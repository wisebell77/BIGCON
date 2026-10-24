"""99. 골목 상권(팀 정의: 길 단위 묶음 58개 구역)에 우리 지표를 붙이고, 관광 동선 구분과 우천 손실 지원금의 규모·상한을 계산한다.

팀 정의(golmok/step1_zones.py, feature/golmok-definition 브랜치): 상가정보(강원 2026.06) 춘천 점포 중 생활업종(음식·소매·수리개인·숙박·예술스포츠) →
전통시장 반경 100m 제외 → 대기업 직영 추정 브랜드 제외 → 도로명이 '~길'인 점포 → 같은 길에서 50m 이내로 이어진 점포(DBSCAN eps 50m, min 3) 15개 이상 →
건물 3동 이상 → 골목을 따라 100m 구간(폭 20m × 100m = 2,000㎡, 춘천 조례 제24조) 안 15개 이상 → 58개 구역, 2,171개 점포.

입력: data/external/골목상권_팀정의/골목상권_구역.csv·골목상권_점포.csv(팀 step1 결과), data/external/관광거점_좌표.csv(역·터미널, 도심·외곽 관광 거점),
      stores_clean, st_industry(업종 손실 단가·취약 등급), r_rate_per_store(업종 × 강수 구간 × 요일 업소당 손실), weather_long_daily(1991~2025 일강수),
      춘천사랑상품권 가맹점(공공데이터포털, EXT_BIG), outputs/demo/district_data.json(지정 골목형상점가 좌표)
출력(outputs/tables)
 al_team_zones.csv        구역별 지표: 점포, 업종 구성, 취약 등급, 호우일 하루 손실, 손실률, 상품권 가맹률, 관광 동선 구분·거리, 지정 상점가 거리, 지급 대상 점포, 하루 지급액
 al_team_zone_stores.csv  점포별(상가업소번호, 업종, 손실 단가, 가맹 여부, 좌표)
 g_payout_table.csv       업종군별 지급표(영업시간 20mm 이상, 20~40mm 단가 × 보전율 50%, 평일/주말)
 g_cap_simulation.csv     지급 일수 상한별 35년(1991~2025) 모의: 평균 지급 일수·지급액, 상한에 걸린 해, 총액 상한 초과 해
 g_cap_years.csv          연도별 모의 결과(1단계·2단계)
 g_scope_summary.csv      1단계(손실 상위 구역)·2단계(전체) 범위의 점포·하루 지급액·시즌 기대액·상한
그림: al04_team_zones_map.png(구역 지도와 관광 동선), al05_team_zones_loss.png(구역별 호우일 손실과 가맹률)
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from analysis import RC, INK, INK2, MUTED, REGION_COLOR, finish

TEAM = C.ROOT / "data" / "external" / "골목상권_팀정의"
POI_PATH = C.ROOT / "data" / "external" / "관광거점_좌표.csv"
PORTAL = C.EXT_BIG / "공공데이터포털"
PAY_GROUPS = ["식품·생필품", "생활·기타소매", "음식점", "패션·잡화", "카페·베이커리", "개인서비스"]  # 지급표 업종군(숙박·여가는 레인체크가 맡고, 주점·대형유통은 제외)
COMP = 0.5          # 보전율
BUDGET1 = 3e8       # 1단계 시범 총액 상한(원)
STAGE1_STORES = 1200  # 1단계 지급 대상 점포 규모(손실 상위 구역부터 누적)
EXPECTED_DAYS_20 = 12.0  # 2026년 7~12월 영업시간 20mm 이상 기대 일수(4.5절)
WD_SHARE = 12.4 / 18.3   # 2026 기대 트리거일 중 평일 비중(rc_trigger_months)
TIER_COLOR = {"1 관광동선 500m": REGION_COLOR["chuncheon"], "2 관광동선 1km": "#f3a683", "3 생활권": "#7f9bbf"}


def read_csv_any(path, **kw):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str, **kw)
        except Exception:
            continue
    raise ValueError(path)


def xy(lon, lat):
    """춘천 중심 기준 평면 좌표(m)."""
    return (np.asarray(lon, float) - 127.73) * 111320 * np.cos(np.radians(37.88)), (np.asarray(lat, float) - 37.88) * 110574


def norm_name(x):
    return re.sub(r"[\s()\-·.,]", "", str(x)).lower()


def road_key(addr):
    m2 = re.search(r"춘천시\s+(?:\S+?[읍면]\s+)?([^\s(]+(?:로|길))\s*(\d+)", str(addr))
    return f"{m2.group(1)} {m2.group(2)}" if m2 else None


def load_stores():
    Z = pd.read_csv(TEAM / "골목상권_구역.csv", encoding="utf-8-sig")
    S = pd.read_csv(TEAM / "골목상권_점포.csv", encoding="utf-8-sig", dtype={"상가업소번호": str, "건물관리번호": str})
    st = C.load("stores_clean")
    st = st[st.region == "chuncheon"][["상가업소번호", "industry", "big"]]
    sti = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")
    sti = sti[sti.region == "chuncheon"][["industry", "group", "rain_loss_day_per_store_만원", "sales_day_per_store_만원", "grade_V"]]
    rate = pd.read_csv(C.TAB / "r_rate_per_store.csv", encoding="utf-8-sig")
    rate = rate[rate.region == "chuncheon"][["industry", "20~40mm·평일", "20~40mm·토", "20~40mm·일"]].rename(columns={"20~40mm·평일": "u_wd", "20~40mm·토": "u_sat", "20~40mm·일": "u_sun"})
    S = S.merge(st, on="상가업소번호", how="left").merge(sti, on="industry", how="left").merge(rate, on="industry", how="left")
    for c in ["u_wd", "u_sat", "u_sun"]:
        S["pay_" + c[2:]] = np.where(S.group.isin(PAY_GROUPS) & S[c].notna(), S[c] * COMP, 0.0)
    S["eligible"] = S.group.isin(PAY_GROUPS) & S.u_wd.notna()
    # 춘천사랑상품권 가맹 여부(96번과 같은 규칙: 상호명 정규화 + 도로명 번지)
    vch = read_csv_any(PORTAL / "강원특별자치도 춘천시_춘천사랑상품권 가맹점 정보_20251119.csv")
    raw = read_csv_any(C.ROOT / "data" / "external" / "소상공인시장진흥공단_상가(상권)정보_강원_202606.csv", usecols=["상가업소번호", "도로명주소", "상호명"])
    vch["k1"], vch["k2"] = vch.상호명.map(norm_name), vch.주소.map(road_key)
    vkeys, vnames = set(zip(vch.k1, vch.k2)), set(vch.k1)
    S = S.merge(raw.rename(columns={"상호명": "상호명_raw"}), on="상가업소번호", how="left")
    S["k1"], S["k2"] = S.상호명_raw.fillna(S.상호명).map(norm_name), S.도로명주소.map(road_key)
    S["voucher"] = [((a, b) in vkeys) or (a in vnames and a != "") for a, b in zip(S.k1, S.k2)]
    print(f"팀 정의 구역 {len(Z)}개, 점포 {len(S):,}개 | 우리 업종 매핑 {S.industry.notna().mean()*100:.1f}%, 지급 대상(지급표 업종군·단가 있음) {S.eligible.sum():,}개({S.eligible.mean()*100:.0f}%), 가맹 추정 {S.voucher.mean()*100:.0f}%")
    return Z, S


def zone_metrics(Z, S):
    g = S.groupby("구역")
    agg = pd.DataFrame({
        "카드 업종 매핑": g.industry.apply(lambda s: int(s.notna().sum())),
        "음식 비중(%)": (g.big.apply(lambda s: (s == "음식").mean()) * 100).round(0),
        "숙박·여가 비중(%)": (g.big.apply(lambda s: s.isin(["숙박", "예술·스포츠"]).mean()) * 100).round(0),
        "취약 등급 점포": g.grade_V.apply(lambda s: int(s.isin(["취약", "매우 취약"]).sum())),
        "호우일 하루 손실(만원)": g["rain_loss_day_per_store_만원"].sum().round(1),
        "하루 매출(만원)": g["sales_day_per_store_만원"].sum().round(1),
        "상품권 가맹률(%)": (g.voucher.mean() * 100).round(0),
        "주요 업종": g.industry.apply(lambda s: ", ".join(f"{i}({n})" for i, n in s.value_counts().head(3).items())),
        "지급 대상 점포": g.eligible.sum().astype(int),
        "하루 지급액 평일(만원)": g.pay_wd.sum().round(1), "하루 지급액 토(만원)": g.pay_sat.sum().round(1), "하루 지급액 일(만원)": g.pay_sun.sum().round(1),
    }).reset_index()
    Z = Z.merge(agg, on="구역", how="left")
    Z["손실률(%)"] = (Z["호우일 하루 손실(만원)"] / Z["하루 매출(만원)"] * 100).round(1)
    # 관광 동선 구분
    poi = pd.read_csv(POI_PATH, encoding="utf-8-sig")
    zx, zy = xy(Z.경도, Z.위도)

    def mind(sub):
        px, py = xy(sub.lon, sub.lat)
        D = np.hypot(zx[:, None] - px[None, :], zy[:, None] - py[None, :])
        return (D.min(1) / 1000).round(2), sub.name.values[D.argmin(1)]

    for k, col in [("교통", "역·터미널"), ("도심관광", "도심 관광 거점"), ("외곽관광", "외곽 관광지")]:
        Z[col + " 거리(km)"], Z["가까운 " + col] = mind(poi[poi.kind == k])

    def tier(r):
        if r["역·터미널 거리(km)"] <= 0.5 or r["도심 관광 거점 거리(km)"] <= 0.5:
            return "1 관광동선 500m"
        if r["역·터미널 거리(km)"] <= 1.0 or r["도심 관광 거점 거리(km)"] <= 1.0 or r["외곽 관광지 거리(km)"] <= 1.5:
            return "2 관광동선 1km"
        return "3 생활권"

    Z["관광 동선 구분"] = Z.apply(tier, axis=1)
    d = json.load(open(C.ROOT / "outputs" / "demo" / "district_data.json", encoding="utf-8"))
    al = pd.DataFrame(d["regions"]["chuncheon"]["alley_official"])
    ax_, ay_ = xy(al.lon, al.lat)
    D = np.hypot(zx[:, None] - ax_[None, :], zy[:, None] - ay_[None, :])
    Z["가까운 지정 상점가"], Z["지정 상점가 거리(km)"] = al.name.values[D.argmin(1)], (D.min(1) / 1000).round(2)
    Z = Z.sort_values("호우일 하루 손실(만원)", ascending=False).reset_index(drop=True)
    Z["손실 순위"] = np.arange(1, len(Z) + 1)
    return Z, poi, al


def payout_table(S):
    sti = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")
    sti = sti[(sti.region == "chuncheon") & sti.group.isin(PAY_GROUPS)][["industry", "group", "n_stores"]]
    rate = pd.read_csv(C.TAB / "r_rate_per_store.csv", encoding="utf-8-sig")
    rate = rate[rate.region == "chuncheon"][["industry", "20~40mm·평일", "20~40mm·토", "20~40mm·일"]]
    m = sti.merge(rate, on="industry", how="inner")
    m["주말"] = (m["20~40mm·토"] + m["20~40mm·일"]) / 2
    rows = []
    for gname, g in m.groupby("group"):
        gv = g.dropna(subset=["20~40mm·평일"])  # 단가가 없는 업종(시간대 β 없음)은 평균에서 뺀다
        w = gv.n_stores.fillna(0)
        rows.append({"업종군": gname, "점포 수(춘천 상가정보)": int(g.n_stores.fillna(0).sum()), "단가 있는 업종 점포": int(w.sum()), "손실 단가 평일(만원)": round(float(np.average(gv["20~40mm·평일"], weights=w)), 1),
                     "손실 단가 주말(만원)": round(float(np.average(gv["주말"], weights=w)), 1),
                     "지급액 평일(만원)": round(float(np.average(gv["20~40mm·평일"], weights=w)) * COMP, 1), "지급액 주말(만원)": round(float(np.average(gv["주말"], weights=w)) * COMP, 1),
                     "골목 구역 안 지급 대상 점포": int(S[(S.group == gname) & S.eligible].shape[0])})
    P = pd.DataFrame(rows).set_index("업종군").loc[PAY_GROUPS].reset_index()
    return P


def simulate(Z, S, scope_zones, label, total_cap):
    """1991~2025년 7~12월 하루 강수 20mm 이상인 날을 날짜 순으로 지급한다고 보고, 지급 일수 상한·총액 상한별 연 지급액을 모의한다."""
    sub = S[S.구역.isin(scope_zones)]
    pay = {"wd": float(sub.pay_wd.sum()) * 1e4, "sat": float(sub.pay_sat.sum()) * 1e4, "sun": float(sub.pay_sun.sum()) * 1e4}  # 원
    avg_day = pay["wd"] * 5 / 7 + pay["sat"] / 7 + pay["sun"] / 7
    w = C.load("weather_long_daily")
    w = w[(w.region == "chuncheon") & (w.month >= 7) & (w.rn_sum >= 20)].sort_values("date")
    w["kind"] = np.select([w.date.dt.dayofweek == 5, w.date.dt.dayofweek == 6], ["sat", "sun"], "wd")
    rows, years = [], []
    n_star = int(np.floor(total_cap / avg_day))
    for ncap in list(dict.fromkeys([6, 8, 10, 11, 12, 15, n_star])) + [None]:
        tot_nocap, tot_cap, days, hit_n, over = [], [], [], 0, 0
        for yr, g in w.groupby("year"):
            seq = [pay[k] for k in g.kind]
            n_use = seq if ncap is None else seq[:ncap]
            s_nocap = float(sum(n_use))
            if ncap is not None and len(seq) > ncap:
                hit_n += 1
            over += int(s_nocap > total_cap)
            # 총액 상한: 넘는 날 비례 지급
            cum, s_cap = 0.0, 0.0
            for v in n_use:
                if cum + v >= total_cap:
                    s_cap = total_cap
                    break
                cum += v
                s_cap = cum
            tot_nocap.append(s_nocap)
            tot_cap.append(s_cap)
            days.append(len(n_use))
            if ncap == n_star:
                years.append({"범위": label, "연도": int(yr), "20mm+ 일수(7~12월)": len(seq), "지급 일수(상한 적용)": len(n_use), "지급액(억, 일수 상한만)": round(s_nocap / 1e8, 2), "지급액(억, 이중 상한)": round(s_cap / 1e8, 2)})
        rows.append({"범위": label, "지급 일수 상한": "없음" if ncap is None else ncap, "평균 지급 일수": round(float(np.mean(days)), 1), "상한에 걸린 해(35년 중)": hit_n,
                     "연 지급액 평균(억, 일수 상한만)": round(float(np.mean(tot_nocap)) / 1e8, 2), "연 지급액 최대(억, 일수 상한만)": round(float(np.max(tot_nocap)) / 1e8, 2),
                     f"총액 {total_cap/1e8:.1f}억 초과 해": over, f"연 지급액 평균(억, 이중 상한 {total_cap/1e8:.1f}억)": round(float(np.mean(tot_cap)) / 1e8, 2)})
    summary = {"범위": label, "구역 수": len(scope_zones), "구역 점포": int(sub.shape[0]), "지급 대상 점포": int(sub.eligible.sum()),
               "하루 지급액 평일(억)": round(pay["wd"] / 1e8, 3), "하루 지급액 토(억)": round(pay["sat"] / 1e8, 3), "하루 지급액 일(억)": round(pay["sun"] / 1e8, 3),
               "하루 지급액 평균(억, 요일 가중)": round(avg_day / 1e8, 3), "2026 기대 지급액(억, 20mm 12일, 상한 없음)": round((pay["wd"] * WD_SHARE + (pay["sat"] + pay["sun"]) / 2 * (1 - WD_SHARE)) * EXPECTED_DAYS_20 / 1e8, 2),
               "총액 상한(억)": total_cap / 1e8, "일수 상한(총액 ÷ 평균 하루 지급액)": n_star}
    return pd.DataFrame(rows), pd.DataFrame(years), summary


def figures(Z, S, poi, al):
    plt.rcParams.update(RC)
    # al04: 구역 지도와 관광 동선
    st = C.load("stores_clean")
    st = st[st.region == "chuncheon"]
    fig, ax = plt.subplots(figsize=(11.5, 8.2), layout="constrained")
    sx, sy = xy(st.lon, st.lat)
    ax.scatter(sx / 1000, sy / 1000, s=1.5, color="#d9d8d2", alpha=0.6, lw=0, label="춘천 전체 업소(상가정보)")
    tier_of = Z.set_index("구역")["관광 동선 구분"]
    for t, color in TIER_COLOR.items():
        sub = S[S.구역.map(tier_of) == t]
        bx, by = xy(sub.경도, sub.위도)
        ax.scatter(bx / 1000, by / 1000, s=6, color=color, lw=0, label=f"{t}: {Z[Z['관광 동선 구분'] == t].shape[0]}개 구역, 점포 {len(sub):,}")
    px, py = xy(poi.lon, poi.lat)
    # 겹치는 라벨은 빼거나 위치를 조정한다
    LABEL = {"춘천역": (6, 4), "남춘천역": (6, 6), "춘천시외버스터미널": (-62, -12), "춘천 명동 닭갈비골목": (8, 6), "소양강스카이워크": (6, 4), "공지천 유원지": (6, 4),
             "국립춘천박물관": (6, -10), "의암호 스카이워크": (6, 4), "레고랜드 코리아 리조트": (6, 4), "삼악산호수케이블카": (6, 4), "춘천중앙시장": (-58, -12), "육림고개": (8, -10)}
    for kind, marker, color, size in [("교통", "s", INK, 60), ("도심관광", "^", "#1b4f9c", 60)]:
        m = (poi.kind == kind).values
        ax.scatter(px[m] / 1000, py[m] / 1000, marker=marker, s=size, color=color, edgecolor="white", lw=0.8, zorder=5, label="역·터미널" if kind == "교통" else "도심 관광 거점")
        for x_, y_, nm in zip(px[m], py[m], poi.name[m]):
            if nm in LABEL:
                ax.annotate(nm.replace("춘천 ", "").replace(" 코리아 리조트", "").replace("춘천시외버스", "시외버스"), (x_ / 1000, y_ / 1000), textcoords="offset points", xytext=LABEL[nm], fontsize=7.5, color=color)
    ax_, ay_ = xy(al.lon, al.lat)
    ax.scatter(ax_ / 1000, ay_ / 1000, marker="*", s=110, color="#e34948", edgecolor="white", lw=0.8, zorder=5, label="지정 골목형상점가")
    top = Z.sort_values("점포수", ascending=False).head(8)
    tx, ty = xy(top.경도, top.위도)
    for x_, y_, nm in zip(tx, ty, top.길):
        if nm == "서부대성로44번길":
            continue  # 명동 라벨과 겹침
        ax.annotate(nm, (x_ / 1000, y_ / 1000), textcoords="offset points", xytext=(4, -9), fontsize=7, color=INK2)
    zx, zy = xy(Z.경도, Z.위도)
    ax.set_xlim(zx.min() / 1000 - 0.8, zx.max() / 1000 + 1.2)
    ax.set_ylim(zy.min() / 1000 - 0.8, zy.max() / 1000 + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("동서(km, 춘천 중심 기준)")
    ax.set_ylabel("남북(km)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=1.6, frameon=False)
    finish(fig, C.FIG, "al04_team_zones_map", "춘천 골목 상권 58개 구역(길 단위 묶음)과 관광 동선: 역·터미널 또는 도심 관광 거점에서 500m·1km 안 구역")
    plt.close(fig)
    # al05: 손실 상위 구역과 가맹률
    top = Z.head(20).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.barh(np.arange(len(top)), top["호우일 하루 손실(만원)"], color=[TIER_COLOR[t] for t in top["관광 동선 구분"]])
    ax.set_yticks(np.arange(len(top)), [f"{r['길']}({r['행정동']}, {int(r['점포수'])}곳)" for _, r in top.iterrows()], fontsize=8)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["호우일 하루 손실(만원)"] + 3, i, f"가맹 {int(r['상품권 가맹률(%)'])}%", va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("호우일 하루 손실(만원, 업종 단가 × 점포 수 배분값)")
    ax.set_title("손실 상위 20개 구역(색: 관광 동선 구분)", loc="left", fontsize=10)
    ax = axes[1]
    for t, color in TIER_COLOR.items():
        sub = Z[Z["관광 동선 구분"] == t]
        ax.scatter(sub["상품권 가맹률(%)"], sub["손실률(%)"], s=sub.점포수 * 1.6 + 10, color=color, alpha=0.75, edgecolor="white", lw=0.6, label=t)
    ax.set_xlabel("춘천사랑상품권 가맹률(%, 상호명·번지 결합 추정)")
    ax.set_ylabel("손실률(%, 호우일 손실 ÷ 하루 매출)")
    ax.set_title("구역별 가맹률과 손실률(점 크기: 점포 수)", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    finish(fig, C.FIG, "al05_team_zones_loss", "골목 상권 58개 구역의 호우일 손실, 상품권 가맹률, 관광 동선 구분")
    plt.close(fig)


def main():
    Z, S = load_stores()
    Z, poi, al = zone_metrics(Z, S)
    P = payout_table(S)
    print("\n=== 지급표(영업시간 20mm 이상, 보전율 50%)")
    print(P.to_string(index=False))
    # 범위: 1단계 = 손실 상위 구역부터 지급 대상 점포 누적 약 1,200곳, 2단계 = 58개 전체
    cum = Z["지급 대상 점포"].cumsum()
    n1 = int((cum < STAGE1_STORES).sum()) + 1
    stage1 = Z.구역.head(n1).tolist()
    Z["1단계 시범 대상"] = Z.구역.isin(stage1)
    sims, years, summ = [], [], []
    s1, y1, m1 = simulate(Z, S, stage1, f"1단계(손실 상위 {n1}개 구역)", BUDGET1)
    sims.append(s1); years.append(y1); summ.append(m1)
    avg2 = (S.pay_wd.sum() * 5 / 7 + S.pay_sat.sum() / 7 + S.pay_sun.sum() / 7) * 1e4
    cap2 = float(np.ceil(m1["일수 상한(총액 ÷ 평균 하루 지급액)"] * avg2 / 0.5e8) * 0.5e8)
    s2, y2, m2 = simulate(Z, S, Z.구역.tolist(), "2단계(58개 구역 전체)", cap2)
    sims.append(s2); years.append(y2); summ.append(m2)
    SIM, YRS, SUM = pd.concat(sims, ignore_index=True), pd.concat(years, ignore_index=True), pd.DataFrame(summ)
    print("\n=== 범위 요약"); print(SUM.T.to_string())
    print("\n=== 지급 일수 상한 모의(1991~2025, 하루 강수 20mm 이상 날 기준)"); print(SIM.to_string(index=False))
    print("\n=== 관광 동선 구분별"); print(Z.groupby("관광 동선 구분").agg(구역=("구역", "size"), 점포=("점포수", "sum"), 손실=("호우일 하루 손실(만원)", "sum"), 가맹률=("상품권 가맹률(%)", "mean"), 취약=("취약 등급 점포", "sum")).round(1).to_string())
    print("\n=== 손실 상위 12개 구역")
    cols = ["손실 순위", "구역", "길", "행정동", "점포수", "관광 동선 구분", "역·터미널 거리(km)", "도심 관광 거점 거리(km)", "음식 비중(%)", "취약 등급 점포", "호우일 하루 손실(만원)", "손실률(%)", "상품권 가맹률(%)", "지급 대상 점포", "하루 지급액 평일(만원)", "주요 업종"]
    print(Z[cols].head(12).to_string(index=False))
    print("\n=== 지정 골목형상점가 500m 안 구역"); print(Z[Z["지정 상점가 거리(km)"] <= 0.5][["구역", "길", "행정동", "점포수", "가까운 지정 상점가", "지정 상점가 거리(km)"]].to_string(index=False))
    out_cols = ["손실 순위", "구역", "길", "행정동", "점포수", "건물수", "100m구간 최대 점포", "구간길이_m", "경도", "위도", "카드 업종 매핑", "음식 비중(%)", "숙박·여가 비중(%)", "취약 등급 점포",
                "호우일 하루 손실(만원)", "하루 매출(만원)", "손실률(%)", "상품권 가맹률(%)", "주요 업종", "관광 동선 구분", "가까운 역·터미널", "역·터미널 거리(km)", "가까운 도심 관광 거점", "도심 관광 거점 거리(km)",
                "가까운 외곽 관광지", "외곽 관광지 거리(km)", "가까운 지정 상점가", "지정 상점가 거리(km)", "지급 대상 점포", "하루 지급액 평일(만원)", "하루 지급액 토(만원)", "하루 지급액 일(만원)", "1단계 시범 대상"]
    Z[out_cols].to_csv(C.TAB / "al_team_zones.csv", index=False, encoding="utf-8-sig")
    S[["구역", "상가업소번호", "상호명", "상권업종소분류명", "industry", "group", "big", "grade_V", "rain_loss_day_per_store_만원", "sales_day_per_store_만원", "eligible", "pay_wd", "pay_sat", "pay_sun", "voucher", "길", "행정동명", "건물관리번호", "경도", "위도", "X", "Y"]].to_csv(C.TAB / "al_team_zone_stores.csv", index=False, encoding="utf-8-sig")
    P.to_csv(C.TAB / "g_payout_table.csv", index=False, encoding="utf-8-sig")
    SIM.to_csv(C.TAB / "g_cap_simulation.csv", index=False, encoding="utf-8-sig")
    YRS.to_csv(C.TAB / "g_cap_years.csv", index=False, encoding="utf-8-sig")
    SUM.to_csv(C.TAB / "g_scope_summary.csv", index=False, encoding="utf-8-sig")
    figures(Z, S, poi, al)
    print("\n저장: al_team_zones, al_team_zone_stores, g_payout_table, g_cap_simulation, g_cap_years, g_scope_summary; 그림 al04, al05")


if __name__ == "__main__":
    main()
