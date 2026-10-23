"""95. 골목 상권 후보 구역: 상가정보 좌표로 '골목형상점가 요건'(2천㎡ 이내 30개 이상 밀집)을 만족하는 구역을 찾고, 손실 단가를 얹는다.

- 법 기준(전통시장법 시행령 제2조의2): 2천㎡ 이내 면적에 소상공인 점포 30개 이상 밀집. 지자체 조례로 완화(예: 15개)되는 곳도 있다.
- 방법: 업소 좌표(UTM-K)로 반경 25.2m(면적 2천㎡인 원) 안의 업소 수를 세고, 30개 이상인 업소를 '밀집 핵'으로 본다. 핵끼리 25m 이내로 이어지면 한 구역(클러스터)으로 묶는다.
  (완화 기준 15개도 같이 계산한다. 상가정보는 소상공인이 아닌 업소도 포함하므로 '전체 업소 수'와 '카드 업종 매핑 업소 수'를 둘 다 낸다.)
- 주의: 강남은 한 건물에 수십 개 업소가 있는 고층 상가가 많아 같은 좌표에 업소가 몰린다. 건물 1개가 구역 업소의 절반 이상이면 '단일 건물' 표시를 한다.
- 손실: 구역 안 업소의 카드 업종별 '호우일 하루 업소당 손실'(st_industry)을 더해 구역의 호우일 하루 손실과 20mm 트리거 시즌 기대 지급액(r_rate_per_store)을 만든다. 측정값이 아니라 업종 단위 손실을 업소 수로 배분한 값이다.
출력: al_clusters.csv(구역 요약), al_cluster_stores.csv(업소→구역), 그림 al01_alley_map.png, al02_alley_top.png
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

import common as C
from analysis import RC, SURFACE, INK, INK2, MUTED, WASH, REGION_COLOR, finish

R_CORE = np.sqrt(2000 / np.pi)  # 25.2m: 면적 2천㎡인 원의 반지름
THRESH = {"법정(30개)": 30, "완화(15개)": 15}
LINK = 50.0  # 핵끼리 이 거리 안이면 같은 구역(길 건너·옆 블록까지 한 상권으로)
ATTACH = 30.0  # 핵이 아닌 업소는 이 거리 안의 핵 구역에 붙인다


def load_raw_extra():
    """상가정보 원본에서 건물관리번호·상호명·상권업종대분류명을 붙인다."""
    parts = []
    for f in ["소상공인시장진흥공단_상가(상권)정보_서울_202606.csv", "소상공인시장진흥공단_상가(상권)정보_강원_202606.csv"]:
        d = pd.read_csv(C.ROOT / "data" / "external" / f, encoding="utf-8-sig", usecols=["상가업소번호", "상호명", "상권업종대분류명", "건물관리번호", "건물명"], dtype=str)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    st = C.load("stores_clean").merge(load_raw_extra(), on="상가업소번호", how="left")
    sti = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")[["region", "industry", "rain_loss_day_per_store_만원", "sales_day_per_store_만원", "grade_V"]]
    rate = pd.read_csv(C.TAB / "r_rate_per_store.csv", encoding="utf-8-sig")[["region", "industry", "시즌 기대 지급액 업소당(만원, 20mm 트리거)", "20~40mm·평일", "20~40mm·토"]]
    st = st.merge(sti, on=["region", "industry"], how="left").merge(rate, on=["region", "industry"], how="left")
    rows, members = [], []
    for region in C.REGIONS:
        d = st[st.region == region].reset_index(drop=True)
        xy = d[["x", "y"]].values
        tree = cKDTree(xy)
        cnt = np.array([len(v) for v in tree.query_ball_point(xy, R_CORE)])
        d["n_within_2000m2"] = cnt
        for label, thr in THRESH.items():
            core = np.where(cnt >= thr)[0]
            if len(core) == 0:
                continue
            ctree = cKDTree(xy[core])
            pairs = ctree.query_pairs(LINK, output_type="ndarray")
            n = len(core)
            adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, lab = connected_components(adj, directed=False)
            # 핵이 아닌 업소도 가장 가까운 핵이 25m 이내면 그 구역에 넣는다(구역 경계 업소)
            dist, idx = ctree.query(xy, k=1)
            cl = np.where(dist <= ATTACH, lab[idx], -1)
            cl[core] = lab
            d[f"cluster_{label}"] = cl
            for k in np.unique(cl[cl >= 0]):
                g = d[cl == k]
                hull_w, hull_h = g.x.max() - g.x.min() + 10, g.y.max() - g.y.min() + 10
                top_b = g.건물관리번호.value_counts(dropna=True)
                single = (top_b.iloc[0] / len(g) >= 0.5) if len(top_b) else False
                mapped = g[g.industry.notna()]
                food = (g["상권업종대분류명"] == "음식").mean() if "상권업종대분류명" in g else np.nan
                rows.append({"region": region, "지역": C.REGION_NAME[region], "기준": label, "cluster": int(k), "행정동": g.emd_name.mode().iloc[0] if len(g.emd_name.mode()) else "", "법정동": g.ldong_name.mode().iloc[0] if len(g.ldong_name.mode()) else "",
                             "업소 수": len(g), "카드 업종 매핑 업소": len(mapped), "밀집 핵 업소(반경 25m 내 30+)": int((g.n_within_2000m2 >= 30).sum()),
                             "가로 m": round(hull_w), "세로 m": round(hull_h), "대략 면적 ㎡": round(hull_w * hull_h), "단일 건물 비중": round(float(top_b.iloc[0] / len(g)), 2) if len(top_b) else np.nan, "단일 건물 구역": single,
                             "대표 건물": top_b.index[0] if len(top_b) and single else "", "건물명": g.건물명.mode().iloc[0] if single and len(g.건물명.mode()) else "",
                             "음식 업종 비중": round(float(food), 2) if food == food else np.nan,
                             "주요 업종(상위 3)": ", ".join(f"{i}({n})" for i, n in mapped.industry.value_counts().head(3).items()),
                             "취약 등급 업소": int(mapped.grade_V.isin(["취약", "매우 취약"]).sum()),
                             "호우일 하루 손실(만원)": round(float(mapped["rain_loss_day_per_store_만원"].fillna(0).sum()), 1),
                             "20mm 시즌 기대 지급액(만원)": round(float(mapped["시즌 기대 지급액 업소당(만원, 20mm 트리거)"].fillna(0).sum()), 1),
                             "업소당 일매출 합(만원)": round(float(mapped["sales_day_per_store_만원"].fillna(0).sum()), 1),
                             "cx": round(float(g.x.mean())), "cy": round(float(g.y.mean())), "lon": round(float(g.lon.mean()), 5), "lat": round(float(g.lat.mean()), 5)})
        members.append(d[["상가업소번호", "region", "emd_name", "industry", "상호명", "상권업종대분류명", "건물관리번호", "x", "y", "lon", "lat", "n_within_2000m2"] + [c for c in d.columns if c.startswith("cluster_")]])
    A = pd.DataFrame(rows)
    A["손실률 proxy(%)"] = (A["호우일 하루 손실(만원)"] / A["업소당 일매출 합(만원)"].replace(0, np.nan) * 100).round(1)
    A = A.sort_values(["기준", "region", "업소 수"], ascending=[True, True, False])
    A.to_csv(C.TAB / "al_clusters.csv", index=False, encoding="utf-8-sig")
    M = pd.concat(members, ignore_index=True)
    M.to_csv(C.TAB / "al_cluster_stores.csv", index=False, encoding="utf-8-sig")
    for label in THRESH:
        a = A[A.기준 == label]
        print(f"== {label}: 구역 수 {a.groupby('지역').size().to_dict()}, 업소 합 {a.groupby('지역')['업소 수'].sum().to_dict()}, 단일 건물 구역 {a.groupby('지역')['단일 건물 구역'].sum().to_dict()}")
    show = A[(A.기준 == "법정(30개)") & (~A["단일 건물 구역"])].groupby("region").head(8)
    sb1 = A[(A.기준 == "법정(30개)") & (A["단일 건물 구역"]) & (A.region == "chuncheon")].sort_values("업소 수", ascending=False).head(8)
    print("춘천 단일 건물 구역(참고):"); print(sb1[["행정동", "법정동", "업소 수", "건물명", "주요 업종(상위 3)"]].to_string(index=False))
    sb2 = A[(A.기준 == "완화(15개)") & (~A["단일 건물 구역"]) & (A.region == "chuncheon")].sort_values("업소 수", ascending=False).head(12)
    print("춘천 완화 기준(15개) 다건물 구역 상위:"); print(sb2[["행정동", "법정동", "업소 수", "카드 업종 매핑 업소", "대략 면적 ㎡", "음식 업종 비중", "주요 업종(상위 3)", "호우일 하루 손실(만원)"]].to_string(index=False))
    print(show[["지역", "행정동", "법정동", "업소 수", "카드 업종 매핑 업소", "대략 면적 ㎡", "음식 업종 비중", "주요 업종(상위 3)", "호우일 하루 손실(만원)", "20mm 시즌 기대 지급액(만원)"]].to_string(index=False))

    # 그림: 지도(법정 기준, 단일 건물 제외)와 상위 구역 막대
    plt.rcParams.update(RC)
    cells = C.load("sk_cell")
    MAP_RULE = {"gangnam": "법정(30개)", "chuncheon": "완화(15개)"}  # 춘천시는 2026년 4월 조례로 점포 수 기준을 완화했다
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, region in zip(axes, C.REGIONS):
        d = st[st.region == region]
        a = A[(A.region == region) & (A.기준 == MAP_RULE[region])]
        c = cells[(cells.region == region) & (cells["pop"] > 0)]
        if region == "gangnam":
            cx, cy, h = (c.x.min() + c.x.max()) / 2, (c.y.min() + c.y.max()) / 2, 5500
        else:
            cx, cy, h = np.average(c.x, weights=c["pop"]), np.average(c.y, weights=c["pop"]), 3200
        ax.scatter(d.x, d.y, s=0.6, color="#c3c2b7", linewidths=0, rasterized=True)
        multi = a[~a["단일 건물 구역"]]; single = a[a["단일 건물 구역"]]
        ax.scatter(single.cx, single.cy, s=18, marker="s", color="#898781", label=f"단일 건물 구역 {len(single)}개")
        sc = ax.scatter(multi.cx, multi.cy, s=multi["업소 수"].clip(upper=400) * 0.6 + 10, c=multi["손실률 proxy(%)"].fillna(0), cmap="Blues", vmin=0, vmax=12, edgecolors=INK, linewidths=0.5, label=f"골목 상권 후보 {len(multi)}개")
        for _, r in multi.sort_values("업소 수", ascending=False).head(6).iterrows():
            ax.annotate(f"{r.법정동} {r['업소 수']}개", (r.cx, r.cy), fontsize=8, color=INK, ha="left", va="bottom", xytext=(6, 4), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f"{C.REGION_NAME[region]}: 밀집 구역 후보, 기준 {MAP_RULE[region]} (원 크기 = 업소 수, 색 = 호우 손실률)"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    cb = fig.colorbar(sc, ax=axes, shrink=0.5, label="호우일 손실 / 평상 매출 (%, 업종 단가 배분)"); cb.outline.set_visible(False)
    finish(fig, C.FIG, "al01_alley_map", "상가정보 좌표로 찾은 밀집 구역(회색 점: 전체 업소, 회색 네모: 한 건물이 절반 이상인 구역)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, region in zip(axes, C.REGIONS):
        a = A[(A.region == region) & (A.기준 == MAP_RULE[region]) & (~A["단일 건물 구역"])].sort_values("호우일 하루 손실(만원)", ascending=False).head(12)
        ax.barh(range(len(a)), a["호우일 하루 손실(만원)"], color=REGION_COLOR[region], edgecolor=SURFACE)
        for i, (n, m) in enumerate(zip(a["업소 수"], a["카드 업종 매핑 업소"])):
            ax.text(a["호우일 하루 손실(만원)"].iloc[i], i, f"  {n}개 업소(매핑 {m})", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(a)), [f"{r.법정동} {r.cluster}" for _, r in a.iterrows()]); ax.invert_yaxis(); ax.set_xlim(0, a["호우일 하루 손실(만원)"].max() * 1.5 if len(a) else 1)
        ax.set_title(f"{C.REGION_NAME[region]}: 구역별 호우일 하루 손실(만원), 기준 {MAP_RULE[region]}"); ax.grid(axis="y", visible=False)
    finish(fig, C.FIG, "al02_alley_top", "골목 상권 후보 구역의 호우일 손실(업종 단가 × 업소 수)")
    print(f"저장: al_clusters.csv({len(A)}행), al_cluster_stores.csv({len(M):,}행), al01, al02")


if __name__ == "__main__":
    main()
