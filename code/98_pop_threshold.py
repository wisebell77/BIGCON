"""98. 단기예보 강수확률(POP)의 신뢰도와 레인체크 알림(가발급) 임계값.

질문: 알림 기준 "강수확률 60% 이상"에 근거가 있는가. 2025년 7~12월 D-1 17시 단기예보의 강수확률을 실제 강수와 비교한다.
 (1) 예보 시각 단위 신뢰도: POP 값별로 실제 강수(0.1mm+, 1mm+)가 관측된 빈도. POP가 가리키는 시간 창(앞 3시간/뒤 3시간/±1시간/해당 1시간)은 기후값 대비 Brier 기술 점수가 가장 높은 것으로 고른다.
 (2) 하루 단위: 영업시간(06~21시 발표 구간) 최대 POP 구간별로 영업시간 비 1/5/10/20mm 이상이 실제 온 비율.
 (3) 임계값 20~90%와 규칙(최대/평균/2구간 이상/오후, 예보 강수량, 합집합)별 적중률 POD·헛경보율 FAR·CSI·알림일 비중, 월별 알림 일수,
     놓침:헛알림 비용비별 최적 임계값.

입력: kma_forecast_point(POP, 6·9·12·15·18·21시, 격자점 값), weather_hourly(ASOS 1시간 강수), weather_daily(영업시간 강수 rn_biz), kma_forecast_daily(fc_rn_biz), calendar
출력(outputs/tables): rc_pop_window.csv, rc_pop_reliability_slot.csv(값별), rc_pop_reliability_slot_grouped.csv(발표 등급별), rc_pop_reliability_day.csv, rc_pop_threshold.csv, rc_pop_threshold_month.csv, rc_pop_costloss.csv
그림: rc06_pop_reliability.png
"""
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

SLOTS = [6, 9, 12, 15, 18, 21]
PM_SLOTS = [12, 15, 18]
THRESH = list(range(20, 100, 10))
EVENTS = {"rain5": ("영업시간 비 5mm+", 5), "rain10": ("영업시간 비 10mm+", 10), "rain20": ("영업시간 비 20mm+", 20)}
WINDOWS = {"앞 3시간(h-3~h-1)": (-3, -1), "뒤 3시간(h~h+2)": (0, 2), "±1시간(h-1~h+1)": (-1, 1), "해당 1시간(h)": (0, 0)}
REGION_KO = {"chuncheon": "춘천", "gangnam": "강남", "all": "합계"}


def load():
    pt = C.load("kma_forecast_point")
    pop = pt[pt["var"] == "POP"][["region", "date", "hour", "value"]].copy()
    pop["date"] = pd.to_datetime(pop.date)
    wh = C.load("weather_hourly")[["region", "date", "hour", "rn"]].copy()
    wh["date"] = pd.to_datetime(wh.date)
    W = wh.pivot_table(index=["region", "date"], columns="hour", values="rn").reindex(columns=range(24))
    wd = C.load("weather_daily")[["region", "date", "rn_biz", "rn_sum"]].copy()
    wd["date"] = pd.to_datetime(wd.date)
    fd = C.load("kma_forecast_daily")[["region", "date", "fc_rn_biz"]].copy()
    fd["date"] = pd.to_datetime(fd.date)
    cal = C.load("calendar")[["date", "month", "is_offday"]].copy()
    cal["date"] = pd.to_datetime(cal.date)
    return pop, W, wd, fd, cal


def slot_table(pop, W):
    """예보 시각별 POP와 시간 창별 실제 강수."""
    s = pop.merge(W, left_on=["region", "date"], right_index=True, how="inner")
    arr = s[list(range(24))].to_numpy(dtype=float)
    h = s.hour.to_numpy()
    rows = np.arange(len(s))
    for name, (a, b) in WINDOWS.items():
        tot = np.zeros(len(s))
        for k in range(a, b + 1):
            hh = h + k
            ok = (hh >= 0) & (hh <= 23)
            v = np.where(ok, arr[rows, np.clip(hh, 0, 23)], np.nan)
            tot = tot + np.nan_to_num(v)
        s["rn_" + name] = tot
    return s[["region", "date", "hour", "value"] + ["rn_" + n for n in WINDOWS]]


def brier(p, y):
    p = np.asarray(p, float) / 100
    y = np.asarray(y, float)
    bs = np.mean((p - y) ** 2)
    base = y.mean()
    ref = base * (1 - base)
    return bs, ref, 1 - bs / ref if ref > 0 else np.nan


def window_test(s):
    out = []
    for name in WINDOWS:
        for region, g in list(s.groupby("region")) + [("all", s)]:
            for ev, thr in [("0.1mm+", 0.1), ("1mm+", 1.0)]:
                y = (g["rn_" + name] >= thr).astype(int)
                bs, ref, bss = brier(g.value, y)
                out.append({"시간 창": name, "지역": REGION_KO[region], "사건": ev, "예보 수": len(g), "관측 빈도": round(y.mean(), 3),
                            "Brier": round(bs, 4), "기후값 Brier": round(ref, 4), "Brier 기술 점수": round(bss, 3)})
    return pd.DataFrame(out)


def pop_class(v):
    """기상청 단기예보 강수확률은 0·10·20·30·60·70·80·90·100%로 발표한다(40·50% 없음). 격자 값의 드문 중간값(61~69 등)은 가까운 등급으로 묶는다."""
    v = float(v)
    if v < 10:
        return 0
    if v < 30:
        return 20
    if v < 60:
        return 30
    if v < 70:
        return 60
    return 70


def slot_reliability(s, win, grouped=False):
    rn = s["rn_" + win]
    s = s.assign(hit01=(rn >= 0.1).astype(int), hit1=(rn >= 1).astype(int), rn=rn, cls=s.value.map(pop_class))
    key = "cls" if grouped else "value"
    out = []
    for region, g in list(s.groupby("region")) + [("all", s)]:
        for v, gg in g.groupby(key):
            out.append({"지역": REGION_KO[region], "강수확률(%)": int(v) if not grouped else {0: "0", 20: "10~20", 30: "30", 60: "60", 70: "70 이상"}[int(v)],
                        "예보 수": len(gg), "실제 강수 0.1mm+ 비율": round(gg.hit01.mean(), 3),
                        "실제 강수 1mm+ 비율": round(gg.hit1.mean(), 3), "평균 강수(mm)": round(gg.rn.mean(), 2)})
    return pd.DataFrame(out)


def daily_table(pop, wd, fd, cal):
    p = pop[pop.hour.isin(SLOTS)]
    g = p.groupby(["region", "date"])
    d = pd.DataFrame({"pop_max": g.value.max(), "pop_mean": g.value.mean(), "n_slots": g.value.size()}).reset_index()
    pm = p[p.hour.isin(PM_SLOTS)].groupby(["region", "date"]).value.max().rename("pop_pm_max").reset_index()
    d = d.merge(pm, on=["region", "date"], how="left")
    for x in THRESH:
        d = d.merge(p.assign(ge=(p.value >= x).astype(int)).groupby(["region", "date"]).ge.sum().rename(f"n_ge{x}").reset_index(), on=["region", "date"], how="left")
    d = d.merge(wd, on=["region", "date"], how="inner").merge(fd, on=["region", "date"], how="left").merge(cal, on="date", how="left")
    d = d[d.n_slots == len(SLOTS)].copy()
    for k, (_, thr) in EVENTS.items():
        d[k] = (d.rn_biz >= thr).astype(int)
    return d


def day_reliability(d):
    bins = [-1, 19, 29, 39, 49, 59, 69, 79, 89, 100]
    labels = ["0~10", "20", "30", "40", "50", "60", "70", "80", "90~100"]
    d = d.assign(bin=pd.cut(d.pop_max, bins=bins, labels=labels))
    out = []
    for region, g in list(d.groupby("region")) + [("all", d)]:
        for b, gg in g.groupby("bin", observed=True):
            out.append({"지역": REGION_KO[region], "영업시간 최대 강수확률(%)": str(b), "일수": len(gg), "비 1mm+ 비율": round((gg.rn_biz >= 1).mean(), 2),
                        "비 5mm+ 비율": round(gg.rain5.mean(), 2), "비 10mm+ 비율": round(gg.rain10.mean(), 2), "비 20mm+ 비율": round(gg.rain20.mean(), 2),
                        "평균 영업시간 강수(mm)": round(gg.rn_biz.mean(), 1)})
    return pd.DataFrame(out)


def rules(d):
    """규칙 이름 → 알림 여부(불리언 시리즈)."""
    r = {}
    for x in THRESH:
        r[f"최대 POP {x}%+"] = d.pop_max >= x
    for x in range(20, 80, 10):
        r[f"평균 POP {x}%+"] = d.pop_mean >= x
    for x in range(40, 90, 10):
        r[f"POP {x}%+ 구간 2개 이상"] = d[f"n_ge{x}"] >= 2
    for x in range(40, 90, 10):
        r[f"오후(12~18시) 최대 POP {x}%+"] = d.pop_pm_max >= x
    r["예보 강수 1mm+"] = d.fc_rn_biz >= 1
    r["예보 강수 5mm+"] = d.fc_rn_biz >= 5
    r["예보 강수 10mm+"] = d.fc_rn_biz >= 10
    for x in [60, 70, 80]:
        r[f"예보 5mm+ 또는 최대 POP {x}%+"] = (d.fc_rn_biz >= 5) | (d.pop_max >= x)
    r["예보 5mm+ 그리고 최대 POP 60%+"] = (d.fc_rn_biz >= 5) & (d.pop_max >= 60)
    return r


def skill(alert, event):
    a = np.asarray(alert, bool)
    e = np.asarray(event, bool)
    hit, miss, fa = int((a & e).sum()), int((~a & e).sum()), int((a & ~e).sum())
    pod = hit / (hit + miss) if hit + miss else np.nan
    far = fa / (hit + fa) if hit + fa else np.nan
    csi = hit / (hit + miss + fa) if hit + miss + fa else np.nan
    return hit, miss, fa, pod, far, csi


def threshold_table(d):
    out = []
    for region, g in list(d.groupby("region")) + [("all", d)]:
        R = rules(g)
        for subset, gg_mask in [("전체", np.ones(len(g), bool)), ("주말·공휴일", g.is_offday.to_numpy() == 1), ("평일", g.is_offday.to_numpy() == 0)]:
            gg = g[gg_mask]
            for name, al in R.items():
                if subset != "전체" and not name.startswith("최대 POP") and not name.startswith("예보 강수 5mm"):
                    continue
                al = al[gg_mask]
                for k, (label, _) in EVENTS.items():
                    hit, miss, fa, pod, far, csi = skill(al, gg[k])
                    missed = gg[(~al.to_numpy()) & (gg[k] == 1)]
                    out.append({"지역": REGION_KO[region], "요일": subset, "규칙": name, "사건": label, "일수": len(gg), "관측 일수": int(gg[k].sum()),
                                "알림 일수": int(al.sum()), "적중": hit, "놓침": miss, "헛경보": fa, "적중률 POD": round(pod, 3), "헛경보율 FAR": round(far, 3),
                                "CSI": round(csi, 3), "알림일 비중(%)": round(al.mean() * 100, 1),
                                "놓친 날(영업시간 mm)": "; ".join(f"{dt:%m-%d} {rn:.0f}" for dt, rn in zip(missed.date, missed.rn_biz)) if k == "rain10" else ""})
    return pd.DataFrame(out)


def month_table(d):
    out = []
    for region, g in d.groupby("region"):
        for m, gm in g.groupby("month"):
            row = {"지역": REGION_KO[region], "월": int(m), "일수": len(gm), "관측 5mm+ 일수": int(gm.rain5.sum()), "관측 10mm+ 일수": int(gm.rain10.sum()),
                   "예보 5mm+ 알림": int((gm.fc_rn_biz >= 5).sum())}
            for x in [40, 50, 60, 70, 80]:
                row[f"최대 POP {x}%+ 알림"] = int((gm.pop_max >= x).sum())
                row[f"최대 POP {x}%+ 적중(10mm)"] = int(((gm.pop_max >= x) & (gm.rain10 == 1)).sum())
            out.append(row)
    return pd.DataFrame(out)


def costloss(d):
    """놓침 1건의 비용을 1, 헛알림 1건의 비용을 r로 두고 기대 비용 = r × 헛경보 + 놓침 을 최소화하는 임계값."""
    out = []
    for region, g in list(d.groupby("region")) + [("all", d)]:
        for k, (label, _) in EVENTS.items():
            if k == "rain20":
                continue
            ev = g[k].to_numpy(bool)
            n_ev, n = int(ev.sum()), len(g)
            for r in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
                costs = {}
                for x in THRESH:
                    al = (g.pop_max >= x).to_numpy()
                    hit, miss, fa, *_ = skill(al, ev)
                    costs[x] = r * fa + miss
                costs["알림 안 함"] = n_ev
                costs["매일 알림"] = r * (n - n_ev)
                best = min(costs, key=costs.get)
                out.append({"지역": REGION_KO[region], "사건": label, "헛알림 비용비 r(놓침=1)": r, "최적 임계값(최대 POP)": best,
                            "기대 비용(최적)": round(costs[best], 1), "기대 비용(60%)": round(costs[60], 1), "기대 비용(알림 안 함)": n_ev,
                            "비용 순서": " < ".join(f"{k2}({v:.1f})" for k2, v in sorted(costs.items(), key=lambda kv: kv[1])[:4])})
    return pd.DataFrame(out)


def figure(slot_rel, day_rel, thr, month):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), layout="constrained")
    # (a) 예보 시각 단위 신뢰도
    ax = axes[0, 0]
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--", label="완전 일치")
    xpos = {"0": 0, "10~20": 20, "30": 30, "60": 60, "70 이상": 75}
    for region in ["gangnam", "chuncheon"]:
        s = slot_rel[slot_rel.지역 == REGION_KO[region]].copy()
        s["x"] = s["강수확률(%)"].astype(str).map(xpos)
        s = s.sort_values("x")
        ax.plot(s.x, s["실제 강수 0.1mm+ 비율"] * 100, marker="o", color=REGION_COLOR[region], lw=1.4, label=f"{REGION_KO[region]} 0.1mm 이상")
        ax.plot(s.x, s["실제 강수 1mm+ 비율"] * 100, marker="s", ms=4, color=REGION_COLOR[region], lw=1, ls=":", label=f"{REGION_KO[region]} 1mm 이상")
        for _, r in s.iterrows():
            ax.annotate(f"{int(r['예보 수'])}", (r.x, r["실제 강수 0.1mm+ 비율"] * 100), textcoords="offset points", xytext=(0, 6 if region == "chuncheon" else -12), ha="center", fontsize=7, color=INK2)
    ax.set_xticks([0, 20, 30, 60, 75])
    ax.set_xticklabels(["0", "10~20", "30", "60", "70 이상"])
    ax.set_xlabel("예보 강수확률(%). 기상청은 40·50%를 발표하지 않는다")
    ax.set_ylabel("뒤이은 3시간에 실제로 비가 온 비율(%)")
    ax.set_title("(a) 예보 시각 단위: 강수확률 대 실제 강수(숫자는 예보 수)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(-3, 103)
    ax.set_ylim(0, 103)
    # (b) 하루 단위 신뢰도(춘천)
    ax = axes[0, 1]
    s = day_rel[day_rel.지역 == "춘천"]
    xs = np.arange(len(s))
    for j, (col, color, lab) in enumerate([("비 1mm+ 비율", "#9bc0ea", "1mm+"), ("비 5mm+ 비율", "#5a98e0", "5mm+"), ("비 10mm+ 비율", REGION_COLOR["gangnam"], "10mm+"), ("비 20mm+ 비율", "#104281", "20mm+")]):
        ax.bar(xs + (j - 1.5) * 0.2, s[col] * 100, width=0.19, color=color, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}\n({n}일)" for b, n in zip(s["영업시간 최대 강수확률(%)"], s["일수"])], fontsize=8)
    ax.set_xlabel("영업시간 최대 강수확률(%)")
    ax.set_ylabel("실제로 온 날의 비율(%)")
    ax.set_title("(b) 하루 단위(춘천): 최대 강수확률 구간별 영업시간 비", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, title="영업시간 강수", title_fontsize=8)
    # (c) 임계값 스윕(춘천, 10mm 사건)
    ax = axes[1, 0]
    t = thr[(thr.지역 == "춘천") & (thr.요일 == "전체") & (thr.사건 == "영업시간 비 10mm+") & thr.규칙.str.startswith("최대 POP")].copy()
    t["x"] = t.규칙.str.extract(r"(\d+)").astype(int)
    t = t.sort_values("x")
    ax.plot(t.x, t["적중률 POD"] * 100, marker="o", color=REGION_COLOR["chuncheon"], label="적중률 POD")
    ax.plot(t.x, t["헛경보율 FAR"] * 100, marker="s", color=INK2, label="헛경보율 FAR")
    ax.plot(t.x, t["알림일 비중(%)"], marker="^", color=MUTED, label="알림일 비중")
    ax.plot(t.x, t["CSI"] * 100, marker="d", color=REGION_COLOR["gangnam"], label="CSI")
    ax.axvline(60, color=MUTED, lw=0.8, ls="--")
    ax.text(66, 80, "40·50%는 발표되지 않아", fontsize=8, color=INK2)
    ax.text(66, 74, "40~60% 기준은 같은 날을 고른다", fontsize=8, color=INK2)
    ax.set_xlabel("알림 임계값: 영업시간 최대 강수확률(%)")
    ax.set_ylabel("%")
    ax.set_title("(c) 임계값별 적중·헛경보·알림일(춘천, 영업시간 비 10mm+)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 105)
    # (d) 월별 알림 일수(춘천)
    ax = axes[1, 1]
    m = month[month.지역 == "춘천"]
    xs = np.arange(len(m))
    ax.bar(xs - 0.3, m["관측 10mm+ 일수"], width=0.18, color=INK, label="관측 10mm+")
    for j, (x, color) in enumerate([(50, "#f3a988"), (60, REGION_COLOR["chuncheon"]), (70, "#a8471f")]):
        ax.bar(xs - 0.1 + j * 0.2, m[f"최대 POP {x}%+ 알림"], width=0.18, color=color, label=f"최대 POP {x}%+ 알림")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(v)}월" for v in m.월])
    ax.set_ylabel("일수")
    ax.set_title("(d) 월별 알림 일수와 관측 호우 일수(춘천, 2025)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    finish(fig, C.FIG, "rc06_pop_reliability", "단기예보 강수확률의 신뢰도와 알림 임계값(2025년 7~12월, D-1 17시 발표)")
    plt.close(fig)


def main():
    pop, W, wd, fd, cal = load()
    s = slot_table(pop, W)
    wt = window_test(s)
    pooled = wt[(wt.지역 == "합계") & (wt.사건 == "0.1mm+")].sort_values("Brier 기술 점수", ascending=False)
    best = pooled.iloc[0]["시간 창"]
    print("=== 시간 창 비교(합계, 0.1mm+)")
    print(pooled.to_string(index=False))
    print(f"→ 신뢰도 표에 쓰는 창: {best}")
    rel = slot_reliability(s, best)
    relg = slot_reliability(s, best, grouped=True)
    print("=== 예보 시각 단위 신뢰도(발표 등급으로 묶음)")
    print(relg.to_string(index=False))
    d = daily_table(pop, wd, fd, cal)
    dr = day_reliability(d)
    print("=== 하루 단위 신뢰도(춘천)")
    print(dr[dr.지역 == "춘천"].to_string(index=False))
    thr = threshold_table(d)
    show = thr[(thr.요일 == "전체") & (thr.사건 == "영업시간 비 10mm+")]
    print("=== 임계값·규칙별 적중(전체, 10mm 사건, 춘천)")
    print(show[show.지역 == "춘천"].drop(columns=["요일", "사건", "일수"]).to_string(index=False))
    print("=== 합계")
    print(show[show.지역 == "합계"].drop(columns=["요일", "사건", "일수", "놓친 날(영업시간 mm)"]).to_string(index=False))
    mt = month_table(d)
    cl = costloss(d)
    print("=== 비용비별 최적 임계값(춘천)")
    print(cl[cl.지역 == "춘천"].drop(columns=["지역"]).to_string(index=False))
    for name, df in [("rc_pop_window", wt), ("rc_pop_reliability_slot", rel), ("rc_pop_reliability_slot_grouped", relg), ("rc_pop_reliability_day", dr), ("rc_pop_threshold", thr), ("rc_pop_threshold_month", mt), ("rc_pop_costloss", cl)]:
        df.to_csv(C.TAB / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  저장: {name}.csv {len(df)}행")
    figure(relg, dr, thr, mt)
    print("  그림: rc06_pop_reliability.png")


if __name__ == "__main__":
    main()
