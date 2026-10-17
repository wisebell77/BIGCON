"""90. 손실 단가표: 지역×업종×요일×강수 구간별 예상 손실과 업소당 정액, 트리거 문턱별 시즌 기대 지급액.

아이디어 2(지수형 휴업손실 지원·보험)의 근거표. 보험 요율 그 자체가 아니라 요율 산정과 지원 단가의 기초(기대 손실)다.
2단계 트리거: 경보는 영업시간 비 10mm(정보·행동 안내), 지급은 20mm 이상(강남 10~20mm 효과 −3.5%로 약하고 잡음이 커서). r_trigger_options.csv에 문턱별 발동 일수·기대 지급액을 낸다. 80번과 같은 엔진(시간대 β × 요일·시간대 평상 매출)을 쓰되,
강수 구간(영업시간 06~24시 강수 10~20 / 20~40 / 40mm+)마다 관측된 '시간대 분포'를 곱해 구간 대표 손실을 만든다.
- 시간대 분포: 2025년 6월~2026년 1월 강남·춘천의 영업시간 비 10mm+ 날을 구간별로 모아 06~11 / 12~17 / 18~23시 강수 비중과 평균 강수량을 구한다(지역 합동).
- 손실: Σ_시간대 평상 매출(요일·시간대) × (1 − exp(β_시간대)), β는 그 시간대 강수가 5mm 이상이면 r5p, 1~5mm면 r1_5(증거 규칙 β, 사후 기대 β는 참고 열).
- 폭염일·열대야: 일 β × 요일 평상 매출(대부분 0 → 지급 대상이 아니라는 근거).
- 시즌 순보험료(2026년 7~12월, 업소 1곳): Σ_날 P(영업시간 비 10mm+ | 날짜) × Σ_구간 P(구간 | 비) × 손실(구간, 요일) / 업소 수.
  P(비 | 날짜)는 80번의 기후학적 확률 모델(a_expected_loss_calendar_2026), P(구간 | 비)는 1991~2025 7~12월 일강수량 분포(일 합계로 근사).
출력: r_rate_table.csv(긴 형식), r_rate_per_store.csv(업소당 요약), r_band_profile.csv, outputs/demo/demo_data.json(데모 페이지용).
"""
import json

import numpy as np
import pandas as pd

import common as C
from analysis import exclusion_days
import priority as P

BANDS = {"10~20mm": (10, 20), "20~40mm": (20, 40), "40mm+": (40, 1e9)}
TBS = ["06_11", "12_17", "18_23"]
DOW_KO = ["월", "화", "수", "목", "금", "토", "일"]


def band_profile():
    """구간별 시간대 강수 비중과 평균 강수량(지역 합동), 장기 자료의 P(구간 | 비 10mm+)."""
    wtb = C.load("weather_tb")
    wd = C.load("weather_daily")[["region", "date", "rn_biz"]]
    t = wtb[wtb.tb.isin(TBS)].pivot_table(index=["region", "date"], columns="tb", values="rn_sum").reindex(columns=TBS).fillna(0).reset_index()
    t = t.merge(wd, on=["region", "date"])
    t = t[t.rn_biz >= 10]
    rows = []
    for band, (lo, hi) in BANDS.items():
        g = t[(t.rn_biz >= lo) & (t.rn_biz < hi)]
        tot = g[TBS].sum(axis=1).replace(0, np.nan)
        share = (g[TBS].div(tot, axis=0)).mean()
        rows.append({"band": band, "n_days": len(g), "mean_rn_biz": g.rn_biz.mean(), **{f"share_{tb}": share[tb] for tb in TBS},
                     **{f"rn_{tb}": share[tb] * g.rn_biz.mean() for tb in TBS}})
    prof = pd.DataFrame(rows)
    # 장기: 7~12월 일강수 10mm+ 날 중 구간 비중(지역별)
    long = C.load("weather_long_daily")
    lg = long[long.date.dt.month.between(7, 12) & (long.rn_sum >= 10)]
    pb = {}
    for region, g in lg.groupby("region"):
        pb[region] = {band: float(((g.rn_sum >= lo) & (g.rn_sum < hi)).mean()) for band, (lo, hi) in BANDS.items()}
    return prof, pb


def main():
    cal = C.load("calendar")
    EXCL = exclusion_days(cal)
    panel = C.load("model_panel_day")
    tb_panel = C.load("model_panel_tb")
    im = C.industry_map()[["industry", "group", "small_biz"]]
    st = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")[["region", "industry", "n_stores", "grade_V", "V"]]
    prof, pband = band_profile()
    prof.round(3).to_csv(C.TAB / "r_band_profile.csv", index=False, encoding="utf-8-sig")
    print("강수 구간 프로필(지역 합동):"); print(prof.round(2).to_string(index=False))

    print("민감도 추정(시간대·일 β)...")
    btb = P.betas_tb_from_panel(tb_panel)
    bday = P.betas_from_panel(panel)
    base_tb = P.base_sales_tb_dow(tb_panel, EXCL)
    base = P.base_sales_dow(panel, EXCL)
    x = base.merge(base_tb, on=["region", "industry", "dow"], how="left").merge(btb, on=["region", "industry"], how="left").merge(bday, on=["region", "industry"], how="left")
    x = x.merge(im, on="industry", how="left").merge(st, on=["region", "industry"], how="left")
    x = x[x.small_biz == 1].copy()  # 지급 대상은 소상공인 중심 업종
    x["dow_ko"] = x.dow.map(dict(enumerate(DOW_KO)))

    def band_loss(df, band, prefix="beta"):
        pr = prof.set_index("band").loc[band]
        loss = np.zeros(len(df))
        for tb in TBS:
            rn = pr[f"rn_{tb}"]
            b = df[f"{prefix}_tb_r5p_{tb}"] if rn >= 5 else (df[f"{prefix}_tb_r1_5_{tb}"] if rn >= 1 else 0.0)
            loss = loss + df[f"base_{tb}"].fillna(0).values * (1 - np.exp(np.minimum(np.asarray(b, dtype=float), 0)))
        return loss

    rows = []
    for band in BANDS:
        d = x.copy()
        d["band"] = band
        d["expected_loss"] = band_loss(d, band)
        d["expected_loss_post"] = band_loss(d, band, "beta_post")
        rows.append(d)
    for ev, name in [("heat_day", "폭염일"), ("tropical", "열대야")]:
        d = x.copy()
        d["band"] = name
        d["expected_loss"] = d.base_sales.fillna(0) * (1 - np.exp(d[f"beta_{ev}"].fillna(0).clip(upper=0)))
        d["expected_loss_post"] = d.base_sales.fillna(0) * (1 - np.exp(np.minimum(d[f"beta_post_{ev}"].fillna(0), 0)))
        rows.append(d)
    R = pd.concat(rows, ignore_index=True)
    R["per_store_만원"] = R.expected_loss / R.n_stores / 1e4
    R["loss_rate_%"] = R.expected_loss / R.base_sales.replace(0, np.nan) * 100
    out_cols = ["region", "industry", "group", "grade_V", "n_stores", "dow", "dow_ko", "band", "base_sales", "expected_loss", "expected_loss_post", "per_store_만원", "loss_rate_%"]
    R[out_cols].round(2).to_csv(C.TAB / "r_rate_table.csv", index=False, encoding="utf-8-sig")

    # 업소당 요약: 평일 평균 / 토 / 일 × 구간
    R["dow_grp"] = np.where(R.dow <= 4, "평일", np.where(R.dow == 5, "토", "일"))
    S = R.groupby(["region", "industry", "group", "grade_V", "n_stores", "dow_grp", "band"], dropna=False).agg(expected_loss=("expected_loss", "mean"), base_sales=("base_sales", "mean")).reset_index()
    S["per_store_만원"] = S.expected_loss / S.n_stores / 1e4
    W = S.pivot_table(index=["region", "industry"], columns=["band", "dow_grp"], values="per_store_만원").reset_index()
    W.columns = [c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}·{c[1]}") for c in W.columns]
    meta = x.drop_duplicates(["region", "industry"])[["region", "industry", "group", "grade_V", "n_stores"]]
    W = meta.merge(W, on=["region", "industry"], how="left")  # 업소 수가 없는 업종도 행을 유지한다
    # 시즌 순보험료(2026년 7~12월): Σ_날 p_rain(날) × Σ_구간 P(구간|비) × 손실(구간, 요일)
    calp = pd.read_csv(C.TAB / "a_expected_loss_calendar_2026.csv", encoding="utf-8-sig", parse_dates=["date"])
    calp["dow"] = calp.date.dt.dayofweek
    L = R[R.band.isin(BANDS)].pivot_table(index=["region", "industry", "dow"], columns="band", values="expected_loss")
    prem = []
    for (region, industry), g in L.groupby(level=[0, 1]):
        gl = g.droplevel([0, 1])
        c = calp[calp.region == region]
        tot = 0.0
        for _, r in c.iterrows():
            if r.dow in gl.index:
                tot += r.p_rain_biz10 * sum(pband[region][b] * gl.loc[r.dow, b] for b in BANDS)
        prem.append({"region": region, "industry": industry, "season_premium": tot})
    prem = pd.DataFrame(prem)
    W = W.merge(prem, on=["region", "industry"], how="left")
    W["시즌 기대 지급액(억, 10mm 트리거·100% 보전)"] = W.season_premium / 1e8
    W["시즌 기대 지급액 업소당(만원)"] = W.season_premium / W.n_stores / 1e4
    # 지급 트리거(20mm 이상)만 더한 값
    prem20 = []
    for (region, industry), g in L.groupby(level=[0, 1]):
        gl = g.droplevel([0, 1]); c = calp[calp.region == region]
        prem20.append({"region": region, "industry": industry, "season_premium_20": sum(r.p_rain_biz10 * sum(pband[region][b] * gl.loc[r.dow, b] for b in ["20~40mm", "40mm+"]) for _, r in c.iterrows() if r.dow in gl.index)})
    W = W.merge(pd.DataFrame(prem20), on=["region", "industry"], how="left")
    W["시즌 기대 지급액(억, 20mm 트리거·100% 보전)"] = W.season_premium_20 / 1e8
    W["시즌 기대 지급액 업소당(만원, 20mm 트리거)"] = W.season_premium_20 / W.n_stores / 1e4
    W = W.sort_values(["region", "season_premium"], ascending=[True, False])
    W.round(2).to_csv(C.TAB / "r_rate_per_store.csv", index=False, encoding="utf-8-sig")
    tot = W.groupby("region")[["시즌 기대 지급액(억, 10mm 트리거·100% 보전)", "시즌 기대 지급액(억, 20mm 트리거·100% 보전)"]].sum().round(1)
    print("시즌 기대 지급액 합계(억, 소상공인 업종):"); print(tot.to_string()); print("P(구간|비):", {k: {b: round(v, 2) for b, v in d.items()} for k, d in pband.items()})
    for region in C.REGIONS:
        w = W[W.region == region].head(8)
        print(f"-- {C.REGION_NAME[region]} 업소당(만원) 상위: ", w[["industry", "20~40mm·평일", "20~40mm·토", "시즌 기대 지급액 업소당(만원, 20mm 트리거)"]].round(1).to_string(index=False))
    # 트리거 문턱별 기대 발동 일수·기대 지급액(지역 합계)
    opts = []
    for region in C.REGIONS:
        c = calp[calp.region == region]; exp_days = c.p_rain_biz10.sum()
        Lr = L.groupby(level=[0, 2]).sum().loc[region]  # 업종 합계 → 요일 × 구간
        for name, bands in [("10mm 이상", list(BANDS)), ("20mm 이상", ["20~40mm", "40mm+"]), ("40mm 이상", ["40mm+"])]:
            pay = sum(r.p_rain_biz10 * sum(pband[region][b] * Lr.loc[r.dow, b] for b in bands) for r in c.itertuples() if r.dow in Lr.index)
            opts.append({"region": region, "지역": C.REGION_NAME[region], "지급 트리거": name, "기대 발동 일수(7~12월)": round(exp_days * sum(pband[region][b] for b in bands), 1),
                         "기대 지급액 100% 보전(억)": round(pay / 1e8, 1), "기대 지급액 50% 보전(억)": round(pay / 2e8, 1), "2025년 실제 해당 일수": None})
    e = pd.read_csv(C.TAB / "e_summary.csv", encoding="utf-8-sig", index_col=0)
    wd = C.load("weather_daily")
    for o in opts:
        thr = {"10mm 이상": 10, "20mm 이상": 20, "40mm 이상": 40}[o["지급 트리거"]]
        w25 = wd[(wd.region == o["region"]) & (wd.date >= "2025-07-01") & (wd.date <= "2025-12-31")]
        o["2025년 실제 해당 일수"] = int((w25.rn_biz >= thr).sum())
    opts = pd.DataFrame(opts)
    opts.to_csv(C.TAB / "r_trigger_options.csv", index=False, encoding="utf-8-sig")
    print("트리거 문턱별 기대 발동 일수·지급액:"); print(opts.drop(columns="region").to_string(index=False))

    # 데모 페이지용 데이터
    demo = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"), "bands": {b: {"lo": lo, "hi": (None if hi > 1e8 else hi)} for b, (lo, hi) in BANDS.items()},
            "band_profile": prof.round(4).to_dict(orient="records"), "p_band_given_rain": pband, "trigger_options": opts.drop(columns="지역").to_dict(orient="records"), "regions": {}}
    emd = pd.read_csv(C.TAB / "st_emd.csv", encoding="utf-8-sig")
    prem_lk = prem.set_index(["region", "industry"]).season_premium
    prem20_lk = W.set_index(["region", "industry"]).season_premium_20
    wk = calp.assign(week_start=calp.date - pd.to_timedelta(calp.dow, unit="D")).groupby(["region", "week_start"]).agg(loss=("exp_loss", "sum"), p_rain=("p_rain_biz10", "mean"), p_heat=("p_heat_day", "mean")).reset_index()
    for region in C.REGIONS:
        xr = x[x.region == region]
        inds = []
        for ind, g in xr.groupby("industry"):
            g = g.set_index("dow").reindex(range(7))
            first = xr[xr.industry == ind].iloc[0]
            inds.append({"industry": ind, "group": first.group, "grade": first.grade_V if isinstance(first.grade_V, str) else "해당 없음", "n_stores": None if pd.isna(first.n_stores) else int(first.n_stores),
                         "V": None if pd.isna(first.V) else round(float(first.V), 3),
                         "base": [[round(float(g.loc[d, f"base_{tb}"]) if pd.notna(g.loc[d, f"base_{tb}"]) else 0.0) for tb in TBS] + [round(float(g.loc[d, "base_sales"])) if pd.notna(g.loc[d, "base_sales"]) else 0.0] for d in range(7)],
                         "beta": {"r1_5": [round(float(first[f"beta_tb_r1_5_{tb}"]), 4) for tb in TBS], "r5p": [round(float(first[f"beta_tb_r5p_{tb}"]), 4) for tb in TBS],
                                  "heat": round(float(first.beta_heat_day), 4), "tropical": round(float(first.beta_tropical), 4)},
                         "premium_per_store_만원": (None if (pd.isna(first.n_stores) or (region, ind) not in prem_lk.index) else round(float(prem_lk.loc[(region, ind)]) / float(first.n_stores) / 1e4, 1)),
                         "premium20_per_store_만원": (None if (pd.isna(first.n_stores) or (region, ind) not in prem20_lk.index) else round(float(prem20_lk.loc[(region, ind)]) / float(first.n_stores) / 1e4, 1))})
        e = emd[emd.region == region]
        demo["regions"][region] = {"name": C.REGION_NAME[region], "industries": inds,
                                   "emd": [{"name": r.emd_name, "n_face": int(r.n_face), "n_vuln": int(r.n_vuln), "vuln_share": round(float(r.vuln_share), 3), "rain_loss_day": round(float(r.rain_loss_day)),
                                            "pop60": None if pd.isna(r.pop60) else round(float(r.pop60)), "share60": None if pd.isna(r.share_age_60p) else round(float(r.share_age_60p), 3),
                                            "cx": round(float(r.cx)), "cy": round(float(r.cy))} for r in e.itertuples()],
                                   "calendar": [{"week": r.week_start.strftime("%m-%d"), "loss": round(float(r.loss) / 1e8, 2), "p_rain": round(float(r.p_rain), 3), "p_heat": round(float(r.p_heat), 3)} for r in wk[wk.region == region].itertuples()]}
    demo_dir = C.ROOT / "outputs" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "demo_data.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")
    print(f"저장: r_rate_table.csv({len(R):,}행), r_rate_per_store.csv({len(W)}행), r_band_profile.csv, r_trigger_options.csv, outputs/demo/demo_data.json({(demo_dir / 'demo_data.json').stat().st_size / 1e3:.0f}KB)")


if __name__ == "__main__":
    main()
