"""장기 기상 자료(ASOS 일자료 1991~2025) → 평년·추세·2025년 맥락.

입력: data/external/weather_long/OBS_ASOS_DD_*.csv (기상자료개방포털 종관기상관측 일자료, 서울 108·춘천 101,
      기간이 겹치거나 나뉘어 있어도 된다. 지점·일시 기준으로 합친다)
출력
- weather_long_daily: 지점×일 핵심 항목과 이벤트 플래그(1991~2025)
- weather_climatology: 지역×일(연중 1~366일)의 평년(1991~2020) 분포. 해당 날짜 앞뒤 7일을 모아(30년×15일) 평균·10·50·90퍼센타일
- weather_annual: 지역×연도 이벤트 일수·기온·강수 통계(연간, 7~12월)
- weather_2025_context: 2025-07-01~12-31 지역×일의 평년 대비 편차·백분위·이상기후 플래그(이상고온 = 평년 상위 10%)

용어: 기상청 이상기후 정의는 1991~2020 평년 대비 상·하위 10%. 폭염일은 일최고 33℃ 이상, 열대야는 밤 최저 25℃ 이상인데
일자료에는 밤 최저기온이 없어 일최저 25℃ 이상으로 대신한다(2025년 서울 44일 vs 공식 46일 수준으로 비슷하다).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXT_WEATHER, OUT, REGION_ASOS, save

SRC = EXT_WEATHER.parent / "weather_long"
NORMAL = (1991, 2020)  # 평년 기간
COLS = {
    "ta_mean": "평균기온(°C)", "ta_max": "최고기온(°C)", "ta_min": "최저기온(°C)",
    "rn_sum": "일강수량(mm)", "rn_max_1h": "1시간 최다강수량(mm)", "rn_hours": "강수 계속시간(hr)",
    "ws_mean": "평균 풍속(m/s)", "ws_max": "최대 풍속(m/s)", "hm_mean": "평균 상대습도(%)",
    "ss_sum": "합계 일조시간(hr)", "snow_new": "일 최심신적설(cm)", "cloud_mean": "평균 전운량(1/10)",
}
REGION_OF = {stn: region for region, stn in REGION_ASOS.items()}


def read_long():
    frames = []
    for f in sorted(SRC.glob("*.csv")):
        d = pd.read_csv(f, encoding="cp949")
        out = pd.DataFrame({"stn": d["지점"].astype(int), "date": pd.to_datetime(d["일시"])})
        for k, c in COLS.items():
            out[k] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
        frames.append(out)
        print(f"  {f.name}: {len(out):,}행 {out.date.min():%Y-%m-%d}~{out.date.max():%Y-%m-%d}")
    df = pd.concat(frames).drop_duplicates(["stn", "date"]).sort_values(["stn", "date"]).reset_index(drop=True)
    df["region"] = df["stn"].map(REGION_OF)
    df = df[df["region"].notna()]
    for c in ["rn_sum", "rn_max_1h", "rn_hours", "snow_new"]:
        df[c] = df[c].fillna(0)  # 기상청 일자료는 강수·적설이 없으면 빈칸
    df["year"], df["month"], df["doy"] = df.date.dt.year, df.date.dt.month, df.date.dt.dayofyear
    # 이벤트 플래그
    df["heat33"] = (df.ta_max >= 33).astype(int)
    df["heat35"] = (df.ta_max >= 35).astype(int)
    df["tropical25"] = (df.ta_min >= 25).astype(int)
    for mm in [1, 10, 30, 50, 80, 100]:
        df[f"rain{mm}"] = (df.rn_sum >= mm).astype(int)
    df["rain_1h20"] = (df.rn_max_1h >= 20).astype(int)
    df["rain_1h30"] = (df.rn_max_1h >= 30).astype(int)
    df["cold12"] = (df.ta_min <= -12).astype(int)
    df["snow5"] = (df.snow_new >= 5).astype(int)
    return df


def climatology(df):
    """지역×연중일: 평년 기간의 앞뒤 7일 창에서 평균·퍼센타일."""
    base = df[df.year.between(*NORMAL)].copy()
    base = base[base.doy <= 365]  # 윤일은 뺀다
    rows = []
    for region, g in base.groupby("region"):
        g = g.set_index("doy")
        for d in range(1, 366):
            win = [((d - 1 + k) % 365) + 1 for k in range(-7, 8)]
            s = g.loc[g.index.isin(win)]
            row = {"region": region, "doy": d, "n": len(s)}
            for c in ["ta_max", "ta_min", "ta_mean"]:
                row[f"{c}_mean"] = s[c].mean()
                row[f"{c}_p10"] = s[c].quantile(0.10)
                row[f"{c}_p90"] = s[c].quantile(0.90)
                row[f"{c}_sd"] = s[c].std()
            row["rn_mean"] = s.rn_sum.mean()
            row["rn_p90"] = s.rn_sum.quantile(0.90)
            row["rn_p95"] = s.rn_sum.quantile(0.95)
            row["rain1_prob"] = s.rain1.mean()
            row["rain10_prob"] = s.rain10.mean()
            row["heat33_prob"] = s.heat33.mean()
            row["tropical_prob"] = s.tropical25.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def annual(df):
    def agg(g):
        rain_days = g.rain1.sum()
        return pd.Series({
            "n_days": len(g), "ta_mean": g.ta_mean.mean(), "ta_max_mean": g.ta_max.mean(), "ta_max_max": g.ta_max.max(),
            "ta_min_min": g.ta_min.min(), "heat33_days": g.heat33.sum(), "heat35_days": g.heat35.sum(),
            "tropical_days": g.tropical25.sum(), "cold12_days": g.cold12.sum(),
            "rain_days": rain_days, "rain10_days": g.rain10.sum(), "rain30_days": g.rain30.sum(), "rain50_days": g.rain50.sum(),
            "rain80_days": g.rain80.sum(), "rain_1h20_days": g.rain_1h20.sum(), "rain_1h30_days": g.rain_1h30.sum(),
            "precip_total": g.rn_sum.sum(), "precip_intensity": g.rn_sum.sum() / rain_days if rain_days else np.nan,
            "rn_max_day": g.rn_sum.max(), "rn_max_1h": g.rn_max_1h.max(), "snow5_days": g.snow5.sum(),
            "summer_ta_mean": g.loc[g.month.between(6, 8), "ta_mean"].mean(),
        })
    full = df.groupby(["region", "year"]).apply(agg, include_groups=False).reset_index().assign(window="연간")
    jd = df[df.month >= 7].groupby(["region", "year"]).apply(agg, include_groups=False).reset_index().assign(window="7~12월")
    return pd.concat([full, jd], ignore_index=True)


def context_2025(df, clim):
    x = df[(df.date >= "2025-07-01") & (df.date <= "2025-12-31")].merge(clim, on=["region", "doy"], how="left")
    x["ta_max_anom"] = x.ta_max - x.ta_max_mean
    x["ta_min_anom"] = x.ta_min - x.ta_min_mean
    x["ta_max_z"] = x.ta_max_anom / x.ta_max_sd
    x["hot_extreme"] = (x.ta_max > x.ta_max_p90).astype(int)    # 기상청 이상고온 정의(평년 상위 10%)
    x["cold_extreme"] = (x.ta_max < x.ta_max_p10).astype(int)
    x["night_extreme"] = (x.ta_min > x.ta_min_p90).astype(int)
    x["rain_extreme"] = ((x.rn_sum >= 1) & (x.rn_sum > x.rn_p95)).astype(int)  # 그 시기 상위 5% 강수
    # 평년 기간 전체 분포 대비 백분위(같은 창)
    base = df[df.year.between(*NORMAL) & (df.doy <= 365)]
    pct_rows = []
    for region, g in x.groupby("region"):
        b = base[base.region == region].set_index("doy")
        for _, r in g.iterrows():
            win = [((r.doy - 1 + k) % 365) + 1 for k in range(-7, 8)]
            s = b.loc[b.index.isin(win)]
            pct_rows.append({"region": region, "date": r.date,
                             "ta_max_pct": (s.ta_max < r.ta_max).mean() * 100,
                             "ta_min_pct": (s.ta_min < r.ta_min).mean() * 100,
                             "rn_pct_rainy": (s.loc[s.rain1 == 1, "rn_sum"] < r.rn_sum).mean() * 100 if r.rn_sum >= 1 else np.nan})
    x = x.merge(pd.DataFrame(pct_rows), on=["region", "date"], how="left")
    keep = ["region", "date", "ta_max", "ta_min", "rn_sum", "ta_max_mean", "ta_max_p90", "ta_min_mean", "ta_min_p90", "rn_mean", "rn_p95",
            "ta_max_anom", "ta_min_anom", "ta_max_z", "ta_max_pct", "ta_min_pct", "rn_pct_rainy",
            "hot_extreme", "cold_extreme", "night_extreme", "rain_extreme", "heat33_prob", "rain10_prob"]
    return x[keep]


def main():
    if not SRC.exists() or not list(SRC.glob("*.csv")):
        print(f"  {SRC} 에 장기 ASOS 일자료가 없습니다. 건너뜁니다.")
        return
    print("[1] 장기 일자료 읽기")
    df = read_long()
    for region, g in df.groupby("region"):
        print(f"  {region}: {g.date.min():%Y-%m-%d}~{g.date.max():%Y-%m-%d} {len(g):,}일, 최고기온 결측 {int(g.ta_max.isna().sum())}")
    save(df, "weather_long_daily")
    print("[2] 평년(1991~2020) 분포")
    clim = climatology(df)
    save(clim, "weather_climatology")
    print("[3] 연도별 통계")
    ann = annual(df)
    save(ann, "weather_annual")
    a = ann[(ann.window == "연간")]
    for region, g in a.groupby("region"):
        n = g[g.year.between(*NORMAL)]
        y25 = g[g.year == 2025].iloc[0]
        print(f"  {region}: 폭염일 평년 {n.heat33_days.mean():.1f} → 2025 {y25.heat33_days:.0f}, 열대야 {n.tropical_days.mean():.1f} → {y25.tropical_days:.0f}, "
              f"일 50mm+ {n.rain50_days.mean():.1f} → {y25.rain50_days:.0f}, 강수강도 {n.precip_intensity.mean():.1f} → {y25.precip_intensity:.1f} mm/일")
    print("[4] 2025년 7~12월 맥락")
    ctx = context_2025(df, clim)
    save(ctx, "weather_2025_context")
    for region, g in ctx.groupby("region"):
        print(f"  {region}: 이상고온일(평년 상위 10%) {int(g.hot_extreme.sum())}, 이상 야간고온 {int(g.night_extreme.sum())}, 극한강수일(상위 5%) {int(g.rain_extreme.sum())}, "
              f"최고기온 편차 평균 {g.ta_max_anom.mean():+.1f}℃")


if __name__ == "__main__":
    main()
