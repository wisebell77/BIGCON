"""분석·모델링용 통합 패널.

- model_panel_day: 지역 × 업종 × 일 + 달력 + 업종 특성(p_) + 거주지 구성 + 일 날씨(w_)
- model_panel_tb : 지역 × 업종 × 일 × 시간대 + 달력 + 시간대 날씨(wtb_) + 일 날씨 일부(w_)

in_scope: 소비성 업종 중 거래가 꾸준한 시리즈(기본 기준은 아래 상수). 민감도 추정과 모델은 이 시리즈로 한다.
SK 유동인구는 일별 값이 없어 패널에 넣지 않는다(지수 단계에서 지역 노출 구조로만 쓴다).
"""
import numpy as np
import pandas as pd

from common import OUT, load, save

SCOPE_MIN_COVERAGE = 0.95  # 184일 중 거래가 있는 날 비율
SCOPE_MIN_CNT_DAY = 20     # 하루 평균 결제건수

FLAG_COLS = ["group", "consumer", "small_biz", "discretionary", "outdoor"]
PROFILE_COLS = ["sales_day", "cnt_day", "ticket", "coverage",
                "share_tb_00_05", "share_tb_06_11", "share_tb_12_17", "share_tb_18_23",
                "share_age_10-20대", "share_age_30-50대", "share_age_60대+", "share_sex_F",
                "ratio_weekend", "ratio_chuseok", "cv_workday", "share_corp", "share_nonlocal"]
CAL_COLS = ["ym", "month", "day", "dow", "week", "days_to_month_end", "is_holiday", "is_weekend", "is_offday",
            "off_block_len", "off_block_pos", "is_pre_long_off", "is_post_long_off", "is_sandwich",
            "is_chuseok_block", "coupon_phase", "is_coupon_last_week", "payback_on", "is_special"]
DAY_WEATHER_FOR_TB = ["rn_biz", "heat_day", "tropical", "tropical_prev", "rain_biz10", "rain_biz20", "feels_max", "ta_max_off"]


def add_history(df, keys, y="y"):
    """과거 매출 피처. 7일 이상 지난 값만 써서 당일·전날 날씨 영향이 섞이지 않게 한다.
    날씨 효과를 추정하는 회귀에는 넣지 말고, 예측 모형에서만 쓴다."""
    df = df.sort_values(keys + ["date"]).copy()
    g = df.groupby(keys, sort=False)[y]
    lags = {k: g.shift(k) for k in (7, 14, 21, 28)}
    df[f"{y}_lag7"], df[f"{y}_lag14"] = lags[7], lags[14]
    df[f"{y}_dow_mean4"] = pd.concat(lags.values(), axis=1).mean(axis=1)  # 직전 4주 같은 요일 평균
    df[f"{y}_ma7_lag7"] = g.transform(lambda s: s.shift(7).rolling(7, min_periods=4).mean())
    return df


def scope_table(prof):
    scope = prof[["region", "industry"] + FLAG_COLS].copy()
    scope["in_scope"] = ((prof["consumer"] == 1) & (prof["coverage"] >= SCOPE_MIN_COVERAGE)
                         & (prof["cnt_day"] >= SCOPE_MIN_CNT_DAY)).astype(int)
    return scope


def merge_weather(df, name, keys, prefix, cols=None):
    path = OUT / f"{name}.parquet"
    if not path.exists():
        print(f"  {name}.parquet 없음 → 날씨 없이 만든다(04_weather.py 실행 후 다시 돌리면 붙는다)")
        return df
    w = pd.read_parquet(path)
    if cols:
        w = w[keys + cols]
    w = w.rename(columns={c: f"{prefix}{c}" for c in w.columns if c not in keys})
    print(f"  {name} 결합: {w.shape[1] - len(keys)}개 열")
    return df.merge(w, on=keys, how="left")


def main():
    cal = load("calendar")[["date"] + CAL_COLS]
    prof = load("industry_profile")
    scope = scope_table(prof)
    keys = ["region", "industry"]

    print("[1] 일 패널")
    day = load("card_day")
    resid = load("card_day_resid")[keys + ["date", "sales_local", "sales_nonlocal", "sales_unknown", "share_nonlocal"]]
    day = day.merge(resid.rename(columns={"share_nonlocal": "share_nonlocal_d"}), on=keys + ["date"], how="left")
    day = day.merge(cal, on="date", how="left").merge(scope, on=keys, how="left")
    p = prof[keys + PROFILE_COLS].rename(columns={c: f"p_{c}" for c in PROFILE_COLS})
    day = day.merge(p, on=keys, how="left")
    day["y"] = np.log1p(day["sales"]).where(day["sales"] > 0)  # 거래가 없는 날은 결측(휴업·결측으로 본다)
    day["y_cnt"] = np.log1p(day["cnt"]).where(day["cnt"] > 0)
    day["ticket"] = day["sales"] / day["cnt"].replace(0, np.nan)
    day = add_history(day, keys)
    day = merge_weather(day, "weather_daily", ["region", "date"], "w_")
    save(day, "model_panel_day")
    s = scope[scope.in_scope == 1]
    print(f"  in_scope 시리즈 {len(s)}개 (강남 {(s.region == 'gangnam').sum()}, 춘천 {(s.region == 'chuncheon').sum()})")

    print("[2] 시간대 패널")
    tb = load("card_day_tb").merge(cal, on="date", how="left").merge(scope, on=keys, how="left")
    tb["y"] = np.log1p(tb["sales"]).where(tb["sales"] > 0)
    tb = add_history(tb, keys + ["tb"])
    tb = merge_weather(tb, "weather_tb", ["region", "date", "tb"], "wtb_")
    tb = merge_weather(tb, "weather_daily", ["region", "date"], "w_", cols=DAY_WEATHER_FOR_TB)
    save(tb, "model_panel_tb")


if __name__ == "__main__":
    main()
