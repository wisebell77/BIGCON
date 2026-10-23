"""신한카드 데이터 정제와 분석용 패널.

- 데이터1(주제1용): 일자 × 시간대 × 시군구 × 업종 × 성별(남/여/법인) × 연령
- 데이터2(주제2용 레이아웃): 일자 × 시군구 × 업종 × 고객 거주 시도 × 성별 × 연령(법인 없음)

정제 원칙
- 결제건수 최솟값이 5다. 작은 칸은 비식별 처리로 값이 조정된 것으로 보인다. 데이터1(개인)과 데이터2의
  지역×업종×일 합계가 몇 원 차이로 같으므로 합계는 보존된다. 행이 없는 날은 0으로 채우고 행 수(n_rows)를 남긴다.
- 법인 결제는 개인 매출과 분리한다. 강남 세금공과금·ZZ_나머지의 법인 결제가 극단값이다.
- 업종 성격(업종군, 소비성, 소상공인 중심 등)은 code/meta/industry_map.csv에서 붙인다.

출력
- card1_clean, card2_clean: 정제한 원자료(지역·업종·성별·연령 코드 통일)
- card_day: 지역 × 업종 × 일 (개인 매출·건수, 법인은 *_corp 열)
- card_day_tb: 지역 × 업종 × 일 × 시간대
- card_day_age: 지역 × 업종 × 일 × 연령군(3구간)
- card_day_tb_age: 지역 × 업종 × 일 × 시간대 × 연령군
- card_day_resid: 지역 × 업종 × 일 × 거주지(지역민/외지인/미상), 데이터2 기준
- industry_profile: 지역 × 업종 고정 특성(규모, 객단가, 시간대·연령·성별 구성, 주말 비, 외지인 비중 등)
"""
import numpy as np
import pandas as pd

from common import (AGE_GROUP, AGE_GROUPS, CARD1, CARD2, DATES, LOCAL_SIDO, REGION_BY_CARD,
                    TIME_BANDS, industry_map, load, save)


def read_card(path):
    raw = pd.read_csv(path, sep="\t", encoding="cp949", dtype=str)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TA_YMD"], format="%Y%m%d"),
        "region": raw["MCT_SGG_CD"].map(REGION_BY_CARD),
        "industry": raw["MCT_RY_CD"],
        "sex": raw["SEX_CCD"].map({"남성": "M", "여성": "F", "법인": "CORP"}),
        "age": pd.to_numeric(raw["AGE_CCD"].str.replace("대", "").str.strip(), errors="coerce").astype("Int64"),
        "sales": raw["TS_AT"].astype("int64"),
        "cnt": raw["USE_CNT"].astype("int64"),
    })
    if "TIME_GB" in raw:
        df.insert(1, "tb", raw["TIME_GB"])
    if "CLN_SGG_CD" in raw:
        df["resid"] = raw["CLN_SGG_CD"]
    assert df["region"].notna().all() and df["sex"].notna().all()
    assert (df["age"].isna() == (df["sex"] == "CORP")).all()  # 법인만 연령이 없다
    assert set(df["industry"]) <= set(industry_map()["industry"]), "industry_map.csv에 없는 업종"
    df["age_group"] = df["age"].map(AGE_GROUP)
    print(f"  {path.name}: {len(df):,}행, {df.date.min():%Y-%m-%d}~{df.date.max():%Y-%m-%d}")
    return df


def full_grid(series, **extra):
    """(지역, 업종) 시리즈마다 184일 × 추가 축을 빠짐없이 만든다."""
    grid = series.merge(pd.DataFrame({"date": DATES}), how="cross")
    for col, values in extra.items():
        grid = grid.merge(pd.DataFrame({col: values}), how="cross")
    return grid


def aggregate(df, keys, series, value_prefix="", **extra):
    agg = df.groupby(keys, observed=True).agg(sales=("sales", "sum"), cnt=("cnt", "sum"),
                                              n_rows=("sales", "size")).reset_index()
    out = full_grid(series, **extra).merge(agg, on=keys, how="left")
    out[["sales", "cnt", "n_rows"]] = out[["sales", "cnt", "n_rows"]].fillna(0).astype("int64")
    if value_prefix:
        out = out.rename(columns={c: f"{c}{value_prefix}" for c in ["sales", "cnt", "n_rows"]})
    return out


def industry_profile(person, day, cal, corp_day):
    """(지역, 업종)별 고정 특성. 예측 성능 평가 때는 학습 기간만으로 다시 계산해야 누수가 없다."""
    keys = ["region", "industry"]
    n_days = day["date"].nunique()
    prof = day.groupby(keys).agg(sales_total=("sales", "sum"), cnt_total=("cnt", "sum"),
                                 days_obs=("n_rows", lambda s: int((s > 0).sum()))).reset_index()
    prof["sales_day"] = prof["sales_total"] / n_days
    prof["cnt_day"] = prof["cnt_total"] / n_days
    prof["ticket"] = prof["sales_total"] / prof["cnt_total"].replace(0, np.nan)  # 객단가(원/건)
    prof["coverage"] = prof["days_obs"] / n_days

    def shares(col, levels, prefix, value="sales"):
        t = person.pivot_table(index=keys, columns=col, values=value, aggfunc="sum", fill_value=0, observed=True)
        t = t.reindex(columns=levels, fill_value=0)
        return t.div(t.sum(axis=1).replace(0, np.nan), axis=0).add_prefix(prefix).reset_index()

    prof = prof.merge(shares("tb", TIME_BANDS, "share_tb_"), on=keys, how="left")
    prof = prof.merge(shares("age_group", AGE_GROUPS, "share_age_"), on=keys, how="left")
    prof = prof.merge(shares("age_group", AGE_GROUPS, "cntshare_age_", value="cnt"), on=keys, how="left")
    prof = prof.merge(shares("sex", ["F", "M"], "share_sex_").drop(columns="share_sex_M"), on=keys, how="left")

    d = day.merge(cal[["date", "is_weekend", "is_holiday", "is_offday", "is_chuseok_block"]], on="date")
    normal = d[d.is_offday == 0].groupby(keys)["sales"].mean()
    weekend = d[(d.is_weekend == 1) & (d.is_holiday == 0)].groupby(keys)["sales"].mean()
    chuseok = d[d.is_chuseok_block == 1].groupby(keys)["sales"].mean()
    cv = d[d.is_offday == 0].groupby(keys)["sales"].agg(lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan)
    extra = pd.DataFrame({"ratio_weekend": weekend / normal, "ratio_chuseok": chuseok / normal,
                          "cv_workday": cv}).reset_index()
    prof = prof.merge(extra, on=keys, how="left")

    corp = corp_day.groupby(keys)["sales_corp"].sum().rename("corp_sales_total").reset_index()
    prof = prof.merge(corp, on=keys, how="left").fillna({"corp_sales_total": 0})
    prof["share_corp"] = prof["corp_sales_total"] / (prof["corp_sales_total"] + prof["sales_total"]).replace(0, np.nan)
    return prof


def residence_panel(c2, series):
    """데이터2: 지역민 / 외지인 / 거주지 미상 결제액."""
    local_sido = c2["region"].map(LOCAL_SIDO)
    c2 = c2.assign(origin=np.select([c2["resid"].eq("정보없음"), c2["resid"].eq(local_sido)],
                                    ["unknown", "local"], "nonlocal"))
    keys = ["region", "industry", "date"]
    t = c2.pivot_table(index=keys, columns="origin", values=["sales", "cnt"], aggfunc="sum", fill_value=0)
    t.columns = [f"{v}_{o}" for v, o in t.columns]
    out = full_grid(series).merge(t.reset_index(), on=keys, how="left")
    val_cols = [f"{v}_{o}" for v in ["sales", "cnt"] for o in ["local", "nonlocal", "unknown"]]
    out = out.reindex(columns=keys + val_cols).fillna(0)
    out[val_cols] = out[val_cols].astype("int64")
    known = out["sales_local"] + out["sales_nonlocal"]
    out["share_nonlocal"] = out["sales_nonlocal"] / known.replace(0, np.nan)
    return out, c2


def main():
    cal = load("calendar")

    print("[1] 데이터1 정제")
    c1 = read_card(CARD1)
    save(c1, "card1_clean")
    series = c1[["region", "industry"]].drop_duplicates().sort_values(["region", "industry"]).reset_index(drop=True)
    person, corp = c1[c1["sex"] != "CORP"], c1[c1["sex"] == "CORP"]
    print(f"  시리즈(지역×업종) {len(series)}개, 개인 {len(person):,}행 / 법인 {len(corp):,}행")

    print("[2] 데이터1 패널 (개인 결제 기준, 법인은 별도 열)")
    day = aggregate(person, ["region", "industry", "date"], series)
    corp_day = aggregate(corp, ["region", "industry", "date"], series, value_prefix="_corp")
    day = day.merge(corp_day, on=["region", "industry", "date"])
    save(day, "card_day")
    save(aggregate(person, ["region", "industry", "date", "tb"], series, tb=TIME_BANDS), "card_day_tb")
    save(aggregate(person, ["region", "industry", "date", "age_group"], series, age_group=AGE_GROUPS), "card_day_age")
    save(aggregate(person, ["region", "industry", "date", "tb", "age_group"], series,
                   tb=TIME_BANDS, age_group=AGE_GROUPS), "card_day_tb_age")

    print("[3] 데이터2 정제와 거주지 패널")
    c2 = read_card(CARD2)
    resid, c2 = residence_panel(c2, series)
    save(c2, "card2_clean")
    save(resid, "card_day_resid")
    chk = day.merge(resid, on=["region", "industry", "date"])
    tot2 = chk[["sales_local", "sales_nonlocal", "sales_unknown"]].sum(axis=1)
    big = chk["sales"] > 0
    ratio = tot2[big] / chk.loc[big, "sales"]
    print(f"  데이터2/데이터1 일매출 비율: 중앙값 {ratio.median():.3f}, 25~75% {ratio.quantile(.25):.3f}~{ratio.quantile(.75):.3f}")

    print("[4] 업종 특성표")
    prof = industry_profile(person, day, cal, corp_day)
    nl = resid.groupby(["region", "industry"])[["sales_local", "sales_nonlocal"]].sum()
    prof = prof.merge((nl["sales_nonlocal"] / nl.sum(axis=1).replace(0, np.nan)).rename("share_nonlocal").reset_index(),
                      on=["region", "industry"], how="left")
    prof = prof.merge(industry_map(), on="industry", how="left")
    save(prof, "industry_profile")


if __name__ == "__main__":
    main()
