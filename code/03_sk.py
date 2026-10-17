"""SK 유동인구 → 지역 단위 노출 구조.

원본: 50m 격자 셀 × 월(STD_YM)의 일평균 유동인구, 표 3종(성연령 / 시간대 / 요일). 일별 값은 없다.
그래서 날씨 효과 추정에는 쓰지 않고, '언제(시간대·요일) 누가(연령) 밖에 있는가'라는 노출 구조로만 쓴다.
팀 결정(2026-10-09): 상권 단위는 지역×업종(×시간대)이고 SK는 노출 보정에만 쓴다. 블록·동 단위 결과는 만들지 않는다.

정제 원칙
- 12월 time·wkdy 파일은 모든 행이 2번씩 들어 있어 중복을 지운다.
- BLOCK_CD 앞 5자리(통계청 시군구 코드)가 강남(11230)·춘천(32010)이 아닌 인접 지역 셀은 뺀다.
- 비중은 같은 표 안에서만 나눈다(표마다 셀 범위가 조금씩 다르다).

출력
- sk_region_month: 지역 × 월 (총량, 60대+ 비중, 시간대 비중, 주말/평일 비, 월 지수)
- sk_region_profile: 지역 × kind(hour/age/wkdy) × key 의 6개월 평균 값과 비중 (그림·노출 계산용)
- sk_region: 지역 요약(6개월 평균)
"""
import numpy as np
import pandas as pd

from common import RAW_SK, REGION_BY_SGG, TIME_BANDS, save

FILES = {"age": "flow_age_pop", "time": "flow_time_pop", "wkdy": "flow_wkdy_pop"}
AGES = ["10", "20", "30", "40", "50", "60"]
AGE_COLS = [f"{s}{a}" for s in "mf" for a in AGES]
HOURS = [f"h{h:02d}" for h in range(24)]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MONTHS = ["202507", "202508", "202509", "202510", "202511", "202512"]
TB_HOURS = {"00_05": range(0, 6), "06_11": range(6, 12), "12_17": range(12, 18), "18_23": range(18, 24)}


def rename_values(df):
    ren = {}
    for c in df.columns:
        if c.startswith(("MAN_", "WMAN_")):
            ren[c] = ("f" if c.startswith("WMAN_") else "m") + c.split("_")[-1][:2]  # m10 … m60(60대 이상)
        elif c.startswith("TMST_"):
            ren[c] = "h" + c[-2:]
        elif c.startswith("FLOW_POP_CNT_"):
            ren[c] = {"TUS": "tue"}.get(c[-3:], c[-3:].lower())
    return df.rename(columns=ren)


def read_kind(kind):
    frames = []
    for f in sorted(RAW_SK.glob(f"{FILES[kind]}_2025*.csv")):
        df = pd.read_csv(f, sep="|", encoding="utf-8-sig", dtype={"STD_YM": str, "BLOCK_CD": str})
        n0 = len(df)
        df = df.drop_duplicates()
        if len(df) < n0:
            print(f"  {f.name}: 완전 중복 {n0 - len(df):,}행 제거 ({n0:,} → {len(df):,})")
        frames.append(df)
    df = rename_values(pd.concat(frames, ignore_index=True))
    df = df.rename(columns={"STD_YM": "ym", "BLOCK_CD": "block_cd", "X_COORD": "x", "Y_COORD": "y"})
    df["region"] = df["block_cd"].str[:5].map(REGION_BY_SGG)
    print(f"  {kind}: 인접 시군구 셀 {df.region.isna().sum()}행 제외, 남은 셀×월 {df.region.notna().sum():,}")
    df = df[df["region"].notna()].copy()
    df["cell_id"] = df["x"].round().astype(int).astype(str) + "_" + df["y"].round().astype(int).astype(str)
    assert not df.duplicated(["ym", "cell_id"]).any()
    return df


def region_month(age, tim, wk):
    a = age.groupby(["region", "ym"])[AGE_COLS].sum()
    t = tim.groupby(["region", "ym"])[HOURS].sum()
    w = wk.groupby(["region", "ym"])[DAYS].sum()
    out = pd.DataFrame({"pop": a.sum(axis=1)})
    out["share_age_60p"] = (a["m60"] + a["f60"]) / out["pop"]
    out["share_age_10_20"] = a[["m10", "f10", "m20", "f20"]].sum(axis=1) / out["pop"]
    out["share_female"] = a[[f"f{g}" for g in AGES]].sum(axis=1) / out["pop"]
    for tb, hrs in TB_HOURS.items():
        out[f"share_tb_{tb}"] = t[[f"h{h:02d}" for h in hrs]].sum(axis=1) / t.sum(axis=1)
    out["ratio_weekend"] = w[["sat", "sun"]].mean(axis=1) / w[["mon", "tue", "wed", "thu", "fri"]].mean(axis=1)
    out["pop_idx"] = out["pop"] / out.groupby("region")["pop"].transform("mean")
    n_cells = age.groupby(["region", "ym"])["cell_id"].nunique()
    out["n_cells"] = n_cells
    return out.reset_index()


def region_profile(age, tim, wk):
    """지역 × (hour/age/wkdy) × key: 6개월 평균 값(value)과 같은 kind 안 비중(share)."""
    rows = []
    for kind, df, cols in [("hour", tim, HOURS), ("age", age, AGE_COLS), ("wkdy", wk, DAYS)]:
        m = df.groupby(["region", "ym"])[cols].sum().groupby("region").mean()  # 월별 합 → 6개월 평균
        share = m.div(m.sum(axis=1), axis=0)
        for region in m.index:
            for c in cols:
                rows.append({"region": region, "kind": kind, "key": c, "value": m.loc[region, c], "share": share.loc[region, c]})
    return pd.DataFrame(rows)


def region_summary(month, prof):
    cols = ["pop", "share_age_60p", "share_age_10_20", "share_female", "ratio_weekend", "n_cells"] + \
           [f"share_tb_{tb}" for tb in TIME_BANDS]
    out = month.groupby("region")[cols].mean()
    hour = prof[prof.kind == "hour"].pivot(index="region", columns="key", values="share")
    out["peak_hour"] = hour.values.argmax(axis=1)
    out["share_night_18_23"] = hour[[f"h{h:02d}" for h in range(18, 24)]].sum(axis=1)
    out["density_km2"] = out["pop"] / (out["n_cells"] * 0.0025)  # 셀 1개 = 50m × 50m
    return out.reset_index()


def cell_table(age, tim, wk):
    """셀(50m 격자) 단위 6개월 평균: 유동인구, 60대+ 비중, 시간대 비중, 주말/평일 비. 상가정보·동 단위 지도에 쓴다."""
    keys = ["region", "block_cd", "cell_id", "x", "y"]
    a = age.groupby(keys)[AGE_COLS].mean()
    t = tim.groupby(keys)[HOURS].mean()
    w = wk.groupby(keys)[DAYS].mean()
    out = pd.DataFrame({"pop": a.sum(axis=1), "n_months": age.groupby(keys).size()})
    out["share_age_60p"] = (a["m60"] + a["f60"]) / out["pop"].replace(0, np.nan)
    out["share_age_10_20"] = a[["m10", "f10", "m20", "f20"]].sum(axis=1) / out["pop"].replace(0, np.nan)
    th = t.sum(axis=1).replace(0, np.nan)
    for tb, hrs in TB_HOURS.items():
        out[f"share_tb_{tb}"] = t[[f"h{h:02d}" for h in hrs]].sum(axis=1) / th
    out["ratio_weekend"] = w[["sat", "sun"]].mean(axis=1) / w[["mon", "tue", "wed", "thu", "fri"]].mean(axis=1).replace(0, np.nan)
    out = out.reset_index()
    out["emd_cd"] = out["block_cd"].str[:8]  # 시군구 5자리 + 읍면동 3자리(통계청 코드)
    return out


def main():
    print("[1] 읽기·중복 제거·지역 필터")
    age, tim, wk = read_kind("age"), read_kind("time"), read_kind("wkdy")

    print("[2] 지역 단위 집계")
    save(cell_table(age, tim, wk), "sk_cell")
    month = region_month(age, tim, wk)
    prof = region_profile(age, tim, wk)
    save(month, "sk_region_month")
    save(prof, "sk_region_profile")
    save(region_summary(month, prof), "sk_region")
    for r, g in month.groupby("region"):
        print(f"  {r}: 셀 {int(g.n_cells.mean()):,}개, 월 지수 {g.pop_idx.round(2).tolist()}, "
              f"60대+ {g.share_age_60p.mean():.1%}, 주말/평일 {g.ratio_weekend.mean():.2f}")


if __name__ == "__main__":
    main()
