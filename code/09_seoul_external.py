"""09. 서울 열린데이터광장·공공데이터포털 외부 자료 수집·정리.

- 서울시 상권분석서비스 추정매출-자치구: Open API(VwsmSignguSelngW)로 전체(25개 구×업종×분기)를 받는다. 받은 JSON 페이지와
  합친 CSV를 원본으로 EXT_SEOUL_SALES/api_raw/ 에 보관하고, parquet(seoul_sales_sgg)로 저장한다.
- 추정매출-행정동: 열린데이터광장에서 받은 연도별 zip(2021~2025)을 읽어 강남구(행정동 코드 11680*)만 parquet(seoul_sales_emd)로 저장한다.
- [내국인] 행정동별 서울 생활인구(250m): 월별 zip에서 강남구만 골라 시간별(seoul_pop_hourly)과 일별(seoul_pop_daily: 일평균,
  시간대 4구간 평균, 60대+)로 저장한다.
- 질병관리청 온열질환 응급실감시체계: 사례 CSV를 parquet(heat_illness)로, 시도×일 건수를 heat_illness_daily로 저장한다.

원본 위치는 common.EXT_BIG(기본 D:\\Bigcon_data, 환경변수 BIGCON_EXT). 인증키는 프로젝트 루트 .env의 openapi_seoul에서 읽고
어디에도 출력하지 않는다. 실행: python 09_seoul_external.py [--no-api]  (--no-api: 이미 받은 API 원본 CSV를 다시 쓴다)
"""
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

import common as C

API = "http://openapi.seoul.go.kr:8088/{key}/json/{svc}/{s}/{e}/"
SVC_SGG = "VwsmSignguSelngW"  # 추정매출-자치구(선택 인자 없음)
SVC_EMD = "VwsmAdstrdSelngW"  # 추정매출-행정동(선택 인자: 기준_년분기_코드)

# 추정매출 공통 53개 열. 자치구 API는 2·3번째가 자치구 코드/명, 행정동 zip은 행정동 코드/명(열 순서는 같다).
SALES_COLS = (
    ["yyq", "area_cd", "area_nm", "ind_cd", "ind_nm", "amt", "cnt", "amt_wkday", "amt_wkend"]
    + [f"amt_{d}" for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]]
    + [f"amt_t{t}" for t in ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]]
    + ["amt_m", "amt_f"] + [f"amt_a{a}" for a in ["10", "20", "30", "40", "50", "60p"]]
    + ["cnt_wkday", "cnt_wkend"] + [f"cnt_{d}" for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]]
    + [f"cnt_t{t}" for t in ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]]
    + ["cnt_m", "cnt_f"] + [f"cnt_a{a}" for a in ["10", "20", "30", "40", "50", "60p"]]
)
assert len(SALES_COLS) == 53
API_COLS = (
    ["STDR_YYQU_CD", "SIGNGU_CD", "SIGNGU_CD_NM", "SVC_INDUTY_CD", "SVC_INDUTY_CD_NM", "THSMON_SELNG_AMT", "THSMON_SELNG_CO",
     "MDWK_SELNG_AMT", "WKEND_SELNG_AMT"]
    + [f"{d}_SELNG_AMT" for d in ["MON", "TUES", "WED", "THUR", "FRI", "SAT", "SUN"]]
    + [f"TMZON_{t}_SELNG_AMT" for t in ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]]
    + ["ML_SELNG_AMT", "FML_SELNG_AMT"] + [f"AGRDE_{a}_SELNG_AMT" for a in ["10", "20", "30", "40", "50", "60_ABOVE"]]
    + ["MDWK_SELNG_CO", "WKEND_SELNG_CO"] + [f"{d}_SELNG_CO" for d in ["MON", "TUES", "WED", "THUR", "FRI", "SAT", "SUN"]]
    + [f"TMZON_{t}_SELNG_CO" for t in ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]]
    + ["ML_SELNG_CO", "FML_SELNG_CO"] + [f"AGRDE_{a}_SELNG_CO" for a in ["10", "20", "30", "40", "50", "60_ABOVE"]]
)
assert len(API_COLS) == 53

LOG = []  # 수집 기록(문서용)


def log(msg):
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------- Open API
def api_fetch_all(svc, raw_dir, extra=""):
    """서비스 전체를 1,000건씩 받아 JSON 페이지를 raw_dir에 저장하고 DataFrame으로 돌려준다. URL(키 포함)은 출력하지 않는다."""
    key = C.seoul_api_key()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_dir.glob("page_*.json"):
        old.unlink()
    rows, start, page, total = [], 1, 1, None
    while True:
        url = API.format(key=key, svc=svc, s=start, e=start + 999) + extra
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    body = r.read()
                break
            except Exception as e:  # 일시 오류는 잠시 쉬고 다시
                if attempt == 2:
                    raise RuntimeError(f"API 호출 실패({svc}, {start}~): {type(e).__name__}") from None
                time.sleep(3)
        js = json.loads(body)
        if svc not in js:  # 인증키·서비스 오류는 RESULT만 온다
            raise RuntimeError(f"API 오류: {js.get('RESULT')}")
        blk = js[svc]
        code = blk["RESULT"]["CODE"]
        if code == "INFO-200":  # 해당 데이터 없음
            break
        if code != "INFO-000":
            raise RuntimeError(f"API 오류: {blk['RESULT']}")
        (raw_dir / f"page_{page:04d}.json").write_bytes(body)
        total = int(blk["list_total_count"])
        rows.extend(blk["row"])
        if start + 999 >= total:
            break
        start += 1000
        page += 1
        time.sleep(0.2)
    log(f"  {svc}: 전체 {total if total is not None else 0:,}건 중 {len(rows):,}건, {page}페이지 → {raw_dir}")
    return pd.DataFrame(rows)


def sales_numeric(df):
    for c in df.columns[5:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("int64")
    return df


def fetch_sgg_sales(use_api=True):
    """추정매출-자치구: API로 전체를 받아 원본 CSV와 parquet를 만든다."""
    raw_dir = C.EXT_SEOUL_SALES / "api_raw" / SVC_SGG
    csv_path = C.EXT_SEOUL_SALES / f"서울시 상권분석서비스(추정매출-자치구)_API_전체.csv"
    if use_api:
        df = api_fetch_all(SVC_SGG, raw_dir)
        df = df[API_COLS]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        meta = {"service": SVC_SGG, "url_pattern": API.replace("{key}", "<인증키>"), "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "rows": int(len(df)), "pages": len(list(raw_dir.glob("page_*.json"))), "csv": csv_path.name}
        (raw_dir / "fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  원본 CSV 저장: {csv_path.name} ({len(df):,}행)")
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        log(f"  API 생략, 원본 CSV 사용: {csv_path.name} ({len(df):,}행)")
    df.columns = SALES_COLS
    df = sales_numeric(df)
    df["yyq"] = df["yyq"].astype(str)
    df["year"] = df["yyq"].str[:4].astype(int)
    df["q"] = df["yyq"].str[4].astype(int)
    df = df.sort_values(["yyq", "area_cd", "ind_cd"]).reset_index(drop=True)
    log(f"  자치구 {df.area_cd.nunique()}개, 업종 {df.ind_cd.nunique()}개, 분기 {df.yyq.min()}~{df.yyq.max()} ({df.yyq.nunique()}개)")
    C.save(df, "seoul_sales_sgg")
    return df


# ---------------------------------------------------------------- 추정매출-행정동 zip
def load_emd_sales():
    files = sorted(C.EXT_SEOUL_SALES.glob("*추정매출-행정동)_20*.zip"))
    if not files:
        raise FileNotFoundError(f"추정매출-행정동 zip이 없다: {C.EXT_SEOUL_SALES}")
    parts = []
    for f in files:
        z = zipfile.ZipFile(f)
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        d = pd.read_csv(z.open(name), encoding="cp949", dtype=str)
        assert len(d.columns) == 53, f"{f.name}: 열 {len(d.columns)}개"
        d.columns = SALES_COLS
        d = d[d.area_cd.str.startswith(C.GANGNAM_ADSTRD)].copy()
        parts.append(d)
        log(f"  {f.name}: 강남구 {len(d):,}행 (분기 {sorted(d.yyq.unique())})")
    df = pd.concat(parts, ignore_index=True)
    df = sales_numeric(df)
    df["year"] = df["yyq"].str[:4].astype(int)
    df["q"] = df["yyq"].str[4].astype(int)
    df = df.sort_values(["yyq", "area_cd", "ind_cd"]).reset_index(drop=True)
    log(f"  강남구 행정동 {df.area_cd.nunique()}개, 업종 {df.ind_cd.nunique()}개, 분기 {df.yyq.min()}~{df.yyq.max()}")
    C.save(df, "seoul_sales_emd")
    return df


# ---------------------------------------------------------------- 생활인구
AGE_BANDS = ["00_09", "10_14", "15_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64", "65_69", "70p"]
POP_COLS = ["date", "hour", "adstrd_cd", "pop"] + [f"m_{a}" for a in AGE_BANDS] + [f"f_{a}" for a in AGE_BANDS]
HOUR_BAND = {h: ("00_05" if h < 6 else "06_11" if h < 12 else "12_17" if h < 18 else "18_23") for h in range(24)}


def load_living_pop():
    files = sorted(C.EXT_SEOUL_POP.glob("250_LOCAL_RESD_ADMDONG_*.zip"))
    if not files:
        raise FileNotFoundError(f"생활인구 zip이 없다: {C.EXT_SEOUL_POP}")
    parts = []
    for f in files:
        z = zipfile.ZipFile(f)
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        d = pd.read_csv(z.open(name), encoding="cp949", dtype={"행정동코드": str})
        assert len(d.columns) == 32, f"{f.name}: 열 {len(d.columns)}개"
        d = d[d["행정동코드"].str.startswith(C.GANGNAM_ADSTRD)].copy()
        d.columns = POP_COLS
        parts.append(d)
        print(f"  {f.name}: 강남구 {len(d):,}행, 동 {d.adstrd_cd.nunique()}개")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["hour"] = df["hour"].astype(int)
    for c in POP_COLS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["pop60"] = df[[f"{s}_{a}" for s in ("m", "f") for a in ["60_64", "65_69", "70p"]]].sum(axis=1)
    df = df.sort_values(["date", "hour", "adstrd_cd"]).reset_index(drop=True)
    log(f"  생활인구 강남구: {df.date.min().date()}~{df.date.max().date()}, 월 {df.date.dt.to_period('M').nunique()}개, 동 {df.adstrd_cd.nunique()}개, {len(df):,}행")
    C.save(df, "seoul_pop_hourly")

    # 일별: 시간 평균(=SK의 '일평균' 정의와 맞춤), 시간대 4구간 평균, 60대+
    df["tb"] = df["hour"].map(HOUR_BAND)
    day = df.groupby(["date", "adstrd_cd"]).agg(pop=("pop", "mean"), pop60=("pop60", "mean"), n_hours=("hour", "size")).reset_index()
    tb = df.groupby(["date", "adstrd_cd", "tb"])["pop"].mean().unstack("tb")
    tb.columns = [f"pop_{c}" for c in tb.columns]
    day = day.merge(tb.reset_index(), on=["date", "adstrd_cd"], how="left")
    day["share_60p"] = day["pop60"] / day["pop"]
    bad = day[day.n_hours != 24]
    if len(bad):
        print(f"  주의: 24시간이 아닌 날·동 {len(bad)}건 (예: {bad.head(3)[['date', 'adstrd_cd', 'n_hours']].values.tolist()})")
    C.save(day, "seoul_pop_daily")
    return df, day


# ---------------------------------------------------------------- 온열질환
SIDO_MAP = {"서울특별시": "seoul", "강원특별자치도": "gangwon", "강원도": "gangwon"}


def load_heat_illness():
    d = pd.read_csv(C.EXT_HEAT_ILL, encoding="utf-8-sig", dtype=str)
    d.columns = ["date", "sex", "age", "sido", "sgg", "indoor", "place"]
    n_blank = int(d.isna().all(axis=1).sum())  # 파일 끝에 ',,,,,,' 빈 줄이 수천 개 있다
    d = d.dropna(how="all")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    n_bad = int(d["date"].isna().sum())
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    log(f"  온열질환 사례 {len(d):,}건 ({d.date.min().date()}~{d.date.max().date()}; 빈 줄 {n_blank}개, 날짜 없는 행 {n_bad}건 제외). "
        "2011~2018년은 신고 의료기관 소재지, 2019년부터는 발생 시도 기준")
    C.save(d, "heat_illness")
    daily = d.groupby(["sido", "date"]).size().rename("cases").reset_index()
    daily["region_sido"] = daily["sido"].map(SIDO_MAP)
    C.save(daily, "heat_illness_daily")
    return d, daily


# ---------------------------------------------------------------- 점검표
def _dev(s, dates):
    """로그값에서 같은 요일 ±1·2주 평균을 뺀 편차(카드 분석과 같은 방식)."""
    s = pd.Series(np.log(s.values), index=dates)
    base = pd.concat([s.shift(7, freq="D"), s.shift(-7, freq="D"), s.shift(14, freq="D"), s.shift(-14, freq="D")], axis=1).mean(axis=1)
    return (s - base.reindex(s.index)).values


def preview():
    """받은 자료가 쓸 만한지 보는 표 세 개를 outputs/tables/ext_*.csv로 저장한다."""
    C.TAB.mkdir(parents=True, exist_ok=True)
    # 1) 강남 생활인구 × 비·폭염 (2023·2024·2025년 7~12월, 서울 ASOS 108 일강수 10mm+, 일최고 33℃+)
    day = C.load("seoul_pop_daily")
    g = day.groupby("date")[["pop", "pop_00_05", "pop_06_11", "pop_12_17", "pop_18_23", "pop60"]].sum().reset_index()
    wl = C.load("weather_long_daily")
    wl = wl[wl.stn == 108][["date", "rn_sum", "ta_max"]].copy()
    wl["date"] = pd.to_datetime(wl["date"])
    g = g.merge(wl, on="date", how="left")
    g["rain10"] = (g.rn_sum >= 10).astype(float)
    g["heat33"] = (g.ta_max >= 33).astype(float)
    rows = []
    for c in ["pop", "pop_00_05", "pop_06_11", "pop_12_17", "pop_18_23", "pop60"]:
        parts = [pd.DataFrame({"dev": _dev(gy[c], gy.date), "rain10": gy.rain10.values, "heat33": gy.heat33.values, "year": y})
                 for y, gy in g.groupby(g.date.dt.year)]
        d = pd.concat(parts).dropna()
        for scope, dd in [("2023~2025", d), ("2025", d[d.year == 2025])]:
            for ev in ["rain10", "heat33"]:
                a, b = dd[dd[ev] == 1].dev, dd[dd[ev] == 0].dev
                rows.append({"기간": scope, "지표": c, "이벤트": ev, "이벤트 일수": len(a), "이벤트일 편차%": round((np.exp(a.mean()) - 1) * 100, 2),
                             "그 외 편차%": round((np.exp(b.mean()) - 1) * 100, 2),
                             "t": round((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)), 2)})
    pd.DataFrame(rows).to_csv(C.TAB / "ext_pop_event.csv", index=False, encoding="utf-8-sig")
    # 2) 강남구 추정매출: 분기 구성비와 하반기 전년 대비
    s = C.load("seoul_sales_sgg")
    gn = s[(s.area_cd == "11680") & (s.year <= 2025)]
    yr = gn.groupby(["year", "q"])["amt"].sum().unstack("q")
    share = (yr.div(yr.sum(axis=1), axis=0) * 100).round(1)
    share.columns = [f"Q{c} 비중%" for c in share.columns]
    h2 = gn[gn.q.isin([3, 4])].groupby("year")["amt"].sum().div(1e8).round(0)
    share["H2 매출(억)"] = h2
    share["H2 전년 대비%"] = (h2.pct_change() * 100).round(1)
    share.reset_index().to_csv(C.TAB / "ext_sales_quarter.csv", index=False, encoding="utf-8-sig")
    # 3) 업종별 2025 하반기 대 2024 하반기
    ind = gn[gn.q.isin([3, 4])].groupby(["ind_nm", "year"])["amt"].sum().unstack("year").div(1e8).round(1)
    ind["2025/2024 %"] = ((ind[2025] / ind[2024] - 1) * 100).round(1)
    ind.sort_values(2025, ascending=False).reset_index().to_csv(C.TAB / "ext_sales_h2_industry.csv", index=False, encoding="utf-8-sig")
    r = pd.DataFrame(rows)
    a = r[(r.기간 == "2023~2025") & (r.지표 == "pop") & (r.이벤트 == "rain10")].iloc[0]
    log(f"  점검: 강남 생활인구는 비 온 날(3년, {a['이벤트 일수']}일) {a['이벤트일 편차%']:+.2f}% (t={a['t']}) → 비는 '사람 수'가 아니라 '소비'를 줄인다. "
        f"2025 하반기 강남구 추정매출은 전년 대비 {share.loc[2025, 'H2 전년 대비%']:+.1f}%")
    print("  저장: ext_pop_event.csv, ext_sales_quarter.csv, ext_sales_h2_industry.csv")


# ---------------------------------------------------------------- 기록
def write_record(use_api):
    path = C.ROOT / "data" / "external" / "수집기록_외부자료.md"
    lines = [
        "# 외부 자료 수집 기록 (자동 생성: 09_seoul_external.py)", "",
        f"마지막 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')}  (API 호출: {'예' if use_api else '아니오, 보관 CSV 사용'})", "",
        f"원본 보관 위치: `{C.EXT_BIG}` (환경변수 BIGCON_EXT로 변경 가능). 제출 시 이 폴더의 원본 파일을 함께 낸다.", "",
        "| 자료 | 출처 | 수집 방법 | 원본 파일 |", "|---|---|---|---|",
        "| 서울시 상권분석서비스(추정매출-자치구) | 서울 열린데이터광장, 서울신용보증재단, 공공누리 1유형 | Open API `VwsmSignguSelngW`를 1,000건씩 전체 호출(인증키는 .env, 제출하지 않음) | `서울열린데이터광장/상권분석서비스/api_raw/VwsmSignguSelngW/page_*.json`, `…_API_전체.csv` |",
        "| 서울시 상권분석서비스(추정매출-행정동) | 같은 곳 | 데이터셋 페이지의 연도별 zip 다운로드(2021~2025) | `서울열린데이터광장/상권분석서비스/*추정매출-행정동)_YYYY년.zip` |",
        "| [내국인] 행정동별 서울 생활인구(250m) | 서울 열린데이터광장(OA-23016) | 월별 zip 다운로드(2023·2024·2025년 7~12월) | `서울열린데이터광장/행정동별서울생활인구(250m)/250_LOCAL_RESD_ADMDONG_YYYYMM.zip` |",
        "| 질병관리청 온열질환 응급실감시체계 | 공공데이터포털(무료) | CSV 다운로드 | `공공데이터포털/질병관리청_온열질환 감시 데이터_20250925.CSV` |",
        "", "실행 로그:", "",
    ] + [f"- {m.strip()}" for m in LOG]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  기록: {path}")


if __name__ == "__main__":
    use_api = "--no-api" not in sys.argv
    print("[1/4] 추정매출-자치구 (Open API)")
    fetch_sgg_sales(use_api)
    print("[2/4] 추정매출-행정동 (zip, 강남구)")
    load_emd_sales()
    print("[3/4] 행정동별 서울 생활인구 (zip, 강남구)")
    load_living_pop()
    print("[4/4] 온열질환 응급실감시체계")
    load_heat_illness()
    print("[점검] 받은 자료 점검표")
    preview()
    write_record(use_api)
    print("완료")
