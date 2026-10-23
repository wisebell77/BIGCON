"""기상청 관측자료 → 시간별·시간대별·일별 날씨 피처와 기상 이벤트 플래그.

입력: data/external/weather/ 에 기상자료개방포털에서 받은 CSV(cp949)를 파일 이름 그대로 넣는다.
- ASOS 시간자료(필수): 서울(108)·춘천(101)
- AWS 시간자료(선택): 강남(400). 있으면 강남 날씨는 강남구 안의 AWS 값을 먼저 쓰고, 빈 시간과
  AWS에 없는 요소(일조·적설·운량)는 서울 ASOS로 채운다. 지점 구성은 common.REGION_STATIONS.
- ASOS 일자료(선택): 공식 일 최고·최저기온과 일강수량(*_asos). 폭염일 판정은 이 공식 일최고기온으로 한다.
  (시간자료 최고값은 분 단위 최고값보다 낮아 폭염일이 적게 잡힌다: 춘천 26일 vs 공식 32일)
기간은 2025-06-01 ~ 2026-01-31로 받는다(전날·다음 날 값과 이동평균용 여유).

시간 처리: 시간자료의 '01:00' 값은 00~01시 구간이다. 한 시간 당겨서 0~23시 구간으로 맞추고,
카드 시간대(00_05, 06_11, 12_17, 18_23)와 같은 기준으로 묶는다.

이벤트 정의(팀 결정 2026-10-09: 폭염·열대야·호우 중심, 한파는 참고)
- 폭염일 heat_day: 공식 일최고기온 33℃ 이상(기상청 폭염일 정의). heat35_day: 35℃ 이상.
  heat_feels33: 시간별 체감온도 최고 33℃ 이상(폭염특보 기준값, 지속 조건 제외).
- 열대야 tropical: 그날 18시 ~ 다음 날 09시 최저기온 25℃ 이상(저녁 영업시간이 열대야인 날).
  tropical_prev: 전날 밤이 열대야(아침 영업시간에 영향).
- 호우 rain_biz10/20/30: 영업시간(06~24시) 강수 합 10/20/30mm 이상. rain_1h20: 1시간 20mm 이상.
  rain_adv: 호우주의보 기준(3시간 60mm 또는 12시간 110mm)에 한 번이라도 닿은 날. rain_warn: 경보(90/180).
- 한파(참고) cold12: 일최저 -12℃ 이하. cold_max0: 공식 일최고 0℃ 이하.
- 생활기상지수: 체감온도(기상청 2022 개정식, 여름은 습구온도 기반), 불쾌지수 DI(기상청 식).

사용법: python code/04_weather.py [--src 입력폴더] [--out 출력폴더]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXT_WEATHER, OUT, REGION_ASOS, REGION_STATIONS, TIME_BANDS, save

# 표준 이름: (열 이름에 들어 있어야 할 글자, 들어 있으면 안 되는 글자)
HOURLY_COLS = {
    "ta": ("기온", ("QC", "지면", "이슬점", "지중")),
    "rn": ("강수량", ("QC",)),
    "ws": ("풍속", ("QC",)),
    "hm": ("습도", ("QC",)),
    "ss": ("일조", ("QC",)),
    "snow": ("적설", ("신적설", "QC")),
    "cloud": ("전운량", ()),
}
DAILY_COLS = {
    "ta_max_asos": ("최고기온", ("시각",)),
    "ta_min_asos": ("최저기온", ("시각",)),
    "rn_sum_asos": ("일강수량", ()),
    "rn_max_1h_asos": ("1시간 최다강수량", ("시각",)),
}


def pick(columns, include, exclude):
    """열 이름을 공백 없이 비교한다(자료 종류마다 '최다 강수량'처럼 띄어쓰기가 다르다)."""
    squash = lambda s: s.replace(" ", "")
    hits = [c for c in columns if squash(include) in squash(c) and not any(squash(x) in squash(c) for x in exclude)]
    return hits[0] if hits else None


def read_kma_csv(path):
    for enc in ("cp949", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def load_kma(src):
    hourly, daily = [], []
    for path in sorted(Path(src).glob("*.csv")):
        df = read_kma_csv(path)
        if "지점" not in df.columns or "일시" not in df.columns:
            print(f"  건너뜀(기상청 관측자료 형식 아님): {path.name}")
            continue
        ts = pd.to_datetime(df["일시"])
        is_hourly = (ts.dt.hour != 0).any()
        is_aws = "AWS" in path.name.upper() or "풍향(deg)" in df.columns
        spec = HOURLY_COLS if is_hourly else DAILY_COLS
        out = pd.DataFrame({"stn": df["지점"].astype(int), "ts": ts})
        for name, (inc, exc) in spec.items():
            col = pick(df.columns, inc, exc)
            out[name] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
        if is_hourly and not is_aws:
            # ASOS는 비·눈이 없으면 빈칸이다(QC플래그 9 포함, 공식 일강수량과 100% 일치 확인).
            # AWS는 비가 없으면 0.0을 적으므로 빈칸은 결측으로 두고 다른 지점 값으로 채운다.
            out[["rn", "snow"]] = out[["rn", "snow"]].fillna(0)
        if not is_hourly:
            out[["rn_sum_asos", "rn_max_1h_asos"]] = out[["rn_sum_asos", "rn_max_1h_asos"]].fillna(0)
        (hourly if is_hourly else daily).append(out)
        kind = f"{'AWS' if is_aws else 'ASOS'} {'시간' if is_hourly else '일'}자료"
        print(f"  {path.name}: {kind} {len(out):,}행, 지점 {sorted(int(s) for s in out.stn.unique())}, "
              f"{ts.min():%Y-%m-%d} ~ {ts.max():%Y-%m-%d}")
    if not hourly:
        return None, None
    h = pd.concat(hourly).drop_duplicates(["stn", "ts"]).sort_values(["stn", "ts"])
    d = pd.concat(daily).drop_duplicates(["stn", "ts"]) if daily else None
    return h, d


def region_hourly(h):
    """지역마다 우선 지점 값을 쓰고, 빈 시간·빈 요소는 다음 지점 값으로 채운다."""
    frames = []
    for region, stns in REGION_STATIONS.items():
        found = [s for s in stns if (h["stn"] == s).any()]
        if not found:
            print(f"  주의: {region} 지점 {stns} 자료가 없습니다.")
            continue
        if found[0] != stns[0]:
            print(f"  주의: {region} 우선 지점 {stns[0]} 자료가 없어 {found[0]}를 씁니다.")
        merged = None
        for s in found:
            part = h[h["stn"] == s].set_index("ts")[list(HOURLY_COLS)]
            merged = part if merged is None else merged.combine_first(part)
        primary = h[h["stn"] == stns[0]].set_index("ts")["rn"]
        merged["from_primary"] = primary.reindex(merged.index).notna().astype(int)
        frames.append(merged.reset_index().assign(region=region))
    return pd.concat(frames, ignore_index=True).sort_values(["region", "ts"]).reset_index(drop=True)


def wet_bulb_stull(ta, rh):
    """Stull(2011) 습구온도 추정식."""
    return (ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(ta + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def feels_like(ta, rh, ws, month):
    """기상청 체감온도: 5~9월은 습구온도 기반 여름 공식, 그 외는 풍속냉각(기온 10℃ 이하, 풍속 1.3m/s 이상)."""
    tw = wet_bulb_stull(ta, rh)
    summer = -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0
    v = ws * 3.6  # km/h
    chill = 13.12 + 0.6215 * ta - 11.37 * v ** 0.16 + 0.3965 * v ** 0.16 * ta
    winter = np.where((ta <= 10) & (ws >= 1.3), chill, ta)
    return np.where(month.between(5, 9), summer, winter)


def discomfort_index(ta, rh):
    """불쾌지수(기상청 생활기상지수 식). 75 이상 절반이 불쾌, 80 이상 대부분 불쾌."""
    return 0.81 * ta + 0.01 * rh * (0.99 * ta - 14.3) + 46.3


def hourly_features(h):
    h = region_hourly(h)
    h[["rn", "snow"]] = h[["rn", "snow"]].fillna(0)  # 모든 지점이 빈 시간(드묾)
    start = h["ts"] - pd.Timedelta(hours=1)  # '01:00' 값 = 00~01시 구간
    h["date"] = start.dt.normalize()
    h["hour"] = start.dt.hour
    h["tb"] = pd.cut(h["hour"], [-1, 5, 11, 17, 23], labels=TIME_BANDS).astype(str)
    h["feels"] = feels_like(h["ta"], h["hm"], h["ws"].fillna(0), h["date"].dt.month)
    h["di"] = discomfort_index(h["ta"], h["hm"])
    h["rain_hr"] = (h["rn"] > 0).astype(int)
    h["heat_hr"] = (h["feels"] >= 33).astype(int)
    g = h.groupby("region")["rn"]
    h["rn_3h"] = g.transform(lambda s: s.rolling(3, min_periods=1).sum())
    h["rn_12h"] = g.transform(lambda s: s.rolling(12, min_periods=1).sum())
    h["rain_adv_hr"] = ((h["rn_3h"] >= 60) | (h["rn_12h"] >= 110)).astype(int)   # 호우주의보 기준
    h["rain_warn_hr"] = ((h["rn_3h"] >= 90) | (h["rn_12h"] >= 180)).astype(int)  # 호우경보 기준
    return h


def daily_features(h, d_obs):
    g = h.groupby(["region", "date"])
    day = g.agg(ta_mean=("ta", "mean"), ta_max=("ta", "max"), ta_min=("ta", "min"),
                feels_max=("feels", "max"), feels_mean=("feels", "mean"), di_max=("di", "max"),
                hm_mean=("hm", "mean"), rn_sum=("rn", "sum"), rn_hours=("rain_hr", "sum"), rn_max_1h=("rn", "max"),
                heat_hours=("heat_hr", "sum"), rain_adv=("rain_adv_hr", "max"), rain_warn=("rain_warn_hr", "max"),
                ws_mean=("ws", "mean"), ws_max=("ws", "max"), ss_sum=("ss", "sum"),
                snow_max=("snow", "max"), cloud_mean=("cloud", "mean"), n_obs=("ts", "size"),
                primary_hours=("from_primary", "sum")).reset_index()
    day = day[day["n_obs"] >= 20]  # 파일 맨 앞·끝처럼 관측이 몇 시간뿐인 날은 뺀다

    biz = h[h["hour"] >= 6].groupby(["region", "date"]).agg(
        rn_biz=("rn", "sum"), rn_biz_hours=("rain_hr", "sum"), rn_biz_max_1h=("rn", "max")).reset_index()
    day = day.merge(biz, on=["region", "date"], how="left")

    if d_obs is not None:  # 공식 일자료
        asos_region = {stn: region for region, stn in REGION_ASOS.items()}
        ref = d_obs.assign(region=d_obs["stn"].map(asos_region), date=d_obs["ts"].dt.normalize())
        ref = ref.dropna(subset=["region"]).drop(columns=["stn", "ts"])
        day = day.merge(ref, on=["region", "date"], how="left")
    else:
        for c in DAILY_COLS:
            day[c] = np.nan
    day["ta_max_off"] = day["ta_max_asos"].fillna(day["ta_max"])  # 공식 일최고기온(없으면 시간자료 최고)

    # 열대야: 그날 18시 ~ 다음 날 09시 최저기온(기상청 정의: 18:01~익일 09:00). night_min은 밤이 시작하는 날짜 기준
    night = h[(h["hour"] >= 18) | (h["hour"] <= 8)].copy()
    night["night_of"] = night["date"] - pd.to_timedelta((night["hour"] <= 8).astype(int), unit="D")
    nmin = night.groupby(["region", "night_of"])["ta"].min().rename("night_min").reset_index()
    day = day.merge(nmin.rename(columns={"night_of": "date"}), on=["region", "date"], how="left")
    day = day.sort_values(["region", "date"]).reset_index(drop=True)
    day["night_min_prev"] = day.groupby("region")["night_min"].shift(1)

    # 이벤트 플래그
    day["heat_day"] = (day["ta_max_off"] >= 33).astype(int)
    day["heat35_day"] = (day["ta_max_off"] >= 35).astype(int)
    day["heat_feels33"] = (day["feels_max"] >= 33).astype(int)
    day["tropical"] = (day["night_min"] >= 25).astype(int)
    day["tropical_prev"] = (day["night_min_prev"] >= 25).astype(int)
    for mm in [10, 20, 30]:
        day[f"rain_biz{mm}"] = (day["rn_biz"] >= mm).astype(int)
    for mm in [1, 10, 30, 80]:
        day[f"rain{mm}"] = (day["rn_sum"] >= mm).astype(int)
    day["rain_1h20"] = (day["rn_max_1h"] >= 20).astype(int)
    day["rain_dawn_only"] = ((day["rn_sum"] >= 10) & (day["rn_biz"] < 1)).astype(int)  # 새벽에만 온 비
    day["cold12"] = (day["ta_min"] <= -12).astype(int)
    day["cold_max0"] = (day["ta_max_off"] <= 0).astype(int)
    day["snow"] = (day["snow_max"] > 0).astype(int)
    m = day["date"].dt.month
    dry = day["rn_sum"] == 0
    day["bok_heat"] = (dry & (day["ta_max_off"] >= 35) & m.between(4, 9)).astype(int)   # 한국은행 이슈노트 정의
    day["bok_cold"] = (dry & (day["ta_max_off"] <= 0) & ~m.between(4, 9)).astype(int)
    day["bok_rain"] = ((day["rn_sum"] >= 20) & (day["bok_heat"] == 0) & (day["bok_cold"] == 0)).astype(int)
    day["event_any"] = ((day["heat_day"] == 1) | (day["tropical"] == 1) | (day["rain_biz10"] == 1)).astype(int)

    # 시차·누적·이상치
    g = day.groupby("region")
    for col in ["rn_biz", "rn_sum", "ta_max_off", "feels_max", "heat_day", "tropical", "rain_biz10", "rain_biz20"]:
        day[f"{col}_lag1"] = g[col].shift(1)
        day[f"{col}_lead1"] = g[col].shift(-1)
    day["rn_3d"] = g["rn_sum"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    day["heat_streak"] = g["heat_day"].transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
    day["tropical_streak"] = g["tropical"].transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
    day["ta_max_dev7"] = day["ta_max_off"] - g["ta_max_off"].transform(lambda s: s.shift(1).rolling(7, min_periods=4).mean())
    day["ta_max_anom31"] = day["ta_max_off"] - g["ta_max_off"].transform(lambda s: s.rolling(31, center=True, min_periods=15).mean())
    day["after_rain"] = ((day["rn_biz_lag1"] >= 20) & (day["rn_biz"] < 1)).astype(int)  # 큰비 다음 날(반등 확인용)
    return day


def timeband_features(h):
    tb = h.groupby(["region", "date", "tb"]).agg(
        rn_sum=("rn", "sum"), rn_hours=("rain_hr", "sum"), rn_max_1h=("rn", "max"),
        ta_mean=("ta", "mean"), ta_max=("ta", "max"), feels_max=("feels", "max"), feels_mean=("feels", "mean"),
        di_max=("di", "max"), hm_mean=("hm", "mean"), heat_hours=("heat_hr", "sum"),
        rain_adv=("rain_adv_hr", "max")).reset_index()
    for mm in [1, 5, 10]:
        tb[f"rain{mm}"] = (tb["rn_sum"] >= mm).astype(int)
    tb["heat"] = (tb["feels_max"] >= 33).astype(int)
    return tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(EXT_WEATHER))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"[1] 관측자료 읽기: {args.src}")
    h, d_obs = load_kma(args.src)
    if h is None:
        print("  시간자료가 없습니다. README의 '날씨 데이터 받기'를 따라 CSV를 넣은 뒤 다시 실행하세요.")
        return

    print("[2] 시간 → 시간대·일 피처")
    h = hourly_features(h)
    day = daily_features(h, d_obs)
    tb = timeband_features(h).merge(day[["region", "date"]], on=["region", "date"])  # 일 단위와 같은 날만
    a = day[day["date"].between("2025-07-01", "2025-12-31")]
    for region, g in a.groupby("region"):
        stns = REGION_STATIONS[region]
        print(f"  {region}: 분석 기간 {len(g)}일, 우선 지점({stns[0]}) 값이 있는 시간 비율 "
              f"{g.primary_hours.sum() / g.n_obs.sum():.1%}, 폭염일 {g.heat_day.sum()}, 열대야 {g.tropical.sum()}, "
              f"영업시간 비 10mm+ {g.rain_biz10.sum()}일, 호우주의보 수준 {g.rain_adv.sum()}일, 한파(-12) {g.cold12.sum()}일")
    out = Path(args.out)
    keep = ["region", "ts", "date", "hour", "tb", "ta", "rn", "ws", "hm", "feels", "di", "rain_hr", "heat_hr",
            "rain_adv_hr", "from_primary"]
    save(h[keep], "weather_hourly", out)
    save(day, "weather_daily", out)
    save(tb, "weather_tb", out)


if __name__ == "__main__":
    main()
