"""09b. 기상청 API허브 단기예보(동네예보) 격자 자료 → 강남·춘천 지점의 '전날 17시 발표' 예보.

- 엔드포인트: https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd?tmfc=&tmef=&vars= (5km 격자 149×253 전국 값, 한 변수·한 예보시각씩)
- 받는 것: 예보 대상일 D(2025-07-01~12-31)마다 전날 17시 발표(tmfc = D-1 17시) 기준으로
  PCP(1시간 강수량) D 0~23시, POP(강수확률) 6·9·12·15·18·21시, TMX(일최고기온) 15시, TMN(일최저기온) 6시.
- 원본: 응답 텍스트를 gzip으로 EXT_KMA/dfs_shrt_grd/<VAR>/<tmfc>_<tmef>.txt.gz 에 그대로 보관(제출용). 이미 있으면 다시 받지 않는다(중단 후 재실행 가능).
- 산출: kma_forecast_point(지점·발표·예보시각·변수·값, 3×3 평균), kma_forecast_daily(지역·날짜별 예보 피처와 플래그).
- 인증키: 프로젝트 루트 .env의 weather_api. 어디에도 출력하지 않는다.

실행: python 09b_kma_forecast.py            (수집 + 정리)
      python 09b_kma_forecast.py --build    (받아 둔 원본만으로 정리)
"""
import gzip
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import common as C

URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
NX, NY = 149, 253  # 동네예보 격자. 응답은 y행(남→북) 안에서 x(서→동) 순서로 값이 나열된다(검증: 2025-07-11 15시 TMP 서울 33℃).
POINTS = {"gangnam": (61, 126), "chuncheon": (73, 134)}  # 기상청 동네예보 격자 번호(강남구 61,126 / 춘천시 73,134)
PLAN = {"PCP": list(range(0, 24)), "POP": [6, 9, 12, 15, 18, 21], "TMX": [15], "TMN": [6]}
ISSUE_HOUR = 17
DATES = C.DATES
RAW = C.EXT_KMA / "dfs_shrt_grd"
NUM = re.compile(r"-?\d+\.\d+")


def fetch(tmfc, tmef, var, key):
    q = urllib.parse.urlencode({"tmfc": tmfc, "tmef": tmef, "vars": var, "authKey": key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(URL + "?" + q, timeout=120) as r:
                body = r.read()
            if len(body) < 1000:  # 오류 메시지만 온 경우
                raise RuntimeError(f"응답이 짧음({len(body)}B): {body[:80]!r}")
            return body
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"{var} {tmfc}->{tmef}: {type(e).__name__} {str(e)[:120]}") from None
            time.sleep(3 * (attempt + 1))


def collect():
    key = C.kma_api_key()
    n_new = n_skip = 0
    t0 = time.time()
    todo = []
    for d in DATES:
        tmfc = (d - timedelta(days=1)).strftime("%Y%m%d") + f"{ISSUE_HOUR:02d}"
        for var, hours in PLAN.items():
            for h in hours:
                tmef = (d + timedelta(hours=h)).strftime("%Y%m%d%H")
                todo.append((tmfc, tmef, var))
    print(f"  호출 계획 {len(todo):,}건")
    for i, (tmfc, tmef, var) in enumerate(todo, 1):
        out = RAW / var / f"{tmfc}_{tmef}.txt.gz"
        if out.exists():
            n_skip += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        body = fetch(tmfc, tmef, var, key)
        with gzip.open(out, "wb") as f:
            f.write(body)
        n_new += 1
        time.sleep(0.15)
        if n_new % 200 == 0:
            print(f"  {i:,}/{len(todo):,} ({time.time() - t0:.0f}초)", flush=True)
    print(f"  새로 받음 {n_new:,}, 이미 있음 {n_skip:,}")


def parse(path):
    txt = gzip.open(path, "rb").read().decode("utf-8", "replace")
    vals = np.array([float(v) for v in NUM.findall(txt)], dtype="float32")
    if len(vals) != NX * NY:
        raise ValueError(f"{path.name}: 값 {len(vals)}개")
    grid = vals.reshape(NY, NX)  # grid[y-1, x-1]
    grid[grid <= -99] = np.nan
    return grid


def build():
    rows = []
    for var in PLAN:
        for p in sorted((RAW / var).glob("*.txt.gz")):
            tmfc, tmef = p.stem.replace(".txt", "").split("_")
            try:
                g = parse(p)
            except Exception as e:  # 수집 중이거나 깨진 파일은 건너뛴다(다시 수집하면 채워진다)
                print(f"  건너뜀 {p.name}: {type(e).__name__}")
                continue
            for region, (x, y) in POINTS.items():
                v = g[y - 1, x - 1]
                nb = g[max(y - 2, 0):y + 1, max(x - 2, 0):x + 1]
                rows.append({"region": region, "tmfc": tmfc, "tmef": tmef, "var": var, "value": float(v) if np.isfinite(v) else np.nan,
                             "value_3x3": float(np.nanmean(nb)) if np.isfinite(nb).any() else np.nan})
    pt = pd.DataFrame(rows)
    pt["tmfc"] = pd.to_datetime(pt["tmfc"], format="%Y%m%d%H")
    pt["tmef"] = pd.to_datetime(pt["tmef"], format="%Y%m%d%H")
    pt["date"] = (pt["tmfc"] + pd.Timedelta(hours=24 - ISSUE_HOUR)).dt.normalize()  # 예보 대상일 D = 발표 다음 날
    pt["hour"] = pt["tmef"].dt.hour
    C.save(pt, "kma_forecast_point")

    # 일별 피처
    pcp = pt[pt["var"] == "PCP"].pivot_table(index=["region", "date"], columns="hour", values="value")
    pop = pt[pt["var"] == "POP"].pivot_table(index=["region", "date"], columns="hour", values="value")
    tmx = pt[pt["var"] == "TMX"].groupby(["region", "date"])["value"].first()
    tmn = pt[pt["var"] == "TMN"].groupby(["region", "date"])["value"].first()
    biz = [h for h in range(6, 24) if h in pcp.columns]
    dawn = [h for h in range(0, 6) if h in pcp.columns]
    day = pd.DataFrame({
        "fc_rn_biz": pcp[biz].sum(axis=1, min_count=1),
        "fc_rn_dawn": pcp[dawn].sum(axis=1, min_count=1),
        "fc_rn_day": pcp.sum(axis=1, min_count=1),
        "fc_rn_biz_max1h": pcp[biz].max(axis=1),
        "fc_rn_hours_biz": (pcp[biz] >= 1).sum(axis=1),
        "fc_rn_12_17": pcp[[h for h in range(12, 18) if h in pcp.columns]].sum(axis=1, min_count=1),
        "fc_rn_18_23": pcp[[h for h in range(18, 24) if h in pcp.columns]].sum(axis=1, min_count=1),
        "fc_rn_06_11": pcp[[h for h in range(6, 12) if h in pcp.columns]].sum(axis=1, min_count=1),
        "fc_pop_max_biz": pop[[h for h in [6, 9, 12, 15, 18, 21] if h in pop.columns]].max(axis=1),
        "fc_pop_mean_biz": pop[[h for h in [6, 9, 12, 15, 18, 21] if h in pop.columns]].mean(axis=1),
        "fc_tmax": tmx, "fc_tmin": tmn,
    }).reset_index()
    day["fc_rain_biz10"] = (day.fc_rn_biz >= 10).astype(int)
    day["fc_rain_biz20"] = (day.fc_rn_biz >= 20).astype(int)
    day["fc_heat_day"] = (day.fc_tmax >= 33).astype(int)
    day["fc_pop60"] = (day.fc_pop_max_biz >= 60).astype(int)
    C.save(day, "kma_forecast_daily")
    print(f"  지역·일 {len(day)}행, 기간 {day.date.min().date()}~{day.date.max().date()}, 예보 영업시간 비 10mm+ 일수: "
          + ", ".join(f"{r} {int(v)}" for r, v in day.groupby('region').fc_rain_biz10.sum().items()))
    return day


if __name__ == "__main__":
    if "--build" not in sys.argv:
        print("[1/2] 격자 예보 수집")
        collect()
    print("[2/2] 지점 추출·일별 피처")
    build()
    print("완료")
