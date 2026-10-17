"""기상특보 발령 이력(2005~) → 강남(서울 동남권)·춘천의 일별 특보 상태.

입력: data/external/warnings/FCT_WRN_*.csv (기상자료개방포털 → 기상특보 → 특보 발령 현황. 열: 발표시간, 지역, 발효시각, 해당지역, 내용)
- 2014년 이전 파일은 '내용' 칸에 줄바꿈이 섞여 있어 CSV가 깨진다. 발표시간이 날짜로 읽히지 않는 줄은 이어진 본문이므로 버린다.
- 한 통보문은 '전국' 행과 지역 행에 중복으로 들어 있다. 항목 단위로 풀어 (발효시각, 종류, 등급, 동작, 지역)으로 중복을 지운다.
- 항목은 "(1) 호우주의보 발표 : 2025년 07월 08일 18시 30분" 꼴이고, 옛 파일은 항목 사이에 구분자가 없어 "(n)" 표시로 자른다.
- 해당지역에서 강남은 '서울', '서울특별시', '서울(동남권 …)'(강남구는 동남권)으로, 춘천은 '춘천', '강원도' 전체, '영서'로 잡는다.
  '… 제외)' 표기는 뒤집어 처리한다.
- 동작: 발표·변경·대치·연장·발효 = 그 등급이 발효 중, 해제·대치해제·제외 = 해제. 같은 시각이면 해제를 먼저 처리한다.

출력
- warnings_events: 지역×종류별 발효 구간(시작, 끝, 등급)
- warnings_daily: 지역×날짜(2005-07-01~)×종류별 최고 등급(0 없음, 1 주의보, 2 경보)과 발효 시간(전체, 06~24시)
- warnings_annual: 지역×연도별 종류별 특보 일수

주의: 폭염특보는 2008년 시작, 2020년부터 체감온도 기준으로 바뀌었다. 서울 권역 세분(동남권 등)은 최근 자료에만 있다.
연도 간 비교에는 이런 제도 변화가 섞인다.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXT_WEATHER, save

SRC = EXT_WEATHER.parent / "warnings"
KINDS = {"호우": "rain", "폭염": "heat", "한파": "cold", "대설": "snow", "강풍": "wind", "태풍": "typhoon", "열대야": "tropical", "폭염중대": "heat_major"}
LEVEL = {"주의보": 1, "경보": 2}
ISSUE = {"발표", "변경", "대치", "연장", "발효", "대치발표", "빌표", "박표", "발표제"}
CLEAR = {"해제", "대치해제", "제외", "해재"}
ITEM = re.compile(r"\((\d+)\)\s*([가-힣]+?)(주의보|경보)\s*([가-힣]+)\s*:\s*(.*?)(?=\(\d+\)\s*[가-힣]+?(?:주의보|경보)|$)", re.S)
TIME = re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(\d{1,2})시\s*(\d{1,2})분")


def read_warnings():
    frames = []
    for f in sorted(SRC.glob("FCT_WRN_*.csv")):
        d = pd.read_csv(f, encoding="cp949", engine="python", usecols=["발표시간", "지역", "발효시각", "해당지역"])
        d["발표시간"] = pd.to_datetime(d["발표시간"], errors="coerce")
        n0 = len(d)
        d = d.dropna(subset=["발표시간", "발효시각", "해당지역"])
        print(f"  {f.name}: {n0:,}줄 중 통보문 {len(d):,}건, {d.발표시간.min():%Y-%m-%d}~{d.발표시간.max():%Y-%m-%d}")
        frames.append(d)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def parse_items(text):
    """'(1) 호우주의보 발표 : 2025년 07월 08일 18시 30분 / (2) …' → [(n, 종류, 등급, 동작, 본문)]"""
    return [(int(n), kind, LEVEL[lvl], act, body.strip(" /\t")) for n, kind, lvl, act, body in ITEM.findall(str(text))]


def area_hits(area):
    """해당지역 문자열에서 강남(서울 동남권)·춘천 포함 여부."""
    area = str(area)
    hits = {"gangnam": False, "chuncheon": False}
    for m in re.finditer(r"서울(?:특별시)?\s*(\([^)]*\))?", area):
        grp = m.group(1)
        if not grp:
            hits["gangnam"] = True
        else:
            inside = "동남권" in grp or "서울 전체" in grp
            hits["gangnam"] = (not inside) if "제외" in grp else inside
    for m in re.finditer(r"강원(?:특별자치도|도)?\s*(\([^)]*\))?", area):
        grp = m.group(1)
        if not grp:
            hits["chuncheon"] = True
        else:
            inside = "춘천" in grp or "영서" in grp or "강원 전체" in grp
            if "제외" in grp:
                hits["chuncheon"] = not ("춘천" in grp)
            else:
                hits["chuncheon"] = hits["chuncheon"] or inside
    if "춘천" in area and "강원" not in area:
        hits["chuncheon"] = True
    return hits


def build_events(w):
    rows = []
    for _, r in w.iterrows():
        times = {n: body for n, _, _, _, body in parse_items(r["발효시각"])}
        areas = {n: (kind, lvl, act, body) for n, kind, lvl, act, body in parse_items(r["해당지역"])}
        for n, (kind, lvl, act, body) in areas.items():
            if kind not in KINDS or n not in times:
                continue
            m = TIME.search(times[n])
            if not m:
                continue
            t = pd.Timestamp(*(int(x) for x in m.groups()))
            hit = area_hits(body)
            for region, ok in hit.items():
                if ok:
                    rows.append({"region": region, "kind": KINDS[kind], "level": lvl, "action": act, "time": t, "announced": r["발표시간"]})
    ev = pd.DataFrame(rows).drop_duplicates(["region", "kind", "level", "action", "time"])
    ev["is_clear"] = ev.action.isin(CLEAR)
    ev = ev[ev.action.isin(ISSUE | CLEAR)]
    return ev.sort_values(["region", "kind", "time", "is_clear"], ascending=[True, True, True, False]).reset_index(drop=True)


MAX_DAYS = {"rain": 3, "wind": 3, "snow": 3, "typhoon": 3, "cold": 10, "heat": 45, "tropical": 10, "heat_major": 10}  # 해제 누락 보정 상한
REPLACED_BY_TYPHOON = {"rain", "wind"}  # 태풍특보가 나오면 호우·강풍특보는 태풍특보로 대치된다


def intervals(ev, end_default):
    """동작 순서대로 상태를 따라가며 (시작, 끝, 등급) 구간을 만든다.
    - 해제는 등급과 상관없이 그 종류의 특보를 끝낸다(옛 통보문은 등급 변경이 누락되기도 한다).
    - 태풍특보 발표 시각에 열려 있던 호우·강풍특보는 닫는다(대치).
    - 종류별 상한(MAX_DAYS)보다 긴 구간은 해제 누락으로 보고 상한에서 자른다."""
    out = []
    typhoon_starts = {r: g.loc[~g.is_clear, "time"].tolist() for r, g in ev[ev.kind == "typhoon"].groupby("region")}
    for (region, kind), g in ev.groupby(["region", "kind"]):
        level, start = 0, None
        g = g.sort_values(["time", "is_clear"], ascending=[True, False])
        for _, r in g.iterrows():
            if kind in REPLACED_BY_TYPHOON and level:
                cut = [t for t in typhoon_starts.get(region, []) if start < t < r.time]
                if cut:
                    out.append({"region": region, "kind": kind, "level": level, "start": start, "end": cut[0]})
                    level, start = 0, None
            if r.is_clear:
                if level:
                    out.append({"region": region, "kind": kind, "level": level, "start": start, "end": r.time})
                    level, start = 0, None
            else:
                if level == 0:
                    level, start = r.level, r.time
                elif r.level != level:  # 등급 변경: 이전 구간을 닫고 새 구간
                    out.append({"region": region, "kind": kind, "level": level, "start": start, "end": r.time})
                    level, start = r.level, r.time
        if level:
            out.append({"region": region, "kind": kind, "level": level, "start": start, "end": end_default})
    iv = pd.DataFrame(out)
    iv = iv[iv.end > iv.start].copy()
    cap = iv.kind.map(MAX_DAYS).fillna(10)
    too_long = (iv.end - iv.start).dt.total_seconds() / 86400 > cap
    if too_long.any():
        print(f"  해제가 누락된 것으로 보이는 구간 {int(too_long.sum())}개를 종류별 상한에서 자름 "
              f"(종류별: {iv[too_long].groupby('kind').size().to_dict()})")
        iv.loc[too_long, "end"] = iv.loc[too_long, "start"] + pd.to_timedelta(cap[too_long], unit="D")
    iv["hours"] = (iv.end - iv.start).dt.total_seconds() / 3600
    return iv.reset_index(drop=True)


def daily_from_intervals(iv, d0, d1):
    days = pd.date_range(d0, d1, freq="D")
    recs = []
    for _, r in iv.iterrows():
        for day in pd.date_range(r.start.normalize(), r.end.normalize(), freq="D"):
            a, b = max(r.start, day), min(r.end, day + pd.Timedelta(days=1))
            a6, b6 = max(r.start, day + pd.Timedelta(hours=6)), min(r.end, day + pd.Timedelta(days=1))
            recs.append({"region": r.region, "kind": r.kind, "date": day, "level": r.level,
                         "hours": max(0.0, (b - a).total_seconds() / 3600), "hours_biz": max(0.0, (b6 - a6).total_seconds() / 3600)})
    x = pd.DataFrame(recs)
    x = x.groupby(["region", "kind", "date"]).agg(level=("level", "max"), hours=("hours", "sum"), hours_biz=("hours_biz", "sum")).reset_index()
    wide = x.pivot(index=["region", "date"], columns="kind", values=["level", "hours", "hours_biz"])
    wide.columns = [f"wrn_{k}_{v}" for v, k in wide.columns]
    idx = pd.MultiIndex.from_product([["gangnam", "chuncheon"], days], names=["region", "date"])
    wide = wide.reindex(idx).fillna(0).reset_index()
    for k in KINDS.values():
        for v in ["level", "hours", "hours_biz"]:
            if f"wrn_{k}_{v}" not in wide:
                wide[f"wrn_{k}_{v}"] = 0.0
        wide[f"wrn_{k}"] = (wide[f"wrn_{k}_level"] > 0).astype(int)
        wide[f"wrn_{k}_level"] = wide[f"wrn_{k}_level"].astype(int)
    return wide


def main():
    if not SRC.exists() or not list(SRC.glob("FCT_WRN_*.csv")):
        print(f"  {SRC} 에 특보 이력 파일이 없습니다. 건너뜁니다.")
        return
    print("[1] 통보문 읽기")
    w = read_warnings()
    print(f"  통보문 {len(w):,}건(전국·지역 중복 포함)")
    print("[2] 항목 → 지역별 이벤트")
    ev = build_events(w)
    print("  이벤트 수:", ev.groupby(["region", "kind"]).size().unstack("kind", fill_value=0).to_string())
    d1 = pd.Timestamp("2025-12-31 23:59")
    iv = intervals(ev[ev.time <= d1], d1)
    save(iv, "warnings_events")
    print("[3] 일별 특보 상태")
    daily = daily_from_intervals(iv, "2005-07-01", "2025-12-31")
    save(daily, "warnings_daily")
    daily["year"] = daily.date.dt.year
    ann = daily.groupby(["region", "year"])[[f"wrn_{k}" for k in KINDS.values()]].sum().reset_index()
    ann_biz = daily.assign(**{f"wrn_{k}_bizday": (daily[f"wrn_{k}_hours_biz"] >= 1).astype(int) for k in KINDS.values()})
    ann_biz = ann_biz.groupby(["region", "year"])[[f"wrn_{k}_bizday" for k in KINDS.values()]].sum().reset_index()
    ann = ann.merge(ann_biz, on=["region", "year"])
    save(ann, "warnings_annual")
    a25 = daily[(daily.date >= "2025-07-01")]
    for region, g in a25.groupby("region"):
        print(f"  {region} 2025.7~12: 호우특보일 {int(g.wrn_rain.sum())}(영업시간 {int((g.wrn_rain_hours_biz >= 1).sum())}), 폭염특보일 {int(g.wrn_heat.sum())}, "
              f"한파 {int(g.wrn_cold.sum())}, 대설 {int(g.wrn_snow.sum())}, 강풍 {int(g.wrn_wind.sum())}, 태풍 {int(g.wrn_typhoon.sum())}")


if __name__ == "__main__":
    main()
