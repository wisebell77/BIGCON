"""09c. 한국관광데이터랩 '지역별 데이터랩' 내려받기(춘천시, 2023~2025년) 정리.

받은 zip(관광소비·방문자수, 연도별)은 월 단위 집계다. 일별 방문자 수는 들어 있지 않다.
- tour_monthly_visitors: 기준년월, 방문자수(연인원), 전년동월, 증감률
- tour_monthly_spend: 기준연월 × 업종대분류 × 방문자 유형(내국인 전체/현지인/외지인) 소비액(천원)
- 나머지 분포 표(거주지·거리·성연령·내비게이션·외국인·업종 비율)는 tour_tables_<파일명>.csv로 outputs/tables에 그대로 둔다.
원본 위치: common.EXT_TOUR (D:\\Bigcon_data\\한국관광데이터랩).
"""
import re
import zipfile
from io import StringIO

import pandas as pd

import common as C


def read_zip_csvs(path):
    z = zipfile.ZipFile(path)
    out = {}
    for info in z.infolist():
        raw = z.open(info).read()
        txt = raw.decode("utf-8-sig", "ignore")
        name = re.sub(r"^\d{14}_", "", info.filename).replace(".csv", "").strip()
        out[name] = pd.read_csv(StringIO(txt), dtype=str)
    return out


def main():
    base = C.EXT_TOUR
    visitors, spend, misc = [], [], {}
    for kind in ["방문자수", "관광소비"]:
        for zp in sorted((base / kind).glob("*_데이터랩_다운로드.zip")):
            m = re.search(r"_(\d{6})-(\d{6})_", zp.name)
            period = f"{m.group(1)}~{m.group(2)}" if m else zp.stem
            tables = read_zip_csvs(zp)
            print(f"  {zp.name}: {list(tables)}")
            for name, df in tables.items():
                if name == "방문자 수(연인원) 추이":
                    d = df.rename(columns={"기준년월": "ym", "방문자수": "visitors", "전년동월방문자수": "visitors_prev_year", "방문자수증감률": "yoy_pct"})
                    visitors.append(d)
                elif name.startswith("관광소비 추이_"):
                    d = df.rename(columns={"기준연월": "ym", "업종대분류명": "category", "소비액(천원)": "spend_k"})
                    d["visitor_type"] = name.split("_")[-1]  # 내국인(전체)/현지인/외지인
                    spend.append(d)
                else:
                    misc.setdefault(name, []).append(df.assign(기간=period))
    v = pd.concat(visitors, ignore_index=True)
    for c in ["visitors", "visitors_prev_year", "yoy_pct"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["ym"] = v["ym"].astype(str)
    v = v.drop_duplicates("ym").sort_values("ym").reset_index(drop=True)
    C.save(v, "tour_monthly_visitors")
    s = pd.concat(spend, ignore_index=True)
    s["spend_k"] = pd.to_numeric(s["spend_k"], errors="coerce")
    s["ym"] = s["ym"].astype(str)
    s = s.drop_duplicates(["ym", "category", "visitor_type"]).sort_values(["visitor_type", "category", "ym"]).reset_index(drop=True)
    C.save(s, "tour_monthly_spend")
    C.TAB.mkdir(parents=True, exist_ok=True)
    for name, parts in misc.items():
        d = pd.concat(parts, ignore_index=True)
        safe = re.sub(r"[^\w가-힣]+", "_", name).strip("_")
        d.to_csv(C.TAB / f"tour_tables_{safe}.csv", index=False, encoding="utf-8-sig")
    print(f"  방문자 월 {v.ym.min()}~{v.ym.max()} ({len(v)}개월), 소비 표 {len(s)}행, 기타 표 {len(misc)}개 → outputs/tables/tour_tables_*.csv")

    # 빠른 점검: 하반기(7~12월) 방문자·외지인 소비 비중
    v["year"] = v.ym.str[:4]
    v["month"] = v.ym.str[4:].astype(int)
    h2 = v[v.month >= 7].groupby("year").visitors.sum() / 1e4
    print("  춘천 7~12월 방문자(만명, 연인원):", {k: round(x, 0) for k, x in h2.items()})
    tot = s[s.category == "전체"].pivot_table(index="ym", columns="visitor_type", values="spend_k")
    if {"내국인", "외지인"} <= set(tot.columns):
        share = (tot["외지인"] / tot["내국인"] * 100)
        print(f"  관광소비 중 외지인 비중: 평균 {share.mean():.1f}% (최소 {share.min():.1f}, 최대 {share.max():.1f})")
    return v, s


if __name__ == "__main__":
    main()
    print("완료")
