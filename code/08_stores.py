"""소상공인시장진흥공단 상가(상권)정보 → 지역×업종 업소 수, 행정동별 업소 수, SK 격자와의 결합.

입력: data/external/소상공인시장진흥공단_상가(상권)정보_{서울,강원}_YYYYMM.csv (공공데이터포털, 무료)
      강남구·춘천시 업소만 쓴다. 기준 시점(파일명의 YYYYMM)이 카드 데이터(2025.7~12)와 다를 수 있다.
업종 매핑: code/meta/store_map.csv (카드 업종 → 상가 업종 중분류(mid) 또는 소분류(sub) 이름). 매핑이 없는 업소는 industry가 비어 있다.
좌표: 경도·위도(WGS84) → UTM-K(EPSG:5179, SK 격자와 같은 좌표계). Karney(2011) 횡축 메르카토르 급수로 직접 계산한다(pyproj 불필요, 오차 mm 수준).
SK 결합: 업소를 가장 가까운 50m 격자 셀에 붙이고(격자 간격 50m, 셀 중심의 좌표 오프셋은 자료에서 읽음), 셀에는 가장 가까운 업소의 행정동명을 준다.

출력
- stores_clean: 업소별 (region, industry, big/mid/sub 분류, 행정동명·코드, lon, lat, x, y, cell_id)
- stores_industry: 지역×업종 업소 수(매핑된 업종만)
- stores_emd: 지역×행정동×업종 업소 수
- sk_cell_emd: SK 셀 → 행정동명(가장 가까운 업소 기준, 300m 안)
- sk_emd_pop: 지역×행정동 유동인구 구조(SK 셀 합계)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import EXT_WEATHER, META, REGION_BY_CARD, load, save

SRC = EXT_WEATHER.parent
SGG = {"강남구": "gangnam", "춘천시": "chuncheon"}
BIG_FACE = ["음식", "소매", "수리·개인", "숙박", "예술·스포츠"]  # 대면 업종 대분류


def utmk(lon, lat):
    """WGS84 경위도 → UTM-K(EPSG:5179). GRS80, 원점 38N 127.5E, 축척 0.9996, 가산 1,000,000/2,000,000."""
    a, f = 6378137.0, 1 / 298.257222101
    k0, lat0, lon0, fe, fn = 0.9996, np.radians(38.0), np.radians(127.5), 1_000_000.0, 2_000_000.0
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    al = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240]
    e = np.sqrt(f * (2 - f))

    def xi_eta(phi, lam):
        t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
        xi_, eta_ = np.arctan2(t, np.cos(lam - lon0)), np.arctanh(np.sin(lam - lon0) / np.sqrt(1 + t ** 2))
        xi, eta = xi_.copy(), eta_.copy()
        for j, aj in enumerate(al, start=1):
            xi = xi + aj * np.sin(2 * j * xi_) * np.cosh(2 * j * eta_)
            eta = eta + aj * np.cos(2 * j * xi_) * np.sinh(2 * j * eta_)
        return xi, eta

    xi, eta = xi_eta(np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)))
    xi0, _ = xi_eta(np.array([lat0]), np.array([lon0]))
    return fe + k0 * A * eta, fn + k0 * A * (xi - xi0[0])


def read_stores():
    frames = []
    for f in sorted(SRC.glob("소상공인시장진흥공단_상가(상권)정보_*.csv")):
        d = pd.read_csv(f, encoding="utf-8", low_memory=False,
                        usecols=["상가업소번호", "상호명", "상권업종대분류명", "상권업종중분류명", "상권업종소분류명", "시군구명", "행정동코드", "행정동명", "법정동명", "경도", "위도"])
        d = d[d["시군구명"].isin(SGG)].copy()
        d["region"] = d["시군구명"].map(SGG)
        d["asof"] = re.search(r"_(\d{6})\.csv$", f.name).group(1) if re.search(r"_(\d{6})\.csv$", f.name) else ""
        print(f"  {f.name}: {len(d):,}개 업소(강남구·춘천시), 기준 {d['asof'].iloc[0] if len(d) else '?'}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("상가업소번호")
    return df.rename(columns={"상권업종대분류명": "big", "상권업종중분류명": "mid", "상권업종소분류명": "sub",
                              "행정동코드": "emd_code", "행정동명": "emd_name", "법정동명": "ldong_name", "경도": "lon", "위도": "lat"})


def apply_map(df):
    m = pd.read_csv(META / "store_map.csv", encoding="utf-8-sig")
    sub_map, mid_map = {}, {}
    for _, r in m.iterrows():
        for name in str(r["names"]).split("|"):
            target = sub_map if r["level"] == "sub" else mid_map
            assert name not in target, f"중복 매핑: {name}"
            target[name] = r["industry"]
    df["industry"] = df["sub"].map(sub_map)
    df.loc[df.industry.isna(), "industry"] = df.loc[df.industry.isna(), "mid"].map(mid_map)
    unused = [n for n in list(sub_map) + list(mid_map) if n not in set(df["sub"]) | set(df["mid"])]
    if unused:
        print(f"  매핑표에는 있지만 자료에 없는 이름 {len(unused)}개: {unused[:8]}")
    face = df[df.big.isin(BIG_FACE)]
    print(f"  매핑된 업소 {df.industry.notna().sum():,}/{len(df):,} (대면 대분류 안에서는 {face.industry.notna().sum():,}/{len(face):,})")
    return df


def snap_to_cells(df, cells):
    """업소를 SK 50m 격자 셀에 붙인다. 격자 오프셋은 셀 좌표에서 읽는다."""
    ox, oy = float(cells.x.iloc[0] % 50), float(cells.y.iloc[0] % 50)
    sx = np.round((df.x - ox) / 50) * 50 + ox
    sy = np.round((df.y - oy) / 50) * 50 + oy
    df["cell_id"] = sx.round().astype(int).astype(str) + "_" + sy.round().astype(int).astype(str)
    have = set(cells.cell_id)
    df["in_sk"] = df.cell_id.isin(have).astype(int)
    return df


def main():
    files = list(SRC.glob("소상공인시장진흥공단_상가(상권)정보_*.csv"))
    if not files:
        print(f"  {SRC} 에 상가정보 파일이 없습니다. 건너뜁니다.")
        return
    print("[1] 상가정보 읽기")
    df = read_stores()
    df = apply_map(df)
    df["x"], df["y"] = utmk(df.lon.values, df.lat.values)
    chk_x, chk_y = utmk(np.array([126.9780]), np.array([37.5665]))  # 서울시청: 약 (953,900, 1,952,000)
    print(f"  좌표 변환 점검 서울시청 → ({chk_x[0]:,.0f}, {chk_y[0]:,.0f})")

    print("[2] SK 격자와 결합")
    cells = load("sk_cell")
    df = snap_to_cells(df, cells)
    for r, g in df.groupby("region"):
        print(f"  {r}: 업소 {len(g):,}, SK 셀이 있는 위치 {g.in_sk.mean():.1%}, 행정동 {g.emd_name.nunique()}개")
    save(df[["상가업소번호", "region", "industry", "big", "mid", "sub", "emd_code", "emd_name", "ldong_name", "lon", "lat", "x", "y", "cell_id", "in_sk", "asof"]], "stores_clean")

    print("[3] 집계")
    ind = df.dropna(subset=["industry"]).groupby(["region", "industry"]).agg(n_stores=("상가업소번호", "size"), n_emd=("emd_name", "nunique")).reset_index()
    save(ind, "stores_industry")
    emd = df.groupby(["region", "emd_name", "industry"], dropna=False).size().rename("n_stores").reset_index()
    save(emd, "stores_emd")

    # SK 셀 → 행정동: 가장 가까운 업소의 행정동(300m 안)
    out = []
    for r, c in cells.groupby("region"):
        s = df[df.region == r]
        tree = cKDTree(s[["x", "y"]].values)
        dist, idx = tree.query(c[["x", "y"]].values, k=1)
        emd_name = np.where(dist <= 300, s.emd_name.values[idx], None)
        out.append(pd.DataFrame({"region": r, "cell_id": c.cell_id.values, "emd_name": emd_name, "dist_store": dist}))
    ce = pd.concat(out, ignore_index=True)
    save(ce, "sk_cell_emd")
    cp = cells.merge(ce[["region", "cell_id", "emd_name"]], on=["region", "cell_id"]).dropna(subset=["emd_name"])
    cp["pop60"] = cp["pop"] * cp["share_age_60p"].fillna(0)
    cp["pop_12_17"] = cp["pop"] * cp["share_tb_12_17"].fillna(0)
    ep = cp.groupby(["region", "emd_name"]).agg(pop=("pop", "sum"), pop60=("pop60", "sum"), pop_12_17=("pop_12_17", "sum"), n_cells=("cell_id", "size"),
                                                x=("x", "mean"), y=("y", "mean")).reset_index()
    ep["share_age_60p"] = ep.pop60 / ep["pop"]
    ep["share_tb_12_17"] = ep.pop_12_17 / ep["pop"]
    save(ep, "sk_emd_pop")
    for r, g in ep.groupby("region"):
        print(f"  {r}: 행정동 {len(g)}개에 유동인구 배정, 배정 못 한 셀 {int((ce[ce.region == r].emd_name.isna()).sum()):,}개")


if __name__ == "__main__":
    main()
