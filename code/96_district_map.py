"""96. 상권 지도 데이터: 강남 공식 상권(서울시 상권분석서비스)과 우리 밀집 후보(강남·춘천), 전통시장·골목형상점가·관광지를 한 JSON·CSV로.

입력(원본은 D:/Bigcon_data)
- 서울 영역-상권 shapefile(EPSG:5181, 상권 1,650개 → 강남구 103개: 골목 54·발달 39·전통시장 9·관광특구 1), 영역-행정동 shapefile
- 추정매출-상권(2024·2025 zip), 점포-상권(2025 zip), 길단위인구·집객시설·상권변화지표-상권(API CSV)
- 95b번의 밀집 후보(al_clusters_rect, al_cluster_stores_rect: 원형 창·길 따라 창 두 방식), 상가정보(stores_clean), st_industry·r_rate_per_store(업종 손실 단가)
- 공공데이터포털: 전국전통시장표준데이터, 전국관광지정보표준데이터, 골목형상점가 현황(2024.12), 춘천시 상권 매출현황(읍면동×업종 구간, 2023.7)
좌표: 5181(중부원점) → 경위도는 뉴턴 반복으로 역변환, 상가정보 x,y(UTM-K 5179)는 같은 식의 정변환으로 맞춘다. shapely로 면적·포함 판정.
밀집 후보는 두 방식(원형 창, 길 따라 창)을 모두 넣어 지도에서 버튼으로 바꿔 볼 수 있게 한다(기본 길 따라 창). 구역 외곽선은 업소 점을 30m 넓힌 뒤 12m 깎은 선(업소 주변 18m)이라 길을 따라 늘어선 구역은 띠 모양이 된다.
출력: outputs/demo/district_data.json, 페이지 outputs/demo/상권지도.html(district_template.html의 __DATA__에 JSON 주입), outputs/tables/al_gangnam_official_districts.csv, al_gangnam_candidates.csv, al_chuncheon_districts.csv
"""
import io
import json
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.strtree import STRtree

import common as C

EXT = C.EXT_SEOUL_SALES
PORTAL = C.EXT_BIG / "공공데이터포털"
P5181 = dict(lon0=127.0, lat0=38.0, k0=1.0, fe=200000.0, fn=500000.0)
P5179 = dict(lon0=127.5, lat0=38.0, k0=0.9996, fe=1_000_000.0, fn=2_000_000.0)
A_, F_ = 6378137.0, 1 / 298.257222101
N_ = F_ / (2 - F_)
AA = A_ / (1 + N_) * (1 + N_ ** 2 / 4 + N_ ** 4 / 64)
ALPHA = [N_ / 2 - 2 * N_ ** 2 / 3 + 5 * N_ ** 3 / 16, 13 * N_ ** 2 / 48 - 3 * N_ ** 3 / 5, 61 * N_ ** 3 / 240]


def tm_forward(lon, lat, lon0, lat0, k0, fe, fn):
    """경위도(도) → 횡메르카토르(m). Karney 급수, GRS80."""
    lon, lat, lon0r, lat0r = np.radians(lon), np.radians(lat), np.radians(lon0), np.radians(lat0)

    def xi_eta(phi, lam):
        t = np.sinh(np.arctanh(np.sin(phi)) - 2 * np.sqrt(N_) / (1 + N_) * np.arctanh(2 * np.sqrt(N_) / (1 + N_) * np.sin(phi)))
        xi_, eta_ = np.arctan2(t, np.cos(lam - lon0r)), np.arctanh(np.sin(lam - lon0r) / np.sqrt(1 + t ** 2))
        xi, eta = xi_.copy(), eta_.copy()
        for j, a in enumerate(ALPHA, 1):
            xi = xi + a * np.sin(2 * j * xi_) * np.cosh(2 * j * eta_)
            eta = eta + a * np.cos(2 * j * xi_) * np.sinh(2 * j * eta_)
        return xi, eta

    xi, eta = xi_eta(lat, lon)
    xi0, _ = xi_eta(np.array([lat0r]), np.array([lon0r]))
    return fe + k0 * AA * eta, fn + k0 * AA * (xi - xi0[0])


def tm_inverse(E, N, **p):
    """횡메르카토르(m) → 경위도(도). 정변환을 뉴턴 반복으로 뒤집는다(오차 1mm 미만)."""
    E, N = np.asarray(E, float), np.asarray(N, float)
    lon = np.full_like(E, p["lon0"]) + (E - p["fe"]) / (p["k0"] * AA) / np.cos(np.radians(p["lat0"])) * 180 / np.pi
    lat = np.full_like(N, p["lat0"]) + (N - p["fn"]) / (p["k0"] * AA) * 180 / np.pi
    for _ in range(8):
        x, y = tm_forward(lon, lat, **p)
        d = 1e-6
        x1, y1 = tm_forward(lon + d, lat, **p)
        x2, y2 = tm_forward(lon, lat + d, **p)
        j11, j21, j12, j22 = (x1 - x) / d, (y1 - y) / d, (x2 - x) / d, (y2 - y) / d
        det = j11 * j22 - j12 * j21
        rx, ry = E - x, N - y
        lon = lon + (j22 * rx - j12 * ry) / det
        lat = lat + (-j21 * rx + j11 * ry) / det
    return lon, lat


def read_shapefile(zip_path):
    """zip 안의 shp/dbf를 읽어 (속성 dict, [링 좌표 배열…]) 목록으로. 폴리곤(type 5)만."""
    z = zipfile.ZipFile(zip_path)
    dbf = z.open([n for n in z.namelist() if n.lower().endswith(".dbf")][0]).read()
    shp = z.open([n for n in z.namelist() if n.lower().endswith(".shp")][0]).read()
    enc = "utf-8"
    cpg = [n for n in z.namelist() if n.lower().endswith(".cpg")]
    if cpg:
        enc = z.open(cpg[0]).read().decode("ascii", "replace").strip() or "utf-8"
    nrec = struct.unpack("<I", dbf[4:8])[0]
    hdrlen, reclen = struct.unpack("<H", dbf[8:10])[0], struct.unpack("<H", dbf[10:12])[0]
    fields, pos = [], 32
    while dbf[pos] != 0x0D:
        fields.append((dbf[pos:pos + 11].split(b"\x00")[0].decode("ascii", "replace"), dbf[pos + 16]))
        pos += 32
    attrs = []
    for i in range(nrec):
        rec, off, vals = dbf[hdrlen + i * reclen: hdrlen + (i + 1) * reclen], 1, {}
        for name, ln in fields:
            vals[name] = rec[off:off + ln].decode(enc, "replace").strip()
            off += ln
        attrs.append(vals)
    geoms, pos = [], 100
    while pos < len(shp):
        _, clen = struct.unpack(">ii", shp[pos:pos + 8])
        content = shp[pos + 8: pos + 8 + clen * 2]
        pos += 8 + clen * 2
        stype = struct.unpack("<i", content[:4])[0]
        if stype != 5:
            geoms.append([])
            continue
        nparts, npts = struct.unpack("<ii", content[36:44])
        parts = struct.unpack(f"<{nparts}i", content[44:44 + 4 * nparts])
        pts = np.frombuffer(content[44 + 4 * nparts: 44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(npts, 2)
        rings = [pts[parts[k]: (parts[k + 1] if k + 1 < nparts else npts)] for k in range(nparts)]
        geoms.append(rings)
    return attrs, geoms


def rings_to_polygon(rings):
    """바깥 링(시계 방향)들을 합치고 안쪽 링(반시계)은 구멍으로 뺀다."""
    outers, holes = [], []
    for r in rings:
        if len(r) < 4:
            continue
        pg = Polygon(r)
        if not pg.is_valid:
            pg = pg.buffer(0)
        (holes if pg.exterior.is_ccw else outers).append(pg)
    if not outers:
        outers, holes = holes, []
    geom = outers[0]
    for o in outers[1:]:
        geom = geom.union(o)
    for h in holes:
        geom = geom.difference(h)
    return geom


def poly_to_lonlat(geom, proj, tol=2.0):
    """shapely 폴리곤(투영 m) → 경위도 링 목록(간략화)."""
    g = geom.simplify(tol)
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    out = []
    for pg in polys:
        xy = np.asarray(pg.exterior.coords)
        lon, lat = tm_inverse(xy[:, 0], xy[:, 1], **proj)
        out.append([[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon, lat)])
    return out


def poly_5181_to_5179(geom):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    res = []
    for pg in polys:
        xy = np.asarray(pg.exterior.coords)
        lon, lat = tm_inverse(xy[:, 0], xy[:, 1], **P5181)
        x, y = tm_forward(lon, lat, **P5179)
        res.append(Polygon(np.column_stack([x, y])))
    g = res[0]
    for r in res[1:]:
        g = g.union(r)
    return g


def read_csv_any(path, **kw):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str, **kw)
        except Exception:
            continue
    raise ValueError(path)


def store_stats(stores_in):
    """구역 안 상가정보 업소의 요약(우리 손실 단가 포함)."""
    g = stores_in
    mapped = g[g.industry.notna()]
    top = ", ".join(f"{i}({n})" for i, n in mapped.industry.value_counts().head(3).items())
    return {"업소 수(상가정보)": int(len(g)), "카드 업종 매핑 업소": int(len(mapped)), "음식 업종 비중": round(float((g.big == "음식").mean()), 2) if len(g) else None,
            "취약 등급 업소": int(mapped.grade_V.isin(["취약", "매우 취약"]).sum()),
            "호우일 하루 손실(만원)": round(float(mapped.rain_loss.fillna(0).sum()), 1), "20mm 시즌 기대 지급액(만원)": round(float(mapped.pay20.fillna(0).sum()), 1),
            "주요 업종(상위 3)": top}


def main():
    out_dir = C.ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 상가정보 업소 + 업종 손실 단가
    st = C.load("stores_clean")
    sti = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")[["region", "industry", "rain_loss_day_per_store_만원", "sales_day_per_store_만원", "grade_V"]].rename(columns={"rain_loss_day_per_store_만원": "rain_loss", "sales_day_per_store_만원": "sales_day"})
    rate = pd.read_csv(C.TAB / "r_rate_per_store.csv", encoding="utf-8-sig")[["region", "industry", "시즌 기대 지급액 업소당(만원, 20mm 트리거)"]].rename(columns={"시즌 기대 지급액 업소당(만원, 20mm 트리거)": "pay20"})
    st = st.merge(sti, on=["region", "industry"], how="left").merge(rate, on=["region", "industry"], how="left")
    data = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"), "regions": {}}
    # 서울시 업종(63) → 카드 업종 → 호우일 손실률(호우일 하루 손실 / 평상 매출, 업종 단가)
    smap = pd.read_csv(C.META / "seoul_ind_map.csv", encoding="utf-8-sig").dropna(subset=["seoul_names"])
    smap = smap.assign(seoul_nm=smap.seoul_names.str.split("|")).explode("seoul_nm")[["industry", "seoul_nm"]]
    sti_full = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")
    rate_card = sti_full[sti_full.region == "gangnam"].set_index("industry").apply(lambda r: r.loss_per_day_rain_biz10 / r.sales_day if r.sales_day else 0, axis=1)
    seoul_rate = smap.assign(rate=smap.industry.map(rate_card)).dropna(subset=["rate"]).set_index("seoul_nm")["rate"].to_dict()
    CHANGE_ORD = {"다이나믹": 0, "상권확장": 1, "정체": 2, "상권축소": 3}

    # ------------------------------------------------------------ 강남 공식 상권
    attrs, geoms = read_shapefile(EXT / "서울시 상권분석서비스(영역-상권).zip")
    gn_idx = [i for i, a in enumerate(attrs) if a["SIGNGU_CD"] == "11680"]
    sales = {}
    for y in [2024, 2025]:
        zp = next(EXT.glob(f"*추정매출-상권)_{y}년.zip"))
        z = zipfile.ZipFile(zp)
        d = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".csv")][0]), encoding="cp949", dtype={"상권_코드": str, "기준_년분기_코드": str})
        sales[y] = d
    S = pd.concat(sales.values(), ignore_index=True)
    S["q"] = S["기준_년분기_코드"].str[4].astype(int)
    S["year"] = S["기준_년분기_코드"].str[:4].astype(int)
    h2 = S[S.q.isin([3, 4])]
    amt = h2.groupby(["상권_코드", "year"])["당월_매출_금액"].sum().unstack("year")
    wk = h2[h2.year == 2025].groupby("상권_코드")[["주중_매출_금액", "주말_매출_금액"]].sum()
    tb = h2[h2.year == 2025].groupby("상권_코드")[["시간대_11~14_매출_금액", "시간대_14~17_매출_금액", "시간대_17~21_매출_금액"]].sum()
    top_ind = h2[h2.year == 2025].groupby(["상권_코드", "서비스_업종_코드_명"])["당월_매출_금액"].sum().reset_index().sort_values("당월_매출_금액", ascending=False)
    top_ind = top_ind.groupby("상권_코드").head(3).groupby("상권_코드")["서비스_업종_코드_명"].apply(lambda s: ", ".join(s))
    zp = next(EXT.glob("*점포-상권)_2025년.zip"))
    z = zipfile.ZipFile(zp)
    P = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".csv")][0]), encoding="cp949", dtype={"trdar_cd": str, "stdr_yyqu_cd": str})
    P = P[P.stdr_yyqu_cd == P.stdr_yyqu_cd.max()]
    stor = P.groupby("trdar_cd").agg(stor_co=("stor_co", "sum"), frc=("frc_stor_co", "sum"), opbiz=("opbiz_stor_co", "sum"), clsbiz=("clsbiz_stor_co", "sum"))
    F = pd.read_csv(EXT / "서울시 상권분석서비스(길단위인구-상권)_API_전체.csv", encoding="utf-8-sig", dtype={"TRDAR_CD": str, "STDR_YYQU_CD": str})
    F = F[F.STDR_YYQU_CD.isin(["20253", "20254"])].groupby("TRDAR_CD")[["TOT_FLPOP_CO", "AGRDE_60_ABOVE_FLPOP_CO", "SAT_FLPOP_CO", "SUN_FLPOP_CO"]].mean()
    Fc = pd.read_csv(EXT / "서울시 상권분석서비스(집객시설-상권)_API_전체.csv", encoding="utf-8-sig", dtype={"TRDAR_CD": str, "STDR_YYQU_CD": str})
    Fc = Fc.sort_values("STDR_YYQU_CD").groupby("TRDAR_CD").last()
    Ix = pd.read_csv(EXT / "서울시 상권분석서비스(상권변화지표-상권)_API_전체.csv", encoding="utf-8-sig", dtype={"TRDAR_CD": str, "STDR_YYQU_CD": str})
    Ix = Ix.sort_values("STDR_YYQU_CD").groupby("TRDAR_CD").last()
    gn_stores = st[st.region == "gangnam"].reset_index(drop=True)
    pts = [Point(x, y) for x, y in zip(gn_stores.x, gn_stores.y)]
    tree = STRtree(pts)
    official, rows = [], []
    for i in gn_idx:
        a = attrs[i]
        geom = rings_to_polygon(geoms[i])
        g79 = poly_5181_to_5179(geom)
        cand = tree.query(g79)
        inside = [k for k in cand if g79.contains(pts[k])]
        ss = store_stats(gn_stores.iloc[inside])
        code = a["TRDAR_CD"]
        s25 = float(amt.loc[code, 2025]) if code in amt.index and 2025 in amt.columns and pd.notna(amt.loc[code, 2025]) else None
        s24 = float(amt.loc[code, 2024]) if code in amt.index and 2024 in amt.columns and pd.notna(amt.loc[code, 2024]) else None
        rec = {"code": code, "name": a["TRDAR_CD_N"], "type": a["TRDAR_SE_1"], "emd": a["ADSTRD_CD_"], "area_m2": float(a["RELM_AR"] or geom.area),
               "sales_h2_2025_억": round(s25 / 1e8, 1) if s25 else None, "sales_h2_yoy_%": round((s25 / s24 - 1) * 100, 1) if s25 and s24 else None,
               "weekend_share_%": round(float(wk.loc[code, "주말_매출_금액"] / max(wk.loc[code].sum(), 1) * 100), 1) if code in wk.index else None,
               "evening_share_%": round(float(tb.loc[code, "시간대_17~21_매출_금액"] / max(h2[(h2.year == 2025) & (h2.상권_코드 == code)]["당월_매출_금액"].sum(), 1) * 100), 1) if code in tb.index else None,
               "top_sales_industries": top_ind.get(code, ""),
               "stores_seoul": int(stor.loc[code, "stor_co"]) if code in stor.index else None, "franchise": int(stor.loc[code, "frc"]) if code in stor.index else None,
               "open_q": int(stor.loc[code, "opbiz"]) if code in stor.index else None, "close_q": int(stor.loc[code, "clsbiz"]) if code in stor.index else None,
               "flpop_q": round(float(F.loc[code, "TOT_FLPOP_CO"])) if code in F.index else None, "flpop_60p_%": round(float(F.loc[code, "AGRDE_60_ABOVE_FLPOP_CO"] / max(F.loc[code, "TOT_FLPOP_CO"], 1) * 100), 1) if code in F.index else None,
               "flpop_weekend_%": round(float((F.loc[code, "SAT_FLPOP_CO"] + F.loc[code, "SUN_FLPOP_CO"]) / max(F.loc[code, "TOT_FLPOP_CO"], 1) * 100), 1) if code in F.index else None,
               "facilities": int(Fc.loc[code, "VIATR_FCLTY_CO"]) if code in Fc.index else None, "subway": int(Fc.loc[code, "SUBWAY_STATN_CO"]) if code in Fc.index else None,
               "change_ix": Ix.loc[code, "TRDAR_CHNGE_IX_NM"] if code in Ix.index else None, "oper_months": float(Ix.loc[code, "OPR_SALE_MT_AVRG"]) if code in Ix.index else None,
               **ss}
        rec["sales_per_store_만원"] = round(s25 / rec["stores_seoul"] / 1e4 / 184, 1) if s25 and rec["stores_seoul"] else None  # 하반기 184일 기준 점포당 하루 매출
        # 서울시 업종 매출 × 카드 업종 호우 손실률 → 측정 매출 기반의 호우일 하루 손실
        hh = h2[(h2.year == 2025) & (h2.상권_코드 == code)]
        if len(hh) and s25:
            ls = float(sum(v * seoul_rate.get(k, 0.0) for k, v in hh.groupby("서비스_업종_코드_명")["당월_매출_금액"].sum().items())) / 184
            rec["loss_seoul_만원"] = round(ls / 1e4, 1)
            rec["loss_rate_seoul_%"] = round(ls / (s25 / 184) * 100, 2)
        else:
            rec["loss_seoul_만원"], rec["loss_rate_seoul_%"] = None, None
        rec["close_rate_%"] = round(rec["close_q"] / rec["stores_seoul"] * 100, 2) if rec.get("close_q") is not None and rec.get("stores_seoul") else None
        rec["change_ord"] = CHANGE_ORD.get(rec.get("change_ix"))
        rec["vuln_share_%"] = round(rec["취약 등급 업소"] / rec["카드 업종 매핑 업소"] * 100, 1) if rec["카드 업종 매핑 업소"] else None
        rec["loss_rate_alloc_%"] = round(rec["호우일 하루 손실(만원)"] / (s25 / 184 / 1e4) * 100, 2) if s25 else None
        rows.append(rec)
        official.append({**rec, "rings": poly_to_lonlat(geom, P5181)})
    G = pd.DataFrame(rows).sort_values(["type", "sales_h2_2025_억"], ascending=[True, False])
    G.to_csv(C.TAB / "al_gangnam_official_districts.csv", index=False, encoding="utf-8-sig")
    print(f"강남 공식 상권 {len(G)}개: {G.type.value_counts().to_dict()}; 2025 하반기 매출 합 {G.sales_h2_2025_억.sum():,.0f}억, 상가정보 업소 {G['업소 수(상가정보)'].sum():,}개")

    # ------------------------------------------------------------ 밀집 후보(강남 법정30, 춘천 완화15): 원형 창·길 따라 창(95b) 두 방식 → 업소 주변 외곽선
    A = pd.read_csv(C.TAB / "al_clusters_rect.csv", encoding="utf-8-sig")
    M = pd.read_csv(C.TAB / "al_cluster_stores_rect.csv", encoding="utf-8-sig", dtype={"상가업소번호": str})
    M = M.merge(st[["상가업소번호", "big", "grade_V", "rain_loss", "pay20", "sales_day"]], on="상가업소번호", how="left")
    RULE = {"gangnam": "법정(30개)", "chuncheon": "완화(15개)"}
    METHODS = {"rect": {"name": "길따라", "label": "길 따라 창", "desc": "업소가 늘어선 방향으로 돌린 2천㎡ 직사각형 창(100×20·80×25·63×32m 중 최댓값) 안 업소 수"},
               "circle": {"name": "원형", "label": "원형 창", "desc": "반경 25.2m 원(2천㎡) 안 업소 수"}}

    def outline(xy):
        """업소 점들을 30m 넓힌 뒤 12m 깎은 외곽선(업소 주변 18m). 핵 연결 거리 50m가 끊기지 않고, 길을 따라 늘어선 구역은 띠 모양이 된다."""
        return MultiPoint(xy).buffer(30, 6).buffer(-12, 6)

    cands = {region: {} for region in C.REGIONS}
    for region in C.REGIONS:
        rule = RULE[region]
        m_all = M[M.region == region]
        for mkey, mt in METHODS.items():
            col = f"cl_{mt['name']}_{rule}"
            a = A[(A.region == region) & (A.방법 == mt["name"]) & (A.기준 == rule) & (~A["단일 건물 구역"])]
            lst = []
            for _, r in a.iterrows():
                g = m_all[m_all[col] == r.cluster]
                if len(g) < 3:
                    continue
                shape = outline(list(zip(g.x, g.y)))
                ss = store_stats(g)
                mapped = g[g.industry.notna()]
                sales_sum = float(mapped.sales_day.fillna(0).sum())
                lst.append({"id": int(r.cluster), "method": mkey, "method_label": mt["label"], "name": f"{r.법정동} 밀집구역 {int(r.cluster)}", "emd": r.행정동, "ldong": r.법정동,
                            "area_m2": round(float(shape.area)), "n_stores": int(len(g)), "mapped": int(len(mapped)), "food_share": ss["음식 업종 비중"], "vuln_stores": ss["취약 등급 업소"],
                            "rain_loss_만원": ss["호우일 하루 손실(만원)"], "pay20_만원": ss["20mm 시즌 기대 지급액(만원)"], "sales_day_만원": round(sales_sum, 1),
                            "loss_rate_%": round(ss["호우일 하루 손실(만원)"] / sales_sum * 100, 1) if sales_sum else None, "top": ss["주요 업종(상위 3)"],
                            "vuln_share_%": round(ss["취약 등급 업소"] / len(mapped) * 100, 1) if len(mapped) else None,
                            "rings": poly_to_lonlat(shape, P5179, tol=1.0)})
            cands[region][mkey] = lst
            print(f"{C.REGION_NAME[region]} 밀집 후보({mt['label']}, {rule}) {len(lst)}개, 업소 {sum(c['n_stores'] for c in lst):,}개, 호우일 하루 손실 {sum(c['rain_loss_만원'] for c in lst) / 1e4:,.1f}억")
    pd.DataFrame([{"방법": c["method_label"], **{k: v for k, v in c.items() if k not in ("rings", "method", "method_label")}} for mk in METHODS for c in cands["gangnam"][mk]]).to_csv(C.TAB / "al_gangnam_candidates.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ 춘천: 읍면동 매출 구간(2023.7), 골목형상점가(지정), 전통시장, 관광지
    cs = read_csv_any(PORTAL / "강원특별자치도 춘천시_상권 매출현황_20230701.csv")
    cs = cs[cs.기준일자 == cs.기준일자.max()]
    order = ["0원", "천만원 미만", "천만원대"] + [f"{i}억대" for i in range(1, 100)]
    cs["rank"] = cs.매출액.map({v: i for i, v in enumerate(order)}).fillna(-1)
    emd_sales = cs.sort_values("rank", ascending=False).groupby("읍면동").apply(lambda g: ", ".join(f"{r.업종구분} {r.매출액}" for r in g.head(3).itertuples()), include_groups=False).to_dict()
    for c in [c for lst in cands["chuncheon"].values() for c in lst]:
        c["emd_sales_2023"] = emd_sales.get(c["emd"], "")
    tm = read_csv_any(PORTAL / "전국전통시장표준데이터.csv")
    tm = tm[tm.소재지도로명주소.fillna("").str.contains("춘천시|강남구")]
    markets = [{"region": "chuncheon" if "춘천" in r.소재지도로명주소 else "gangnam", "name": r.시장명, "type": r.시장유형, "stores": None if pd.isna(r.점포수) else int(float(r.점포수)),
                "items": r.취급품목 if isinstance(r.취급품목, str) else "", "lon": float(r.경도), "lat": float(r.위도), "addr": r.소재지도로명주소} for r in tm.itertuples()]
    tour = read_csv_any(PORTAL / "전국관광지정보표준데이터.csv")
    tour = tour[tour.소재지도로명주소.fillna("").str.contains("춘천시")]
    tourist = [{"name": r.관광지명, "type": r.관광지구분, "lon": float(r.경도), "lat": float(r.위도), "addr": r.소재지도로명주소, "area_m2": None if pd.isna(r.면적) else r.면적} for r in tour.itertuples()]
    al = read_csv_any(PORTAL / "소상공인시장진흥공단_골목형상점가 현황정보(241231).csv")
    al = al[al.주소.fillna("").str.contains("춘천시")]
    raw_addr = read_csv_any(C.ROOT / "data" / "external" / "소상공인시장진흥공단_상가(상권)정보_강원_202606.csv", usecols=["상가업소번호", "도로명주소", "상호명"])
    cc = st[st.region == "chuncheon"].merge(raw_addr, on="상가업소번호", how="left")
    # 춘천사랑상품권 가맹점 ↔ 상가정보: 상호명(공백 제거) + 도로명 번지로 맞춘다
    vch = read_csv_any(PORTAL / "강원특별자치도 춘천시_춘천사랑상품권 가맹점 정보_20251119.csv")
    def norm_name(x):
        return re.sub(r"[\s()\-·.,]", "", str(x)).lower()
    def road_key(addr):
        m2 = re.search(r"춘천시\s+(?:\S+?[읍면]\s+)?([^\s(]+(?:로|길))\s*(\d+)", str(addr))
        return f"{m2.group(1)} {m2.group(2)}" if m2 else None
    vch["k1"] = vch.상호명.map(norm_name)
    vch["k2"] = vch.주소.map(road_key)
    vkeys = set(zip(vch.k1, vch.k2))
    vnames = set(vch.k1)
    cc["k1"] = cc.상호명.map(norm_name) if "상호명" in cc else ""
    if "상호명" not in cc:
        raw_nm = read_csv_any(C.ROOT / "data" / "external" / "소상공인시장진흥공단_상가(상권)정보_강원_202606.csv", usecols=["상가업소번호", "상호명"])
        cc = cc.merge(raw_nm, on="상가업소번호", how="left")
        cc["k1"] = cc.상호명.map(norm_name)
    cc["k2"] = cc.도로명주소.map(road_key)
    cc["voucher"] = [((a, b) in vkeys) or (a in vnames and a != "") for a, b in zip(cc.k1, cc.k2)]
    print(f"상품권 가맹점 매칭: 상가정보 춘천 업소 {len(cc):,}개 중 {int(cc.voucher.sum()):,}개 가맹 추정({cc.voucher.mean()*100:.0f}%)")
    cols_c = {mk: f"cl_{mt['name']}_{RULE['chuncheon']}" for mk, mt in METHODS.items()}
    mm = M[M.region == "chuncheon"][["상가업소번호", *cols_c.values()]].merge(cc[["상가업소번호", "voucher"]], on="상가업소번호", how="left")
    for mk, col in cols_c.items():
        vshare = mm[mm[col] >= 0].groupby(col).voucher.mean()
        for c in cands["chuncheon"][mk]:
            c["voucher_share_%"] = round(float(vshare[c["id"]]) * 100, 1) if c["id"] in vshare.index else None
    alley_pts = []
    for r in al.itertuples():
        m = re.search(r"춘천시\s+(\S+?)\s+(\d+)", r.주소)
        road, num = (m.group(1), int(m.group(2))) if m else (r.주소.split()[-2], None)
        hit = cc[cc.도로명주소.fillna("").str.contains(re.escape(road))]
        if num is not None and len(hit):
            nums = hit.도로명주소.str.extract(re.escape(road) + r"\s+(\d+)")[0].astype(float)
            near = hit[nums.between(num - 40, num + 40)]
            hit = near if len(near) >= 3 else hit
        if len(hit):
            lon, lat = tm_inverse(np.array([hit.x.mean()]), np.array([hit.y.mean()]), **P5179)
            alley_pts.append({"name": r.시장명, "stores": int(float(r.전체점포)), "items": r.취급상품 if isinstance(r.취급상품, str) else "", "addr": r.주소, "lon": round(float(lon[0]), 5), "lat": round(float(lat[0]), 5), "geocode_n": int(len(hit))})
    hit = cc[cc.도로명주소.fillna("").str.contains("강촌로")]
    if len(hit):
        lon, lat = tm_inverse(np.array([hit.x.mean()]), np.array([hit.y.mean()]), **P5179)
        alley_pts.append({"name": "강촌리 골목형상점가(2026.6 지정)", "stores": 80, "items": "음식·숙박", "addr": "춘천시 남산면 강촌리", "lon": round(float(lon[0]), 5), "lat": round(float(lat[0]), 5), "geocode_n": int(len(hit))})
    print(f"춘천 지정 골목형상점가 {len(alley_pts)}개(좌표는 상가정보 주소 매칭), 전통시장 {sum(m['region']=='chuncheon' for m in markets)}개, 관광지 {len(tourist)}개")

    # ------------------------------------------------------------ 배경: SK 유동인구 200m 격자, 행정동 경계(강남)·라벨
    cells = C.load("sk_cell")
    emd_pop = C.load("sk_emd_pop")
    bg = {}
    for region in C.REGIONS:
        c = cells[(cells.region == region) & (cells["pop"] > 0)].copy()
        # 상권 폴리곤 주변(±4km)만, 춘천은 400m 격자로(셀 수를 줄여 지도가 가볍게)
        pts_xy = M[(M.region == region) & (M[[f"cl_{mt['name']}_{RULE[region]}" for mt in METHODS.values()]].ge(0).any(axis=1))][["x", "y"]]
        cx0, cx1, cy0, cy1 = pts_xy.x.min() - 4000, pts_xy.x.max() + 4000, pts_xy.y.min() - 4000, pts_xy.y.max() + 4000
        c = c[c.x.between(cx0, cx1) & c.y.between(cy0, cy1)]
        gs = 200 if region == "gangnam" else 400
        c["gx"], c["gy"] = (c.x // gs) * gs + gs / 2, (c.y // gs) * gs + gs / 2
        g = c.groupby(["gx", "gy"])["pop"].sum().reset_index()
        lon, lat = tm_inverse(g.gx.values, g.gy.values, **P5179)
        bg[region] = {"size_m": gs, "cells": [[round(float(a), 5), round(float(b), 5), round(float(p))] for a, b, p in zip(lon, lat, g["pop"])]}
    eattrs, egeoms = read_shapefile(EXT / "서울시 상권분석서비스(영역-행정동).zip")
    emd_lines = []
    for a, gg in zip(eattrs, egeoms):
        if a.get("SIGNGU_CD") == "11680" or (a.get("ADSTRD_CD", "").startswith("11680")):
            geom = rings_to_polygon(gg)
            emd_lines.append({"name": a.get("ADSTRD_CD_", a.get("ADSTRD_NM", "")), "rings": poly_to_lonlat(geom, P5181, tol=5.0)})
    labels = {}
    for region in C.REGIONS:
        e = emd_pop[emd_pop.region == region]
        lon, lat = tm_inverse(e.x.values, e.y.values, **P5179)
        labels[region] = [{"name": n, "lon": round(float(a), 5), "lat": round(float(b), 5), "pop60": round(float(p))} for n, a, b, p in zip(e.emd_name, lon, lat, e.pop60)]

    data["cand_methods"] = {mk: {"label": mt["label"], "desc": mt["desc"]} for mk, mt in METHODS.items()}
    data["cand_default"] = "rect"
    data["regions"]["gangnam"] = {"name": "강남구", "official": official, "candidates": cands["gangnam"], "cand_rule": RULE["gangnam"], "markets": [m for m in markets if m["region"] == "gangnam"], "alley_official": [], "tourist": [],
                                  "bg": bg["gangnam"], "emd_lines": emd_lines, "emd_labels": labels["gangnam"]}
    data["regions"]["chuncheon"] = {"name": "춘천시", "official": [], "candidates": cands["chuncheon"], "cand_rule": RULE["chuncheon"], "markets": [m for m in markets if m["region"] == "chuncheon"], "alley_official": alley_pts, "tourist": tourist,
                                    "bg": bg["chuncheon"], "emd_lines": [], "emd_labels": labels["chuncheon"]}

    def clean(o):
        """numpy 형과 NaN을 JSON에 넣을 수 있는 값으로."""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        return o

    js = json.dumps(clean(data), ensure_ascii=False)
    (out_dir / "district_data.json").write_text(js, encoding="utf-8")
    tpl = (out_dir / "district_template.html").read_text(encoding="utf-8")
    assert tpl.count("__DATA__") == 1
    (out_dir / "상권지도.html").write_text(tpl.replace("__DATA__", js.replace("</", "<\\/")), encoding="utf-8")
    # 춘천 표(두 방식 모두, 기본인 길 따라 창 먼저)
    rows = [{"구분": f"밀집 후보({c['method_label']}, {RULE['chuncheon']})", "방법": c["method_label"], "이름": c["name"], "행정동": c["emd"], "업소 수": c["n_stores"], "면적(㎡, 업소 외곽)": c["area_m2"], "음식 비중": c["food_share"], "취약 등급 업소": c["vuln_stores"],
             "호우일 하루 손실(만원)": c["rain_loss_만원"], "20mm 시즌 기대 지급액(만원)": c["pay20_만원"], "손실률(%)": c["loss_rate_%"], "취약 업소 비율(%)": c.get("vuln_share_%"), "상품권 가맹 비율(%)": c.get("voucher_share_%"), "주요 업종": c["top"], "읍면동 매출 구간(2023.7 상위3)": c["emd_sales_2023"]}
            for mk in METHODS for c in cands["chuncheon"][mk]]
    rows += [{"구분": "골목형상점가(지정)", "이름": p["name"], "행정동": "", "업소 수": p["stores"], "주요 업종": p["items"], "비고": p["addr"]} for p in alley_pts]
    rows += [{"구분": "전통시장", "이름": m["name"], "업소 수": m["stores"], "주요 업종": m["items"], "비고": m["addr"]} for m in markets if m["region"] == "chuncheon"]
    rows += [{"구분": "관광지(지정)", "이름": t["name"], "주요 업종": t["type"], "비고": t["addr"]} for t in tourist]
    pd.DataFrame(rows).to_csv(C.TAB / "al_chuncheon_districts.csv", index=False, encoding="utf-8-sig")
    print(f"저장: district_data.json({(out_dir / 'district_data.json').stat().st_size / 1e6:.1f}MB), 상권지도.html, al_gangnam_official_districts.csv, al_gangnam_candidates.csv, al_chuncheon_districts.csv")


if __name__ == "__main__":
    main()
