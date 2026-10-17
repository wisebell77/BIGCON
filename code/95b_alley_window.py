"""95b. 골목 상권 후보, 길 따라 창 방식: 원형(반경 25.2m) 대신 '2천㎡ 직사각형 창'을 업소가 늘어선 방향으로 돌려 센다.

법령은 "2천㎡ 이내 면적에 30개 이상"이라는 밀도만 말하고 모양은 정하지 않는다. 길 따라 늘어선 상권(폭 20m × 길이 100m도 2천㎡)은 원으로는 못 잡는다.
- 각 업소마다 반경 60m 안 업소들의 주성분 방향(길 방향)을 구하고, 그 방향으로 100×20, 80×25, 63×32m(모두 2천㎡) 직사각형 창과 원(반경 25.2m) 안의 업소 수를 센다. 최댓값이 기준(30 또는 15) 이상이면 밀집 핵.
- 핵끼리 50m 이내면 같은 구역, 핵이 아닌 업소는 30m 이내 구역에 붙인다. 한 건물이 절반 이상이면 '단일 건물 구역'.
- 검증: 춘천 지정 골목형상점가 5곳(주소로 맞춘 업소)이 구역에 들어오는 비율, 강남 공식 골목상권 폴리곤 안 업소의 포함 비율을 원형 방식과 비교한다.
출력: al_clusters_rect.csv, al_cluster_stores_rect.csv, al_window_compare.csv, 그림 al03_alley_window.png
"""
import re
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Point
from shapely.strtree import STRtree
import matplotlib.pyplot as plt

import common as C
from analysis import RC, SURFACE, INK, INK2, MUTED, WASH, REGION_COLOR, finish

WINDOWS = [(100, 20), (80, 25), (63.2, 31.6)]  # (길이, 폭) m, 모두 2천㎡
R_CORE = np.sqrt(2000 / np.pi)
R_DIR = 60.0
THRESH = {"법정(30개)": 30, "완화(15개)": 15}
LINK, ATTACH = 50.0, 30.0


def window_counts(xy):
    """각 업소의 원형 창 수와 길 따라 직사각형 창 수(최댓값)."""
    tree = cKDTree(xy)
    rmax = max(np.hypot(L / 2, W / 2) for L, W in WINDOWS)
    nb_dir = tree.query_ball_point(xy, R_DIR)
    nb_win = tree.query_ball_point(xy, rmax)
    n_circle = np.array([len(v) for v in tree.query_ball_point(xy, R_CORE)])
    n_rect = np.zeros(len(xy), dtype=int)
    angle = np.full(len(xy), np.nan)
    for i in range(len(xy)):
        idx = nb_dir[i]
        if len(idx) >= 3:
            rel = xy[idx] - xy[i]
            cov = np.cov(rel.T) if len(idx) > 2 else np.eye(2)
            w, v = np.linalg.eigh(cov)
            u = v[:, np.argmax(w)]
        else:
            u = np.array([1.0, 0.0])
        angle[i] = np.degrees(np.arctan2(u[1], u[0]))
        rel = xy[nb_win[i]] - xy[i]
        along = rel @ u
        across = rel @ np.array([-u[1], u[0]])
        best = 0
        for L, W in WINDOWS:
            best = max(best, int(((np.abs(along) <= L / 2) & (np.abs(across) <= W / 2)).sum()))
        n_rect[i] = best
    return n_circle, n_rect, angle


def cluster(xy, core_mask):
    core = np.where(core_mask)[0]
    cl = np.full(len(xy), -1)
    if len(core) == 0:
        return cl
    ctree = cKDTree(xy[core])
    pairs = ctree.query_pairs(LINK, output_type="ndarray")
    n = len(core)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(adj, directed=False)
    dist, idx = ctree.query(xy, k=1)
    cl = np.where(dist <= ATTACH, lab[idx], -1)
    cl[core] = lab
    return cl


def main():
    st = C.load("stores_clean")
    raw = pd.concat([pd.read_csv(C.ROOT / "data" / "external" / f, encoding="utf-8-sig", usecols=["상가업소번호", "건물관리번호", "도로명주소"], dtype=str)
                     for f in ["소상공인시장진흥공단_상가(상권)정보_서울_202606.csv", "소상공인시장진흥공단_상가(상권)정보_강원_202606.csv"]])
    st = st.merge(raw, on="상가업소번호", how="left")
    sti = pd.read_csv(C.TAB / "st_industry.csv", encoding="utf-8-sig")[["region", "industry", "rain_loss_day_per_store_만원", "grade_V"]]
    st = st.merge(sti, on=["region", "industry"], how="left")
    rows, members, compare = [], [], []
    for region in C.REGIONS:
        d = st[st.region == region].reset_index(drop=True)
        xy = d[["x", "y"]].values
        n_c, n_r, ang = window_counts(xy)
        d["n_circle"], d["n_rect"], d["road_angle"] = n_c, n_r, np.round(ang, 1)
        print(f"{C.REGION_NAME[region]}: 업소 {len(d):,}개, 원형 30+ {int((n_c >= 30).sum()):,}개, 직사각형 30+ {int((n_r >= 30).sum()):,}개, 15+ 원형 {int((n_c >= 15).sum()):,} / 직사각형 {int((n_r >= 15).sum()):,}")
        for label, thr in THRESH.items():
            for method, n_arr in [("원형", n_c), ("길따라", n_r)]:
                cl = cluster(xy, n_arr >= thr)
                d[f"cl_{method}_{label}"] = cl
                for k in np.unique(cl[cl >= 0]):
                    g = d[cl == k]
                    top_b = g.건물관리번호.value_counts(dropna=True)
                    single = (top_b.iloc[0] / len(g) >= 0.5) if len(top_b) else False
                    mapped = g[g.industry.notna()]
                    rows.append({"region": region, "지역": C.REGION_NAME[region], "방법": method, "기준": label, "cluster": int(k), "행정동": g.emd_name.mode().iloc[0], "법정동": g.ldong_name.mode().iloc[0],
                                 "업소 수": len(g), "카드 업종 매핑 업소": len(mapped), "단일 건물 구역": single,
                                 "가로 m": round(g.x.max() - g.x.min() + 10), "세로 m": round(g.y.max() - g.y.min() + 10),
                                 "음식 업종 비중": round(float((g.big == "음식").mean()), 2), "취약 등급 업소": int(mapped.grade_V.isin(["취약", "매우 취약"]).sum()),
                                 "호우일 하루 손실(만원)": round(float(mapped["rain_loss_day_per_store_만원"].fillna(0).sum()), 1),
                                 "주요 업종(상위 3)": ", ".join(f"{i}({n})" for i, n in mapped.industry.value_counts().head(3).items()),
                                 "cx": round(float(g.x.mean())), "cy": round(float(g.y.mean())), "lon": round(float(g.lon.mean()), 5), "lat": round(float(g.lat.mean()), 5)})
        members.append(d[["상가업소번호", "region", "emd_name", "industry", "x", "y", "lon", "lat", "n_circle", "n_rect", "road_angle"] + [c for c in d.columns if c.startswith("cl_")]])
    A = pd.DataFrame(rows)
    A.to_csv(C.TAB / "al_clusters_rect.csv", index=False, encoding="utf-8-sig")
    M = pd.concat(members, ignore_index=True)
    M.to_csv(C.TAB / "al_cluster_stores_rect.csv", index=False, encoding="utf-8-sig")

    # ---------------- 검증 1: 춘천 지정 골목형상점가 5곳(주소 매칭 업소)이 구역에 들어오나
    cc = M[M.region == "chuncheon"].merge(raw[["상가업소번호", "도로명주소"]], on="상가업소번호", how="left")
    targets = {"강원대 후문(서부대성로 227)": ("서부대성로", 227), "한림대(성심로 4)": ("성심로", 4), "후평3동 행복마을(대양당길9번길)": ("대양당길", None), "뒤뜰(삭주로 110)": ("삭주로", 110), "강촌리(강촌로)": ("강촌로", None)}
    for name, (road, num) in targets.items():
        hit = cc[cc.도로명주소.fillna("").str.contains(re.escape(road))]
        if num is not None and len(hit):
            nums = hit.도로명주소.str.extract(re.escape(road) + r"\s+(\d+)")[0].astype(float)
            near = hit[nums.between(num - 40, num + 40)]
            hit = near if len(near) >= 3 else hit
        rec = {"대상": name, "주변 업소": len(hit)}
        for label in THRESH:
            for method in ["원형", "길따라"]:
                col = f"cl_{method}_{label}"
                inside = hit[hit[col] >= 0]
                rec[f"{method}·{label} 포함 비율(%)"] = round(len(inside) / max(len(hit), 1) * 100, 0)
        compare.append(rec)
    # ---------------- 검증 2: 강남 공식 골목상권 폴리곤 안 업소 포함 비율
    import importlib
    dm = importlib.import_module("96_district_map")
    attrs, geoms = dm.read_shapefile(C.EXT_SEOUL_SALES / "서울시 상권분석서비스(영역-상권).zip")
    gn = M[M.region == "gangnam"].reset_index(drop=True)
    pts = [Point(x, y) for x, y in zip(gn.x, gn.y)]
    tree = STRtree(pts)
    in_golmok = np.zeros(len(gn), dtype=bool)
    for a, gg in zip(attrs, geoms):
        if a["SIGNGU_CD"] == "11680" and a["TRDAR_SE_1"] == "골목상권":
            g79 = dm.poly_5181_to_5179(dm.rings_to_polygon(gg))
            for k in tree.query(g79):
                if g79.contains(pts[k]):
                    in_golmok[k] = True
    rec = {"대상": "강남 공식 골목상권 54곳 안 업소", "주변 업소": int(in_golmok.sum())}
    for label in THRESH:
        for method in ["원형", "길따라"]:
            rec[f"{method}·{label} 포함 비율(%)"] = round(float((gn.loc[in_golmok, f"cl_{method}_{label}"] >= 0).mean() * 100), 0)
    compare.append(rec)
    cmp = pd.DataFrame(compare)
    cmp.to_csv(C.TAB / "al_window_compare.csv", index=False, encoding="utf-8-sig")
    print("\n=== 검증: 지정·공식 상권 업소가 후보 구역에 들어오는 비율(%)"); print(cmp.to_string(index=False))
    summ = A[~A["단일 건물 구역"]].groupby(["지역", "방법", "기준"]).agg(구역=("cluster", "size"), 업소=("업소 수", "sum"), 손실=("호우일 하루 손실(만원)", "sum")).round(0)
    print("\n=== 구역 수·업소 수(단일 건물 제외)"); print(summ.to_string())

    # ---------------- 그림: 춘천 원형 vs 길따라(완화 기준)
    plt.rcParams.update(RC)
    cells = C.load("sk_cell")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    c = cells[(cells.region == "chuncheon") & (cells["pop"] > 0)]
    cx, cy, h = np.average(c.x, weights=c["pop"]), np.average(c.y, weights=c["pop"]), 3200
    d = M[M.region == "chuncheon"]
    for ax, method in zip(axes, ["원형", "길따라"]):
        col = f"cl_{method}_완화(15개)"
        a = A[(A.region == "chuncheon") & (A.방법 == method) & (A.기준 == "완화(15개)")]
        ax.scatter(d.x, d.y, s=0.6, color="#c3c2b7", linewidths=0, rasterized=True)
        inside = d[d[col] >= 0]
        ax.scatter(inside.x, inside.y, s=2.2, color=REGION_COLOR["chuncheon"], linewidths=0, rasterized=True, label=f"구역 안 업소 {len(inside):,}개")
        single = a[a["단일 건물 구역"]]
        ax.scatter(single.cx, single.cy, s=16, marker="s", color="#898781", label=f"단일 건물 구역 {len(single)}개")
        for _, r in a[~a["단일 건물 구역"]].sort_values("업소 수", ascending=False).head(6).iterrows():
            ax.annotate(f"{r.법정동} {r['업소 수']}개", (r.cx, r.cy), fontsize=8, color=INK, ha="left", va="bottom", xytext=(5, 4), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        for name, (road, num) in targets.items():
            hit = cc[cc.도로명주소.fillna("").str.contains(re.escape(road))]
            if num is not None and len(hit):
                nums = hit.도로명주소.str.extract(re.escape(road) + r"\s+(\d+)")[0].astype(float); near = hit[nums.between(num - 40, num + 40)]; hit = near if len(near) >= 3 else hit
            if len(hit) and abs(hit.x.mean() - cx) < h and abs(hit.y.mean() - cy) < h:
                ax.scatter([hit.x.mean()], [hit.y.mean()], s=70, marker="*", color="#c0392b", edgecolors=SURFACE, zorder=6)
                ax.annotate(name.split("(")[0], (hit.x.mean(), hit.y.mean()), fontsize=8, color="#c0392b", ha="left", va="top", xytext=(6, -4), textcoords="offset points")
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"춘천 밀집 후보({method} 창, 완화 15개): 구역 {int((~a['단일 건물 구역']).sum())}곳"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    finish(fig, C.FIG, "al03_alley_window", "같은 2천㎡를 원으로 셀 때와 길 방향 직사각형으로 셀 때(별: 지정 골목형상점가 대표점)")
    print("저장: al_clusters_rect.csv, al_cluster_stores_rect.csv, al_window_compare.csv, al03_alley_window.png")


if __name__ == "__main__":
    main()
