# -*- coding: utf-8 -*-
"""95c. 골목 상권 묶음 규칙(핵 연결 거리 LINK, 주변 업소 부착 거리 ATTACH)의 민감도: 춘천 완화 15개·길 따라 창 기준.
al_cluster_stores_rect.csv의 n_rect(창 안 업소 수)를 그대로 쓰고 묶음만 다시 한다."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common as C

M = pd.read_csv(C.TAB / "al_cluster_stores_rect.csv", encoding="utf-8-sig", dtype={"상가업소번호": str})
raw = pd.read_csv(ROOT / "data" / "external" / "소상공인시장진흥공단_상가(상권)정보_강원_202606.csv", encoding="utf-8-sig", usecols=["상가업소번호", "건물관리번호", "도로명주소"], dtype=str)
d = M[M.region == "chuncheon"].merge(raw, on="상가업소번호", how="left").reset_index(drop=True)
xy = d[["x", "y"]].values
core_mask = d.n_rect.values >= 15
targets = {"강원대 후문": ("서부대성로", 227), "뒤뜰(삭주로)": ("삭주로", 110), "한림대": ("성심로", 4), "후평3동 행복마을": ("대양당길", None)}


def cluster(xy, core_mask, link, attach):
    core = np.where(core_mask)[0]
    ctree = cKDTree(xy[core])
    pairs = ctree.query_pairs(link, output_type="ndarray")
    n = len(core)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(adj, directed=False)
    dist, idx = ctree.query(xy, k=1)
    cl = np.where(dist <= attach, lab[idx], -1)
    cl[core] = lab
    return cl


def hit_mask(road, num):
    m = d.도로명주소.fillna("").str.contains(re.escape(road))
    if num is not None and m.sum():
        nums = d.loc[m, "도로명주소"].str.extract(re.escape(road) + r"\s+(\d+)")[0].astype(float)
        near = nums.between(num - 40, num + 40)
        if near.sum() >= 3:
            mm = pd.Series(False, index=d.index); mm.loc[near[near].index] = True
            return mm.values
    return m.values


rows = []
for link in [30, 50, 70, 100]:
    for attach in [20, 30, 40]:
        cl = cluster(xy, core_mask, link, attach)
        ids, cnt = np.unique(cl[cl >= 0], return_counts=True)
        # 단일 건물 구역 제외
        multi = 0; stores_multi = 0
        for k, c in zip(ids, cnt):
            g = d[cl == k]
            top = g.건물관리번호.value_counts(dropna=True)
            single = (top.iloc[0] / len(g) >= 0.5) if len(top) else False
            if not single and len(g) >= 3:
                multi += 1; stores_multi += len(g)
        rec = {"핵 연결 거리(m)": link, "부착 거리(m)": attach, "구역 수(단일 건물 제외)": multi, "업소 수": stores_multi, "최대 구역 업소 수": int(cnt.max()) if len(cnt) else 0}
        for name, (road, num) in targets.items():
            hm = hit_mask(road, num)
            rec[f"{name} 포함률(%)"] = round(float((cl[hm] >= 0).mean() * 100), 0) if hm.sum() else np.nan
        rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(C.TAB / "al_link_sensitivity.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(out.to_string(index=False))
