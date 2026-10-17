"""
1단계: 춘천 골목상권 구역 도출  (필요 데이터: data/ 폴더만 → 레포만 받아도 실행 가능)

근거: 전통시장법 시행령 제2조의2 단서 → 춘천시 전통시장 및 상점가 육성을 위한 조례 제24조(2026.4.23 개정)
      "소상공인이 운영하는 점포가 2,000㎡ 이내에 15개 이상 밀집한 구역"
적용: 지정은 상인이 구역 도면을 그려 신청(제25조) → 2,000㎡ 안에 15개가 들어가게 그릴 수 있는 골목
      구역 폭 20m(도로 + 양쪽 점포) × 길이 100m = 2,000㎡  →  골목 따라 100m 구간 안 점포 15개 이상

실행:  python step1_zones.py          (기본 폭 20m)
       python step1_zones.py 15       (민감도: 폭 15m → 133m 구간)
출력:  output/골목상권_구역.csv, output/골목상권_점포.csv
"""
import sys, numpy as np, pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

MIN_STORES = 15                                   # 조례 제24조
WIDTH = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
SPAN = 2000 / WIDTH                               # 폭 20m → 100m
LIFE = ['음식', '소매', '수리·개인', '숙박', '예술·스포츠']
DIRECT = r'스타벅스|올리브영|다이소|이마트(?!24)|홈플러스|롯데마트|하이마트|맥도날드|유니클로|씨지브이|CGV|메가박스|롯데시네마|아웃백|빕스|삼성디지털프라자'
MARKET = r'중앙시장|풍물시장|번개시장|제일시장|낭만시장'

d = pd.read_csv('data/춘천_상가정보_202606.csv', low_memory=False)
mp = pd.read_csv('data/업종매핑_상가소분류_카드업종.csv')
d['카드업종'] = d.상권업종소분류명.map(dict(zip(mp.상권업종소분류명, mp.카드업종)))
tr = Transformer.from_crs('EPSG:4326', 'EPSG:5179', always_xy=True)   # 위경도 → 미터 좌표
d['X'], d['Y'] = tr.transform(d.경도.values, d.위도.values)
log = [('춘천 전체 점포', len(d))]

# ① 생활업종
d = d[d.상권업종대분류명.isin(LIFE)].reset_index(drop=True); log.append(('① 생활업종', len(d)))
# ② 전통시장 반경 100m 제외 (건물명에 '시장' 또는 시장 상호)
mk = d[d.건물명.fillna('').str.contains('시장') | d.상호명.str.contains(MARKET, na=False)]
cent = mk.groupby('건물명')[['X', 'Y']].median().dropna()
dist, _ = BallTree(cent.values).query(d[['X', 'Y']].values, k=1)
d = d[~((dist[:, 0] <= 100) | d.index.isin(mk.index))]; log.append(('② 전통시장 100m 제외', len(d)))
# ③ 대기업 직영 추정 제외 (조례: 소상공인이 운영하는 점포)
d = d[~d.상호명.fillna('').str.contains(DIRECT)]; log.append(('③ 대기업 직영 제외', len(d)))
# ④ 도로명 '~길'만 (대로·로 = 대로변 제외)
d['길'] = d.도로명.astype(str).str.split().str[-1]
d = d[d.길.str.endswith('길')]; log.append(("④ 도로명 '~길'", len(d)))

# ⑤ 같은 길에서 50m 이내로 이어진 점포를 한 구역으로
def max_in_span(z, L):
    """골목 방향으로 점포를 한 줄로 세우고, 길이 L 자를 밀면서 안에 들어오는 최대 점포 수"""
    P = z[['X', 'Y']].values - z[['X', 'Y']].values.mean(0)
    axis = np.linalg.svd(P, full_matrices=False)[2][0]
    t = np.sort(P @ axis)
    return int(max(np.searchsorted(t, t + L, side='right') - np.arange(len(t))))

zones, zid = [], 0
for road, g in d.groupby('길'):
    if len(g) < MIN_STORES: continue
    lab = DBSCAN(eps=50, min_samples=3).fit_predict(g[['X', 'Y']].values)
    for k in set(lab) - {-1}:
        z = g[lab == k]
        if len(z) >= MIN_STORES:
            zid += 1; zones.append(z.assign(구역=zid))
S = pd.concat(zones); log.append(('⑤ 50m 연속 15개 이상', f'{S.구역.nunique()}구역 / {len(S)}점포'))
# ⑥ 건물 3동 이상 (아파트·복합 상가건물 1동에 몰린 곳 제외)
nb = S.groupby('구역').건물관리번호.nunique()
S = S[S.구역.isin(nb[nb >= 3].index)]; log.append(('⑥ 건물 3동 이상', f'{S.구역.nunique()}구역 / {len(S)}점포'))
# ⑦ 골목 따라 SPAN m 구간 안 15개 이상 (조례 2,000㎡)
span = {k: max_in_span(z, SPAN) for k, z in S.groupby('구역')}
S = S[S.구역.map(span) >= MIN_STORES]; log.append((f'⑦ {SPAN:.0f}m 구간 15개 이상', f'{S.구역.nunique()}구역 / {len(S)}점포'))

Z = S.groupby('구역').agg(길=('길', lambda s: s.mode()[0]), 행정동=('행정동명', lambda s: s.mode()[0]),
        점포수=('상호명', 'size'), 건물수=('건물관리번호', 'nunique'),
        경도=('경도', 'median'), 위도=('위도', 'median')).reset_index()
Z[f'{SPAN:.0f}m구간 최대 점포'] = Z.구역.map(span)
Z['구간길이_m'] = Z.구역.map({k: round(float(np.hypot(np.ptp(z.X), np.ptp(z.Y)))) for k, z in S.groupby('구역')})
Z.to_csv('output/골목상권_구역.csv', index=False, encoding='utf-8-sig')
S[['구역', '상가업소번호', '상호명', '상권업종소분류명', '카드업종', '길', '행정동명', '건물관리번호', '경도', '위도', 'X', 'Y']] \
    .to_csv('output/골목상권_점포.csv', index=False, encoding='utf-8-sig')
for a, b in log: print(f'{a:<22} {b}')
