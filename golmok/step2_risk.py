"""
2단계: 골목 구역별 우천 위험 지수  (필요: 1단계 출력 + private/ 폴더의 빅콘 기반 표 2개 → 레포에 올리지 않음)
  private/춘천_업종별_우천효과.csv   : 신한카드 춘천, 업종별 '영업시간 비 10mm 이상' 날 매출 변화(%), q값, 1일 피해액, 업종군
  private/춘천_SKT_격자_주말평일.csv  : SKT 50m 격자 평일·주말 평균 유동인구 (EPSG:5179 좌표)

위험 지수 = 백분위 평균( ① 예상 우천 손실  ② 비 취약 업종(A·B군) 비중  ③ 주말/평일 유동인구 비 )
① 예상 우천 손실 (추정값, 실제 구역 매출 아님)
   업종 1일 피해액(춘천 전체) ÷ 춘천 그 업종 점포 수 = 점포당 평균 피해 → 구역 점포만큼 합산
   감소가 통계적으로 확실한 업종만 (q<0.1, 효과<0), 나머지는 0
"""
import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.neighbors import BallTree

Z = pd.read_csv('output/골목상권_구역.csv'); S = pd.read_csv('output/골목상권_점포.csv')
I = pd.read_csv('private/춘천_업종별_우천효과.csv').set_index('업종')
F = pd.read_csv('private/춘천_SKT_격자_주말평일.csv')
# 춘천 전체 생활업종 점포 수 (점포당 피해 나눌 때 분모)
allst = pd.read_csv('data/춘천_상가정보_202606.csv', low_memory=False)
mp = pd.read_csv('data/업종매핑_상가소분류_카드업종.csv')
allst = allst[allst.상권업종대분류명.isin(['음식', '소매', '수리·개인', '숙박', '예술·스포츠'])]
n_all = allst.상권업종소분류명.map(dict(zip(mp.상권업종소분류명, mp.카드업종))).value_counts()

ok = (I.q < 0.1) & (I['효과(%)'] < 0)
I['점포당피해_만원'] = (-I['1일피해(만원)']).where(ok, 0) / I.index.map(n_all).fillna(np.inf)
S['점포당피해_만원'] = S.카드업종.map(I.점포당피해_만원).fillna(0)
S['업종군'] = S.카드업종.map(I.업종군)

bt = BallTree(F[['X_COORD', 'Y_COORD']].values)
rows = []
for k, z in S.groupby('구역'):
    f = F.iloc[np.unique(np.concatenate(bt.query_radius(z[['X', 'Y']].values, r=50)))]
    m = z[z.업종군.notna()]
    rows.append({'구역': k, '업종매칭점포': len(m), '취약업종(A·B)점포': int(m.업종군.isin(['A 고민감', 'B 중민감']).sum()),
                 '예상우천손실_만원/일': z.점포당피해_만원.sum(), '주말/평일 유동비': f.주말.sum() / max(f.평일.sum(), 1e-9)})
R = Z.merge(pd.DataFrame(rows), on='구역')
R['취약비중(%)'] = R['취약업종(A·B)점포'] / R.업종매칭점포.clip(lower=1) * 100
pct = lambda x: x.rank(pct=True) * 100
R['위험지수'] = (pct(R['예상우천손실_만원/일']) + pct(R['취약비중(%)']) + pct(R['주말/평일 유동비'])) / 3
R['유형'] = np.where(R['주말/평일 유동비'] > 1.1, '나들이·외지형', '주민 생활형')
R = R.sort_values('위험지수', ascending=False).reset_index(drop=True); R.index += 1

# 손실 계산 내역 (구역 × 업종)
B = S.groupby(['구역', '카드업종']).agg(점포수=('상호명', 'size'), 점포당피해_만원=('점포당피해_만원', 'first'),
                                    합계_만원=('점포당피해_만원', 'sum')).reset_index()
B['춘천전체_업종점포'] = B.카드업종.map(n_all); B['업종_1일피해_만원'] = B.카드업종.map(I['1일피해(만원)'])
B['업종_매출변화(%)'] = B.카드업종.map(I['효과(%)']).round(1)
B = B.merge(R.reset_index()[['index', '구역', '길']].rename(columns={'index': '순위'}), on='구역').sort_values(['순위', '합계_만원'], ascending=[True, False])

with pd.ExcelWriter('output/골목상권_위험지수.xlsx') as w:
    R.round(2).to_excel(w, sheet_name='구역별 위험지수', index_label='순위')
    B[['순위', '길', '카드업종', '점포수', '업종_매출변화(%)', '업종_1일피해_만원', '춘천전체_업종점포', '점포당피해_만원', '합계_만원']].round(1) \
        .to_excel(w, sheet_name='손실 계산 내역', index=False)

# 앱용 데이터
rows = [{'rank': int(i), 'road': r.길, 'dong': r.행정동, 'type': r.유형, 'stores': int(r.점포수),
         'lossPerRainDay_manwon': round(float(r['예상우천손실_만원/일']), 1), 'vulnPct': round(float(r['취약비중(%)']), 1),
         'weekendRatio': round(float(r['주말/평일 유동비']), 2), 'risk': round(float(r.위험지수), 1),
         'lat': round(float(r.위도), 6), 'lng': round(float(r.경도), 6)} for i, r in R.iterrows()]
D = R.groupby('행정동').agg(n=('길', 'size'), stores=('점포수', 'sum'), loss=('예상우천손실_만원/일', 'sum')).sort_values('loss', ascending=False)
D['share'] = D.loss / D.loss.sum() * 100
dong = [{'name': k, 'txt': f"골목 {v.n}곳 · 점포 {v.stores}개 · 비 오는 날 하루 예상 손실 {v.loss:,.0f}만 원 ({v.share:.0f}%)",
         'w': round(v.loss / D.loss.max() * 100)} for k, v in D.head(5).iterrows()]
with open('output/golmok_data.js', 'w') as f:
    f.write('// 춘천 골목상권 (조례 제24조 15개 기준) 구역별 우천 위험 — 손실은 업종별 감소율로 추정한 값\n')
    f.write('const GOLMOK = ' + json.dumps(rows, ensure_ascii=False) + ';\n')
    f.write('const GOLMOK_DONG = ' + json.dumps(dong, ensure_ascii=False, indent=1) + ';\n')

# 지도
have = {f.name for f in font_manager.fontManager.ttflist}
mpl.rcParams['font.family'] = next((x for x in ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'AppleGothic', 'Malgun Gothic'] if x in have), 'sans-serif')
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.4), gridspec_kw={'width_ratios': [1.3, 1]})
ax.scatter(allst.경도, allst.위도, s=1.5, color='#d9d9d9')
cm = mpl.colormaps['OrRd']
for k, z in S.groupby('구역'):
    ax.scatter(z.경도, z.위도, s=4, color=cm(0.25 + 0.75 * R.loc[R.구역 == k, '위험지수'].iloc[0] / 100))
for rank, r in R.head(10).iterrows():
    ax.annotate(str(rank), (r.경도, r.위도), xytext=(9, 8), textcoords='offset points', fontsize=8.5, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#222', lw=0.7), arrowprops=dict(arrowstyle='-', color='#222', lw=0.5))
ax.set_xlim(127.69, 127.79); ax.set_ylim(37.84, 37.91); ax.set_aspect(1.26); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title(f'춘천 골목상권 {len(R)}개 구역 · 색 = 우천 위험 지수', loc='left', fontweight='bold')
T = R.head(15); y = np.arange(len(T))[::-1]
ax2.barh(y, T.위험지수, color=[cm(0.25 + 0.75 * v / 100) for v in T.위험지수])
ax2.set_yticks(y, [f'{i}. {r.길} ({r.행정동})' for i, r in T.iterrows()], fontsize=8.5)
for yi, (_, r) in zip(y, T.iterrows()):
    ax2.text(r.위험지수 + 1, yi, f"{r.점포수}개 · 예상손실 {r['예상우천손실_만원/일']:.0f}만원/일 · 취약 {r['취약비중(%)']:.0f}%", va='center', fontsize=7.3, color='#444')
ax2.set_xlim(0, 160); ax2.set_xlabel('위험 지수 (0~100)'); ax2.set_title('위험 상위 15개 골목', loc='left', fontweight='bold')
for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)
fig.text(0.01, 0.01, "골목상권 = 춘천시 조례 제24조(2026.4.23) 소상공인 점포 2,000㎡ 이내 15개 → 골목 따라 100m(폭 20m) 구간 안 15개 이상\n"
         "위험 지수 = 백분위 평균(예상 우천 손실[업종별 감소율×점포 구성 추정], 비 취약 업종 비중, 주말/평일 유동인구 비) · 자료: 소진공 상가정보, 신한카드, SKT", fontsize=7.3, color='#777')
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig('output/골목상권_위험지수_지도.png', dpi=160, bbox_inches='tight')
print(R[['길', '행정동', '유형', '점포수', '예상우천손실_만원/일', '취약비중(%)', '주말/평일 유동비', '위험지수']].round(1).head(10).to_string())
print(R.유형.value_counts().to_dict())
