"""82개 구역(팀 95b_alley_window.py, 길 따라 창·완화 15개) 기준으로 관광지 거리 다시 계산 (보고서 6.3용)."""
import pandas as pd, numpy as np
from pathlib import Path
H = Path(__file__).parent
src = open(H/'6.3_참고_골목41_관광지지도_생성.py', encoding='utf-8').read()
exec(src[src.index('A = pd.DataFrame'):src.index("c['가까운 관광지']")].replace('Dm = ', '_ = 0 and '))
c = pd.read_csv(H/'6.3_입력_골목구역_al_clusters_rect.csv')
c = c[(c.region=='chuncheon') & (c['방법']=='길따라') & c['기준'].str.startswith('완화') & (~c['단일 건물 구역'].astype(bool))].copy()
D = np.array([[dist_m(r.lon, r.lat, a.lon, a.lat) for a in A.itertuples()] for r in c.itertuples()])
c['가까운 관광지'] = A.name.values[D.argmin(1)]
c['관광지 거리 m'] = D.min(1).round(-1).astype(int)
c['구분'] = np.select([c['관광지 거리 m']<=1000, c['관광지 거리 m']<=2000], ['1km 안','1~2km'], '2km 밖')
c.sort_values('관광지 거리 m').to_csv(H/'6.3_골목82_관광지거리.csv', index=False, encoding='utf-8-sig')
print('구역 수', len(c))
print(c.groupby('구분').agg(구역=('업소 수','size'), 업소=('업소 수','sum'), 하루손실만원=('호우일 하루 손실(만원)','sum')).round(0))
print(c[c['구분']!='2km 밖'].sort_values('업소 수',ascending=False)[['법정동','행정동','업소 수','가까운 관광지','관광지 거리 m','호우일 하루 손실(만원)']].head(12).to_string(index=False))
print(c.groupby('가까운 관광지')['관광지 거리 m'].min().sort_values())

# 정적 그림 (보고서용)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='AppleGothic'; plt.rcParams['axes.unicode_minus']=False
col = {'1km 안':'#d1495b','1~2km':'#edae49','2km 밖':'#8d99ae'}
fig, ax = plt.subplots(figsize=(11,10))
for k,v in col.items():
    t=c[c['구분']==k]; ax.scatter(t.lon,t.lat,s=t['업소 수']*0.8+15,c=v,alpha=.8,edgecolor='k',lw=.5,label=f'관광지 {k} ({len(t)}곳, 업소 {t["업소 수"].sum():,}개)')
ax.scatter(A.lon,A.lat,marker='*',s=260,c='#2e86ab',edgecolor='k',zorder=5,label='주요 관광지(대략 위치)')
for a in A.itertuples(): ax.annotate(a.name,(a.lon,a.lat),xytext=(5,5),textcoords='offset points',fontsize=9,color='#1b4965')
for r in c[c['업소 수']>=90].itertuples(): ax.annotate(f'{r.법정동} {r._8}개',(r.lon,r.lat),xytext=(6,-12),textcoords='offset points',fontsize=8.5)
ax.set_xlim(127.62,127.81); ax.set_ylim(37.80,37.95); ax.set_aspect(1/np.cos(37.88*np.pi/180))
ax.set_title('춘천 골목상권 82개 구역과 주요 관광지까지의 거리', fontsize=13)
ax.legend(loc='lower right',fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.text(.01,.01,'골목상권: 팀 95b_alley_window.py(길 따라 창, 완화 15개, 상가정보 2026.06). 관광지 위치는 인접 가게 좌표로 잡은 대략값.',fontsize=8,color='gray')
fig.savefig(H/'6.3_골목82_관광지거리.png',dpi=150,bbox_inches='tight')
