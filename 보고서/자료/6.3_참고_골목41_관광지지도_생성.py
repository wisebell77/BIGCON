"""춘천 골목상권 후보(팀 95_alley.py 결과) x 주요 관광지 거리 -> 레인체크 사용처 후보 지도."""
import pandas as pd, numpy as np, json
from pathlib import Path
H = Path(__file__).parent
c = pd.read_csv(H/'al_clusters.csv')
c = c[(c.region=='chuncheon') & (c['기준'].str.startswith('완화')) & (~c['단일 건물 구역'])].copy()
# 관광지 위치: 상가정보에서 관광지 바로 옆 가게 좌표로 잡은 대략 위치(proxy). 레고랜드는 상가정보에 없어 대략값.
A = pd.DataFrame([
 ('레고랜드(하중도)', 127.6945, 37.8865, '대략값(지도 확인 필요)', '테마파크'),
 ('삼악산 호수케이블카', 127.6914, 37.8644, '인디커피삼악산케이블카지점', '체험'),
 ('소양강스카이워크', 127.7252, 37.8929, '춘천스카이워크닭갈비막국수', '관광지'),
 ('공지천 유원지', 127.7077, 37.8726, '호텔공지천·카페공지천', '관광지'),
 ('의암호 스카이워크', 127.6845, 37.8459, '커피홀의암스카이워크점', '관광지'),
 ('구봉산 전망대 카페거리', 127.7772, 37.8934, '이디야구봉산전망대점', '관광지'),
 ('김유정역·김유정레일바이크', 127.7152, 37.8179, 'GS25김유정역점', '체험'),
 ('강촌(강촌역·레일바이크)', 127.6341, 37.8133, '지에스25강촌중앙', '체험'),
 ('엘리시안 강촌', 127.5522, 37.8409, '엘리시안강촌펀스키', '리조트·골프'),
 ('소양강댐', 127.8015, 37.9372, '소양강댐시민의숲', '관광지'),
], columns=['name','lon','lat','근거','종류'])
def dist_m(lon1, lat1, lon2, lat2):
    R=6371000; p=np.pi/180
    a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
Dm = np.array([[dist_m(r.lon, r.lat, a.lon, a.lat) for a in A.itertuples()] for r in c.itertuples()])
c['가까운 관광지'] = A.name.values[Dm.argmin(1)]
c['관광지 거리 m'] = Dm.min(1).round(-1).astype(int)
c['구분'] = np.select([c['관광지 거리 m']<=1000, c['관광지 거리 m']<=2000], ['1km 안(걸어서)','1~2km(차로 5분)'], '2km 밖(생활 상권)')
cols = ['법정동','행정동','업소 수','음식 업종 비중','주요 업종(상위 3)','취약 등급 업소','호우일 하루 손실(만원)','가까운 관광지','관광지 거리 m','구분','lon','lat']
T = c.sort_values(['관광지 거리 m'])[cols]
T.to_csv(H/'6.3_참고_골목41_관광지거리.csv', index=False, encoding='utf-8-sig')
print(T.drop(columns=['lon','lat']).to_string(index=False))
print('\n구분별', T.groupby('구분').agg(구역=('업소 수','size'), 업소=('업소 수','sum'), 하루손실만원=('호우일 하루 손실(만원)','sum')))
# 업소 점(구역 소속)
s = pd.read_csv(H/'al_cluster_stores.csv')
s = s[(s.region=='chuncheon') & (s['cluster_완화(15개)']>=0)]
cid = dict(zip(c.cluster, c['구분']))
s['구분'] = s['cluster_완화(15개)'].map(cid)
s = s.dropna(subset=['구분'])
data = {
 'clusters': T.assign(**{'음식 업종 비중': (T['음식 업종 비중']*100).round()}).to_dict('records'),
 'attractions': A.to_dict('records'),
 'stores': s[['lon','lat','구분','상호명','industry']].round(5).fillna('').values.tolist(),
}
html = (H/'6.3_참고_골목41_관광지지도_템플릿.html').read_text(encoding='utf-8').replace('__DATA__', json.dumps(data, ensure_ascii=False))
(H/'6.3_참고_골목41_관광지지도.html').write_text(html, encoding='utf-8')
# 정적 그림
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='AppleGothic'; plt.rcParams['axes.unicode_minus']=False
col = {'1km 안(걸어서)':'#d1495b','1~2km(차로 5분)':'#edae49','2km 밖(생활 상권)':'#8d99ae'}
fig, ax = plt.subplots(figsize=(11,10))
for k,v in col.items():
    t=T[T['구분']==k]; ax.scatter(t.lon,t.lat,s=t['업소 수']*1.2+15,c=v,alpha=.8,edgecolor='k',lw=.5,label=f'{k} ({len(t)}곳, {t["업소 수"].sum():,}개 업소)')
ax.scatter(A.lon,A.lat,marker='*',s=260,c='#2e86ab',edgecolor='k',zorder=5,label='주요 관광지(대략 위치)')
for a in A.itertuples(): ax.annotate(a.name,(a.lon,a.lat),xytext=(5,5),textcoords='offset points',fontsize=9,color='#1b4965')
for r in T[T['업소 수']>=90].itertuples(): ax.annotate(f'{r.법정동} {r._3}개',(r.lon,r.lat),xytext=(6,-12),textcoords='offset points',fontsize=8.5)
ax.set_xlim(127.62,127.81); ax.set_ylim(37.80,37.95); ax.set_aspect(1/np.cos(37.88*np.pi/180))
ax.set_title('춘천 골목상권 후보(완화 15개 기준) × 주요 관광지 거리\n레인체크 쿠폰 사용처 우선 후보 = 관광지 1~2km 안 구역', fontsize=13)
ax.legend(loc='lower right',fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.text(.01,.01,'골목상권 구역: 팀 analysis 브랜치 95_alley.py(상가정보 2026.06). 관광지 위치는 바로 옆 가게 좌표로 잡은 대략값(레고랜드는 지도 확인 필요). 엘리시안 강촌·소양강댐은 범위 밖.',fontsize=8,color='gray')
fig.savefig(H/'6.3_참고_골목41_관광지지도.png',dpi=150,bbox_inches='tight')
