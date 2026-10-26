import pandas as pd, numpy as np, glob
from scipy import stats
fs=sorted(glob.glob('raw/inner/**/*.csv',recursive=True))
d=pd.concat([pd.read_csv(f,encoding='cp949') for f in fs],ignore_index=True)
d=d.rename(columns={'일시':'date','일강수량(mm)':'R','1시간 최다강수량(mm)':'R1h','최고기온(°C)':'Tmax'})
d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').sort_values('date'); d['yr']=d.date.dt.year
d['R']=d.R.fillna(0)  # 일강수량 공란 = 무강수
cnt=d.groupby('yr').size()
print('연도 수',cnt.size, cnt.min(), cnt.idxmin(), '| 2025 일수',cnt.get(2025))
print('1시간최다 첫 유효연도', d[d.R1h.notna()].yr.min())
a=d.groupby('yr').agg(연강수량=('R','sum'),강수일수_1mm=('R',lambda s:(s>=1).sum()),
    R10mm=('R',lambda s:(s>=10).sum()),R20mm=('R',lambda s:(s>=20).sum()),R50mm=('R',lambda s:(s>=50).sum()),
    호우일수_80mm=('R',lambda s:(s>=80).sum()),Rx1day=('R','max'),R1h_max=('R1h','max'),
    R1h_30mm일수=('R1h',lambda s:(s>=30).sum()))
a['강수강도_SDII']=d[d.R>=1].groupby('yr').R.mean()
a['R20mm_강수비중(%)']=d[d.R>=20].groupby('yr').R.sum()/a.연강수량*100
a.to_pickle('annual.pkl')
rows=[]
for c in a.columns:
    s=a[c].dropna(); x=s.index.values
    sl=stats.linregress(x,s.values); tau=stats.kendalltau(x,s.values)
    p1=s[(s.index>=1966)&(s.index<=1995)].mean(); p2=s[(s.index>=1996)&(s.index<=2025)].mean()
    rows.append({'지표':c,'기간':f'{x.min()}-{x.max()}','10년당 변화':round(sl.slope*10,2),'회귀 p':round(sl.pvalue,3),'MK p':round(tau.pvalue,3),
                 '전반 30년 평균':round(p1,1),'후반 30년 평균':round(p2,1),'변화율(%)':round((p2/p1-1)*100,1) if p1 else None})
T=pd.DataFrame(rows); T.to_pickle('trend.pkl'); pd.set_option('display.width',200); print(T.to_string(index=False))
# 최근 10년
print(a.loc[2016:2025,['R20mm','R50mm','강수강도_SDII','Rx1day']])
