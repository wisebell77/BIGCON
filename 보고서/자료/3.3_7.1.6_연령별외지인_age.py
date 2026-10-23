import pandas as pd, numpy as np, statsmodels.formula.api as smf, sys
S=sys.argv[1]; CARDF=sys.argv[2]
d=pd.read_csv(CARDF, sep='\t', encoding='cp949', dtype={'TA_YMD':str})
d=d[d.MCT_SGG_CD=='강원 춘천시']; d['date']=pd.to_datetime(d.TA_YMD)
m=pd.read_csv(f'{S}/team/industry_map.csv'); d=d.merge(m[['industry','group','small_biz']],left_on='MCT_RY_CD',right_on='industry',how='left')
FACE=["음식점","카페·베이커리","주점·유흥","식품·생필품","패션·잡화","생활·기타소매","대형유통","숙박","여가·스포츠","개인서비스"]
d=d[d.group.isin(FACE)]
print(d.AGE_CCD.unique())
age=d.AGE_CCD.str.extract(r'(\d+)')[0].astype(float)
d['ag']=pd.cut(age,[0,29,39,49,59,200],labels=['20대이하','30대','40대','50대','60대+'])
d['who']=np.where(d.CLN_SGG_CD!='강원','외지인','현지인')
cal=pd.read_parquet(f'{S}/team/calendar.parquet')[['date','dow','is_offday','is_holiday','week','is_chuseok_block','coupon_phase']]; cal['date']=pd.to_datetime(cal.date)
w=pd.read_parquet(f'{S}/team/weather_daily.parquet'); w=w[w.region=='chuncheon'][['date','rain_biz10','heat_day','tropical']]; w['date']=pd.to_datetime(w.date)
o=d[d.who=='외지인']
tot=o.groupby('ag',observed=True).agg(매출=('TS_AT','sum'),건수=('USE_CNT','sum'))
tot['매출비중%']=tot.매출/tot.매출.sum()*100; tot['건당금액(원)']=tot.매출/tot.건수
# 평일 비중
x=o.merge(cal,on='date'); tot['평일매출비중%']=x[x.is_offday==0].groupby('ag',observed=True).TS_AT.sum()/x.groupby('ag',observed=True).TS_AT.sum()*100
small=x[x.small_biz==1].groupby('ag',observed=True).TS_AT.sum()/x.groupby('ag',observed=True).TS_AT.sum()*100; tot['소상공인비중%']=small
# 비 효과 by age (외지인)
eff={}; heat={}
for a in tot.index:
    y=x[x.ag==a].groupby('date').TS_AT.sum().rename('y').reset_index().merge(cal,on='date').merge(w,on='date')
    r=smf.ols('np.log(y) ~ rain_biz10 + heat_day + C(dow)*is_offday + is_holiday + is_chuseok_block + C(week)',data=y.fillna({'coupon_phase':'none'})).fit(cov_type='HC1')
    eff[a]=f"{(np.exp(r.params['rain_biz10'])-1)*100:+.1f}% (p={r.pvalues['rain_biz10']:.2f})"; heat[a]=f"{(np.exp(r.params['heat_day'])-1)*100:+.1f}% (p={r.pvalues['heat_day']:.2f})"
tot['비오는날']=pd.Series(eff); tot['폭염일']=pd.Series(heat)
pd.set_option('display.width',250); print(tot.drop(columns=['매출','건수']).round(1))
# 평일 하루 vs 휴일 하루 비율
z=x.groupby(['ag','is_offday'],observed=True).TS_AT.sum().unstack(); nd=cal.groupby('is_offday').size()
print('\n휴일 하루 / 평일 하루 매출 배수', (z[1]/nd[1])/(z[0]/nd[0]).round(2))
# 업종별 외지인 연령 구성 (상위 업종)
top=o.groupby(['MCT_RY_CD','ag'],observed=True).TS_AT.sum().unstack(); top=top[top.sum(1)>top.sum(1).quantile(.8)]
print((top.div(top.sum(1),axis=0)*100).round(0).sort_values('60대+',ascending=False).head(12))
