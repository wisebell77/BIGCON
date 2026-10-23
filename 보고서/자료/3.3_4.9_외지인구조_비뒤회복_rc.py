import pandas as pd, numpy as np, statsmodels.formula.api as smf, sys
S=sys.argv[1]; CARDF=sys.argv[2]
d=pd.read_csv(CARDF, sep='\t', encoding='cp949', dtype={'TA_YMD':str})
d=d[d.MCT_SGG_CD=='강원 춘천시']
d['date']=pd.to_datetime(d.TA_YMD); d['out']=d.CLN_SGG_CD!='강원'
m=pd.read_csv(f'{S}/team/industry_map.csv')
FACE=["음식점","카페·베이커리","주점·유흥","식품·생필품","패션·잡화","생활·기타소매","대형유통","숙박","여가·스포츠","개인서비스"]
d=d.merge(m[['industry','group','small_biz']],left_on='MCT_RY_CD',right_on='industry',how='left')
d=d[d.group.isin(FACE)]
BIG=['호텔/콘도','실내/실외골프장','종합레저타운/놀이동산','영화/공연']
d['seg']=np.where(d.MCT_RY_CD.isin(BIG),'big_tour',np.where(d.small_biz==1,'small','other_big'))
print('춘천 대면 업종 연결 안 된 업종 수', d.group.isna().sum())
# 1. monthly
mo=d.groupby([d.date.dt.month,'out']).TS_AT.sum().unstack()/1e8
mo.columns=['현지인','외지인']; print('\n[월별 매출 억, 신한카드 표본]\n', mo.round(1))
tv=pd.read_parquet(f'{S}/team/tour_monthly_visitors.parquet'); print('\n[관광데이터랩 월 방문자 만명]'); tv['ym']=tv.ym.astype(int); tv['m']=tv.ym%100; tv['y']=tv.ym//100
print(tv.pivot(index='m',columns='y',values='visitors').div(1e4).round(0))
# daily panel
w=pd.read_parquet(f'{S}/team/weather_daily.parquet'); w=w[w.region=='chuncheon'][['date','rain_biz10','rain_biz20','rn_biz']]
cal=pd.read_parquet(f'{S}/team/calendar.parquet')[['date','dow','is_offday','is_holiday','month','week','is_chuseok_block','coupon_phase']]
w['date']=pd.to_datetime(w.date); cal['date']=pd.to_datetime(cal.date)
def daily(mask,name):
    x=d[mask].groupby('date').TS_AT.sum().rename(name); return x
D=pd.concat([daily(d.out,'out_all'),daily(d.out&(d.seg=='small'),'out_small'),daily(d.out&(d.seg=='big_tour'),'out_big'),daily(~d.out,'loc_all'),daily(~d.out&(d.seg=='small'),'loc_small')],axis=1).reset_index()
D=D.merge(cal,on='date').merge(w,on='date',how='left').sort_values('date').reset_index(drop=True)
r=D.set_index('date').rain_biz10.fillna(0)
for a,b in [(1,7),(8,14),(15,28),(29,42)]:
    D[f'r_{a}_{b}']=[r[(r.index>=t-pd.Timedelta(days=b))&(r.index<=t-pd.Timedelta(days=a))].sum() for t in D.date]
D['lead1']=r.shift(-1).reindex(D.date).fillna(0).values
print('\n[비 온 뒤 1~6주 매출이 더 늘어나나] (계수 = 과거 구간 비 온 날 1일당 % 변화)')
for y in ['out_all','out_small','loc_all']:
    f=f'np.log({y}) ~ rain_biz10 + lead1 + r_1_7 + r_8_14 + r_15_28 + r_29_42 + C(dow)*is_offday + is_holiday + is_chuseok_block + C(coupon_phase) + C(month)'
    res=smf.ols(f,data=D.fillna({'coupon_phase':'none'})).fit(cov_type='HAC',cov_kwds={'maxlags':7})
    out={k:f"{(np.exp(res.params[k])-1)*100:+.1f}% (p={res.pvalues[k]:.2f})" for k in ['rain_biz10','lead1','r_1_7','r_8_14','r_15_28','r_29_42']}
    print(y, out)
# 누적: 비 1일 손실 vs 다음 6주 합 회복
# 3. co-movement big vs small (residual)
print('\n[대형 관광시설과 소상공인 외지인 매출이 같이 움직이나]')
for y in ['out_big','out_small','loc_small']:
    D['res_'+y]=smf.ols(f'np.log({y}) ~ C(dow)*is_offday + is_holiday + C(week)',data=D).fit().resid
nr=D.rain_biz10==0
print('같은 주·요일 효과 뺀 뒤 상관: 대형↔외지인 소상공인', D[['res_out_big','res_out_small']].corr().iloc[0,1].round(2),
      '| 비 안 온 날만', D.loc[nr,['res_out_big','res_out_small']].corr().iloc[0,1].round(2),
      '| 대형↔현지인 소상공인(비교)', D.loc[nr,['res_out_big','res_loc_small']].corr().iloc[0,1].round(2))
res=smf.ols('res_out_small ~ res_out_big',data=D[nr]).fit(); print('대형 외지인 매출 +10%일 때 소상공인 외지인 매출', f"{res.params['res_out_big']*10:+.1f}%")
sh=d[d.out].groupby('seg').TS_AT.sum(); print('외지인 매출 구성', (sh/sh.sum()*100).round(1).to_dict())
# 소상공인 업종별 외지인 비중
g=d.groupby(['MCT_RY_CD','out']).TS_AT.sum().unstack(); g['외지인비중']=g[True]/g.sum(axis=1)*100; g['외지인억']=g[True]/1e8
print(g.sort_values('외지인억',ascending=False).head(15)[['외지인억','외지인비중']].round(1))
# weekday vs weekend 외지인 per day, by month
D['wk']=np.where(D.is_offday==1,'휴일','평일')
print('\n[월×평일/휴일 하루 외지인 매출 억]\n', (D.groupby(['month','wk']).out_all.mean()/1e8).unstack().round(2))
# rain loss by month (외지인)
print('\n[달별 영업시간 10mm 비 일수]', D.groupby('month').rain_biz10.sum().to_dict())
