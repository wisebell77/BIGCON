import pandas as pd, numpy as np, sys
S=sys.argv[1]
d=pd.read_csv(sys.argv[2], sep='\t', encoding='cp949', dtype={'TA_YMD':str})
d=d[(d.MCT_SGG_CD=='강원 춘천시')&(d.CLN_SGG_CD!='강원')]; d['date']=pd.to_datetime(d.TA_YMD)
cal=pd.read_parquet(f'{S}/team/calendar.parquet')[['date','is_offday','month']]; cal['date']=pd.to_datetime(cal.date)
d=d.merge(cal,on='date'); age=d.AGE_CCD.str.extract(r'(\d+)')[0].astype(int)
d['ag']=np.select([age<=20,age<=40,age<=50],['10~20대','30~40대','50대'],'60대+')
d['기간']=np.where(d.month.isin([7,8]),'7~8월(방학)','9~12월(학기)')
nd=cal.assign(기간=np.where(cal.month.isin([7,8]),'7~8월(방학)','9~12월(학기)')).groupby(['기간','is_offday']).size()
for ind in ['종합레저타운/놀이동산','실내/실외골프장','한식']:
    x=d[d.MCT_RY_CD==ind].groupby(['ag','기간','is_offday']).TS_AT.sum()
    r=(x.xs(1,level=2)/nd.xs(1,level=1).reindex(x.xs(1,level=2).index.get_level_values(1)).values)/(x.xs(0,level=2)/nd.xs(0,level=1).reindex(x.xs(0,level=2).index.get_level_values(1)).values)
    wk=x.xs(0,level=2)/x.groupby(level=[0,1]).sum()*100
    print('==',ind,'(휴일 하루÷평일 하루 / 평일 매출 비중%)'); print(pd.concat([r.round(1).rename('휴일배수'),wk.round(0).rename('평일비중%')],axis=1).unstack(1))
