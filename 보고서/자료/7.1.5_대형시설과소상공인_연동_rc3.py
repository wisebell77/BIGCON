src=open('3.3_4.9_외지인구조_비뒤회복_rc.py').read().split("print('\\n[비 온 뒤")[0].replace("print(","(lambda *a,**k:None)(")
src=src.replace(".TS_AT.sum().rename(name)",".USE_CNT.sum().rename(name)")
exec(src)
for y in ['out_big','out_small','loc_small']:
    D['res_'+y]=smf.ols(f'np.log({y}) ~ C(dow)*is_offday + is_holiday + C(week)',data=D).fit().resid
nr=D.rain_biz10==0
print('결제건수 기준 상관: 전체', D[['res_out_big','res_out_small']].corr().iloc[0,1].round(2), '| 비 안 온 날', D.loc[nr,['res_out_big','res_out_small']].corr().iloc[0,1].round(2), '| 대형↔현지인 소상공인(비 안 온 날)', D.loc[nr,['res_out_big','res_loc_small']].corr().iloc[0,1].round(2))
