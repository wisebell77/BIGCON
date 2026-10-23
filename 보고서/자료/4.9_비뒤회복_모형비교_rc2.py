exec(open('3.3_4.9_외지인구조_비뒤회복_rc.py').read().split("print('\\n[비 온 뒤")[0].replace("print(","(lambda *a,**k:None)("))
D['t']=np.arange(len(D)); D=D.fillna({'coupon_phase':'none'})
base='rain_biz10 + lead1 + r_1_7 + r_8_14 + r_15_28 + r_29_42 + C(dow)*is_offday + is_holiday + is_chuseok_block + C(coupon_phase)'
specs={'월 고정':base+' + C(month)','2주 고정':base+' + C(week//2)','추세(3차)':base+' + t + I(t**2) + I(t**3)','8월~10월만, 월 고정':base+' + C(month)'}
for y in ['out_all','out_small']:
  for k,f in specs.items():
    dd=D if '만' not in k else D[D.month.isin([8,9,10])]
    res=smf.ols(f'np.log({y}) ~ '+f,data=dd).fit(cov_type='HAC',cov_kwds={'maxlags':7})
    p=res.params; L={kk:(np.exp(p[kk])-1)*100 for kk in ['rain_biz10','r_1_7','r_8_14','r_15_28','r_29_42']}
    net=L['rain_biz10']+7*L['r_1_7']+7*L['r_8_14']+14*L['r_15_28']+14*L['r_29_42']
    print(y,k,{kk:f'{v:+.1f}({res.pvalues[kk]:.2f})' for kk,v in L.items()}, '6주 순합(하루치 %)', round(net))
