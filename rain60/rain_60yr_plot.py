import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
have={f.name for f in font_manager.fontManager.ttflist}
mpl.rcParams['font.family']=next(c for c in ['NanumGothic','Noto Sans CJK KR','Noto Sans KR'] if c in have); mpl.rcParams['axes.unicode_minus']=False
a=pd.read_pickle('annual.pkl'); T=pd.read_pickle('trend.pkl').set_index('지표')
P=[('R20mm','일 20mm 이상 일수 (일/년)'),('강수강도_SDII','강수강도: 비 온 날 평균 강수량 (mm/일)'),
   ('R1h_max','1시간 최다강수량 연최대 (mm)'),('R1h_30mm일수','시간당 30mm 이상 내린 날 (일/년)')]
fig,axes=plt.subplots(2,2,figsize=(12,7))
for ax,(c,t) in zip(axes.flat,P):
    s=a[c]; x=s.index.values
    ax.bar(x,s.values,color='#C9D6E8',width=0.8)
    ax.plot(x,s.rolling(10,center=True,min_periods=5).mean(),color='#2A78D6',lw=2,label='10년 이동평균')
    lr=stats.linregress(x,s.values); ax.plot(x,lr.intercept+lr.slope*x,'--',color='#E8693A',lw=1.6,label='추세선')
    r=T.loc[c]; sig='*' if r['MK p']<0.05 else ''
    ax.set_title(f"{t}\n10년당 {r['10년당 변화']:+.2f}{sig} · 전반 {r['전반 30년 평균']} → 후반 {r['후반 30년 평균']} ({r['변화율(%)']:+.0f}%)",loc='left',fontsize=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.grid(axis='y',color='#eee')
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('춘천 60년(1966~2025) 강수 추세 — 비 오는 날 수는 그대로, 짧고 강한 비가 늘었다',x=0.01,ha='left',fontweight='bold')
fig.text(0.01,-0.01,'자료: 기상청 ASOS 춘천(101) 일자료 · * Mann-Kendall 검정 p<0.05 · 전반=1966~1995, 후반=1996~2025',fontsize=8,color='#777')
fig.tight_layout(); fig.savefig('춘천_60년_강수추세.png',dpi=160,bbox_inches='tight')
T.reset_index().to_excel('춘천_60년_강수추세.xlsx',index=False)
with pd.ExcelWriter('춘천_60년_강수추세.xlsx') as xw:
    T.reset_index().to_excel(xw,sheet_name='추세요약',index=False); a.round(1).to_excel(xw,sheet_name='연도별')
