"""노트북이 같이 쓰는 분석 함수: 기준선 편차, 이벤트 대비, 고정효과 회귀, 그림 설정."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import DATES

# ---------- 그림 공통 ----------
SURFACE, INK, INK2, MUTED, GRID, AXIS, WASH = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
REGION_COLOR = {"gangnam": "#2a78d6", "chuncheon": "#eb6834"}
AGE_COLOR = {"10-20대": "#86b6ef", "30-50대": "#2a78d6", "60대+": "#104281"}
EVENT_COLOR = {"heat_day": "#e34948", "tropical": "#8b4fd6", "rain_biz10": "#2a78d6", "rain_biz20": "#1b4f9c", "cold12": "#5fa8c9"}
RC = {
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 9,
}


def exclusion_days(cal):
    """기준선(평상시)에서 빼는 날: 공휴일, 추석 연휴, 연휴 전후, 징검다리, 특별일, 소비쿠폰 마지막 주."""
    m = ((cal.is_holiday == 1) | (cal.is_chuseok_block == 1) | (cal.is_pre_long_off == 1)
         | (cal.is_post_long_off == 1) | (cal.is_sandwich == 1) | (cal.is_special == 1)
         | (cal.is_coupon_last_week == 1))
    return set(cal.loc[m, "date"])


def deviation(df, keys, value="sales", excl=frozenset(), min_nb=2):
    """각 날의 로그 매출 - 같은 요일 앞뒤 1·2주 평균(제외일은 빼고, 이웃이 min_nb개 이상일 때만).
    날씨처럼 며칠 단위로 바뀌는 충격을 추세·요일·정책 효과와 떼어 놓는 가장 단순한 기준선이다."""
    wide = np.log1p(df.pivot_table(index="date", columns=keys, values=value, aggfunc="sum")).reindex(DATES)
    clean = wide.copy()
    clean.loc[clean.index.isin(excl)] = np.nan
    nb = [clean.shift(k) for k in (7, 14, -7, -14)]
    cnt = sum(x.notna().astype(int) for x in nb)
    base = pd.concat(nb).groupby(level=0).mean().where(cnt >= min_nb)
    dev = wide - base
    dev.loc[dev.index.isin(excl)] = np.nan
    out = dev.stack(list(range(len(keys))), future_stack=True).rename("dev").reset_index()
    out.columns = ["date"] + keys + ["dev"]
    return out.dropna(subset=["dev"])


def pct(x):
    return (np.exp(x) - 1) * 100


def contrast(df, keys, cond_a, cond_b, col="dev"):
    """조건 A인 날과 조건 B인 날의 평균 편차 차이를 %로(95% 신뢰구간, z값)."""
    a = df[cond_a].groupby(keys)[col].agg(["mean", "var", "count"])
    b = df[cond_b].groupby(keys)[col].agg(["mean", "var", "count"])
    diff = a["mean"] - b["mean"]
    se = np.sqrt(a["var"] / a["count"] + b["var"] / b["count"])
    return pd.DataFrame({"pct": pct(diff), "lo": pct(diff - 1.96 * se), "hi": pct(diff + 1.96 * se),
                         "n": a["count"], "n_base": b["count"], "z": diff / se, "dlog": diff, "se": se})


def fe_ols(df, y, x_cols, fe_cols, cov="hac", cluster="date", maxlags=7):
    """더미 고정효과 OLS.
    cov="hac": Newey-West(시계열 자기상관, 시리즈 하나짜리 회귀에 사용)
    cov="cluster": cluster 열(기본 날짜)로 묶은 표준오차(여러 시리즈를 합친 회귀에 사용)
    돌려주는 표: x_cols별 계수, 표준오차, p값, %효과와 95% 구간."""
    d = df.dropna(subset=[y] + x_cols).copy()
    X = d[x_cols].astype(float)
    if fe_cols:
        fe = pd.get_dummies(d[fe_cols].astype(str), drop_first=True, dtype=float)
        X = pd.concat([X, fe], axis=1)
    X = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(d[y].astype(float), X)
    if cov == "hac":
        m = ols.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif cov == "cluster":
        m = ols.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    else:
        m = ols.fit(cov_type="HC1")
    res = pd.DataFrame({"coef": m.params[x_cols], "se": m.bse[x_cols], "p": m.pvalues[x_cols]})
    res["pct"] = pct(res["coef"])
    res["lo"] = pct(res["coef"] - 1.96 * res["se"])
    res["hi"] = pct(res["coef"] + 1.96 * res["se"])
    res["n"] = len(d)
    res["r2"] = m.rsquared
    return res, m


def run_series(df, keys, xvars, ctrl, fe=("dow", "week"), y="y", cov="hac", min_obs=60):
    """keys로 묶은 시리즈마다 고정효과 회귀(fe_ols). 그 시리즈에서 변동이 없는 변수는 뺀다.
    돌려주는 표: var, coef, se, p, pct, lo, hi, n, r2, keys…, n_event(변수가 1인 날 수)."""
    rows = []
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        xs = [c for c in xvars if g[c].nunique() > 1]
        cs = [c for c in ctrl if g[c].nunique() > 1]
        if not xs or g[y].notna().sum() < min_obs:
            continue
        res, _ = fe_ols(g, y, xs + cs, list(fe), cov=cov)
        res = res.loc[xs].reset_index().rename(columns={"index": "var"})
        for k, v in zip(keys, key):
            res[k] = v
        res["n_event"] = [int(g.loc[g[y].notna(), c].sum()) for c in xs]
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def pool(res, by, coef="coef", se="se"):
    """시리즈별 계수를 역분산 가중 평균으로 합친다(고정효과 메타분석)."""
    r = res.dropna(subset=[coef, se]).copy()
    r["w"] = 1 / r[se] ** 2
    g = r.groupby(by)
    out = pd.DataFrame({"coef": g.apply(lambda x: (x[coef] * x["w"]).sum() / x["w"].sum(), include_groups=False),
                        "se": g.apply(lambda x: np.sqrt(1 / x["w"].sum()), include_groups=False),
                        "k": g.size()})
    out["pct"] = pct(out["coef"])
    out["lo"] = pct(out["coef"] - 1.96 * out["se"])
    out["hi"] = pct(out["coef"] + 1.96 * out["se"])
    out["z"] = out["coef"] / out["se"]
    return out


def finish(fig, fig_dir, name, title=None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    return fig
