"""우선 지원 순위 알고리즘.

입력: 예보(지역, 요일, 시간대별 강수, 최고·체감·밤 최저기온 등) → 출력: 상권(지역×업종)별 경보 확률, 예상 손실, 우선순위 점수.
- 경보 확률 p: 이벤트일에 매출이 10% 이상 줄어들 확률(LightGBM 분류, models.py)
- 예상 손실(원), 세 가지 방식을 모두 낸다
  (1) expected_loss(기본): 호우는 시간대별 β — Σ_시간대 (그 요일·시간대 평상 매출 × (1 − exp(β_시간대))). β_시간대는 그 시간대에 5mm 이상(또는 1~5mm)
      비가 올 때의 시리즈별 민감도(20번과 같은 시간대 회귀, 경험적 베이즈 축소 + 증거 규칙). 폭염·열대야는 일 β.
  (2) expected_loss_post: 같은 구조이되 β 대신 사후분포에서의 기대 감소분 E[min(β,0)]을 쓴다(증거 규칙의 0 대신 연속값).
  (3) expected_loss_cal: 분류기 확률을 손실률로 보정(등방 회귀, 학습 기간 안 밖-표본 확률 기준)한 값 × 요일 평상 매출. β를 쓰지 않는다.
  비교용 expected_loss_day: 호우도 일 β로 계산.
- 점수(priority): 기본은 (1) 예상 손실(시간대 β). 호우일 백테스트(80번)에서 상위 10이 실제 손실의 가장 큰 몫을 포함했다.
  경보 확률 p는 점수에 곱하지 않고 경보 여부(임계값)로만 쓴다(곱하면 순위가 흐려진다). 비교용 점수 priority_p(= p × (1)), priority_post((2)),
  priority_day(일 β), priority_cal((3))을 같이 낸다.
- 지원 대상은 소상공인 중심 업종(small_biz=1). 80번 노트북에서 백테스트와 시나리오 격자에 쓴다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from common import FACE_GROUPS
from analysis import run_series
import models as M
import vulnerability as vul

EVS = ["heat_day", "tropical", "rain_biz10"]
CTRL = ["is_holiday", "is_chuseok_block", "is_pre_long_off", "is_post_long_off", "is_sandwich", "is_special"]
BIZ_TB = ["06_11", "12_17", "18_23"]
RN_COL = {"06_11": "w_rn_06", "12_17": "w_rn_12", "18_23": "w_rn_18"}


def scenario_weather(rn_06=0.0, rn_12=0.0, rn_18=0.0, ta_max=28.0, feels_max=28.0, night_min=20.0, hm=60.0, heat_streak=0, rn_lag1=0.0):
    """예보 시나리오를 모델 입력 열로 바꾼다."""
    rn_biz = rn_06 + rn_12 + rn_18
    return {"w_rn_biz": rn_biz, "w_rn_sum": rn_biz, "w_rn_biz_hours": sum(int(x > 0) * 6 for x in (rn_06, rn_12, rn_18)) / 2,
            "w_rn_06": rn_06, "w_rn_12": rn_12, "w_rn_18": rn_18, "w_rain_biz10": int(rn_biz >= 10),
            "w_heat_day": int(ta_max >= 33), "w_tropical": int(night_min >= 25), "w_ta_max_off": ta_max, "w_feels_max": feels_max,
            "w_night_min": night_min, "w_hm_mean": hm, "w_heat_streak": heat_streak if ta_max >= 33 else 0, "w_rn_biz_lag1": rn_lag1}


def _evidence_and_posterior(eb, by, evid_p, evid_z):
    """축소 계수에 (1) 증거 규칙 β와 (2) 사후 기대 감소분 E[min(β,0)]을 붙인다."""
    pooled_z = eb.groupby(by + ["var"]).apply(lambda g: np.average(g.coef, weights=1 / g.se ** 2) / np.sqrt(1 / np.sum(1 / g.se ** 2)), include_groups=False)
    key = list(zip(*[eb[c] for c in by + ["var"]]))
    eb = eb.copy()
    eb["region_sig"] = [pooled_z.get(k, 0) < evid_z for k in key]
    eb["own_sig"] = (eb.coef < 0) & (eb.p < evid_p)
    eb["beta"] = np.where(eb.region_sig | eb.own_sig, eb.coef_eb, 0.0).clip(max=0)
    mu, sd = eb.coef_eb.values, np.sqrt(np.clip(eb.shrink.values, 1e-6, 1)) * eb.se.values  # 사후 표준편차 = sqrt(w)·se
    eb["beta_post"] = mu * norm.cdf(-mu / sd) - sd * norm.pdf(mu / sd)  # E[min(β,0)] (정규 사후분포)
    return eb


def betas_from_panel(panel, evid_p=0.10, evid_z=-1.96):
    """일 패널에서 시리즈별 이벤트 민감도(폭염·열대야·호우): 증거 규칙 β와 사후 기대 β_post."""
    d = panel[(panel.in_scope == 1) & panel.group.isin(FACE_GROUPS)].copy()
    d.loc[d.sales == 0, "y"] = np.nan
    for c in EVS + ["rain_biz10_lag1", "cold12"]:
        d[c] = d["w_" + c]
    res = run_series(d, ["region", "industry"], EVS + ["cold12", "rain_biz10_lag1"], CTRL)
    eb = _evidence_and_posterior(vul.eb_shrink(res[res["var"].isin(EVS)].copy(), ["region", "var"]), ["region"], evid_p, evid_z)
    out = eb.pivot(index=["region", "industry"], columns="var", values=["beta", "beta_post"]).reindex(columns=pd.MultiIndex.from_product([["beta", "beta_post"], EVS])).fillna(0)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def betas_tb_from_panel(tb_panel, evid_p=0.10, evid_z=-1.96):
    """시간대 패널에서 시리즈×시간대별 비 민감도(1~5mm, 5mm+)."""
    t = tb_panel[(tb_panel.in_scope == 1) & tb_panel.group.isin(FACE_GROUPS) & tb_panel.tb.isin(BIZ_TB)].copy()
    t.loc[t.sales == 0, "y"] = np.nan
    t["tb_r1_5"] = ((t.wtb_rn_sum >= 1) & (t.wtb_rn_sum < 5)).astype(int)
    t["tb_r5p"] = (t.wtb_rn_sum >= 5).astype(int)
    res = run_series(t, ["region", "industry", "tb"], ["tb_r1_5", "tb_r5p"], CTRL)
    eb = _evidence_and_posterior(vul.eb_shrink(res, ["region", "tb", "var"]), ["region", "tb"], evid_p, evid_z)
    out = eb.pivot(index=["region", "industry"], columns=["var", "tb"], values=["beta", "beta_post"])
    out.columns = [f"{a}_{v}_{tb}" for a, v, tb in out.columns]
    for a in ["beta", "beta_post"]:
        for v in ["tb_r1_5", "tb_r5p"]:
            for tb in BIZ_TB:
                if f"{a}_{v}_{tb}" not in out:
                    out[f"{a}_{v}_{tb}"] = 0.0
    return out.fillna(0).reset_index()


def base_sales_dow(panel, excl):
    d = panel[(panel.in_scope == 1) & panel.group.isin(FACE_GROUPS) & (panel.sales > 0) & ~panel.date.isin(excl)]
    return d.groupby(["region", "industry", "dow"]).sales.mean().rename("base_sales").reset_index()


def base_sales_tb_dow(tb_panel, excl):
    t = tb_panel[(tb_panel.in_scope == 1) & tb_panel.group.isin(FACE_GROUPS) & ~tb_panel.date.isin(excl) & tb_panel.tb.isin(BIZ_TB)]
    wide = t.groupby(["region", "industry", "dow", "tb"]).sales.mean().unstack("tb").reindex(columns=BIZ_TB).fillna(0)
    wide.columns = [f"base_{tb}" for tb in wide.columns]
    return wide.reset_index()


def _loss_day(x, prefix):
    loss = np.zeros(len(x))
    for ev in EVS:
        loss = loss + x[f"w_{ev}"].values * x.base_sales.fillna(0).values * (1 - np.exp(x[f"{prefix}_{ev}"].fillna(0).clip(upper=0).values))
    return loss


def _loss_tb(x, prefix):
    loss = np.zeros(len(x))
    for tb in BIZ_TB:
        rn = x[RN_COL[tb]].fillna(0).values
        b = np.where(rn >= 5, x[f"{prefix}_tb_r5p_{tb}"].fillna(0), np.where(rn >= 1, x[f"{prefix}_tb_r1_5_{tb}"].fillna(0), 0.0))
        loss = loss + x[f"base_{tb}"].fillna(0).values * (1 - np.exp(np.minimum(b, 0)))
    for ev in ["heat_day", "tropical"]:
        loss = loss + x[f"w_{ev}"].values * x.base_sales.fillna(0).values * (1 - np.exp(x[f"{prefix}_{ev}"].fillna(0).clip(upper=0).values))
    return loss


class PriorityModel:
    """분류기 + β(증거 규칙/사후 기대, 일/시간대) + 확률→손실률 보정 + 요일(·시간대) 평상 매출."""

    def __init__(self, clf, betas, betas_tb, base, base_tb, categories, iso, feats):
        self.clf, self.betas, self.betas_tb, self.base, self.base_tb, self.categories, self.iso, self.feats = clf, betas, betas_tb, base, base_tb, categories, iso, feats

    @classmethod
    def fit(cls, ev_train, panel_train, tb_panel_train, excl, cls_params=None, feats=None):
        feats = feats or M.CLS_FEAT
        params = cls_params or M.CLS_PARAMS
        clf = lgb.LGBMClassifier(**params).fit(ev_train[feats], ev_train.label, categorical_feature=M.CAT)
        cats = {c: ev_train[c].cat.categories for c in M.CAT}
        # 확률 → 손실률 보정: 학습 기간 안에서 날짜 묶음 밖-표본 확률에 실제 손실률(1 − exp(dev), 0 이상)을 등방 회귀로 맞춘다
        p_oof = np.full(len(ev_train), np.nan)
        for tr_i, te_i in GroupKFold(n_splits=5).split(ev_train, ev_train.label, groups=ev_train.date):
            m = lgb.LGBMClassifier(**params).fit(ev_train.iloc[tr_i][feats], ev_train.iloc[tr_i].label, categorical_feature=M.CAT)
            p_oof[te_i] = m.predict_proba(ev_train.iloc[te_i][feats])[:, 1]
        # 목표는 '순' 감소율(1 − exp(dev), 음수 허용). 0에서 자른 값을 쓰면 평상시 변동(잡음)까지 손실로 학습해 폭염일처럼 효과가 없는 날에도
        # 손실이 생긴다. 순효과는 잡음이 평균 0으로 상쇄되어 날씨 때문에 생긴 몫만 남는다. 이벤트 종류마다 따로 맞춘다(호우일과 폭염일의 손실 구조가 다르다).
        # 목표는 로그 편차(−dev, 양수면 손실). 금액 비율(1−exp(dev))은 큰 증가가 극단값이 되어 평균을 흔든다.
        # dev의 극단값(기준선이 거의 0인 날 등)은 2~98% 분위에서 자른다(윈저화)
        lo, hi = np.nanquantile(ev_train.dev.values, [0.02, 0.98])
        log_loss = -np.clip(ev_train.dev.values, lo, hi)
        etype = np.select([ev_train.w_rain_biz10.values == 1, ev_train.w_heat_day.values == 1], ["rain", "heat"], "tropical")
        iso = {}
        for k in ["rain", "heat", "tropical"]:
            m_ = etype == k
            iso[k] = IsotonicRegression(out_of_bounds="clip").fit(p_oof[m_], log_loss[m_]) if m_.sum() >= 50 else None
        return cls(clf, betas_from_panel(panel_train), betas_tb_from_panel(tb_panel_train), base_sales_dow(panel_train, excl),
                   base_sales_tb_dow(tb_panel_train, excl), cats, iso, feats)

    def _rate_cal(self, out):
        """E[−dev | p, 이벤트 종류]를 손실률 1 − exp(−E)로 바꾼다(음수면 0)."""
        etype = np.select([out.w_rain_biz10.values == 1, out.w_heat_day.values == 1], ["rain", "heat"], "tropical")
        rate = np.zeros(len(out))
        for k, iso in self.iso.items():
            m_ = etype == k
            if m_.any() and iso is not None:
                rate[m_] = 1 - np.exp(-np.clip(iso.predict(out.p_alert.values[m_]), 0, None))
        return rate

    def score(self, rows):
        X = rows.copy()
        for c in M.CAT:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories[c])
        keep = [c for c in ["date"] if c in rows.columns]  # 날짜가 있으면 그대로 넘겨 실제 손실과 날짜로 맞출 수 있게 한다
        out = rows[keep + ["region", "industry", "dow", "small_biz", "w_rn_06", "w_rn_12", "w_rn_18"] + [f"w_{ev}" for ev in EVS]].copy()
        out["region"], out["industry"] = out.region.astype(str), out.industry.astype(str)
        out["p_alert"] = self.clf.predict_proba(X[self.feats])[:, 1]
        out = out.merge(self.base, on=["region", "industry", "dow"], how="left")
        x = out.merge(self.betas, on=["region", "industry"], how="left").merge(self.betas_tb, on=["region", "industry"], how="left")
        x = x.merge(self.base_tb, on=["region", "industry", "dow"], how="left")
        out["expected_loss_day"] = _loss_day(x, "beta")
        out["expected_loss"] = _loss_tb(x, "beta")
        out["expected_loss_post"] = _loss_tb(x, "beta_post")
        out["rate_cal"] = self._rate_cal(out)  # E[순 감소율 | p, 이벤트 종류], 음수면 손실 없음
        out["expected_loss_cal"] = out.base_sales.fillna(0).values * np.clip(out.rate_cal.values, 0, None)
        # 점수: 기본은 예상 손실(시간대 β). 호우일 백테스트에서 확률을 곱하면 오히려 순위가 흐려져 확률은 경보 여부(임계값)로만 쓴다.
        out["priority"] = out.expected_loss
        out["priority_p"] = out.p_alert * out.expected_loss
        out["priority_post"] = out.expected_loss_post
        out["priority_day"] = out.expected_loss_day
        out["priority_cal"] = out.expected_loss_cal
        out["rank_all"] = out.priority.rank(ascending=False, method="first")
        sb = out.small_biz == 1
        out.loc[sb, "rank_small_biz"] = out.loc[sb, "priority"].rank(ascending=False, method="first")
        return out.sort_values("priority", ascending=False)

    def scenario(self, template_rows, wx):
        rows = template_rows.copy()
        for k, v in wx.items():
            rows[k] = v
        rows = M.add_exposure_features(rows)
        return self.score(rows)
