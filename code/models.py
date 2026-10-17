"""예측·분류 모델의 데이터 준비와 교차검증 함수. 40번 노트북과 같은 논리를 70·80번에서 다시 쓰기 위해 모듈로 뺐다."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import FACE_GROUPS, TIME_BANDS, load
from analysis import deviation, exclusion_days

SEED = 42
CAT = ["region", "industry", "group"]
CAL = ["dow", "is_holiday", "is_pre_long_off", "is_post_long_off", "is_sandwich", "is_special", "is_chuseok_block",
       "coupon_phase", "is_coupon_last_week", "days_to_month_end", "off_block_len", "off_block_pos", "payback_on"]
HIST = ["y_dow_mean4", "h_lag7", "h_lag14", "h_ma7"]
PROF = ["p_ticket", "p_share_tb_00_05", "p_share_tb_06_11", "p_share_tb_12_17", "p_share_tb_18_23", "p_age_young", "p_age_mid", "p_age_old",
        "p_share_sex_F", "p_ratio_weekend", "p_share_nonlocal", "small_biz", "outdoor", "discretionary"]
WX = ["w_ta_max_off", "w_feels_max", "w_night_min", "w_rn_biz", "w_rn_sum", "w_rn_biz_hours", "w_rn_06", "w_rn_12", "w_rn_18", "w_hm_mean",
      "w_heat_day", "w_tropical", "w_rain_biz10", "w_rn_biz_lag1", "w_rn_biz_lead1", "w_heat_streak", "w_ta_max_anom31"]
FEAT_FULL = CAT + CAL + HIST + PROF + WX
FEAT_NOW = CAT + CAL + HIST + PROF
CLS_FEAT = CAT + PROF + ["dow", "is_weekend", "w_ta_max_off", "w_feels_max", "w_night_min", "w_rn_biz", "w_rn_06", "w_rn_12", "w_rn_18",
                         "w_rn_biz_hours", "w_hm_mean", "w_heat_day", "w_tropical", "w_rain_biz10", "w_heat_streak", "w_rn_biz_lag1"]
# 노출 × 위해 상호작용 피처: 비가 '그 업종의 장사 시간'에 왔는가, 주말 장사 업종에 주말 비가 왔는가 등
EXP = ["x_rain_exp", "x_rain_exp_share", "x_heat_exp", "x_wkend_rain", "x_nonlocal_rain"]
CLS_FEAT_EXP = CLS_FEAT + EXP
FEAT_FULL_EXP = FEAT_FULL + EXP


def add_exposure_features(df):
    """시리즈의 시간대·주말·외지인 구조와 그날 날씨를 곱한 피처."""
    df = df.copy()
    df["x_rain_exp"] = df.p_share_tb_06_11 * df.w_rn_06 + df.p_share_tb_12_17 * df.w_rn_12 + df.p_share_tb_18_23 * df.w_rn_18
    df["x_rain_exp_share"] = (df.x_rain_exp / df.w_rn_biz.replace(0, np.nan)).fillna(0)  # 영업시간 비 중 그 업종 시간대 비중으로 가중한 몫(0~1)
    df["x_heat_exp"] = df.p_share_tb_12_17 * df.w_feels_max
    df["x_wkend_rain"] = df.is_weekend * df.w_rn_biz * df.p_ratio_weekend
    df["x_nonlocal_rain"] = df.p_share_nonlocal.fillna(0) * df.w_rn_biz
    return df
REG_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, verbose=-1)
CLS_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)


def load_daily():
    """40번 A와 같은 일 패널: 매출 0인 날 제외, 기준선 대비 편차 r, 시간대별 강수."""
    panel = load("model_panel_day")
    wtb = load("weather_tb")
    rn_tb = wtb.pivot_table(index=["region", "date"], columns="tb", values="rn_sum").reindex(columns=TIME_BANDS).fillna(0)
    rn_tb.columns = [f"w_rn_{t[:2]}" for t in TIME_BANDS]
    panel = panel.merge(rn_tb.reset_index(), on=["region", "date"], how="left")
    d = panel[(panel.in_scope == 1) & (panel.sales > 0)].copy()
    d["series"] = d.region + "|" + d.industry
    d["face"] = d.group.isin(FACE_GROUPS).astype(int)
    d = d.rename(columns={"p_share_age_10-20대": "p_age_young", "p_share_age_30-50대": "p_age_mid", "p_share_age_60대+": "p_age_old"})
    d["h_lag7"] = d.y_lag7 - d.y_dow_mean4
    d["h_lag14"] = d.y_lag14 - d.y_dow_mean4
    d["h_ma7"] = d.y_ma7_lag7 - d.y_dow_mean4
    d["r"] = d.y - d.y_dow_mean4
    d = add_exposure_features(d)
    for c in CAT:
        d[c] = d[c].astype("category")
    return d


def fit_predict(train, test, feats, params=None):
    m = lgb.LGBMRegressor(**(params or REG_PARAMS))
    m.fit(train[feats], train.r, categorical_feature=[c for c in CAT if c in feats])
    return m, test.y_dow_mean4.values + m.predict(test[feats])


def metrics(df, pred_col):
    out = {}
    for name, sub in [("전체", df), ("대면소비", df[df.face == 1]), ("비 10mm+ 날", df[(df.face == 1) & (df.w_rain_biz10 == 1)]),
                      ("폭염일", df[(df.face == 1) & (df.w_heat_day == 1)]), ("열대야", df[(df.face == 1) & (df.w_tropical == 1)])]:
        s = sub.dropna(subset=[pred_col, "y"])
        if len(s) < 30:
            out[name] = {"MAE(log)": np.nan, "R2(log)": np.nan, "WAPE(%)": np.nan, "편향(%)": np.nan, "n": len(s)}
            continue
        err = np.expm1(s[pred_col]) - s.sales
        out[name] = {"MAE(log)": mean_absolute_error(s.y, s[pred_col]), "R2(log)": r2_score(s.y, s[pred_col]),
                     "WAPE(%)": np.abs(err).sum() / s.sales.sum() * 100, "편향(%)": err.sum() / s.sales.sum() * 100, "n": len(s)}
    return pd.DataFrame(out).T


def cv_month(dm, feats, params=None, col="pred"):
    """한 달씩 빼고 학습하는 교차검증. dm에 예측 열을 채워 돌려준다."""
    dm = dm.copy()
    dm[col] = np.nan
    for m_ in sorted(dm.month.unique()):
        tr, te = dm[dm.month != m_], dm[dm.month == m_]
        _, dm.loc[te.index, col] = fit_predict(tr, te, feats, params)
    return dm


def load_events(d=None, threshold=0.9):
    """40번 B와 같은 이벤트일 관측(대면소비 시리즈 × 이벤트일)과 라벨(기준선 대비 threshold 이하면 1)."""
    d = d if d is not None else load_daily()
    cal = load("calendar")
    EXCL = exclusion_days(cal)
    face = d[d.face == 1].copy()
    dev = deviation(face, ["region", "industry"], excl=EXCL)
    dev["region"], dev["industry"] = dev.region.astype(str), dev.industry.astype(str)
    fb = face.assign(region=face.region.astype(str), industry=face.industry.astype(str)).merge(dev, on=["region", "industry", "date"])
    ev = fb[(fb.w_rain_biz10 == 1) | (fb.w_heat_day == 1) | (fb.w_tropical == 1)].copy()
    ev["label"] = (ev.dev <= np.log(threshold)).astype(int)
    ev["label20"] = (ev.dev <= np.log(0.8)).astype(int)
    for c in CAT:
        ev[c] = ev[c].astype("category")
    return ev


def cv_group(ev, feats, params=None, n_splits=5, col="p_hat", y="label"):
    """날짜로 묶은 GroupKFold 교차검증. 밖-표본 확률을 채워 돌려준다."""
    ev = ev.copy()
    ev[col] = np.nan
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(ev, ev[y], groups=ev.date):
        clf = lgb.LGBMClassifier(**(params or CLS_PARAMS)).fit(ev.iloc[tr_i][feats], ev.iloc[tr_i][y], categorical_feature=[c for c in CAT if c in feats])
        ev.iloc[te_i, ev.columns.get_loc(col)] = clf.predict_proba(ev.iloc[te_i][feats])[:, 1]
    return ev


def cls_metrics(y, p, top=0.2):
    k = int(len(y) * top)
    order = np.argsort(-np.asarray(p))[:k]
    y = pd.Series(np.asarray(y))
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "양성 비율": y.mean(),
            "정밀도@상위20%": y.iloc[order].mean(), "재현율@상위20%": y.iloc[order].sum() / y.sum()}
