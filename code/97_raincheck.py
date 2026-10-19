"""97. 레인체크 파라미터 튜닝: 회의안의 세부 조건(할인율 평일/주말, 결제 1회 상한, 횟수 제한, 유효기간, 월 한도, 트리거 문턱)을 우리 데이터로 검토한다.

입력: card_day_resid(데이터2 거주지 구분, 지역×업종×일), card2_clean(성·연령 셀, 건당 결제액), model_panel_day(업종군·분석 범위), calendar, weather_daily,
      a_expected_loss_calendar_2026(기후학적 호우 확률), f_forecast_skill(단기예보 적중), tour_monthly_visitors(관광데이터랩 월 방문자)
대상 업종군(레인체크 사용처, 회의안의 숙소·식당·어트랙션): 음식점, 카페·베이커리, 숙박, 여가·스포츠. 분석 범위(in_scope) 업종만.
출력(outputs/tables/rc_*.csv, 그림 rc01~rc04)
 A. rc_sensitivity_dow.csv   비 10mm+ 민감도(매출·건수)를 현지인/외지인 × 평일/주말·공휴일로 나눠 추정(업종군별 회귀 → 역분산 합성)
 B. rc_trigger_day_pool.csv  트리거일(비 10mm+)의 외지인 결제 규모: 실제, 기준선, 감소분(= 사라진 방문), 관광데이터랩 방문자 환산
 C. rc_ticket_cap.csv        외지인 건당 결제액 분포와 할인율×상한별 기대 할인액, 상한에 걸리는 비중
 D. rc_trigger_months.csv    2026년 월별 기대 트리거 일수(관측 10mm 확정, 예보 가발급)와 2025년 실제
 E. rc_cost_per_1000.csv, rc_cost_grid.csv, rc_budget.csv  쿠폰 1,000장당 비용·순증 매출, 파라미터 격자, 예산 → 월 한도
 F. rc_validity.csv          유효기간별 도달 월·방문자 커버·유효기간 안 추가 트리거 기대 수
 G. rc_threshold.csv         확정 문턱(1/5/10/20/30mm)별 외지인 손실 커버와 손실 없는 날 비중
 H. rc_monthly_cap.csv       월 2회 발급 제한이 깎는 발급 비중(월별 트리거 일수 분포)
 rc_assumptions.csv          비용 모델 가정 목록
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from analysis import RC, SURFACE, INK, INK2, MUTED, WASH, REGION_COLOR, exclusion_days, deviation, run_series, pool, pct, finish

RC_GROUPS = ["음식점", "카페·베이커리", "숙박", "여가·스포츠"]
CTRL = ["is_holiday", "is_chuseok_block", "is_pre_long_off", "is_post_long_off", "is_sandwich", "is_special"]
ORIGIN_NAME = {"local": "현지인", "nonlocal": "외지인"}
# 2026년 7~12월 공휴일(대체공휴일 포함, 추정): 광복절 대체 8/17, 추석 9/24~26 + 대체 9/28, 개천절 대체 10/5, 한글날 10/9, 성탄절 12/25
HOLIDAYS_2026 = pd.to_datetime(["2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-28", "2026-10-03", "2026-10-05", "2026-10-09", "2026-12-25"])
# 비용 모델 기본 가정
BASE = dict(use_rate=0.5, uses_per_coupon=2, offday_use_share=0.3, deadweight=0.4, rate_wd=0.20, rate_we=0.10, cap=10000)
CAP_BY_GROUP = {"음식점": 10000, "카페·베이커리": 10000, "숙박": 20000, "여가·스포츠": 20000}  # 비에 가장 민감하고 결제액이 큰 숙박·여가는 상한을 높인다
ASSUMPTIONS = [
    ("사용률(발급 쿠폰 중 유효기간 안에 한 번이라도 쓰는 비율)", BASE["use_rate"], "0.3~0.7", "자료 없음. 시범 KPI. 지역사랑상품권 정책발행 사용률이 높은 편이라 0.5를 기본"),
    ("사용 횟수(쓴 쿠폰 1장당 결제 횟수, 1일 3회 상한 안)", BASE["uses_per_coupon"], "1~3", "숙소·식당·어트랙션 중 두 곳 이용을 기본"),
    ("주말·공휴일 사용 비중", BASE["offday_use_share"], "0.2~0.5", "평일 할인율을 높여 평일로 유도한다는 설계. 외지인 결제 건수의 주말 비중(B표)보다 낮게"),
    ("사중손실(쿠폰 없이도 왔을 비중)", BASE["deadweight"], "0.3~0.5", "자료 없음. 구현안 4절과 같은 범위"),
    ("건당 결제액 분포", "데이터2 셀(업종×성×연령×일) 평균", "", "거래 1건 단위 분산이 빠져 상한에 걸리는 비중은 실제보다 작게 잡힘"),
    ("방문자 환산", "관광데이터랩 월 방문자 ÷ 신한 외지인 결제 건수", "", "방문자에는 관광 목적이 아닌 외지인도 섞여 상한(최대치)으로 본다"),
    ("운영비(앱·플랫폼·홍보·검증)", "제외", "", "쿠폰 할인액만 계산. 운영비는 시범 설계 후 별도"),
]


def load_all():
    cal = C.load("calendar")
    excl = exclusion_days(cal)
    w = C.load("weather_daily")
    pan = C.load("model_panel_day")
    sc = pan[["region", "industry", "group", "in_scope", "small_biz"]].drop_duplicates()
    rs = C.load("card_day_resid").merge(sc, on=["region", "industry"])
    rs = rs[rs.in_scope == 1]
    return cal, excl, w, sc, rs


# ---------------------------------------------------------------- A. 민감도: 거주지 × 평일/주말
def sensitivity_dow(rs, cal, w):
    x = rs[rs.group.isin(C.FACE_GROUPS)].groupby(["region", "group", "date"])[["sales_local", "sales_nonlocal", "cnt_local", "cnt_nonlocal"]].sum().reset_index()
    parts = []
    for origin in ["local", "nonlocal"]:
        d = x[["region", "group", "date"]].copy()
        d["origin"], d["sales"], d["cnt"] = origin, x[f"sales_{origin}"], x[f"cnt_{origin}"]
        parts.append(d)
    rl = pd.concat(parts, ignore_index=True)
    rl["y"] = np.log1p(rl.sales).where(rl.sales > 0)
    rl["y_cnt"] = np.log1p(rl.cnt).where(rl.cnt > 0)
    rl = rl.merge(cal, on="date").merge(w[["region", "date", "rain_biz10", "heat_day", "tropical", "rain_biz10_lag1", "rain_biz10_lead1"]], on=["region", "date"])
    rl["rain_wd"] = rl.rain_biz10 * (1 - rl.is_offday)
    rl["rain_we"] = rl.rain_biz10 * rl.is_offday
    XV = ["rain_wd", "rain_we", "heat_day", "tropical", "rain_biz10_lag1"]
    out = []
    for yv, yname in [("y", "매출"), ("y_cnt", "결제 건수")]:
        res = run_series(rl, ["region", "group", "origin"], XV, CTRL, y=yv)
        for scope_name, groups in [("대면 업종군 전체", C.FACE_GROUPS), ("레인체크 업종군(음식·카페·숙박·여가)", RC_GROUPS)]:
            sub = res[res.group.isin(groups) & res["var"].isin(["rain_wd", "rain_we"])]
            p = pool(sub, ["var", "region", "origin"]).reset_index()
            nev = sub.groupby(["var", "region", "origin"]).n_event.max().rename("n_event_max").reset_index()
            p = p.merge(nev, on=["var", "region", "origin"])
            p["지표"], p["범위"] = yname, scope_name
            out.append(p)
        # 업종군별도 남긴다(레인체크 업종군만)
        sub = res[res.group.isin(RC_GROUPS) & res["var"].isin(["rain_wd", "rain_we"])].copy()
        sub["지표"] = yname
        out.append(sub.rename(columns={"group": "업종군"}).assign(범위="업종군별", k=1))
    A = pd.concat(out, ignore_index=True)
    A["요일"] = A["var"].map({"rain_wd": "평일", "rain_we": "주말·공휴일"})
    A["거주지"] = A.origin.map(ORIGIN_NAME)
    A["지역"] = A.region.map(C.REGION_NAME)
    cols = ["지표", "범위", "지역", "거주지", "요일", "업종군", "pct", "lo", "hi", "se", "z", "k", "n_event", "n_event_max"]
    A = A.reindex(columns=cols).sort_values(["지표", "범위", "지역", "거주지", "요일"])
    A.to_csv(C.TAB / "rc_sensitivity_dow.csv", index=False, encoding="utf-8-sig")
    show = A[(A.범위 != "업종군별")].pivot_table(index=["지표", "범위", "지역", "거주지"], columns="요일", values="pct").round(1)
    print("=== A. 비 10mm+ 민감도(%), 평일 vs 주말·공휴일"); print(show.to_string())
    return A, rl


# ---------------------------------------------------------------- B. 트리거일 외지인 결제 규모
def trigger_pool(rs, cal, excl, w):
    x = rs[(rs.region == "chuncheon") & rs.group.isin(RC_GROUPS)].groupby("date")[["sales_nonlocal", "cnt_nonlocal", "sales_local", "cnt_local"]].sum().reset_index()
    x["region"] = "chuncheon"
    rows = {}
    for v in ["cnt_nonlocal", "sales_nonlocal", "cnt_local"]:
        dv = deviation(x, ["region"], value=v, excl=excl).rename(columns={"dev": f"dev_{v}"})
        rows[v] = dv[["date", f"dev_{v}"]]
    d = x.merge(rows["cnt_nonlocal"], on="date", how="left").merge(rows["sales_nonlocal"], on="date", how="left").merge(rows["cnt_local"], on="date", how="left")
    d = d.merge(cal[["date", "month", "is_offday", "is_weekend"]], on="date").merge(w[w.region == "chuncheon"][["date", "rain_biz10", "rn_biz"]], on="date")
    d["base_cnt_nonlocal"] = d.cnt_nonlocal / np.exp(d.dev_cnt_nonlocal)
    d["base_sales_nonlocal"] = d.sales_nonlocal / np.exp(d.dev_sales_nonlocal)
    d["lost_cnt_nonlocal"] = d.base_cnt_nonlocal - d.cnt_nonlocal
    d["lost_sales_nonlocal"] = d.base_sales_nonlocal - d.sales_nonlocal
    # 관광데이터랩 방문자 환산: 월 방문자 ÷ 월 신한 외지인 결제 건수
    tv = C.load("tour_monthly_visitors")
    tv["ym"] = tv.ym.astype(int)
    m = x.assign(ym=x.date.dt.year * 100 + x.date.dt.month).groupby("ym").agg(cnt=("cnt_nonlocal", "sum"), days=("date", "nunique")).reset_index().merge(tv[["ym", "visitors"]], on="ym")
    m["visitors_per_day"] = m.visitors / m.days
    m["cnt_per_visitor"] = m.cnt / m.visitors
    ratio = float(m.cnt.sum() / m.visitors.sum())
    d["visitors_est"] = d.cnt_nonlocal / ratio
    d["lost_visitors_est"] = d.lost_cnt_nonlocal / ratio
    trig = d[(d.rain_biz10 == 1) & d.dev_cnt_nonlocal.notna()]
    summ = []
    for label, sub in [("평일", trig[trig.is_offday == 0]), ("주말·공휴일", trig[trig.is_offday == 1]), ("전체", trig)]:
        allday = d[d.dev_cnt_nonlocal.notna()]
        ref = allday if label == "전체" else allday[allday.is_offday == (label != "평일")]
        summ.append({"요일": label, "트리거일 수(2025 7~12월)": len(sub),
                     "외지인 결제 건수(평상시 평균)": round(ref[ref.rain_biz10 == 0].cnt_nonlocal.mean()),
                     "트리거일 외지인 결제 건수(실제 평균)": round(sub.cnt_nonlocal.mean()), "트리거일 기준선 건수": round(sub.base_cnt_nonlocal.mean()),
                     "감소 건수(사라진 방문)": round(sub.lost_cnt_nonlocal.mean()), "감소율(%)": round(float(pct(sub.dev_cnt_nonlocal.mean())), 1),
                     "트리거일 외지인 매출(억, 실제)": round(sub.sales_nonlocal.mean() / 1e8, 2), "매출 감소(억)": round(sub.lost_sales_nonlocal.mean() / 1e8, 2), "매출 감소율(%)": round(float(pct(sub.dev_sales_nonlocal.mean())), 1),
                     "현지인 건수 감소율(%)": round(float(pct(sub.dev_cnt_local.mean())), 1),
                     "방문자 환산(실제, 명)": round(sub.visitors_est.mean()), "방문자 환산(사라진, 명)": round(sub.lost_visitors_est.mean())})
    B = pd.DataFrame(summ)
    B.to_csv(C.TAB / "rc_trigger_day_pool.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== B. 트리거일 외지인 결제 규모(레인체크 업종군, 신한카드 기준). 방문자 환산 비율: 결제 {ratio:.3f}건/방문자, 월 방문자 평균 {m.visitors.mean()/1e4:,.0f}만 명(일 {m.visitors_per_day.mean():,.0f}명)")
    print(B.to_string(index=False))
    return B, d, ratio, m


# ---------------------------------------------------------------- C. 건당 결제액과 상한
def ticket_cap(sc):
    c2 = C.load("card2_clean").merge(sc, on=["region", "industry"])
    t = c2[(c2.region == "chuncheon") & (c2.origin == "nonlocal") & c2.group.isin(RC_GROUPS) & (c2.in_scope == 1) & (c2.cnt > 0)].copy()
    t["ticket"] = t.sales / t.cnt
    t = t[t.ticket > 0]
    rates = [0.10, 0.15, 0.20, 0.25, 0.30]
    caps = [5000, 10000, 15000, 20000, 30000, np.inf]

    def wq(s, wgt, q):
        o = np.argsort(s.values); cs = np.cumsum(wgt.values[o]) / wgt.sum()
        return float(np.interp(q, cs, s.values[o]))

    rows = []
    for gname, g in [("전체", t)] + [(k, t[t.group == k]) for k in RC_GROUPS]:
        mean_t = float(np.average(g.ticket, weights=g.cnt))
        for r in rates:
            for K in caps:
                disc = np.minimum(r * g.ticket, K)
                rows.append({"업종군": gname, "할인율": r, "상한(원)": K, "건당 결제액 평균(원)": round(mean_t), "건당 결제액 중앙값(원)": round(wq(g.ticket, g.cnt, 0.5)),
                             "기대 할인액(원/건)": round(float(np.average(disc, weights=g.cnt))), "상한에 걸리는 건 비중(%)": round(float(np.average(r * g.ticket > K, weights=g.cnt)) * 100, 1),
                             "실효 할인율(%)": round(float(np.average(disc, weights=g.cnt)) / mean_t * 100, 1), "결제 건수 비중(%)": round(g.cnt.sum() / t.cnt.sum() * 100, 1)})
    Cc = pd.DataFrame(rows)
    Cc.to_csv(C.TAB / "rc_ticket_cap.csv", index=False, encoding="utf-8-sig")
    q = {f"p{int(p*100)}": round(wq(t.ticket, t.cnt, p)) for p in [0.1, 0.25, 0.5, 0.75, 0.9]}
    print("\n=== C. 외지인 건당 결제액(원, 건수 가중 분위):", q)
    print(Cc[(Cc.할인율.isin([0.1, 0.2])) & (Cc["상한(원)"].isin([10000, 20000, np.inf]))].pivot_table(index=["업종군", "할인율"], columns="상한(원)", values=["기대 할인액(원/건)", "상한에 걸리는 건 비중(%)"]).to_string())
    return Cc, t


# ---------------------------------------------------------------- D. 2026 월별 기대 트리거 일수
def trigger_months(w, cal):
    cal26 = pd.read_csv(C.TAB / "a_expected_loss_calendar_2026.csv", parse_dates=["date"])
    c = cal26[cal26.region == "chuncheon"].copy()
    c["offday"] = (c.date.dt.dayofweek >= 5) | c.date.isin(HOLIDAYS_2026)
    c["month"] = c.date.dt.month
    skill = pd.read_csv(C.TAB / "f_forecast_skill.csv", encoding="utf-8-sig")
    sk = skill[skill.지역 == "춘천"].set_index("정의")
    mult = {"예보 5mm+": sk.loc["영업시간 비 10mm+ (예보 강수량 5mm+)", "예보 일수"] / sk.loc["영업시간 비 10mm+ (예보 강수량 5mm+)", "관측 일수"],
            "예보 5mm+ 또는 강수확률 60%+": sk.loc["영업시간 비 10mm+ (강수확률 60%+ 또는 강수량 5mm+)", "예보 일수"] / sk.loc["영업시간 비 10mm+ (강수확률 60%+ 또는 강수량 5mm+)", "관측 일수"]}
    wc = w[(w.region == "chuncheon") & w.date.isin(cal.date)].merge(cal[["date", "month", "is_offday"]], on="date")
    rows = []
    for mth, g in c.groupby("month"):
        act = wc[wc.month == mth]
        rows.append({"월": mth, "2026 기대 트리거 일수(관측 10mm 확정)": round(g.p_rain_biz10.sum(), 1), "그중 평일": round(g.loc[~g.offday, "p_rain_biz10"].sum(), 1), "그중 주말·공휴일": round(g.loc[g.offday, "p_rain_biz10"].sum(), 1),
                     "가발급 기대 일수(예보 5mm+)": round(g.p_rain_biz10.sum() * mult["예보 5mm+"], 1), "가발급 기대 일수(예보 5mm+ 또는 확률 60%+)": round(g.p_rain_biz10.sum() * mult["예보 5mm+ 또는 강수확률 60%+"], 1),
                     "2025 실제 10mm+ 일수": int(act.rain_biz10.sum()), "2025 실제 그중 주말·공휴일": int(act[act.is_offday == 1].rain_biz10.sum())})
    D = pd.DataFrame(rows)
    tot = D.drop(columns="월").sum(numeric_only=True).round(1)
    D = pd.concat([D, pd.DataFrame([{"월": "합계", **tot.to_dict()}])], ignore_index=True)
    D.to_csv(C.TAB / "rc_trigger_months.csv", index=False, encoding="utf-8-sig")
    print("\n=== D. 월별 기대 트리거 일수(2026, 기후학적 확률 모델) / 2025 실제"); print(D.to_string(index=False))
    return D, c, mult


# ---------------------------------------------------------------- E. 비용 모델: 쿠폰 1,000장당 비용·효과와 파라미터 격자
def cost_model(t, B, D, c26, ratio):
    g_all = t  # 외지인 레인체크 업종 셀-일 (건수 가중)
    tick_mean = float(np.average(g_all.ticket, weights=g_all.cnt))

    def exp_disc(r, K):
        """사용 1건당 기대 할인액. K는 금액(모든 업종 같은 상한) 또는 {업종군: 상한} 딕셔너리."""
        cap = g_all.group.map(K).values if isinstance(K, dict) else K
        return float(np.average(np.minimum(r * g_all.ticket, cap), weights=g_all.cnt))

    def per_1000(rate_wd, rate_we, cap, use_rate, uses, offday_share, deadweight):
        uses_total = 1000 * use_rate * uses
        disc = (1 - offday_share) * exp_disc(rate_wd, cap) + offday_share * exp_disc(rate_we, cap)
        cost = uses_total * disc
        sales = uses_total * tick_mean
        net = sales * (1 - deadweight)
        return cost, sales, net, disc

    rows = []
    grid = [(rw, re_, K) for rw in [0.10, 0.15, 0.20, 0.25, 0.30] for re_ in [0.0, 0.05, 0.10, 0.15, 0.20] for K in [5000, 10000, 15000, 20000, 30000, np.inf]]
    for rw, re_, K in grid:
        cost, sales, net, disc = per_1000(rw, re_, K, BASE["use_rate"], BASE["uses_per_coupon"], BASE["offday_use_share"], BASE["deadweight"])
        rows.append({"평일 할인율": rw, "주말 할인율": re_, "상한(원)": K, "사용 1건당 기대 할인액(원)": round(disc), "1,000장당 할인 비용(만원)": round(cost / 1e4, 1),
                     "1,000장당 재방문 매출(만원)": round(sales / 1e4, 1), "1,000장당 순증 매출(만원)": round(net / 1e4, 1), "순증÷비용": round(net / cost, 2) if cost else None,
                     "상한에 걸리는 평일 사용 비중(%)": round(float(np.average(rw * g_all.ticket > K, weights=g_all.cnt)) * 100, 1)})
    G = pd.DataFrame(rows)
    G.to_csv(C.TAB / "rc_cost_grid.csv", index=False, encoding="utf-8-sig")
    # 기본안과 가정 민감도
    base_cost, base_sales, base_net, base_disc = per_1000(BASE["rate_wd"], BASE["rate_we"], BASE["cap"], BASE["use_rate"], BASE["uses_per_coupon"], BASE["offday_use_share"], BASE["deadweight"])
    sens = [{"항목": "기본안(평일 20%·주말 10%·상한 1만원, 사용률 50%, 2회 사용, 주말 사용 30%, 사중손실 40%)", "1,000장당 비용(만원)": round(base_cost / 1e4, 1), "1,000장당 순증(만원)": round(base_net / 1e4, 1), "순증÷비용": round(base_net / base_cost, 2)}]
    for name, kw in [("사용률 30%", dict(use_rate=0.3)), ("사용률 70%", dict(use_rate=0.7)), ("사용 1회", dict(uses_per_coupon=1)), ("사용 3회", dict(uses_per_coupon=3)),
                     ("주말 사용 50%", dict(offday_use_share=0.5)), ("사중손실 30%", dict(deadweight=0.3)), ("사중손실 50%", dict(deadweight=0.5)),
                     ("상한 없음", dict(cap=np.inf)), ("상한 1.5만원", dict(cap=15000)), ("상한 2만원", dict(cap=20000)),
                     ("상한 업종별(식당·카페 1만원, 숙박·여가 2만원)", dict(cap=CAP_BY_GROUP)),
                     ("평일·주말 모두 15%", dict(rate_wd=0.15, rate_we=0.15)), ("평일 15%·주말 10%", dict(rate_wd=0.15)), ("평일 20%·주말 0%", dict(rate_we=0.0)),
                     ("추천안: 평일 20%·주말 10%, 상한 업종별, 주말 사용 30%", dict(cap=CAP_BY_GROUP))]:
        p = {**BASE, **kw}
        cost, sales, net, disc = per_1000(p["rate_wd"], p["rate_we"], p["cap"], p["use_rate"], p["uses_per_coupon"], p["offday_use_share"], p["deadweight"])
        sens.append({"항목": name, "1,000장당 비용(만원)": round(cost / 1e4, 1), "1,000장당 순증(만원)": round(net / 1e4, 1), "순증÷비용": round(net / cost, 2)})
    S = pd.DataFrame(sens)
    S.to_csv(C.TAB / "rc_cost_per_1000.csv", index=False, encoding="utf-8-sig")
    print("\n=== E. 쿠폰 1,000장당 비용·효과(만원)"); print(S.to_string(index=False))
    # 예산 → 발급 가능 장수와 월 한도, 발급 수요(상한) 비교
    trig = D[D.월 != "합계"].copy()
    trig["월"] = trig.월.astype(int)
    b = B.set_index("요일")
    demand = []
    for _, r in trig.iterrows():
        n_wd, n_we = r["그중 평일"], r["그중 주말·공휴일"]
        came = n_wd * b.loc["평일", "방문자 환산(실제, 명)"] + n_we * b.loc["주말·공휴일", "방문자 환산(실제, 명)"]
        lost = n_wd * b.loc["평일", "방문자 환산(사라진, 명)"] + n_we * b.loc["주말·공휴일", "방문자 환산(사라진, 명)"]
        came_card = n_wd * b.loc["평일", "트리거일 외지인 결제 건수(실제 평균)"] + n_we * b.loc["주말·공휴일", "트리거일 외지인 결제 건수(실제 평균)"]
        demand.append({"월": int(r.월), "기대 트리거 일수": r["2026 기대 트리거 일수(관측 10mm 확정)"], "트리거일 방문 외지인 합(명, 상한)": round(came), "사라진 방문 합(명, 상한)": round(lost), "트리거일 외지인 결제 건수 합(신한)": round(came_card)})
    Dm = pd.DataFrame(demand)
    cost_per_coupon = base_cost / 1000
    rows = []
    for budget in [0.5e8, 1e8, 2e8, 3e8]:
        n = budget / cost_per_coupon
        rows.append({"시즌 예산(억, 할인액만)": budget / 1e8, "쿠폰 1장당 기대 비용(원)": round(cost_per_coupon), "시즌 발급 가능(장)": round(n), "월 한도(장, 트리거 일수 비례 배분·최대 월)": round(n * trig["2026 기대 트리거 일수(관측 10mm 확정)"].max() / trig["2026 기대 트리거 일수(관측 10mm 확정)"].sum()),
                     "트리거일당 발급 가능(장)": round(n / trig["2026 기대 트리거 일수(관측 10mm 확정)"].sum()),
                     "참여율 10% 가정 시 발급 수요 대비(%)": round(n / (0.1 * Dm["트리거일 방문 외지인 합(명, 상한)"].sum()) * 100, 1),
                     "사라진 방문 전원 발급 대비(%)": round(n / Dm["사라진 방문 합(명, 상한)"].sum() * 100, 1)})
    Bud = pd.DataFrame(rows)
    Bud.to_csv(C.TAB / "rc_budget.csv", index=False, encoding="utf-8-sig")
    Dm.to_csv(C.TAB / "rc_demand_months.csv", index=False, encoding="utf-8-sig")
    # 월 한도: 예산을 기대 트리거 일수 비례로 나눈 값과 수요 시나리오
    alloc = trig[["월", "2026 기대 트리거 일수(관측 10mm 확정)"]].rename(columns={"2026 기대 트리거 일수(관측 10mm 확정)": "기대 트리거 일수"}).copy()
    share = alloc["기대 트리거 일수"] / alloc["기대 트리거 일수"].sum()
    for budget in [0.5e8, 1e8, 2e8, 3e8]:
        alloc[f"월 한도(장, 예산 {budget/1e8:g}억)"] = (budget / cost_per_coupon * share).round()
    alloc = alloc.merge(Dm[["월", "사라진 방문 합(명, 상한)", "트리거일 방문 외지인 합(명, 상한)"]], on="월")
    alloc["수요: 사라진 방문 × 플랫폼 연동 20%(명)"] = (alloc["사라진 방문 합(명, 상한)"] * 0.2).round()
    alloc["수요: 트리거일 방문 외지인 × 앱 인증 10%(명)"] = (alloc["트리거일 방문 외지인 합(명, 상한)"] * 0.1).round()
    alloc.to_csv(C.TAB / "rc_budget_months.csv", index=False, encoding="utf-8-sig")
    print("\n=== E2. 예산 → 발급 가능 장수(기본안)"); print(Bud.to_string(index=False))
    print("\n=== E3. 월별 발급 수요 상한(2026 기대 트리거 일수 × 트리거일 외지인 규모)"); print(Dm.to_string(index=False))
    print("\n=== E4. 월 한도(예산 비례)와 수요 시나리오"); print(alloc.to_string(index=False))
    pd.DataFrame(ASSUMPTIONS, columns=["가정", "기본값", "범위", "근거·메모"]).to_csv(C.TAB / "rc_assumptions.csv", index=False, encoding="utf-8-sig")
    return G, S, Bud, Dm, alloc, tick_mean, exp_disc


# ---------------------------------------------------------------- F. 유효기간
def validity(c26):
    tv = C.load("tour_monthly_visitors")
    tv["ym"] = tv.ym.astype(int)
    v = tv[(tv.ym >= 202501) & (tv.ym <= 202512)]
    v25 = v.set_index(v.ym % 100).visitors
    annual = v25.mean()
    p = c26.set_index("date").p_rain_biz10
    rows = []
    for m in range(7, 13):
        issue = pd.Timestamp(2026, m, 15)
        for weeks in [4, 6, 8, 12]:
            end = issue + pd.Timedelta(weeks=weeks)
            months = list(range(m, min(end.month if end.year == 2026 else 12, 12) + 1))
            cover = v25.reindex(months).mean() / annual * 100
            extra = p[(p.index > issue) & (p.index <= end)].sum()
            rows.append({"발급 월(15일 기준)": m, "유효기간(주)": weeks, "만료일": end.strftime("%m-%d"), "닿는 월": "~".join(f"{x}월" for x in [months[0], months[-1]]) if len(months) > 1 else f"{months[0]}월",
                         "10월(연중 최대 방문) 포함": "O" if 10 in months and (end >= pd.Timestamp(2026, 10, 1)) else "", "유효기간 월 방문자 평균(연평균=100)": round(cover, 0),
                         "유효기간 안 추가 트리거 기대 일수": round(float(extra), 1)})
    F = pd.DataFrame(rows)
    F.to_csv(C.TAB / "rc_validity.csv", index=False, encoding="utf-8-sig")
    print("\n=== F. 유효기간별 커버(2025 방문자 월 패턴, 2026 트리거 확률)"); print(F[F["유효기간(주)"].isin([6, 8, 12])].to_string(index=False))
    return F


# ---------------------------------------------------------------- G. 확정 문턱별 외지인 손실 커버
def threshold_cov(d):
    dd = d[d.dev_sales_nonlocal.notna()].copy()
    dd["loss"] = (dd.base_sales_nonlocal * (1 - np.exp(dd.dev_sales_nonlocal.clip(-1, 1)))).clip(lower=0)
    rain_any = dd[dd.rn_biz >= 1]
    total = rain_any.loss.sum()
    rows = []
    for thr in [1, 5, 10, 20, 30]:
        s = dd[dd.rn_biz >= thr]
        rows.append({"확정 문턱(영업시간 강수, mm)": thr, "해당 일수(2025 7~12월)": len(s), "그중 주말·공휴일": int(s.is_offday.sum()), "외지인 손실 합(억)": round(s.loss.sum() / 1e8, 2),
                     "비 온 날(1mm+) 외지인 손실 중 커버(%)": round(s.loss.sum() / total * 100, 1), "하루 평균 외지인 손실(억)": round(s.loss.mean() / 1e8, 2),
                     "손실 없는 날 비중(%)": round(float((s.dev_sales_nonlocal >= 0).mean() * 100), 0), "외지인 결제 건수 감소율 평균(%)": round(float(pct(s.dev_cnt_nonlocal.mean())), 1)})
    Gt = pd.DataFrame(rows)
    Gt.to_csv(C.TAB / "rc_threshold.csv", index=False, encoding="utf-8-sig")
    print("\n=== G. 확정 문턱별 외지인 손실 커버(레인체크 업종군)"); print(Gt.to_string(index=False))
    return Gt


# ---------------------------------------------------------------- H. 월 2회 발급 제한
def monthly_cap(c26, n_sim=20000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, g in c26.groupby("month"):
        p = g.p_rain_biz10.values
        sim = (rng.random((n_sim, len(p))) < p).sum(axis=1)
        for cap in [1, 2, 3]:
            rows.append({"월": m, "기대 트리거 일수": round(p.sum(), 1), "P(트리거 3일 이상)": round(float((sim >= 3).mean()), 2), "월 발급 상한(회)": cap,
                         "상한이 깎는 발급 비중(%, 매일 등록한 사람 기준)": round((1 - np.minimum(sim, cap).mean() / sim.mean()) * 100, 1)})
    H = pd.DataFrame(rows)
    H.to_csv(C.TAB / "rc_monthly_cap.csv", index=False, encoding="utf-8-sig")
    print("\n=== H. 월 발급 상한이 깎는 비중"); print(H[H["월 발급 상한(회)"] == 2].to_string(index=False))
    return H


# ---------------------------------------------------------------- 그림
def figures(A, Cc, t, D, G, alloc, exp_disc, tick_mean):
    plt.rcParams.update(RC)
    # rc01: 민감도 평일/주말 × 거주지(매출·건수), 춘천 레인체크 업종군
    a = A[(A.범위 == "레인체크 업종군(음식·카페·숙박·여가)") & (A.지역 == "춘천")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, ind in zip(axes, ["매출", "결제 건수"]):
        s = a[a.지표 == ind]
        xs = np.arange(2)
        for j, (org, color) in enumerate([("현지인", MUTED), ("외지인", REGION_COLOR["chuncheon"])]):
            v = s[s.거주지 == org].set_index("요일").reindex(["평일", "주말·공휴일"])
            ax.bar(xs + (j - 0.5) * 0.36, v.pct, width=0.34, color=color, label=org)
            ax.errorbar(xs + (j - 0.5) * 0.36, v.pct, yerr=np.vstack([v.pct - v.lo, v.hi - v.pct]), fmt="none", ecolor=INK, elinewidth=1, capsize=3)
            for xx, (pp, ne) in zip(xs + (j - 0.5) * 0.36, zip(v.pct, v.n_event_max)):
                ax.text(xx, min(pp, 0) - 1.5, f"{pp:.0f}%\n(n={int(ne)})", ha="center", va="top", fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xs, ["평일 비 10mm+", "주말·공휴일 비 10mm+"]); ax.set_ylabel(f"{ind} 변화(%)"); ax.set_title(f"춘천 {ind}: 비 온 날의 변화")
        ax.set_ylim(min(s.lo.min() - 6, -20), max(s.hi.max() + 4, 8)); ax.grid(axis="x", visible=False); ax.legend(loc="upper right")
    finish(fig, C.FIG, "rc01_sensitivity_dow", "레인체크 업종군(음식·카페·숙박·여가) 비 10mm+ 민감도: 거주지 × 요일 (업종군별 회귀의 역분산 합성, 95% 구간)")
    # rc02: 건당 결제액 분포와 상한별 기대 할인액
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    ax = axes[0]
    for gname, color in zip(RC_GROUPS, ["#eb6834", "#f3a683", "#2a78d6", "#104281"]):
        g = t[t.group == gname]
        o = np.argsort(g.ticket.values); cs = np.cumsum(g.cnt.values[o]) / g.cnt.sum()
        ax.plot(g.ticket.values[o] / 1e4, cs * 100, color=color, lw=1.6, label=f"{gname} (평균 {np.average(g.ticket, weights=g.cnt)/1e4:.1f}만원)")
    ax.axvline(5, color=INK2, lw=0.8, ls="--"); ax.text(5.1, 5, "20% 할인이 1만원 상한에\n걸리는 결제액(5만원)", fontsize=8, color=INK2, va="bottom")
    ax.set_xlim(0, 15); ax.set_xlabel("건당 결제액(만원)"); ax.set_ylabel("누적 결제 건수 비중(%)"); ax.set_title("외지인 건당 결제액 분포(셀 평균, 건수 가중)"); ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    rates = np.linspace(0.05, 0.30, 26)
    for K, color, lab in [(np.inf, INK, "상한 없음"), (20000, "#2a78d6", "상한 2만원"), (10000, REGION_COLOR["chuncheon"], "상한 1만원"), (5000, MUTED, "상한 5천원")]:
        ax.plot(rates * 100, [exp_disc(r, K) for r in rates], color=color, lw=1.6, label=lab)
    ax.set_xlabel("할인율(%)"); ax.set_ylabel("사용 1건당 기대 할인액(원)"); ax.set_title(f"할인율 × 상한 → 건당 기대 할인액 (평균 결제액 {tick_mean/1e4:.1f}만원)"); ax.legend(loc="upper left", fontsize=8)
    finish(fig, C.FIG, "rc02_ticket_cap", "상한은 20% 할인에서 숙박·여가 결제에만 걸리고 식당·카페는 거의 영향이 없다")
    # rc03: 월별 기대 트리거 일수
    d = D[D.월 != "합계"].copy(); d["월"] = d.월.astype(int)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    xs = np.arange(len(d))
    ax.bar(xs - 0.3, d["2026 기대 트리거 일수(관측 10mm 확정)"], width=0.28, color=REGION_COLOR["chuncheon"], label="2026 기대(관측 10mm 확정)")
    ax.bar(xs, d["가발급 기대 일수(예보 5mm+)"], width=0.28, color="#f3a683", label="2026 기대 가발급(예보 5mm+)")
    ax.bar(xs + 0.3, d["2025 실제 10mm+ 일수"], width=0.28, color=MUTED, label="2025 실제(관측 10mm+)")
    for x, r in zip(xs, d.itertuples()):
        ax.text(x - 0.3, r[2] + 0.15, f"{r[2]:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xs, [f"{m}월" for m in d.월]); ax.set_ylabel("일수"); ax.set_title("춘천 월별 트리거 일수: 기후학적 기대값과 2025년 실제"); ax.grid(axis="x", visible=False); ax.legend()
    finish(fig, C.FIG, "rc03_trigger_months", "가발급(예보)은 확정(관측)의 약 1.7배. 7~9월에 몰린다")
    # rc04: 왼쪽 파라미터 격자(평일 할인율 × 상한 → 1,000장당 비용), 오른쪽 월 한도(예산 비례) vs 발급 수요 시나리오
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    g = G[G["주말 할인율"] == 0.10]
    for K, color, lab in [(np.inf, INK, "상한 없음"), (20000, "#2a78d6", "상한 2만원"), (10000, REGION_COLOR["chuncheon"], "상한 1만원"), (5000, MUTED, "상한 5천원")]:
        s = g[g["상한(원)"] == K].sort_values("평일 할인율")
        ax.plot(s["평일 할인율"] * 100, s["1,000장당 할인 비용(만원)"], "-o", color=color, lw=1.5, ms=4, label=lab)
    base = g[(g["상한(원)"] == 10000) & (g["평일 할인율"] == 0.20)].iloc[0]
    ax.scatter([20], [base["1,000장당 할인 비용(만원)"]], s=130, facecolors="none", edgecolors="#c0392b", lw=1.8, zorder=5)
    ax.annotate(f"회의안: 1,000장당 {base['1,000장당 할인 비용(만원)']:.0f}만원\n(장당 {base['1,000장당 할인 비용(만원)']*10:.0f}원)", (20, base["1,000장당 할인 비용(만원)"]), fontsize=8, color="#c0392b", xytext=(-95, 18), textcoords="offset points")
    ax.set_xlabel("평일 할인율(%)  (주말 10%, 사용률 50%, 2회 사용, 주말 사용 30%)"); ax.set_ylabel("쿠폰 1,000장당 할인 비용(만원)"); ax.set_title("할인율·상한 → 쿠폰 1,000장당 비용"); ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    xs = np.arange(len(alloc))
    ax.bar(xs - 0.3, alloc["수요: 트리거일 방문 외지인 × 앱 인증 10%(명)"] / 1e3, width=0.28, color="#f3a683", label="수요: 트리거일 방문 외지인 × 앱 인증 10%")
    ax.bar(xs, alloc["수요: 사라진 방문 × 플랫폼 연동 20%(명)"] / 1e3, width=0.28, color=REGION_COLOR["chuncheon"], label="수요: 사라진 방문 × 플랫폼 연동 20%")
    ax.bar(xs + 0.3, alloc["월 한도(장, 예산 1억)"] / 1e3, width=0.28, color=INK, label="월 한도: 시즌 예산 1억(할인액)")
    for k, ls, lab in [("월 한도(장, 예산 2억)", "--", "예산 2억"), ("월 한도(장, 예산 3억)", ":", "예산 3억")]:
        ax.plot(xs + 0.3, alloc[k] / 1e3, ls, color=INK, lw=1, marker="_", ms=9, label=f"월 한도: {lab}")
    ax.set_xticks(xs, [f"{int(m)}월" for m in alloc.월]); ax.set_ylabel("천 장(명)"); ax.set_title("월 한도(예산을 트리거 일수 비례로 배분) vs 발급 수요 시나리오"); ax.grid(axis="x", visible=False); ax.legend(fontsize=7.5)
    finish(fig, C.FIG, "rc04_cost_grid", "순증 매출은 사용률·사용 횟수·결제액으로 정해지고 할인율·상한은 비용만 바꾼다. 수요는 예산을 크게 넘으므로 월 한도가 필수다")


def main():
    cal, excl, w, sc, rs = load_all()
    A, rl = sensitivity_dow(rs, cal, w)
    B, d, ratio, m = trigger_pool(rs, cal, excl, w)
    Cc, t = ticket_cap(sc)
    D, c26, mult = trigger_months(w, cal)
    G, S, Bud, Dm, alloc, tick_mean, exp_disc = cost_model(t, B, D, c26, ratio)
    F = validity(c26)
    Gt = threshold_cov(d)
    H = monthly_cap(c26)
    figures(A, Cc, t, D, G, alloc, exp_disc, tick_mean)
    print("\n저장: rc_sensitivity_dow, rc_trigger_day_pool, rc_ticket_cap, rc_trigger_months, rc_cost_per_1000, rc_cost_grid, rc_budget, rc_budget_months, rc_demand_months, rc_validity, rc_threshold, rc_monthly_cap, rc_assumptions; 그림 rc01~rc04")


if __name__ == "__main__":
    main()
