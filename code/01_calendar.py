"""달력 피처: 요일, 공휴일, 연휴, 추석, 정책 이벤트(소비쿠폰·상생페이백).

공휴일과 정책 일정은 정부 발표로 확인한 값이다.
- 추석 연휴 10/3~10/9(10/8 대체공휴일), 10/10은 임시공휴일 지정 안 됨
- 민생회복 소비쿠폰: 1차 지급 7/21, 2차 지급 9/22, 사용 기한 11/30
  (연매출 30억 이하 소상공인 가맹점, 주소지 지자체 안에서만 사용)
- 상생페이백: 9~11월 카드 소비 증가분 20% 환급(월 10만원 한도), 12월까지 연장(월 3만원 한도)
"""
import numpy as np
import pandas as pd

from common import DATES, save

HOLIDAYS = {
    "2025-08-15": "광복절",
    "2025-10-03": "개천절",
    "2025-10-05": "추석 연휴",
    "2025-10-06": "추석",
    "2025-10-07": "추석 연휴",
    "2025-10-08": "대체공휴일",
    "2025-10-09": "한글날",
    "2025-12-25": "성탄절",
}
SPECIAL_DAYS = {"2025-11-13": "수능", "2025-12-24": "크리스마스이브", "2025-12-31": "연말"}
CHUSEOK = pd.Timestamp("2025-10-06")
COUPON_1ST, COUPON_2ND, COUPON_END = map(pd.Timestamp, ["2025-07-21", "2025-09-22", "2025-11-30"])
PAYBACK_START, PAYBACK_END = map(pd.Timestamp, ["2025-09-01", "2025-12-31"])


def run_lengths(flag):
    """연속 구간마다 (길이, 구간 안 순번)을 돌려준다."""
    block_id = (flag != flag.shift()).cumsum()
    length = flag.groupby(block_id).transform("size")
    pos = flag.groupby(block_id).cumcount() + 1
    return length, pos


def build_calendar():
    cal = pd.DataFrame({"date": DATES})
    d = cal["date"]
    cal["ym"] = d.dt.strftime("%Y%m")
    cal["month"] = d.dt.month
    cal["day"] = d.dt.day
    cal["dow"] = d.dt.dayofweek  # 0=월 … 6=일
    cal["week"] = d.dt.isocalendar().week.astype(int)
    cal["days_to_month_end"] = d.dt.days_in_month - d.dt.day

    cal["holiday_name"] = d.dt.strftime("%Y-%m-%d").map(HOLIDAYS).fillna("")
    cal["is_holiday"] = (cal["holiday_name"] != "").astype(int)
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["is_offday"] = ((cal["is_holiday"] == 1) | (cal["is_weekend"] == 1)).astype(int)

    # 쉬는 날 묶음: 길이와 순번(근무일은 0)
    off = cal["is_offday"].astype(bool)
    length, pos = run_lengths(off)
    cal["off_block_len"] = np.where(off, length, 0)
    cal["off_block_pos"] = np.where(off, pos, 0)
    long_off = pd.Series(np.where(off, length, 0) >= 3, index=cal.index)
    cal["is_pre_long_off"] = ((~off) & long_off.shift(-1, fill_value=False)).astype(int)   # 3일 이상 연휴 전날
    cal["is_post_long_off"] = ((~off) & long_off.shift(1, fill_value=False)).astype(int)   # 3일 이상 연휴 다음 날
    cal["is_sandwich"] = ((~off) & off.shift(1, fill_value=False) & off.shift(-1, fill_value=False)).astype(int)

    cal["chuseok_rel"] = (d - CHUSEOK).dt.days.where(lambda s: s.abs() <= 14)  # 추석 기준 D-14 ~ D+14
    cal["is_chuseok_block"] = d.between("2025-10-03", "2025-10-09").astype(int)

    cal["coupon_phase"] = np.select(
        [d < COUPON_1ST, d < COUPON_2ND, d <= COUPON_END], [0, 1, 2], default=3
    )  # 0 지급 전, 1 1차 지급 이후, 2 2차 지급 이후, 3 사용 기한 종료 후
    cal["coupon_days_left"] = (COUPON_END - d).dt.days.where(d.between(COUPON_1ST, COUPON_END))
    cal["is_coupon_last_week"] = d.between(COUPON_END - pd.Timedelta(days=6), COUPON_END).astype(int)
    cal["payback_on"] = d.between(PAYBACK_START, PAYBACK_END).astype(int)
    cal["special_day"] = d.dt.strftime("%Y-%m-%d").map(SPECIAL_DAYS).fillna("")
    cal["is_special"] = (cal["special_day"] != "").astype(int)
    return cal


if __name__ == "__main__":
    cal = build_calendar()
    print("공휴일:", cal.loc[cal.is_holiday == 1, ["date", "holiday_name"]].assign(date=lambda x: x.date.dt.strftime("%m-%d")).values.tolist())
    print("3일 이상 연휴 전날:", cal.loc[cal.is_pre_long_off == 1, "date"].dt.strftime("%m-%d").tolist(),
          "/ 다음 날:", cal.loc[cal.is_post_long_off == 1, "date"].dt.strftime("%m-%d").tolist(),
          "/ 징검다리:", cal.loc[cal.is_sandwich == 1, "date"].dt.strftime("%m-%d").tolist())
    save(cal, "calendar")
