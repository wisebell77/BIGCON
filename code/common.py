"""경로, 공통 상수, 저장·불러오기 함수."""
import os
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 콘솔에서 한글 출력이 깨지지 않게

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
META = CODE / "meta"
RAW_CARD = ROOT / "신한카드_데이터_레이아웃"
RAW_SK = ROOT / "sk통신인구_데이터_레이아웃"
EXT_WEATHER = ROOT / "data" / "external" / "weather"
OUT = ROOT / "data" / "processed"
# 큰 외부 원본(서울 생활인구, 상권분석서비스, 온열질환)은 프로젝트 밖에 둔다. 환경변수 BIGCON_EXT로 위치를 바꿀 수 있다.
EXT_BIG = Path(os.environ.get("BIGCON_EXT", r"D:\Bigcon_data"))
EXT_SEOUL_SALES = EXT_BIG / "서울열린데이터광장" / "상권분석서비스"  # 추정매출-행정동 연도별 zip + API 원본
EXT_SEOUL_POP = EXT_BIG / "서울열린데이터광장" / "행정동별서울생활인구(250m)"  # 월별 zip
EXT_HEAT_ILL = EXT_BIG / "공공데이터포털" / "질병관리청_온열질환 감시 데이터_20250925.CSV"
EXT_KMA = EXT_BIG / "기상청API허브"  # 단기예보 격자 원본(gzip 텍스트)
EXT_TOUR = EXT_BIG / "한국관광데이터랩"  # 지역별 데이터랩 내려받기(춘천시, 월 단위)
GANGNAM_ADSTRD = "11680"  # 행정안전부 행정기관코드 앞 5자리 = 강남구
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

CARD1 = RAW_CARD / "신한카드_빅콘테스트2026_데이터1.txt"  # 주제1용: 시간대 있음, 법인 포함
CARD2 = RAW_CARD / "신한카드_빅콘테스트2026_데이터2.txt"  # 주제2용 레이아웃: 고객 거주 시도 있음, 법인 제외

DATES = pd.date_range("2025-07-01", "2025-12-31", freq="D")

# 지역 키를 하나로 맞춘다: 카드(시군구명), SK(통계청 시군구 코드), 기상청(지점번호)
REGIONS = ["gangnam", "chuncheon"]
REGION_NAME = {"gangnam": "강남", "chuncheon": "춘천"}
REGION_BY_CARD = {"서울 강남구": "gangnam", "강원 춘천시": "chuncheon"}
REGION_BY_SGG = {"11230": "gangnam", "32010": "chuncheon"}
# 지역 날씨에 쓸 관측소. 앞 지점 값을 먼저 쓰고, 빈 시간과 없는 요소(일조·적설·운량)는 뒤 지점으로 채운다.
# 400 강남 AWS(강남구 안), 108 서울 ASOS(종로구 송월동), 101 춘천 ASOS
REGION_STATIONS = {"gangnam": [400, 108], "chuncheon": [101]}
REGION_ASOS = {"gangnam": 108, "chuncheon": 101}  # 공식 일자료(일최고·최저기온, 일강수량)를 붙일 ASOS 지점
LOCAL_SIDO = {"gangnam": "서울", "chuncheon": "강원"}  # 데이터2 거주지가 이 시도면 '지역민'

TIME_BANDS = ["00_05", "06_11", "12_17", "18_23"]
TB_NAME = {"00_05": "0~6시", "06_11": "6~12시", "12_17": "12~18시", "18_23": "18~24시"}
AGE_GROUP = {10: "10-20대", 20: "10-20대", 30: "30-50대", 40: "30-50대", 50: "30-50대",
             60: "60대+", 70: "60대+", 80: "60대+", 90: "60대+"}
AGE_GROUPS = ["10-20대", "30-50대", "60대+"]

# 대면소비 업종군: 손님이 매장에 와야 매출이 생기는 업종. 취약 상권 분석의 기본 범위.
FACE_GROUPS = ["음식점", "카페·베이커리", "주점·유흥", "식품·생필품", "패션·잡화", "생활·기타소매",
               "대형유통", "숙박", "여가·스포츠", "개인서비스"]

# 본 분석 이벤트(팀 결정: 폭염·열대야·호우 중심, 한파는 참고). 열 이름은 weather_daily의 플래그.
EVENTS = {
    "heat_day": "폭염일(일최고기온 33℃ 이상)",
    "tropical": "열대야(밤 최저기온 25℃ 이상)",
    "rain_biz10": "호우(영업시간 06~24시 강수 10mm 이상)",
    "rain_biz20": "호우(영업시간 강수 20mm 이상)",
    "cold12": "한파(일최저 -12℃ 이하, 참고)",
}


def save(df, name, out_dir=OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  저장: {path.name}  {df.shape[0]:,}행 × {df.shape[1]}열")


def load(name):
    return pd.read_parquet(OUT / f"{name}.parquet")


def industry_map():
    return pd.read_csv(META / "industry_map.csv", encoding="utf-8-sig")


def seoul_api_key():
    """프로젝트 루트 .env의 openapi_seoul 값을 돌려준다. 값은 어디에도 출력하지 않는다."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("openapi_seoul")
    if not key:
        raise RuntimeError("인증키가 없다. 프로젝트 루트 .env에 openapi_seoul=... 를 넣어라.")
    return key


def kma_api_key():
    """프로젝트 루트 .env의 weather_api(기상청 API허브 인증키)를 돌려준다. 값은 어디에도 출력하지 않는다."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("weather_api")
    if not key:
        raise RuntimeError("기상청 API허브 키가 없다. 프로젝트 루트 .env에 weather_api=... 를 넣어라.")
    return key
