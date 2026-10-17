"""전처리 전체 실행: python code/run_all.py"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEPS = ["01_calendar.py", "02_card.py", "03_sk.py", "04_weather.py", "05_panel.py", "06_weather_long.py", "07_warnings.py", "08_stores.py",
         "09_seoul_external.py", "09b_kma_forecast.py", "09c_tourism.py"]  # 09*는 D 드라이브 원본(common.EXT_BIG)이 있을 때만 돈다. API는 보관된 원본이 있으면 다시 부르지 않는다

for step in STEPS:
    t0 = time.time()
    print(f"\n===== {step}")
    sys.argv = [step]  # 04_weather.py의 인자 파싱이 run_all 인자를 읽지 않게
    if step.startswith("09"):
        import common as C
        need = {"09_seoul_external.py": C.EXT_SEOUL_SALES, "09b_kma_forecast.py": C.EXT_KMA / "dfs_shrt_grd", "09c_tourism.py": C.EXT_TOUR}[step]
        if not need.exists():
            print("  외부 원본 폴더가 없어 건너뜀:", need)
            continue
        if step.startswith("09_") and (C.EXT_SEOUL_SALES / "서울시 상권분석서비스(추정매출-자치구)_API_전체.csv").exists():
            sys.argv.append("--no-api")
        if step.startswith("09b"):
            sys.argv.append("--build")  # 격자 예보 수집(약 5,900회 호출, 80분)은 한 번만 직접 돌리고, 여기서는 보관된 원본으로 정리만 한다
    runpy.run_path(str(HERE / step), run_name="__main__")
    print(f"===== {step} 끝 ({time.time() - t0:.0f}초)")
