"""Hurst-ADF-KPSS 三重趋势验证器

通过Hurst指数、ADF检验、KPSS检验三重验证判断市场趋势性。

用法:
    from research.hurst_adf_kpss import TrendValidator

    validator = TrendValidator(window_size=40, step=5)
    results = validator.validate(fusion_bars)
    summary = validator.summarize(results)
"""

from .hurst import calculate_hurst, get_lag_params
from .stationarity import run_adf_test, run_kpss_test
from .trend_validator import TrendValidator

__all__ = [
    "TrendValidator",
    "calculate_hurst",
    "get_lag_params",
    "run_adf_test",
    "run_kpss_test",
]
