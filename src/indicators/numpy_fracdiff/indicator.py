import numpy as np
from jesse.helpers import get_candle_source
from .fracdiff_fn import fracdiff
from ...utils.math_tools import np_shift


def np_fracdiff(
    candles: np.ndarray,
    sequential=False,
    source_type: str = "close",
    minus_type: str = "close",
    frac: float = 0.7,
    lag: int = 1,
) -> np.ndarray:
    # 参数验证
    if frac < 0 or frac > 1:
        raise ValueError(f"frac must be between 0 and 1, got {frac}")
    
    # 避免重复计算
    if source_type == minus_type:
        src = np.log(get_candle_source(candles, source_type=source_type))
        minus = np_shift(src, lag)
    else:
        src = np.log(get_candle_source(candles, source_type=source_type))
        minus = np_shift(
            np.log(get_candle_source(candles, source_type=minus_type)), lag
        )
    
    raw = src - minus
    fracdiff_res = fracdiff(raw, order=frac)
    
    return fracdiff_res if sequential else fracdiff_res[-1:]
