import numpy as np
from jesse.helpers import get_candle_source
from .fracdiff_fn import fracdiff
from .find_truncation import find_truncation
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
    
    # 当 sequential=False 时，只需要计算最后的值
    if not sequential:
        # 获取权重长度来确定需要多少历史数据
        truncation, _ = find_truncation(frac, tau=1e-5, mmax=20000)
        # 保留足够的历史数据用于计算，加上一些buffer确保准确性
        window_size = min(len(candles), truncation + lag + 10)
        candles_subset = candles[-window_size:]
        
        # 计算截断后的数据
        if source_type == minus_type:
            src = np.log(get_candle_source(candles_subset, source_type=source_type))
            minus = np_shift(src, lag)
        else:
            src = np.log(get_candle_source(candles_subset, source_type=source_type))
            minus = np_shift(
                np.log(get_candle_source(candles_subset, source_type=minus_type)), lag
            )
        
        raw = src - minus
        fracdiff_res = fracdiff(raw, order=frac)
        return fracdiff_res[-1:]
    
    # sequential=True 时，计算完整序列
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
    
    return fracdiff_res
