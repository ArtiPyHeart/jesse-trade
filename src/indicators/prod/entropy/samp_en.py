import numpy as np
from jesse.helpers import get_candle_source

from pyrs_indicators.util_entropy import sample_entropy


def sample_entropy_indicator(
    candles: np.ndarray,
    period: int = 32,
    use_array_price: bool = False,
    source_type: str = "close",
    sequential: bool = False,
):
    """Sample Entropy (SampEn) 指标

    使用 Rust 高性能计算，相比 Python numba 性能提升 10-50x。

    Args:
        candles: OHLCV K线数据
        period: 滑动窗口大小 (default: 32)
        use_array_price: log return 计算方式 (default: False)
            - True: 相邻收益率 log(p[i]/p[i-1])
            - False: 相对当前价格 log(p_current/p[i])
        source_type: 价格来源 (default: "close")
        sequential: 是否返回全序列 (default: False)

    Returns:
        sequential=True: 完整熵值数组，前 period 个为 NaN
        sequential=False: 单个熵值的数组
    """
    src = get_candle_source(candles, source_type).astype(np.float64)
    n = len(src)

    if sequential:
        result = np.full(n, np.nan)
        for i in range(period, n):
            if use_array_price:
                # 相邻收益率: log(p[i-period+1:i] / p[i-period:i-1])
                log_ret = np.log(src[i - period + 1 : i] / src[i - period : i - 1])
            else:
                # 相对当前价格: log(p[i] / p[i-period:i])
                log_ret = np.log(src[i] / src[i - period : i])

            result[i] = sample_entropy(
                log_ret,
                m=2,
                r_ratio=0.3,
                mode="range",
            )
        return result
    else:
        # 单点计算
        if use_array_price:
            log_ret = np.log(src[-period:-1] / src[-period - 1 : -2])
        else:
            log_ret = np.log(src[-1] / src[-period - 1 : -1])

        entropy = sample_entropy(
            log_ret,
            m=2,
            r_ratio=0.3,
            mode="range",
        )
        return np.array([entropy])
