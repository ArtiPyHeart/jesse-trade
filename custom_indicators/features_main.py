import jesse.indicators as ta
import numpy as np
from jesse import helpers
from numba import jit

from custom_indicators.td_sequential import td_sequential


@jit(nopython=True, fastmath=True)
def _dt(array):
    return np.diff(array, prepend=np.nan)


@jit(nopython=True, fastmath=True)
def _std(array, n=20):
    # 使用cumsum方法创建rolling window
    ret = np.full_like(array, np.nan)

    # 计算移动窗口的标准差
    for i in range(n - 1, len(array)):
        ret[i] = np.std(array[i - n + 1 : i + 1], ddof=1)
    return ret


@jit(nopython=True, fastmath=True)
def _skew(array, n=20):
    ret = np.full_like(array, np.nan)

    for i in range(n - 1, len(array)):
        window = array[i - n + 1 : i + 1]
        # 计算中心矩
        m3 = np.mean((window - np.mean(window)) ** 3)
        std = np.std(window, ddof=1)
        # 偏度计算公式
        ret[i] = m3 / (std**3) if std != 0 else 0
    return ret


@jit(nopython=True, fastmath=True)
def _kurtosis(array, n=20):
    ret = np.full_like(array, np.nan)

    for i in range(n - 1, len(array)):
        window = array[i - n + 1 : i + 1]
        # 计算中心矩
        m4 = np.mean((window - np.mean(window)) ** 4)
        var = np.var(window, ddof=1)
        # 峰度计算公式
        ret[i] = (m4 / var**2) - 3 if var != 0 else 0
    return ret


def feature_matrix(candles: np.array, sequential: bool = False):
    candles = helpers.slice_candles(candles, sequential)
    close = helpers.get_candle_source(candles, "close")

    st_trend, st_changed = ta.supertrend(candles, sequential=sequential)
    fe_supertrend = close / st_trend

    kama = ta.kama(candles, sequential=sequential)
    fe_kama = close / kama

    lrsi = ta.lrsi(candles, sequential=sequential)

    boll_upper, boll_middle, boll_lower = ta.bollinger_bands(
        candles, sequential=sequential
    )
    fe_boll_upper = close / boll_upper
    fe_boll_middle = close / boll_middle
    fe_boll_lower = close / boll_lower

    boll_width = ta.bollinger_bands_width(candles, sequential=sequential)

    atr = ta.atr(candles, sequential=sequential)
    vwap = ta.vwap(candles, anchor="h", sequential=sequential)
    fe_vwap = close / vwap

    sar = ta.sar(candles, sequential=sequential)
    fe_sar = close / sar

    td_buy, td_sell = td_sequential(candles, sequential=sequential)

    final_fe = np.concatenate(
        [
            i.reshape(-1, 1)
            for i in [
                fe_supertrend,
                fe_kama,
                lrsi,
                fe_boll_upper,
                fe_boll_middle,
                fe_boll_lower,
                boll_width,
                atr,
                fe_vwap,
                fe_sar,
                td_buy,
                td_sell,
            ]
        ],
        axis=1,
    )

    if sequential:
        return final_fe
    else:
        return final_fe[-1, :].reshape(1, -1)
