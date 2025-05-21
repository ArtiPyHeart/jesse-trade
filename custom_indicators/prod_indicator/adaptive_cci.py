import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math_tools import deg_cos, deg_sin


@njit
def _calculate_adaptive_cci_numba(
    filt: np.ndarray,
    length: int,
    c1: float,
    c2: float,
    c3: float,
    period_min: int,
    period_max: int,
    max_lag: int,
) -> np.ndarray:
    """
    使用 numba 加速的核心计算逻辑，返回自适应CCI序列。
    """
    # 相关系数数组
    corr_array = np.zeros((length, max_lag + 1))
    # 计算相关系数
    for i in range(length):
        for lag in range(max_lag + 1):
            M = 3
            if i - lag - (M - 1) < 0:
                continue
            sx = sy = sxx = syy = sxy = 0.0
            for count in range(M):
                x = filt[i - count]
                y = filt[i - lag - count]
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
            denom = (M * sxx - sx * sx) * (M * syy - sy * sy)
            if denom > 0:
                corr_array[i, lag] = (M * sxy - sx * sy) / np.sqrt(denom)

    # 计算 CosinePart, SinePart, SqSum
    cosine_part = np.zeros((length, period_max + 1))
    sine_part = np.zeros((length, period_max + 1))
    sq_sum = np.zeros((length, period_max + 1))
    for i in range(length):
        for period in range(period_min, period_max + 1):
            cos_acc = 0.0
            sin_acc = 0.0
            for n in range(3, max_lag + 1):
                val = corr_array[i, n]
                cos_acc += val * deg_cos(360.0 * n / period)
                sin_acc += val * deg_sin(360.0 * n / period)
            cosine_part[i, period] = cos_acc
            sine_part[i, period] = sin_acc
            sq_sum[i, period] = cos_acc * cos_acc + sin_acc * sin_acc

    # R, Pwr 计算
    R = np.zeros((length, period_max + 1, 2))
    pwr = np.zeros((length, period_max + 1))
    max_pwr_array = np.zeros(length)

    for i in range(length):
        for period in range(period_min, period_max + 1):
            old_val = R[i - 1, period, 1] if i > 0 else 0.0
            R[i, period, 1] = (
                0.2 * sq_sum[i, period] * sq_sum[i, period] + 0.8 * old_val
            )

        if i > 0:
            max_pwr_array[i] = 0.991 * max_pwr_array[i - 1]
        for period in range(period_min, period_max + 1):
            if R[i, period, 1] > max_pwr_array[i]:
                max_pwr_array[i] = R[i, period, 1]

        for period in range(3, period_max + 1):
            if max_pwr_array[i] != 0:
                pwr[i, period] = R[i, period, 1] / max_pwr_array[i]

    # 主导周期
    dominant_cycle = np.zeros(length)
    for i in range(length):
        spx = 0.0
        sp = 0.0
        for period in range(period_min, period_max + 1):
            if pwr[i, period] >= 0.5:
                spx += period * pwr[i, period]
                sp += pwr[i, period]
        if sp != 0.0:
            dc = spx / sp
            if dc < period_min:
                dc = period_min
            elif dc > period_max:
                dc = period_max
        else:
            dc = 10.0
        dominant_cycle[i] = dc

    # 计算自适应CCI
    mycci = np.zeros(length)
    for i in range(2, length):
        dc = int(dominant_cycle[i])
        if dc < 2:
            dc = 2
        start_idx = i - dc + 1 if (i - dc + 1) >= 0 else 0
        window = filt[start_idx : i + 1]
        ave_price = np.mean(window)
        rms = np.sqrt(np.mean((window - ave_price) ** 2))

        denom = 0.015 * rms
        ratio = (filt[i] - ave_price) / denom if denom != 0 else 0.0

        prev_m1 = mycci[i - 1] if i > 0 else 0.0
        prev_m2 = mycci[i - 2] if i > 1 else 0.0

        mycci[i] = c1 * (ratio + prev_m1) / 2.0 + c2 * prev_m1 + c3 * prev_m2

    return mycci


def adaptive_cci(
    candles: np.ndarray, source_type: str = "close", sequential: bool = False
) -> float | np.ndarray:
    """
    自适应CCI (Adaptive CCI) 指标实现基于 Ehlers 方法:
    1. 高通滤波去除短周期噪声
    2. 超级平滑滤波
    3. 相关系数累积计算获取主导周期 (DominantCycle)
    4. 基于主导周期自适应计算CCI

    :param candles: np.ndarray
    :param source_type: str
    :param sequential: bool
    :return: np.ndarray 或 float
    """
    # 截取或完整返回数据
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    # 若数据过短，直接返回零值或最后一个值
    length = len(src)
    if length < 3:
        if sequential:
            return np.zeros_like(src, dtype=float)
        else:
            return 0.0

    # ------------------------------
    # Step 1: 高通滤波 (Highpass Filter)
    # ------------------------------
    # 使用 deg_cos 与 deg_sin 计算角度制的三角函数
    alpha1 = (
        deg_cos(0.707 * 360.0 / 48.0) + deg_sin(0.707 * 360.0 / 48.0) - 1.0
    ) / deg_cos(0.707 * 360.0 / 48.0)

    hp = np.zeros(length)
    hp[0:2] = src[0:2]  # 初始化前2个值

    for i in range(2, length):
        hp[i] = (
            (1 - alpha1 / 2) * (1 - alpha1 / 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
            + 2 * (1 - alpha1) * hp[i - 1]
            - (1 - alpha1) * (1 - alpha1) * hp[i - 2]
        )

    # ------------------------------
    # Step 2: 超级平滑滤波 (Super Smoother)
    # ------------------------------
    a1 = np.exp(-1.414 * np.pi / 10.0)
    # 修改：使用 deg_cos 实现角度计算
    b1 = 2.0 * a1 * deg_cos(1.414 * 180.0 / 10.0)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    filt = np.zeros(length)
    filt[0:2] = hp[0:2]

    for i in range(2, length):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

    # ------------------------------
    # Step 3: 相关系数计算并确定主导周期
    # ------------------------------
    # 参照 Ehlers 思路，在 [10, 48] 范围内对各周期进行分析
    period_min, period_max = 10, 48
    max_lag = 48

    # 3) 调用 numba 加速的核心函数
    mycci = _calculate_adaptive_cci_numba(
        filt, length, c1, c2, c3, period_min, period_max, max_lag
    )

    if sequential:
        return mycci
    else:
        return mycci[-1]
