import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _compute_hp(src: np.ndarray, alpha1: float) -> np.ndarray:
    length = len(src)
    HP = np.zeros(length)
    HP[0] = 0
    HP[1] = 0
    for i in range(2, length):
        HP[i] = (
            (1 - alpha1 / 2) * (1 - alpha1 / 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
            + 2 * (1 - alpha1) * HP[i - 1]
            - (1 - alpha1) * (1 - alpha1) * HP[i - 2]
        )
    return HP


@njit
def _compute_super_smoother(
    HP: np.ndarray, c1: float, c2: float, c3: float
) -> np.ndarray:
    length = len(HP)
    Filt = np.zeros(length)
    Filt[0] = HP[0]
    Filt[1] = HP[1]
    for i in range(2, length):
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
    return Filt


@njit
def _compute_corr(Filt: np.ndarray, avg_length: int, max_lag: int) -> np.ndarray:
    length = len(Filt)
    Corr = np.zeros((length, max_lag + 1))
    for i in range(length):
        for lag in range(max_lag + 1):
            M = avg_length if avg_length != 0 else lag
            if M <= 0 or (i - (lag + M - 1) < 0):
                Corr[i, lag] = 0.0
                continue

            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            start_idx = i - (lag + M - 1)
            for count in range(M):
                X = Filt[start_idx + count]
                Y = Filt[start_idx + count + lag]
                sx += X
                sy += Y
                sxx += X * X
                sxy += X * Y
                syy += Y * Y
            denom = (M * sxx - sx * sx) * (M * syy - sy * sy)
            if denom > 0:
                Corr[i, lag] = (M * sxy - sx * sy) / np.sqrt(denom)
            else:
                Corr[i, lag] = 0.0
    return Corr


@njit
def _compute_power(Corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length, max_lag_plus = Corr.shape
    max_lag = max_lag_plus - 1

    # 为了配合原逻辑，需要三个数组
    R = np.zeros((length, max_lag + 1, 2))
    CosinePart = np.zeros((length, max_lag + 1))
    SinePart = np.zeros((length, max_lag + 1))
    SqSum = np.zeros((length, max_lag + 1))

    # 计算CosinePart / SinePart / SqSum
    for i in range(length):
        for period in range(10, max_lag + 1):
            cosine_acc = 0.0
            sine_acc = 0.0
            for n in range(3, max_lag + 1):
                c_val = deg_cos(360 * n / period)
                s_val = deg_sin(360 * n / period)
                cosine_acc += Corr[i, n] * c_val
                sine_acc += Corr[i, n] * s_val
            CosinePart[i, period] = cosine_acc
            SinePart[i, period] = sine_acc
            SqSum[i, period] = cosine_acc * cosine_acc + sine_acc * sine_acc

        # 用 .2 / .8 做指数平滑
        if i == 0:
            for period in range(10, max_lag + 1):
                R[i, period, 0] = 0.2 * SqSum[i, period] * SqSum[i, period]
                R[i, period, 1] = 0
        else:
            for period in range(10, max_lag + 1):
                prev_r1 = R[i - 1, period, 0]
                R[i, period, 0] = (
                    0.2 * SqSum[i, period] * SqSum[i, period] + 0.8 * prev_r1
                )
                R[i, period, 1] = R[i - 1, period, 0]

    # 计算Pwr
    Pwr = np.zeros((length, max_lag + 1))
    MaxPwr = np.zeros(length)
    for i in range(length):
        if i == 0:
            MaxPwr[i] = 1e-9
        else:
            MaxPwr[i] = 0.991 * MaxPwr[i - 1]
        for period in range(10, max_lag + 1):
            if R[i, period, 0] > MaxPwr[i]:
                MaxPwr[i] = R[i, period, 0]
        for period in range(3, max_lag + 1):
            if MaxPwr[i] > 0:
                Pwr[i, period] = R[i, period, 0] / MaxPwr[i]
            else:
                Pwr[i, period] = 0.0

    return Pwr, CosinePart, SinePart


@njit
def _compute_dominant_cycle(Pwr: np.ndarray) -> np.ndarray:
    length, max_lag_plus = Pwr.shape
    max_lag = max_lag_plus - 1
    DominantCycle = np.zeros(length)
    for i in range(length):
        spx = 0.0
        sp = 0.0
        for period in range(10, max_lag + 1):
            if Pwr[i, period] >= 0.5:
                spx += period * Pwr[i, period]
                sp += Pwr[i, period]
        if sp != 0:
            dc = spx / sp
            if dc < 10:
                dc = 10
            elif dc > 48:
                dc = 48
            DominantCycle[i] = dc
        else:
            DominantCycle[i] = 10
    return DominantCycle


@njit
def _compute_adaptive_rsi(
    Filt: np.ndarray, DominantCycle: np.ndarray, c1: float, c2: float, c3: float
) -> np.ndarray:
    length = len(Filt)
    MyRSI = np.zeros(length)
    if length > 1:
        MyRSI[0] = 0.0
        MyRSI[1] = 0.0

    for i in range(length):
        half_cycle = int(DominantCycle[i] // 2)
        if half_cycle < 1:
            half_cycle = 1
        if i - (half_cycle) < 0:
            MyRSI[i] = 0.0
            continue

        cu = 0.0
        cd = 0.0
        for count in range(half_cycle):
            idx_now = i - count
            idx_next = i - count - 1
            if idx_next < 0:
                break
            diff_val = Filt[idx_now] - Filt[idx_next]
            if diff_val > 0:
                cu += diff_val
            else:
                cd -= diff_val  # diff_val<0时，-diff_val 等于上文的 (Filt[idx_next]-Filt[idx_now])

        # 计算前一个bar的cu/cd
        if i >= 1:
            half_cycle_prev = int(DominantCycle[i - 1] // 2)
            if half_cycle_prev < 1:
                half_cycle_prev = 1
            cu_prev = 0.0
            cd_prev = 0.0
            for count in range(half_cycle_prev):
                idx_now = (i - 1) - count
                idx_next = (i - 1) - count - 1
                if idx_next < 0:
                    break
                diff_val_prev = Filt[idx_now] - Filt[idx_next]
                if diff_val_prev > 0:
                    cu_prev += diff_val_prev
                else:
                    cd_prev -= diff_val_prev
            denom_prev = cu_prev + cd_prev
        else:
            denom_prev = 0.0
            cu_prev = 0.0

        denom = cu + cd
        if denom != 0 and denom_prev != 0:
            rsi_component = (cu / denom + cu_prev / denom_prev) / 2
        elif denom != 0:
            rsi_component = cu / denom
        else:
            rsi_component = 0.0

        if i >= 2:
            MyRSI[i] = c1 * rsi_component + c2 * MyRSI[i - 1] + c3 * MyRSI[i - 2]
        elif i == 1:
            MyRSI[i] = rsi_component
        else:
            MyRSI[i] = 0.0

    return MyRSI


def adaptive_rsi(
    candles: np.ndarray,
    source_type: str = "close",
    avg_length: int = 3,
    sequential: bool = False,
):
    """
    自适应RSI指标 (基于John F. Ehlers在Cycle Analytics for Traders中的示例代码改写)，
    已使用 numba 对部分循环进行加速。
    :param candles: Numpy数组，形状为[..., 6]，依次为 [时间戳, 开盘价, 最高价, 最低价, 收盘价, 成交量]
    :param source_type: 输入参考的价格类型，默认 "close"
    :param avg_length: 相关系数计算时的平均长度参数
    :param sequential: 是否返回整个序列，True 返回与candles等长的RSI序列，否则只返回最后一个值
    :return: 自适应RSI序列或最后一个值
    """
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    length = len(src)
    if length < 50:
        rsi_dummy = np.full_like(src, np.nan, dtype=float)
        return rsi_dummy if sequential else rsi_dummy[-1]

    # Highpass滤波系数
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )

    # 1. 计算HP
    HP = _compute_hp(src, alpha1)

    # 2. 超级平滑器
    a1 = np.exp(-1.414 * np.pi / 10)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    Filt = _compute_super_smoother(HP, c1, c2, c3)

    # 3. 计算相关系数Corr
    max_lag = 48
    Corr = _compute_corr(Filt, avg_length, max_lag)

    # 4. 计算Spectral Power
    Pwr, CosinePart, SinePart = _compute_power(Corr)

    # 5. 计算DominantCycle
    DominantCycle = _compute_dominant_cycle(Pwr)

    # 6. 计算Adaptive RSI
    MyRSI = _compute_adaptive_rsi(Filt, DominantCycle, c1, c2, c3)

    if sequential:
        return MyRSI
    else:
        return MyRSI[-1]
