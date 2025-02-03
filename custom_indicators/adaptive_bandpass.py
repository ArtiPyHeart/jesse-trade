import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit


@njit
def _highpass_filter(src: np.ndarray, alpha1: float) -> np.ndarray:
    """
    计算高通滤波后的序列。
    """
    hp = np.zeros_like(src)
    for i in range(len(src)):
        if i < 2:
            hp[i] = 0.0
        else:
            hp[i] = (
                (1 - alpha1 / 2)
                * (1 - alpha1 / 2)
                * (src[i] - 2 * src[i - 1] + src[i - 2])
                + 2 * (1 - alpha1) * hp[i - 1]
                - (1 - alpha1) * (1 - alpha1) * hp[i - 2]
            )
    return hp


@njit
def _super_smoother_filter(
    src: np.ndarray, c1: float, c2: float, c3: float
) -> np.ndarray:
    """
    超平滑滤波器。根据HP序列平滑得到Filt。
    """
    filt = np.zeros_like(src)
    for i in range(len(src)):
        if i < 2:
            filt[i] = 0.0
        else:
            filt[i] = (
                c1 * (src[i] + src[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
            )
    return filt


@njit
def _compute_pearson_correlation(
    filt: np.ndarray, lag_max: int, avg_length: int
) -> np.ndarray:
    """
    计算从 0 到 lag_max 之间各个滞后的皮尔森相关系数。
    """
    length = len(filt)
    corr = np.zeros(lag_max + 1)
    for lag in range(lag_max + 1):
        # 当自适应均值长度为 0 时，使用 lag 做为长度
        M = avg_length if avg_length > 0 else lag
        if M <= 0:
            corr[lag] = 0.0
            continue
        if M + lag > length:  # 数据不足无法计算
            corr[lag] = 0.0
            continue

        Sx, Sy, Sxx, Syy, Sxy = 0.0, 0.0, 0.0, 0.0, 0.0
        for c in range(M):
            X = filt[length - 1 - c]
            Y = filt[length - 1 - c - lag]
            Sx += X
            Sy += Y
            Sxx += X * X
            Syy += Y * Y
            Sxy += X * Y

        denom = (M * Sxx - Sx * Sx) * (M * Syy - Sy * Sy)
        if denom > 0:
            corr[lag] = (M * Sxy - Sx * Sy) / np.sqrt(denom)
        else:
            corr[lag] = 0.0
    return corr


@njit
def _calculate_power_spectrum(corr: np.ndarray, lag_max: int) -> np.ndarray:
    """
    累加各滞后与正弦、余弦的分量得到功率谱，并进行指数平滑。
    """
    # 只有在 10-48 才计算 Period
    cosine_part = np.zeros(lag_max + 1)
    sine_part = np.zeros(lag_max + 1)
    sqsum = np.zeros(lag_max + 1)
    R = np.zeros((lag_max + 1, 2))

    for Period in range(10, lag_max + 1):
        # 计算余弦正弦部分
        cp, sp = 0.0, 0.0
        for N in range(3, lag_max + 1):
            cp += corr[N] * np.cos(np.deg2rad(360 * N / Period))
            sp += corr[N] * np.sin(np.deg2rad(360 * N / Period))
        cosine_part[Period] = cp
        sine_part[Period] = sp
        sqsum[Period] = cp * cp + sp * sp

    # 将 sqsum 做平滑存储到 R
    for Period in range(10, lag_max + 1):
        R[Period, 1] = 0.2 * sqsum[Period] * sqsum[Period] + 0.8 * R[Period, 0]
        R[Period, 0] = R[Period, 1]  # shift

    # 找最大能量，用于归一化
    max_pwr = 0.0
    for Period in range(10, lag_max + 1):
        if R[Period, 1] > max_pwr:
            max_pwr = R[Period, 1]

    # 计算功率谱
    pwr = np.zeros(lag_max + 1)
    if max_pwr > 0:
        for Period in range(3, lag_max + 1):
            pwr[Period] = R[Period, 1] / max_pwr

    return pwr


@njit
def _dominant_cycle(pwr: np.ndarray, threshold: float = 0.5):
    """
    根据大于阈值的功率值求加权平均得到主周期（DominantCycle）。
    """
    lag_max = pwr.shape[0] - 1
    spx, sp = 0.0, 0.0
    for Period in range(10, lag_max + 1):
        if pwr[Period] >= threshold:
            spx += Period * pwr[Period]
            sp += pwr[Period]
    if sp != 0.0:
        dc = spx / sp
        if dc < 10:
            dc = 10
        return dc
    else:
        return 10.0


@njit
def _adaptive_bandpass_filter(
    filt: np.ndarray, dominant_cycle: float, bandwidth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据主周期和带宽参数对滤波后的数据做自适应带通（BandPass）滤波，得到：
    - Signal：当前带通输出单位化
    - LeadSignal：在当前Signal基础上适度超前并单位化
    """
    length = len(filt)
    bp = np.zeros_like(filt)
    peak = np.zeros_like(filt)
    signal = np.zeros_like(filt)
    lead = np.zeros_like(filt)
    lead_peak = np.zeros_like(filt)
    lead_signal = np.zeros_like(filt)

    # 带通滤波参数
    beta1 = np.cos(np.deg2rad(360.0 / (0.9 * dominant_cycle)))
    gamma1 = 1.0 / np.cos(np.deg2rad(360.0 * bandwidth / (0.9 * dominant_cycle)))
    alpha2 = gamma1 - np.sqrt(gamma1**2 - 1.0)

    for i in range(length):
        if i < 2:
            bp[i] = 0.0
        else:
            bp[i] = (
                0.5 * (1 - alpha2) * (filt[i] - filt[i - 2])
                + beta1 * (1 + alpha2) * bp[i - 1]
                - alpha2 * bp[i - 2]
            )

        if i < 1:
            peak[i] = 0.0
        else:
            peak[i] = 0.991 * peak[i - 1]
            if abs(bp[i]) > peak[i]:
                peak[i] = abs(bp[i])

        if peak[i] != 0.0:
            signal[i] = bp[i] / peak[i]

    # 计算 Lead & LeadSignal
    for i in range(length):
        if i >= 3:
            lead[i] = (
                1.3 * (signal[i] + signal[i - 1] - signal[i - 2] - signal[i - 3]) / 4.0
            )
            lead_peak[i] = 0.93 * lead_peak[i - 1]
            if abs(lead[i]) > lead_peak[i]:
                lead_peak[i] = abs(lead[i])
            if lead_peak[i] != 0.0:
                lead_signal[i] = 0.7 * lead[i] / lead_peak[i]

    return signal, lead_signal, bp


def adaptive_bandpass(
    candles: np.ndarray,
    source_type: str = "close",
    bandwidth: float = 0.3,
    avg_length: int = 3,
    sequential: bool = False,
):
    """
    自适应带通滤波器（Adaptive BandPass）示例:
      - candles: OHLCV 数据 (np.ndarray)
      - source_type: 用于计算的价格类型（默认为 "close"）
      - bandwidth: 带宽控制参数
      - avg_length: 相关计算时的平滑长度
      - sequential: 是否返回整个序列
    返回值:
      - signal: 带通输出信号（单位化）
      - lead_signal: 超前带通信号（单位化）
      - bp: 带通原始滤波值，可自行参考或忽略
    """
    # 截取或保留全部candle记录
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    length = len(src)

    if length < 50:
        # 数据太短，无法计算时，返回空或仅返回src
        if sequential:
            return np.zeros(length), np.zeros(length), np.zeros(length)
        else:
            return 0.0, 0.0, 0.0

    # 1) 高通滤波参数 & 执行
    #    例中 alpha1 来自原公式: alpha1 = (cos(0.707*360/48)+sin(0.707*360/48)-1)/cos(0.707*360/48)
    alpha1 = (
        np.cos(np.deg2rad(0.707 * 360 / 48.0))
        + np.sin(np.deg2rad(0.707 * 360 / 48.0))
        - 1
    ) / np.cos(np.deg2rad(0.707 * 360 / 48.0))

    hp = _highpass_filter(src, alpha1)

    # 2) 超平滑滤波
    #    a1 = exp(-1.414*π / 10)；b1, c1, c2, c3依公式计算
    a1 = np.exp(-1.414 * np.pi / 10.0)
    b1 = 2.0 * a1 * np.cos(np.deg2rad(1.414 * 180 / 10.0))
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3
    filt = _super_smoother_filter(hp, c1, c2, c3)

    # 3) 相关系数 & 功率谱分析，决定主周期
    lag_max = 48
    corr = _compute_pearson_correlation(filt, lag_max, avg_length)
    pwr = _calculate_power_spectrum(corr, lag_max)
    dc = _dominant_cycle(pwr)

    # 4) 自适应带通滤波
    signal, lead_signal, bp = _adaptive_bandpass_filter(filt, dc, bandwidth)

    if sequential:
        return signal, lead_signal, bp
    else:
        return signal[-1], lead_signal[-1], bp[-1]
