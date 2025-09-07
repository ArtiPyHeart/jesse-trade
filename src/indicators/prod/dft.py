import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _calc_dft(window, L, p_min, p_max, spectral_dilation_compensation):
    num_periods = p_max - p_min + 1
    raw_pwr = np.zeros(num_periods)
    for p in range(p_min, p_max + 1):
        Comp = p if spectral_dilation_compensation else 1
        cosine_sum = 0.0
        sine_sum = 0.0
        for N in range(L):
            # 使用角度制，确保与 EasyLanguage 中 Cosine(360*N/p) 一致
            angle_deg = 360.0 * N / p
            cosine_sum += window[N] * deg_cos(angle_deg)
            sine_sum += window[N] * deg_sin(angle_deg)
        cosine_sum /= Comp
        sine_sum /= Comp
        raw_pwr[p - p_min] = cosine_sum**2 + sine_sum**2
    max_pwr = np.max(raw_pwr)
    norm_pwr = np.zeros(num_periods)
    if max_pwr > 0:
        for i in range(num_periods):
            norm_pwr[i] = raw_pwr[i] / max_pwr
    else:
        norm_pwr = raw_pwr.copy()
    Spx = 0.0
    Sp = 0.0
    for idx in range(num_periods):
        period = idx + p_min
        if norm_pwr[idx] >= 0.5:
            Spx += period * norm_pwr[idx]
            Sp += norm_pwr[idx]
    if Sp != 0:
        dominant_cycle = Spx / Sp
    else:
        dominant_cycle = np.nan
    return norm_pwr, dominant_cycle


def dft(
    candles: np.ndarray,
    source_type: str = "close",
    spectral_dilation_compensation: bool = True,
    sequential: bool = False,
):
    # 按照 sequential 参数裁剪 candles
    candles = slice_candles(candles, sequential)
    # 使用指定数据源（例如收盘价）
    close = get_candle_source(candles, source_type)
    length = len(close)

    # 定义窗口长度及周期范围
    L = 48
    p_min = 10
    p_max = 48
    num_periods = p_max - p_min + 1  # 共39个周期

    # 初始化各序列
    HP = np.zeros(length)
    Filt = np.zeros(length)
    dominant_cycle = np.full(length, np.nan)
    # spectrum 数组的 shape 为 (length, 39)，每列对应一个周期的归一化功率
    spectrum = np.full((length, num_periods), np.nan)

    # 使用角度制计算高通滤波器参数，不再通过弧度转换
    alpha1 = (deg_cos(0.707 * 360 / L) + deg_sin(0.707 * 360 / L) - 1) / deg_cos(
        0.707 * 360 / L
    )

    # 超级平滑滤波器参数
    a1 = np.exp(-1.414 * np.pi / 10)
    angle2 = 1.414 * 180 / 10
    b1 = 2 * a1 * deg_cos(angle2)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # 逐个计算HP与Filt
    for i in range(length):
        if i < 2:
            HP[i] = 0
            Filt[i] = 0
        else:
            HP[i] = (
                (1 - alpha1 / 2) ** 2 * (close[i] - 2 * close[i - 1] + close[i - 2])
                + 2 * (1 - alpha1) * HP[i - 1]
                - (1 - alpha1) ** 2 * HP[i - 2]
            )
            Filt[i] = (
                c1 * ((HP[i] + HP[i - 1]) / 2) + c2 * Filt[i - 1] + c3 * Filt[i - 2]
            )

        # 当历史数据足够（至少48个点）时进行DFT计算
        if i >= L - 1:
            window = Filt[i - L + 1 : i + 1]  # 长度为48的滑动窗口
            norm_pwr, d_cycle = _calc_dft(
                window, L, p_min, p_max, spectral_dilation_compensation
            )
            spectrum[i, :] = norm_pwr
            dominant_cycle[i] = d_cycle

    # 根据 sequential 返回全量的指标序列或最新一根K线的指标值
    if sequential:
        return dominant_cycle, spectrum
    else:
        return dominant_cycle[-1:], spectrum[-1:]
