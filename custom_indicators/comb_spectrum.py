import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math import cos_radians, sin_radians


@njit
def _calc_comb_spectrum(
    src,
    length,
    shortest_period,
    longest_period,
    spectral_dilation_compensation,
    bandwidth,
):
    """
    使用 numba 加速计算核心循环，包括高通滤波、超级平滑滤波以及带通滤波等部分。
    """
    # 初始化数组
    hpc = np.zeros(length)
    filt = np.zeros(length)
    BP = np.zeros((49, 49))  # 10..48 用到
    pwr_list = np.zeros((length, 49))
    dominant_cycle = np.zeros(length)

    alpha1_hp = (
        cos_radians(0.707 * 360 / 48) + sin_radians(0.707 * 360 / 48) - 1.0
    ) / cos_radians(0.707 * 360 / 48)

    # 超级平滑滤波 (Super Smoother) 参数
    a1 = math.exp(-1.414 * math.pi / 10.0)
    b1 = 2.0 * a1 * math.cos(math.radians(1.414 * 180 / 10.0))
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3

    max_pwr = 0.0  # 用于动态调整

    for i in range(length):
        # --- 1) 高通滤波 HPC ---
        if i >= 2:
            hpc[i] = (
                ((1 - alpha1_hp / 2.0) ** 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
                + 2 * (1 - alpha1_hp) * hpc[i - 1]
                - ((1 - alpha1_hp) ** 2) * hpc[i - 2]
            )
        else:
            hpc[i] = 0.0

        # --- 2) 超级平滑滤波 Filt ---
        if i >= 2:
            filt[i] = (
                c1 * (hpc[i] + hpc[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]
            )
        else:
            filt[i] = 0.0

        # --- 3) 带通滤波 ---
        if i > 12:
            # 衰减 max_pwr
            max_pwr *= 0.995

            for n in range(shortest_period, longest_period + 1):
                # 后移 BP[n,m]
                for m in range(48, 1, -1):
                    BP[n, m] = BP[n, m - 1]

                comp = float(n) if spectral_dilation_compensation else 1.0
                beta1 = cos_radians(360.0 / n)
                gamma1 = 1.0 / cos_radians(360.0 * bandwidth / n)
                alpha1 = gamma1 - math.sqrt(gamma1 * gamma1 - 1.0)

                if i >= 2:
                    BP[n, 1] = (
                        0.5 * (1 - alpha1) * (filt[i] - filt[i - 2])
                        + beta1 * (1 + alpha1) * BP[n, 2]
                        - alpha1 * BP[n, 3]
                    )
                else:
                    BP[n, 1] = 0.0

                # 计算 Pwr
                p_sum = 0.0
                for m in range(1, n + 1):
                    val = BP[n, m] / comp
                    p_sum += val * val
                pwr_list[i, n] = p_sum

                if pwr_list[i, n] > max_pwr:
                    max_pwr = pwr_list[i, n]

            # --- 归一化 Pwr ---
            if max_pwr > 0:
                for n in range(shortest_period, longest_period + 1):
                    pwr_list[i, n] = pwr_list[i, n] / max_pwr

            # --- 计算主周期 ---
            spx = 0.0
            sp = 0.0
            for period in range(shortest_period, longest_period + 1):
                if pwr_list[i, period] >= 0.5:
                    spx += period * pwr_list[i, period]
                    sp += pwr_list[i, period]

            if abs(sp) > 1e-12:
                dominant_cycle[i] = spx / sp
            else:
                dominant_cycle[i] = 0.0
        else:
            for n in range(shortest_period, longest_period + 1):
                pwr_list[i, n] = 0.0
            dominant_cycle[i] = 0.0

    return dominant_cycle, pwr_list


def comb_spectrum(
    candles: np.ndarray,
    source_type: str = "close",
    sequential: bool = False,
    shortest_period: int = 10,
    longest_period: int = 48,
    spectral_dilation_compensation: bool = True,
    bandwidth: float = 0.3,
):
    # 预处理
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    length = src.shape[0]
    # 调用 numba 加速函数
    dominant_cycle, pwr_list = _calc_comb_spectrum(
        src,
        length,
        shortest_period,
        longest_period,
        spectral_dilation_compensation,
        bandwidth,
    )

    if sequential:
        # 返回全序列
        return dominant_cycle, pwr_list[:, shortest_period : longest_period + 1]
    else:
        # 仅返回最后一根K线
        return dominant_cycle[-1], pwr_list[-1, shortest_period : longest_period + 1]
