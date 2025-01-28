from typing import Union

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit


@njit
def _high_pass_filter(source: np.ndarray) -> np.ndarray:
    alpha1 = (
        np.cos(0.707 * 2 * np.pi / 48) + np.sin(0.707 * 2 * np.pi / 48) - 1
    ) / np.cos(0.707 * 2 * np.pi / 48)
    hp = np.zeros_like(source)
    for i in range(2, len(source)):
        hp[i] = (
            (1 - alpha1 / 2)
            * (1 - alpha1 / 2)
            * (source[i] - 2 * source[i - 1] + source[i - 2])
            + 2 * (1 - alpha1) * hp[i - 1]
            - (1 - alpha1) * (1 - alpha1) * hp[i - 2]
        )
    return hp


@njit
def _super_smoother(hp: np.ndarray, length: int) -> np.ndarray:
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c1 = 1 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1

    filt = np.zeros_like(hp)
    for i in range(2, len(hp)):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
    return filt


@njit
def _calculate_mod_rsi(filt: np.ndarray, length: int) -> np.ndarray:
    closes_up = np.zeros_like(filt)
    closes_dn = np.zeros_like(filt)
    rsi = np.zeros_like(filt)

    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c1 = 1 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1

    for i in range(length, len(filt)):
        up_sum = 0.0
        dn_sum = 0.0
        for j in range(length):
            if filt[i - j] > filt[i - j - 1]:
                up_sum += filt[i - j] - filt[i - j - 1]
            if filt[i - j] < filt[i - j - 1]:
                dn_sum += filt[i - j - 1] - filt[i - j]

        denom = up_sum + dn_sum
        if denom != 0 and i > 2:
            prev_denom = closes_up[i - 1] + closes_dn[i - 1]
            if prev_denom != 0:
                rsi[i] = (
                    c1 * (up_sum / denom + closes_up[i - 1] / prev_denom) / 2
                    + c2 * rsi[i - 1]
                    + c3 * rsi[i - 2]
                )

        closes_up[i] = up_sum
        closes_dn[i] = dn_sum

    return rsi


def mod_rsi(
    candles: np.ndarray,
    length: int = 10,
    source_type: str = "close",
    sequential: bool = False,
) -> Union[float, np.ndarray]:
    """
    改进版RSI指标

    :param candles: np.ndarray
    :param length: int - 默认周期为10
    :param source_type: str - 价格数据来源
    :param sequential: bool - 是否返回序列数据
    :return: Union[float, np.ndarray]
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)

    # 计算高通滤波
    hp = _high_pass_filter(source)

    # 使用Super Smoother进行平滑
    filt = _super_smoother(hp, length)

    # 计算改进版RSI
    rsi = _calculate_mod_rsi(filt, length)

    return rsi if sequential else rsi[-1]
