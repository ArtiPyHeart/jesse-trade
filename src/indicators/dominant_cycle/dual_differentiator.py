import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _calc_dual_differentiator_core(src, alpha1, c1, c2, c3, LPPeriod, eps):
    n = len(src)
    HP = np.zeros(n, dtype=np.float64)
    Filt = np.zeros(n, dtype=np.float64)
    IPeak = np.zeros(n, dtype=np.float64)
    Real_arr = np.zeros(n, dtype=np.float64)
    Quad = np.zeros(n, dtype=np.float64)
    QPeak = np.zeros(n, dtype=np.float64)
    Imag = np.zeros(n, dtype=np.float64)
    IDot = np.zeros(n, dtype=np.float64)
    QDot = np.zeros(n, dtype=np.float64)
    Period_arr = np.zeros(n, dtype=np.float64)
    DomCycle = np.zeros(n, dtype=np.float64)

    # 初始化前两个K线数据
    HP[0] = 0.0
    Filt[0] = 0.0
    IPeak[0] = eps
    Real_arr[0] = 0.0
    Quad[0] = 0.0
    QPeak[0] = eps
    Imag[0] = 0.0
    IDot[0] = 0.0
    QDot[0] = 0.0
    Period_arr[0] = LPPeriod
    DomCycle[0] = LPPeriod

    if n > 1:
        HP[1] = 0.0
        Filt[1] = 0.0
        IPeak[1] = eps
        Real_arr[1] = 0.0
        Quad[1] = 0.0
        QPeak[1] = eps
        Imag[1] = 0.0
        IDot[1] = 0.0
        QDot[1] = 0.0
        Period_arr[1] = LPPeriod
        DomCycle[1] = LPPeriod

    for i in range(2, n):
        HP[i] = (
            (1 - alpha1 / 2.0) ** 2 * (src[i] - 2.0 * src[i - 1] + src[i - 2])
            + 2.0 * (1 - alpha1) * HP[i - 1]
            - ((1 - alpha1) ** 2) * HP[i - 2]
        )
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
        IPeak[i] = 0.991 * IPeak[i - 1]
        if abs(Filt[i]) > IPeak[i]:
            IPeak[i] = abs(Filt[i])
        Real_arr[i] = Filt[i] / (IPeak[i] if abs(IPeak[i]) > eps else eps)
        Quad[i] = Real_arr[i] - Real_arr[i - 1]
        QPeak[i] = 0.991 * QPeak[i - 1]
        if abs(Quad[i]) > QPeak[i]:
            QPeak[i] = abs(Quad[i])
        Imag[i] = Quad[i] / (QPeak[i] if abs(QPeak[i]) > eps else eps)
        IDot[i] = Real_arr[i] - Real_arr[i - 1]
        QDot[i] = Imag[i] - Imag[i - 1]
        denominator = -Real_arr[i] * QDot[i] + Imag[i] * IDot[i]
        if abs(Real_arr[i] * QDot[i] - Imag[i] * IDot[i]) > eps:
            Period_arr[i] = 6.28318 * (Real_arr[i] ** 2 + Imag[i] ** 2) / denominator
        else:
            Period_arr[i] = Period_arr[i - 1]
        if Period_arr[i] < 8:
            Period_arr[i] = 8.0
        elif Period_arr[i] > 48:
            Period_arr[i] = 48.0
        DomCycle[i] = (
            c1 * (Period_arr[i] + Period_arr[i - 1]) / 2.0
            + c2 * DomCycle[i - 1]
            + c3 * DomCycle[i - 2]
        )
    return DomCycle


def dual_differentiator(
    candles: np.ndarray,
    source_type: str = "close",
    LPPeriod: int = 20,
    sequential: bool = False,
):
    """
    根据 John F. Ehlers 的 Dual Differentiator 算法计算支配周期 (Dominant Cycle)。

    算法步骤（参考 EasyLanguage 代码）：
      1. 使用高通滤波器过滤周期低于48根K线的成分，计算 HP。
      2. 使用超级平滑滤波器对 HP 进行平滑得到 Filt。
      3. 利用 Filt 的局部绝对峰值更新 IPeak，并计算实部 Real = Filt / IPeak。
      4. 以 Real 的差分计算虚部 Quad 与峰值 QPeak，再得到虚部 Imag = Quad / QPeak。
      5. 利用 Real 与 Imag 及其导数计算原始周期，并限制其范围在 8 到 48 内。
      6. 最后对周期进行再次平滑，得到最终支配周期 DomCycle。

    :param candles: 输入的K线数据 (numpy.ndarray)
    :param source_type: 指定数据类型(默认选取收盘价 "close")
    :param sequential: 是否返回全序列 (True 返回整个计算数组，False 仅返回最新值)
    :param LPPeriod: 超级平滑滤波器周期，默认值为20
    :return: 支配周期，当 sequential=False 时返回最后一个周期值，否则返回整个数组
    """
    # 切片数据（根据 sequential 参数截取全部或仅最近数据）
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    n = len(src)
    if n == 0:
        return []

    # 预先计算常数
    # alpha1 = (Cosine(0.707*360/48) + Sine(0.707*360/48) - 1) / Cosine(0.707*360/48)
    alpha_val = 0.707 * 360.0 / 48.0
    alpha1 = (deg_cos(alpha_val) + deg_sin(alpha_val) - 1) / deg_cos(alpha_val)

    # 超级平滑滤波器参数（公式中的 a1, b1, c1, c2, c3）
    a1 = math.exp(-1.414 * math.pi / LPPeriod)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / LPPeriod)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    eps = 1e-10

    DomCycle = _calc_dual_differentiator_core(src, alpha1, c1, c2, c3, LPPeriod, eps)
    if sequential:
        return DomCycle
    else:
        return DomCycle[-1]
