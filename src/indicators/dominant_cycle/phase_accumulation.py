import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _phase_accumulation_loop(
    close, n, hp_factor, hp_coef2, hp_coef3, c1, c2, c3, c1_dom, c2_dom, c3_dom, eps
):
    # 预先分配各个中间变量数组
    HP = np.zeros(n)  # 高通滤波结果
    Filt = np.zeros(n)  # 平滑后的序列
    IPeak = np.zeros(n)  # 滤波器瞬时峰值（用于归一化）
    Real_arr = np.zeros(n)  # 实部
    Quad = np.zeros(n)  # 差分序列（用于计算虚部）
    QPeak = np.zeros(n)  # 虚部峰值（用于归一化）
    Imag = np.zeros(n)  # 虚部
    Phase = np.zeros(n)  # 相位
    DeltaPhase = np.zeros(n)  # 差分相位
    InstPeriod = np.zeros(n)  # 瞬时周期（累积的 delta phase 达到360度后的周期长度）
    DomCycle = np.zeros(n)  # 主导周期的平滑结果

    # 初始化初值：前两个bar直接取0，且峰值设定为一个很小的值防止除零
    IPeak[0] = eps
    QPeak[0] = eps
    # InstPeriod 和 DomCycle的初始值直接为0（也可以根据需求调整）

    ##### 主循环，从第三个bar开始（i从2开始） #####
    for i in range(2, n):
        # 1. 高通滤波 (HP)
        # HP[i] = hp_factor*(close[i] - 2*close[i-1] + close[i-2])
        #         + hp_coef2*HP[i-1] - hp_coef3*HP[i-2]
        HP[i] = (
            hp_factor * (close[i] - 2 * close[i - 1] + close[i - 2])
            + hp_coef2 * HP[i - 1]
            - hp_coef3 * HP[i - 2]
        )

        # 2. 超级平滑滤波 (Filt)
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

        # 3. 归一化：更新 IPeak
        IPeak[i] = 0.991 * IPeak[i - 1]
        if abs(Filt[i]) > IPeak[i]:
            IPeak[i] = abs(Filt[i])
        # 计算实部
        Real_arr[i] = Filt[i] / (IPeak[i] if IPeak[i] > eps else eps)

        # 4. 虚部差分 (Quad) and 更新 QPeak，再计算 Imag
        Quad[i] = Real_arr[i] - Real_arr[i - 1]
        QPeak[i] = 0.991 * QPeak[i - 1]
        if abs(Quad[i]) > QPeak[i]:
            QPeak[i] = abs(Quad[i])
        Imag[i] = Quad[i] / (QPeak[i] if QPeak[i] > eps else eps)

        # 5. 计算当前相位（单位：度）
        if abs(Real_arr[i]) > eps:
            # 这里采用正值的反正切，然后转换为角度
            phase = math.degrees(math.atan(abs(Imag[i] / Real_arr[i])))
        else:
            phase = 0.0

        # 根据不同象限修正相位
        if Real_arr[i] < 0 < Imag[i]:
            phase = 180 - phase
        elif Real_arr[i] < 0 and Imag[i] < 0:
            phase = 180 + phase
        elif Real_arr[i] > 0 > Imag[i]:
            phase = 360 - phase
        Phase[i] = phase

        # 6. 计算相位差 DeltaPhase
        # 对于第一个可计算的值(i>=1)
        d_phase = Phase[i - 1] - Phase[i]
        # 解决相位跳跃问题（例如：前一根K线相位小于90，当前K线相位大于270时）
        if Phase[i - 1] < 90 and Phase[i] > 270:
            d_phase = 360 + Phase[i - 1] - Phase[i]
        # 限定 DeltaPhase 的范围 [10, 48]
        d_phase = max(10, min(d_phase, 48))
        DeltaPhase[i] = d_phase

        # 7. 累计 DeltaPhase 达到360度来确定瞬时周期 InstPeriod
        phase_sum = 0.0
        period = 0
        # 向后回溯最多50个周期（如果可用）
        for k in range(0, min(51, i + 1)):
            phase_sum += DeltaPhase[i - k]
            if phase_sum > 360:
                period = k
                break
        # 如果未能累计超过360度，则延用上一周期
        if period == 0 and i > 0:
            period = InstPeriod[i - 1]
        InstPeriod[i] = period

        # 8. 用超级平滑滤波器对 InstPeriod 进行平滑得到 DomCycle
        if i >= 2:
            DomCycle[i] = (
                c1_dom * ((InstPeriod[i] + InstPeriod[i - 1]) / 2)
                + c2_dom * DomCycle[i - 1]
                + c3_dom * DomCycle[i - 2]
            )
        else:
            DomCycle[i] = InstPeriod[i]

    return DomCycle


def phase_accumulation(
    candles: np.ndarray,
    source_type: str = "close",
    LPPeriod: int = 20,
    sequential: bool = False,
):
    """
    通过相位累积法测量主导周期，基于 John F. Ehlers 的指标实现。
    参数:
        candles: k线数据数组
        source_type: 选取的价格类型（默认使用收盘价）
        sequential: 是否返回所有周期值，True返回全序列，False仅返回最后一个值
        LPPeriod: 超级平滑滤波器的参数，默认20
    返回:
        主导周期 (numpy 数组或单个值)
    """
    # 如果不返回全序列时取截断的最后一部分
    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type)
    n = len(close)
    if n < 3:
        # 数据不足无法计算
        return close[-1] if not sequential else np.zeros(n)

    eps = 1e-8

    ##### 1. 高通滤波器部分 #####
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / (
        deg_cos(0.707 * 360 / 48) + eps
    )
    hp_factor = (1 - alpha1 / 2) ** 2
    hp_coef2 = 2 * (1 - alpha1)
    hp_coef3 = (1 - alpha1) ** 2

    ##### 2. 超级平滑滤波器（用于 Filt）的系数 #####
    a1 = math.exp(-1.414 * math.pi / LPPeriod)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / LPPeriod)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    ##### 3. 主导周期平滑的系数 #####
    a1_dom = math.exp(-1.414 * math.pi / 250)
    b1_dom = 2 * a1_dom * deg_cos(1.414 * 180 / 250)
    c2_dom = b1_dom
    c3_dom = -a1_dom * a1_dom
    c1_dom = 1 - c2_dom - c3_dom

    # 调用 numba 优化的子函数，完成 heavy loop 部分
    DomCycle = _phase_accumulation_loop(
        close, n, hp_factor, hp_coef2, hp_coef3, c1, c2, c3, c1_dom, c2_dom, c3_dom, eps
    )

    if sequential:
        return DomCycle  # 返回整个周期数组
    else:
        return DomCycle[-1]  # 返回最后一个周期值
