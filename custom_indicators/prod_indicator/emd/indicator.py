import numpy as np
from jesse.helpers import get_candle_source, slice_candles

from custom_indicators.prod_indicator.emd.nrbo import nrbo
from custom_indicators.prod_indicator.emd.vmdpy import VMD

ALPHA = 2000  ###  数据保真度约束
TAU = 0.0  ###  噪声容限
K = 5  ###  模态数量
DC = 0  ###  直流分量
INIT = 1  ###  初始化中心频率
TOL = 1e-7  ### 收敛容忍度


def vmd_indicator(
    candles: np.ndarray,
    source_type: str = "close",
    sequential: bool = False,
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    u, u_hat, omega = VMD(src, ALPHA, TAU, K, DC, INIT, TOL)
    # 排除前两行u
    u = u[2:]

    u_nrbo = np.zeros_like(u)

    for i in range(u.shape[0]):
        u_nrbo[i] = nrbo(u[i])

    if sequential:
        return u_nrbo.T  # (row, 3)
    else:
        return u_nrbo.T[-1, :]  # (3,)
