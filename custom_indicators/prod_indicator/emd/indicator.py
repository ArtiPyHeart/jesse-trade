import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from mpire.pool import WorkerPool

from custom_indicators.prod_indicator.emd.nrbo import nrbo
from custom_indicators.prod_indicator.emd.vmdpy import VMD

ALPHA = 2000  ###  数据保真度约束
TAU = 0.0  ###  噪声容限
K = 5  ###  模态数量
DC = 0  ###  直流分量
INIT = 1  ###  初始化中心频率
TOL = 1e-7  ### 收敛容忍度


def _calc_vmd_nrbo(src: np.ndarray):
    u, u_hat, omega = VMD(src, ALPHA, TAU, K, DC, INIT, TOL)
    u = u[2:]
    u_nrbo = np.zeros_like(u)
    for i in range(u.shape[0]):
        u_nrbo[i] = nrbo(u[i])
    return u_nrbo.T[-1].tolist()


def vmd_indicator(
    candles: np.ndarray,
    window: int,
    parallel: bool = False,
    source_type: str = "close",
    sequential: bool = False,
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    if sequential:
        if parallel:
            res = [src[idx - window : idx] for idx in range(window, len(src))]
            with WorkerPool() as pool:
                res = pool.map(_calc_vmd_nrbo, res)
        else:
            res = []
            for idx in range(window, len(src)):
                res.append(_calc_vmd_nrbo(src[idx - window : idx]))
        res = np.asarray(res)
        columns = res.shape[1]
        # padding res with nan (window, columns)
        res = np.vstack([np.full((window, columns), np.nan), res])
        return res
    else:
        return np.asarray(_calc_vmd_nrbo(src[-window:]))
