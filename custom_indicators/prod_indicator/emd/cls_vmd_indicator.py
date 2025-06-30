import numpy as np
from jesse.helpers import get_candle_source
from joblib import Parallel, delayed

from custom_indicators.prod_indicator._indicator_base._cls_ind import IndicatorBase
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
    return u_nrbo.T


class VMD_NRBO(IndicatorBase):
    def __init__(
        self,
        candles: np.ndarray,
        window: int,
        source_type: str = "close",
        sequential: bool = False,
    ):
        super().__init__(candles, sequential)
        self.window = window
        self.src = get_candle_source(candles, source_type)

        self.process()

    def _single_process(self):
        single_res = _calc_vmd_nrbo(self.src[-self.window :])
        self.raw_result.append(single_res)

    def _sequential_process(self):
        src_with_window = [
            self.src[idx - self.window : idx]
            for idx in range(self.window, len(self.src) + 1)
        ]
        res = Parallel(n_jobs=-2)(delayed(_calc_vmd_nrbo)(i) for i in src_with_window)

        self.raw_result.extend(res)
