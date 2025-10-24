import numpy as np
from jesse.helpers import get_candle_source

from pyrs_indicators.ind_decomposition import vmd
from pyrs_indicators._core import _rust_nrbo  # NRBO 内部使用

from src.indicators.prod._indicator_base._cls_ind import IndicatorBase

ALPHA = 2000  ###  数据保真度约束
TAU = 0.0  ###  噪声容限
K = 5  ###  模态数量
DC = 0  ###  直流分量
INIT = 1  ###  初始化中心频率
TOL = 1e-7  ### 收敛容忍度


def _calc_vmd_nrbo(src: np.ndarray):
    """
    Calculate VMD + NRBO using Rust implementation.

    Rust version provides 50-100x speedup over Python/Numba implementation
    while maintaining numerical alignment (error < 1e-10).
    """
    # 使用新的 Python 接口
    u, u_hat, omega = vmd(
        src, alpha=ALPHA, tau=TAU, K=K, DC=bool(DC), init=INIT, tol=TOL, return_full=True
    )
    u = u[2:]  # Skip first 2 modes
    u_nrbo = np.zeros_like(u)
    for i in range(u.shape[0]):
        u_nrbo[i] = _rust_nrbo(u[i], max_iter=10, tol=1e-6)
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
        # Rust implementation is fast enough without parallel processing
        res = [_calc_vmd_nrbo(i) for i in src_with_window]

        self.raw_result.extend(res)
