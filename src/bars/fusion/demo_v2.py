import numpy as np

from src.bars.fusion.base import FusionBarContainerBase


class DemoV2Bar(FusionBarContainerBase):
    """
    abs(ln(close / close_lag1)) * ln(high / low)
    """

    def __init__(
        self,
        clip_r=0.0055,
        max_bars=-1,
        threshold=0.47,
    ):
        super().__init__(max_bars, threshold)
        self.clip_r = clip_r

    @property
    def max_lookback(self) -> int:
        return 1

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        close_arr = candles[:, 2]
        high_arr = candles[:, 3]
        low_arr = candles[:, 4]
        res = (
            np.abs(np.log(close_arr[1:] / (close_arr[:-1] + 1e-12)))
            * np.log(high_arr[1:] / (low_arr[1:] + 1e-12))
            * 10000
        )
        # 根据 clip_r 过滤：小于 clip_r 的值设为 0
        res[res < self.clip_r] = 0
        return res
