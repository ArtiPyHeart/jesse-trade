import numpy as np

from src.bars.fusion.base import FusionBarContainerBase


class DemoBar(FusionBarContainerBase):
    """
    abs(close - close_lag1) * (high - low) / close
    """

    def __init__(
        self,
        max_bars=50000,
        threshold=0.2,
    ):
        super().__init__(max_bars, threshold)

    @property
    def max_lookback(self) -> int:
        return 1

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        close_arr = candles[:, 2]
        high_arr = candles[:, 3]
        low_arr = candles[:, 4]
        res = (
            np.abs(close_arr[1:] - close_arr[:-1])
            * (high_arr[1:] - low_arr[1:])
            / close_arr[1:]
        )
        return res
