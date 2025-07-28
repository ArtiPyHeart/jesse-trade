import numpy as np
from bar.fusion.base import FusionBarContainerBase


class DeapBarV1(FusionBarContainerBase):
    """
    公式：
    min(
      sub(
        vol_1,
        min(
          price_position,
          vol_2
        )
      ),
      min(
        min(
          price_position,
          abs(r_1)
        ),
        abs(r_2)
      )
    )
    """

    def __init__(
        self,
        VOL_1: int,
        VOL_2: int,
        R_1: int,
        R_2: int,
        threshold: float,
        max_bars=50000,
    ):
        super().__init__(max_bars, threshold)
        self.VOL_1 = VOL_1
        self.VOL_2 = VOL_2
        self.R_1 = R_1
        self.R_2 = R_2

    @property
    def max_lookback(self) -> int:
        return max(self.VOL_1, self.VOL_2, self.R_1, self.R_2)

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        close_arr = candles[:, 2]
        high_arr = candles[:, 3]
        low_arr = candles[:, 4]
        vol_arr = candles[:, 5]
        price_pos = (close_arr - low_arr) / (high_arr - low_arr + 1e-10)
        price_pos = price_pos[self.max_lookback :]

        log_vol_1 = np.log(vol_arr[self.VOL_1 :] / vol_arr[: -self.VOL_1])
        if self.max_lookback > self.VOL_1:
            log_vol_1 = log_vol_1[self.max_lookback - self.VOL_1 :]
        log_vol_2 = np.log(vol_arr[self.VOL_2 :] / vol_arr[: -self.VOL_2])
        if self.max_lookback > self.VOL_2:
            log_vol_2 = log_vol_2[self.max_lookback - self.VOL_2 :]
        log_ret_1 = np.log(close_arr[self.R_1 :] / close_arr[: -self.R_1])
        if self.max_lookback > self.R_1:
            log_ret_1 = log_ret_1[self.max_lookback - self.R_1 :]
        log_ret_2 = np.log(high_arr[self.R_2 :] / high_arr[: -self.R_2])
        if self.max_lookback > self.R_2:
            log_ret_2 = log_ret_2[self.max_lookback - self.R_2 :]

        return np.min(
            [
                log_vol_1 - np.min([price_pos, log_vol_2], axis=0),
                np.min(
                    [np.min([price_pos, np.abs(log_ret_1)], axis=0), np.abs(log_ret_2)],
                    axis=0,
                ),
            ],
            axis=0,
        )
