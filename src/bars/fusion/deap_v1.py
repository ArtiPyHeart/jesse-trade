import numpy as np

from src.bars.fusion.base import FusionBarContainerBase


class DeapBarV1(FusionBarContainerBase):
    """
    公式：
    min(
      sub(
        vol_1,
        min(
          price_position,
          vol_1
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
        max_bars=-1,
    ):
        super().__init__(max_bars, 0.3044867289147951)
        self.VOL_1 = 51
        self.R_1 = 88
        self.R_2 = 287

    @property
    def max_lookback(self) -> int:
        return max(self.VOL_1, self.R_1, self.R_2)

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
        log_ret_1 = np.log(close_arr[self.R_1 :] / close_arr[: -self.R_1])
        if self.max_lookback > self.R_1:
            log_ret_1 = log_ret_1[self.max_lookback - self.R_1 :]
        log_ret_2 = np.log(high_arr[self.R_2 :] / high_arr[: -self.R_2])
        if self.max_lookback > self.R_2:
            log_ret_2 = log_ret_2[self.max_lookback - self.R_2 :]

        return np.min(
            [
                log_vol_1 - np.min([price_pos, log_vol_1], axis=0),
                np.min(
                    [np.min([price_pos, np.abs(log_ret_1)], axis=0), np.abs(log_ret_2)],
                    axis=0,
                ),
            ],
            axis=0,
        )
