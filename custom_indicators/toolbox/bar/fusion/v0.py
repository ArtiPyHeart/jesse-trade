import numpy as np
from mpire.pool import WorkerPool

from custom_indicators.toolbox.bar.fusion.base import FusionBarContainerBase
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret_from_candles


class FusionBarContainerV0(FusionBarContainerBase):
    def __init__(self, max_bars=5000):
        super().__init__(max_bars, 0.6204344631779585)
        self.N_1 = 244
        self.N_2 = 201
        self.N_ENTROPY = 140

    @property
    def max_lookback(self) -> int:
        return max(self.N_1, self.N_2, self.N_ENTROPY)

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        log_ret_n_1 = np.log(candles[self.N_1 :, 2] / candles[: -self.N_1, 2])
        if self.max_lookback > self.N_1:
            log_ret_n_1 = log_ret_n_1[self.max_lookback - self.N_1 :]
        log_ret_n_2 = np.log(candles[self.N_2 :, 2] / candles[: -self.N_2, 2])
        if self.max_lookback > self.N_2:
            log_ret_n_2 = log_ret_n_2[self.max_lookback - self.N_2 :]

        if self.max_lookback > self.N_ENTROPY:
            entropy_log_ret_list = log_ret_from_candles(
                candles[self.max_lookback - self.N_ENTROPY :], self.N_ENTROPY
            )
        else:
            entropy_log_ret_list = log_ret_from_candles(candles, self.N_ENTROPY)

        with WorkerPool() as pool:
            entropy_array = pool.map(sample_entropy_numba, entropy_log_ret_list)

        return np.min([np.abs(log_ret_n_1), log_ret_n_2 + entropy_array], axis=0)
