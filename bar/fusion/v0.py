import numpy as np
from joblib import delayed, Parallel

from bar.fusion.base import FusionBarContainerBase
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret_from_candles


class FusionBarContainerV0(FusionBarContainerBase):
    def __init__(self, max_bars=5000):
        super().__init__(max_bars, 0.3166890909253018)
        self.N_1 = 10
        self.N_2 = 168
        self.N_ENTROPY = 54

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

        entropy_array = Parallel()(
            delayed(sample_entropy_numba)(i) for i in entropy_log_ret_list
        )
        entropy_array = np.array(entropy_array)
        threshold = np.min([np.abs(log_ret_n_1), log_ret_n_2 + entropy_array], axis=0)
        return threshold
