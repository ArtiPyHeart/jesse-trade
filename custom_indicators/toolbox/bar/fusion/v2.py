import numpy as np

from custom_indicators.utils.import_tools import ensure_package

ensure_package("mpire")
from mpire import WorkerPool  # noqa: E402

from custom_indicators.toolbox.bar.fusion.base import FusionBarContainerBase
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret


class FusionBarContainerV2(FusionBarContainerBase):
    """
    Only valid for BTC futures
    """

    def __init__(self, max_bars: int = 1000, threshold: float = 1.3334030570418345):
        super().__init__(max_bars, threshold)

        self.N_1 = 200
        self.N_2 = 121
        self.ENTROPY_N = 72

    @property
    def max_lookback(self) -> int:
        return self.N_1

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        """
        min(abs(r200), add(r121, r72_entropy))
        """
        log_ret_n_1 = np.log(candles[self.N_1 :, 2] / candles[: -self.N_1, 2])

        log_ret_n_2 = np.log(candles[self.N_2 :, 2] / candles[: -self.N_2, 2])[
            self.N_1 - self.N_2 :
        ]

        entropy_log_ret_list = log_ret(
            candles[self.N_1 - self.ENTROPY_N :, :], self.ENTROPY_N
        )

        with WorkerPool() as pool:
            entropy_array = pool.map(sample_entropy_numba, entropy_log_ret_list)

        return np.min([np.abs(log_ret_n_1), log_ret_n_2 + entropy_array], axis=0)


if __name__ == "__main__":
    candles = np.load("data/btc_1m.npy")

    container = FusionBarContainerV2()
    # init
    container.update_with_candles(candles[:-10000])
    # update
    for i in reversed(range(1, 10000)):
        container.update_with_candles(candles[:-i])
        len_timestamp = len(container._unfinished_bars_timestamps)
        len_threshold = len(container._unfinished_bars_thresholds)
        print(
            f"{candles[:-i].shape = } -> {container.is_latest_bar_complete = }, {container.get_fusion_bars().shape = }, {len_timestamp = }, {len_threshold = }"
        )
