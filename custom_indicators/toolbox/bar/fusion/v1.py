import numpy as np

from custom_indicators.utils.import_tools import ensure_package

ensure_package("mpire")
from mpire import WorkerPool  # noqa: E402

from custom_indicators.toolbox.bar.fusion.base import FusionBarContainerBase
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret


class FusionBarContainerV1(FusionBarContainerBase):
    """
    Only valid for BTC futures
    """

    def __init__(self, max_bars: int = 1000, threshold: float = 0.8548507667918396):
        super().__init__(max_bars, threshold)

        self.SHORT_N = 21
        self.LONG_N = 180
        self.ENTROPY_N = 34

    @property
    def max_lookback(self) -> int:
        return self.LONG_N

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        """
        min(abs(r21), add(r180, r34_entropy))
        """
        log_ret_short_n = np.log(
            candles[self.SHORT_N :, 2] / candles[: -self.SHORT_N, 2]
        )[self.LONG_N - self.SHORT_N :]

        log_ret_long_n = np.log(candles[self.LONG_N :, 2] / candles[: -self.LONG_N, 2])
        entropy_log_ret_list = log_ret(
            candles[self.LONG_N - self.ENTROPY_N :, :], self.ENTROPY_N
        )

        with WorkerPool() as pool:
            entropy_array = pool.map(sample_entropy_numba, entropy_log_ret_list)

        return np.min([np.abs(log_ret_short_n), log_ret_long_n + entropy_array], axis=0)


if __name__ == "__main__":
    candles = np.load("/Users/yangqiuyu/Github/jesse-trade/data/btc_1m.npy")

    container = FusionBarContainerV1()
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
