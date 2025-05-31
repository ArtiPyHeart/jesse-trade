import warnings

import arrow
import numpy as np
from mpire import WorkerPool

from custom_indicators.toolbox.bar.build import build_bar_by_cumsum
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret


class FusionBarContainerV1:
    SHORT_N = 21
    LONG_N = 180
    ENTROPY_N = 34
    THRESHOLD = 0.8548507667918396

    def __init__(self):
        # unfinished bars info
        self._unfinished_bars_timestamps: np.ndarray | None = None
        self._unfinished_bars_thresholds: np.ndarray | None = None

        self._merged_bars: np.ndarray | None = None

        self._is_latest_bar_complete = False

    def _get_thresholds(self, candles: np.ndarray) -> np.ndarray:
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

    def _init_bars(self, candles: np.ndarray):
        thresholds = self._get_thresholds(candles)
        candles = candles[len(candles) - len(thresholds) :]
        self._merged_bars = build_bar_by_cumsum(
            candles, thresholds, self.THRESHOLD, reverse=False
        )
        timestamp_mask = candles[:, 0].astype(int) > self._merged_bars[-1, 0].astype(
            int
        )
        self._unfinished_bars_timestamps = candles[:, 0].astype(int)[timestamp_mask]
        self._unfinished_bars_thresholds = thresholds[timestamp_mask]

    def _update_bars(self, candles: np.ndarray):
        # 1. 分离所有需要新增的candles
        todo_candles = candles[
            candles[:, 0].astype(int) > self._merged_bars[-1, 0].astype(int)
        ]
        # 2. 根据_unfinished_bars_timestamps来确定新的threshold从哪个index开始新增
        anchor_index_array = np.where(
            candles[:, 0].astype(int) > self._unfinished_bars_timestamps[-1]
        )[0]
        if len(anchor_index_array) == 0:
            warnings.warn(
                f"No new candles after {arrow.get(int(self._unfinished_bars_timestamps[-1])).format('YYYY-MM-DD HH:mm:ss ZZ')}",
                UserWarning,
            )
            return
        anchor_index = anchor_index_array[0] - self.LONG_N
        assert anchor_index > 0, (
            f"Not enough data to build fusion bars, {anchor_index = }"
        )
        # 3. 根据anchor_index来计算新的threshold
        new_thresholds = self._get_thresholds(candles[anchor_index:])
        # 4. 根据新的threshold来更新_unfinished_bars_thresholds
        self._unfinished_bars_thresholds = np.hstack(
            [self._unfinished_bars_thresholds, new_thresholds]
        )
        self._unfinished_bars_timestamps = todo_candles[:, 0].astype(int)
        merged_bars = build_bar_by_cumsum(
            todo_candles,
            self._unfinished_bars_thresholds,
            self.THRESHOLD,
            reverse=False,
        )
        if len(merged_bars) > 0:
            self._merged_bars = np.vstack([self._merged_bars, merged_bars])
            self._is_latest_bar_complete = True

            timestamp_mask = self._unfinished_bars_timestamps > self._merged_bars[-1, 0]
            self._unfinished_bars_timestamps = self._unfinished_bars_timestamps[
                timestamp_mask
            ]
            self._unfinished_bars_thresholds = self._unfinished_bars_thresholds[
                timestamp_mask
            ]
        else:
            self._is_latest_bar_complete = False

    def update_with_candles(self, candles: np.ndarray):
        candles = candles[candles[:, 5] > 0]
        if len(candles) == 0:
            return

        if self._unfinished_bars_timestamps is None:
            # 初始化
            self._init_bars(candles)
        else:
            # 更新
            self._update_bars(candles)

    def get_fusion_bars(self) -> np.ndarray:
        return self._merged_bars.copy()

    @property
    def is_latest_bar_complete(self) -> bool:
        return self._is_latest_bar_complete


if __name__ == "__main__":
    candles = np.load("data/btc_1m.npy")

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
