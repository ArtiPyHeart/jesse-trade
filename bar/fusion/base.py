import warnings
from abc import ABC, abstractmethod

import arrow
import numpy as np


from bar.build import build_bar_by_cumsum


class FusionBarContainerBase(ABC):
    def __init__(self, max_bars: int, threshold: float):
        self.THRESHOLD = threshold
        self.max_bars = max_bars

        # unfinished bars info
        self._unfinished_bars_timestamps: np.ndarray | None = None
        self._unfinished_bars_thresholds: np.ndarray | None = None

        self._merged_bars: np.ndarray | None = None

        self._is_latest_bar_complete = False

    @property
    def is_latest_bar_complete(self) -> bool:
        return self._is_latest_bar_complete

    @property
    def merged_bars(self) -> np.ndarray:
        return self._merged_bars

    @merged_bars.setter
    def merged_bars(self, merged_bars: np.ndarray):
        if len(merged_bars) > self.max_bars:
            merged_bars = merged_bars[-self.max_bars :]
        self._merged_bars = merged_bars

    @property
    @abstractmethod
    def max_lookback(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _init_bars(self, candles: np.ndarray):
        thresholds = self.get_thresholds(candles)
        candles = candles[len(candles) - len(thresholds) :]
        self.merged_bars = build_bar_by_cumsum(
            candles, thresholds, self.THRESHOLD, reverse=False
        )
        if len(self.merged_bars) > 0:
            timestamp_mask = candles[:, 0].astype(int) > self.merged_bars[-1, 0].astype(
                int
            )
            self._unfinished_bars_timestamps = candles[:, 0].astype(int)[timestamp_mask]
            self._unfinished_bars_thresholds = thresholds[timestamp_mask]
        else:
            self._unfinished_bars_timestamps = candles[:, 0].astype(int)
            self._unfinished_bars_thresholds = thresholds

    def _update_bars(self, candles: np.ndarray):
        # 1. 分离所有需要新增的candles
        todo_candles = candles[
            candles[:, 0].astype(int) > self.merged_bars[-1, 0].astype(int)
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
        anchor_index = anchor_index_array[0] - self.max_lookback
        assert (
            anchor_index > 0
        ), f"Not enough data to build fusion bars, {anchor_index = }"
        # 3. 根据anchor_index来计算新的threshold
        new_thresholds = self.get_thresholds(candles[anchor_index:])
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
            self.merged_bars = np.vstack([self.merged_bars, merged_bars])
            self._is_latest_bar_complete = True

            timestamp_mask = self._unfinished_bars_timestamps > self.merged_bars[-1, 0]
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
        if len(candles) < self.max_lookback:
            return
        if len(candles) == 0:
            return

        if self._merged_bars is None or len(self._merged_bars) == 0:
            # 初始化
            self._init_bars(candles)
        else:
            # 更新
            self._update_bars(candles)

    def get_fusion_bars(self) -> np.ndarray:
        return self.merged_bars.copy()
