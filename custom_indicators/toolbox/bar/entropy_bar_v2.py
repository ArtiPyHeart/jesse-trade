import numpy as np
from joblib import parallel_backend, delayed

from custom_indicators.toolbox.bar.build import build_bar_by_cumsum
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import log_ret_from_candles
from custom_indicators.utils.parallel import joblib_pool
from custom_indicators.volitility_indicator.yang_zhang import yang_zhang_volatility


class EntropyBarContainer:
    MIN_WINDOW = 20
    MAX_WINDOW = 60 * 24

    def __init__(
        self,
        window: int,
        window_vol_t: int,
        window_vol_ref: int,
        entropy_threshold: float,
        max_bars: int = 5000,
    ):
        self.window = window
        self.window_vol_t = window_vol_t
        self.window_vol_ref = window_vol_ref
        self.entropy_threshold = entropy_threshold
        self.max_bars = max_bars

        self.latest_timestamp = 0
        self.bars = np.empty((0, 6), dtype=np.float64)
        self._is_latest_bar_complete = False

        self._unfinished_bars = np.empty((0, 6), dtype=np.float64)
        self._unfinished_bars_entropy = []

    def _check_bar_limit(self):
        if self.bars.shape[0] > self.max_bars:
            self.bars = self.bars[-self.max_bars :]

    def update_with_candle(self, candles: np.ndarray):
        if self.bars.size == 0:
            candle_for_vol = candles
        else:
            if self._unfinished_bars.size == 0:
                # 没有遗留bar
                candle_for_vol = candles[
                    candles[:, 0].astype(int)
                    >= self.latest_timestamp - self.MAX_WINDOW * 60000
                ]
            else:
                # 已经有部分遗留bar
                candle_for_vol = candles[
                    candles[:, 0].astype(int)
                    >= int(self._unfinished_bars[-1, 0]) - self.MAX_WINDOW * 60000
                ]

        vol_t = yang_zhang_volatility(
            candle_for_vol, period=self.window_vol_t, sequential=True
        )
        vol_ref = yang_zhang_volatility(
            candle_for_vol, period=self.window_vol_ref, sequential=True
        )

        window_on_vol = self.window * vol_ref / (vol_t + 1e-10)
        window_on_vol = np.clip(window_on_vol, self.MIN_WINDOW, self.MAX_WINDOW)
        log_ret_list = log_ret_from_candles(candle_for_vol, window_on_vol)
        with parallel_backend(joblib_pool._backend):
            entropy_array = [delayed(sample_entropy_numba)(i) for i in log_ret_list]
        # 统一处理长度
        if self.bars.size == 0:
            len_gap = len(candles) - len(entropy_array)
            candle_to_merge = candles[len_gap:]
        else:
            # 要考虑unfinished bar的情况
            candle_to_stack = candles[
                candles[:, 0].astype(int) > int(self._unfinished_bars[-1, 0])
            ]
            if len(entropy_array) > len(candle_to_stack):
                len_gap = len(entropy_array) - len(candle_to_stack)
                entropy_array = entropy_array[len_gap:]
            elif len(entropy_array) < len(candle_to_stack):
                raise ValueError("Not enough entropy calculated.")
            else:
                pass
            self._unfinished_bars = np.vstack((self._unfinished_bars, candle_to_stack))
            self._unfinished_bars_entropy = (
                self._unfinished_bars_entropy + entropy_array
            )
            candle_to_merge = self._unfinished_bars
            entropy_array = self._unfinished_bars_entropy

        merged_bars = build_bar_by_cumsum(
            candle_to_merge,
            entropy_array,
            self.entropy_threshold,
            reverse=False,
        )

        if len(merged_bars) > 0:
            if self.bars.size == 0:
                # 初始化的情况
                self.bars = merged_bars
                self.latest_timestamp = int(self.bars[-1, 0])
            else:
                # 新增bar的情况
                self.bars = np.vstack((self.bars, merged_bars))
                self.latest_timestamp = int(self.bars[-1, 0])

            unfinish_mask = candle_to_merge[:, 0].astype(int) > self.latest_timestamp
            self._unfinished_bars = candle_to_merge[unfinish_mask]
            self._unfinished_bars_entropy = [
                e for e, u in zip(entropy_array, unfinish_mask) if u
            ]

            if len(self._unfinished_bars) == 1:
                self._is_latest_bar_complete = True
            else:
                self._is_latest_bar_complete = False
        else:
            self._is_latest_bar_complete = False

        self._check_bar_limit()

    def get_entropy_bar(self) -> np.ndarray:
        return self.bars.copy()

    def is_latest_bar_complete(self) -> bool:
        return self._is_latest_bar_complete


if __name__ == "__main__":
    candles = np.load("/Users/yangqiuyu/Github/jesse-trade/data/btc_1m.npy")[-200000:]
    entropy_bar_container = EntropyBarContainer(
        window=52,
        window_vol_t=55,
        window_vol_ref=9424,
        entropy_threshold=53.37804908688859,
    )
    entropy_bar_container.update_with_candle(candles[:-10000])
    for i in reversed(range(10000)):
        entropy_bar_container.update_with_candle(candles[:-i])
        is_complete = entropy_bar_container.is_latest_bar_complete()
        latest_timestamp = entropy_bar_container.latest_timestamp
        print(
            f"{i} {latest_timestamp = }: {is_complete = } {len(entropy_bar_container._unfinished_bars) = }"
        )
