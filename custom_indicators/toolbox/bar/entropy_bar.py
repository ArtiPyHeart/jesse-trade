import numba
import numpy as np


@numba.njit
def calculate_entropy(ret_array: np.ndarray, bins: int) -> float:
    hist_count, _ = np.histogram(ret_array, bins=bins)
    probabilities = hist_count / len(ret_array)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


@numba.njit
def build_entropy_bar(
    candles: np.ndarray,
    window: int,
    threshold: float = 1.0,
    bins: int = 10,
    max_bars: int = -1,
) -> tuple[np.ndarray, np.ndarray, bool]:
    bars = np.zeros((candles.shape[0], 6), dtype=np.float64)
    bar_index = 0

    bar_timestamp = candles[0, 0]
    bar_open = candles[0, 1]
    bar_close = candles[0, 2]
    bar_high = candles[0, 3]
    bar_low = candles[0, 4]
    bar_volume = candles[0, 5]

    all_ret_array = np.log(candles[1:, 2]) / np.log(candles[:-1, 2])
    candles = candles[1:]

    for i in range(window):
        bar_close = candles[i, 2]
        bar_high = max(candles[i, 3], bar_high)
        bar_low = min(candles[i, 4], bar_low)
        bar_volume += candles[i, 5]

    is_new_bar = False

    for i in range(window, candles.shape[0]):
        ret_array = all_ret_array[i - window : i]
        entropy = calculate_entropy(ret_array, bins=bins)
        ts = candles[i, 0]
        o = candles[i, 1]
        c = candles[i, 2]
        h = candles[i, 3]
        l = candles[i, 4]
        v = candles[i, 5]

        if entropy <= threshold:
            bar_close = c
            bar_high = max(h, bar_high)
            bar_low = min(l, bar_low)
            bar_volume += v
            is_new_bar = False
        else:
            bars[bar_index, 0] = bar_timestamp
            bars[bar_index, 1] = bar_open
            bars[bar_index, 2] = bar_close
            bars[bar_index, 3] = bar_high
            bars[bar_index, 4] = bar_low
            bars[bar_index, 5] = bar_volume

            bar_index += 1

            bar_timestamp = ts
            bar_open = o
            bar_close = c
            bar_high = h
            bar_low = l
            bar_volume = v

            is_new_bar = True

    bars[bar_index, 0] = bar_timestamp
    bars[bar_index, 1] = bar_open
    bars[bar_index, 2] = bar_close
    bars[bar_index, 3] = bar_high
    bars[bar_index, 4] = bar_low
    bars[bar_index, 5] = bar_volume

    bars = bars[1:bar_index]
    if max_bars > 0:
        bars = bars[-max_bars:]

    return bars, all_ret_array[-window:], is_new_bar


class EntropyBarContainer:
    def __init__(
        self,
        window: int,
        threshold: float = 1.0,
        bins: int = 10,
        max_bars: int = 5000,
    ):
        self.window = window
        self.threshold = threshold
        self.bins = bins
        self.max_bars = max_bars

        self.bars = np.zeros((0, 6), dtype=np.float64)
        self.is_new_bar_ready = False

        self._ret_array = []

    @property
    def ret_array(self):
        return np.array(self._ret_array)

    @ret_array.setter
    def ret_array(self, value):
        if isinstance(value, np.ndarray):
            self._ret_array = value.tolist()
        else:
            self._ret_array.append(float(value))
            if len(self._ret_array) > self.window:
                self._ret_array.pop(0)

    def _load_initial_candles(self, candles: np.ndarray):
        initial_bars, ret_array, is_new_bar = build_entropy_bar(
            candles,
            self.window,
            threshold=self.threshold,
            bins=self.bins,
            max_bars=self.max_bars,
        )

        self.is_new_bar_ready = is_new_bar
        self.bars = initial_bars
        self.ret_array = ret_array

    def _update_with_candle(self, candle: np.ndarray):
        candle = candle[-1]
        ts = candle[0]
        o = candle[1]
        c = candle[2]
        h = candle[3]
        l = candle[4]
        v = candle[5]
        ret = np.log(c) - np.log(self.bars[-1, 2])
        self.ret_array = ret
        entropy = calculate_entropy(self.ret_array, self.bins)

        if entropy <= self.threshold:
            self.bars[-1, 2] = c
            self.bars[-1, 3] = max(h, self.bars[-1, 3])
            self.bars[-1, 4] = min(l, self.bars[-1, 4])
            self.bars[-1, 5] += v
            self.is_new_bar_ready = False
        else:
            new_bar = np.array([ts, o, c, h, l, v], dtype=np.float64)
            self.bars = np.vstack((self.bars, new_bar))
            self.is_new_bar_ready = True

            if self.max_bars > 0 and self.bars.shape[0] > self.max_bars:
                self.bars = self.bars[1:]

    def update_with_candle(self, candle: np.ndarray):
        if self.bars.size == 0:
            self._load_initial_candles(candle)
        else:
            self._update_with_candle(candle)

    def get_entropy_bars(self):
        return self.bars.copy()

    def is_latest_bar_complete(self):
        return self.is_new_bar_ready
