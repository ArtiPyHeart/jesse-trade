import numba
import numpy as np


@numba.njit
def calculate_entropy(ret_array: np.ndarray, bins: int) -> float:
    hist_count, _ = np.histogram(ret_array, bins=bins)
    probabilities = hist_count / len(ret_array)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


class ReturnArray:
    def __init__(self, window: int, bins: int = 10):
        self.window = window
        self.bins = bins
        self._ret_array = []

    @property
    def entropy(self):
        return calculate_entropy(np.array(self._ret_array), bins=self.bins)

    @property
    def ret_array(self):
        return self._ret_array

    @ret_array.setter
    def ret_array(self, value):
        if isinstance(value, list):
            self._ret_array = value
        elif isinstance(value, np.ndarray):
            value = value.tolist()
            self._ret_array = value
        else:
            raise TypeError(
                f"ret_array must be a list or numpy array, not {type(value)}"
            )

    def append(self, value):
        if len(self._ret_array) > self.window:
            self._ret_array.pop(0)
        self._ret_array.append(value)


@numba.njit
def build_entropy_bar(
    candles: np.ndarray,
    window: int,
    threshold: float,
    bins: int = 10,
    max_bars: int = -1,
):
    bars = np.zeros((candles.shape[0], 6), dtype=np.float64)
    bar_index = 0
    ret_array: list[float] = []

    bar_timestamp = candles[0, 0]
    bar_open = candles[0, 1]
    bar_close = candles[0, 2]
    bar_high = candles[0, 3]
    bar_low = candles[0, 4]
    bar_volume = candles[0, 5]

    candles = candles[1:]
    is_new_bar = False

    for i in range(window):
        ts = candles[i, 0]
        o = candles[i, 1]
        c = candles[i, 2]
        h = candles[i, 3]
        l = candles[i, 4]
        v = candles[i, 5]
        ret = np.log(c) - np.log(bar_close)
        ret_array.append(ret)
        bar_close = c
        bar_high = max(h, bar_high)
        bar_low = min(l, bar_low)
        bar_volume += v

    for i in range(window, candles.shape[0]):
        ts = candles[i, 0]
        o = candles[i, 1]
        c = candles[i, 2]
        h = candles[i, 3]
        l = candles[i, 4]
        v = candles[i, 5]

        ret = np.log(c) - np.log(bar_close)
        ret_array.append(ret)
        if len(ret_array) > window:
            ret_array.pop(0)

        if is_new_bar:
            bar_timestamp = ts
            bar_open = o
            bar_close = c
            bar_high = h
            bar_low = l
            bar_volume += v
            is_new_bar = False
        else:
            bar_close = c
            bar_high = max(h, bar_high)
            bar_low = min(l, bar_low)
            bar_volume += v

        entropy = calculate_entropy(np.array(ret_array), bins=bins)

        if entropy > threshold:
            bars[bar_index] = [
                bar_timestamp,
                bar_open,
                bar_close,
                bar_high,
                bar_low,
                bar_volume,
            ]
            bar_index += 1
            is_new_bar = True

    bars = bars[:bar_index]
    if not is_new_bar:
        uncompleted_bar = [
            bar_timestamp,
            bar_open,
            bar_close,
            bar_high,
            bar_low,
            bar_volume,
        ]
    else:
        uncompleted_bar = None
    if max_bars > 0:
        return bars[-max_bars:], is_new_bar, ret_array, uncompleted_bar
    return bars, is_new_bar, ret_array, uncompleted_bar


class EntropyBarContainer:
    def __init__(
        self, window: int, threshold: float, bins: int = 10, max_bars: int = 5000
    ):
        self.window = window
        self.threshold = threshold
        self.bins = bins
        self.max_bars = max_bars
        self.ret_array: ReturnArray = None

        self.bars = np.zeros((0, 6), dtype=np.float64)

        self.is_new_bar_ready = False

        # 构建中的bar的数据
        self._bar_open = 0.0
        self._bar_close = 0.0
        self._bar_high = 0.0
        self._bar_low = 9999999999999.0
        self._bar_volume = 0.0
        self._bar_timestamp = 0.0

    def _reset_current_bar(self):
        """重置当前构建中的bar数据"""
        self.current_dollar_volume = 0.0
        self._bar_open = 0.0
        self._bar_close = 0.0
        self._bar_high = 0.0
        self._bar_low = 9999999999999.0
        self._bar_volume = 0.0
        self._bar_timestamp = 0.0

    def _load_initial_candles(self, candles: np.ndarray):
        initial_bars, is_new_bar, ret_array, uncompleted_bar = build_entropy_bar(
            candles,
            self.window,
            self.threshold,
            bins=self.bins,
            max_bars=self.max_bars,
        )

        self.bars = initial_bars
        self.is_new_bar_ready = is_new_bar
        self.ret_array.ret_array = ret_array
        if uncompleted_bar is not None:
            self._bar_timestamp = uncompleted_bar[0]
            self._bar_open = uncompleted_bar[1]
            self._bar_close = uncompleted_bar[2]
            self._bar_high = uncompleted_bar[3]
            self._bar_low = uncompleted_bar[4]
            self._bar_volume = uncompleted_bar[5]

    def _update_with_candle(self, candle: np.ndarray):
        candle = candle[-1]
        ts = candle[0]
        o = candle[1]
        c = candle[2]
        h = candle[3]
        l = candle[4]
        v = candle[5]

        ret = np.log(c) - np.log(self._bar_close)
        self.ret_array.append(ret)

        if self.is_new_bar_ready:
            self._bar_timestamp = ts
            self._bar_open = o
            self._bar_close = c
            self._bar_high = h
            self._bar_low = l
            self._bar_volume += v
            self.is_new_bar_ready = False
        else:
            self._bar_close = c
            self._bar_high = max(h, self._bar_high)
            self._bar_low = min(l, self._bar_low)
            self._bar_volume += v

        entropy = self.ret_array.entropy
        if entropy > self.threshold:
            new_bar = np.array(
                [
                    self._bar_timestamp,
                    self._bar_open,
                    self._bar_close,
                    self._bar_high,
                    self._bar_low,
                    self._bar_volume,
                ]
            ).reshape(1, 6)
            self.bars = np.vstack([self.bars, new_bar])
            if self.bars.shape[0] > self.max_bars:
                self.bars = self.bars[1:]

            self._reset_current_bar()
            self.is_new_bar_ready = True
        else:
            # 如果当前bar还未完成，则继续累积
            self.is_new_bar_ready = False

    def update_with_candle(self, candle: np.ndarray):
        if self.bars.size == 0:
            self._load_initial_candles(candle)
        else:
            self._update_with_candle(candle)

    def get_entropy_bars(self, include_uncompleted_bar=False) -> np.ndarray:
        if include_uncompleted_bar:
            if not self.is_new_bar_ready:
                uncompleted_bar = np.array(
                    [
                        self._bar_timestamp,
                        self._bar_open,
                        self._bar_close,
                        self._bar_high,
                        self._bar_low,
                        self._bar_volume,
                    ]
                ).reshape(1, 6)
                return np.vstack([self.bars, uncompleted_bar])
            else:
                return self.bars.copy()
        else:
            return self.bars.copy()

    def is_latest_bar_complete(self) -> bool:
        """
        检查最新的bar是否完成
        :return: True if the latest bar is complete, False otherwise.
        """
        return self.is_new_bar_ready
