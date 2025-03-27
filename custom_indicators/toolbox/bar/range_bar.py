import numba
import numpy as np


@numba.njit
def build_range_bar(
    candles: np.ndarray,
    threshold: float,
    max_bars: int = -1,
) -> np.ndarray:
    """
    Build range bars from a given set of candles.

    Args:
        candles (np.ndarray): Input candles with shape (n, 6).
        threshold (float): The threshold for range bar construction.
        max_bars (int): Maximum number of bars to return. Default is -1 (no limit).

    Returns:
        np.ndarray: Constructed range bars with shape (m, 6).
    """
    n = candles.shape[0]
    # 预分配一个足够大的数组存储中间结果
    bars = np.zeros((n, 6), dtype=np.float64)
    bar_index = 0

    # 用于累积 dollar bar
    bar_open = 0.0
    bar_close = 0.0
    bar_high = 0.0
    bar_low = 9999999999999.0
    bar_volume = 0.0
    bar_timestamp = 0.0

    is_new_bar = True

    # 从时间最近（candles末尾）向时间最早（candles开头）构建
    for i in range(n - 1, -1, -1):
        ts = candles[i, 0]
        o = candles[i, 1]
        c = candles[i, 2]
        h = candles[i, 3]
        l = candles[i, 4]
        v = candles[i, 5]

        bar_timestamp = ts

        if is_new_bar:
            # 开启新bar时，保存这根(离现在最近的)K线的close作为bar_close
            bar_open = o
            bar_close = c  # ← 只在新bar时记录 close
            bar_high = h
            bar_low = l
            bar_volume = v
            is_new_bar = False
        else:
            # 对 open 取更早的 K 线的 open（因为时间更远、开盘更早）
            bar_open = o
            bar_high = max(h, bar_high)
            bar_low = min(l, bar_low)
            bar_volume += v

        # 达到阈值后把这根 bar 放进结果
        price_change = np.log(max(bar_close, bar_open)) - np.log(
            min(bar_close, bar_open)
        )
        if price_change >= threshold:
            # 记录 bar
            bars[bar_index, 0] = bar_timestamp
            bars[bar_index, 1] = bar_open
            bars[bar_index, 2] = bar_close
            bars[bar_index, 3] = bar_high
            bars[bar_index, 4] = bar_low
            bars[bar_index, 5] = bar_volume

            # 更新状态
            bar_index += 1
            is_new_bar = True
            current_dollar_volume = 0.0
            bar_open = 0.0
            bar_close = 0.0
            bar_high = 0.0
            bar_low = 9999999999999.0
            bar_volume = 0.0

            if 0 < max_bars < bar_index:
                break

    return bars[:bar_index][::-1]


class RangeBarContainer:
    """
    Range Bar容器类，用于在jesse回测和实盘交易中管理和构建range bar

    该类能够：
    1. 存储和管理range bar，确保不超过最大数量限制
    2. 在首次加载时构建初始range bar
    3. 在后续每分钟轮询时更新range bar
    4. 提供完整构建的range bar数据
    5. 提供状态指示器，表明最新range bar是否已完成构建

    """

    def __init__(self, threshold: float, max_bars: int = 5000):
        self.threshold = threshold
        self.max_bars = max_bars

        # 存储构建完成的range bar
        self.bars = np.zeros((0, 6), dtype=np.float64)

        # 存储当前正在构建的的range bar
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
        """
        加载初始的所有1分钟K线数据，构建初始dollar bar
        适用于jesse回测开始或实盘交易启动时

        参数:
            candles: np.ndarray - jesse的1分钟K线数据，按照时间从远到近排列
        """
        if candles.size == 0:
            return

        initial_bars = build_range_bar(candles, self.threshold, max_bars=self.max_bars)

        self.bars = initial_bars

        # 初始化当前构建中的bar
        self._reset_current_bar()
        self.is_new_bar_ready = False

    def _update_with_candle(self, candle: np.ndarray):
        """
        使用最新的一根1分钟K线更新dollar bar
        适用于jesse每分钟轮询时调用

        参数:
            candle: np.ndarray - 最新的一根jesse 1分钟K线，形状为(6,)
        """
        if candle.size == 0:
            return

        # 只取最后一行
        if len(candle.shape) > 1 and candle.shape[0] > 1:
            candle = candle[-1]

        # 提取candle数据
        ts = candle[0]
        o = candle[1]
        c = candle[2]
        h = candle[3]
        l = candle[4]
        v = candle[5]

        # 记录时间戳
        self.bar_timestamp = ts

        # 更新当前构建中的bar数据
        if self.is_new_bar_ready:  # 开始新的bar
            self._bar_open = o
            self._bar_close = c
            self._bar_high = h
            self._bar_low = l
            self._bar_volume = v
        else:  # 继续累积当前bar
            # bar_open已经是最早K线的open，不需要更新
            self._bar_close = c  # 更新为最新close
            self._bar_high = max(h, self._bar_high)
            self._bar_low = min(l, self._bar_low)
            self._bar_volume += v

        price_change = np.log(max(self._bar_close, self._bar_open)) - np.log(
            min(self._bar_close, self._bar_open)
        )
        if price_change >= self.threshold:
            # 构建新的完整bar
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

            # 将新bar添加到现有bars中
            self.bars = np.vstack([self.bars, new_bar])

            # 如果超出最大数量限制，移除最旧的bar
            if self.bars.shape[0] > self.max_bars:
                self.bars = self.bars[1:]

            # 重置当前bar并设置标志
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

    def get_range_bars(self) -> np.ndarray:
        """
        获取当前所有完整构建的range bars

        返回:
            np.ndarray - 所有完整的range bars，不包含当前构建中的bar
        """
        return self.bars.copy()

    def is_latest_bar_complete(self) -> bool:
        """
        检查最新的range bar是否已完成构建

        返回:
            bool - 如果最新的range bar已完成构建，则返回True，否则返回False
        """
        return self.is_new_bar_ready
