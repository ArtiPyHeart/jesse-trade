import numba
import numpy as np
from jesse import helpers


def estimate_dollar_bar_threshold(candles: np.ndarray, num_minutes: int) -> float:
    """
    通过jesse的1分钟k线数据，估计出更高时间周期的dollar bar的阈值
    candles: np.ndarray, shape=(num_candles, 6), columns=['timestamp', 'open', 'close', 'high', 'low', 'volume']
    num_minutes: 参考的更高时间周期的分钟数
    """
    if candles.size == 0:
        return 0.0
    # 使用close价格计算每根k线的成交额
    dollar_volume = helpers.get_candle_source(
        candles, source_type="close"
    ) * helpers.get_candle_source(candles, source_type="volume")
    total_dollar_volume = dollar_volume.sum()
    # 期望的时间周期k线数量
    expected_bar_count = candles.shape[0] // num_minutes

    # 下面用二分法来搜索最优阈值，使 build_dollar_bar(candles, threshold) 的输出
    # bar 数量与 expected_bar_count 相差不超过 1%

    # 将二分搜索的阈值区间设为 [min_threshold, max_threshold]
    # 可根据实际需求灵活调整这里的上下界
    min_threshold = 1.0
    max_threshold = total_dollar_volume * 5  # 粗略给个较高上限

    # 最大迭代次数（防止循环太久）
    max_iterations = 100
    best_threshold = (min_threshold + max_threshold) / 2

    for _ in range(max_iterations):
        mid_threshold = (min_threshold + max_threshold) / 2

        # 调用本文件中的 build_dollar_bar (numba.njit) 函数来检查产出数量
        bars = build_dollar_bar(candles, mid_threshold)
        bar_count = bars.shape[0]

        # 计算 bar 数量与期望值的偏差
        error_ratio = abs(bar_count - expected_bar_count) / (expected_bar_count + 1e-9)

        if error_ratio <= 0.01:
            # 如果误差已在1%以内，直接返回
            return mid_threshold

        # 如果 bar_count 比期望多，说明 threshold 太小，需要增加阈值
        if bar_count > expected_bar_count:
            min_threshold = mid_threshold
        else:
            # bar_count 太少，说明 threshold 太大
            max_threshold = mid_threshold

        best_threshold = mid_threshold

    # 如果在 max_iterations 次迭代内未达成 1% 的精度，就返回当前 best_threshold
    return best_threshold


@numba.njit
def build_dollar_bar(candles: np.ndarray, threshold: float) -> np.ndarray:
    """
    通过jesse的1分钟k线数据，构建更高时间周期的dollar bar，由近到远构建dollar bar
    candles: np.ndarray, shape=(num_candles, 6), columns=['timestamp', 'open', 'close', 'high', 'low', 'volume']
    threshold: dollar bar的阈值，约等于成交量*价格

    返回: np.ndarray, shape=(num_dollar_bars, 6), 与输入candles格式相同
    """
    n = candles.shape[0]
    # 预分配一个足够大的数组存储中间结果
    bars = np.zeros((n, 6), dtype=np.float64)
    bar_index = 0

    # 用于累积 dollar bar
    current_dollar_volume = 0.0
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

        dv = c * v  # 本根K线的 dollar volume
        bar_timestamp = ts

        if is_new_bar:
            # 开启新bar时，保存这根(离现在最近的)K线的close作为bar_close
            bar_open = o
            bar_close = c  # ← 只在新bar时记录 close
            bar_high = h
            bar_low = l
            bar_volume = v
            current_dollar_volume = dv
            is_new_bar = False
        else:
            # 对 open 取更早的 K 线的 open（因为时间更远、开盘更早）
            bar_open = o
            bar_high = max(h, bar_high)
            bar_low = min(l, bar_low)
            bar_volume += v
            current_dollar_volume += dv

        # 达到阈值后把这根 bar 放进结果
        if current_dollar_volume > threshold:
            bars[bar_index, 0] = bar_timestamp
            bars[bar_index, 1] = bar_open
            bars[bar_index, 2] = bar_close
            bars[bar_index, 3] = bar_high
            bars[bar_index, 4] = bar_low
            bars[bar_index, 5] = bar_volume

            # 重置
            bar_index += 1
            is_new_bar = True
            current_dollar_volume = 0.0
            bar_open = 0.0
            bar_close = 0.0
            bar_high = 0.0
            bar_low = 9999999999999.0
            bar_volume = 0.0

    # 舍弃未达阈值的剩余部分 —— 用切片替代原先的for循环
    return bars[:bar_index][::-1]


class DollarBarContainer:
    """
    Dollar Bar容器类，用于在jesse回测和实盘交易中管理和构建dollar bar

    该类能够：
    1. 存储和管理dollar bar，确保不超过最大数量限制
    2. 在首次加载时构建初始dollar bar
    3. 在后续每分钟轮询时更新dollar bar
    4. 提供完整构建的dollar bar数据
    5. 提供状态指示器，表明最新dollar bar是否已完成构建
    """

    def __init__(self, threshold: float, max_bars: int = 500):
        """
        初始化Dollar Bar容器

        参数:
            threshold: float - dollar bar的阈值 (price * volume)
            max_bars: int - 最大保存的dollar bar数量
        """
        self.threshold = threshold
        self.max_bars = max_bars

        # 存储完整构建的dollar bar
        self.bars = np.zeros((0, 6), dtype=np.float64)

        # 存储当前正在构建的dollar bar
        self.current_dollar_volume = 0.0
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

    def load_initial_candles(self, candles: np.ndarray):
        """
        加载初始的所有1分钟K线数据，构建初始dollar bar
        适用于jesse回测开始或实盘交易启动时

        参数:
            candles: np.ndarray - jesse的1分钟K线数据，按照时间从远到近排列
        """
        if candles.size == 0:
            return

        # 使用现有的build_dollar_bar函数构建初始dollar bar
        initial_bars = build_dollar_bar(candles, self.threshold)

        # 限制数量不超过max_bars
        if initial_bars.shape[0] > self.max_bars:
            initial_bars = initial_bars[-self.max_bars :]

        self.bars = initial_bars

        # 初始化当前构建中的bar
        self._reset_current_bar()
        self.is_new_bar_ready = False

    def update_with_candle(self, candle: np.ndarray):
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

        # 计算当前K线的dollar volume
        dv = c * v

        # 记录时间戳
        self.bar_timestamp = ts

        # 更新当前构建中的bar数据
        if self.current_dollar_volume == 0.0:  # 开始新的bar
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

        # 累加dollar volume
        self.current_dollar_volume += dv

        # 检查是否达到阈值
        if self.current_dollar_volume > self.threshold:
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
            self.is_new_bar_ready = False

    def get_dollar_bars(self) -> np.ndarray:
        """
        获取当前所有完整构建的dollar bars

        返回:
            np.ndarray - 所有完整的dollar bars，不包含当前构建中的bar
        """
        return self.bars.copy()

    def is_latest_bar_complete(self) -> bool:
        """
        检查最新的dollar bar是否构建完成

        返回:
            bool - 如果最新的dollar bar刚刚构建完成，返回True，否则返回False
        """
        return self.is_new_bar_ready
