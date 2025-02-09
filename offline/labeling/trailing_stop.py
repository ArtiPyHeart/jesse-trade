import jesse.indicators as ta
import numpy as np
from jesse import helpers


def _trailing_stop_label(candles, n_bar=15, min_r=0.00025, k=1.5, vol_break_only=False):
    """
    通过当前K线到最多n_bar后的K线，找到第一个回报率大于target_r的K线，进行标记，1=多方持仓，-1=空方持仓，0=空仓
    candles: np.ndarray, jesse生成的k线
    n_bar: int, 计算trailing stop的bar数
    min_r: float, 最小回报率，至少要高过交易所的手续费磨损
    k: float, natr的倍数回报率
    """
    LABELING_INDEX = 0
    SKIP_INDEX = 0
    LABEL = 0
    labels = np.zeros(candles.shape[0], dtype=np.int8)
    realized_r = np.zeros(candles.shape[0], dtype=np.float64)
    bar_duration = np.zeros(candles.shape[0], dtype=np.int64)

    close = helpers.get_candle_source(candles, "close")
    natr = ta.natr(candles, sequential=True) * k / 100
    vol, vol_ma = ta.volume(candles, sequential=True)
    for idx, (p, r) in enumerate(zip(close, natr)):
        if np.isnan(r) or candles.shape[0] - idx < n_bar or idx < LABELING_INDEX:
            continue
        else:
            LABELING_INDEX = idx
            SKIP_INDEX = idx + n_bar + 1
            target_r = r if r > min_r else min_r
            # 找到第一个大于target_r的r
            for i in range(idx, idx + n_bar + 1):
                current_r = close[i] / p - 1
                if np.abs(current_r) > target_r:
                    LABEL = 1 if current_r > 0 else -1
                    SKIP_INDEX = i
                    break
            if vol_break_only:
                # 从idx到i中, 找到第一个volume大于vol_ma的index作为起点，如果找不到，则维持原状
                for j in range(idx, SKIP_INDEX):
                    if vol[j] > vol_ma[j]:
                        LABELING_INDEX = j
                        break
            # 找到起点后还需要修正真正的回报率，如果回报率小于target_r，则不标记
            real_r = close[SKIP_INDEX] / close[LABELING_INDEX] - 1
            if np.abs(real_r) < target_r:
                LABEL = 0

            if LABEL != 0:
                labels[LABELING_INDEX] = LABEL
                realized_r[LABELING_INDEX] = (
                    close[SKIP_INDEX] / close[LABELING_INDEX] - 1
                )
                bar_duration[LABELING_INDEX] = SKIP_INDEX - LABELING_INDEX
    return labels, realized_r, bar_duration


def _adaptive_trailing_stop_label(
    candles: np.ndarray,
    base_n_bar: int = 15,
    min_r: float = 0.00025,
    k: float = 1.5,
    vol_lookback: int = 20,
    vol_scale_high: float = 0.5,
    vol_scale_low: float = 2.0,
    use_natr: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    自适应版 trailing stop 标记

    参数说明：
    -----------
    candles : np.ndarray
        K线数据，形状 [n, 6] 或与 jesse 原生兼容(N,6)：
        [时间戳, 开盘价, 最高价, 最低价, 收盘价, 成交量]

    base_n_bar : int
        基础窗口长度，类似于原先的 n_bar。

    min_r : float
        最小回报率，需要超过该值才视为可以开仓。

    k : float
        用于放大 NATR（或ATR）来确定目标回报率的倍数因子。

    vol_lookback : int
        计算"当前波动率"时的回溯周期。

    vol_scale_high : float
        当出现较大波动时，用于缩小有效 n_bar 的系数；
        例如 vol_scale_high=0.5，则当波动率大于参考值时，会把 n_bar *= 0.5。

    vol_scale_low : float
        当出现较小波动时，用于放大有效 n_bar 的系数；
        例如 vol_scale_low=2.0，则当波动率小于参考值时，会把 n_bar *= 2.

    use_natr : bool
        若为 True，则使用 NATR 进行波动率判断，否则使用 ATR。

    返回：
    -----------
    labels : np.ndarray[int8]
        与原始长度相同的多/空/平标记（1/-1/0）

    realized_r : np.ndarray[float64]
        真正实现的回报率

    bar_duration : np.ndarray[int64]
        持仓 bar 数，表示在何时退出
    """
    labels = np.zeros(candles.shape[0], dtype=np.int8)
    realized_r = np.zeros(candles.shape[0], dtype=np.float64)
    bar_duration = np.zeros(candles.shape[0], dtype=np.int64)

    close = helpers.get_candle_source(candles, "close")
    if use_natr:
        vol_array = ta.natr(candles, period=vol_lookback, sequential=True)  # NATR
    else:
        # 若不用NATR，也可用 ATR
        # vol_array = ta.atr(candles, period=vol_lookback, sequential=True)
        # 或者使用其他自定义波动率度量
        vol_array = ta.atr(candles, period=vol_lookback, sequential=True)

    # 为了演示"当前波动率"相对平均波动率做比较
    # 1) 计算 vol_array 在过去一段时间的均值（或中位数）
    # 2) 动态判定波动大/小
    # 这里只演示一种较简易的做法
    # 你可以自行用移动平均、EWMA、分位数等更灵活的方式
    vol_mean = np.nanmean(vol_array)

    LABELING_INDEX = 0
    SKIP_INDEX = 0
    LABEL = 0

    for idx in range(candles.shape[0]):
        # 检查越界
        if idx < base_n_bar or candles.shape[0] - idx <= 1:
            continue

        current_vol = vol_array[idx]
        if np.isnan(current_vol):
            continue

        # 根据当前波动率 与历史平均波动率比较
        # 演示：波动率大 -> 缩短 n_bar；波动率小 -> 放大 n_bar
        if current_vol > vol_mean * 1.2:
            # 波动率显著大
            n_bar_adaptive = int(base_n_bar * vol_scale_high)
        elif current_vol < vol_mean * 0.8:
            # 波动率显著小
            n_bar_adaptive = int(base_n_bar * vol_scale_low)
        else:
            # 波动率正常
            n_bar_adaptive = base_n_bar

        # n_bar_adaptive 不可小于 1
        n_bar_adaptive = max(n_bar_adaptive, 1)

        # 确保不与上次 LABELING_INDEX 冲突，否则会发生重复标记或跳跃错误
        if idx < LABELING_INDEX:
            continue

        # 如果可用的未来bar数量不够，就终止
        if idx + n_bar_adaptive >= candles.shape[0]:
            break

        # 设定起点
        LABELING_INDEX = idx
        SKIP_INDEX = idx + n_bar_adaptive

        p = close[idx]
        # 使用 NATR * k 或者 ATR * k 作为目标阈值
        # 当波动率极小但 k 又很大时，要注意结果是否过度放大
        if not np.isnan(vol_array[idx]):
            r_vol = vol_array[idx]
            target_r = max(r_vol * k / 100, min_r)
        else:
            target_r = min_r

        LABEL = 0
        # 找到第一个收益率绝对值大于 target_r 的位置
        for i in range(idx, idx + n_bar_adaptive + 1):
            if i >= candles.shape[0]:
                break
            current_r = close[i] / p - 1
            if np.abs(current_r) > target_r:
                LABEL = 1 if current_r > 0 else -1
                SKIP_INDEX = i
                break

        # 若有需要，也可以引入额外的交易量过滤等
        # ... (vol_break_only等逻辑在此省略)

        # 校正最终收益率，若仍未过阈值就维持0
        real_r = close[SKIP_INDEX] / close[LABELING_INDEX] - 1
        if np.abs(real_r) < target_r:
            LABEL = 0

        if LABEL != 0:
            labels[LABELING_INDEX] = LABEL
            realized_r[LABELING_INDEX] = real_r
            bar_duration[LABELING_INDEX] = SKIP_INDEX - LABELING_INDEX

    return labels, realized_r, bar_duration


class TrailingStopLabel:
    def __init__(self, candles, n_bar=15, min_r=0.00025, k=1):
        self._candles = candles
        self._n_bar = n_bar
        self._min_r = min_r

        self._labels, self._realized_r, self._bar_duration = (
            _adaptive_trailing_stop_label(candles, base_n_bar=n_bar, min_r=min_r, k=k)
        )

    @property
    def vol_ratio(self):
        vol, vol_ma = ta.volume(self._candles, sequential=True)
        return vol / vol_ma

    @property
    def labels(self):
        return self._labels

    @property
    def realized_r(self):
        return self._realized_r

    @property
    def bar_duration(self):
        return self._bar_duration

    @property
    def return_of_label(self):
        close = helpers.get_candle_source(self._candles, "close")
        HOLD_RETURN = close[-1] - close[0]
        PROFIT = 0
        START_PRICE = 0
        END_PRICE = 0
        for idx, (c, l) in enumerate(zip(close, self._labels)):
            if idx == 0:
                continue
            else:
                # 开多
                if l == 1 and self._labels[idx - 1] != 1:
                    START_PRICE = c
                # 开空
                if l == -1 and self._labels[idx - 1] != -1:
                    START_PRICE = c
                # 平仓
                if l != self._labels[idx - 1]:
                    if self._labels[idx - 1] == 1:
                        END_PRICE = c
                        PROFIT += END_PRICE - START_PRICE
                        START_PRICE = 0
                        END_PRICE = 0
                    elif self._labels[idx - 1] == -1:
                        END_PRICE = c
                        PROFIT += START_PRICE - END_PRICE
                        START_PRICE = 0
                        END_PRICE = 0
        return PROFIT / HOLD_RETURN
