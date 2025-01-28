import jesse.indicators as ta
import numpy as np
import plotly.graph_objects as go
from jesse import helpers
from plotly.subplots import make_subplots


def _trailing_stop_label(candles, n_bar=15, min_r=0.00025, k=1.5, vol_break_only=False):
    """
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


class TrailingStopLabel:
    def __init__(self, candles, n_bar=15, min_r=0.00025, k=1):
        self._candles = candles
        self._n_bar = n_bar
        self._min_r = min_r

        self._labels, self._realized_r, self._bar_duration = _trailing_stop_label(
            candles, n_bar, min_r, k
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
                elif l == -1 and self._labels[idx - 1] != -1:
                    START_PRICE = c
                # 平仓
                elif l != self._labels[idx - 1] and self._labels[idx - 1] != 0:
                    END_PRICE = c
                    PROFIT += END_PRICE - START_PRICE
                    START_PRICE = 0
                    END_PRICE = 0
        return PROFIT / HOLD_RETURN


def plot(candles, lines: dict[str, np.ndarray]):
    time = [helpers.timestamp_to_time(i) for i in candles[:, 0]]
    o = helpers.get_candle_source(candles, "open")
    h = helpers.get_candle_source(candles, "high")
    l = helpers.get_candle_source(candles, "low")
    c = helpers.get_candle_source(candles, "close")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(x=time, open=o, high=h, low=l, close=c, name="Candlestick")
    )
    for k, v in lines.items():
        fig.add_trace(go.Scatter(x=time, y=v, name=k, mode="lines"))
    fig.show()


def subplot(candles, lines: dict[str, np.ndarray]):
    time = [helpers.timestamp_to_time(i) for i in candles[:, 0]]
    o = helpers.get_candle_source(candles, "open")
    h = helpers.get_candle_source(candles, "high")
    l = helpers.get_candle_source(candles, "low")
    c = helpers.get_candle_source(candles, "close")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Candlestick(x=time, open=o, high=h, low=l, close=c, name="Candlestick"),
        row=1,
        col=1,
    )

    for line_name, values in lines.items():
        fig.add_trace(
            go.Scatter(x=time, y=values, name=line_name, mode="lines"), row=2, col=1
        )

    fig.update_layout(height=800, showlegend=False)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()
