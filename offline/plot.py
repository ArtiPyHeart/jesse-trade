import numpy as np
import plotly.graph_objects as go
from jesse import helpers
from plotly.subplots import make_subplots


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
