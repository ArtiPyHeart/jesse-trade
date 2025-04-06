from jesse import helpers
from jesse.indicators import ma


def ma_difference(
    candles,
    short_period=14,
    short_matype=6,
    long_period=20,
    long_matype=6,
    source_type="close",
    sequential=False,
):
    candles = helpers.slice_candles(candles, sequential)
    short_ma = ma(
        candles,
        period=short_period,
        matype=short_matype,
        source_type=source_type,
        sequential=True,
    )
    long_ma = ma(
        candles,
        period=long_period,
        matype=long_matype,
        source_type=source_type,
        sequential=True,
    )

    ma_difference = short_ma - long_ma
    return ma_difference if sequential else ma_difference[-1]
