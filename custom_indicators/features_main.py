import jesse.indicators as ta
import numpy as np
from jesse import helpers

from custom_indicators import hurst_coefficient, roofing_filter, td_sequential
from custom_indicators.utils.math import ddt, dt, lag


def feature_matrix(candles: np.array, sequential: bool = False):
    candles = helpers.slice_candles(candles, sequential)
    all_names = []
    final_fe = []

    # Demark td sequential
    names = [
        "td_buy",
        "td_buy_lag1",
        "td_buy_lag2",
        "td_buy_lag3",
        "td_buy_lag4",
        "td_buy_lag5",
        "td_buy_dt",
        "td_sell",
        "td_sell_lag1",
        "td_sell_lag2",
        "td_sell_lag3",
        "td_sell_lag4",
        "td_sell_lag5",
        "td_sell_dt",
    ]
    td_buy, td_sell = td_sequential(candles, sequential=True)
    td_buy_lag1 = lag(td_buy, 1)
    td_buy_lag2 = lag(td_buy, 2)
    td_buy_lag3 = lag(td_buy, 3)
    td_buy_lag4 = lag(td_buy, 4)
    td_buy_lag5 = lag(td_buy, 5)
    td_buy_dt = dt(td_buy)
    td_sell_lag1 = lag(td_sell, 1)
    td_sell_lag2 = lag(td_sell, 2)
    td_sell_lag3 = lag(td_sell, 3)
    td_sell_lag4 = lag(td_sell, 4)
    td_sell_lag5 = lag(td_sell, 5)
    td_sell_dt = dt(td_sell)
    final_fe.extend(
        [
            td_buy.reshape(-1, 1),
            td_buy_lag1.reshape(-1, 1),
            td_buy_lag2.reshape(-1, 1),
            td_buy_lag3.reshape(-1, 1),
            td_buy_lag4.reshape(-1, 1),
            td_buy_lag5.reshape(-1, 1),
            td_buy_dt.reshape(-1, 1),
            td_sell.reshape(-1, 1),
            td_sell_lag1.reshape(-1, 1),
            td_sell_lag2.reshape(-1, 1),
            td_sell_lag3.reshape(-1, 1),
            td_sell_lag4.reshape(-1, 1),
            td_sell_lag5.reshape(-1, 1),
            td_sell_dt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # bandpass & highpass
    names = [
        "bandpass",
        "bandpass_lag1",
        "bandpass_lag2",
        "bandpass_lag3",
        "bandpass_lag4",
        "bandpass_lag5",
        "bandpass_dt",
        "bandpass_ddt",
        "highpass_bp",
        "highpass_bp_lag1",
        "highpass_bp_lag2",
        "highpass_bp_lag3",
        "highpass_bp_lag4",
        "highpass_bp_lag5",
        "highpass_bp_dt",
        "highpass_bp_ddt",
    ]
    bandpass_tuple = ta.bandpass(candles, sequential=True)
    bandpass = bandpass_tuple.bp_normalized
    bandpass_lag1 = lag(bandpass, 1)
    bandpass_lag2 = lag(bandpass, 2)
    bandpass_lag3 = lag(bandpass, 3)
    bandpass_lag4 = lag(bandpass, 4)
    bandpass_lag5 = lag(bandpass, 5)
    bandpass_dt = dt(bandpass)
    bandpass_ddt = ddt(bandpass)
    highpass_bp = bandpass_tuple.trigger
    highpass_bp_lag1 = lag(highpass_bp, 1)
    highpass_bp_lag2 = lag(highpass_bp, 2)
    highpass_bp_lag3 = lag(highpass_bp, 3)
    highpass_bp_lag4 = lag(highpass_bp, 4)
    highpass_bp_lag5 = lag(highpass_bp, 5)
    highpass_bp_dt = dt(highpass_bp)
    highpass_bp_ddt = ddt(highpass_bp)
    final_fe.extend(
        [
            bandpass.reshape(-1, 1),
            bandpass_lag1.reshape(-1, 1),
            bandpass_lag2.reshape(-1, 1),
            bandpass_lag3.reshape(-1, 1),
            bandpass_lag4.reshape(-1, 1),
            bandpass_lag5.reshape(-1, 1),
            bandpass_dt.reshape(-1, 1),
            bandpass_ddt.reshape(-1, 1),
            highpass_bp.reshape(-1, 1),
            highpass_bp_lag1.reshape(-1, 1),
            highpass_bp_lag2.reshape(-1, 1),
            highpass_bp_lag3.reshape(-1, 1),
            highpass_bp_lag4.reshape(-1, 1),
            highpass_bp_lag5.reshape(-1, 1),
            highpass_bp_dt.reshape(-1, 1),
            highpass_bp_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # hurst
    names = [
        "hurst_coef_fast",
        "hurst_coef_fast_lag1",
        "hurst_coef_fast_lag2",
        "hurst_coef_fast_lag3",
        "hurst_coef_fast_lag4",
        "hurst_coef_fast_lag5",
        "hurst_coef_fast_dt",
        "hurst_coef_fast_ddt",
        "hurst_coef_slow",
        "hurst_coef_slow_lag1",
        "hurst_coef_slow_lag2",
        "hurst_coef_slow_lag3",
        "hurst_coef_slow_lag4",
        "hurst_coef_slow_lag5",
        "hurst_coef_slow_dt",
        "hurst_coef_slow_ddt",
    ]
    hurst_coef_fast = hurst_coefficient(candles, period=30, sequential=True)
    hurst_coef_fast_lag1 = lag(hurst_coef_fast, 1)
    hurst_coef_fast_lag2 = lag(hurst_coef_fast, 2)
    hurst_coef_fast_lag3 = lag(hurst_coef_fast, 3)
    hurst_coef_fast_lag4 = lag(hurst_coef_fast, 4)
    hurst_coef_fast_lag5 = lag(hurst_coef_fast, 5)
    hurst_coef_fast_dt = dt(hurst_coef_fast)
    hurst_coef_fast_ddt = ddt(hurst_coef_fast)
    hurst_coef_slow = hurst_coefficient(candles, period=200, sequential=True)
    hurst_coef_slow_lag1 = lag(hurst_coef_slow, 1)
    hurst_coef_slow_lag2 = lag(hurst_coef_slow, 2)
    hurst_coef_slow_lag3 = lag(hurst_coef_slow, 3)
    hurst_coef_slow_lag4 = lag(hurst_coef_slow, 4)
    hurst_coef_slow_lag5 = lag(hurst_coef_slow, 5)
    hurst_coef_slow_lag5 = lag(hurst_coef_slow, 5)
    hurst_coef_slow_dt = dt(hurst_coef_slow)
    hurst_coef_slow_ddt = ddt(hurst_coef_slow)
    final_fe.extend(
        [
            hurst_coef_fast.reshape(-1, 1),
            hurst_coef_fast_lag1.reshape(-1, 1),
            hurst_coef_fast_lag2.reshape(-1, 1),
            hurst_coef_fast_lag3.reshape(-1, 1),
            hurst_coef_fast_lag4.reshape(-1, 1),
            hurst_coef_fast_lag5.reshape(-1, 1),
            hurst_coef_fast_dt.reshape(-1, 1),
            hurst_coef_fast_ddt.reshape(-1, 1),
            hurst_coef_slow.reshape(-1, 1),
            hurst_coef_slow_lag1.reshape(-1, 1),
            hurst_coef_slow_lag2.reshape(-1, 1),
            hurst_coef_slow_lag3.reshape(-1, 1),
            hurst_coef_slow_lag4.reshape(-1, 1),
            hurst_coef_slow_lag5.reshape(-1, 1),
            hurst_coef_slow_dt.reshape(-1, 1),
            hurst_coef_slow_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # roofing filter
    names = [
        "roofing_filter",
        "roofing_filter_lag1",
        "roofing_filter_lag2",
        "roofing_filter_lag3",
        "roofing_filter_lag4",
        "roofing_filter_lag5",
        "roofing_filter_dt",
        "roofing_filter_ddt",
    ]
    rf = roofing_filter(candles, sequential=True)
    rf_lag1 = lag(rf, 1)
    rf_lag2 = lag(rf, 2)
    rf_lag3 = lag(rf, 3)
    rf_lag4 = lag(rf, 4)
    rf_lag5 = lag(rf, 5)
    rf_dt = dt(rf)
    rf_ddt = ddt(rf)
    final_fe.extend(
        [
            rf.reshape(-1, 1),
            rf_lag1.reshape(-1, 1),
            rf_lag2.reshape(-1, 1),
            rf_lag3.reshape(-1, 1),
            rf_lag4.reshape(-1, 1),
            rf_lag5.reshape(-1, 1),
            rf_dt.reshape(-1, 1),
            rf_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    final_fe = np.concatenate(final_fe, axis=1)
    if sequential:
        return final_fe
    else:
        return final_fe[-1, :].reshape(1, -1)


if __name__ == "__main__":
    from jesse import research

    warmup_1m, trading_1m = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp("2024-06-01"),
        helpers.date_to_timestamp("2024-12-31"),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )

    fe_seq = feature_matrix(trading_1m, sequential=True)
    assert fe_seq.shape[0] == trading_1m.shape[0]

    fe_last = feature_matrix(trading_1m, sequential=False)
    assert fe_last.shape[0] == 1
    assert fe_last.shape[1] == fe_seq.shape[1]
