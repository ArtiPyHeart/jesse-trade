from collections import namedtuple

import jesse.indicators as ta
import numpy as np
from jesse import helpers

from custom_indicators import (
    adaptive_bandpass,
    adaptive_cci,
    adaptive_rsi,
    adaptive_stochastic,
    autocorrelation,
    autocorrelation_periodogram,
    autocorrelation_reversals,
    comb_spectrum,
    dft,
    ehlers_convolution,
    evenbetter_sinewave,
    hurst_coefficient,
    mod_rsi,
    mod_stochastic,
    roofing_filter,
    swamicharts_rsi,
    swamicharts_stochastic,
    td_sequential,
)
from custom_indicators.dominant_cycle import (
    dual_differentiator,
    homodyne,
    phase_accumulation,
)
from custom_indicators.utils.math import ddt, dt, lag

Features = namedtuple("Features", ["names", "features"])


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
        "td_buy_dt",
        "td_sell",
        "td_sell_lag1",
        "td_sell_lag2",
        "td_sell_lag3",
        "td_sell_dt",
    ]
    td_buy, td_sell = td_sequential(candles, sequential=True)
    td_buy_lag1 = lag(td_buy, 1)
    td_buy_lag2 = lag(td_buy, 2)
    td_buy_lag3 = lag(td_buy, 3)
    td_buy_dt = dt(td_buy)
    td_sell_lag1 = lag(td_sell, 1)
    td_sell_lag2 = lag(td_sell, 2)
    td_sell_lag3 = lag(td_sell, 3)
    td_sell_dt = dt(td_sell)
    final_fe.extend(
        [
            td_buy.reshape(-1, 1),
            td_buy_lag1.reshape(-1, 1),
            td_buy_lag2.reshape(-1, 1),
            td_buy_lag3.reshape(-1, 1),
            td_buy_dt.reshape(-1, 1),
            td_sell.reshape(-1, 1),
            td_sell_lag1.reshape(-1, 1),
            td_sell_lag2.reshape(-1, 1),
            td_sell_lag3.reshape(-1, 1),
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
        "bandpass_dt",
        "bandpass_ddt",
        "highpass_bp",
        "highpass_bp_lag1",
        "highpass_bp_lag2",
        "highpass_bp_lag3",
        "highpass_bp_dt",
        "highpass_bp_ddt",
    ]
    bandpass_tuple = ta.bandpass(candles, sequential=True)
    bandpass = bandpass_tuple.bp_normalized
    bandpass_lag1 = lag(bandpass, 1)
    bandpass_lag2 = lag(bandpass, 2)
    bandpass_lag3 = lag(bandpass, 3)
    bandpass_dt = dt(bandpass)
    bandpass_ddt = ddt(bandpass)
    highpass_bp = bandpass_tuple.trigger
    highpass_bp_lag1 = lag(highpass_bp, 1)
    highpass_bp_lag2 = lag(highpass_bp, 2)
    highpass_bp_lag3 = lag(highpass_bp, 3)
    highpass_bp_dt = dt(highpass_bp)
    highpass_bp_ddt = ddt(highpass_bp)
    final_fe.extend(
        [
            bandpass.reshape(-1, 1),
            bandpass_lag1.reshape(-1, 1),
            bandpass_lag2.reshape(-1, 1),
            bandpass_lag3.reshape(-1, 1),
            bandpass_dt.reshape(-1, 1),
            bandpass_ddt.reshape(-1, 1),
            highpass_bp.reshape(-1, 1),
            highpass_bp_lag1.reshape(-1, 1),
            highpass_bp_lag2.reshape(-1, 1),
            highpass_bp_lag3.reshape(-1, 1),
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
        "hurst_coef_fast_dt",
        "hurst_coef_fast_ddt",
        "hurst_coef_slow",
        "hurst_coef_slow_lag1",
        "hurst_coef_slow_lag2",
        "hurst_coef_slow_lag3",
        "hurst_coef_slow_dt",
        "hurst_coef_slow_ddt",
    ]
    hurst_coef_fast = hurst_coefficient(candles, period=30, sequential=True)
    hurst_coef_fast_lag1 = lag(hurst_coef_fast, 1)
    hurst_coef_fast_lag2 = lag(hurst_coef_fast, 2)
    hurst_coef_fast_lag3 = lag(hurst_coef_fast, 3)
    hurst_coef_fast_dt = dt(hurst_coef_fast)
    hurst_coef_fast_ddt = ddt(hurst_coef_fast)
    hurst_coef_slow = hurst_coefficient(candles, period=200, sequential=True)
    hurst_coef_slow_lag1 = lag(hurst_coef_slow, 1)
    hurst_coef_slow_lag2 = lag(hurst_coef_slow, 2)
    hurst_coef_slow_lag3 = lag(hurst_coef_slow, 3)
    hurst_coef_slow_dt = dt(hurst_coef_slow)
    hurst_coef_slow_ddt = ddt(hurst_coef_slow)
    final_fe.extend(
        [
            hurst_coef_fast.reshape(-1, 1),
            hurst_coef_fast_lag1.reshape(-1, 1),
            hurst_coef_fast_lag2.reshape(-1, 1),
            hurst_coef_fast_lag3.reshape(-1, 1),
            hurst_coef_fast_dt.reshape(-1, 1),
            hurst_coef_fast_ddt.reshape(-1, 1),
            hurst_coef_slow.reshape(-1, 1),
            hurst_coef_slow_lag1.reshape(-1, 1),
            hurst_coef_slow_lag2.reshape(-1, 1),
            hurst_coef_slow_lag3.reshape(-1, 1),
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
        "roofing_filter_dt",
        "roofing_filter_ddt",
    ]
    rf = roofing_filter(candles, sequential=True)
    rf_lag1 = lag(rf, 1)
    rf_lag2 = lag(rf, 2)
    rf_lag3 = lag(rf, 3)
    rf_dt = dt(rf)
    rf_ddt = ddt(rf)
    final_fe.extend(
        [
            rf.reshape(-1, 1),
            rf_lag1.reshape(-1, 1),
            rf_lag2.reshape(-1, 1),
            rf_lag3.reshape(-1, 1),
            rf_dt.reshape(-1, 1),
            rf_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # modified stochastic
    names = [
        "mod_stochastic",
        "mod_stochastic_lag1",
        "mod_stochastic_lag2",
        "mod_stochastic_lag3",
        "mod_stochastic_dt",
        "mod_stochastic_ddt",
    ]
    mod_stochastic_ = mod_stochastic(candles, roofing_filter=True, sequential=True)
    mod_stochastic_lag1 = lag(mod_stochastic_, 1)
    mod_stochastic_lag2 = lag(mod_stochastic_, 2)
    mod_stochastic_lag3 = lag(mod_stochastic_, 3)
    mod_stochastic_dt = dt(mod_stochastic_)
    mod_stochastic_ddt = ddt(mod_stochastic_)
    final_fe.extend(
        [
            mod_stochastic_.reshape(-1, 1),
            mod_stochastic_lag1.reshape(-1, 1),
            mod_stochastic_lag2.reshape(-1, 1),
            mod_stochastic_lag3.reshape(-1, 1),
            mod_stochastic_dt.reshape(-1, 1),
            mod_stochastic_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # modified rsi
    names = [
        "mod_rsi",
        "mod_rsi_lag1",
        "mod_rsi_lag2",
        "mod_rsi_lag3",
        "mod_rsi_dt",
        "mod_rsi_ddt",
    ]
    mod_rsi_ = mod_rsi(candles, sequential=True)
    mod_rsi_lag1 = lag(mod_rsi_, 1)
    mod_rsi_lag2 = lag(mod_rsi_, 2)
    mod_rsi_lag3 = lag(mod_rsi_, 3)
    mod_rsi_dt = dt(mod_rsi_)
    mod_rsi_ddt = ddt(mod_rsi_)
    final_fe.extend(
        [
            mod_rsi_.reshape(-1, 1),
            mod_rsi_lag1.reshape(-1, 1),
            mod_rsi_lag2.reshape(-1, 1),
            mod_rsi_lag3.reshape(-1, 1),
            mod_rsi_dt.reshape(-1, 1),
            mod_rsi_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # autocorrelation
    auto_corr = autocorrelation(candles, sequential=True)
    names = [f"ac_{i}" for i in range(auto_corr.shape[1])]
    final_fe.append(auto_corr)
    all_names.extend(names)

    # autocorrelation periodogram
    names = [
        "acp_dom_cycle",
        "acp_dom_cycle_lag1",
        "acp_dom_cycle_lag2",
        "acp_dom_cycle_lag3",
        "acp_dom_cycle_dt",
        "acp_dom_cycle_ddt",
    ]
    acp_dom_cycle, pwr = autocorrelation_periodogram(candles, sequential=True)
    acp_dom_cycle_lag1 = lag(acp_dom_cycle, 1)
    acp_dom_cycle_lag2 = lag(acp_dom_cycle, 2)
    acp_dom_cycle_lag3 = lag(acp_dom_cycle, 3)
    acp_dom_cycle_dt = dt(acp_dom_cycle)
    acp_dom_cycle_ddt = ddt(acp_dom_cycle)
    names.extend([f"acp_pwr_{i}" for i in range(pwr.shape[1])])
    final_fe.extend(
        [
            acp_dom_cycle.reshape(-1, 1),
            acp_dom_cycle_lag1.reshape(-1, 1),
            acp_dom_cycle_lag2.reshape(-1, 1),
            acp_dom_cycle_lag3.reshape(-1, 1),
            acp_dom_cycle_dt.reshape(-1, 1),
            acp_dom_cycle_ddt.reshape(-1, 1),
        ]
    )
    final_fe.append(pwr)
    all_names.extend(names)

    # autocorrelation reversals
    names = ["acr"]
    acr = autocorrelation_reversals(candles, sequential=True)
    final_fe.extend(
        [
            acr.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # dft
    names = [
        "dft_dom_cycle",
        "dft_dom_cycle_lag1",
        "dft_dom_cycle_lag2",
        "dft_dom_cycle_lag3",
        "dft_dom_cycle_dt",
        "dft_dom_cycle_ddt",
    ]
    dft_dom_cycle, spectrum = dft(candles, sequential=True)
    dft_dom_cycle_lag1 = lag(dft_dom_cycle, 1)
    dft_dom_cycle_lag2 = lag(dft_dom_cycle, 2)
    dft_dom_cycle_lag3 = lag(dft_dom_cycle, 3)
    dft_dom_cycle_dt = dt(dft_dom_cycle)
    dft_dom_cycle_ddt = ddt(dft_dom_cycle)
    names.extend([f"dft_spectrum_{i}" for i in range(spectrum.shape[1])])
    final_fe.extend(
        [
            dft_dom_cycle.reshape(-1, 1),
            dft_dom_cycle_lag1.reshape(-1, 1),
            dft_dom_cycle_lag2.reshape(-1, 1),
            dft_dom_cycle_lag3.reshape(-1, 1),
            dft_dom_cycle_dt.reshape(-1, 1),
            dft_dom_cycle_ddt.reshape(-1, 1),
        ]
    )
    final_fe.append(spectrum)
    all_names.extend(names)

    # comb spectrum
    names = [
        "comb_spectrum_dom_cycle",
        "comb_spectrum_dom_cycle_lag1",
        "comb_spectrum_dom_cycle_lag2",
        "comb_spectrum_dom_cycle_lag3",
        "comb_spectrum_dom_cycle_dt",
        "comb_spectrum_dom_cycle_ddt",
    ]
    comb_spectrum_dom_cycle, pwr = comb_spectrum(candles, sequential=True)
    comb_spectrum_dom_cycle_lag1 = lag(comb_spectrum_dom_cycle, 1)
    comb_spectrum_dom_cycle_lag2 = lag(comb_spectrum_dom_cycle, 2)
    comb_spectrum_dom_cycle_lag3 = lag(comb_spectrum_dom_cycle, 3)
    comb_spectrum_dom_cycle_dt = dt(comb_spectrum_dom_cycle)
    comb_spectrum_dom_cycle_ddt = ddt(comb_spectrum_dom_cycle)
    names.extend([f"comb_spectrum_pwr_{i}" for i in range(pwr.shape[1])])
    final_fe.extend(
        [
            comb_spectrum_dom_cycle.reshape(-1, 1),
            comb_spectrum_dom_cycle_lag1.reshape(-1, 1),
            comb_spectrum_dom_cycle_lag2.reshape(-1, 1),
            comb_spectrum_dom_cycle_lag3.reshape(-1, 1),
            comb_spectrum_dom_cycle_dt.reshape(-1, 1),
            comb_spectrum_dom_cycle_ddt.reshape(-1, 1),
        ]
    )
    final_fe.append(pwr)
    all_names.extend(names)

    # adaptive rsi
    names = [
        "adaptive_rsi",
        "adaptive_rsi_lag1",
        "adaptive_rsi_lag2",
        "adaptive_rsi_lag3",
        "adaptive_rsi_dt",
        "adaptive_rsi_ddt",
    ]
    adaptive_rsi_ = adaptive_rsi(candles, sequential=True)
    adaptive_rsi_lag1 = lag(adaptive_rsi_, 1)
    adaptive_rsi_lag2 = lag(adaptive_rsi_, 2)
    adaptive_rsi_lag3 = lag(adaptive_rsi_, 3)
    adaptive_rsi_dt = dt(adaptive_rsi_)
    adaptive_rsi_ddt = ddt(adaptive_rsi_)
    final_fe.extend(
        [
            adaptive_rsi_.reshape(-1, 1),
            adaptive_rsi_lag1.reshape(-1, 1),
            adaptive_rsi_lag2.reshape(-1, 1),
            adaptive_rsi_lag3.reshape(-1, 1),
            adaptive_rsi_dt.reshape(-1, 1),
            adaptive_rsi_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # adaptive stochastic
    names = [
        "adaptive_stochastic",
        "adaptive_stochastic_lag1",
        "adaptive_stochastic_lag2",
        "adaptive_stochastic_lag3",
        "adaptive_stochastic_dt",
        "adaptive_stochastic_ddt",
    ]
    adaptive_stochastic_ = adaptive_stochastic(candles, sequential=True)
    adaptive_stochastic_lag1 = lag(adaptive_stochastic_, 1)
    adaptive_stochastic_lag2 = lag(adaptive_stochastic_, 2)
    adaptive_stochastic_lag3 = lag(adaptive_stochastic_, 3)
    adaptive_stochastic_dt = dt(adaptive_stochastic_)
    adaptive_stochastic_ddt = ddt(adaptive_stochastic_)
    final_fe.extend(
        [
            adaptive_stochastic_.reshape(-1, 1),
            adaptive_stochastic_lag1.reshape(-1, 1),
            adaptive_stochastic_lag2.reshape(-1, 1),
            adaptive_stochastic_lag3.reshape(-1, 1),
            adaptive_stochastic_dt.reshape(-1, 1),
            adaptive_stochastic_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # adaptive cci
    names = [
        "adaptive_cci",
        "adaptive_cci_lag1",
        "adaptive_cci_lag2",
        "adaptive_cci_lag3",
        "adaptive_cci_dt",
        "adaptive_cci_ddt",
    ]
    adaptive_cci_ = adaptive_cci(candles, sequential=True)
    adaptive_cci_lag1 = lag(adaptive_cci_, 1)
    adaptive_cci_lag2 = lag(adaptive_cci_, 2)
    adaptive_cci_lag3 = lag(adaptive_cci_, 3)
    adaptive_cci_dt = dt(adaptive_cci_)
    adaptive_cci_ddt = ddt(adaptive_cci_)
    final_fe.extend(
        [
            adaptive_cci_.reshape(-1, 1),
            adaptive_cci_lag1.reshape(-1, 1),
            adaptive_cci_lag2.reshape(-1, 1),
            adaptive_cci_lag3.reshape(-1, 1),
            adaptive_cci_dt.reshape(-1, 1),
            adaptive_cci_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # adaptive bandpass
    names = [
        "adaptive_bp",
        "adaptive_bp_lead",
        "adaptive_bp_lead_minus_bp",
        "adaptive_bp_lag1",
        "adaptive_bp_lag2",
        "adaptive_bp_lag3",
        "adaptive_bp_dt",
        "adaptive_bp_ddt",
        "adaptive_bp_lead_lag1",
        "adaptive_bp_lead_lag2",
        "adaptive_bp_lead_lag3",
        "adaptive_bp_lead_dt",
        "adaptive_bp_lead_ddt",
    ]
    adaptive_bp, adaptive_bp_lead, _ = adaptive_bandpass(candles, sequential=True)
    adaptive_bp_lead_minus_bp = adaptive_bp_lead - adaptive_bp
    adaptive_bp_lag1 = lag(adaptive_bp, 1)
    adaptive_bp_lag2 = lag(adaptive_bp, 2)
    adaptive_bp_lag3 = lag(adaptive_bp, 3)
    adaptive_bp_dt = dt(adaptive_bp)
    adaptive_bp_ddt = ddt(adaptive_bp)
    adaptive_bp_lead_lag1 = lag(adaptive_bp_lead, 1)
    adaptive_bp_lead_lag2 = lag(adaptive_bp_lead, 2)
    adaptive_bp_lead_lag3 = lag(adaptive_bp_lead, 3)
    adaptive_bp_lead_dt = dt(adaptive_bp_lead)
    adaptive_bp_lead_ddt = ddt(adaptive_bp_lead)
    final_fe.extend(
        [
            adaptive_bp.reshape(-1, 1),
            adaptive_bp_lead.reshape(-1, 1),
            adaptive_bp_lead_minus_bp.reshape(-1, 1),
            adaptive_bp_lag1.reshape(-1, 1),
            adaptive_bp_lag2.reshape(-1, 1),
            adaptive_bp_lag3.reshape(-1, 1),
            adaptive_bp_dt.reshape(-1, 1),
            adaptive_bp_ddt.reshape(-1, 1),
            adaptive_bp_lead_lag1.reshape(-1, 1),
            adaptive_bp_lead_lag2.reshape(-1, 1),
            adaptive_bp_lead_lag3.reshape(-1, 1),
            adaptive_bp_lead_dt.reshape(-1, 1),
            adaptive_bp_lead_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # evenbetter sinewave
    names = [
        "evenbetter_sinewave_long",
        "evenbetter_sinewave_long_lag1",
        "evenbetter_sinewave_long_lag2",
        "evenbetter_sinewave_long_lag3",
        "evenbetter_sinewave_short",
        "evenbetter_sinewave_short_lag1",
        "evenbetter_sinewave_short_lag2",
        "evenbetter_sinewave_short_lag3",
    ]
    eb_sw_long = evenbetter_sinewave(candles, duration=40, sequential=True)
    eb_sw_long_lag1 = lag(eb_sw_long, 1)
    eb_sw_long_lag2 = lag(eb_sw_long, 2)
    eb_sw_long_lag3 = lag(eb_sw_long, 3)
    eb_sw_short = evenbetter_sinewave(candles, duration=20, sequential=True)
    eb_sw_short_lag1 = lag(eb_sw_short, 1)
    eb_sw_short_lag2 = lag(eb_sw_short, 2)
    eb_sw_short_lag3 = lag(eb_sw_short, 3)
    final_fe.extend(
        [
            eb_sw_long.reshape(-1, 1),
            eb_sw_long_lag1.reshape(-1, 1),
            eb_sw_long_lag2.reshape(-1, 1),
            eb_sw_long_lag3.reshape(-1, 1),
            eb_sw_short.reshape(-1, 1),
            eb_sw_short_lag1.reshape(-1, 1),
            eb_sw_short_lag2.reshape(-1, 1),
            eb_sw_short_lag3.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # convolution
    names = []
    _, _, conv = ehlers_convolution(candles, sequential=True)
    names.extend([f"conv_{i}" for i in range(conv.shape[1])])
    final_fe.append(conv)
    all_names.extend(names)

    # dual differentiator
    names = [
        "dual_diff",
        "dual_diff_lag1",
        "dual_diff_lag2",
        "dual_diff_lag3",
        "dual_diff_dt",
        "dual_diff_ddt",
    ]
    dual_diff = dual_differentiator(candles, sequential=True)
    dual_diff_lag1 = lag(dual_diff, 1)
    dual_diff_lag2 = lag(dual_diff, 2)
    dual_diff_lag3 = lag(dual_diff, 3)
    dual_diff_dt = dt(dual_diff)
    dual_diff_ddt = ddt(dual_diff)
    final_fe.extend(
        [
            dual_diff.reshape(-1, 1),
            dual_diff_lag1.reshape(-1, 1),
            dual_diff_lag2.reshape(-1, 1),
            dual_diff_lag3.reshape(-1, 1),
            dual_diff_dt.reshape(-1, 1),
            dual_diff_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # phase accumulation
    names = [
        "phase_accumulation",
        "phase_accumulation_lag1",
        "phase_accumulation_lag2",
        "phase_accumulation_lag3",
        "phase_accumulation_dt",
        "phase_accumulation_ddt",
    ]
    phase_accumulation_ = phase_accumulation(candles, sequential=True)
    phase_accumulation_lag1 = lag(phase_accumulation_, 1)
    phase_accumulation_lag2 = lag(phase_accumulation_, 2)
    phase_accumulation_lag3 = lag(phase_accumulation_, 3)
    phase_accumulation_dt = dt(phase_accumulation_)
    phase_accumulation_ddt = ddt(phase_accumulation_)
    final_fe.extend(
        [
            phase_accumulation_.reshape(-1, 1),
            phase_accumulation_lag1.reshape(-1, 1),
            phase_accumulation_lag2.reshape(-1, 1),
            phase_accumulation_lag3.reshape(-1, 1),
            phase_accumulation_dt.reshape(-1, 1),
            phase_accumulation_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # homodyne
    names = [
        "homodyne",
        "homodyne_lag1",
        "homodyne_lag2",
        "homodyne_lag3",
        "homodyne_dt",
        "homodyne_ddt",
    ]
    homodyne_ = homodyne(candles, sequential=True)
    homodyne_lag1 = lag(homodyne_, 1)
    homodyne_lag2 = lag(homodyne_, 2)
    homodyne_lag3 = lag(homodyne_, 3)
    homodyne_dt = dt(homodyne_)
    homodyne_ddt = ddt(homodyne_)
    final_fe.extend(
        [
            homodyne_.reshape(-1, 1),
            homodyne_lag1.reshape(-1, 1),
            homodyne_lag2.reshape(-1, 1),
            homodyne_lag3.reshape(-1, 1),
            homodyne_dt.reshape(-1, 1),
            homodyne_ddt.reshape(-1, 1),
        ]
    )
    all_names.extend(names)

    # swamicharts rsi
    names = []
    lookback, swamicharts_rsi_ = swamicharts_rsi(candles, sequential=True)
    names.extend([f"swamicharts_rsi_{i}" for i in lookback])
    final_fe.append(swamicharts_rsi_)
    all_names.extend(names)

    # swamicharts stochastic
    names = []
    lookback, swamicharts_stochastic_ = swamicharts_stochastic(candles, sequential=True)
    names.extend([f"swamicharts_stochastic_{i}" for i in lookback])
    final_fe.append(swamicharts_stochastic_)
    all_names.extend(names)

    final_fe = np.concatenate(final_fe, axis=1)
    if sequential:
        return Features(all_names, final_fe)
    else:
        return Features(all_names, final_fe[-1, :].reshape(1, -1))


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

    fe = feature_matrix(trading_1m, sequential=True)
    assert (
        len(fe.names) == fe.features.shape[1]
    ), f"{len(fe.names)} != {fe.features.shape[1]}"
    assert fe.features.shape[0] == trading_1m.shape[0]

    fe_last = feature_matrix(trading_1m, sequential=False)
    assert fe_last.features.shape[0] == 1
    assert fe_last.features.shape[1] == fe.features.shape[1]
