import jesse.indicators as ta
import numpy as np
from jesse import helpers

from custom_indicators import (
    accumulated_swing_index,
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
    ehlers_early_onset_trend,
    evenbetter_sinewave,
    hurst_coefficient,
    mod_rsi,
    mod_stochastic,
    roofing_filter,
    swamicharts_rsi,
    swamicharts_stochastic,
)
from custom_indicators.dominant_cycle import (
    dual_differentiator,
    homodyne,
    phase_accumulation,
)
from custom_indicators.utils.math import ddt, dt, lag


def feature_bundle(candles: np.array, sequential: bool = False) -> dict[str, np.array]:
    candles = helpers.slice_candles(candles, sequential)
    res_fe = {}

    # accumulated swing index
    acc_swing_index = accumulated_swing_index(candles, sequential=True)
    res_fe["acc_swing_index"] = acc_swing_index
    res_fe["acc_swing_index_lag1"] = lag(acc_swing_index, 1)
    res_fe["acc_swing_index_lag2"] = lag(acc_swing_index, 2)
    res_fe["acc_swing_index_lag3"] = lag(acc_swing_index, 3)
    res_fe["acc_swing_index_dt"] = dt(acc_swing_index)
    res_fe["acc_swing_index_ddt"] = ddt(acc_swing_index)

    # ehlers early onset trend
    ehlers_early_onset_trend_ = ehlers_early_onset_trend(candles, sequential=True)
    res_fe["ehlers_early_onset_trend"] = ehlers_early_onset_trend_
    res_fe["ehlers_early_onset_trend_lag1"] = lag(ehlers_early_onset_trend_, 1)
    res_fe["ehlers_early_onset_trend_lag2"] = lag(ehlers_early_onset_trend_, 2)
    res_fe["ehlers_early_onset_trend_lag3"] = lag(ehlers_early_onset_trend_, 3)
    res_fe["ehlers_early_onset_trend_dt"] = dt(ehlers_early_onset_trend_)
    res_fe["ehlers_early_onset_trend_ddt"] = ddt(ehlers_early_onset_trend_)

    # bandpass & highpass
    bandpass_tuple = ta.bandpass(candles, sequential=True)
    res_fe["bandpass"] = bandpass_tuple.bp_normalized
    res_fe["bandpass_lag1"] = lag(bandpass_tuple.bp_normalized, 1)
    res_fe["bandpass_lag2"] = lag(bandpass_tuple.bp_normalized, 2)
    res_fe["bandpass_lag3"] = lag(bandpass_tuple.bp_normalized, 3)
    res_fe["bandpass_dt"] = dt(bandpass_tuple.bp_normalized)
    res_fe["bandpass_ddt"] = ddt(bandpass_tuple.bp_normalized)
    res_fe["highpass_bp"] = bandpass_tuple.trigger
    res_fe["highpass_bp_lag1"] = lag(bandpass_tuple.trigger, 1)
    res_fe["highpass_bp_lag2"] = lag(bandpass_tuple.trigger, 2)
    res_fe["highpass_bp_lag3"] = lag(bandpass_tuple.trigger, 3)
    res_fe["highpass_bp_dt"] = dt(bandpass_tuple.trigger)
    res_fe["highpass_bp_ddt"] = ddt(bandpass_tuple.trigger)

    # hurst
    hurst_coef_fast = hurst_coefficient(candles, period=30, sequential=True)
    hurst_coef_slow = hurst_coefficient(candles, period=200, sequential=True)
    res_fe["hurst_coef_fast"] = hurst_coef_fast
    res_fe["hurst_coef_fast_lag1"] = lag(hurst_coef_fast, 1)
    res_fe["hurst_coef_fast_lag2"] = lag(hurst_coef_fast, 2)
    res_fe["hurst_coef_fast_lag3"] = lag(hurst_coef_fast, 3)
    res_fe["hurst_coef_fast_dt"] = dt(hurst_coef_fast)
    res_fe["hurst_coef_fast_ddt"] = ddt(hurst_coef_fast)
    res_fe["hurst_coef_slow"] = hurst_coef_slow
    res_fe["hurst_coef_slow_lag1"] = lag(hurst_coef_slow, 1)
    res_fe["hurst_coef_slow_lag2"] = lag(hurst_coef_slow, 2)
    res_fe["hurst_coef_slow_lag3"] = lag(hurst_coef_slow, 3)
    res_fe["hurst_coef_slow_dt"] = dt(hurst_coef_slow)
    res_fe["hurst_coef_slow_ddt"] = ddt(hurst_coef_slow)

    # roofing filter
    rf = roofing_filter(candles, sequential=True)
    res_fe["roofing_filter"] = rf
    res_fe["roofing_filter_lag1"] = lag(rf, 1)
    res_fe["roofing_filter_lag2"] = lag(rf, 2)
    res_fe["roofing_filter_lag3"] = lag(rf, 3)
    res_fe["roofing_filter_dt"] = dt(rf)
    res_fe["roofing_filter_ddt"] = ddt(rf)

    # modified stochastic
    mod_stochastic_ = mod_stochastic(candles, roofing_filter=True, sequential=True)
    res_fe["mod_stochastic"] = mod_stochastic_
    res_fe["mod_stochastic_lag1"] = lag(mod_stochastic_, 1)
    res_fe["mod_stochastic_lag2"] = lag(mod_stochastic_, 2)
    res_fe["mod_stochastic_lag3"] = lag(mod_stochastic_, 3)
    res_fe["mod_stochastic_dt"] = dt(mod_stochastic_)
    res_fe["mod_stochastic_ddt"] = ddt(mod_stochastic_)

    # modified rsi
    mod_rsi_ = mod_rsi(candles, sequential=True)
    res_fe["mod_rsi"] = mod_rsi_
    res_fe["mod_rsi_lag1"] = lag(mod_rsi_, 1)
    res_fe["mod_rsi_lag2"] = lag(mod_rsi_, 2)
    res_fe["mod_rsi_lag3"] = lag(mod_rsi_, 3)
    res_fe["mod_rsi_dt"] = dt(mod_rsi_)
    res_fe["mod_rsi_ddt"] = ddt(mod_rsi_)

    # autocorrelation
    auto_corr = autocorrelation(candles, sequential=True)
    for i in range(auto_corr.shape[1]):
        res_fe[f"ac_{i}"] = auto_corr[:, i]

    # autocorrelation periodogram
    acp_dom_cycle, pwr = autocorrelation_periodogram(candles, sequential=True)
    for i in range(pwr.shape[1]):
        res_fe[f"acp_pwr_{i}"] = pwr[:, i]

    # autocorrelation reversals
    acr = autocorrelation_reversals(candles, sequential=True)
    res_fe["acr"] = acr

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
    res_fe["dft_dom_cycle"] = dft_dom_cycle
    res_fe["dft_dom_cycle_lag1"] = lag(dft_dom_cycle, 1)
    res_fe["dft_dom_cycle_lag2"] = lag(dft_dom_cycle, 2)
    res_fe["dft_dom_cycle_lag3"] = lag(dft_dom_cycle, 3)
    res_fe["dft_dom_cycle_dt"] = dt(dft_dom_cycle)
    res_fe["dft_dom_cycle_ddt"] = ddt(dft_dom_cycle)
    for i in range(spectrum.shape[1]):
        res_fe[f"dft_spectrum_{i}"] = spectrum[:, i]

    # comb spectrum
    comb_spectrum_dom_cycle, pwr = comb_spectrum(candles, sequential=True)
    res_fe["comb_spectrum_dom_cycle"] = comb_spectrum_dom_cycle
    res_fe["comb_spectrum_dom_cycle_lag1"] = lag(comb_spectrum_dom_cycle, 1)
    res_fe["comb_spectrum_dom_cycle_lag2"] = lag(comb_spectrum_dom_cycle, 2)
    res_fe["comb_spectrum_dom_cycle_lag3"] = lag(comb_spectrum_dom_cycle, 3)
    res_fe["comb_spectrum_dom_cycle_dt"] = dt(comb_spectrum_dom_cycle)
    res_fe["comb_spectrum_dom_cycle_ddt"] = ddt(comb_spectrum_dom_cycle)
    for i in range(pwr.shape[1]):
        res_fe[f"comb_spectrum_pwr_{i}"] = pwr[:, i]

    # adaptive rsi
    adaptive_rsi_ = adaptive_rsi(candles, sequential=True)
    res_fe["adaptive_rsi"] = adaptive_rsi_
    res_fe["adaptive_rsi_lag1"] = lag(adaptive_rsi_, 1)
    res_fe["adaptive_rsi_lag2"] = lag(adaptive_rsi_, 2)
    res_fe["adaptive_rsi_lag3"] = lag(adaptive_rsi_, 3)
    res_fe["adaptive_rsi_dt"] = dt(adaptive_rsi_)
    res_fe["adaptive_rsi_ddt"] = ddt(adaptive_rsi_)

    # adaptive stochastic
    adaptive_stochastic_ = adaptive_stochastic(candles, sequential=True)
    res_fe["adaptive_stochastic"] = adaptive_stochastic_
    res_fe["adaptive_stochastic_lag1"] = lag(adaptive_stochastic_, 1)
    res_fe["adaptive_stochastic_lag2"] = lag(adaptive_stochastic_, 2)
    res_fe["adaptive_stochastic_lag3"] = lag(adaptive_stochastic_, 3)
    res_fe["adaptive_stochastic_dt"] = dt(adaptive_stochastic_)
    res_fe["adaptive_stochastic_ddt"] = ddt(adaptive_stochastic_)

    # adaptive cci
    adaptive_cci_ = adaptive_cci(candles, sequential=True)
    res_fe["adaptive_cci"] = adaptive_cci_
    res_fe["adaptive_cci_lag1"] = lag(adaptive_cci_, 1)
    res_fe["adaptive_cci_lag2"] = lag(adaptive_cci_, 2)
    res_fe["adaptive_cci_lag3"] = lag(adaptive_cci_, 3)
    res_fe["adaptive_cci_dt"] = dt(adaptive_cci_)
    res_fe["adaptive_cci_ddt"] = ddt(adaptive_cci_)

    # adaptive bandpass
    adaptive_bp, adaptive_bp_lead, _ = adaptive_bandpass(candles, sequential=True)
    res_fe["adaptive_bp"] = adaptive_bp
    res_fe["adaptive_bp_lag1"] = lag(adaptive_bp, 1)
    res_fe["adaptive_bp_lag2"] = lag(adaptive_bp, 2)
    res_fe["adaptive_bp_lag3"] = lag(adaptive_bp, 3)
    res_fe["adaptive_bp_dt"] = dt(adaptive_bp)
    res_fe["adaptive_bp_ddt"] = ddt(adaptive_bp)
    res_fe["adaptive_bp_lead"] = adaptive_bp_lead
    res_fe["adaptive_bp_lead_lag1"] = lag(adaptive_bp_lead, 1)
    res_fe["adaptive_bp_lead_lag2"] = lag(adaptive_bp_lead, 2)
    res_fe["adaptive_bp_lead_lag3"] = lag(adaptive_bp_lead, 3)
    res_fe["adaptive_bp_lead_dt"] = dt(adaptive_bp_lead)
    res_fe["adaptive_bp_lead_ddt"] = ddt(adaptive_bp_lead)

    # evenbetter sinewave
    eb_sw_long = evenbetter_sinewave(candles, duration=40, sequential=True)
    res_fe["evenbetter_sinewave_long"] = eb_sw_long
    res_fe["evenbetter_sinewave_long_lag1"] = lag(eb_sw_long, 1)
    res_fe["evenbetter_sinewave_long_lag2"] = lag(eb_sw_long, 2)
    res_fe["evenbetter_sinewave_long_lag3"] = lag(eb_sw_long, 3)
    eb_sw_short = evenbetter_sinewave(candles, duration=20, sequential=True)
    res_fe["evenbetter_sinewave_short"] = eb_sw_short
    res_fe["evenbetter_sinewave_short_lag1"] = lag(eb_sw_short, 1)
    res_fe["evenbetter_sinewave_short_lag2"] = lag(eb_sw_short, 2)
    res_fe["evenbetter_sinewave_short_lag3"] = lag(eb_sw_short, 3)

    # convolution
    _, _, conv = ehlers_convolution(candles, sequential=True)
    for i in range(conv.shape[1]):
        res_fe[f"conv_{i}"] = conv[:, i]

    # dual differentiator
    dual_diff = dual_differentiator(candles, sequential=True)
    res_fe["dual_diff"] = dual_diff
    res_fe["dual_diff_lag1"] = lag(dual_diff, 1)
    res_fe["dual_diff_lag2"] = lag(dual_diff, 2)
    res_fe["dual_diff_lag3"] = lag(dual_diff, 3)
    res_fe["dual_diff_dt"] = dt(dual_diff)
    res_fe["dual_diff_ddt"] = ddt(dual_diff)

    # phase accumulation
    phase_accumulation_ = phase_accumulation(candles, sequential=True)
    res_fe["phase_accumulation"] = phase_accumulation_
    res_fe["phase_accumulation_lag1"] = lag(phase_accumulation_, 1)
    res_fe["phase_accumulation_lag2"] = lag(phase_accumulation_, 2)
    res_fe["phase_accumulation_lag3"] = lag(phase_accumulation_, 3)
    res_fe["phase_accumulation_dt"] = dt(phase_accumulation_)
    res_fe["phase_accumulation_ddt"] = ddt(phase_accumulation_)

    # homodyne
    homodyne_ = homodyne(candles, sequential=True)
    res_fe["homodyne"] = homodyne_
    res_fe["homodyne_lag1"] = lag(homodyne_, 1)
    res_fe["homodyne_lag2"] = lag(homodyne_, 2)
    res_fe["homodyne_lag3"] = lag(homodyne_, 3)
    res_fe["homodyne_dt"] = dt(homodyne_)
    res_fe["homodyne_ddt"] = ddt(homodyne_)

    # swamicharts rsi
    lookback, swamicharts_rsi_ = swamicharts_rsi(candles, sequential=True)
    for i in range(swamicharts_rsi_.shape[1]):
        res_fe[f"swamicharts_rsi_{i}"] = swamicharts_rsi_[:, i]

    # swamicharts stochastic
    lookback, swamicharts_stochastic_ = swamicharts_stochastic(candles, sequential=True)
    for i in range(swamicharts_stochastic_.shape[1]):
        res_fe[f"swamicharts_stochastic_{i}"] = swamicharts_stochastic_[:, i]

    if sequential:
        return res_fe
    else:
        return {k: v[-1:] for k, v in res_fe.items()}


if __name__ == "__main__":
    from jesse import research

    warmup_1m, trading_1m = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp("2024-12-01"),
        helpers.date_to_timestamp("2024-12-31"),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )

    fe = feature_bundle(trading_1m, sequential=True)
    for k, v in fe.items():
        assert len(v) == len(
            trading_1m
        ), f"{k} has length {len(v)} but candles has length {len(trading_1m)}"
    fe_last = feature_bundle(trading_1m, sequential=False)
    for k, v in fe_last.items():
        assert len(v) == 1, f"{k} has length {len(v)} not 1"
