import jesse.indicators as ta
import numpy as np
from jesse import helpers

from custom_indicators import (
    accumulated_swing_index,
    adaptive_cci,
    adaptive_rsi,
    adaptive_stochastic,
    autocorrelation,
    autocorrelation_periodogram,
    comb_spectrum,
    dft,
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


def features_1m(candles: np.ndarray, sequential: bool = False) -> dict[str, np.ndarray]:
    candles = helpers.slice_candles(candles, sequential)
    res_fe = {}

    # autocorrelation
    auto_corr = autocorrelation(candles, sequential=True)
    for i in range(auto_corr.shape[1]):
        res_fe[f"ac_{i}"] = auto_corr[:, i]

    # accumulated swing index
    acc_swing_index = accumulated_swing_index(candles, sequential=True)
    res_fe["acc_swing_index"] = acc_swing_index
    res_fe["acc_swing_index_dt"] = dt(acc_swing_index)
    res_fe["acc_swing_index_ddt"] = ddt(acc_swing_index)
    res_fe["acc_swing_index_lag1"] = lag(acc_swing_index, 1)
    res_fe["acc_swing_index_lag2"] = lag(acc_swing_index, 2)
    res_fe["acc_swing_index_lag3"] = lag(acc_swing_index, 3)

    # autocorrelation periodogram
    acp_dom_cycle, pwr = autocorrelation_periodogram(candles, sequential=True)
    for i in range(pwr.shape[1]):
        res_fe[f"acp_pwr_{i}"] = pwr[:, i]

    # adaptive cci
    adaptive_cci_ = adaptive_cci(candles, sequential=True)
    res_fe["adaptive_cci"] = adaptive_cci_
    res_fe["adaptive_cci_lag1"] = lag(adaptive_cci_, 1)
    res_fe["adaptive_cci_lag2"] = lag(adaptive_cci_, 2)
    res_fe["adaptive_cci_lag3"] = lag(adaptive_cci_, 3)

    # adaptive rsi
    adaptive_rsi_ = adaptive_rsi(candles, sequential=True)
    res_fe["adaptive_rsi"] = adaptive_rsi_
    res_fe["adaptive_rsi_lag3"] = lag(adaptive_rsi_, 3)

    # adaptive stochastic
    adaptive_stochastic_ = adaptive_stochastic(candles, sequential=True)
    res_fe["adaptive_stochastic"] = adaptive_stochastic_
    res_fe["adaptive_stochastic_lag1"] = lag(adaptive_stochastic_, 1)
    res_fe["adaptive_stochastic_lag2"] = lag(adaptive_stochastic_, 2)
    res_fe["adaptive_stochastic_lag3"] = lag(adaptive_stochastic_, 3)

    # bandpass & highpass
    bandpass_tuple = ta.bandpass(candles, sequential=True)
    res_fe["bandpass"] = bandpass_tuple.bp_normalized
    res_fe["bandpass_ddt"] = ddt(bandpass_tuple.bp_normalized)
    res_fe["bandpass_lag1"] = lag(bandpass_tuple.bp_normalized, 1)
    res_fe["bandpass_lag2"] = lag(bandpass_tuple.bp_normalized, 2)
    res_fe["bandpass_lag3"] = lag(bandpass_tuple.bp_normalized, 3)

    # comb spectrum
    comb_spectrum_dom_cycle, pwr = comb_spectrum(candles, sequential=True)
    res_fe["comb_spectrum_dom_cycle"] = comb_spectrum_dom_cycle
    res_fe["comb_spectrum_dom_cycle_lag1"] = lag(comb_spectrum_dom_cycle, 1)
    res_fe["comb_spectrum_dom_cycle_lag2"] = lag(comb_spectrum_dom_cycle, 2)
    res_fe["comb_spectrum_dom_cycle_lag3"] = lag(comb_spectrum_dom_cycle, 3)
    for i in range(pwr.shape[1]):
        res_fe[f"comb_spectrum_pwr_{i}"] = pwr[:, i]

    # dft
    dft_dom_cycle, spectrum = dft(candles, sequential=True)
    for i in range(spectrum.shape[1]):
        res_fe[f"dft_spectrum_{i}"] = spectrum[:, i]

    # dual differentiator
    dual_diff = dual_differentiator(candles, sequential=True)
    res_fe["dual_diff"] = dual_diff
    res_fe["dual_diff_lag1"] = lag(dual_diff, 1)
    res_fe["dual_diff_lag2"] = lag(dual_diff, 2)
    res_fe["dual_diff_lag3"] = lag(dual_diff, 3)

    # ehlers early onset trend
    ehlers_early_onset_trend_ = ehlers_early_onset_trend(candles, sequential=True)
    res_fe["ehlers_early_onset_trend"] = ehlers_early_onset_trend_
    res_fe["ehlers_early_onset_trend_dt"] = dt(ehlers_early_onset_trend_)
    res_fe["ehlers_early_onset_trend_lag1"] = lag(ehlers_early_onset_trend_, 1)
    res_fe["ehlers_early_onset_trend_lag2"] = lag(ehlers_early_onset_trend_, 2)
    res_fe["ehlers_early_onset_trend_lag3"] = lag(ehlers_early_onset_trend_, 3)

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

    # homodyne
    homodyne_ = homodyne(candles, sequential=True)
    res_fe["homodyne"] = homodyne_
    res_fe["homodyne_lag1"] = lag(homodyne_, 1)
    res_fe["homodyne_lag2"] = lag(homodyne_, 2)
    res_fe["homodyne_lag3"] = lag(homodyne_, 3)

    # hurst
    hurst_coef_fast = hurst_coefficient(candles, period=30, sequential=True)
    hurst_coef_slow = hurst_coefficient(candles, period=200, sequential=True)
    res_fe["hurst_coef_fast"] = hurst_coef_fast
    res_fe["hurst_coef_fast_lag1"] = lag(hurst_coef_fast, 1)
    res_fe["hurst_coef_fast_lag2"] = lag(hurst_coef_fast, 2)
    res_fe["hurst_coef_fast_lag3"] = lag(hurst_coef_fast, 3)
    res_fe["hurst_coef_slow"] = hurst_coef_slow
    res_fe["hurst_coef_slow_lag1"] = lag(hurst_coef_slow, 1)
    res_fe["hurst_coef_slow_lag2"] = lag(hurst_coef_slow, 2)
    res_fe["hurst_coef_slow_lag3"] = lag(hurst_coef_slow, 3)

    # modified rsi
    mod_rsi_ = mod_rsi(candles, sequential=True)
    res_fe["mod_rsi"] = mod_rsi_
    res_fe["mod_rsi_lag1"] = lag(mod_rsi_, 1)
    res_fe["mod_rsi_lag2"] = lag(mod_rsi_, 2)
    res_fe["mod_rsi_lag3"] = lag(mod_rsi_, 3)

    # modified stochastic
    mod_stochastic_ = mod_stochastic(candles, roofing_filter=True, sequential=True)
    res_fe["mod_stochastic"] = mod_stochastic_
    res_fe["mod_stochastic_dt"] = dt(mod_stochastic_)
    res_fe["mod_stochastic_lag1"] = lag(mod_stochastic_, 1)
    res_fe["mod_stochastic_lag2"] = lag(mod_stochastic_, 2)
    res_fe["mod_stochastic_lag3"] = lag(mod_stochastic_, 3)

    # phase accumulation
    phase_accumulation_ = phase_accumulation(candles, sequential=True)
    res_fe["phase_accumulation"] = phase_accumulation_
    res_fe["phase_accumulation_ddt"] = ddt(phase_accumulation_)
    res_fe["phase_accumulation_lag1"] = lag(phase_accumulation_, 1)
    res_fe["phase_accumulation_lag2"] = lag(phase_accumulation_, 2)
    res_fe["phase_accumulation_lag3"] = lag(phase_accumulation_, 3)

    # roofing filter
    rf = roofing_filter(candles, sequential=True)
    res_fe["roofing_filter"] = rf
    res_fe["roofing_filter_dt"] = dt(rf)
    res_fe["roofing_filter_lag1"] = lag(rf, 1)
    res_fe["roofing_filter_lag2"] = lag(rf, 2)
    res_fe["roofing_filter_lag3"] = lag(rf, 3)

    # swamicharts rsi
    lookback, swamicharts_rsi_ = swamicharts_rsi(candles, sequential=True)
    for i in range(swamicharts_rsi_.shape[1]):
        res_fe[f"swamicharts_rsi_{i}"] = swamicharts_rsi_[:, i]

    # swamicharts stochastic
    lookback, swamicharts_stochastic_ = swamicharts_stochastic(candles, sequential=True)
    for i in range(swamicharts_stochastic_.shape[1]):
        res_fe[f"swamicharts_stochastic_{i}"] = swamicharts_stochastic_[:, i]

    if sequential:
        return {f"1m_{k}": v for k, v in res_fe.items()}
    else:
        return {f"1m_{k}": v[-1:] for k, v in res_fe.items()}
