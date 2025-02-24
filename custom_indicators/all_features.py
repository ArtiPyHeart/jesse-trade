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
    td_sequential,
)
from custom_indicators.dominant_cycle import (
    dual_differentiator,
    homodyne,
    phase_accumulation,
)
from custom_indicators.utils.math import ddt, dt, lag

LAG_MAX = 40


def feature_bundle(candles: np.array, sequential: bool = False) -> dict[str, np.array]:
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
    for lag in range(1, LAG_MAX):
        res_fe[f"acc_swing_index_lag{lag}"] = lag(acc_swing_index, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"acc_swing_index_dt_lag{lag}"] = lag(res_fe["acc_swing_index_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"acc_swing_index_ddt_lag{lag}"] = lag(
            res_fe["acc_swing_index_ddt"], lag
        )

    # autocorrelation periodogram
    acp_dom_cycle, pwr = autocorrelation_periodogram(candles, sequential=True)
    for i in range(pwr.shape[1]):
        res_fe[f"acp_pwr_{i}"] = pwr[:, i]

    # autocorrelation reversals
    acr = autocorrelation_reversals(candles, sequential=True)
    res_fe["acr"] = acr

    # adaptive bandpass
    adaptive_bp, adaptive_bp_lead, _ = adaptive_bandpass(candles, sequential=True)
    res_fe["adaptive_bp"] = adaptive_bp
    res_fe["adaptive_bp_dt"] = dt(adaptive_bp)
    res_fe["adaptive_bp_ddt"] = ddt(adaptive_bp)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lag{lag}"] = lag(adaptive_bp, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_dt_lag{lag}"] = lag(res_fe["adaptive_bp_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_ddt_lag{lag}"] = lag(res_fe["adaptive_bp_ddt"], lag)
    res_fe["adaptive_bp_lead"] = adaptive_bp_lead
    res_fe["adaptive_bp_lead_dt"] = dt(adaptive_bp_lead)
    res_fe["adaptive_bp_lead_ddt"] = ddt(adaptive_bp_lead)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lead_lag{lag}"] = lag(adaptive_bp_lead, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lead_dt_lag{lag}"] = lag(
            res_fe["adaptive_bp_lead_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lead_ddt_lag{lag}"] = lag(
            res_fe["adaptive_bp_lead_ddt"], lag
        )

    # adaptive cci
    adaptive_cci_ = adaptive_cci(candles, sequential=True)
    res_fe["adaptive_cci"] = adaptive_cci_
    res_fe["adaptive_cci_dt"] = dt(adaptive_cci_)
    res_fe["adaptive_cci_ddt"] = ddt(adaptive_cci_)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_cci_lag{lag}"] = lag(adaptive_cci_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_cci_dt_lag{lag}"] = lag(res_fe["adaptive_cci_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_cci_ddt_lag{lag}"] = lag(res_fe["adaptive_cci_ddt"], lag)

    # adaptive rsi
    adaptive_rsi_ = adaptive_rsi(candles, sequential=True)
    res_fe["adaptive_rsi"] = adaptive_rsi_
    res_fe["adaptive_rsi_dt"] = dt(adaptive_rsi_)
    res_fe["adaptive_rsi_ddt"] = ddt(adaptive_rsi_)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_rsi_lag{lag}"] = lag(adaptive_rsi_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_rsi_dt_lag{lag}"] = lag(res_fe["adaptive_rsi_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_rsi_ddt_lag{lag}"] = lag(res_fe["adaptive_rsi_ddt"], lag)

    # adaptive stochastic
    adaptive_stochastic_ = adaptive_stochastic(candles, sequential=True)
    res_fe["adaptive_stochastic"] = adaptive_stochastic_
    res_fe["adaptive_stochastic_dt"] = dt(adaptive_stochastic_)
    res_fe["adaptive_stochastic_ddt"] = ddt(adaptive_stochastic_)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_stochastic_lag{lag}"] = lag(adaptive_stochastic_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_stochastic_dt_lag{lag}"] = lag(
            res_fe["adaptive_stochastic_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"adaptive_stochastic_ddt_lag{lag}"] = lag(
            res_fe["adaptive_stochastic_ddt"], lag
        )

    # bandpass & highpass
    bandpass_tuple = ta.bandpass(candles, sequential=True)
    res_fe["bandpass"] = bandpass_tuple.bp_normalized
    res_fe["bandpass_dt"] = dt(bandpass_tuple.bp_normalized)
    res_fe["bandpass_ddt"] = ddt(bandpass_tuple.bp_normalized)
    for lag in range(1, LAG_MAX):
        res_fe[f"bandpass_lag{lag}"] = lag(bandpass_tuple.bp_normalized, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"bandpass_dt_lag{lag}"] = lag(res_fe["bandpass_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"bandpass_ddt_lag{lag}"] = lag(res_fe["bandpass_ddt"], lag)
    res_fe["highpass_bp"] = bandpass_tuple.trigger
    res_fe["highpass_bp_dt"] = dt(bandpass_tuple.trigger)
    res_fe["highpass_bp_ddt"] = ddt(bandpass_tuple.trigger)
    for lag in range(1, LAG_MAX):
        res_fe[f"highpass_bp_lag{lag}"] = lag(bandpass_tuple.trigger, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"highpass_bp_dt_lag{lag}"] = lag(res_fe["highpass_bp_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"highpass_bp_ddt_lag{lag}"] = lag(res_fe["highpass_bp_ddt"], lag)

    # comb spectrum
    comb_spectrum_dom_cycle, pwr = comb_spectrum(candles, sequential=True)
    res_fe["comb_spectrum_dom_cycle"] = comb_spectrum_dom_cycle
    res_fe["comb_spectrum_dom_cycle_dt"] = dt(comb_spectrum_dom_cycle)
    res_fe["comb_spectrum_dom_cycle_ddt"] = ddt(comb_spectrum_dom_cycle)
    for lag in range(1, LAG_MAX):
        res_fe[f"comb_spectrum_dom_cycle_lag{lag}"] = lag(comb_spectrum_dom_cycle, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"comb_spectrum_dom_cycle_dt_lag{lag}"] = lag(
            res_fe["comb_spectrum_dom_cycle_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"comb_spectrum_dom_cycle_ddt_lag{lag}"] = lag(
            res_fe["comb_spectrum_dom_cycle_ddt"], lag
        )
    for i in range(pwr.shape[1]):
        res_fe[f"comb_spectrum_pwr_{i}"] = pwr[:, i]

    # convolution
    _, _, conv = ehlers_convolution(candles, sequential=True)
    for i in range(conv.shape[1]):
        res_fe[f"conv_{i}"] = conv[:, i]

    # damiani volatmeter
    damiani_vol = ta.damiani_volatmeter(candles, sequential=True)
    res_fe["damiani_vol"] = damiani_vol

    # dft
    dft_dom_cycle, spectrum = dft(candles, sequential=True)
    res_fe["dft_dom_cycle"] = dft_dom_cycle
    res_fe["dft_dom_cycle_dt"] = dt(dft_dom_cycle)
    res_fe["dft_dom_cycle_ddt"] = ddt(dft_dom_cycle)
    for lag in range(1, LAG_MAX):
        res_fe[f"dft_dom_cycle_lag{lag}"] = lag(dft_dom_cycle, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"dft_dom_cycle_dt_lag{lag}"] = lag(res_fe["dft_dom_cycle_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"dft_dom_cycle_ddt_lag{lag}"] = lag(res_fe["dft_dom_cycle_ddt"], lag)
    for i in range(spectrum.shape[1]):
        res_fe[f"dft_spectrum_{i}"] = spectrum[:, i]

    # dual differentiator
    dual_diff = dual_differentiator(candles, sequential=True)
    res_fe["dual_diff"] = dual_diff
    res_fe["dual_diff_dt"] = dt(dual_diff)
    res_fe["dual_diff_ddt"] = ddt(dual_diff)
    for lag in range(1, LAG_MAX):
        res_fe[f"dual_diff_lag{lag}"] = lag(dual_diff, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"dual_diff_dt_lag{lag}"] = lag(res_fe["dual_diff_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"dual_diff_ddt_lag{lag}"] = lag(res_fe["dual_diff_ddt"], lag)

    # ehlers early onset trend
    ehlers_early_onset_trend_ = ehlers_early_onset_trend(candles, sequential=True)
    res_fe["ehlers_early_onset_trend"] = ehlers_early_onset_trend_
    res_fe["ehlers_early_onset_trend_dt"] = dt(ehlers_early_onset_trend_)
    res_fe["ehlers_early_onset_trend_ddt"] = ddt(ehlers_early_onset_trend_)
    for lag in range(1, LAG_MAX):
        res_fe[f"ehlers_early_onset_trend_lag{lag}"] = lag(
            ehlers_early_onset_trend_, lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"ehlers_early_onset_trend_dt_lag{lag}"] = lag(
            res_fe["ehlers_early_onset_trend_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"ehlers_early_onset_trend_ddt_lag{lag}"] = lag(
            res_fe["ehlers_early_onset_trend_ddt"], lag
        )

    # evenbetter sinewave
    eb_sw_long = evenbetter_sinewave(candles, duration=40, sequential=True)
    res_fe["evenbetter_sinewave_long"] = eb_sw_long
    res_fe["evenbetter_sinewave_long_dt"] = dt(eb_sw_long)
    res_fe["evenbetter_sinewave_long_ddt"] = ddt(eb_sw_long)
    for lag in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_long_lag{lag}"] = lag(eb_sw_long, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_long_dt_lag{lag}"] = lag(
            res_fe["evenbetter_sinewave_long_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_long_ddt_lag{lag}"] = lag(
            res_fe["evenbetter_sinewave_long_ddt"], lag
        )
    eb_sw_short = evenbetter_sinewave(candles, duration=20, sequential=True)
    res_fe["evenbetter_sinewave_short"] = eb_sw_short
    res_fe["evenbetter_sinewave_short_dt"] = dt(eb_sw_short)
    res_fe["evenbetter_sinewave_short_ddt"] = ddt(eb_sw_short)
    for lag in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_short_lag{lag}"] = lag(eb_sw_short, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_short_dt_lag{lag}"] = lag(
            res_fe["evenbetter_sinewave_short_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_short_ddt_lag{lag}"] = lag(
            res_fe["evenbetter_sinewave_short_ddt"], lag
        )

    # fisher
    fisher_ind = ta.fisher(candles, sequential=True)
    res_fe["fisher"] = fisher_ind
    res_fe["fisher_dt"] = dt(fisher_ind)
    res_fe["fisher_ddt"] = ddt(fisher_ind)
    for lag in range(1, LAG_MAX):
        res_fe[f"fisher_lag{lag}"] = lag(fisher_ind, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"fisher_dt_lag{lag}"] = lag(res_fe["fisher_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"fisher_ddt_lag{lag}"] = lag(res_fe["fisher_ddt"], lag)

    # forecast oscillator
    forecast_oscillator = ta.forecast_oscillator(candles, sequential=True)
    res_fe["forecast_oscillator"] = forecast_oscillator
    for lag in range(1, LAG_MAX):
        res_fe[f"forecast_oscillator_lag{lag}"] = lag(forecast_oscillator, lag)

    # homodyne
    homodyne_ = homodyne(candles, sequential=True)
    res_fe["homodyne"] = homodyne_
    res_fe["homodyne_dt"] = dt(homodyne_)
    res_fe["homodyne_ddt"] = ddt(homodyne_)
    for lag in range(1, LAG_MAX):
        res_fe[f"homodyne_lag{lag}"] = lag(homodyne_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"homodyne_dt_lag{lag}"] = lag(res_fe["homodyne_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"homodyne_ddt_lag{lag}"] = lag(res_fe["homodyne_ddt"], lag)

    # hurst
    hurst_coef_fast = hurst_coefficient(candles, period=30, sequential=True)
    hurst_coef_slow = hurst_coefficient(candles, period=200, sequential=True)
    res_fe["hurst_coef_fast"] = hurst_coef_fast
    res_fe["hurst_coef_fast_dt"] = dt(hurst_coef_fast)
    res_fe["hurst_coef_fast_ddt"] = ddt(hurst_coef_fast)
    for lag in range(1, LAG_MAX):
        res_fe[f"hurst_coef_fast_lag{lag}"] = lag(hurst_coef_fast, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"hurst_coef_fast_dt_lag{lag}"] = lag(res_fe["hurst_coef_fast_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"hurst_coef_fast_ddt_lag{lag}"] = lag(
            res_fe["hurst_coef_fast_ddt"], lag
        )
    res_fe["hurst_coef_slow"] = hurst_coef_slow
    res_fe["hurst_coef_slow_dt"] = dt(hurst_coef_slow)
    res_fe["hurst_coef_slow_ddt"] = ddt(hurst_coef_slow)
    for lag in range(1, LAG_MAX):
        res_fe[f"hurst_coef_slow_lag{lag}"] = lag(hurst_coef_slow, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"hurst_coef_slow_dt_lag{lag}"] = lag(res_fe["hurst_coef_slow_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"hurst_coef_slow_ddt_lag{lag}"] = lag(
            res_fe["hurst_coef_slow_ddt"], lag
        )

    # modified rsi
    mod_rsi_ = mod_rsi(candles, sequential=True)
    res_fe["mod_rsi"] = mod_rsi_
    res_fe["mod_rsi_dt"] = dt(mod_rsi_)
    res_fe["mod_rsi_ddt"] = ddt(mod_rsi_)
    for lag in range(1, LAG_MAX):
        res_fe[f"mod_rsi_lag{lag}"] = lag(mod_rsi_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"mod_rsi_dt_lag{lag}"] = lag(res_fe["mod_rsi_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"mod_rsi_ddt_lag{lag}"] = lag(res_fe["mod_rsi_ddt"], lag)

    # modified stochastic
    mod_stochastic_ = mod_stochastic(candles, roofing_filter=True, sequential=True)
    res_fe["mod_stochastic"] = mod_stochastic_
    res_fe["mod_stochastic_dt"] = dt(mod_stochastic_)
    res_fe["mod_stochastic_ddt"] = ddt(mod_stochastic_)
    for lag in range(1, LAG_MAX):
        res_fe[f"mod_stochastic_lag{lag}"] = lag(mod_stochastic_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"mod_stochastic_dt_lag{lag}"] = lag(res_fe["mod_stochastic_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"mod_stochastic_ddt_lag{lag}"] = lag(res_fe["mod_stochastic_ddt"], lag)

    # natr
    natr_ = ta.natr(candles, sequential=True)
    res_fe["natr"] = natr_
    res_fe["natr_dt"] = dt(natr_)
    res_fe["natr_ddt"] = ddt(natr_)
    for lag in range(1, LAG_MAX):
        res_fe[f"natr_lag{lag}"] = lag(natr_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"natr_dt_lag{lag}"] = lag(res_fe["natr_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"natr_ddt_lag{lag}"] = lag(res_fe["natr_ddt"], lag)

    # phase accumulation
    phase_accumulation_ = phase_accumulation(candles, sequential=True)
    res_fe["phase_accumulation"] = phase_accumulation_
    res_fe["phase_accumulation_dt"] = dt(phase_accumulation_)
    res_fe["phase_accumulation_ddt"] = ddt(phase_accumulation_)
    for lag in range(1, LAG_MAX):
        res_fe[f"phase_accumulation_lag{lag}"] = lag(phase_accumulation_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"phase_accumulation_dt_lag{lag}"] = lag(
            res_fe["phase_accumulation_dt"], lag
        )
    for lag in range(1, LAG_MAX):
        res_fe[f"phase_accumulation_ddt_lag{lag}"] = lag(
            res_fe["phase_accumulation_ddt"], lag
        )

    # Polarized Fractal Efficiency
    pfe_ = ta.pfe(candles, sequential=True)
    res_fe["pfe"] = pfe_
    res_fe["pfe_dt"] = dt(pfe_)
    res_fe["pfe_ddt"] = ddt(pfe_)
    for lag in range(1, LAG_MAX):
        res_fe[f"pfe_lag{lag}"] = lag(pfe_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"pfe_dt_lag{lag}"] = lag(res_fe["pfe_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"pfe_ddt_lag{lag}"] = lag(res_fe["pfe_ddt"], lag)

    # roofing filter
    rf = roofing_filter(candles, sequential=True)
    res_fe["roofing_filter"] = rf
    res_fe["roofing_filter_dt"] = dt(rf)
    res_fe["roofing_filter_ddt"] = ddt(rf)
    for lag in range(1, LAG_MAX):
        res_fe[f"roofing_filter_lag{lag}"] = lag(rf, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"roofing_filter_dt_lag{lag}"] = lag(res_fe["roofing_filter_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"roofing_filter_ddt_lag{lag}"] = lag(res_fe["roofing_filter_ddt"], lag)

    # Schaff Trend Cycle
    stc_ = ta.stc(candles, sequential=True)
    res_fe["stc"] = stc_
    res_fe["stc_dt"] = dt(stc_)
    res_fe["stc_ddt"] = ddt(stc_)
    for lag in range(1, LAG_MAX):
        res_fe[f"stc_lag{lag}"] = lag(stc_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"stc_dt_lag{lag}"] = lag(res_fe["stc_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"stc_ddt_lag{lag}"] = lag(res_fe["stc_ddt"], lag)

    # stiffness
    stiffness_ = ta.stiffness(candles, sequential=True)
    res_fe["stiffness"] = stiffness_
    res_fe["stiffness_dt"] = dt(stiffness_)
    res_fe["stiffness_ddt"] = ddt(stiffness_)
    for lag in range(1, LAG_MAX):
        res_fe[f"stiffness_lag{lag}"] = lag(stiffness_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"stiffness_dt_lag{lag}"] = lag(res_fe["stiffness_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"stiffness_ddt_lag{lag}"] = lag(res_fe["stiffness_ddt"], lag)

    # swamicharts rsi
    lookback, swamicharts_rsi_ = swamicharts_rsi(candles, sequential=True)
    for i in range(swamicharts_rsi_.shape[1]):
        res_fe[f"swamicharts_rsi_{i}"] = swamicharts_rsi_[:, i]

    # swamicharts stochastic
    lookback, swamicharts_stochastic_ = swamicharts_stochastic(candles, sequential=True)
    for i in range(swamicharts_stochastic_.shape[1]):
        res_fe[f"swamicharts_stochastic_{i}"] = swamicharts_stochastic_[:, i]

    # td sequential
    td_sequential_buy, td_sequential_sell = td_sequential(candles, sequential=True)
    res_fe["td_sequential_buy"] = td_sequential_buy
    res_fe["td_sequential_sell"] = td_sequential_sell
    td_sequential_buy_aggressive, td_sequential_sell_aggressive = td_sequential(
        candles, sequential=True, aggressive=True
    )
    res_fe["td_sequential_buy_aggressive"] = td_sequential_buy_aggressive
    res_fe["td_sequential_sell_aggressive"] = td_sequential_sell_aggressive
    td_sequential_buy_stealth, td_sequential_sell_stealth = td_sequential(
        candles, sequential=True, stealth_actions=True
    )
    res_fe["td_sequential_buy_stealth"] = td_sequential_buy_stealth
    res_fe["td_sequential_sell_stealth"] = td_sequential_sell_stealth
    td_sequential_buy_aggressive_stealth, td_sequential_sell_aggressive_stealth = (
        td_sequential(candles, sequential=True, aggressive=True, stealth_actions=True)
    )
    res_fe["td_sequential_buy_aggressive_stealth"] = (
        td_sequential_buy_aggressive_stealth
    )
    res_fe["td_sequential_sell_aggressive_stealth"] = (
        td_sequential_sell_aggressive_stealth
    )

    # trendflex
    trendflex_ = ta.trendflex(candles, sequential=True)
    res_fe["trendflex"] = trendflex_
    res_fe["trendflex_dt"] = dt(trendflex_)
    res_fe["trendflex_ddt"] = ddt(trendflex_)
    for lag in range(1, LAG_MAX):
        res_fe[f"trendflex_lag{lag}"] = lag(trendflex_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"trendflex_dt_lag{lag}"] = lag(res_fe["trendflex_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"trendflex_ddt_lag{lag}"] = lag(res_fe["trendflex_ddt"], lag)

    # Voss Filter
    voss_filter_ = ta.voss_filter(candles, sequential=True)
    res_fe["voss_filter"] = voss_filter_
    res_fe["voss_filter_dt"] = dt(voss_filter_)
    res_fe["voss_filter_ddt"] = ddt(voss_filter_)
    for lag in range(1, LAG_MAX):
        res_fe[f"voss_filter_lag{lag}"] = lag(voss_filter_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"voss_filter_dt_lag{lag}"] = lag(res_fe["voss_filter_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"voss_filter_ddt_lag{lag}"] = lag(res_fe["voss_filter_ddt"], lag)

    # vwap
    vwap_ = ta.vwap(candles, sequential=True)
    res_fe["vwap"] = vwap_
    res_fe["vwap_dt"] = dt(vwap_)
    res_fe["vwap_ddt"] = ddt(vwap_)
    for lag in range(1, LAG_MAX):
        res_fe[f"vwap_lag{lag}"] = lag(vwap_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"vwap_dt_lag{lag}"] = lag(res_fe["vwap_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"vwap_ddt_lag{lag}"] = lag(res_fe["vwap_ddt"], lag)

    # Williams' %R
    williams_r_ = ta.willr(candles, sequential=True)
    res_fe["williams_r"] = williams_r_
    res_fe["williams_r_dt"] = dt(williams_r_)
    res_fe["williams_r_ddt"] = ddt(williams_r_)
    for lag in range(1, LAG_MAX):
        res_fe[f"williams_r_lag{lag}"] = lag(williams_r_, lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"williams_r_dt_lag{lag}"] = lag(res_fe["williams_r_dt"], lag)
    for lag in range(1, LAG_MAX):
        res_fe[f"williams_r_ddt_lag{lag}"] = lag(res_fe["williams_r_ddt"], lag)

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
