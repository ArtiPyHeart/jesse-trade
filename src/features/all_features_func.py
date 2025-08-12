import jesse.indicators as ta
import numpy as np
from jesse import helpers
from jesse.indicators import adx, aroon
from jesse.indicators.aroon import AROON

from src.indicators.dominant_cycle import (
    dual_differentiator,
    homodyne,
    phase_accumulation,
)
from src.indicators.prod import (
    accumulated_swing_index,
    adaptive_bandpass,
    adaptive_cci,
    adaptive_rsi,
    adaptive_stochastic,
    autocorrelation,
    autocorrelation_periodogram,
    autocorrelation_reversals,
    chaiken_money_flow,
    change_variance_ratio,
    cmma,
    comb_spectrum,
    dft,
    ehlers_convolution,
    ehlers_early_onset_trend,
    evenbetter_sinewave,
    fti,
    hurst_coefficient,
    iqr_ratio,
    ma_difference,
    mod_rsi,
    mod_stochastic,
    norm_on_balance_volume,
    price_change_oscillator,
    price_variance_ratio,
    reactivity,
    roofing_filter,
    swamicharts_rsi,
    swamicharts_stochastic,
    td_sequential,
    sample_entropy_indicator,
    approximate_entropy_indicator,
    frac_diff_ffd_candle,
    FTIResult,
    VMD_NRBO,
    amihud_lambda,
    bekker_parkinson_vol,
    corwin_schultz_estimator,
    hasbrouck_lambda,
    kyle_lambda,
    roll_impact,
    roll_measure,
    entropy_for_jesse,
    CWT_SWT
)
from src.utils.math_tools import ddt, dt, lag

LAG_MAX = 20

def feature_bundle(
        candles: np.ndarray, sequential: bool = False, lightweighted: bool = False
) -> dict[str, np.ndarray]:
    """
    一次性计算所有特征的函数，主要用于线下训练模型时使用
    Args:
        candles: jesse风格K线
        sequential: 是否只输出最新值
        lightweighted: 轻量模型，用于忽略计算较重的特征。通常在寻找自定义轴的tuning_pipeline.py中使用

    Returns:
        Dict[特征名称, 特征numpy array]
    """
    candles = helpers.slice_candles(candles, sequential)
    res_fe = {}

    # adx
    adx_7 = adx(candles, period=7, sequential=True)
    adx_14 = adx(candles, period=14, sequential=True)
    res_fe["adx_7"] = adx_7
    res_fe["adx_7_dt"] = dt(adx_7)
    res_fe["adx_7_ddt"] = ddt(adx_7)
    for lg in range(1, LAG_MAX):
        res_fe[f"adx_7_lag{lg}"] = lag(adx_7, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adx_7_dt_lag{lg}"] = lag(res_fe["adx_7_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adx_7_ddt_lag{lg}"] = lag(res_fe["adx_7_ddt"], lg)
    res_fe["adx_14"] = adx_14
    res_fe["adx_14_dt"] = dt(adx_14)
    res_fe["adx_14_ddt"] = ddt(adx_14)
    for lg in range(1, LAG_MAX):
        res_fe[f"adx_14_lag{lg}"] = lag(adx_14, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adx_14_dt_lag{lg}"] = lag(res_fe["adx_14_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adx_14_ddt_lag{lg}"] = lag(res_fe["adx_14_ddt"], lg)

    # aroon diff
    aroon_: AROON = aroon(candles, sequential=True)
    res_fe["aroon_diff"] = aroon_.up - aroon_.down
    res_fe["aroon_diff_dt"] = dt(res_fe["aroon_diff"])
    res_fe["aroon_diff_ddt"] = ddt(res_fe["aroon_diff"])
    for lg in range(1, LAG_MAX):
        res_fe[f"aroon_diff_lag{lg}"] = lag(res_fe["aroon_diff"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"aroon_diff_dt_lag{lg}"] = lag(res_fe["aroon_diff_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"aroon_diff_ddt_lag{lg}"] = lag(res_fe["aroon_diff_ddt"], lg)

    # autocorrelation
    auto_corr = autocorrelation(candles, sequential=True)
    for i in range(auto_corr.shape[1]):
        res_fe[f"ac_{i}"] = auto_corr[:, i]

    # accumulated swing index
    acc_swing_index = accumulated_swing_index(candles, sequential=True)
    res_fe["acc_swing_index"] = acc_swing_index
    res_fe["acc_swing_index_dt"] = dt(acc_swing_index)
    res_fe["acc_swing_index_ddt"] = ddt(acc_swing_index)
    for lg in range(1, LAG_MAX):
        res_fe[f"acc_swing_index_lag{lg}"] = lag(acc_swing_index, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"acc_swing_index_dt_lag{lg}"] = lag(res_fe["acc_swing_index_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"acc_swing_index_ddt_lag{lg}"] = lag(res_fe["acc_swing_index_ddt"], lg)

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
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lag{lg}"] = lag(adaptive_bp, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_dt_lag{lg}"] = lag(res_fe["adaptive_bp_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_ddt_lag{lg}"] = lag(res_fe["adaptive_bp_ddt"], lg)
    res_fe["adaptive_bp_lead"] = adaptive_bp_lead
    res_fe["adaptive_bp_lead_dt"] = dt(adaptive_bp_lead)
    res_fe["adaptive_bp_lead_ddt"] = ddt(adaptive_bp_lead)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lead_lag{lg}"] = lag(adaptive_bp_lead, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lead_dt_lag{lg}"] = lag(res_fe["adaptive_bp_lead_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_bp_lead_ddt_lag{lg}"] = lag(
            res_fe["adaptive_bp_lead_ddt"], lg
        )

    # adaptive cci
    adaptive_cci_ = adaptive_cci(candles, sequential=True)
    res_fe["adaptive_cci"] = adaptive_cci_
    res_fe["adaptive_cci_dt"] = dt(adaptive_cci_)
    res_fe["adaptive_cci_ddt"] = ddt(adaptive_cci_)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_cci_lag{lg}"] = lag(adaptive_cci_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_cci_dt_lag{lg}"] = lag(res_fe["adaptive_cci_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_cci_ddt_lag{lg}"] = lag(res_fe["adaptive_cci_ddt"], lg)

    # adaptive rsi
    adaptive_rsi_ = adaptive_rsi(candles, sequential=True)
    res_fe["adaptive_rsi"] = adaptive_rsi_
    res_fe["adaptive_rsi_dt"] = dt(adaptive_rsi_)
    res_fe["adaptive_rsi_ddt"] = ddt(adaptive_rsi_)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_rsi_lag{lg}"] = lag(adaptive_rsi_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_rsi_dt_lag{lg}"] = lag(res_fe["adaptive_rsi_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_rsi_ddt_lag{lg}"] = lag(res_fe["adaptive_rsi_ddt"], lg)

    # adaptive stochastic
    adaptive_stochastic_ = adaptive_stochastic(candles, sequential=True)
    res_fe["adaptive_stochastic"] = adaptive_stochastic_
    res_fe["adaptive_stochastic_dt"] = dt(adaptive_stochastic_)
    res_fe["adaptive_stochastic_ddt"] = ddt(adaptive_stochastic_)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_stochastic_lag{lg}"] = lag(adaptive_stochastic_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_stochastic_dt_lag{lg}"] = lag(
            res_fe["adaptive_stochastic_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"adaptive_stochastic_ddt_lag{lg}"] = lag(
            res_fe["adaptive_stochastic_ddt"], lg
        )

    # amihud lambda
    amihud_lambda_ = amihud_lambda(candles, sequential=True)
    res_fe["amihud_lambda"] = amihud_lambda_
    res_fe["amihud_lambda_dt"] = dt(amihud_lambda_)
    res_fe["amihud_lambda_ddt"] = ddt(amihud_lambda_)
    for lg in range(1, LAG_MAX):
        res_fe[f"amihud_lambda_lag{lg}"] = lag(amihud_lambda_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"amihud_lambda_dt_lag{lg}"] = lag(res_fe["amihud_lambda_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"amihud_lambda_ddt_lag{lg}"] = lag(res_fe["amihud_lambda_ddt"], lg)

    # bandpass & highpass
    bandpass_tuple = ta.bandpass(candles, sequential=True)
    res_fe["bandpass"] = bandpass_tuple.bp_normalized
    res_fe["bandpass_dt"] = dt(bandpass_tuple.bp_normalized)
    res_fe["bandpass_ddt"] = ddt(bandpass_tuple.bp_normalized)
    for lg in range(1, LAG_MAX):
        res_fe[f"bandpass_lag{lg}"] = lag(bandpass_tuple.bp_normalized, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"bandpass_dt_lag{lg}"] = lag(res_fe["bandpass_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"bandpass_ddt_lag{lg}"] = lag(res_fe["bandpass_ddt"], lg)
    res_fe["highpass_bp"] = bandpass_tuple.trigger
    res_fe["highpass_bp_dt"] = dt(bandpass_tuple.trigger)
    res_fe["highpass_bp_ddt"] = ddt(bandpass_tuple.trigger)
    for lg in range(1, LAG_MAX):
        res_fe[f"highpass_bp_lag{lg}"] = lag(bandpass_tuple.trigger, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"highpass_bp_dt_lag{lg}"] = lag(res_fe["highpass_bp_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"highpass_bp_ddt_lag{lg}"] = lag(res_fe["highpass_bp_ddt"], lg)

    # bekker_parkinson_vol
    bekker_parkinson_vol_ = bekker_parkinson_vol(candles, sequential=True)
    res_fe["bekker_parkinson_vol"] = bekker_parkinson_vol_
    res_fe["bekker_parkinson_vol_dt"] = dt(bekker_parkinson_vol_)
    res_fe["bekker_parkinson_vol_ddt"] = ddt(bekker_parkinson_vol_)
    for lg in range(1, LAG_MAX):
        res_fe[f"bekker_parkinson_vol_lag{lg}"] = lag(bekker_parkinson_vol_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"bekker_parkinson_vol_dt_lag{lg}"] = lag(
            res_fe["bekker_parkinson_vol_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"bekker_parkinson_vol_ddt_lag{lg}"] = lag(
            res_fe["bekker_parkinson_vol_ddt"], lg
        )

    # chaiken money flow
    chaiken_money_flow_ = chaiken_money_flow(candles, sequential=True)
    res_fe["chaiken_money_flow"] = chaiken_money_flow_
    res_fe["chaiken_money_flow_dt"] = dt(chaiken_money_flow_)
    res_fe["chaiken_money_flow_ddt"] = ddt(chaiken_money_flow_)
    for lg in range(1, LAG_MAX):
        res_fe[f"chaiken_money_flow_lag{lg}"] = lag(chaiken_money_flow_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"chaiken_money_flow_dt_lag{lg}"] = lag(
            res_fe["chaiken_money_flow_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"chaiken_money_flow_ddt_lag{lg}"] = lag(
            res_fe["chaiken_money_flow_ddt"], lg
        )

    # change variance ratio
    change_variance_ratio_ = change_variance_ratio(candles, sequential=True)
    res_fe["change_variance_ratio"] = change_variance_ratio_
    res_fe["change_variance_ratio_dt"] = dt(change_variance_ratio_)
    res_fe["change_variance_ratio_ddt"] = ddt(change_variance_ratio_)
    for lg in range(1, LAG_MAX):
        res_fe[f"change_variance_ratio_lag{lg}"] = lag(change_variance_ratio_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"change_variance_ratio_dt_lag{lg}"] = lag(
            res_fe["change_variance_ratio_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"change_variance_ratio_ddt_lag{lg}"] = lag(
            res_fe["change_variance_ratio_ddt"], lg
        )

    # cmma
    cmma_ = cmma(candles, sequential=True)
    res_fe["cmma"] = cmma_
    res_fe["cmma_dt"] = dt(cmma_)
    res_fe["cmma_ddt"] = ddt(cmma_)
    for lg in range(1, LAG_MAX):
        res_fe[f"cmma_lag{lg}"] = lag(cmma_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"cmma_dt_lag{lg}"] = lag(res_fe["cmma_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"cmma_ddt_lag{lg}"] = lag(res_fe["cmma_ddt"], lg)

    # corwin_schultz_estimator
    corwin_schultz_estimator_ = corwin_schultz_estimator(candles, sequential=True)
    res_fe["corwin_schultz_estimator"] = corwin_schultz_estimator_
    res_fe["corwin_schultz_estimator_dt"] = dt(corwin_schultz_estimator_)
    res_fe["corwin_schultz_estimator_ddt"] = ddt(corwin_schultz_estimator_)
    for lg in range(1, LAG_MAX):
        res_fe[f"corwin_schultz_estimator_lag{lg}"] = lag(corwin_schultz_estimator_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"corwin_schultz_estimator_dt_lag{lg}"] = lag(
            res_fe["corwin_schultz_estimator_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"corwin_schultz_estimator_ddt_lag{lg}"] = lag(
            res_fe["corwin_schultz_estimator_ddt"], lg
        )

    # comb spectrum
    comb_spectrum_dom_cycle, pwr = comb_spectrum(candles, sequential=True)
    res_fe["comb_spectrum_dom_cycle"] = comb_spectrum_dom_cycle
    res_fe["comb_spectrum_dom_cycle_dt"] = dt(comb_spectrum_dom_cycle)
    res_fe["comb_spectrum_dom_cycle_ddt"] = ddt(comb_spectrum_dom_cycle)
    for lg in range(1, LAG_MAX):
        res_fe[f"comb_spectrum_dom_cycle_lag{lg}"] = lag(comb_spectrum_dom_cycle, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"comb_spectrum_dom_cycle_dt_lag{lg}"] = lag(
            res_fe["comb_spectrum_dom_cycle_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"comb_spectrum_dom_cycle_ddt_lag{lg}"] = lag(
            res_fe["comb_spectrum_dom_cycle_ddt"], lg
        )
    for i in range(pwr.shape[1]):
        res_fe[f"comb_spectrum_pwr_{i}"] = pwr[:, i]

    # convolution
    _, _, conv = ehlers_convolution(candles, sequential=True)
    for i in range(conv.shape[1]):
        res_fe[f"conv_{i}"] = conv[:, i]

    # cwt wavelet
    if not lightweighted:
        cwt_win32 = CWT_SWT(candles, 32, sequential=True)
        cwt_win32_res = cwt_win32.res()
        for i in range(cwt_win32_res.shape[1]):
            res_fe[f"cwt_win32_{i}"] = cwt_win32_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win32_lag_res = cwt_win32.res(lag=lg)
            for i in range(cwt_win32_lag_res.shape[1]):
                res_fe[f"cwt_win32_{i}_lag{lg}"] = cwt_win32_lag_res[:, i]
        cwt_win32_dt_res = cwt_win32.res(dt=True)
        for i in range(cwt_win32_dt_res.shape[1]):
            res_fe[f"cwt_win32_{i}_dt"] = cwt_win32_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win32_dt_lag_res = cwt_win32.res(dt=True, lag=lg)
            for i in range(cwt_win32_dt_lag_res.shape[1]):
                res_fe[f"cwt_win32_{i}_dt_lag{lg}"] = cwt_win32_dt_lag_res[:, i]
        cwt_win32_ddt_res = cwt_win32.res(ddt=True)
        for i in range(cwt_win32_ddt_res.shape[1]):
            res_fe[f"cwt_win32_{i}_ddt"] = cwt_win32_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win32_ddt_lag_res = cwt_win32.res(ddt=True, lag=lg)
            for i in range(cwt_win32_ddt_lag_res.shape[1]):
                res_fe[f"cwt_win32_{i}_ddt_lag{lg}"] = cwt_win32_ddt_lag_res[:, i]

    if not lightweighted:
        cwt_win64 = CWT_SWT(candles, 64, sequential=True)
        cwt_win64_res = cwt_win64.res()
        for i in range(cwt_win64_res.shape[1]):
            res_fe[f"cwt_win64_{i}"] = cwt_win64_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win64_lag_res = cwt_win64.res(lag=lg)
            for i in range(cwt_win64_lag_res.shape[1]):
                res_fe[f"cwt_win64_{i}_lag{lg}"] = cwt_win64_lag_res[:, i]
        cwt_win64_dt_res = cwt_win64.res(dt=True)
        for i in range(cwt_win64_dt_res.shape[1]):
            res_fe[f"cwt_win64_{i}_dt"] = cwt_win64_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win64_dt_lag_res = cwt_win64.res(dt=True, lag=lg)
            for i in range(cwt_win64_dt_lag_res.shape[1]):
                res_fe[f"cwt_win64_{i}_dt_lag{lg}"] = cwt_win64_dt_lag_res[:, i]
        cwt_win64_ddt_res = cwt_win64.res(ddt=True)
        for i in range(cwt_win64_ddt_res.shape[1]):
            res_fe[f"cwt_win64_{i}_ddt"] = cwt_win64_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win64_ddt_lag_res = cwt_win64.res(ddt=True, lag=lg)
            for i in range(cwt_win64_ddt_lag_res.shape[1]):
                res_fe[f"cwt_win64_{i}_ddt_lag{lg}"] = cwt_win64_ddt_lag_res[:, i]

    if not lightweighted:
        cwt_win128 = CWT_SWT(candles, 128, sequential=True)
        cwt_win128_res = cwt_win128.res()
        for i in range(cwt_win128_res.shape[1]):
            res_fe[f"cwt_win128_{i}"] = cwt_win128_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win128_lag_res = cwt_win128.res(lag=lg)
            for i in range(cwt_win128_lag_res.shape[1]):
                res_fe[f"cwt_win128_{i}_lag{lg}"] = cwt_win128_lag_res[:, i]
        cwt_win128_dt_res = cwt_win128.res(dt=True)
        for i in range(cwt_win128_dt_res.shape[1]):
            res_fe[f"cwt_win128_{i}_dt"] = cwt_win128_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win128_dt_lag_res = cwt_win128.res(dt=True, lag=lg)
            for i in range(cwt_win128_dt_lag_res.shape[1]):
                res_fe[f"cwt_win128_{i}_dt_lag{lg}"] = cwt_win128_dt_lag_res[:, i]
        cwt_win128_ddt_res = cwt_win128.res(ddt=True)
        for i in range(cwt_win128_ddt_res.shape[1]):
            res_fe[f"cwt_win128_{i}_ddt"] = cwt_win128_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win128_ddt_lag_res = cwt_win128.res(ddt=True, lag=lg)
            for i in range(cwt_win128_ddt_lag_res.shape[1]):
                res_fe[f"cwt_win128_{i}_ddt_lag{lg}"] = cwt_win128_ddt_lag_res[:, i]

    if not lightweighted:
        cwt_win256 = CWT_SWT(candles, 256, sequential=True)
        cwt_win256_res = cwt_win256.res()
        for i in range(cwt_win256_res.shape[1]):
            res_fe[f"cwt_win256_{i}"] = cwt_win256_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win256_lag_res = cwt_win256.res(lag=lg)
            for i in range(cwt_win256_lag_res.shape[1]):
                res_fe[f"cwt_win256_{i}_lag{lg}"] = cwt_win256_lag_res[:, i]
        cwt_win256_dt_res = cwt_win256.res(dt=True)
        for i in range(cwt_win256_dt_res.shape[1]):
            res_fe[f"cwt_win256_{i}_dt"] = cwt_win256_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win256_dt_lag_res = cwt_win256.res(dt=True, lag=lg)
            for i in range(cwt_win256_dt_lag_res.shape[1]):
                res_fe[f"cwt_win256_{i}_dt_lag{lg}"] = cwt_win256_dt_lag_res[:, i]
        cwt_win256_ddt_res = cwt_win256.res(ddt=True)
        for i in range(cwt_win256_ddt_res.shape[1]):
            res_fe[f"cwt_win256_{i}_ddt"] = cwt_win256_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win256_ddt_lag_res = cwt_win256.res(ddt=True, lag=lg)
            for i in range(cwt_win256_ddt_lag_res.shape[1]):
                res_fe[f"cwt_win256_{i}_ddt_lag{lg}"] = cwt_win256_ddt_lag_res[:, i]

    # lightweighted模式下仅保留512长度特征防止耗尽内存
    cwt_win512 = CWT_SWT(candles, 512, sequential=True)
    cwt_win512_res = cwt_win512.res()
    for i in range(cwt_win512_res.shape[1]):
        res_fe[f"cwt_win512_{i}"] = cwt_win512_res[:, i]
    for lg in range(1, LAG_MAX):
        cwt_win512_lag_res = cwt_win512.res(lag=lg)
        for i in range(cwt_win512_lag_res.shape[1]):
            res_fe[f"cwt_win512_{i}_lag{lg}"] = cwt_win512_lag_res[:, i]
    cwt_win512_dt_res = cwt_win512.res(dt=True)
    for i in range(cwt_win512_dt_res.shape[1]):
        res_fe[f"cwt_win512_{i}_dt"] = cwt_win512_dt_res[:, i]
    for lg in range(1, LAG_MAX):
        cwt_win512_dt_lag_res = cwt_win512.res(dt=True, lag=lg)
        for i in range(cwt_win512_dt_lag_res.shape[1]):
            res_fe[f"cwt_win512_{i}_dt_lag{lg}"] = cwt_win512_dt_lag_res[:, i]
    cwt_win512_ddt_res = cwt_win512.res(ddt=True)
    for i in range(cwt_win512_ddt_res.shape[1]):
        res_fe[f"cwt_win512_{i}_ddt"] = cwt_win512_ddt_res[:, i]
    for lg in range(1, LAG_MAX):
        cwt_win512_ddt_lag_res = cwt_win512.res(ddt=True, lag=lg)
        for i in range(cwt_win512_ddt_lag_res.shape[1]):
            res_fe[f"cwt_win512_{i}_ddt_lag{lg}"] = cwt_win512_ddt_lag_res[:, i]

    if not lightweighted:
        cwt_win1024 = CWT_SWT(candles, 1024, sequential=True)
        cwt_win1024_res = cwt_win1024.res()
        for i in range(cwt_win1024_res.shape[1]):
            res_fe[f"cwt_win1024_{i}"] = cwt_win1024_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win1024_lag_res = cwt_win1024.res(lag=lg)
            for i in range(cwt_win1024_lag_res.shape[1]):
                res_fe[f"cwt_win1024_{i}_lag{lg}"] = cwt_win1024_lag_res[:, i]
        cwt_win1024_dt_res = cwt_win1024.res(dt=True)
        for i in range(cwt_win1024_dt_res.shape[1]):
            res_fe[f"cwt_win1024_{i}_dt"] = cwt_win1024_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win1024_dt_lag_res = cwt_win1024.res(dt=True, lag=lg)
            for i in range(cwt_win1024_dt_lag_res.shape[1]):
                res_fe[f"cwt_win1024_{i}_dt_lag{lg}"] = cwt_win1024_dt_lag_res[:, i]
        cwt_win1024_ddt_res = cwt_win1024.res(ddt=True)
        for i in range(cwt_win1024_ddt_res.shape[1]):
            res_fe[f"cwt_win1024_{i}_ddt"] = cwt_win1024_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            cwt_win1024_ddt_lag_res = cwt_win1024.res(ddt=True, lag=lg)
            for i in range(cwt_win1024_ddt_lag_res.shape[1]):
                res_fe[f"cwt_win1024_{i}_ddt_lag{lg}"] = cwt_win1024_ddt_lag_res[:, i]

    # dft
    dft_dom_cycle, spectrum = dft(candles, sequential=True)
    res_fe["dft_dom_cycle"] = dft_dom_cycle
    res_fe["dft_dom_cycle_dt"] = dt(dft_dom_cycle)
    res_fe["dft_dom_cycle_ddt"] = ddt(dft_dom_cycle)
    for lg in range(1, LAG_MAX):
        res_fe[f"dft_dom_cycle_lag{lg}"] = lag(dft_dom_cycle, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"dft_dom_cycle_dt_lag{lg}"] = lag(res_fe["dft_dom_cycle_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"dft_dom_cycle_ddt_lag{lg}"] = lag(res_fe["dft_dom_cycle_ddt"], lg)
    for i in range(spectrum.shape[1]):
        res_fe[f"dft_spectrum_{i}"] = spectrum[:, i]

    # dual differentiator
    dual_diff = dual_differentiator(candles, sequential=True)
    res_fe["dual_diff"] = dual_diff
    res_fe["dual_diff_dt"] = dt(dual_diff)
    res_fe["dual_diff_ddt"] = ddt(dual_diff)
    for lg in range(1, LAG_MAX):
        res_fe[f"dual_diff_lag{lg}"] = lag(dual_diff, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"dual_diff_dt_lag{lg}"] = lag(res_fe["dual_diff_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"dual_diff_ddt_lag{lg}"] = lag(res_fe["dual_diff_ddt"], lg)

    # ehlers early onset trend
    ehlers_early_onset_trend_ = ehlers_early_onset_trend(candles, sequential=True)
    res_fe["ehlers_early_onset_trend"] = ehlers_early_onset_trend_
    res_fe["ehlers_early_onset_trend_dt"] = dt(ehlers_early_onset_trend_)
    res_fe["ehlers_early_onset_trend_ddt"] = ddt(ehlers_early_onset_trend_)
    for lg in range(1, LAG_MAX):
        res_fe[f"ehlers_early_onset_trend_lag{lg}"] = lag(ehlers_early_onset_trend_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"ehlers_early_onset_trend_dt_lag{lg}"] = lag(
            res_fe["ehlers_early_onset_trend_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"ehlers_early_onset_trend_ddt_lag{lg}"] = lag(
            res_fe["ehlers_early_onset_trend_ddt"], lg
        )

    # entropy sample
    if not lightweighted:
        sample_entropy_win32_spot = sample_entropy_indicator(
            candles, period=32, use_array_price=False, sequential=True
        )
        res_fe["sample_entropy_win32_spot"] = sample_entropy_win32_spot

        sample_entropy_win32_array = sample_entropy_indicator(
            candles, period=32, use_array_price=True, sequential=True
        )
        res_fe["sample_entropy_win32_array"] = sample_entropy_win32_array

        sample_entropy_win64_spot = sample_entropy_indicator(
            candles, period=64, use_array_price=False, sequential=True
        )
        res_fe["sample_entropy_win64_spot"] = sample_entropy_win64_spot

        sample_entropy_win64_array = sample_entropy_indicator(
            candles, period=64, use_array_price=True, sequential=True
        )
        res_fe["sample_entropy_win64_array"] = sample_entropy_win64_array

        sample_entropy_win128_spot = sample_entropy_indicator(
            candles, period=128, use_array_price=False, sequential=True
        )
        res_fe["sample_entropy_win128_spot"] = sample_entropy_win128_spot

        sample_entropy_win128_array = sample_entropy_indicator(
            candles, period=128, use_array_price=True, sequential=True
        )
        res_fe["sample_entropy_win128_array"] = sample_entropy_win128_array

        sample_entropy_win256_spot = sample_entropy_indicator(
            candles, period=256, use_array_price=False, sequential=True
        )
        res_fe["sample_entropy_win256_spot"] = sample_entropy_win256_spot

        sample_entropy_win256_array = sample_entropy_indicator(
            candles, period=256, use_array_price=True, sequential=True
        )
        res_fe["sample_entropy_win256_array"] = sample_entropy_win256_array

    sample_entropy_win512_spot = sample_entropy_indicator(
        candles, period=512, use_array_price=False, sequential=True
    )
    res_fe["sample_entropy_win512_spot"] = sample_entropy_win512_spot

    sample_entropy_win512_array = sample_entropy_indicator(
        candles, period=512, use_array_price=True, sequential=True
    )
    res_fe["sample_entropy_win512_array"] = sample_entropy_win512_array

    # entropy approximate
    if not lightweighted:
        approximate_entropy_win32_spot = approximate_entropy_indicator(
            candles, period=32, use_array_price=False, sequential=True
        )
        res_fe["approximate_entropy_win32_spot"] = approximate_entropy_win32_spot

        approximate_entropy_win32_array = approximate_entropy_indicator(
            candles, period=32, use_array_price=True, sequential=True
        )
        res_fe["approximate_entropy_win32_array"] = approximate_entropy_win32_array

        approximate_entropy_win64_spot = approximate_entropy_indicator(
            candles, period=64, use_array_price=False, sequential=True
        )
        res_fe["approximate_entropy_win64_spot"] = approximate_entropy_win64_spot

        approximate_entropy_win64_array = approximate_entropy_indicator(
            candles, period=64, use_array_price=True, sequential=True
        )
        res_fe["approximate_entropy_win64_array"] = approximate_entropy_win64_array

        approximate_entropy_win128_spot = approximate_entropy_indicator(
            candles, period=128, use_array_price=False, sequential=True
        )
        res_fe["approximate_entropy_win128_spot"] = approximate_entropy_win128_spot

        approximate_entropy_win128_array = approximate_entropy_indicator(
            candles, period=128, use_array_price=True, sequential=True
        )
        res_fe["approximate_entropy_win128_array"] = approximate_entropy_win128_array

        approximate_entropy_win256_spot = approximate_entropy_indicator(
            candles, period=256, use_array_price=False, sequential=True
        )
        res_fe["approximate_entropy_win256_spot"] = approximate_entropy_win256_spot

        approximate_entropy_win256_array = approximate_entropy_indicator(
            candles, period=256, use_array_price=True, sequential=True
        )
        res_fe["approximate_entropy_win256_array"] = approximate_entropy_win256_array

    approximate_entropy_win512_spot = approximate_entropy_indicator(
        candles, period=512, use_array_price=False, sequential=True
    )
    res_fe["approximate_entropy_win512_spot"] = approximate_entropy_win512_spot

    approximate_entropy_win512_array = approximate_entropy_indicator(
        candles, period=512, use_array_price=True, sequential=True
    )
    res_fe["approximate_entropy_win512_array"] = approximate_entropy_win512_array

    # entropy for jesse
    if not lightweighted:
        entropy_for_jesse_ = entropy_for_jesse(candles, sequential=True)
        res_fe["entropy_for_jesse"] = entropy_for_jesse_
        res_fe["entropy_for_jesse_dt"] = dt(entropy_for_jesse_)
        res_fe["entropy_for_jesse_ddt"] = ddt(entropy_for_jesse_)
        for lg in range(1, LAG_MAX):
            res_fe[f"entropy_for_jesse_lag{lg}"] = lag(entropy_for_jesse_, lg)
        for lg in range(1, LAG_MAX):
            res_fe[f"entropy_for_jesse_dt_lag{lg}"] = lag(
                res_fe["entropy_for_jesse_dt"], lg
            )
        for lg in range(1, LAG_MAX):
            res_fe[f"entropy_for_jesse_ddt_lag{lg}"] = lag(
                res_fe["entropy_for_jesse_ddt"], lg
            )

    # evenbetter sinewave
    eb_sw_long = evenbetter_sinewave(candles, duration=40, sequential=True)
    res_fe["evenbetter_sinewave_long"] = eb_sw_long
    res_fe["evenbetter_sinewave_long_dt"] = dt(eb_sw_long)
    res_fe["evenbetter_sinewave_long_ddt"] = ddt(eb_sw_long)
    for lg in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_long_lag{lg}"] = lag(eb_sw_long, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_long_dt_lag{lg}"] = lag(
            res_fe["evenbetter_sinewave_long_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_long_ddt_lag{lg}"] = lag(
            res_fe["evenbetter_sinewave_long_ddt"], lg
        )
    eb_sw_short = evenbetter_sinewave(candles, duration=20, sequential=True)
    res_fe["evenbetter_sinewave_short"] = eb_sw_short
    res_fe["evenbetter_sinewave_short_dt"] = dt(eb_sw_short)
    res_fe["evenbetter_sinewave_short_ddt"] = ddt(eb_sw_short)
    for lg in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_short_lag{lg}"] = lag(eb_sw_short, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_short_dt_lag{lg}"] = lag(
            res_fe["evenbetter_sinewave_short_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"evenbetter_sinewave_short_ddt_lag{lg}"] = lag(
            res_fe["evenbetter_sinewave_short_ddt"], lg
        )

    # fisher
    fisher_ind = ta.fisher(candles, sequential=True)
    res_fe["fisher"] = fisher_ind.fisher
    res_fe["fisher_dt"] = dt(fisher_ind.fisher)
    res_fe["fisher_ddt"] = ddt(fisher_ind.fisher)
    for lg in range(1, LAG_MAX):
        res_fe[f"fisher_lag{lg}"] = lag(fisher_ind.fisher, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"fisher_dt_lag{lg}"] = lag(res_fe["fisher_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"fisher_ddt_lag{lg}"] = lag(res_fe["fisher_ddt"], lg)

    # forecast oscillator
    forecast_oscillator = ta.fosc(candles, sequential=True)
    res_fe["forecast_oscillator"] = forecast_oscillator
    for lg in range(1, LAG_MAX):
        res_fe[f"forecast_oscillator_lag{lg}"] = lag(forecast_oscillator, lg)

    # frac diff ffd
    frac_diff_ffd_ = frac_diff_ffd_candle(candles, diff_amt=0.35, sequential=True)
    res_fe["frac_diff_ffd"] = frac_diff_ffd_
    for lg in range(1, LAG_MAX):
        res_fe[f"frac_diff_ffd_lag{lg}"] = lag(frac_diff_ffd_, lg)

    # fti
    fti_: FTIResult = fti(candles, sequential=True)
    res_fe["fti"] = fti_.fti
    res_fe["fti_dt"] = dt(fti_.fti)
    res_fe["fti_ddt"] = ddt(fti_.fti)
    for lg in range(1, LAG_MAX):
        res_fe[f"fti_lag{lg}"] = lag(fti_.fti, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"fti_dt_lag{lg}"] = lag(res_fe["fti_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"fti_ddt_lag{lg}"] = lag(res_fe["fti_ddt"], lg)
    res_fe["fti_best_period"] = fti_.best_period
    res_fe["fti_best_period_dt"] = dt(fti_.best_period)
    res_fe["fti_best_period_ddt"] = ddt(fti_.best_period)
    for lg in range(1, LAG_MAX):
        res_fe[f"fti_best_period_lag{lg}"] = lag(fti_.best_period, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"fti_best_period_dt_lag{lg}"] = lag(res_fe["fti_best_period_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"fti_best_period_ddt_lag{lg}"] = lag(res_fe["fti_best_period_ddt"], lg)

    # hasbrouck lambda
    hasbrouck_lambda_ = hasbrouck_lambda(candles, sequential=True)
    res_fe["hasbrouck_lambda"] = hasbrouck_lambda_
    res_fe["hasbrouck_lambda_dt"] = dt(hasbrouck_lambda_)
    res_fe["hasbrouck_lambda_ddt"] = ddt(hasbrouck_lambda_)
    for lg in range(1, LAG_MAX):
        res_fe[f"hasbrouck_lambda_lag{lg}"] = lag(hasbrouck_lambda_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"hasbrouck_lambda_dt_lag{lg}"] = lag(res_fe["hasbrouck_lambda_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"hasbrouck_lambda_ddt_lag{lg}"] = lag(
            res_fe["hasbrouck_lambda_ddt"], lg
        )

    # homodyne
    homodyne_ = homodyne(candles, sequential=True)
    res_fe["homodyne"] = homodyne_
    res_fe["homodyne_dt"] = dt(homodyne_)
    res_fe["homodyne_ddt"] = ddt(homodyne_)
    for lg in range(1, LAG_MAX):
        res_fe[f"homodyne_lag{lg}"] = lag(homodyne_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"homodyne_dt_lag{lg}"] = lag(res_fe["homodyne_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"homodyne_ddt_lag{lg}"] = lag(res_fe["homodyne_ddt"], lg)

    # hurst
    hurst_coef_fast = hurst_coefficient(candles, period=30, sequential=True)
    hurst_coef_slow = hurst_coefficient(candles, period=200, sequential=True)
    res_fe["hurst_coef_fast"] = hurst_coef_fast
    res_fe["hurst_coef_fast_dt"] = dt(hurst_coef_fast)
    res_fe["hurst_coef_fast_ddt"] = ddt(hurst_coef_fast)
    for lg in range(1, LAG_MAX):
        res_fe[f"hurst_coef_fast_lag{lg}"] = lag(hurst_coef_fast, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"hurst_coef_fast_dt_lag{lg}"] = lag(res_fe["hurst_coef_fast_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"hurst_coef_fast_ddt_lag{lg}"] = lag(res_fe["hurst_coef_fast_ddt"], lg)
    res_fe["hurst_coef_slow"] = hurst_coef_slow
    res_fe["hurst_coef_slow_dt"] = dt(hurst_coef_slow)
    res_fe["hurst_coef_slow_ddt"] = ddt(hurst_coef_slow)
    for lg in range(1, LAG_MAX):
        res_fe[f"hurst_coef_slow_lag{lg}"] = lag(hurst_coef_slow, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"hurst_coef_slow_dt_lag{lg}"] = lag(res_fe["hurst_coef_slow_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"hurst_coef_slow_ddt_lag{lg}"] = lag(res_fe["hurst_coef_slow_ddt"], lg)

    # iqr ratio
    iqr_ratio_ = iqr_ratio(candles, sequential=True)
    res_fe["iqr_ratio"] = iqr_ratio_
    res_fe["iqr_ratio_dt"] = dt(iqr_ratio_)
    res_fe["iqr_ratio_ddt"] = ddt(iqr_ratio_)
    for lg in range(1, LAG_MAX):
        res_fe[f"iqr_ratio_lag{lg}"] = lag(iqr_ratio_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"iqr_ratio_dt_lag{lg}"] = lag(res_fe["iqr_ratio_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"iqr_ratio_ddt_lag{lg}"] = lag(res_fe["iqr_ratio_ddt"], lg)

    # kyle lambda
    kyle_lambda_ = kyle_lambda(candles, sequential=True)
    res_fe["kyle_lambda"] = kyle_lambda_
    res_fe["kyle_lambda_dt"] = dt(kyle_lambda_)
    res_fe["kyle_lambda_ddt"] = ddt(kyle_lambda_)
    for lg in range(1, LAG_MAX):
        res_fe[f"kyle_lambda_lag{lg}"] = lag(kyle_lambda_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"kyle_lambda_dt_lag{lg}"] = lag(res_fe["kyle_lambda_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"kyle_lambda_ddt_lag{lg}"] = lag(res_fe["kyle_lambda_ddt"], lg)

    # ma_difference
    ma_difference_ = ma_difference(candles, sequential=True)
    res_fe["ma_difference"] = ma_difference_
    res_fe["ma_difference_dt"] = dt(ma_difference_)
    res_fe["ma_difference_ddt"] = ddt(ma_difference_)
    for lg in range(1, LAG_MAX):
        res_fe[f"ma_difference_lag{lg}"] = lag(ma_difference_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"ma_difference_dt_lag{lg}"] = lag(res_fe["ma_difference_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"ma_difference_ddt_lag{lg}"] = lag(res_fe["ma_difference_ddt"], lg)

    # modified rsi
    mod_rsi_ = mod_rsi(candles, sequential=True)
    res_fe["mod_rsi"] = mod_rsi_
    res_fe["mod_rsi_dt"] = dt(mod_rsi_)
    res_fe["mod_rsi_ddt"] = ddt(mod_rsi_)
    for lg in range(1, LAG_MAX):
        res_fe[f"mod_rsi_lag{lg}"] = lag(mod_rsi_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"mod_rsi_dt_lag{lg}"] = lag(res_fe["mod_rsi_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"mod_rsi_ddt_lag{lg}"] = lag(res_fe["mod_rsi_ddt"], lg)

    # modified stochastic
    mod_stochastic_ = mod_stochastic(candles, roofing_filter=True, sequential=True)
    res_fe["mod_stochastic"] = mod_stochastic_
    res_fe["mod_stochastic_dt"] = dt(mod_stochastic_)
    res_fe["mod_stochastic_ddt"] = ddt(mod_stochastic_)
    for lg in range(1, LAG_MAX):
        res_fe[f"mod_stochastic_lag{lg}"] = lag(mod_stochastic_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"mod_stochastic_dt_lag{lg}"] = lag(res_fe["mod_stochastic_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"mod_stochastic_ddt_lag{lg}"] = lag(res_fe["mod_stochastic_ddt"], lg)

    # natr
    natr_ = ta.natr(candles, sequential=True)
    res_fe["natr"] = natr_
    res_fe["natr_dt"] = dt(natr_)
    res_fe["natr_ddt"] = ddt(natr_)
    for lg in range(1, LAG_MAX):
        res_fe[f"natr_lag{lg}"] = lag(natr_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"natr_dt_lag{lg}"] = lag(res_fe["natr_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"natr_ddt_lag{lg}"] = lag(res_fe["natr_ddt"], lg)

    # norm on balance volume
    norm_on_balance_volume_ = norm_on_balance_volume(candles, sequential=True)
    res_fe["norm_on_balance_volume"] = norm_on_balance_volume_
    res_fe["norm_on_balance_volume_dt"] = dt(norm_on_balance_volume_)
    res_fe["norm_on_balance_volume_ddt"] = ddt(norm_on_balance_volume_)
    for lg in range(1, LAG_MAX):
        res_fe[f"norm_on_balance_volume_lag{lg}"] = lag(norm_on_balance_volume_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"norm_on_balance_volume_dt_lag{lg}"] = lag(
            res_fe["norm_on_balance_volume_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"norm_on_balance_volume_ddt_lag{lg}"] = lag(
            res_fe["norm_on_balance_volume_ddt"], lg
        )

    # phase accumulation
    phase_accumulation_ = phase_accumulation(candles, sequential=True)
    res_fe["phase_accumulation"] = phase_accumulation_
    res_fe["phase_accumulation_dt"] = dt(phase_accumulation_)
    res_fe["phase_accumulation_ddt"] = ddt(phase_accumulation_)
    for lg in range(1, LAG_MAX):
        res_fe[f"phase_accumulation_lag{lg}"] = lag(phase_accumulation_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"phase_accumulation_dt_lag{lg}"] = lag(
            res_fe["phase_accumulation_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"phase_accumulation_ddt_lag{lg}"] = lag(
            res_fe["phase_accumulation_ddt"], lg
        )

    # Polarized Fractal Efficiency
    pfe_ = ta.pfe(candles, sequential=True)
    res_fe["pfe"] = pfe_
    res_fe["pfe_dt"] = dt(pfe_)
    res_fe["pfe_ddt"] = ddt(pfe_)
    for lg in range(1, LAG_MAX):
        res_fe[f"pfe_lag{lg}"] = lag(pfe_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"pfe_dt_lag{lg}"] = lag(res_fe["pfe_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"pfe_ddt_lag{lg}"] = lag(res_fe["pfe_ddt"], lg)

    # price change oscillator
    price_change_oscillator_ = price_change_oscillator(candles, sequential=True)
    res_fe["price_change_oscillator"] = price_change_oscillator_
    res_fe["price_change_oscillator_dt"] = dt(price_change_oscillator_)
    res_fe["price_change_oscillator_ddt"] = ddt(price_change_oscillator_)
    for lg in range(1, LAG_MAX):
        res_fe[f"price_change_oscillator_lag{lg}"] = lag(price_change_oscillator_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"price_change_oscillator_dt_lag{lg}"] = lag(
            res_fe["price_change_oscillator_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"price_change_oscillator_ddt_lag{lg}"] = lag(
            res_fe["price_change_oscillator_ddt"], lg
        )

    # price variance ratio
    price_variance_ratio_ = price_variance_ratio(candles, sequential=True)
    res_fe["price_variance_ratio"] = price_variance_ratio_
    res_fe["price_variance_ratio_dt"] = dt(price_variance_ratio_)
    res_fe["price_variance_ratio_ddt"] = ddt(price_variance_ratio_)
    for lg in range(1, LAG_MAX):
        res_fe[f"price_variance_ratio_lag{lg}"] = lag(price_variance_ratio_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"price_variance_ratio_dt_lag{lg}"] = lag(
            res_fe["price_variance_ratio_dt"], lg
        )
    for lg in range(1, LAG_MAX):
        res_fe[f"price_variance_ratio_ddt_lag{lg}"] = lag(
            res_fe["price_variance_ratio_ddt"], lg
        )

    # reactivity
    reactivity_ = reactivity(candles, sequential=True)
    res_fe["reactivity"] = reactivity_
    res_fe["reactivity_dt"] = dt(reactivity_)
    res_fe["reactivity_ddt"] = ddt(reactivity_)
    for lg in range(1, LAG_MAX):
        res_fe[f"reactivity_lag{lg}"] = lag(reactivity_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"reactivity_dt_lag{lg}"] = lag(res_fe["reactivity_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"reactivity_ddt_lag{lg}"] = lag(res_fe["reactivity_ddt"], lg)

    # roll impact
    roll_impact_ = roll_impact(candles, sequential=True)
    res_fe["roll_impact"] = roll_impact_
    res_fe["roll_impact_dt"] = dt(roll_impact_)
    res_fe["roll_impact_ddt"] = ddt(roll_impact_)
    for lg in range(1, LAG_MAX):
        res_fe[f"roll_impact_lag{lg}"] = lag(roll_impact_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"roll_impact_dt_lag{lg}"] = lag(res_fe["roll_impact_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"roll_impact_ddt_lag{lg}"] = lag(res_fe["roll_impact_ddt"], lg)

    # roll measure
    roll_measure_ = roll_measure(candles, sequential=True)
    res_fe["roll_measure"] = roll_measure_
    res_fe["roll_measure_dt"] = dt(roll_measure_)
    res_fe["roll_measure_ddt"] = ddt(roll_measure_)
    for lg in range(1, LAG_MAX):
        res_fe[f"roll_measure_lag{lg}"] = lag(roll_measure_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"roll_measure_dt_lag{lg}"] = lag(res_fe["roll_measure_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"roll_measure_ddt_lag{lg}"] = lag(res_fe["roll_measure_ddt"], lg)

    # roofing filter
    rf = roofing_filter(candles, sequential=True)
    res_fe["roofing_filter"] = rf
    res_fe["roofing_filter_dt"] = dt(rf)
    res_fe["roofing_filter_ddt"] = ddt(rf)
    for lg in range(1, LAG_MAX):
        res_fe[f"roofing_filter_lag{lg}"] = lag(rf, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"roofing_filter_dt_lag{lg}"] = lag(res_fe["roofing_filter_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"roofing_filter_ddt_lag{lg}"] = lag(res_fe["roofing_filter_ddt"], lg)

    # Schaff Trend Cycle
    stc_ = ta.stc(candles, sequential=True)
    res_fe["stc"] = stc_
    res_fe["stc_dt"] = dt(stc_)
    res_fe["stc_ddt"] = ddt(stc_)
    for lg in range(1, LAG_MAX):
        res_fe[f"stc_lag{lg}"] = lag(stc_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"stc_dt_lag{lg}"] = lag(res_fe["stc_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"stc_ddt_lag{lg}"] = lag(res_fe["stc_ddt"], lg)

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
    for lg in range(1, LAG_MAX):
        res_fe[f"trendflex_lag{lg}"] = lag(trendflex_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"trendflex_dt_lag{lg}"] = lag(res_fe["trendflex_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"trendflex_ddt_lag{lg}"] = lag(res_fe["trendflex_ddt"], lg)

    # VMD
    if not lightweighted:
        vmd_win32 = VMD_NRBO(candles, 32, sequential=True)
        vmd_win32_res = vmd_win32.res()
        for i in range(vmd_win32_res.shape[1]):
            res_fe[f"vmd_win32_{i}"] = vmd_win32_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win32_lag_res = vmd_win32.res(lag=lg)
            for i in range(vmd_win32_lag_res.shape[1]):
                res_fe[f"vmd_win32_{i}_lag{lg}"] = vmd_win32_lag_res[:, i]
        vmd_win32_dt_res = vmd_win32.res(dt=True)
        for i in range(vmd_win32_dt_res.shape[1]):
            res_fe[f"vmd_win32_{i}_dt"] = vmd_win32_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win32_dt_lag_res = vmd_win32.res(dt=True, lag=lg)
            for i in range(vmd_win32_dt_lag_res.shape[1]):
                res_fe[f"vmd_win32_{i}_dt_lag{lg}"] = vmd_win32_dt_lag_res[:, i]
        vmd_win32_ddt_res = vmd_win32.res(ddt=True)
        for i in range(vmd_win32_ddt_res.shape[1]):
            res_fe[f"vmd_win32_{i}_ddt"] = vmd_win32_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win32_ddt_lag_res = vmd_win32.res(ddt=True, lag=lg)
            for i in range(vmd_win32_ddt_lag_res.shape[1]):
                res_fe[f"vmd_win32_{i}_ddt_lag{lg}"] = vmd_win32_ddt_lag_res[:, i]

    if not lightweighted:
        vmd_win64 = VMD_NRBO(candles, 64, sequential=True)
        vmd_win64_res = vmd_win64.res()
        for i in range(vmd_win64_res.shape[1]):
            res_fe[f"vmd_win64_{i}"] = vmd_win64_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win64_lag_res = vmd_win64.res(lag=lg)
            for i in range(vmd_win64_lag_res.shape[1]):
                res_fe[f"vmd_win64_{i}_lag{lg}"] = vmd_win64_lag_res[:, i]
        vmd_win64_dt_res = vmd_win64.res(dt=True)
        for i in range(vmd_win64_dt_res.shape[1]):
            res_fe[f"vmd_win64_{i}_dt"] = vmd_win64_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win64_dt_lag_res = vmd_win64.res(dt=True, lag=lg)
            for i in range(vmd_win64_dt_lag_res.shape[1]):
                res_fe[f"vmd_win64_{i}_dt_lag{lg}"] = vmd_win64_dt_lag_res[:, i]
        vmd_win64_ddt_res = vmd_win64.res(ddt=True)
        for i in range(vmd_win64_ddt_res.shape[1]):
            res_fe[f"vmd_win64_{i}_ddt"] = vmd_win64_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win64_ddt_lag_res = vmd_win64.res(ddt=True, lag=lg)
            for i in range(vmd_win64_ddt_lag_res.shape[1]):
                res_fe[f"vmd_win64_{i}_ddt_lag{lg}"] = vmd_win64_ddt_lag_res[:, i]

    if not lightweighted:
        vmd_win128 = VMD_NRBO(candles, 128, sequential=True)
        vmd_win128_res = vmd_win128.res()
        for i in range(vmd_win128_res.shape[1]):
            res_fe[f"vmd_win128_{i}"] = vmd_win128_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win128_lag_res = vmd_win128.res(lag=lg)
            for i in range(vmd_win128_lag_res.shape[1]):
                res_fe[f"vmd_win128_{i}_lag{lg}"] = vmd_win128_lag_res[:, i]
        vmd_win128_dt_res = vmd_win128.res(dt=True)
        for i in range(vmd_win128_dt_res.shape[1]):
            res_fe[f"vmd_win128_{i}_dt"] = vmd_win128_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win128_dt_lag_res = vmd_win128.res(dt=True, lag=lg)
            for i in range(vmd_win128_dt_lag_res.shape[1]):
                res_fe[f"vmd_win128_{i}_dt_lag{lg}"] = vmd_win128_dt_lag_res[:, i]
        vmd_win128_ddt_res = vmd_win128.res(ddt=True)
        for i in range(vmd_win128_ddt_res.shape[1]):
            res_fe[f"vmd_win128_{i}_ddt"] = vmd_win128_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win128_ddt_lag_res = vmd_win128.res(ddt=True, lag=lg)
            for i in range(vmd_win128_ddt_lag_res.shape[1]):
                res_fe[f"vmd_win128_{i}_ddt_lag{lg}"] = vmd_win128_ddt_lag_res[:, i]

    if not lightweighted:
        vmd_win256 = VMD_NRBO(candles, 256, sequential=True)
        vmd_win256_res = vmd_win256.res()
        for i in range(vmd_win256_res.shape[1]):
            res_fe[f"vmd_win256_{i}"] = vmd_win256_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win256_lag_res = vmd_win256.res(lag=lg)
            for i in range(vmd_win256_lag_res.shape[1]):
                res_fe[f"vmd_win256_{i}_lag{lg}"] = vmd_win256_lag_res[:, i]
        vmd_win256_dt_res = vmd_win256.res(dt=True)
        for i in range(vmd_win256_dt_res.shape[1]):
            res_fe[f"vmd_win256_{i}_dt"] = vmd_win256_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win256_dt_lag_res = vmd_win256.res(dt=True, lag=lg)
            for i in range(vmd_win256_dt_lag_res.shape[1]):
                res_fe[f"vmd_win256_{i}_dt_lag{lg}"] = vmd_win256_dt_lag_res[:, i]
        vmd_win256_ddt_res = vmd_win256.res(ddt=True)
        for i in range(vmd_win256_ddt_res.shape[1]):
            res_fe[f"vmd_win256_{i}_ddt"] = vmd_win256_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win256_ddt_lag_res = vmd_win256.res(ddt=True, lag=lg)
            for i in range(vmd_win256_ddt_lag_res.shape[1]):
                res_fe[f"vmd_win256_{i}_ddt_lag{lg}"] = vmd_win256_ddt_lag_res[:, i]

    vmd_win512 = VMD_NRBO(candles, 512, sequential=True)
    vmd_win512_res = vmd_win512.res()
    for i in range(vmd_win512_res.shape[1]):
        res_fe[f"vmd_win512_{i}"] = vmd_win512_res[:, i]
    for lg in range(1, LAG_MAX):
        vmd_win512_lag_res = vmd_win512.res(lag=lg)
        for i in range(vmd_win512_lag_res.shape[1]):
            res_fe[f"vmd_win512_{i}_lag{lg}"] = vmd_win512_lag_res[:, i]
    vmd_win512_dt_res = vmd_win512.res(dt=True)
    for i in range(vmd_win512_dt_res.shape[1]):
        res_fe[f"vmd_win512_{i}_dt"] = vmd_win512_dt_res[:, i]
    for lg in range(1, LAG_MAX):
        vmd_win512_dt_lag_res = vmd_win512.res(dt=True, lag=lg)
        for i in range(vmd_win512_dt_lag_res.shape[1]):
            res_fe[f"vmd_win512_{i}_dt_lag{lg}"] = vmd_win512_dt_lag_res[:, i]
    vmd_win512_ddt_res = vmd_win512.res(ddt=True)
    for i in range(vmd_win512_ddt_res.shape[1]):
        res_fe[f"vmd_win512_{i}_ddt"] = vmd_win512_ddt_res[:, i]
    for lg in range(1, LAG_MAX):
        vmd_win512_ddt_lag_res = vmd_win512.res(ddt=True, lag=lg)
        for i in range(vmd_win512_ddt_lag_res.shape[1]):
            res_fe[f"vmd_win512_{i}_ddt_lag{lg}"] = vmd_win512_ddt_lag_res[:, i]

    if not lightweighted:
        vmd_win1024 = VMD_NRBO(candles, 1024, sequential=True)
        vmd_win1024_res = vmd_win1024.res()
        for i in range(vmd_win1024_res.shape[1]):
            res_fe[f"vmd_win1024_{i}"] = vmd_win1024_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win1024_lag_res = vmd_win1024.res(lag=lg)
            for i in range(vmd_win1024_lag_res.shape[1]):
                res_fe[f"vmd_win1024_{i}_lag{lg}"] = vmd_win1024_lag_res[:, i]
        vmd_win1024_dt_res = vmd_win1024.res(dt=True)
        for i in range(vmd_win1024_dt_res.shape[1]):
            res_fe[f"vmd_win1024_{i}_dt"] = vmd_win1024_dt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win1024_dt_lag_res = vmd_win1024.res(dt=True, lag=lg)
            for i in range(vmd_win1024_dt_lag_res.shape[1]):
                res_fe[f"vmd_win1024_{i}_dt_lag{lg}"] = vmd_win1024_dt_lag_res[:, i]
        vmd_win1024_ddt_res = vmd_win1024.res(ddt=True)
        for i in range(vmd_win1024_ddt_res.shape[1]):
            res_fe[f"vmd_win1024_{i}_ddt"] = vmd_win1024_ddt_res[:, i]
        for lg in range(1, LAG_MAX):
            vmd_win1024_ddt_lag_res = vmd_win1024.res(ddt=True, lag=lg)
            for i in range(vmd_win1024_ddt_lag_res.shape[1]):
                res_fe[f"vmd_win1024_{i}_ddt_lag{lg}"] = vmd_win1024_ddt_lag_res[:, i]

    # Voss Filter
    voss_filter_ = ta.voss(candles, sequential=True)
    voss_ = voss_filter_.voss
    filt_ = voss_filter_.filt
    res_fe["voss"] = voss_
    res_fe["voss_filt"] = filt_
    res_fe["voss_dt"] = dt(voss_)
    res_fe["voss_filt_dt"] = dt(filt_)
    res_fe["voss_ddt"] = ddt(voss_)
    res_fe["voss_filt_ddt"] = ddt(filt_)
    for lg in range(1, LAG_MAX):
        res_fe[f"voss_lag{lg}"] = lag(voss_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"voss_filt_lag{lg}"] = lag(filt_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"voss_dt_lag{lg}"] = lag(res_fe["voss_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"voss_filt_dt_lag{lg}"] = lag(res_fe["voss_filt_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"voss_ddt_lag{lg}"] = lag(res_fe["voss_ddt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"voss_filt_ddt_lag{lg}"] = lag(res_fe["voss_filt_ddt"], lg)

    # vwap
    vwap_ = ta.vwap(candles, sequential=True)
    res_fe["vwap"] = vwap_
    res_fe["vwap_dt"] = dt(vwap_)
    res_fe["vwap_ddt"] = ddt(vwap_)
    for lg in range(1, LAG_MAX):
        res_fe[f"vwap_lag{lg}"] = lag(vwap_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"vwap_dt_lag{lg}"] = lag(res_fe["vwap_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"vwap_ddt_lag{lg}"] = lag(res_fe["vwap_ddt"], lg)

    # Williams' %R
    williams_r_ = ta.willr(candles, sequential=True)
    res_fe["williams_r"] = williams_r_
    res_fe["williams_r_dt"] = dt(williams_r_)
    res_fe["williams_r_ddt"] = ddt(williams_r_)
    for lg in range(1, LAG_MAX):
        res_fe[f"williams_r_lag{lg}"] = lag(williams_r_, lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"williams_r_dt_lag{lg}"] = lag(res_fe["williams_r_dt"], lg)
    for lg in range(1, LAG_MAX):
        res_fe[f"williams_r_ddt_lag{lg}"] = lag(res_fe["williams_r_ddt"], lg)

    if sequential:
        return res_fe
    else:
        return {k: v[-1:] for k, v in res_fe.items()}