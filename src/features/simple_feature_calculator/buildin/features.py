import jesse.indicators as ta
import numpy as np
from jesse.indicators import adx, aroon
from jesse.indicators.aroon import AROON

from src.features.simple_feature_calculator import (
    feature,
)
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
    hurst_coefficient,
    iqr_ratio,
    FTIResult,
    amihud_lambda,
    bekker_parkinson_vol,
    corwin_schultz_estimator,
    hasbrouck_lambda,
    kyle_lambda,
    entropy_for_jesse,
    ma_difference,
    mod_rsi,
    mod_stochastic,
    norm_on_balance_volume,
    price_change_oscillator,
    price_variance_ratio,
    reactivity,
    roll_impact,
    roll_measure,
    roofing_filter,
    swamicharts_rsi,
    swamicharts_stochastic,
    bandpass,
    voss,
    bar_duration,
    bar_open,
    bar_high,
    bar_low,
    bar_close,
)
from src.indicators.prod.fti_rust import fti

# 小波与VMD特征
from . import _cwt_vmd_features  # noqa

# 多窗口熵特征 - 从独立模块导入
from . import _entropy_features  # 这会自动注册所有熵特征  # noqa

# 分数阶差分特征
from . import _np_fracdiff_features  # noqa

# WorldQuant 101 Alphas
from . import _wq_alpha  # noqa


@feature(
    name="bar_duration",
    description="fusion bar duration",
)
def bar_durations_feature(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return bar_duration(candles, sequential=sequential)


@feature(
    name="bar_open",
    description="open price",
)
def bar_open_feature(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return bar_open(candles, sequential=sequential)


@feature(
    name="bar_high",
    description="high price",
)
def bar_high_feature(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return bar_high(candles, sequential=sequential)


@feature(
    name="bar_low",
    description="low price",
)
def bar_low_feature(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return bar_low(candles, sequential=sequential)


@feature(
    name="bar_close",
)
def bar_close_feature(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return bar_close(candles, sequential=sequential)


@feature(name="adx_7", description="ADX with period 7")
def adx_7_feature(candles: np.ndarray, sequential: bool = True):
    """ADX周期7"""
    if sequential:
        return adx(candles, period=7, sequential=sequential)
    else:
        return np.array([adx(candles, period=7, sequential=sequential)])


@feature(name="adx_14", description="ADX with period 14")
def adx_14_feature(candles: np.ndarray, sequential: bool = True):
    """ADX周期14"""
    if sequential:
        return adx(candles, period=14, sequential=sequential)
    else:
        return np.array([adx(candles, period=14, sequential=sequential)])


@feature(name="aroon_diff", description="Aroon Difference")
def aroon_diff_feature(candles: np.ndarray, sequential: bool = True):
    """Aroon差值"""
    aroon_: AROON = aroon(candles, sequential=sequential)
    if sequential:
        return aroon_.up - aroon_.down
    else:
        return np.array([aroon_.up - aroon_.down])


@feature(name="ac", returns_multiple=True, description="Autocorrelation")
def autocorrelation_feature(candles: np.ndarray, sequential: bool = True):
    """自相关"""
    return autocorrelation(candles, sequential=sequential)  # 47列


@feature(name="acc_swing_index", description="Accumulated Swing Index")
def acc_swing_index_feature(candles: np.ndarray, sequential: bool = True):
    """累积摆动指数"""
    return accumulated_swing_index(candles, sequential=sequential)


@feature(
    name="acp_pwr", returns_multiple=True, description="Autocorrelation Periodogram"
)
def acp_feature(candles: np.ndarray, sequential: bool = True):
    """自相关周期图"""
    dom_cycle, pwr = autocorrelation_periodogram(candles, sequential=sequential)
    return pwr  # 返回功率谱,39列


@feature(name="acr", description="Autocorrelation Reversals")
def acr_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自相关反转"""
    return autocorrelation_reversals(candles, sequential=sequential)


@feature(
    name="adaptive_bp", returns_multiple=True, description="Adaptive Bandpass Filter"
)
def adaptive_bp_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自适应带通滤波器"""
    bp, bp_lead, _ = adaptive_bandpass(candles, sequential=sequential)
    # 一维数列先堆叠（candles长度 * 列），再转置
    res = np.array([bp, bp_lead]).T  # 2列
    if sequential:
        return res
    else:
        return res.reshape(1, -1)


@feature(name="adaptive_cci", description="Adaptive CCI")
def adaptive_cci_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自适应CCI"""
    return adaptive_cci(candles, sequential=sequential)


@feature(name="adaptive_rsi", description="Adaptive RSI")
def adaptive_rsi_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自适应RSI"""
    return adaptive_rsi(candles, sequential=sequential)


@feature(name="adaptive_stochastic", description="Adaptive Stochastic")
def adaptive_stochastic_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自适应随机指标"""
    return adaptive_stochastic(candles, sequential=sequential)


@feature(name="amihud_lambda", description="Amihud Lambda")
def amihud_lambda_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Amihud Lambda流动性指标"""
    return amihud_lambda(candles, sequential=sequential)


@feature(name="bandpass", returns_multiple=True, description="Bandpass Filter")
def bandpass_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """带通滤波器"""
    bandpass_tuple = bandpass(candles, sequential=sequential)
    res = np.array(
        [
            bandpass_tuple.bp_normalized,
            bandpass_tuple.trigger,
        ]
    ).T
    if sequential:
        return res
    else:
        return res.reshape(1, -1)


@feature(name="bekker_parkinson_vol", description="Bekker-Parkinson Volatility")
def bekker_parkinson_vol_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Bekker-Parkinson波动率"""
    return bekker_parkinson_vol(candles, sequential=sequential)


@feature(name="chaiken_money_flow", description="Chaiken Money Flow")
def chaiken_money_flow_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Chaiken资金流"""
    return chaiken_money_flow(candles, sequential=sequential)


@feature(name="change_variance_ratio", description="Change Variance Ratio")
def change_variance_ratio_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """变化方差比"""
    return change_variance_ratio(candles, sequential=sequential)


@feature(name="cmma", description="Compound Moving Average")
def cmma_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """复合移动平均"""
    return cmma(candles, sequential=sequential)


@feature(name="corwin_schultz_estimator", description="Corwin-Schultz Spread Estimator")
def corwin_schultz_estimator_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Corwin-Schultz价差估计器"""
    return corwin_schultz_estimator(candles, sequential=sequential)


@feature(
    name="comb_spectrum",
    returns_multiple=True,
    description="Comb Spectrum Dominant Cycle",
)
def comb_spectrum_dom_cycle_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """梳状谱主导周期"""
    dom_cycle, pwr = comb_spectrum(candles, sequential=sequential)
    return np.hstack([dom_cycle.reshape(-1, 1), pwr])  # 40列


@feature(name="conv", returns_multiple=True, description="Ehlers Convolution")
def conv_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Ehlers卷积"""
    _, _, conv = ehlers_convolution(candles, sequential=sequential)
    return conv  # 46列


@feature(name="dft", returns_multiple=True, description="DFT Dominant Cycle")
def dft_dom_cycle_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """DFT主导周期"""
    dom_cycle, spectrum = dft(candles, sequential=sequential)
    return np.hstack([dom_cycle.reshape(-1, 1), spectrum])  # 40列


@feature(name="dual_diff", description="Dual Differentiator")
def dual_diff_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """双重微分器"""
    return dual_differentiator(candles, sequential=sequential)


@feature(name="ehlers_early_onset_trend", description="Ehlers Early Onset Trend")
def ehlers_early_onset_trend_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Ehlers早期趋势"""
    return ehlers_early_onset_trend(candles, sequential=sequential)


@feature(name="entropy_for_jesse", description="Entropy for Jesse")
def entropy_for_jesse_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """熵指标"""
    return entropy_for_jesse(candles, sequential=sequential)


@feature(
    name="evenbetter_sinewave_long",
    params={"duration": 40},
    description="EvenBetter Sinewave Long",
)
def evenbetter_sinewave_long_feature(
    candles: np.ndarray,
    sequential: bool = True,
    duration: int = 40,
):
    """改进正弦波长期"""
    return evenbetter_sinewave(candles, duration=duration, sequential=sequential)


@feature(
    name="evenbetter_sinewave_short",
    params={"duration": 20},
    description="EvenBetter Sinewave Short",
)
def evenbetter_sinewave_short_feature(
    candles: np.ndarray,
    sequential: bool = True,
    duration: int = 20,
):
    """改进正弦波短期"""
    return evenbetter_sinewave(candles, duration=duration, sequential=sequential)


@feature(name="fisher", description="Fisher Transform")
def fisher_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Fisher变换"""
    fisher_ind = ta.fisher(candles, sequential=sequential)
    if sequential:
        return fisher_ind.fisher
    else:
        return np.array([fisher_ind.fisher])


# @feature(
#     name="frac_diff_ffd",
#     params={"diff_amt": 0.35},
#     description="Fractional Differentiation FFD",
# )
# def frac_diff_ffd_feature(
#     candles: np.ndarray,
#     sequential: bool = True,
#     diff_amt: float = 0.35,
# ):
#     """分数阶差分FFD"""
#     return frac_diff_ffd_candle(candles, diff_amt=diff_amt, sequential=sequential)


@feature(name="fti", returns_multiple=True, description="Fishers Transform Indicator")
def fti_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Fisher变换指标"""
    fti_: FTIResult = fti(candles, sequential=sequential)
    res = np.array(
        [
            fti_.fti,
            fti_.best_period,
        ]
    ).T  # 2列
    if sequential:
        return res
    else:
        return res.reshape(1, -1)


@feature(name="forecast_oscillator", description="Forecast Oscillator")
def forecast_oscillator_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """预测振荡器"""
    if sequential:
        return ta.fosc(candles, sequential=sequential)
    else:
        return np.array([ta.fosc(candles, sequential=sequential)])


@feature(name="hasbrouck_lambda", description="Hasbrouck Lambda")
def hasbrouck_lambda_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Hasbrouck Lambda"""
    return hasbrouck_lambda(candles, sequential=sequential)


@feature(name="homodyne", description="Homodyne Discriminator")
def homodyne_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """同步鉴别器"""
    return homodyne(candles, sequential=sequential)


@feature(
    name="hurst_coef_fast", params={"period": 30}, description="Hurst Coefficient Fast"
)
def hurst_coef_fast_feature(
    candles: np.ndarray,
    sequential: bool = True,
    period: int = 30,
):
    """Hurst系数快速"""
    return hurst_coefficient(candles, period=period, sequential=sequential)


@feature(
    name="hurst_coef_slow", params={"period": 200}, description="Hurst Coefficient Slow"
)
def hurst_coef_slow_feature(
    candles: np.ndarray,
    sequential: bool = True,
    period: int = 200,
):
    """Hurst系数慢速"""
    return hurst_coefficient(candles, period=period, sequential=sequential)


@feature(name="iqr_ratio", description="IQR Ratio")
def iqr_ratio_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """IQR比率"""
    return iqr_ratio(candles, sequential=sequential)


@feature(name="kyle_lambda", description="Kyle Lambda")
def kyle_lambda_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Kyle Lambda"""
    return kyle_lambda(candles, sequential=sequential)


@feature(
    name="ma_difference",
    description="MA Difference",
)
def ma_difference_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return ma_difference(candles, sequential=sequential)


@feature(
    name="mod_rsi",
    description="Modified RSI",
)
def mod_rsi_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return mod_rsi(candles, sequential=sequential)


@feature(
    name="mod_stochastic",
    description="Modified Stochastic",
)
def mod_stochastic_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return mod_stochastic(candles, sequential=sequential)


@feature(
    name="natr",
    description="NATR",
)
def natr_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    if sequential:
        return ta.natr(candles, sequential=sequential)
    else:
        return np.array([ta.natr(candles, sequential=sequential)])


@feature(
    name="norm_on_balance_volume",
    description="Normalized Balance Volume",
)
def norm_on_balance_volume_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return norm_on_balance_volume(candles, sequential=sequential)


@feature(
    name="phase_accumulation",
    description="Phase Accumulation",
)
def phase_accumulation_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return phase_accumulation(candles, sequential=sequential)


@feature(
    name="pfe",
    description="Polarized Fractal Efficiency",
)
def pfe_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    if sequential:
        return ta.pfe(candles, sequential=sequential)
    else:
        return np.array([ta.pfe(candles, sequential=sequential)])


@feature(
    name="price_change_oscillator",
    description="Price Change Oscillator",
)
def price_change_oscillator_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return price_change_oscillator(candles, sequential=sequential)


@feature(
    name="price_variance_ratio",
    description="Price Variance Ratio",
)
def price_variance_ratio_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return price_variance_ratio(candles, sequential=sequential)


@feature(
    name="reactivity",
    description="Reactivity",
)
def reactivity_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return reactivity(candles, sequential=sequential)


@feature(
    name="roll_impact",
    description="Roll Impact",
)
def roll_impact_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return roll_impact(candles, sequential=sequential)


@feature(
    name="roll_measure",
    description="Roll Measure",
)
def roll_measure_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return roll_measure(candles, sequential=sequential)


@feature(
    name="roofing_filter",
    description="Roofing Filter",
)
def roofing_filter_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    return roofing_filter(candles, sequential=sequential)


@feature(
    name="stc",
    description="Schaff Trend Cycle",
)
def stc_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    if sequential:
        return ta.stc(candles, sequential=sequential)
    else:
        return np.array([ta.stc(candles, sequential=sequential)])


@feature(
    name="swamicharts_rsi",
    description="Swamicharts RSI",
    returns_multiple=True,
)
def swamicharts_rsi_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    lookback, swamicharts_rsi_ = swamicharts_rsi(candles, sequential=sequential)
    return swamicharts_rsi_  # 44列


@feature(
    name="swamicharts_stochastic",
    description="Swamicharts Stochastic",
    returns_multiple=True,
)
def swamicharts_stochastic_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    lookback, swamicharts_stochastic_ = swamicharts_stochastic(
        candles, sequential=sequential
    )
    return swamicharts_stochastic_  # 44列


@feature(
    name="trendflex",
    description="Trendflex",
)
def trendflex_feature(candles: np.ndarray, sequential: bool = True):
    if sequential:
        return ta.trendflex(candles, sequential=sequential)
    else:
        return np.array([ta.trendflex(candles, sequential=sequential)])


@feature(
    name="voss",
    description="VOSS",
    returns_multiple=True,
)
def voss_feature(candles: np.ndarray, sequential: bool = True):
    voss_filter_ = voss(candles, sequential=sequential)
    res = np.array(
        [
            voss_filter_.voss,
            voss_filter_.filt,
        ]
    ).T
    if sequential:
        return res
    else:
        return res.reshape(1, -1)


@feature(
    name="vwap",
    description="VWAP",
)
def vwap_feature(candles: np.ndarray, sequential: bool = True):
    if sequential:
        return ta.vwap(candles, sequential=sequential)
    else:
        return np.array([ta.vwap(candles, sequential=sequential)])


@feature(
    name="williams_r",
    description="Williams R",
)
def williams_r_feature(candles: np.ndarray, sequential: bool = True):
    if sequential:
        return ta.willr(candles, sequential=sequential)
    else:
        return np.array([ta.willr(candles, sequential=sequential)])
