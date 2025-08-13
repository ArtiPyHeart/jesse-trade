"""
内置特征定义

这个模块包含了从原始 FeatureCalculator 迁移的特征
"""

import jesse.indicators as ta
import numpy as np
from jesse.indicators import adx, aroon
from jesse.indicators.aroon import AROON

from src.features.flexible_feature_calculator.registry import feature, class_feature
from src.indicators.dominant_cycle import (
    dual_differentiator,
    homodyne,
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
    frac_diff_ffd_candle,
    FTIResult,
    VMD_NRBO,
    amihud_lambda,
    bekker_parkinson_vol,
    corwin_schultz_estimator,
    hasbrouck_lambda,
    kyle_lambda,
    entropy_for_jesse,
    CWT_SWT,
)


# 注册函数型特征


@feature(name="acc_swing_index", description="Accumulated Swing Index")
def acc_swing_index_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """累积摆动指数"""
    return accumulated_swing_index(candles, sequential=sequential)


@feature(name="ac", returns_multiple=True, description="Autocorrelation")
def autocorrelation_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自相关"""
    return autocorrelation(candles, sequential=sequential)


@feature(name="acp", returns_multiple=True, description="Autocorrelation Periodogram")
def acp_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自相关周期图"""
    dom_cycle, pwr = autocorrelation_periodogram(candles, sequential=sequential)
    return pwr  # 返回功率谱


@feature(name="acr", description="Autocorrelation Reversals")
def acr_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自相关反转"""
    return autocorrelation_reversals(candles, sequential=sequential)


@feature(name="adx", params={"period": 14}, description="Average Directional Index")
def adx_feature(
    candles: np.ndarray, period: int = 14, sequential: bool = True, **kwargs
):
    """平均趋向指数"""
    return adx(candles, period=period, sequential=sequential)


@feature(name="adx_7", description="ADX with period 7")
def adx_7_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """ADX周期7"""
    return adx(candles, period=7, sequential=sequential)


@feature(name="adx_14", description="ADX with period 14")
def adx_14_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """ADX周期14"""
    return adx(candles, period=14, sequential=sequential)


@feature(name="adaptive_bp", description="Adaptive Bandpass Filter")
def adaptive_bp_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自适应带通滤波器"""
    bp, bp_lead, _ = adaptive_bandpass(candles, sequential=sequential)
    return bp


@feature(name="adaptive_bp_lead", description="Adaptive Bandpass Lead")
def adaptive_bp_lead_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自适应带通滤波器领先值"""
    bp, bp_lead, _ = adaptive_bandpass(candles, sequential=sequential)
    return bp_lead


@feature(name="adaptive_cci", description="Adaptive CCI")
def adaptive_cci_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自适应CCI"""
    return adaptive_cci(candles, sequential=sequential)


@feature(name="adaptive_rsi", description="Adaptive RSI")
def adaptive_rsi_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自适应RSI"""
    return adaptive_rsi(candles, sequential=sequential)


@feature(name="adaptive_stochastic", description="Adaptive Stochastic")
def adaptive_stochastic_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """自适应随机指标"""
    return adaptive_stochastic(candles, sequential=sequential)


@feature(name="amihud_lambda", description="Amihud Lambda")
def amihud_lambda_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Amihud Lambda流动性指标"""
    return amihud_lambda(candles, sequential=sequential)


@feature(name="aroon_diff", description="Aroon Difference")
def aroon_diff_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Aroon差值"""
    aroon_: AROON = aroon(candles, sequential=sequential)
    return aroon_.up - aroon_.down


@feature(name="bandpass", description="Bandpass Filter")
def bandpass_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """带通滤波器"""
    bandpass_tuple = ta.bandpass(candles, sequential=sequential)
    return bandpass_tuple.bp_normalized


@feature(name="highpass_bp", description="Highpass Bandpass Trigger")
def highpass_bp_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """高通带通触发器"""
    bandpass_tuple = ta.bandpass(candles, sequential=sequential)
    return bandpass_tuple.trigger


@feature(name="bekker_parkinson_vol", description="Bekker-Parkinson Volatility")
def bekker_parkinson_vol_feature(
    candles: np.ndarray, sequential: bool = True, **kwargs
):
    """Bekker-Parkinson波动率"""
    return bekker_parkinson_vol(candles, sequential=sequential)


@feature(name="chaiken_money_flow", description="Chaiken Money Flow")
def chaiken_money_flow_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Chaiken资金流"""
    return chaiken_money_flow(candles, sequential=sequential)


@feature(name="change_variance_ratio", description="Change Variance Ratio")
def change_variance_ratio_feature(
    candles: np.ndarray, sequential: bool = True, **kwargs
):
    """变化方差比"""
    return change_variance_ratio(candles, sequential=sequential)


@feature(name="cmma", description="Compound Moving Average")
def cmma_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """复合移动平均"""
    return cmma(candles, sequential=sequential)


@feature(name="corwin_schultz_estimator", description="Corwin-Schultz Spread Estimator")
def corwin_schultz_estimator_feature(
    candles: np.ndarray, sequential: bool = True, **kwargs
):
    """Corwin-Schultz价差估计器"""
    return corwin_schultz_estimator(candles, sequential=sequential)


@feature(name="comb_spectrum", returns_multiple=True, description="Comb Spectrum")
def comb_spectrum_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """梳状谱"""
    dom_cycle, pwr = comb_spectrum(candles, sequential=sequential)
    return pwr


@feature(name="comb_spectrum_dom_cycle", description="Comb Spectrum Dominant Cycle")
def comb_spectrum_dom_cycle_feature(
    candles: np.ndarray, sequential: bool = True, **kwargs
):
    """梳状谱主导周期"""
    dom_cycle, pwr = comb_spectrum(candles, sequential=sequential)
    return dom_cycle


@feature(name="conv", returns_multiple=True, description="Ehlers Convolution")
def conv_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Ehlers卷积"""
    _, _, conv = ehlers_convolution(candles, sequential=sequential)
    return conv


@feature(name="dft_spectrum", returns_multiple=True, description="DFT Spectrum")
def dft_spectrum_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """DFT频谱"""
    dom_cycle, spectrum = dft(candles, sequential=sequential)
    return spectrum


@feature(name="dft_dom_cycle", description="DFT Dominant Cycle")
def dft_dom_cycle_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """DFT主导周期"""
    dom_cycle, spectrum = dft(candles, sequential=sequential)
    return dom_cycle


@feature(name="dual_diff", description="Dual Differentiator")
def dual_diff_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """双重微分器"""
    return dual_differentiator(candles, sequential=sequential)


@feature(name="ehlers_early_onset_trend", description="Ehlers Early Onset Trend")
def ehlers_early_onset_trend_feature(
    candles: np.ndarray, sequential: bool = True, **kwargs
):
    """Ehlers早期趋势"""
    return ehlers_early_onset_trend(candles, sequential=sequential)


@feature(name="entropy_for_jesse", description="Entropy for Jesse")
def entropy_for_jesse_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """熵指标"""
    return entropy_for_jesse(candles, sequential=sequential)


@feature(
    name="evenbetter_sinewave_long",
    params={"duration": 40},
    description="EvenBetter Sinewave Long",
)
def evenbetter_sinewave_long_feature(
    candles: np.ndarray, duration: int = 40, sequential: bool = True, **kwargs
):
    """改进正弦波长期"""
    return evenbetter_sinewave(candles, duration=duration, sequential=sequential)


@feature(
    name="evenbetter_sinewave_short",
    params={"duration": 20},
    description="EvenBetter Sinewave Short",
)
def evenbetter_sinewave_short_feature(
    candles: np.ndarray, duration: int = 20, sequential: bool = True, **kwargs
):
    """改进正弦波短期"""
    return evenbetter_sinewave(candles, duration=duration, sequential=sequential)


@feature(name="fisher", description="Fisher Transform")
def fisher_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Fisher变换"""
    fisher_ind = ta.fisher(candles, sequential=sequential)
    return fisher_ind.fisher


@feature(
    name="frac_diff_ffd",
    params={"diff_amt": 0.35},
    description="Fractional Differentiation FFD",
)
def frac_diff_ffd_feature(
    candles: np.ndarray, diff_amt: float = 0.35, sequential: bool = True, **kwargs
):
    """分数阶差分FFD"""
    return frac_diff_ffd_candle(candles, diff_amt=diff_amt, sequential=sequential)


@feature(name="fti", description="Fishers Transform Indicator")
def fti_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Fisher变换指标"""
    fti_: FTIResult = fti(candles, sequential=sequential)
    return fti_.fti


@feature(name="fti_best_period", description="FTI Best Period")
def fti_best_period_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """FTI最佳周期"""
    fti_: FTIResult = fti(candles, sequential=sequential)
    return fti_.best_period


@feature(name="forecast_oscillator", description="Forecast Oscillator")
def forecast_oscillator_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """预测振荡器"""
    return ta.fosc(candles, sequential=sequential)


@feature(name="hasbrouck_lambda", description="Hasbrouck Lambda")
def hasbrouck_lambda_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Hasbrouck Lambda"""
    return hasbrouck_lambda(candles, sequential=sequential)


@feature(name="homodyne", description="Homodyne Discriminator")
def homodyne_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """同步鉴别器"""
    return homodyne(candles, sequential=sequential)


@feature(
    name="hurst_coef_fast", params={"period": 30}, description="Hurst Coefficient Fast"
)
def hurst_coef_fast_feature(
    candles: np.ndarray, period: int = 30, sequential: bool = True, **kwargs
):
    """Hurst系数快速"""
    return hurst_coefficient(candles, period=period, sequential=sequential)


@feature(name="hurst_coef_30", description="Hurst Coefficient 30")
def hurst_coef_30_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Hurst系数30"""
    return hurst_coefficient(candles, period=30, sequential=sequential)


@feature(
    name="hurst_coef_slow", params={"period": 200}, description="Hurst Coefficient Slow"
)
def hurst_coef_slow_feature(
    candles: np.ndarray, period: int = 200, sequential: bool = True, **kwargs
):
    """Hurst系数慢速"""
    return hurst_coefficient(candles, period=period, sequential=sequential)


@feature(name="hurst_coef_200", description="Hurst Coefficient 200")  
def hurst_coef_200_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Hurst系数200"""
    return hurst_coefficient(candles, period=200, sequential=sequential)


@feature(name="iqr_ratio", description="IQR Ratio")
def iqr_ratio_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """IQR比率"""
    return iqr_ratio(candles, sequential=sequential)


@feature(name="kyle_lambda", description="Kyle Lambda")
def kyle_lambda_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
    """Kyle Lambda"""
    return kyle_lambda(candles, sequential=sequential)


# 注册类型特征


@class_feature(
    name="vmd",
    params={"window": 32},
    returns_multiple=True,
    description="VMD NRBO Indicator",
)
class VMDFeature:
    """VMD NRBO特征"""

    def __init__(
        self, candles: np.ndarray, window: int = 32, sequential: bool = False, **kwargs
    ):
        self.indicator = VMD_NRBO(candles, window, sequential=sequential)

    def res(self, **kwargs):
        return self.indicator.res(**kwargs)


@class_feature(
    name="cwt",
    params={"window": 32},
    returns_multiple=True,
    description="CWT SWT Indicator",
)
class CWTFeature:
    """CWT SWT特征"""

    def __init__(
        self, candles: np.ndarray, window: int = 32, sequential: bool = False, **kwargs
    ):
        self.indicator = CWT_SWT(candles, window, sequential=sequential)

    def res(self, **kwargs):
        return self.indicator.res(**kwargs)
