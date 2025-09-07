import jesse.indicators as ta
import numpy as np
from jesse.indicators import adx, aroon
from jesse.indicators.aroon import AROON

from src.features.simple_feature_calculator import (
    feature,
    class_feature,
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
    sample_entropy_indicator,
    approximate_entropy_indicator,
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
)


@feature(name="adx_7", description="ADX with period 7")
def adx_7_feature(candles: np.ndarray, sequential: bool = True):
    """ADX周期7"""
    return adx(candles, period=7, sequential=sequential)


@feature(name="adx_14", description="ADX with period 14")
def adx_14_feature(candles: np.ndarray, sequential: bool = True):
    """ADX周期14"""
    return adx(candles, period=14, sequential=sequential)


@feature(name="aroon_diff", description="Aroon Difference")
def aroon_diff_feature(candles: np.ndarray, sequential: bool = True):
    """Aroon差值"""
    aroon_: AROON = aroon(candles, sequential=sequential)
    return aroon_.up - aroon_.down


@feature(name="ac", returns_multiple=True, description="Autocorrelation")
def autocorrelation_feature(candles: np.ndarray, sequential: bool = True):
    """自相关"""
    return autocorrelation(candles, sequential=sequential)


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
    return pwr  # 返回功率谱


@feature(name="acr", description="Autocorrelation Reversals")
def acr_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自相关反转"""
    return autocorrelation_reversals(candles, sequential=sequential)


@feature(name="adaptive_bp", description="Adaptive Bandpass Filter")
def adaptive_bp_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自适应带通滤波器"""
    bp, bp_lead, _ = adaptive_bandpass(candles, sequential=sequential)
    return bp


@feature(name="adaptive_bp_lead", description="Adaptive Bandpass Lead")
def adaptive_bp_lead_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """自适应带通滤波器领先值"""
    bp, bp_lead, _ = adaptive_bandpass(candles, sequential=sequential)
    return bp_lead


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


@feature(name="bandpass", description="Bandpass Filter")
def bandpass_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """带通滤波器"""
    bandpass_tuple = ta.bandpass(candles, sequential=sequential)
    return bandpass_tuple.bp_normalized


@feature(name="highpass_bp", description="Highpass Bandpass Trigger")
def highpass_bp_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """高通带通触发器"""
    bandpass_tuple = ta.bandpass(candles, sequential=sequential)
    return bandpass_tuple.trigger


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


@feature(name="comb_spectrum_dom_cycle", description="Comb Spectrum Dominant Cycle")
def comb_spectrum_dom_cycle_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """梳状谱主导周期"""
    dom_cycle, pwr = comb_spectrum(candles, sequential=sequential)
    return dom_cycle


@feature(name="comb_spectrum_pwr", returns_multiple=True, description="Comb Spectrum")
def comb_spectrum_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """梳状谱"""
    dom_cycle, pwr = comb_spectrum(candles, sequential=sequential)
    return pwr


@feature(name="conv", returns_multiple=True, description="Ehlers Convolution")
def conv_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Ehlers卷积"""
    _, _, conv = ehlers_convolution(candles, sequential=sequential)
    return conv


@feature(name="dft_dom_cycle", description="DFT Dominant Cycle")
def dft_dom_cycle_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """DFT主导周期"""
    dom_cycle, spectrum = dft(candles, sequential=sequential)
    return dom_cycle


@feature(name="dft_spectrum", returns_multiple=True, description="DFT Spectrum")
def dft_spectrum_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """DFT频谱"""
    dom_cycle, spectrum = dft(candles, sequential=sequential)
    return spectrum


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


for wind in [32, 64, 128, 256, 512]:

    @feature(name=f"sample_entropy_win{wind}_spot", description="Sample Entropy")
    def sample_entropy_spot(
        candles: np.ndarray,
        sequential: bool = True,
    ):
        return sample_entropy_indicator(
            candles, period=wind, use_array_price=False, sequential=sequential
        )

    @feature(name=f"sample_entropy_win{wind}_array", description="Sample Entropy")
    def sample_entropy_array(
        candles: np.ndarray,
        sequential: bool = True,
    ):
        return sample_entropy_indicator(
            candles, period=wind, use_array_price=True, sequential=sequential
        )

    @feature(
        name=f"approximate_entropy_win{wind}_spot", description="Approximate Entropy"
    )
    def approximate_entropy_spot(
        candles: np.ndarray,
        sequential: bool = True,
    ):
        return approximate_entropy_indicator(
            candles, period=wind, use_array_price=False, sequential=sequential
        )

    @feature(
        name=f"approximate_entropy_win{wind}_array", description="Approximate Entropy"
    )
    def approximate_entropy_array(
        candles: np.ndarray,
        sequential: bool = True,
    ):
        return approximate_entropy_indicator(
            candles, period=wind, use_array_price=True, sequential=sequential
        )


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
    duration: int = 40,
    sequential: bool = True,
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
    duration: int = 20,
    sequential: bool = True,
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
    return fisher_ind.fisher


@feature(
    name="frac_diff_ffd",
    params={"diff_amt": 0.35},
    description="Fractional Differentiation FFD",
)
def frac_diff_ffd_feature(
    candles: np.ndarray,
    diff_amt: float = 0.35,
    sequential: bool = True,
):
    """分数阶差分FFD"""
    return frac_diff_ffd_candle(candles, diff_amt=diff_amt, sequential=sequential)


@feature(name="fti", description="Fishers Transform Indicator")
def fti_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """Fisher变换指标"""
    fti_: FTIResult = fti(candles, sequential=sequential)
    return fti_.fti


@feature(name="fti_best_period", description="FTI Best Period")
def fti_best_period_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """FTI最佳周期"""
    fti_: FTIResult = fti(candles, sequential=sequential)
    return fti_.best_period


@feature(name="forecast_oscillator", description="Forecast Oscillator")
def forecast_oscillator_feature(
    candles: np.ndarray,
    sequential: bool = True,
):
    """预测振荡器"""
    return ta.fosc(candles, sequential=sequential)


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
    period: int = 30,
    sequential: bool = True,
):
    """Hurst系数快速"""
    return hurst_coefficient(candles, period=period, sequential=sequential)


@feature(
    name="hurst_coef_slow", params={"period": 200}, description="Hurst Coefficient Slow"
)
def hurst_coef_slow_feature(
    candles: np.ndarray,
    period: int = 200,
    sequential: bool = True,
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
    return ta.natr(candles, sequential=sequential)


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
    return ta.pfe(candles, sequential=sequential)


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
    return ta.stc(candles, sequential=sequential)


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
    return swamicharts_rsi_


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
    return swamicharts_stochastic_


@feature(
    name="trendflex",
    description="Trendflex",
)
def trendflex_feature(candles: np.ndarray, sequential: bool = True):
    return ta.trendflex(candles, sequential=sequential)


@feature(
    name="voss",
    description="VOSS",
)
def voss_feature(candles: np.ndarray, sequential: bool = True):
    voss_filter_ = ta.voss(candles, sequential=sequential)
    return voss_filter_.voss


@feature(
    name="voss_filt",
    description="VOSS",
)
def voss_feature(candles: np.ndarray, sequential: bool = True):
    voss_filter_ = ta.voss(candles, sequential=sequential)
    return voss_filter_.filt


@feature(
    name="vwap",
    description="VWAP",
)
def vwap_feature(candles: np.ndarray, sequential: bool = True):
    return ta.vwap(candles, sequential=sequential)


@feature(
    name="williams_r",
    description="Williams R",
)
def williams_r_feature(candles: np.ndarray, sequential: bool = True):
    return ta.willr(candles, sequential=sequential)


# 注册类型特征
for wind in [32, 64, 128, 256, 512]:

    @class_feature(
        name=f"cwt_win{wind}",
        params={"window": wind},
        returns_multiple=True,
        description="CWT SWT Indicator",
    )
    class CWTFeature:
        """CWT SWT特征"""

        def __init__(
            self,
            candles: np.ndarray,
            window: int = wind,
            sequential: bool = False,
            **kwargs,
        ):
            self.indicator = CWT_SWT(candles, window, sequential=sequential)

        def res(self, **kwargs):
            return self.indicator.res(**kwargs)


for wind in [32, 64, 128, 256, 512]:

    @class_feature(
        name=f"vmd_win{wind}",
        params={"window": wind},
        returns_multiple=True,
        description="VMD NRBO Indicator",
    )
    class VMDFeature:
        """VMD NRBO特征"""

        def __init__(
            self,
            candles: np.ndarray,
            window: int = wind,
            sequential: bool = False,
            **kwargs,
        ):
            self.indicator = VMD_NRBO(candles, window, sequential=sequential)

        def res(self, **kwargs):
            return self.indicator.res(**kwargs)
