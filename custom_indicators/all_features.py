import re

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

LAG_MAX = 20


class FeatureCalculator:
    """
    实例化后用于计算所有特征的类
    1. 每次load后需初始化所有状态；
    2. 计算时，首先从cache中获取，若不存在，则通过字符串形式调用类中的函数；
    3. 若特征名称中包含数字，则需要通过匹配的方式对应特征名称和参数，并调用对应的函数。
    """

    def __init__(self):
        self.candles = None
        self.sequential = False
        self.cache = {}

    def load(self, candles: np.array, sequential: bool = False):
        self.candles = helpers.slice_candles(candles, sequential)
        self.sequential = sequential
        # 每次load后，缓存清空
        self.cache = {}

    def get(self, features: list[str]):
        res_fe = {}
        for feat in features:
            if feat in self.cache:
                res_fe[feat] = self.cache[feat]
            else:
                # 通过字符串形式调用类中的函数
                if hasattr(self, feat):
                    getattr(self, feat)()
                    res_fe[feat] = self.cache[feat]
                else:
                    # 更复杂的特征名称解析
                    # 按步骤拆解特征名称
                    try:
                        # 初始化参数
                        kwargs = {}
                        feature_parts = feat.split("_")
                        remain_parts = feature_parts.copy()

                        # 步骤1: 匹配方法(dt,ddt)和lag
                        # 检查最后部分是否是lag
                        if len(remain_parts) > 0 and remain_parts[-1].startswith("lag"):
                            lag_pattern = r"^lag(\d+)$"
                            lag_match = re.match(lag_pattern, remain_parts[-1])
                            if lag_match:
                                kwargs["lag"] = int(lag_match.group(1))
                                remain_parts.pop()

                        # 检查是否包含dt或ddt部分
                        if len(remain_parts) > 0 and remain_parts[-1] in ["dt", "ddt"]:
                            method = remain_parts[-1]
                            kwargs[method] = True
                            remain_parts.pop()

                        # 步骤2: 检查最后部分是否是数字(作为index)
                        if len(remain_parts) > 0 and remain_parts[-1].isdigit():
                            kwargs["index"] = int(remain_parts[-1])
                            remain_parts.pop()

                        # 步骤3: 剩余部分组合成基础特征名称
                        base_name = "_".join(remain_parts)

                        # 检查是否有对应的特征计算方法
                        if hasattr(self, base_name):
                            # 调用方法并传入解析出的参数
                            getattr(self, base_name)(**kwargs)
                            if feat in self.cache:
                                res_fe[feat] = self.cache[feat]
                            else:
                                # 特征计算方法没有正确设置缓存
                                raise ValueError(
                                    f"特征计算方法 '{base_name}' 未正确设置缓存值 '{feat}'"
                                )
                        else:
                            raise ValueError(
                                f"特征 '{feat}' 的基础计算方法 '{base_name}' 在 FeatureCalculator 类中未定义"
                            )
                    except Exception as e:
                        raise ValueError(f"特征 '{feat}' 解析失败: {str(e)}")
        if self.sequential:
            return res_fe
        else:
            return {k: v[-1:] for k, v in res_fe.items()}

    def _process_transformations(self, base_key: str, base_value: np.array, **kwargs):
        """处理特征的变换操作：dt, ddt, lag等"""
        feature_name = base_key
        feature_value = base_value
        # 如果需要dt变换
        if "dt" in kwargs:
            feature_name = f"{feature_name}_dt"
            if feature_name not in self.cache:
                dt_value = dt(feature_value)
                self.cache[feature_name] = dt_value
            else:
                dt_value = self.cache[feature_name]

            feature_value = dt_value

        # 如果需要ddt变换
        if "ddt" in kwargs:
            feature_name = f"{feature_name}_ddt"
            if feature_name not in self.cache:
                ddt_value = ddt(feature_value)
                self.cache[feature_name] = ddt_value
            else:
                ddt_value = self.cache[feature_name]

            feature_value = ddt_value

        # 如果需要lag变换
        if "lag" in kwargs:
            lag_value = kwargs["lag"]
            feature_name = f"{feature_name}_lag{lag_value}"
            if feature_name not in self.cache:
                lag_value = lag(feature_value, lag_value)
                self.cache[feature_name] = lag_value
            else:
                lag_value = self.cache[feature_name]

            feature_value = lag_value

    def ac(self, **kwargs):
        index = kwargs["index"]
        if f"ac_{index}" not in self.cache:
            # 如果找不到任意一个ac_index，则计算所有ac_index
            auto_corr = autocorrelation(self.candles, sequential=True)
            for i in range(auto_corr.shape[1]):
                self.cache[f"ac_{i}"] = auto_corr[:, i]

    def acc_swing_index(self, **kwargs):
        if "acc_swing_index" not in self.cache:
            self.cache["acc_swing_index"] = accumulated_swing_index(
                self.candles, sequential=True
            )

        self._process_transformations(
            "acc_swing_index", self.cache["acc_swing_index"], **kwargs
        )

    def acp_pwr(self, **kwargs):
        index = kwargs["index"]
        if f"acp_pwr_{index}" not in self.cache:
            acp_dom_cycle, pwr = autocorrelation_periodogram(
                self.candles, sequential=True
            )
            for i in range(pwr.shape[1]):
                self.cache[f"acp_pwr_{i}"] = pwr[:, i]

    def acr(self, **kwargs):
        if "acr" not in self.cache:
            acr = autocorrelation_reversals(self.candles, sequential=True)
            self.cache["acr"] = acr

    def adaptive_bp(self, **kwargs):
        if "adaptive_bp" not in self.cache or "adaptive_bp_lead" not in self.cache:
            adaptive_bp, adaptive_bp_lead, _ = adaptive_bandpass(
                self.candles, sequential=True
            )
            self.cache["adaptive_bp"] = adaptive_bp
            self.cache["adaptive_bp_lead"] = adaptive_bp_lead

        self._process_transformations(
            "adaptive_bp", self.cache["adaptive_bp"], **kwargs
        )

    def adaptive_bp_lead(self, **kwargs):
        if "adaptive_bp" not in self.cache or "adaptive_bp_lead" not in self.cache:
            adaptive_bp, adaptive_bp_lead, _ = adaptive_bandpass(
                self.candles, sequential=True
            )
            self.cache["adaptive_bp"] = adaptive_bp
            self.cache["adaptive_bp_lead"] = adaptive_bp_lead

        self._process_transformations(
            "adaptive_bp_lead", self.cache["adaptive_bp_lead"], **kwargs
        )

    def adaptive_cci(self, **kwargs):
        if "adaptive_cci" not in self.cache:
            adaptive_cci_ = adaptive_cci(self.candles, sequential=True)
            self.cache["adaptive_cci"] = adaptive_cci_

        self._process_transformations(
            "adaptive_cci", self.cache["adaptive_cci"], **kwargs
        )

    def adaptive_rsi(self, **kwargs):
        if "adaptive_rsi" not in self.cache:
            adaptive_rsi_ = adaptive_rsi(self.candles, sequential=True)
            self.cache["adaptive_rsi"] = adaptive_rsi_

        self._process_transformations(
            "adaptive_rsi", self.cache["adaptive_rsi"], **kwargs
        )

    def adaptive_stochastic(self, **kwargs):
        if "adaptive_stochastic" not in self.cache:
            adaptive_stochastic_ = adaptive_stochastic(self.candles, sequential=True)
            self.cache["adaptive_stochastic"] = adaptive_stochastic_

        self._process_transformations(
            "adaptive_stochastic", self.cache["adaptive_stochastic"], **kwargs
        )

    def bandpass(self, **kwargs):
        if "bandpass" not in self.cache or "highpass_bp" not in self.cache:
            bandpass_tuple = ta.bandpass(self.candles, sequential=True)
            self.cache["bandpass"] = bandpass_tuple.bp_normalized
            self.cache["highpass_bp"] = bandpass_tuple.trigger

        self._process_transformations("bandpass", self.cache["bandpass"], **kwargs)

    def highpass_bp(self, **kwargs):
        if "bandpass" not in self.cache or "highpass_bp" not in self.cache:
            bandpass_tuple = ta.bandpass(self.candles, sequential=True)
            self.cache["bandpass"] = bandpass_tuple.bp_normalized
            self.cache["highpass_bp"] = bandpass_tuple.trigger

        self._process_transformations(
            "highpass_bp", self.cache["highpass_bp"], **kwargs
        )

    def comb_spectrum_dom_cycle(self, **kwargs):
        if "comb_spectrum_dom_cycle" not in self.cache:
            comb_spectrum_dom_cycle, pwr = comb_spectrum(self.candles, sequential=True)
            self.cache["comb_spectrum_dom_cycle"] = comb_spectrum_dom_cycle
            for i in range(pwr.shape[1]):
                self.cache[f"comb_spectrum_pwr_{i}"] = pwr[:, i]

        self._process_transformations(
            "comb_spectrum_dom_cycle", self.cache["comb_spectrum_dom_cycle"], **kwargs
        )

    def comb_spectrum_pwr(self, **kwargs):
        index = kwargs["index"]
        if f"comb_spectrum_pwr_{index}" not in self.cache:
            comb_spectrum_dom_cycle, pwr = comb_spectrum(self.candles, sequential=True)
            self.cache["comb_spectrum_dom_cycle"] = comb_spectrum_dom_cycle
            for i in range(pwr.shape[1]):
                self.cache[f"comb_spectrum_pwr_{i}"] = pwr[:, i]

    def conv(self, **kwargs):
        index = kwargs["index"]
        if f"conv_{index}" not in self.cache:
            _, _, conv = ehlers_convolution(self.candles, sequential=True)
            for i in range(conv.shape[1]):
                self.cache[f"conv_{i}"] = conv[:, i]

    def dft_dom_cycle(self, **kwargs):
        if "dft_dom_cycle" not in self.cache:
            dft_dom_cycle, spectrum = dft(self.candles, sequential=True)
            self.cache["dft_dom_cycle"] = dft_dom_cycle
            for i in range(spectrum.shape[1]):
                self.cache[f"dft_spectrum_{i}"] = spectrum[:, i]

        self._process_transformations(
            "dft_dom_cycle", self.cache["dft_dom_cycle"], **kwargs
        )

    def dft_spectrum(self, **kwargs):
        index = kwargs["index"]
        if f"dft_spectrum_{index}" not in self.cache:
            dft_dom_cycle, spectrum = dft(self.candles, sequential=True)
            self.cache["dft_dom_cycle"] = dft_dom_cycle
            for i in range(spectrum.shape[1]):
                self.cache[f"dft_spectrum_{i}"] = spectrum[:, i]

    def dual_diff(self, **kwargs):
        if "dual_diff" not in self.cache:
            dual_diff = dual_differentiator(self.candles, sequential=True)
            self.cache["dual_diff"] = dual_diff

        self._process_transformations("dual_diff", self.cache["dual_diff"], **kwargs)

    def ehlers_early_onset_trend(self, **kwargs):
        if "ehlers_early_onset_trend" not in self.cache:
            ehlers_early_onset_trend_ = ehlers_early_onset_trend(
                self.candles, sequential=True
            )
            self.cache["ehlers_early_onset_trend"] = ehlers_early_onset_trend_

        self._process_transformations(
            "ehlers_early_onset_trend", self.cache["ehlers_early_onset_trend"], **kwargs
        )

    def evenbetter_sinewave_long(self, **kwargs):
        if "evenbetter_sinewave_long" not in self.cache:
            eb_sw_long = evenbetter_sinewave(self.candles, duration=40, sequential=True)
            self.cache["evenbetter_sinewave_long"] = eb_sw_long

        self._process_transformations(
            "evenbetter_sinewave_long", self.cache["evenbetter_sinewave_long"], **kwargs
        )

    def evenbetter_sinewave_short(self, **kwargs):
        if "evenbetter_sinewave_short" not in self.cache:
            eb_sw_short = evenbetter_sinewave(
                self.candles, duration=20, sequential=True
            )
            self.cache["evenbetter_sinewave_short"] = eb_sw_short

        self._process_transformations(
            "evenbetter_sinewave_short",
            self.cache["evenbetter_sinewave_short"],
            **kwargs,
        )

    def fisher(self, **kwargs):
        if "fisher" not in self.cache:
            fisher_ind = ta.fisher(self.candles, sequential=True)
            self.cache["fisher"] = fisher_ind.fisher

        self._process_transformations("fisher", self.cache["fisher"], **kwargs)

    def forecast_oscillator(self, **kwargs):
        if "forecast_oscillator" not in self.cache:
            forecast_oscillator = ta.fosc(self.candles, sequential=True)
            self.cache["forecast_oscillator"] = forecast_oscillator

        self._process_transformations(
            "forecast_oscillator", self.cache["forecast_oscillator"], **kwargs
        )

    def homodyne(self, **kwargs):
        if "homodyne" not in self.cache:
            homodyne_ = homodyne(self.candles, sequential=True)
            self.cache["homodyne"] = homodyne_

        self._process_transformations("homodyne", self.cache["homodyne"], **kwargs)

    def hurst_coef_fast(self, **kwargs):
        if "hurst_coef_fast" not in self.cache:
            hurst_coef_fast = hurst_coefficient(
                self.candles, period=30, sequential=True
            )
            self.cache["hurst_coef_fast"] = hurst_coef_fast

        self._process_transformations(
            "hurst_coef_fast", self.cache["hurst_coef_fast"], **kwargs
        )

    def hurst_coef_slow(self, **kwargs):
        if "hurst_coef_slow" not in self.cache:
            hurst_coef_slow = hurst_coefficient(
                self.candles, period=200, sequential=True
            )
            self.cache["hurst_coef_slow"] = hurst_coef_slow

        self._process_transformations(
            "hurst_coef_slow", self.cache["hurst_coef_slow"], **kwargs
        )

    def mod_rsi(self, **kwargs):
        if "mod_rsi" not in self.cache:
            mod_rsi_ = mod_rsi(self.candles, sequential=True)
            self.cache["mod_rsi"] = mod_rsi_

        self._process_transformations("mod_rsi", self.cache["mod_rsi"], **kwargs)

    def mod_stochastic(self, **kwargs):
        if "mod_stochastic" not in self.cache:
            mod_stochastic_ = mod_stochastic(
                self.candles, roofing_filter=True, sequential=True
            )
            self.cache["mod_stochastic"] = mod_stochastic_

        self._process_transformations(
            "mod_stochastic", self.cache["mod_stochastic"], **kwargs
        )

    def natr(self, **kwargs):
        if "natr" not in self.cache:
            natr_ = ta.natr(self.candles, sequential=True)
            self.cache["natr"] = natr_

        self._process_transformations("natr", self.cache["natr"], **kwargs)

    def phase_accumulation(self, **kwargs):
        if "phase_accumulation" not in self.cache:
            phase_accumulation_ = phase_accumulation(self.candles, sequential=True)
            self.cache["phase_accumulation"] = phase_accumulation_

        self._process_transformations(
            "phase_accumulation", self.cache["phase_accumulation"], **kwargs
        )

    def pfe(self, **kwargs):
        if "pfe" not in self.cache:
            pfe_ = ta.pfe(self.candles, sequential=True)
            self.cache["pfe"] = pfe_

        self._process_transformations("pfe", self.cache["pfe"], **kwargs)

    def roofing_filter(self, **kwargs):
        if "roofing_filter" not in self.cache:
            roofing_filter_ = roofing_filter(self.candles, sequential=True)
            self.cache["roofing_filter"] = roofing_filter_

        self._process_transformations(
            "roofing_filter", self.cache["roofing_filter"], **kwargs
        )

    def stc(self, **kwargs):
        if "stc" not in self.cache:
            stc_ = ta.stc(self.candles, sequential=True)
            self.cache["stc"] = stc_

        self._process_transformations("stc", self.cache["stc"], **kwargs)

    def swamicharts_rsi(self, **kwargs):
        index = kwargs["index"]
        if f"swamicharts_rsi_{index}" not in self.cache:
            lookback, swamicharts_rsi_ = swamicharts_rsi(self.candles, sequential=True)
            for i in range(swamicharts_rsi_.shape[1]):
                self.cache[f"swamicharts_rsi_{i}"] = swamicharts_rsi_[:, i]

    def swamicharts_stochastic(self, **kwargs):
        index = kwargs["index"]
        if f"swamicharts_stochastic_{index}" not in self.cache:
            lookback, swamicharts_stochastic_ = swamicharts_stochastic(
                self.candles, sequential=True
            )
            for i in range(swamicharts_stochastic_.shape[1]):
                self.cache[f"swamicharts_stochastic_{i}"] = swamicharts_stochastic_[
                    :, i
                ]

    def td_sequential_buy(self, **kwargs):
        if "td_sequential_buy" not in self.cache:
            td_sequential_buy, td_sequential_sell = td_sequential(
                self.candles, sequential=True
            )
            self.cache["td_sequential_buy"] = td_sequential_buy
            self.cache["td_sequential_sell"] = td_sequential_sell

    def td_sequential_sell(self, **kwargs):
        if "td_sequential_sell" not in self.cache:
            td_sequential_buy, td_sequential_sell = td_sequential(
                self.candles, sequential=True
            )
            self.cache["td_sequential_buy"] = td_sequential_buy
            self.cache["td_sequential_sell"] = td_sequential_sell

    def td_sequential_buy_aggressive(self, **kwargs):
        if "td_sequential_buy_aggressive" not in self.cache:
            td_sequential_buy_aggressive, td_sequential_sell_aggressive = td_sequential(
                self.candles, sequential=True, aggressive=True
            )
            self.cache["td_sequential_buy_aggressive"] = td_sequential_buy_aggressive
            self.cache["td_sequential_sell_aggressive"] = td_sequential_sell_aggressive

    def td_sequential_sell_aggressive(self, **kwargs):
        if "td_sequential_sell_aggressive" not in self.cache:
            td_sequential_buy_aggressive, td_sequential_sell_aggressive = td_sequential(
                self.candles, sequential=True, aggressive=True
            )
            self.cache["td_sequential_buy_aggressive"] = td_sequential_buy_aggressive
            self.cache["td_sequential_sell_aggressive"] = td_sequential_sell_aggressive

    def td_sequential_buy_stealth(self, **kwargs):
        if "td_sequential_buy_stealth" not in self.cache:
            td_sequential_buy_stealth, td_sequential_sell_stealth = td_sequential(
                self.candles, sequential=True, stealth_actions=True
            )
            self.cache["td_sequential_buy_stealth"] = td_sequential_buy_stealth
            self.cache["td_sequential_sell_stealth"] = td_sequential_sell_stealth

    def td_sequential_sell_stealth(self, **kwargs):
        if "td_sequential_sell_stealth" not in self.cache:
            td_sequential_buy_stealth, td_sequential_sell_stealth = td_sequential(
                self.candles, sequential=True, stealth_actions=True
            )
            self.cache["td_sequential_buy_stealth"] = td_sequential_buy_stealth
            self.cache["td_sequential_sell_stealth"] = td_sequential_sell_stealth

    def td_sequential_buy_aggressive_stealth(self, **kwargs):
        if "td_sequential_buy_aggressive_stealth" not in self.cache:
            (
                td_sequential_buy_aggressive_stealth,
                td_sequential_sell_aggressive_stealth,
            ) = td_sequential(
                self.candles, sequential=True, aggressive=True, stealth_actions=True
            )
            self.cache["td_sequential_buy_aggressive_stealth"] = (
                td_sequential_buy_aggressive_stealth
            )
            self.cache["td_sequential_sell_aggressive_stealth"] = (
                td_sequential_sell_aggressive_stealth
            )

    def td_sequential_sell_aggressive_stealth(self, **kwargs):
        if "td_sequential_sell_aggressive_stealth" not in self.cache:
            (
                td_sequential_buy_aggressive_stealth,
                td_sequential_sell_aggressive_stealth,
            ) = td_sequential(
                self.candles, sequential=True, aggressive=True, stealth_actions=True
            )
            self.cache["td_sequential_buy_aggressive_stealth"] = (
                td_sequential_buy_aggressive_stealth
            )
            self.cache["td_sequential_sell_aggressive_stealth"] = (
                td_sequential_sell_aggressive_stealth
            )

    def trendflex(self, **kwargs):
        if "trendflex" not in self.cache:
            trendflex_ = ta.trendflex(self.candles, sequential=True)
            self.cache["trendflex"] = trendflex_

        self._process_transformations("trendflex", self.cache["trendflex"], **kwargs)

    def voss(self, **kwargs):
        if "voss" not in self.cache:
            voss_filter_ = ta.voss(self.candles, sequential=True)
            self.cache["voss"] = voss_filter_.voss
            self.cache["voss_filt"] = voss_filter_.filt
        self._process_transformations("voss", self.cache["voss"], **kwargs)

    def voss_filt(self, **kwargs):
        if "voss_filt" not in self.cache:
            voss_filter_ = ta.voss(self.candles, sequential=True)
            self.cache["voss"] = voss_filter_.voss
            self.cache["voss_filt"] = voss_filter_.filt
        self._process_transformations("voss_filt", self.cache["voss_filt"], **kwargs)

    def vwap(self, **kwargs):
        if "vwap" not in self.cache:
            vwap_ = ta.vwap(self.candles, sequential=True)
            self.cache["vwap"] = vwap_

        self._process_transformations("vwap", self.cache["vwap"], **kwargs)

    def williams_r(self, **kwargs):
        if "williams_r" not in self.cache:
            williams_r_ = ta.willr(self.candles, sequential=True)
            self.cache["williams_r"] = williams_r_

        self._process_transformations("williams_r", self.cache["williams_r"], **kwargs)


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
