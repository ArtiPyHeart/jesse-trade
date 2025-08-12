import re

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


class FeatureCalculator:
    """
    实例化后用于计算所有特征的类
    1. 每次load后需初始化所有状态；
    2. 计算时，首先从cache中获取，若不存在，则通过字符串形式调用类中的函数；
    3. 若特征名称中包含数字，则需要通过匹配的方式对应特征名称和参数，并调用对应的函数。
    """

    def __init__(self):
        self.candles: np.ndarray = None
        self.sequential = False
        self.cache = {}
        self.cache_class_indicator = {}

    def load(self, candles: np.ndarray, sequential: bool = False):
        self.candles = helpers.slice_candles(candles, sequential)
        self.sequential = sequential
        # 每次load后，缓存清空
        self.cache = {}
        self.cache_class_indicator = {}

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
            return {
                k: v[-1:] if isinstance(v, (np.ndarray, list)) else np.array([v])
                for k, v in res_fe.items()
            }

    def _process_transformations(self, base_key: str, base_value: np.ndarray, **kwargs):
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

    def adx(self, **kwargs):
        index = kwargs["index"]
        if f"adx_{index}" not in self.cache:
            adx_ = adx(self.candles, period=index, sequential=True)
            self.cache[f"adx_{index}"] = adx_

        self._process_transformations(
            f"adx_{index}", self.cache[f"adx_{index}"], **kwargs
        )

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

    def amihud_lambda(self, **kwargs):
        if "amihud_lambda" not in self.cache:
            amihud_lambda_ = amihud_lambda(self.candles, sequential=True)
            self.cache["amihud_lambda"] = amihud_lambda_

        self._process_transformations(
            "amihud_lambda", self.cache["amihud_lambda"], **kwargs
        )

    def aroon_diff(self, **kwargs):
        if "aroon_diff" not in self.cache:
            aroon_: AROON = aroon(self.candles, sequential=True)
            self.cache["aroon_diff"] = aroon_.up - aroon_.down

        self._process_transformations("aroon_diff", self.cache["aroon_diff"], **kwargs)

    def bandpass(self, **kwargs):
        if "bandpass" not in self.cache or "highpass_bp" not in self.cache:
            bandpass_tuple = ta.bandpass(self.candles, sequential=True)
            self.cache["bandpass"] = bandpass_tuple.bp_normalized
            self.cache["highpass_bp"] = bandpass_tuple.trigger

        self._process_transformations("bandpass", self.cache["bandpass"], **kwargs)

    def bekker_parkinson_vol(self, **kwargs):
        if "bekker_parkinson_vol" not in self.cache:
            bekker_parkinson_vol_ = bekker_parkinson_vol(self.candles, sequential=True)
            self.cache["bekker_parkinson_vol"] = bekker_parkinson_vol_

        self._process_transformations(
            "bekker_parkinson_vol", self.cache["bekker_parkinson_vol"], **kwargs
        )

    def highpass_bp(self, **kwargs):
        if "bandpass" not in self.cache or "highpass_bp" not in self.cache:
            bandpass_tuple = ta.bandpass(self.candles, sequential=True)
            self.cache["bandpass"] = bandpass_tuple.bp_normalized
            self.cache["highpass_bp"] = bandpass_tuple.trigger

        self._process_transformations(
            "highpass_bp", self.cache["highpass_bp"], **kwargs
        )

    def chaiken_money_flow(self, **kwargs):
        if "chaiken_money_flow" not in self.cache:
            chaiken_money_flow_ = chaiken_money_flow(self.candles, sequential=True)
            self.cache["chaiken_money_flow"] = chaiken_money_flow_

        self._process_transformations(
            "chaiken_money_flow", self.cache["chaiken_money_flow"], **kwargs
        )

    def change_variance_ratio(self, **kwargs):
        if "change_variance_ratio" not in self.cache:
            change_variance_ratio_ = change_variance_ratio(
                self.candles, sequential=True
            )
            self.cache["change_variance_ratio"] = change_variance_ratio_

        self._process_transformations(
            "change_variance_ratio", self.cache["change_variance_ratio"], **kwargs
        )

    def cmma(self, **kwargs):
        if "cmma" not in self.cache:
            cmma_ = cmma(self.candles, sequential=True)
            self.cache["cmma"] = cmma_

        self._process_transformations("cmma", self.cache["cmma"], **kwargs)

    def corwin_schultz_estimator(self, **kwargs):
        if "corwin_schultz_estimator" not in self.cache:
            corwin_schultz_estimator_ = corwin_schultz_estimator(
                self.candles, sequential=True
            )
            self.cache["corwin_schultz_estimator"] = corwin_schultz_estimator_

        self._process_transformations(
            "corwin_schultz_estimator", self.cache["corwin_schultz_estimator"], **kwargs
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

    def cwt_win32(self, **kwargs):
        index = kwargs["index"]
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False
        lag_value = kwargs["lag"] if "lag" in kwargs else None

        cwt_win32 = (
            CWT_SWT(self.candles, 32, sequential=self.sequential)
            if "cwt_win32" not in self.cache_class_indicator
            else self.cache_class_indicator["cwt_win32"]
        )
        self.cache_class_indicator["cwt_win32"] = cwt_win32

        if f"cwt_win32_{index}" not in self.cache:
            cwt_win32_res = cwt_win32.res()
            for i in range(cwt_win32_res.shape[1]):
                self.cache[f"cwt_win32_{i}"] = cwt_win32_res[:, i]

        if lag_value is not None:
            feature_name = f"cwt_win32_{index}_lag{lag_value}"
            if feature_name not in self.cache:
                cwt_win32_lag = cwt_win32.res(lag=lag_value)
                for i in range(cwt_win32_lag.shape[1]):
                    self.cache[f"cwt_win32_{i}_lag{lag_value}"] = cwt_win32_lag[:, i]

        if dt:
            feature_name = f"cwt_win32_{index}_dt"
            if feature_name not in self.cache:
                cwt_win32_dt = cwt_win32.res(dt=True)
                for i in range(cwt_win32_dt.shape[1]):
                    self.cache[f"cwt_win32_{i}_dt"] = cwt_win32_dt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win32_{index}_dt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win32_dt_lag = cwt_win32.res(dt=True, lag=lag_value)
                    for i in range(cwt_win32_dt_lag.shape[1]):
                        self.cache[f"cwt_win32_{i}_dt_lag{lag_value}"] = (
                            cwt_win32_dt_lag[:, i]
                        )

        if ddt:
            feature_name = f"cwt_win32_{index}_ddt"
            if feature_name not in self.cache:
                cwt_win32_ddt = cwt_win32.res(ddt=True)
                for i in range(cwt_win32_ddt.shape[1]):
                    self.cache[f"cwt_win32_{i}_ddt"] = cwt_win32_ddt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win32_{index}_ddt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win32_ddt_lag = cwt_win32.res(ddt=True, lag=lag_value)
                    for i in range(cwt_win32_ddt_lag.shape[1]):
                        self.cache[f"cwt_win32_{i}_ddt_lag{lag_value}"] = (
                            cwt_win32_ddt_lag[:, i]
                        )

    def cwt_win64(self, **kwargs):
        index = kwargs["index"]
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False
        lag_value = kwargs["lag"] if "lag" in kwargs else None

        cwt_win64 = (
            CWT_SWT(self.candles, 64, sequential=self.sequential)
            if "cwt_win64" not in self.cache_class_indicator
            else self.cache_class_indicator["cwt_win64"]
        )
        self.cache_class_indicator["cwt_win64"] = cwt_win64

        if f"cwt_win64_{index}" not in self.cache:
            cwt_win64_res = cwt_win64.res()
            for i in range(cwt_win64_res.shape[1]):
                self.cache[f"cwt_win64_{i}"] = cwt_win64_res[:, i]

        if lag_value is not None:
            feature_name = f"cwt_win64_{index}_lag{lag_value}"
            if feature_name not in self.cache:
                cwt_win64_lag = cwt_win64.res(lag=lag_value)
                for i in range(cwt_win64_lag.shape[1]):
                    self.cache[f"cwt_win64_{i}_lag{lag_value}"] = cwt_win64_lag[:, i]

        if dt:
            feature_name = f"cwt_win64_{index}_dt"
            if feature_name not in self.cache:
                cwt_win64_dt = cwt_win64.res(dt=True)
                for i in range(cwt_win64_dt.shape[1]):
                    self.cache[f"cwt_win64_{i}_dt"] = cwt_win64_dt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win64_{index}_dt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win64_dt_lag = cwt_win64.res(dt=True, lag=lag_value)
                    for i in range(cwt_win64_dt_lag.shape[1]):
                        self.cache[f"cwt_win64_{i}_dt_lag{lag_value}"] = (
                            cwt_win64_dt_lag[:, i]
                        )

        if ddt:
            feature_name = f"cwt_win64_{index}_ddt"
            if feature_name not in self.cache:
                cwt_win64_ddt = cwt_win64.res(ddt=True)
                for i in range(cwt_win64_ddt.shape[1]):
                    self.cache[f"cwt_win64_{i}_ddt"] = cwt_win64_ddt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win64_{index}_ddt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win64_ddt_lag = cwt_win64.res(ddt=True, lag=lag_value)
                    for i in range(cwt_win64_ddt_lag.shape[1]):
                        self.cache[f"cwt_win64_{i}_ddt_lag{lag_value}"] = (
                            cwt_win64_ddt_lag[:, i]
                        )

    def cwt_win128(self, **kwargs):
        index = kwargs["index"]
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False
        lag_value = kwargs["lag"] if "lag" in kwargs else None

        cwt_win128 = (
            CWT_SWT(self.candles, 128, sequential=self.sequential)
            if "cwt_win128" not in self.cache_class_indicator
            else self.cache_class_indicator["cwt_win128"]
        )
        self.cache_class_indicator["cwt_win128"] = cwt_win128

        if f"cwt_win128_{index}" not in self.cache:
            cwt_win128_res = cwt_win128.res()
            for i in range(cwt_win128_res.shape[1]):
                self.cache[f"cwt_win128_{i}"] = cwt_win128_res[:, i]

        if lag_value is not None:
            feature_name = f"cwt_win128_{index}_lag{lag_value}"
            if feature_name not in self.cache:
                cwt_win128_lag = cwt_win128.res(lag=lag_value)
                for i in range(cwt_win128_lag.shape[1]):
                    self.cache[f"cwt_win128_{i}_lag{lag_value}"] = cwt_win128_lag[:, i]

        if dt:
            feature_name = f"cwt_win128_{index}_dt"
            if feature_name not in self.cache:
                cwt_win128_dt = cwt_win128.res(dt=True)
                for i in range(cwt_win128_dt.shape[1]):
                    self.cache[f"cwt_win128_{i}_dt"] = cwt_win128_dt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win128_{index}_dt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win128_dt_lag = cwt_win128.res(dt=True, lag=lag_value)
                    for i in range(cwt_win128_dt_lag.shape[1]):
                        self.cache[f"cwt_win128_{i}_dt_lag{lag_value}"] = (
                            cwt_win128_dt_lag[:, i]
                        )

        if ddt:
            feature_name = f"cwt_win128_{index}_ddt"
            if feature_name not in self.cache:
                cwt_win128_ddt = cwt_win128.res(ddt=True)
                for i in range(cwt_win128_ddt.shape[1]):
                    self.cache[f"cwt_win128_{i}_ddt"] = cwt_win128_ddt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win128_{index}_ddt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win128_ddt_lag = cwt_win128.res(ddt=True, lag=lag_value)
                    for i in range(cwt_win128_ddt_lag.shape[1]):
                        self.cache[f"cwt_win128_{i}_ddt_lag{lag_value}"] = (
                            cwt_win128_ddt_lag[:, i]
                        )

    def cwt_win256(self, **kwargs):
        index = kwargs["index"]
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False
        lag_value = kwargs["lag"] if "lag" in kwargs else None

        cwt_win256 = (
            CWT_SWT(self.candles, 256, sequential=self.sequential)
            if "cwt_win256" not in self.cache_class_indicator
            else self.cache_class_indicator["cwt_win256"]
        )
        self.cache_class_indicator["cwt_win256"] = cwt_win256

        if f"cwt_win256_{index}" not in self.cache:
            cwt_win256_res = cwt_win256.res()
            for i in range(cwt_win256_res.shape[1]):
                self.cache[f"cwt_win256_{i}"] = cwt_win256_res[:, i]

        if lag_value is not None:
            feature_name = f"cwt_win256_{index}_lag{lag_value}"
            if feature_name not in self.cache:
                cwt_win256_lag = cwt_win256.res(lag=lag_value)
                for i in range(cwt_win256_lag.shape[1]):
                    self.cache[f"cwt_win256_{i}_lag{lag_value}"] = cwt_win256_lag[:, i]

        if dt:
            feature_name = f"cwt_win256_{index}_dt"
            if feature_name not in self.cache:
                cwt_win256_dt = cwt_win256.res(dt=True)
                for i in range(cwt_win256_dt.shape[1]):
                    self.cache[f"cwt_win256_{i}_dt"] = cwt_win256_dt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win256_{index}_dt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win256_dt_lag = cwt_win256.res(dt=True, lag=lag_value)
                    for i in range(cwt_win256_dt_lag.shape[1]):
                        self.cache[f"cwt_win256_{i}_dt_lag{lag_value}"] = (
                            cwt_win256_dt_lag[:, i]
                        )

        if ddt:
            feature_name = f"cwt_win256_{index}_ddt"
            if feature_name not in self.cache:
                cwt_win256_ddt = cwt_win256.res(ddt=True)
                for i in range(cwt_win256_ddt.shape[1]):
                    self.cache[f"cwt_win256_{i}_ddt"] = cwt_win256_ddt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win256_{index}_ddt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win256_ddt_lag = cwt_win256.res(ddt=True, lag=lag_value)
                    for i in range(cwt_win256_ddt_lag.shape[1]):
                        self.cache[f"cwt_win256_{i}_ddt_lag{lag_value}"] = (
                            cwt_win256_ddt_lag[:, i]
                        )

    def cwt_win512(self, **kwargs):
        index = kwargs["index"]
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False
        lag_value = kwargs["lag"] if "lag" in kwargs else None

        cwt_win512 = (
            CWT_SWT(self.candles, 512, sequential=self.sequential)
            if "cwt_win512" not in self.cache_class_indicator
            else self.cache_class_indicator["cwt_win512"]
        )
        self.cache_class_indicator["cwt_win512"] = cwt_win512

        if f"cwt_win512_{index}" not in self.cache:
            cwt_win512_res = cwt_win512.res()
            for i in range(cwt_win512_res.shape[1]):
                self.cache[f"cwt_win512_{i}"] = cwt_win512_res[:, i]

        if lag_value is not None:
            feature_name = f"cwt_win512_{index}_lag{lag_value}"
            if feature_name not in self.cache:
                cwt_win512_lag = cwt_win512.res(lag=lag_value)
                for i in range(cwt_win512_lag.shape[1]):
                    self.cache[f"cwt_win512_{i}_lag{lag_value}"] = cwt_win512_lag[:, i]

        if dt:
            feature_name = f"cwt_win512_{index}_dt"
            if feature_name not in self.cache:
                cwt_win512_dt = cwt_win512.res(dt=True)
                for i in range(cwt_win512_dt.shape[1]):
                    self.cache[f"cwt_win512_{i}_dt"] = cwt_win512_dt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win512_{index}_dt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win512_dt_lag = cwt_win512.res(dt=True, lag=lag_value)
                    for i in range(cwt_win512_dt_lag.shape[1]):
                        self.cache[f"cwt_win512_{i}_dt_lag{lag_value}"] = (
                            cwt_win512_dt_lag[:, i]
                        )

        if ddt:
            feature_name = f"cwt_win512_{index}_ddt"
            if feature_name not in self.cache:
                cwt_win512_ddt = cwt_win512.res(ddt=True)
                for i in range(cwt_win512_ddt.shape[1]):
                    self.cache[f"cwt_win512_{i}_ddt"] = cwt_win512_ddt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win512_{index}_ddt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win512_ddt_lag = cwt_win512.res(ddt=True, lag=lag_value)
                    for i in range(cwt_win512_ddt_lag.shape[1]):
                        self.cache[f"cwt_win512_{i}_ddt_lag{lag_value}"] = (
                            cwt_win512_ddt_lag[:, i]
                        )

    def cwt_win1024(self, **kwargs):
        index = kwargs["index"]
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False
        lag_value = kwargs["lag"] if "lag" in kwargs else None

        cwt_win1024 = (
            CWT_SWT(self.candles, 1024, sequential=self.sequential)
            if "cwt_win1024" not in self.cache_class_indicator
            else self.cache_class_indicator["cwt_win1024"]
        )
        self.cache_class_indicator["cwt_win1024"] = cwt_win1024

        if f"cwt_win1024_{index}" not in self.cache:
            cwt_win1024_res = cwt_win1024.res()
            for i in range(cwt_win1024_res.shape[1]):
                self.cache[f"cwt_win1024_{i}"] = cwt_win1024_res[:, i]

        if lag_value is not None:
            feature_name = f"cwt_win1024_{index}_lag{lag_value}"
            if feature_name not in self.cache:
                cwt_win1024_lag = cwt_win1024.res(lag=lag_value)
                for i in range(cwt_win1024_lag.shape[1]):
                    self.cache[f"cwt_win1024_{i}_lag{lag_value}"] = cwt_win1024_lag[
                        :, i
                    ]

        if dt:
            feature_name = f"cwt_win1024_{index}_dt"
            if feature_name not in self.cache:
                cwt_win1024_dt = cwt_win1024.res(dt=True)
                for i in range(cwt_win1024_dt.shape[1]):
                    self.cache[f"cwt_win1024_{i}_dt"] = cwt_win1024_dt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win1024_{index}_dt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win1024_dt_lag = cwt_win1024.res(dt=True, lag=lag_value)
                    for i in range(cwt_win1024_dt_lag.shape[1]):
                        self.cache[f"cwt_win1024_{i}_dt_lag{lag_value}"] = (
                            cwt_win1024_dt_lag[:, i]
                        )

        if ddt:
            feature_name = f"cwt_win1024_{index}_ddt"
            if feature_name not in self.cache:
                cwt_win1024_ddt = cwt_win1024.res(ddt=True)
                for i in range(cwt_win1024_ddt.shape[1]):
                    self.cache[f"cwt_win1024_{i}_ddt"] = cwt_win1024_ddt[:, i]

            if lag_value is not None:
                feature_name = f"cwt_win1024_{index}_ddt_lag{lag_value}"
                if feature_name not in self.cache:
                    cwt_win1024_ddt_lag = cwt_win1024.res(ddt=True, lag=lag_value)
                    for i in range(cwt_win1024_ddt_lag.shape[1]):
                        self.cache[f"cwt_win1024_{i}_ddt_lag{lag_value}"] = (
                            cwt_win1024_ddt_lag[:, i]
                        )

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

    def sample_entropy_win32_spot(self, **kwargs):
        if "sample_entropy_win32_spot" not in self.cache:
            sample_entropy_win32_spot_ = sample_entropy_indicator(
                self.candles, period=32, sequential=self.sequential
            )
            self.cache["sample_entropy_win32_spot"] = sample_entropy_win32_spot_

    def sample_entropy_win32_array(self, **kwargs):
        if "sample_entropy_win32_array" not in self.cache:
            sample_entropy_win32_array_ = sample_entropy_indicator(
                self.candles,
                period=32,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["sample_entropy_win32_array"] = sample_entropy_win32_array_

    def sample_entropy_win64_spot(self, **kwargs):
        if "sample_entropy_win64_spot" not in self.cache:
            sample_entropy_win64_spot_ = sample_entropy_indicator(
                self.candles, period=64, sequential=self.sequential
            )
            self.cache["sample_entropy_win64_spot"] = sample_entropy_win64_spot_

    def sample_entropy_win64_array(self, **kwargs):
        if "sample_entropy_win64_array" not in self.cache:
            sample_entropy_win64_array_ = sample_entropy_indicator(
                self.candles,
                period=64,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["sample_entropy_win64_array"] = sample_entropy_win64_array_

    def sample_entropy_win128_spot(self, **kwargs):
        if "sample_entropy_win128_spot" not in self.cache:
            sample_entropy_win128_spot_ = sample_entropy_indicator(
                self.candles, period=128, sequential=self.sequential
            )
            self.cache["sample_entropy_win128_spot"] = sample_entropy_win128_spot_

    def sample_entropy_win128_array(self, **kwargs):
        if "sample_entropy_win128_array" not in self.cache:
            sample_entropy_win128_array_ = sample_entropy_indicator(
                self.candles,
                period=128,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["sample_entropy_win128_array"] = sample_entropy_win128_array_

    def sample_entropy_win256_spot(self, **kwargs):
        if "sample_entropy_win256_spot" not in self.cache:
            sample_entropy_win256_spot_ = sample_entropy_indicator(
                self.candles, period=256, sequential=self.sequential
            )
            self.cache["sample_entropy_win256_spot"] = sample_entropy_win256_spot_

    def sample_entropy_win256_array(self, **kwargs):
        if "sample_entropy_win256_array" not in self.cache:
            sample_entropy_win256_array_ = sample_entropy_indicator(
                self.candles,
                period=256,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["sample_entropy_win256_array"] = sample_entropy_win256_array_

    def sample_entropy_win512_spot(self, **kwargs):
        if "sample_entropy_win512_spot" not in self.cache:
            sample_entropy_win512_spot_ = sample_entropy_indicator(
                self.candles, period=512, sequential=self.sequential
            )
            self.cache["sample_entropy_win512_spot"] = sample_entropy_win512_spot_

    def sample_entropy_win512_array(self, **kwargs):
        if "sample_entropy_win512_array" not in self.cache:
            sample_entropy_win512_array_ = sample_entropy_indicator(
                self.candles,
                period=512,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["sample_entropy_win512_array"] = sample_entropy_win512_array_

    def approximate_entropy_win32_spot(self, **kwargs):
        if "approximate_entropy_win32_spot" not in self.cache:
            approximate_entropy_win32_spot_ = approximate_entropy_indicator(
                self.candles, period=32, sequential=self.sequential
            )
            self.cache["approximate_entropy_win32_spot"] = (
                approximate_entropy_win32_spot_
            )

    def approximate_entropy_win32_array(self, **kwargs):
        if "approximate_entropy_win32_array" not in self.cache:
            approximate_entropy_win32_array_ = approximate_entropy_indicator(
                self.candles,
                period=32,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["approximate_entropy_win32_array"] = (
                approximate_entropy_win32_array_
            )

    def approximate_entropy_win64_spot(self, **kwargs):
        if "approximate_entropy_win64_spot" not in self.cache:
            approximate_entropy_win64_spot_ = approximate_entropy_indicator(
                self.candles, period=64, sequential=self.sequential
            )
            self.cache["approximate_entropy_win64_spot"] = (
                approximate_entropy_win64_spot_
            )

    def approximate_entropy_win64_array(self, **kwargs):
        if "approximate_entropy_win64_array" not in self.cache:
            approximate_entropy_win64_array_ = approximate_entropy_indicator(
                self.candles,
                period=64,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["approximate_entropy_win64_array"] = (
                approximate_entropy_win64_array_
            )

    def approximate_entropy_win128_spot(self, **kwargs):
        if "approximate_entropy_win128_spot" not in self.cache:
            approximate_entropy_win128_spot_ = approximate_entropy_indicator(
                self.candles, period=128, sequential=self.sequential
            )
            self.cache["approximate_entropy_win128_spot"] = (
                approximate_entropy_win128_spot_
            )

    def approximate_entropy_win128_array(self, **kwargs):
        if "approximate_entropy_win128_array" not in self.cache:
            approximate_entropy_win128_array_ = approximate_entropy_indicator(
                self.candles,
                period=128,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["approximate_entropy_win128_array"] = (
                approximate_entropy_win128_array_
            )

    def approximate_entropy_win256_spot(self, **kwargs):
        if "approximate_entropy_win256_spot" not in self.cache:
            approximate_entropy_win256_spot_ = approximate_entropy_indicator(
                self.candles, period=256, sequential=self.sequential
            )
            self.cache["approximate_entropy_win256_spot"] = (
                approximate_entropy_win256_spot_
            )

    def approximate_entropy_win256_array(self, **kwargs):
        if "approximate_entropy_win256_array" not in self.cache:
            approximate_entropy_win256_array_ = approximate_entropy_indicator(
                self.candles,
                period=256,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["approximate_entropy_win256_array"] = (
                approximate_entropy_win256_array_
            )

    def approximate_entropy_win512_spot(self, **kwargs):
        if "approximate_entropy_win512_spot" not in self.cache:
            approximate_entropy_win512_spot_ = approximate_entropy_indicator(
                self.candles, period=512, sequential=self.sequential
            )
            self.cache["approximate_entropy_win512_spot"] = (
                approximate_entropy_win512_spot_
            )

    def approximate_entropy_win512_array(self, **kwargs):
        if "approximate_entropy_win512_array" not in self.cache:
            approximate_entropy_win512_array_ = approximate_entropy_indicator(
                self.candles,
                period=512,
                use_array_price=True,
                sequential=self.sequential,
            )
            self.cache["approximate_entropy_win512_array"] = (
                approximate_entropy_win512_array_
            )

    def entropy_for_jesse(self, **kwargs):
        if "entropy_for_jesse" not in self.cache:
            entropy_for_jesse_ = entropy_for_jesse(self.candles, sequential=True)
            self.cache["entropy_for_jesse"] = entropy_for_jesse_

        self._process_transformations(
            "entropy_for_jesse", self.cache["entropy_for_jesse"], **kwargs
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

    def frac_diff_ffd(self, **kwargs):
        if "frac_diff_ffd" not in self.cache:
            frac_diff_ffd_ = frac_diff_ffd_candle(
                self.candles, diff_amt=0.35, sequential=True
            )
            self.cache["frac_diff_ffd"] = frac_diff_ffd_

        self._process_transformations(
            "frac_diff_ffd", self.cache["frac_diff_ffd"], **kwargs
        )

    def fti(self, **kwargs):
        if "fti" not in self.cache:
            fti_: FTIResult = fti(self.candles, sequential=True)
            self.cache["fti"] = fti_.fti
            self.cache["fti_best_period"] = fti_.best_period

        self._process_transformations("fti", self.cache["fti"], **kwargs)

    def fti_best_period(self, **kwargs):
        if "fti_best_period" not in self.cache:
            fti_: FTIResult = fti(self.candles, sequential=True)
            self.cache["fti"] = fti_.fti
            self.cache["fti_best_period"] = fti_.best_period

        self._process_transformations(
            "fti_best_period", self.cache["fti_best_period"], **kwargs
        )

    def forecast_oscillator(self, **kwargs):
        if "forecast_oscillator" not in self.cache:
            forecast_oscillator = ta.fosc(self.candles, sequential=True)
            self.cache["forecast_oscillator"] = forecast_oscillator

        self._process_transformations(
            "forecast_oscillator", self.cache["forecast_oscillator"], **kwargs
        )

    def hasbrouck_lambda(self, **kwargs):
        if "hasbrouck_lambda" not in self.cache:
            hasbrouck_lambda_ = hasbrouck_lambda(self.candles, sequential=True)
            self.cache["hasbrouck_lambda"] = hasbrouck_lambda_

        self._process_transformations(
            "hasbrouck_lambda", self.cache["hasbrouck_lambda"], **kwargs
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

    def iqr_ratio(self, **kwargs):
        if "iqr_ratio" not in self.cache:
            iqr_ratio_ = iqr_ratio(self.candles, sequential=True)
            self.cache["iqr_ratio"] = iqr_ratio_

        self._process_transformations("iqr_ratio", self.cache["iqr_ratio"], **kwargs)

    def kyle_lambda(self, **kwargs):
        if "kyle_lambda" not in self.cache:
            kyle_lambda_ = kyle_lambda(self.candles, sequential=True)
            self.cache["kyle_lambda"] = kyle_lambda_

        self._process_transformations(
            "kyle_lambda", self.cache["kyle_lambda"], **kwargs
        )

    def ma_difference(self, **kwargs):
        if "ma_difference" not in self.cache:
            ma_difference_ = ma_difference(self.candles, sequential=True)
            self.cache["ma_difference"] = ma_difference_

        self._process_transformations(
            "ma_difference", self.cache["ma_difference"], **kwargs
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

    def norm_on_balance_volume(self, **kwargs):
        if "norm_on_balance_volume" not in self.cache:
            norm_on_balance_volume_ = norm_on_balance_volume(
                self.candles, sequential=True
            )
            self.cache["norm_on_balance_volume"] = norm_on_balance_volume_

        self._process_transformations(
            "norm_on_balance_volume", self.cache["norm_on_balance_volume"], **kwargs
        )

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

    def price_change_oscillator(self, **kwargs):
        if "price_change_oscillator" not in self.cache:
            price_change_oscillator_ = price_change_oscillator(
                self.candles, sequential=True
            )
            self.cache["price_change_oscillator"] = price_change_oscillator_

        self._process_transformations(
            "price_change_oscillator", self.cache["price_change_oscillator"], **kwargs
        )

    def price_variance_ratio(self, **kwargs):
        if "price_variance_ratio" not in self.cache:
            price_variance_ratio_ = price_variance_ratio(self.candles, sequential=True)
            self.cache["price_variance_ratio"] = price_variance_ratio_

        self._process_transformations(
            "price_variance_ratio", self.cache["price_variance_ratio"], **kwargs
        )

    def reactivity(self, **kwargs):
        if "reactivity" not in self.cache:
            reactivity_ = reactivity(self.candles, sequential=True)
            self.cache["reactivity"] = reactivity_

        self._process_transformations("reactivity", self.cache["reactivity"], **kwargs)

    def roll_impact(self, **kwargs):
        if "roll_impact" not in self.cache:
            roll_impact_ = roll_impact(self.candles, sequential=True)
            self.cache["roll_impact"] = roll_impact_

        self._process_transformations(
            "roll_impact", self.cache["roll_impact"], **kwargs
        )

    def roll_measure(self, **kwargs):
        if "roll_measure" not in self.cache:
            roll_measure_ = roll_measure(self.candles, sequential=True)
            self.cache["roll_measure"] = roll_measure_

        self._process_transformations(
            "roll_measure", self.cache["roll_measure"], **kwargs
        )

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

    def vmd_win32(self, **kwargs):
        index = kwargs["index"]
        lag = kwargs["lag"] if "lag" in kwargs else None
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False

        vmd_win32 = (
            VMD_NRBO(self.candles, 32, sequential=self.sequential)
            if "vmd_win32" not in self.cache_class_indicator
            else self.cache_class_indicator["vmd_win32"]
        )
        self.cache_class_indicator["vmd_win32"] = vmd_win32

        if f"vmd_win32_{index}" not in self.cache:
            vmd_win32_res = vmd_win32.res()
            for i in range(vmd_win32_res.shape[1]):
                self.cache[f"vmd_win32_{i}"] = vmd_win32_res[:, i]

        if lag is not None:
            feature_name = f"vmd_win32_{index}_lag{lag}"
            if feature_name not in self.cache:
                vmd_win32_lag = vmd_win32.res(lag=lag)
                for i in range(vmd_win32_lag.shape[1]):
                    self.cache[f"vmd_win32_{i}_lag{lag}"] = vmd_win32_lag[:, i]

        if dt:
            feature_name = f"vmd_win32_{index}_dt"
            if feature_name not in self.cache:
                vmd_win32_dt = vmd_win32.res(dt=True)
                for i in range(vmd_win32_dt.shape[1]):
                    self.cache[f"vmd_win32_{i}_dt"] = vmd_win32_dt[:, i]

            if lag is not None:
                feature_name = f"vmd_win32_{index}_dt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win32_dt_lag = vmd_win32.res(dt=True, lag=lag)
                    for i in range(vmd_win32_dt_lag.shape[1]):
                        self.cache[f"vmd_win32_{i}_dt_lag{lag}"] = vmd_win32_dt_lag[
                            :, i
                        ]

        if ddt:
            feature_name = f"vmd_win32_{index}_ddt"
            if feature_name not in self.cache:
                vmd_win32_ddt = vmd_win32.res(ddt=True)
                for i in range(vmd_win32_ddt.shape[1]):
                    self.cache[f"vmd_win32_{i}_ddt"] = vmd_win32_ddt[:, i]

            if lag is not None:
                feature_name = f"vmd_win32_{index}_ddt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win32_ddt_lag = vmd_win32.res(ddt=True, lag=lag)
                    for i in range(vmd_win32_ddt_lag.shape[1]):
                        self.cache[f"vmd_win32_{i}_ddt_lag{lag}"] = vmd_win32_ddt_lag[
                            :, i
                        ]

    def vmd_win64(self, **kwargs):
        index = kwargs["index"]
        lag = kwargs["lag"] if "lag" in kwargs else None
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False

        vmd_win64 = (
            VMD_NRBO(self.candles, 64, sequential=self.sequential)
            if "vmd_win64" not in self.cache_class_indicator
            else self.cache_class_indicator["vmd_win64"]
        )
        self.cache_class_indicator["vmd_win64"] = vmd_win64

        if f"vmd_win64_{index}" not in self.cache:
            vmd_win64_res = vmd_win64.res()
            for i in range(vmd_win64_res.shape[1]):
                self.cache[f"vmd_win64_{i}"] = vmd_win64_res[:, i]

        if lag is not None:
            feature_name = f"vmd_win64_{index}_lag{lag}"
            if feature_name not in self.cache:
                vmd_win64_lag = vmd_win64.res(lag=lag)
                for i in range(vmd_win64_lag.shape[1]):
                    self.cache[f"vmd_win64_{i}_lag{lag}"] = vmd_win64_lag[:, i]

        if dt:
            feature_name = f"vmd_win64_{index}_dt"
            if feature_name not in self.cache:
                vmd_win64_dt = vmd_win64.res(dt=True)
                for i in range(vmd_win64_dt.shape[1]):
                    self.cache[f"vmd_win64_{i}_dt"] = vmd_win64_dt[:, i]

            if lag is not None:
                feature_name = f"vmd_win64_{index}_dt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win64_dt_lag = vmd_win64.res(dt=True, lag=lag)
                    for i in range(vmd_win64_dt_lag.shape[1]):
                        self.cache[f"vmd_win64_{i}_dt_lag{lag}"] = vmd_win64_dt_lag[
                            :, i
                        ]

        if ddt:
            feature_name = f"vmd_win64_{index}_ddt"
            if feature_name not in self.cache:
                vmd_win64_ddt = vmd_win64.res(ddt=True)
                for i in range(vmd_win64_ddt.shape[1]):
                    self.cache[f"vmd_win64_{i}_ddt"] = vmd_win64_ddt[:, i]

            if lag is not None:
                feature_name = f"vmd_win64_{index}_ddt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win64_ddt_lag = vmd_win64.res(ddt=True, lag=lag)
                    for i in range(vmd_win64_ddt_lag.shape[1]):
                        self.cache[f"vmd_win64_{i}_ddt_lag{lag}"] = vmd_win64_ddt_lag[
                            :, i
                        ]

    def vmd_win128(self, **kwargs):
        index = kwargs["index"]
        lag = kwargs["lag"] if "lag" in kwargs else None
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False

        vmd_win128 = (
            VMD_NRBO(self.candles, 128, sequential=self.sequential)
            if "vmd_win128" not in self.cache_class_indicator
            else self.cache_class_indicator["vmd_win128"]
        )
        self.cache_class_indicator["vmd_win128"] = vmd_win128

        if f"vmd_win128_{index}" not in self.cache:
            vmd_win128_res = vmd_win128.res()
            for i in range(vmd_win128_res.shape[1]):
                self.cache[f"vmd_win128_{i}"] = vmd_win128_res[:, i]

        if lag is not None:
            feature_name = f"vmd_win128_{index}_lag{lag}"
            if feature_name not in self.cache:
                vmd_win128_lag = vmd_win128.res(lag=lag)
                for i in range(vmd_win128_lag.shape[1]):
                    self.cache[f"vmd_win128_{i}_lag{lag}"] = vmd_win128_lag[:, i]

        if dt:
            feature_name = f"vmd_win128_{index}_dt"
            if feature_name not in self.cache:
                vmd_win128_dt = vmd_win128.res(dt=True)
                for i in range(vmd_win128_dt.shape[1]):
                    self.cache[f"vmd_win128_{i}_dt"] = vmd_win128_dt[:, i]

            if lag is not None:
                feature_name = f"vmd_win128_{index}_dt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win128_dt_lag = vmd_win128.res(dt=True, lag=lag)
                    for i in range(vmd_win128_dt_lag.shape[1]):
                        self.cache[f"vmd_win128_{i}_dt_lag{lag}"] = vmd_win128_dt_lag[
                            :, i
                        ]

        if ddt:
            feature_name = f"vmd_win128_{index}_ddt"
            if feature_name not in self.cache:
                vmd_win128_ddt = vmd_win128.res(ddt=True)
                for i in range(vmd_win128_ddt.shape[1]):
                    self.cache[f"vmd_win128_{i}_ddt"] = vmd_win128_ddt[:, i]

            if lag is not None:
                feature_name = f"vmd_win128_{index}_ddt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win128_ddt_lag = vmd_win128.res(ddt=True, lag=lag)
                    for i in range(vmd_win128_ddt_lag.shape[1]):
                        self.cache[f"vmd_win128_{i}_ddt_lag{lag}"] = vmd_win128_ddt_lag[
                            :, i
                        ]

    def vmd_win256(self, **kwargs):
        index = kwargs["index"]
        lag = kwargs["lag"] if "lag" in kwargs else None
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False

        vmd_win256 = (
            VMD_NRBO(self.candles, 256, sequential=self.sequential)
            if "vmd_win256" not in self.cache_class_indicator
            else self.cache_class_indicator["vmd_win256"]
        )
        self.cache_class_indicator["vmd_win256"] = vmd_win256

        if f"vmd_win256_{index}" not in self.cache:
            vmd_win256_res = vmd_win256.res()
            for i in range(vmd_win256_res.shape[1]):
                self.cache[f"vmd_win256_{i}"] = vmd_win256_res[:, i]

        if lag is not None:
            feature_name = f"vmd_win256_{index}_lag{lag}"
            if feature_name not in self.cache:
                vmd_win256_lag = vmd_win256.res(lag=lag)
                for i in range(vmd_win256_lag.shape[1]):
                    self.cache[f"vmd_win256_{i}_lag{lag}"] = vmd_win256_lag[:, i]

        if dt:
            feature_name = f"vmd_win256_{index}_dt"
            if feature_name not in self.cache:
                vmd_win256_dt = vmd_win256.res(dt=True)
                for i in range(vmd_win256_dt.shape[1]):
                    self.cache[f"vmd_win256_{i}_dt"] = vmd_win256_dt[:, i]

            if lag is not None:
                feature_name = f"vmd_win256_{index}_dt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win256_dt_lag = vmd_win256.res(dt=True, lag=lag)
                    for i in range(vmd_win256_dt_lag.shape[1]):
                        self.cache[f"vmd_win256_{i}_dt_lag{lag}"] = vmd_win256_dt_lag[
                            :, i
                        ]

        if ddt:
            feature_name = f"vmd_win256_{index}_ddt"
            if feature_name not in self.cache:
                vmd_win256_ddt = vmd_win256.res(ddt=True)
                for i in range(vmd_win256_ddt.shape[1]):
                    self.cache[f"vmd_win256_{i}_ddt"] = vmd_win256_ddt[:, i]

            if lag is not None:
                feature_name = f"vmd_win256_{index}_ddt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win256_ddt_lag = vmd_win256.res(ddt=True, lag=lag)
                    for i in range(vmd_win256_ddt_lag.shape[1]):
                        self.cache[f"vmd_win256_{i}_ddt_lag{lag}"] = vmd_win256_ddt_lag[
                            :, i
                        ]

    def vmd_win512(self, **kwargs):
        index = kwargs["index"]
        lag = kwargs["lag"] if "lag" in kwargs else None
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False

        vmd_win512 = (
            VMD_NRBO(self.candles, 512, sequential=self.sequential)
            if "vmd_win512" not in self.cache_class_indicator
            else self.cache_class_indicator["vmd_win512"]
        )
        self.cache_class_indicator["vmd_win512"] = vmd_win512

        if f"vmd_win512_{index}" not in self.cache:
            vmd_win512_res = vmd_win512.res()
            for i in range(vmd_win512_res.shape[1]):
                self.cache[f"vmd_win512_{i}"] = vmd_win512_res[:, i]

        if lag is not None:
            feature_name = f"vmd_win512_{index}_lag{lag}"
            if feature_name not in self.cache:
                vmd_win512_lag = vmd_win512.res(lag=lag)
                for i in range(vmd_win512_lag.shape[1]):
                    self.cache[f"vmd_win512_{i}_lag{lag}"] = vmd_win512_lag[:, i]

        if dt:
            feature_name = f"vmd_win512_{index}_dt"
            if feature_name not in self.cache:
                vmd_win512_dt = vmd_win512.res(dt=True)
                for i in range(vmd_win512_dt.shape[1]):
                    self.cache[f"vmd_win512_{i}_dt"] = vmd_win512_dt[:, i]

            if lag is not None:
                feature_name = f"vmd_win512_{index}_dt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win512_dt_lag = vmd_win512.res(dt=True, lag=lag)
                    for i in range(vmd_win512_dt_lag.shape[1]):
                        self.cache[f"vmd_win512_{i}_dt_lag{lag}"] = vmd_win512_dt_lag[
                            :, i
                        ]

        if ddt:
            feature_name = f"vmd_win512_{index}_ddt"
            if feature_name not in self.cache:
                vmd_win512_ddt = vmd_win512.res(ddt=True)
                for i in range(vmd_win512_ddt.shape[1]):
                    self.cache[f"vmd_win512_{i}_ddt"] = vmd_win512_ddt[:, i]

            if lag is not None:
                feature_name = f"vmd_win512_{index}_ddt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win512_ddt_lag = vmd_win512.res(ddt=True, lag=lag)
                    for i in range(vmd_win512_ddt_lag.shape[1]):
                        self.cache[f"vmd_win512_{i}_ddt_lag{lag}"] = vmd_win512_ddt_lag[
                            :, i
                        ]

    def vmd_win1024(self, **kwargs):
        index = kwargs["index"]
        lag = kwargs["lag"] if "lag" in kwargs else None
        dt = True if "dt" in kwargs else False
        ddt = True if "ddt" in kwargs else False

        vmd_win1024 = (
            VMD_NRBO(self.candles, 1024, sequential=self.sequential)
            if "vmd_win1024" not in self.cache_class_indicator
            else self.cache_class_indicator["vmd_win1024"]
        )
        self.cache_class_indicator["vmd_win1024"] = vmd_win1024

        if f"vmd_win1024_{index}" not in self.cache:
            vmd_win1024_res = vmd_win1024.res()
            for i in range(vmd_win1024_res.shape[1]):
                self.cache[f"vmd_win1024_{i}"] = vmd_win1024_res[:, i]

        if lag is not None:
            feature_name = f"vmd_win1024_{index}_lag{lag}"
            if feature_name not in self.cache:
                vmd_win1024_lag = vmd_win1024.res(lag=lag)
                for i in range(vmd_win1024_lag.shape[1]):
                    self.cache[f"vmd_win1024_{i}_lag{lag}"] = vmd_win1024_lag[:, i]

        if dt:
            feature_name = f"vmd_win1024_{index}_dt"
            if feature_name not in self.cache:
                vmd_win1024_dt = vmd_win1024.res(dt=True)
                for i in range(vmd_win1024_dt.shape[1]):
                    self.cache[f"vmd_win1024_{i}_dt"] = vmd_win1024_dt[:, i]

            if lag is not None:
                feature_name = f"vmd_win1024_{index}_dt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win1024_dt_lag = vmd_win1024.res(dt=True, lag=lag)
                    for i in range(vmd_win1024_dt_lag.shape[1]):
                        self.cache[f"vmd_win1024_{i}_dt_lag{lag}"] = vmd_win1024_dt_lag[
                            :, i
                        ]

        if ddt:
            feature_name = f"vmd_win1024_{index}_ddt"
            if feature_name not in self.cache:
                vmd_win1024_ddt = vmd_win1024.res(ddt=True)
                for i in range(vmd_win1024_ddt.shape[1]):
                    self.cache[f"vmd_win1024_{i}_ddt"] = vmd_win1024_ddt[:, i]

            if lag is not None:
                feature_name = f"vmd_win1024_{index}_ddt_lag{lag}"
                if feature_name not in self.cache:
                    vmd_win1024_ddt_lag = vmd_win1024.res(ddt=True, lag=lag)
                    for i in range(vmd_win1024_ddt_lag.shape[1]):
                        self.cache[f"vmd_win1024_{i}_ddt_lag{lag}"] = (
                            vmd_win1024_ddt_lag[:, i]
                        )

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


if __name__ == "__main__":
    from pathlib import Path

    candles = np.load(Path(__file__).parent.parent.parent / "data" / "btc_1m.npy")[-10_0000:]
    check_features = [
        # class风格，多列返回，额外加工
        "vmd_win32_0_dt_lag3",
        "cwt_win128_1_ddt_lag1",
        # 函数型，多列返回
        "ac_2",
        # 函数型，单列返回
        "williams_r",
    ]

    feature_calculator = FeatureCalculator()
    feature_calculator.load(candles, sequential=True)
    for feature in check_features:
        print(feature, feature_calculator.get([feature]))
