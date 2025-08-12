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


if __name__ == "__main__":
    from pathlib import Path

    candles = np.load(Path(__file__).parent.parent / "data" / "btc_1m.npy")[-10_0000:]
    check_features = [
        "vmd_0_dt_lag3",
        "cwt_1_ddt_lag1",
        "vmd_0_lag1",
        "cwt_1_lag1",
        "vmd_0",
        "cwt_3",
    ]

    feature_calculator = FeatureCalculator()
    feature_calculator.load(candles, sequential=False)
    for feature in check_features:
        print(feature, feature_calculator.get([feature]))
