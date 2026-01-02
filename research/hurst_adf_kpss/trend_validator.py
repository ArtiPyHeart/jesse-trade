"""三重趋势验证器

通过Hurst指数、ADF检验、KPSS检验三重验证判断市场趋势性。

评分规则 (0-5分):
- Hurst > 0.6: +2
- Hurst > 0.55: +1
- ADF p > 0.05 (非平稳): +1
- KPSS p < 0.05 (非平稳): +1
- 三重共识: +1
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .hurst import _calculate_hurst_from_log_returns, get_lag_params
from .stationarity import run_adf_test, run_kpss_test


def _run_stationarity_tests(series: np.ndarray) -> tuple[float, float]:
    _, adf_pvalue = run_adf_test(series)
    _, kpss_pvalue = run_kpss_test(series)
    return adf_pvalue, kpss_pvalue


class TrendValidator:
    """三重趋势验证器

    对Jesse style的K线数据执行滑动窗口三重检验。
    """

    def __init__(self, window_size: int, step: int = 5, n_jobs: int = os.cpu_count()):
        """初始化验证器

        Args:
            window_size: 滑动窗口大小（K线数量）
            step: 滑动步长
            n_jobs: 并行进程数，1 表示串行
        """
        assert window_size >= 20, f"window_size must be >= 20, got {window_size}"
        assert step >= 1, f"step must be >= 1, got {step}"
        assert n_jobs >= 1, f"n_jobs must be >= 1, got {n_jobs}"

        self.window_size = window_size
        self.step = step
        self.n_jobs = n_jobs

    def validate(self, candles: np.ndarray) -> pd.DataFrame:
        """对K线数据执行滑动窗口三重检验

        Args:
            candles: Jesse style K线数据，shape=(N, 6)
                     [timestamp, open, close, high, low, volume]

        Returns:
            包含检验结果的DataFrame
        """
        assert candles.ndim == 2, f"candles must be 2D, got {candles.ndim}D"
        assert (
            candles.shape[1] == 6
        ), f"candles must have 6 columns, got {candles.shape[1]}"

        close = candles[:, 2]  # close价格在索引2
        n = len(close)

        if n < self.window_size:
            raise ValueError(f"candles length({n}) < window_size({self.window_size})")

        min_lag, max_lag = get_lag_params(self.window_size)
        max_lag = min(max_lag, self.window_size // 2)
        lags = np.arange(min_lag, max_lag + 1)
        log_lags = np.log(lags) if len(lags) > 0 else lags

        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(close))
        start_indices = list(range(0, n - self.window_size + 1, self.step))
        window_slices = []
        hurst_values = []

        for start_idx in start_indices:
            end_idx = start_idx + self.window_size
            window_data = close[start_idx:end_idx]
            window_log_returns = log_returns[start_idx : end_idx - 1]

            hurst = _calculate_hurst_from_log_returns(
                window_log_returns,
                min_lag,
                max_lag,
                self.window_size,
                lags=lags,
                log_lags=log_lags,
            )
            hurst_values.append(hurst)
            window_slices.append(window_data)

        total_windows = len(window_slices)
        worker_limit = os.cpu_count() or 1
        worker_count = min(self.n_jobs, worker_limit, total_windows)

        if worker_count > 1:
            chunk_size = max(1, total_windows // (worker_count * 4))
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                stationarity_results = list(
                    executor.map(
                        _run_stationarity_tests,
                        window_slices,
                        chunksize=chunk_size,
                    )
                )
        else:
            stationarity_results = [
                _run_stationarity_tests(window_data) for window_data in window_slices
            ]

        results = []
        for idx, start_idx in enumerate(start_indices):
            end_idx = start_idx + self.window_size
            adf_pvalue, kpss_pvalue = stationarity_results[idx]
            hurst = hurst_values[idx]

            score = self._calculate_score(hurst, adf_pvalue, kpss_pvalue)
            trend_type = self._classify_trend(hurst, adf_pvalue, kpss_pvalue)

            results.append(
                {
                    "window_idx": idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "hurst": hurst,
                    "adf_pvalue": adf_pvalue,
                    "kpss_pvalue": kpss_pvalue,
                    "score": score,
                    "trend_type": trend_type,
                }
            )

        return pd.DataFrame(results)

    def _calculate_score(self, hurst: float, adf_p: float, kpss_p: float) -> int:
        """计算趋势适配性评分 (0-5分)"""
        if np.isnan(hurst):
            return 0

        score = 0

        # Hurst评分
        if hurst > 0.6:
            score += 2
        elif hurst > 0.55:
            score += 1

        # ADF评分 (p > 0.05 表示非平稳)
        if not np.isnan(adf_p) and adf_p > 0.05:
            score += 1

        # KPSS评分 (p < 0.05 表示非平稳)
        if not np.isnan(kpss_p) and kpss_p < 0.05:
            score += 1

        # 三重共识加分
        if (
            hurst > 0.55
            and not np.isnan(adf_p)
            and adf_p > 0.05
            and not np.isnan(kpss_p)
            and kpss_p < 0.05
        ):
            score += 1

        return min(score, 5)

    def _classify_trend(self, hurst: float, adf_p: float, kpss_p: float) -> str:
        """分类趋势类型（与notebook保持一致）"""
        if np.isnan(hurst):
            return "无法确定（Hurst指数计算失败）"

        # 使用严格不等号，与notebook一致
        adf_nonstationary = not np.isnan(adf_p) and adf_p > 0.05
        kpss_nonstationary = not np.isnan(kpss_p) and kpss_p < 0.05
        adf_stationary = not np.isnan(adf_p) and adf_p < 0.05  # 严格 <
        kpss_stationary = not np.isnan(kpss_p) and kpss_p > 0.05  # 严格 >

        if hurst > 0.55 and adf_nonstationary and kpss_nonstationary:
            return "强趋势且非平稳（适合趋势策略）"
        elif hurst > 0.55 and adf_stationary and kpss_stationary:
            return "趋势但平稳（短期趋势可能）"
        elif hurst < 0.5 and adf_stationary and kpss_stationary:
            return "震荡平稳（不适合趋势策略）"
        elif hurst > 0.55 and adf_nonstationary and kpss_stationary:
            return "矛盾（需进一步验证）"
        else:
            return "弱趋势或反趋势"

    def summarize(self, results: pd.DataFrame) -> dict:
        """汇总统计结果

        Args:
            results: validate()返回的DataFrame

        Returns:
            统计汇总字典
        """
        scores = results["score"]

        return {
            "window_size": self.window_size,
            "step": self.step,
            "total_windows": len(results),
            "mean_score": float(scores.mean()),
            "median_score": float(scores.median()),
            "std_score": float(scores.std()),
            "high_score_ratio": float((scores == 5).sum() / len(scores)),
            "low_score_ratio": float((scores <= 2).sum() / len(scores)),
        }
