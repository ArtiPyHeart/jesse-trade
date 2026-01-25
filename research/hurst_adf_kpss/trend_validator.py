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
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd

from .hurst import _calculate_hurst_from_log_returns, get_lag_params
from .stationarity import run_adf_test, run_kpss_test


def _run_stationarity_tests(series: np.ndarray) -> tuple[float, float]:
    _, adf_pvalue = run_adf_test(series)
    _, kpss_pvalue = run_kpss_test(series)
    return adf_pvalue, kpss_pvalue


_WORKER_CLOSE: Optional[np.ndarray] = None
_WORKER_WINDOW_SIZE: Optional[int] = None


def _init_stationarity_worker(close: np.ndarray, window_size: int) -> None:
    global _WORKER_CLOSE, _WORKER_WINDOW_SIZE
    _WORKER_CLOSE = close
    _WORKER_WINDOW_SIZE = window_size


def _run_stationarity_chunk(start_indices: np.ndarray) -> np.ndarray:
    close = _WORKER_CLOSE
    window_size = _WORKER_WINDOW_SIZE
    if close is None or window_size is None:
        raise RuntimeError("Stationarity worker not initialized")

    results = np.empty((len(start_indices), 2), dtype=np.float64)
    for idx, start_idx in enumerate(start_indices):
        window_data = close[start_idx : start_idx + window_size]
        adf_pvalue, kpss_pvalue = _run_stationarity_tests(window_data)
        results[idx, 0] = adf_pvalue
        results[idx, 1] = kpss_pvalue

    return results


def _run_stationarity_chunk_local(
    close: np.ndarray, window_size: int, start_indices: np.ndarray
) -> np.ndarray:
    results = np.empty((len(start_indices), 2), dtype=np.float64)
    for idx, start_idx in enumerate(start_indices):
        window_data = close[start_idx : start_idx + window_size]
        adf_pvalue, kpss_pvalue = _run_stationarity_tests(window_data)
        results[idx, 0] = adf_pvalue
        results[idx, 1] = kpss_pvalue

    return results


class TrendValidator:
    """三重趋势验证器

    对Jesse style的K线数据执行滑动窗口三重检验。
    """

    def __init__(
        self, window_size: int, step: int = 5, n_jobs: int = os.cpu_count() or 1
    ):
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
        self._use_thread_pool = sys.platform == "darwin"

        min_lag, max_lag = get_lag_params(window_size)
        max_lag = min(max_lag, window_size // 2)
        self._min_lag = min_lag
        self._max_lag = max_lag
        self._lags = np.arange(min_lag, max_lag + 1)
        self._log_lags = np.log(self._lags) if len(self._lags) > 0 else self._lags

    def validate(self, candles: np.ndarray) -> pd.DataFrame:
        """对K线数据执行滑动窗口三重检验

        Args:
            candles: Jesse style K线数据，shape=(N, 6)
                     [timestamp, open, close, high, low, volume]

        Returns:
            包含检验结果的DataFrame
        """
        assert candles.ndim == 2, f"candles must be 2D, got {candles.ndim}D"
        assert candles.shape[1] == 6, (
            f"candles must have 6 columns, got {candles.shape[1]}"
        )

        close = np.ascontiguousarray(candles[:, 2])  # close价格在索引2
        n = len(close)

        if n < self.window_size:
            raise ValueError(f"candles length({n}) < window_size({self.window_size})")

        min_lag = self._min_lag
        max_lag = self._max_lag
        lags = self._lags
        log_lags = self._log_lags

        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(close))
        start_indices = np.arange(0, n - self.window_size + 1, self.step, dtype=int)
        total_windows = len(start_indices)
        end_indices = start_indices + self.window_size
        hurst_values = np.empty(total_windows, dtype=np.float64)

        for idx, start_idx in enumerate(start_indices):
            end_idx = end_indices[idx]
            window_log_returns = log_returns[start_idx : end_idx - 1]
            hurst = _calculate_hurst_from_log_returns(
                window_log_returns,
                min_lag,
                max_lag,
                self.window_size,
                lags=lags,
                log_lags=log_lags,
            )
            hurst_values[idx] = hurst
        worker_limit = os.cpu_count() or 1
        worker_count = min(self.n_jobs, worker_limit, total_windows)
        if worker_count > 1:
            chunk_size = max(1, total_windows // (worker_count * 4))
            chunks = [
                start_indices[i : i + chunk_size]
                for i in range(0, total_windows, chunk_size)
            ]
            if self._use_thread_pool:
                worker_fn = partial(
                    _run_stationarity_chunk_local,
                    close,
                    self.window_size,
                )
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    stationarity_chunks = list(
                        executor.map(worker_fn, chunks, chunksize=1)
                    )
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_stationarity_worker,
                    initargs=(close, self.window_size),
                ) as executor:
                    stationarity_chunks = list(
                        executor.map(
                            _run_stationarity_chunk,
                            chunks,
                            chunksize=1,
                        )
                    )
            stationarity_results = np.vstack(stationarity_chunks)
        else:
            stationarity_results = _run_stationarity_chunk_local(
                close, self.window_size, start_indices
            )

        adf_pvalues = stationarity_results[:, 0]
        kpss_pvalues = stationarity_results[:, 1]

        scores, trend_types = self._calculate_scores_and_trends(
            hurst_values, adf_pvalues, kpss_pvalues
        )

        return pd.DataFrame(
            {
                "window_idx": np.arange(total_windows, dtype=int),
                "start_idx": start_indices,
                "end_idx": end_indices,
                "hurst": hurst_values,
                "adf_pvalue": adf_pvalues,
                "kpss_pvalue": kpss_pvalues,
                "score": scores,
                "trend_type": trend_types,
            }
        )

    def _calculate_scores_and_trends(
        self,
        hurst_values: np.ndarray,
        adf_pvalues: np.ndarray,
        kpss_pvalues: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """向量化评分与趋势分类"""
        assert hurst_values.ndim == 1, (
            f"hurst_values must be 1D, got {hurst_values.ndim}D"
        )
        assert adf_pvalues.shape == hurst_values.shape, (
            "adf_pvalues must match hurst_values shape"
        )
        assert kpss_pvalues.shape == hurst_values.shape, (
            "kpss_pvalues must match hurst_values shape"
        )

        valid_hurst = ~np.isnan(hurst_values)
        adf_nonstationary = ~np.isnan(adf_pvalues) & (adf_pvalues > 0.05)
        kpss_nonstationary = ~np.isnan(kpss_pvalues) & (kpss_pvalues < 0.05)
        adf_stationary = ~np.isnan(adf_pvalues) & (adf_pvalues < 0.05)
        kpss_stationary = ~np.isnan(kpss_pvalues) & (kpss_pvalues > 0.05)
        hurst_gt_06 = hurst_values > 0.6
        hurst_gt_055 = hurst_values > 0.55

        scores = np.zeros(len(hurst_values), dtype=np.int64)
        scores += np.where(hurst_gt_06, 2, np.where(hurst_gt_055, 1, 0)).astype(
            np.int64
        )
        scores += adf_nonstationary.astype(np.int64)
        scores += kpss_nonstationary.astype(np.int64)
        consensus = hurst_gt_055 & adf_nonstationary & kpss_nonstationary
        scores += consensus.astype(np.int64)
        scores = np.minimum(scores, 5)
        scores[~valid_hurst] = 0

        trend_types = np.full(len(hurst_values), "弱趋势或反趋势", dtype=object)
        trend_types[~valid_hurst] = "无法确定（Hurst指数计算失败）"
        valid_mask = valid_hurst

        mask = valid_mask & hurst_gt_055 & adf_nonstationary & kpss_nonstationary
        trend_types[mask] = "强趋势且非平稳（适合趋势策略）"

        mask = valid_mask & hurst_gt_055 & adf_stationary & kpss_stationary
        trend_types[mask] = "趋势但平稳（短期趋势可能）"

        mask = valid_mask & (hurst_values < 0.5) & adf_stationary & kpss_stationary
        trend_types[mask] = "震荡平稳（不适合趋势策略）"

        mask = valid_mask & hurst_gt_055 & adf_nonstationary & kpss_stationary
        trend_types[mask] = "矛盾（需进一步验证）"

        return scores, trend_types

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
        n = len(scores)

        # 计算 1-5 分各占比例
        score_distribution = {
            i: float((scores == i).sum() / n) for i in range(1, 6)
        }

        return {
            "window_size": self.window_size,
            "step": self.step,
            "total_windows": len(results),
            "mean_score": float(scores.mean()),
            "median_score": float(scores.median()),
            "std_score": float(scores.std()),
            "high_score_ratio": float((scores == 5).sum() / n),
            "low_score_ratio": float((scores <= 2).sum() / n),
            "score_distribution": score_distribution,  # {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.25, 5: 0.15}
        }
