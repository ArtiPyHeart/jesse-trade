"""Hurst-ADF-KPSS 三重趋势验证器单元测试"""

import numpy as np
import pytest

from research.hurst_adf_kpss import (
    TrendValidator,
    calculate_hurst,
    get_lag_params,
    run_adf_test,
    run_kpss_test,
)


def _generate_random_walk(n: int, seed: int = 42) -> np.ndarray:
    """生成随机游走序列"""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


def _generate_trending(n: int, seed: int = 42) -> np.ndarray:
    """生成趋势序列"""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 0.5, n)
    noise = rng.normal(0, 0.005, n)
    prices = 100 * np.exp(trend + np.cumsum(noise))
    return prices


def _generate_mean_reverting(n: int, seed: int = 42) -> np.ndarray:
    """生成均值回归序列"""
    rng = np.random.default_rng(seed)
    prices = np.zeros(n)
    prices[0] = 100
    for i in range(1, n):
        prices[i] = prices[i - 1] + 0.1 * (100 - prices[i - 1]) + rng.normal(0, 0.5)
    return prices


def _make_candles(prices: np.ndarray) -> np.ndarray:
    """将价格序列转换为Jesse style K线"""
    n = len(prices)
    candles = np.zeros((n, 6))
    candles[:, 0] = np.arange(n) * 60000  # timestamp
    candles[:, 1] = prices  # open
    candles[:, 2] = prices  # close
    candles[:, 3] = prices * 1.001  # high
    candles[:, 4] = prices * 0.999  # low
    candles[:, 5] = 1000  # volume
    return candles


class TestHurst:
    """Hurst指数测试"""

    def test_random_walk_hurst_near_half(self):
        """随机游走的Hurst应接近0.5"""
        prices = _generate_random_walk(500)
        hurst = calculate_hurst(prices, min_lag=5, max_lag=50)
        assert 0.3 < hurst < 0.7

    def test_trending_hurst_above_half(self):
        """趋势序列的Hurst应大于0.5"""
        prices = _generate_trending(500)
        hurst = calculate_hurst(prices, min_lag=5, max_lag=50)
        assert hurst > 0.5

    def test_short_series_returns_nan(self):
        """过短序列应返回NaN"""
        prices = np.array([100, 101, 102, 103, 104])
        hurst = calculate_hurst(prices, min_lag=5, max_lag=10)
        assert np.isnan(hurst)

    def test_get_lag_params(self):
        """测试lag参数计算"""
        min_lag, max_lag = get_lag_params(60)
        assert min_lag == 10
        assert max_lag == 20


class TestStationarity:
    """平稳性检验测试"""

    def test_adf_random_walk(self):
        """随机游走ADF p值应较高（非平稳）"""
        prices = _generate_random_walk(200)
        stat, pvalue = run_adf_test(prices)
        assert not np.isnan(pvalue)
        # 随机游走通常非平稳，p > 0.05

    def test_kpss_random_walk(self):
        """随机游走KPSS检验"""
        prices = _generate_random_walk(200)
        stat, pvalue = run_kpss_test(prices)
        assert not np.isnan(pvalue)

    def test_adf_short_series(self):
        """过短序列应抛出断言错误"""
        prices = np.array([100, 101, 102])
        with pytest.raises(AssertionError):
            run_adf_test(prices)


class TestTrendValidator:
    """TrendValidator类测试"""

    def test_validate_basic(self):
        """基本验证流程测试"""
        prices = _generate_random_walk(200)
        candles = _make_candles(prices)

        validator = TrendValidator(window_size=40, step=10)
        results = validator.validate(candles)

        assert len(results) > 0
        assert "score" in results.columns
        assert "trend_type" in results.columns
        assert all(results["score"] >= 0)
        assert all(results["score"] <= 5)

    def test_summarize(self):
        """汇总统计测试"""
        prices = _generate_random_walk(200)
        candles = _make_candles(prices)

        validator = TrendValidator(window_size=40, step=10)
        results = validator.validate(candles)
        summary = validator.summarize(results)

        assert "mean_score" in summary
        assert "median_score" in summary
        assert "high_score_ratio" in summary
        assert 0 <= summary["mean_score"] <= 5

    def test_short_candles_raises(self):
        """K线数量不足应抛出异常"""
        prices = _generate_random_walk(30)
        candles = _make_candles(prices)

        validator = TrendValidator(window_size=40, step=5)
        with pytest.raises(ValueError):
            validator.validate(candles)

    def test_score_calculation(self):
        """评分计算测试"""
        validator = TrendValidator(window_size=40)

        # 强趋势：Hurst=0.7, ADF p=0.1, KPSS p=0.01
        score = validator._calculate_score(0.7, 0.1, 0.01)
        assert score == 5  # 2+1+1+1

        # 无法确定：Hurst=NaN
        score = validator._calculate_score(np.nan, 0.1, 0.01)
        assert score == 0

    def test_trend_classification(self):
        """趋势分类测试"""
        validator = TrendValidator(window_size=40)

        # 强趋势且非平稳（适合趋势策略）
        trend = validator._classify_trend(0.6, 0.1, 0.01)
        assert "强趋势且非平稳" in trend

        # 震荡平稳（不适合趋势策略）
        trend = validator._classify_trend(0.4, 0.01, 0.1)
        assert "震荡平稳" in trend

        # 弱趋势或反趋势
        trend = validator._classify_trend(0.5, 0.1, 0.1)
        assert "弱趋势或反趋势" in trend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
