"""Trend Optimizer 单元测试"""

import numpy as np
import pytest

from research.trend_optimizer import (
    MultiWindowEvaluator,
    TrendOptimizer,
    TrialResult,
)
from src.bars.fusion.demo import DemoBar


def _generate_random_walk(n: int, seed: int = 42) -> np.ndarray:
    """生成随机游走价格序列"""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


def _make_candles(prices: np.ndarray) -> np.ndarray:
    """将价格序列转换为 Jesse style K线"""
    n = len(prices)
    candles = np.zeros((n, 6))
    candles[:, 0] = np.arange(n) * 60000  # timestamp
    candles[:, 1] = prices  # open
    candles[:, 2] = prices  # close
    candles[:, 3] = prices * 1.001  # high
    candles[:, 4] = prices * 0.999  # low
    candles[:, 5] = 1000  # volume
    return candles


class TestMultiWindowEvaluator:
    """MultiWindowEvaluator 测试"""

    def test_check_constraints_pass(self):
        """约束检查通过"""
        evaluator = MultiWindowEvaluator(
            window_sizes=(20, 40, 60),
            min_bar_ratio_minutes=360,
        )
        # 1000 bars > max_window(60), 1000 > 10000//360 ≈ 27
        satisfied, reason = evaluator.check_constraints(1000, 10000)
        assert satisfied is True
        assert reason == "ok"

    def test_check_constraints_fail_max_window(self):
        """约束检查失败：bars < max_window"""
        evaluator = MultiWindowEvaluator(
            window_sizes=(20, 40, 60),
            min_bar_ratio_minutes=360,
        )
        satisfied, reason = evaluator.check_constraints(50, 10000)
        assert satisfied is False
        assert "max_window" in reason

    def test_check_constraints_fail_6h_baseline(self):
        """约束检查失败：bars <= 6h 基准"""
        evaluator = MultiWindowEvaluator(
            window_sizes=(20, 40, 60),
            min_bar_ratio_minutes=360,
        )
        # 100 bars <= 100000//360 ≈ 277
        satisfied, reason = evaluator.check_constraints(100, 100000)
        assert satisfied is False
        assert "6h_baseline" in reason

    def test_evaluate_basic(self):
        """基本评估测试"""
        prices = _generate_random_walk(500)
        candles = _make_candles(prices)

        # 生成 fusion bars
        container = DemoBar(threshold=0.5)
        container.update_with_candles(candles)
        fusion_bars = container.get_fusion_bars()

        # 确保有足够的 bars
        if len(fusion_bars) < 60:
            pytest.skip("Not enough fusion bars for evaluation")

        evaluator = MultiWindowEvaluator(window_sizes=(20, 40, 60))
        result = evaluator.evaluate(fusion_bars)

        assert "final_score" in result
        assert "window_scores" in result
        assert "window_summaries" in result
        assert 0 <= result["final_score"] <= 5
        assert all(ws in result["window_scores"] for ws in (20, 40, 60))


class TestTrendOptimizer:
    """TrendOptimizer 测试"""

    def test_init_valid(self):
        """初始化测试"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
            window_sizes=(20, 40, 60),
        )

        assert optimizer.fusion_bar_cls is DemoBar
        assert optimizer._filtered_count == 1000  # 所有 candles 的 volume > 0

    def test_validate_param_names_valid(self):
        """参数名校验：有效参数"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
        )

        # 不应抛出异常
        optimizer._validate_param_names(
            {"clip_r": (0.001, 0.01), "threshold": (0.5, 3.0)}
        )

    def test_validate_param_names_invalid(self):
        """参数名校验：无效参数"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
        )

        with pytest.raises(ValueError, match="Unknown param"):
            optimizer._validate_param_names({"invalid_param": (0.1, 0.5)})

    def test_validate_param_spec_not_tuple(self):
        """参数范围校验：非 tuple"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
        )

        with pytest.raises(ValueError, match="must be tuple"):
            optimizer._validate_param_names({"threshold": [0.1, 0.5]})

    def test_validate_param_spec_wrong_length(self):
        """参数范围校验：元素数量错误"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
        )

        with pytest.raises(ValueError, match="2-3 elements"):
            optimizer._validate_param_names({"threshold": (0.1,)})

    def test_validate_param_spec_min_ge_max(self):
        """参数范围校验：min >= max"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
        )

        with pytest.raises(ValueError, match="must be < max"):
            optimizer._validate_param_names({"threshold": (0.5, 0.1)})

    def test_validate_param_spec_invalid_type(self):
        """参数范围校验：无效类型标识"""
        prices = _generate_random_walk(1000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
        )

        with pytest.raises(ValueError, match="'int' or 'log'"):
            optimizer._validate_param_names({"threshold": (0.1, 0.5, "float")})

    def test_optimize_basic(self):
        """基本优化测试（少量试验）"""
        prices = _generate_random_walk(2000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
            window_sizes=(20, 40),  # 使用较小窗口加速测试
            n_top_results=3,
        )

        results = optimizer.optimize(
            n_trials=5,
            n_startup_trials=3,
            show_progress=False,
            threshold=(0.3, 1.0),
        )

        assert isinstance(results, list)
        assert len(results) <= 3
        if results:
            assert isinstance(results[0], TrialResult)
            assert results[0].rank == 1

    def test_optimize_default_startup_trials(self):
        """默认探索配置应可正常运行"""
        prices = _generate_random_walk(1500)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
            window_sizes=(20, 40),
            n_top_results=3,
        )

        results = optimizer.optimize(
            n_trials=3,
            show_progress=False,
            threshold=(0.3, 1.0),
        )

        assert isinstance(results, list)

    def test_optimize_with_int_param(self):
        """带整数参数的优化测试"""
        prices = _generate_random_walk(2000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
            window_sizes=(20, 40),
            n_top_results=3,
        )

        results = optimizer.optimize(
            n_trials=3,
            n_startup_trials=2,
            show_progress=False,
            threshold=(0.3, 1.0),
            max_bars=(100, 500, "int"),
        )

        assert isinstance(results, list)

    def test_optimize_with_log_param(self):
        """带对数尺度参数的优化测试"""
        prices = _generate_random_walk(2000)
        candles = _make_candles(prices)

        optimizer = TrendOptimizer(
            fusion_bar_cls=DemoBar,
            candles=candles,
            window_sizes=(20, 40),
            n_top_results=3,
        )

        results = optimizer.optimize(
            n_trials=3,
            n_startup_trials=2,
            show_progress=False,
            threshold=(0.1, 10.0, "log"),
        )

        assert isinstance(results, list)


class TestTrialResult:
    """TrialResult 测试"""

    def test_dataclass_fields(self):
        """数据类字段测试"""
        result = TrialResult(
            rank=1,
            params={"threshold": 1.0},
            final_score=3.5,
            window_scores={20: 3.0, 40: 3.5, 60: 4.0},
            fusion_bar_count=200,
            constraint_satisfied=True,
            constraint_reason="ok",
        )

        assert result.rank == 1
        assert result.params == {"threshold": 1.0}
        assert result.final_score == 3.5
        assert result.fusion_bar_count == 200
        assert result.constraint_satisfied is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
