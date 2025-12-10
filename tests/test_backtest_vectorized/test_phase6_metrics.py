"""
Phase 6: 指标计算测试

测试目标:
- total_return 计算
- max_drawdown 使用 cummax 计算
- sharpe_ratio 年化因子计算
- win_rate 计算
- calmar_ratio 计算
- profit_factor 计算

运行方式:
    pytest tests/test_backtest_vectorized/test_phase6_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 导入被测类
from backtest_vectorized_no_jesse import (
    BacktestAnalyzer,
    EquityPoint,
    Trade,
)


class TestTotalReturnCalculation:
    """测试总收益率计算"""

    def test_positive_return(self):
        """测试正收益"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=11000, benchmark_equity=10500),
            EquityPoint(timestamp=3000, equity=12000, benchmark_equity=11000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # total_return = (12000 - 10000) / 10000 = 0.2
        assert metrics["total_return"] == pytest.approx(0.2)
        assert metrics["total_return_pct"] == "20.00%"

    def test_negative_return(self):
        """测试负收益"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=9000, benchmark_equity=9500),
            EquityPoint(timestamp=3000, equity=8000, benchmark_equity=9000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # total_return = (8000 - 10000) / 10000 = -0.2
        assert metrics["total_return"] == pytest.approx(-0.2)
        assert metrics["total_return_pct"] == "-20.00%"

    def test_zero_return(self):
        """测试零收益"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=11000, benchmark_equity=10000),
            EquityPoint(timestamp=3000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        assert metrics["total_return"] == pytest.approx(0.0)


class TestMaxDrawdownCalculation:
    """测试最大回撤计算"""

    def test_simple_drawdown(self):
        """测试简单回撤"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=12000, benchmark_equity=10000),  # 峰值
            EquityPoint(timestamp=3000, equity=11000, benchmark_equity=10000),
            EquityPoint(timestamp=4000, equity=10000, benchmark_equity=10000),  # 谷底
            EquityPoint(timestamp=5000, equity=11000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # max_drawdown = (10000 - 12000) / 12000 = -0.1667
        assert metrics["max_drawdown"] == pytest.approx(-0.1667, rel=1e-3)

    def test_multiple_drawdowns(self):
        """测试多次回撤取最大"""
        starting_balance = 100
        equity_curve = [
            EquityPoint(timestamp=1000, equity=100, benchmark_equity=100),
            EquityPoint(timestamp=2000, equity=120, benchmark_equity=100),  # 第一个峰值
            EquityPoint(timestamp=3000, equity=110, benchmark_equity=100),  # -8.3%
            EquityPoint(timestamp=4000, equity=130, benchmark_equity=100),  # 第二个峰值
            EquityPoint(timestamp=5000, equity=100, benchmark_equity=100),  # -23%
            EquityPoint(timestamp=6000, equity=140, benchmark_equity=100),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # 最大回撤: (100 - 130) / 130 = -0.2308
        assert metrics["max_drawdown"] == pytest.approx(-0.2308, rel=1e-3)

    def test_no_drawdown(self):
        """测试无回撤（持续上涨）"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=11000, benchmark_equity=10000),
            EquityPoint(timestamp=3000, equity=12000, benchmark_equity=10000),
            EquityPoint(timestamp=4000, equity=13000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # 无回撤，应该是 0
        assert metrics["max_drawdown"] == pytest.approx(0.0)

    def test_drawdown_formula(self):
        """验证回撤公式: drawdown = (equity - peak) / peak"""
        equity = np.array([100, 120, 110, 130, 100, 140])
        peak = np.maximum.accumulate(equity)  # [100, 120, 120, 130, 130, 140]
        drawdown = (equity - peak) / peak  # [0, 0, -0.083, 0, -0.231, 0]
        max_drawdown = np.min(drawdown)

        assert peak.tolist() == [100, 120, 120, 130, 130, 140]
        assert max_drawdown == pytest.approx(-0.2308, rel=1e-3)


class TestSharpeRatioCalculation:
    """测试夏普比率计算"""

    def test_sharpe_formula(self):
        """验证夏普比率公式: sharpe = mean(returns) / std(returns) * sqrt(samples_per_year)"""
        starting_balance = 10000

        # 创建已知收益率序列
        # 假设每 1 秒采样一次 (用于简化计算)
        equity_values = [10000, 10100, 10200, 10150, 10300, 10400]
        timestamps = [1000, 2000, 3000, 4000, 5000, 6000]  # 每 1 秒

        equity_curve = [
            EquityPoint(timestamp=t, equity=e, benchmark_equity=10000)
            for t, e in zip(timestamps, equity_values)
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # 手动计算
        returns = np.diff(equity_values) / np.array(equity_values[:-1])
        avg_interval_ms = 1000  # 1 秒
        ms_per_year = 365 * 24 * 60 * 60 * 1000
        samples_per_year = ms_per_year / avg_interval_ms
        annualization_factor = np.sqrt(samples_per_year)

        expected_sharpe = np.mean(returns) / np.std(returns) * annualization_factor

        assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-3)

    def test_sharpe_with_negative_returns(self):
        """测试负收益的夏普比率"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=9800, benchmark_equity=10000),
            EquityPoint(timestamp=3000, equity=9600, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # 负收益 → 夏普比率应为负
        assert metrics["sharpe_ratio"] < 0

    def test_sharpe_with_zero_std(self):
        """测试收益率标准差为零时夏普比率为 0"""
        starting_balance = 10000
        # 所有权益相同
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=3000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        assert metrics["sharpe_ratio"] == 0.0


class TestWinRateCalculation:
    """测试胜率计算"""

    def test_win_rate_50_percent(self):
        """测试 50% 胜率"""
        trades = [
            Trade(timestamp=1000, action="open_long", price=50000, qty=0.1, fee=2),
            Trade(timestamp=2000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100),
            Trade(timestamp=3000, action="open_short", price=51000, qty=0.1, fee=2),
            Trade(timestamp=4000, action="close_short", price=52000, qty=0.1, fee=2, pnl=-100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=4000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        # 1 盈利 + 1 亏损 = 50%
        assert metrics["win_rate"] == pytest.approx(0.5)
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["total_trades"] == 2

    def test_win_rate_100_percent(self):
        """测试 100% 胜率"""
        trades = [
            Trade(timestamp=1000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100),
            Trade(timestamp=2000, action="close_short", price=49000, qty=0.1, fee=2, pnl=100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        assert metrics["win_rate"] == pytest.approx(1.0)
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 0

    def test_win_rate_0_percent(self):
        """测试 0% 胜率"""
        trades = [
            Trade(timestamp=1000, action="close_long", price=49000, qty=0.1, fee=2, pnl=-100),
            Trade(timestamp=2000, action="close_short", price=51000, qty=0.1, fee=2, pnl=-100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        assert metrics["win_rate"] == pytest.approx(0.0)
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 2

    def test_win_rate_excludes_open_trades(self):
        """测试胜率计算排除开仓交易"""
        trades = [
            Trade(timestamp=1000, action="open_long", price=50000, qty=0.1, fee=2, pnl=0),
            Trade(timestamp=2000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        # 只有 1 笔平仓交易（盈利）
        assert metrics["total_trades"] == 1
        assert metrics["win_rate"] == pytest.approx(1.0)


class TestCalmarRatioCalculation:
    """测试卡尔玛比率计算"""

    def test_calmar_ratio(self):
        """测试卡尔玛比率: total_return / |max_drawdown|"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=12000, benchmark_equity=10000),  # 峰值
            EquityPoint(timestamp=3000, equity=10000, benchmark_equity=10000),  # 回撤 -16.67%
            EquityPoint(timestamp=4000, equity=15000, benchmark_equity=10000),  # 收益 50%
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # total_return = 0.5, max_drawdown = -0.1667
        # calmar = 0.5 / 0.1667 = 3.0
        expected_calmar = 0.5 / 0.1667
        assert metrics["calmar_ratio"] == pytest.approx(expected_calmar, rel=1e-2)

    def test_calmar_zero_when_no_drawdown(self):
        """测试无回撤时卡尔玛比率为 0"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=11000, benchmark_equity=10000),
            EquityPoint(timestamp=3000, equity=12000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # max_drawdown = 0, calmar = 0
        assert metrics["calmar_ratio"] == 0.0


class TestProfitFactorCalculation:
    """测试盈亏比计算"""

    def test_profit_factor(self):
        """测试盈亏比: sum(wins) / |sum(losses)|"""
        trades = [
            Trade(timestamp=1000, action="close_long", price=51000, qty=0.1, fee=2, pnl=200),
            Trade(timestamp=2000, action="close_short", price=52000, qty=0.1, fee=2, pnl=-100),
            Trade(timestamp=3000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        # sum(wins) = 200 + 100 = 300
        # sum(losses) = -100
        # profit_factor = 300 / 100 = 3.0
        assert metrics["profit_factor"] == pytest.approx(3.0)

    def test_profit_factor_no_losses(self):
        """测试无亏损时盈亏比为 0"""
        trades = [
            Trade(timestamp=1000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100),
            Trade(timestamp=2000, action="close_short", price=49000, qty=0.1, fee=2, pnl=100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        # 无亏损，profit_factor = 0
        assert metrics["profit_factor"] == 0.0


class TestAvgWinLossCalculation:
    """测试平均盈亏计算"""

    def test_avg_win_and_loss(self):
        """测试平均盈亏"""
        trades = [
            Trade(timestamp=1000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100),
            Trade(timestamp=2000, action="close_short", price=52000, qty=0.1, fee=2, pnl=-50),
            Trade(timestamp=3000, action="close_long", price=51000, qty=0.1, fee=2, pnl=200),
            Trade(timestamp=4000, action="close_short", price=52000, qty=0.1, fee=2, pnl=-100),
        ]

        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        # avg_win = (100 + 200) / 2 = 150
        # avg_loss = (-50 + -100) / 2 = -75
        assert metrics["avg_win"] == pytest.approx(150.0)
        assert metrics["avg_loss"] == pytest.approx(-75.0)


class TestBenchmarkReturnCalculation:
    """测试基准收益率计算"""

    def test_benchmark_return(self):
        """测试基准收益率"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=11000, benchmark_equity=11500),
            EquityPoint(timestamp=3000, equity=12000, benchmark_equity=13000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        # benchmark_return = (13000 - 10000) / 10000 = 0.3
        assert metrics["benchmark_return"] == pytest.approx(0.3)
        assert metrics["benchmark_return_pct"] == "30.00%"


class TestEmptyData:
    """测试边界条件"""

    def test_empty_equity_curve(self):
        """测试空权益曲线返回空字典"""
        metrics = BacktestAnalyzer.calculate_metrics([], [], 10000)
        assert metrics == {}

    def test_single_equity_point(self):
        """测试单个权益点"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        assert metrics["total_return"] == pytest.approx(0.0)
        assert metrics["max_drawdown"] == pytest.approx(0.0)

    def test_no_trades(self):
        """测试无交易记录"""
        starting_balance = 10000
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=10000, benchmark_equity=10000),
        ]

        metrics = BacktestAnalyzer.calculate_metrics([], equity_curve, starting_balance)

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
