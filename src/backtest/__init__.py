"""
回测分析模块

提供金融指标计算和可视化功能。
"""

from .metrics import (
    BacktestMetrics,
    BenchmarkMetrics,
    DrawdownMetrics,
    ReturnMetrics,
    RiskMetrics,
    TradeMetrics,
    calculate_all_metrics,
    calculate_benchmark_metrics,
    calculate_drawdown_metrics,
    calculate_returns,
    calculate_risk_metrics,
    calculate_trade_metrics,
)
from .visualization import (
    plot_backtest_results,
    print_metrics_report,
    save_metrics_json,
    save_metrics_report,
)

__all__ = [
    # Metrics
    "BacktestMetrics",
    "ReturnMetrics",
    "TradeMetrics",
    "DrawdownMetrics",
    "RiskMetrics",
    "BenchmarkMetrics",
    "calculate_all_metrics",
    "calculate_returns",
    "calculate_trade_metrics",
    "calculate_drawdown_metrics",
    "calculate_risk_metrics",
    "calculate_benchmark_metrics",
    # Visualization
    "plot_backtest_results",
    "print_metrics_report",
    "save_metrics_json",
    "save_metrics_report",
]
