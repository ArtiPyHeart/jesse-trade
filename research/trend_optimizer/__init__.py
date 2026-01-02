"""Trend Optimizer 模块

基于 Optuna 的 Fusion Bar 参数优化器，最大化 Hurst/ADF/KPSS 三重验证得分。

Example:
    >>> from research.trend_optimizer import TrendOptimizer
    >>> from src.bars.fusion.demo import DemoBar
    >>>
    >>> optimizer = TrendOptimizer(
    ...     fusion_bar_cls=DemoBar,
    ...     candles=candles_1m,
    ...     window_sizes=(20, 40, 60),
    ... )
    >>> results = optimizer.optimize(
    ...     n_trials=100,
    ...     clip_r=(0.001, 0.01),
    ...     threshold=(0.5, 3.0),
    ... )
    >>> for r in results:
    ...     print(f"Rank {r.rank}: score={r.final_score:.3f}, params={r.params}")
"""

from .evaluator import MultiWindowEvaluator, TrialResult, extract_top_n
from .optimizer import TrendOptimizer

__all__ = [
    "TrendOptimizer",
    "MultiWindowEvaluator",
    "TrialResult",
    "extract_top_n",
]
