"""多窗口评估器和结果类

提供 Fusion Bar 的多窗口三重验证评估能力。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from research.hurst_adf_kpss import TrendValidator

if TYPE_CHECKING:
    import optuna


@dataclass
class TrialResult:
    """单次试验结果（完整信息）"""

    rank: int
    params: dict[str, float]
    final_score: float
    window_scores: dict[int, float]  # {20: 3.2, 40: 3.5, 60: 3.8}
    fusion_bar_count: int
    constraint_satisfied: bool
    constraint_reason: str


class MultiWindowEvaluator:
    """多窗口三重验证评估器

    在多个窗口大小上分别运行 TrendValidator，并等权聚合得分。
    """

    def __init__(
        self,
        window_sizes: tuple[int, ...] = (20, 40, 60),
        step: int = 5,
        min_bar_ratio_minutes: int = 360,  # 6h = 360 min
    ):
        """初始化评估器

        Args:
            window_sizes: 评估窗口大小元组
            step: TrendValidator 滑动步长
            min_bar_ratio_minutes: 最小 bar 数量基准（分钟）
        """
        assert len(window_sizes) > 0, "window_sizes must not be empty"
        assert all(ws >= 20 for ws in window_sizes), "window_size must be >= 20"
        assert step >= 1, "step must be >= 1"
        assert min_bar_ratio_minutes > 0, "min_bar_ratio_minutes must be > 0"

        self.window_sizes = window_sizes
        self.step = step
        self.min_bar_ratio_minutes = min_bar_ratio_minutes

    def check_constraints(
        self,
        fusion_bar_count: int,
        filtered_candle_count: int,
    ) -> tuple[bool, str]:
        """检查约束条件

        Args:
            fusion_bar_count: 生成的 fusion bar 数量
            filtered_candle_count: 过滤后的 1min candles 数量

        Returns:
            (is_satisfied, reason) 元组
        """
        # 约束1: fusion bar >= max(window_sizes)
        max_window = max(self.window_sizes)
        if fusion_bar_count < max_window:
            return False, f"bars({fusion_bar_count}) < max_window({max_window})"

        # 约束2: fusion bar > 6h 基准
        min_bars = filtered_candle_count // self.min_bar_ratio_minutes
        if fusion_bar_count <= min_bars:
            return False, f"bars({fusion_bar_count}) <= 6h_baseline({min_bars})"

        return True, "ok"

    def evaluate(self, fusion_bars: np.ndarray) -> dict:
        """在多个窗口上评估 fusion bars（等权聚合）

        Args:
            fusion_bars: Jesse style 6 列 fusion bars 数组

        Returns:
            {
                'final_score': float,
                'window_scores': {20: float, 40: float, 60: float},
                'window_summaries': {20: dict, 40: dict, 60: dict},
            }
        """
        window_scores = {}
        window_summaries = {}

        for ws in self.window_sizes:
            validator = TrendValidator(window_size=ws, step=self.step)
            results = validator.validate(fusion_bars)
            summary = validator.summarize(results)
            window_scores[ws] = summary["mean_score"]
            window_summaries[ws] = summary

        # 等权聚合
        final_score = float(np.mean(list(window_scores.values())))

        return {
            "final_score": final_score,
            "window_scores": window_scores,
            "window_summaries": window_summaries,
        }


def extract_top_n(
    study: "optuna.Study",
    n_top: int = 10,
    only_valid: bool = True,
) -> list[TrialResult]:
    """从 study 提取 Top-N 结果

    Args:
        study: Optuna study 对象
        n_top: 返回前 N 个结果
        only_valid: 是否只返回约束满足的结果（默认 True）

    Returns:
        按得分降序排列的 TrialResult 列表
    """
    import optuna

    # 筛选已完成的试验
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # 如果只要有效结果，过滤掉约束不满足的
    if only_valid:
        completed = [
            t for t in completed if t.user_attrs.get("constraint_satisfied", False)
        ]

    # 按得分降序排列
    sorted_trials = sorted(
        completed,
        key=lambda t: t.value if t.value is not None else -1e10,
        reverse=True,
    )

    results = []
    for rank, trial in enumerate(sorted_trials[:n_top], start=1):
        results.append(
            TrialResult(
                rank=rank,
                params=dict(trial.params),
                final_score=trial.value if trial.value is not None else 0.0,
                window_scores=trial.user_attrs.get("window_scores", {}),
                fusion_bar_count=trial.user_attrs.get("fusion_bar_count", 0),
                constraint_satisfied=trial.user_attrs.get(
                    "constraint_satisfied", False
                ),
                constraint_reason=trial.user_attrs.get("constraint_reason", ""),
            )
        )

    return results
