"""多窗口评估器和结果类

提供 Fusion Bar 的多窗口三重验证评估能力。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from research.hurst_adf_kpss import TrendValidator

if TYPE_CHECKING:
    import optuna


@dataclass
class EvaluationReport:
    """Fusion Bar 综合评估报告

    提供多维度、多角度的评估结果，便于直观理解 Fusion Bar 的质量。
    """

    # ========== 核心评分 ==========
    overall_score: float  # 综合评分 (0-100)，越高越好
    overall_grade: str  # 等级: A/B/C/D/F

    # ========== 分项得分 (0-5) ==========
    trend_score: float  # 趋势性得分 (各窗口 mean_score 的均值)
    consistency_score: float  # 跨窗口一致性得分
    stability_score: float  # 高分稳定性得分

    # ========== 窗口级统计 ==========
    window_scores: dict[int, float]  # {窗口大小: 平均得分}
    window_details: dict[int, dict]  # {窗口大小: 详细统计}

    # ========== 汇总统计 ==========
    bar_count: int  # Fusion Bar 数量
    total_windows_evaluated: int  # 评估的滑动窗口总数
    high_score_ratio: float  # 高分 (5分) 窗口占比
    low_score_ratio: float  # 低分 (≤2分) 窗口占比

    # ========== Hurst/ADF/KPSS 汇总 ==========
    mean_hurst: float  # 平均 Hurst 指数
    hurst_above_055_ratio: float  # Hurst > 0.55 的比例
    adf_nonstationary_ratio: float  # ADF 非平稳 (p>0.05) 的比例
    kpss_nonstationary_ratio: float  # KPSS 非平稳 (p<0.05) 的比例
    triple_consensus_ratio: float  # 三重共识的比例

    # ========== 原始数据 ==========
    raw_results: dict = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        """格式化输出评估报告"""
        lines = [
            "=" * 60,
            "Fusion Bar 评估报告",
            "=" * 60,
            "",
            f"【综合评分】{self.overall_score:.1f}/100 ({self.overall_grade})",
            "",
            "【分项得分】(0-5)",
            f"  趋势性:     {self.trend_score:.2f}",
            f"  一致性:     {self.consistency_score:.2f}",
            f"  稳定性:     {self.stability_score:.2f}",
            "",
            "【窗口得分】",
        ]
        for ws, score in sorted(self.window_scores.items()):
            lines.append(f"  窗口 {ws:3d}: {score:.2f}")

        lines.extend(
            [
                "",
                "【统计概览】",
                f"  Bar 数量:        {self.bar_count}",
                f"  评估窗口数:      {self.total_windows_evaluated}",
                f"  高分(5分)占比:   {self.high_score_ratio:.1%}",
                f"  低分(≤2分)占比:  {self.low_score_ratio:.1%}",
                "",
                "【三重检验统计】",
                f"  平均 Hurst:      {self.mean_hurst:.3f}",
                f"  Hurst>0.55:      {self.hurst_above_055_ratio:.1%}",
                f"  ADF 非平稳:      {self.adf_nonstationary_ratio:.1%}",
                f"  KPSS 非平稳:     {self.kpss_nonstationary_ratio:.1%}",
                f"  三重共识:        {self.triple_consensus_ratio:.1%}",
                "=" * 60,
            ]
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转换为字典（不含 raw_results）"""
        return {
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "trend_score": self.trend_score,
            "consistency_score": self.consistency_score,
            "stability_score": self.stability_score,
            "window_scores": self.window_scores,
            "bar_count": self.bar_count,
            "total_windows_evaluated": self.total_windows_evaluated,
            "high_score_ratio": self.high_score_ratio,
            "low_score_ratio": self.low_score_ratio,
            "mean_hurst": self.mean_hurst,
            "hurst_above_055_ratio": self.hurst_above_055_ratio,
            "adf_nonstationary_ratio": self.adf_nonstationary_ratio,
            "kpss_nonstationary_ratio": self.kpss_nonstationary_ratio,
            "triple_consensus_ratio": self.triple_consensus_ratio,
        }


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

    def evaluate_detailed(self, fusion_bars: np.ndarray) -> EvaluationReport:
        """生成详细的评估报告

        Args:
            fusion_bars: Jesse style 6 列 fusion bars 数组

        Returns:
            EvaluationReport 对象，包含多维度统计和综合评分
        """
        window_scores: dict[int, float] = {}
        window_details: dict[int, dict] = {}
        all_results: list[pd.DataFrame] = []

        # 收集各窗口的评估结果
        for ws in self.window_sizes:
            validator = TrendValidator(window_size=ws, step=self.step)
            results = validator.validate(fusion_bars)
            summary = validator.summarize(results)

            window_scores[ws] = summary["mean_score"]
            window_details[ws] = summary
            all_results.append(results)

        # 合并所有窗口的原始结果
        combined = pd.concat(all_results, ignore_index=True)

        # ========== 计算分项得分 ==========
        scores_list = list(window_scores.values())
        scores_arr = np.asarray(scores_list, dtype=float)

        # 趋势性得分: 各窗口 mean_score 的均值 (0-5)
        trend_score = float(np.mean(scores_arr))

        # 一致性得分: 基于窗口得分的标准差，归一化到 [0, 5]
        # 使用理论最大标准差进行归一化 (二值分布的最大 std)
        n = scores_arr.size
        if n <= 1:
            consistency_score = 5.0
        else:
            window_std = float(np.std(scores_arr, ddof=0))
            p = (n // 2) / n
            max_std = 5.0 * np.sqrt(p * (1.0 - p))  # 理论最大标准差
            normalized_std = 0.0 if max_std == 0.0 else window_std / max_std
            consistency_score = float(5.0 * (1.0 - np.clip(normalized_std, 0.0, 1.0)))

        # 稳定性得分: 基于高分比例和低分比例，线性缩放到 [0, 5]
        high_ratio = float((combined["score"] == 5).sum() / len(combined))
        low_ratio = float((combined["score"] <= 2).sum() / len(combined))
        stability_raw = 5.0 * high_ratio + 2.0 * (1.0 - low_ratio)  # 原始值范围 [0, 7]
        stability_score = float(5.0 * stability_raw / 7.0)  # 缩放到 [0, 5]

        # ========== 计算综合评分 (0-100) ==========
        # 权重: 趋势性 50%, 一致性 30%, 稳定性 20%
        # 公式: trend*10 + consistency*6 + stability*4, 满分 = 50+30+20 = 100
        overall_score = float(
            np.clip(
                trend_score * 10.0 + consistency_score * 6.0 + stability_score * 4.0,
                0.0,
                100.0,
            )
        )

        # 等级划分
        if overall_score >= 80:
            overall_grade = "A"
        elif overall_score >= 65:
            overall_grade = "B"
        elif overall_score >= 50:
            overall_grade = "C"
        elif overall_score >= 35:
            overall_grade = "D"
        else:
            overall_grade = "F"

        # ========== 三重检验统计 ==========
        valid_hurst = combined["hurst"].dropna()
        mean_hurst = float(valid_hurst.mean()) if len(valid_hurst) > 0 else 0.0
        hurst_above_055_ratio = (
            float((valid_hurst > 0.55).sum() / len(valid_hurst))
            if len(valid_hurst) > 0
            else 0.0
        )

        valid_adf = combined["adf_pvalue"].dropna()
        adf_nonstationary_ratio = (
            float((valid_adf > 0.05).sum() / len(valid_adf))
            if len(valid_adf) > 0
            else 0.0
        )

        valid_kpss = combined["kpss_pvalue"].dropna()
        kpss_nonstationary_ratio = (
            float((valid_kpss < 0.05).sum() / len(valid_kpss))
            if len(valid_kpss) > 0
            else 0.0
        )

        # 三重共识: Hurst > 0.55 且 ADF p > 0.05 且 KPSS p < 0.05
        triple_mask = (
            (combined["hurst"] > 0.55)
            & (combined["adf_pvalue"] > 0.05)
            & (combined["kpss_pvalue"] < 0.05)
        )
        triple_consensus_ratio = float(triple_mask.sum() / len(combined))

        return EvaluationReport(
            # 核心评分
            overall_score=overall_score,
            overall_grade=overall_grade,
            # 分项得分
            trend_score=trend_score,
            consistency_score=consistency_score,
            stability_score=stability_score,
            # 窗口级统计
            window_scores=window_scores,
            window_details=window_details,
            # 汇总统计
            bar_count=len(fusion_bars),
            total_windows_evaluated=len(combined),
            high_score_ratio=high_ratio,
            low_score_ratio=low_ratio,
            # 三重检验统计
            mean_hurst=mean_hurst,
            hurst_above_055_ratio=hurst_above_055_ratio,
            adf_nonstationary_ratio=adf_nonstationary_ratio,
            kpss_nonstationary_ratio=kpss_nonstationary_ratio,
            triple_consensus_ratio=triple_consensus_ratio,
            # 原始数据
            raw_results={
                "combined_df": combined,
                "window_results": all_results,
            },
        )


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
