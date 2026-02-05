"""Fusion Bar 趋势优化器

使用 Optuna 搜索 Fusion Bar 的最佳参数，最大化 Hurst/ADF/KPSS 三重验证得分。
"""

import inspect
from typing import Any

import numpy as np
import optuna

from src.bars.fusion.base import FusionBarContainerBase

from .evaluator import MultiWindowEvaluator, TrialResult, extract_top_n

# 约束违反时的惩罚值（direction=maximize 时使用负值）
PENALTY_SCORE = -1e6
EXPLORATION_STARTUP_RATIO = 0.99  # 只有最后 1% 用于收束，探索比微调更重要
MIN_STARTUP_TRIALS = 50
MIN_EI_CANDIDATES = 64
MAX_EI_CANDIDATES = 256


def _exploration_gamma(trial_count: int) -> int:
    return min(int(trial_count**0.5), 50)


def _resolve_startup_trials(n_trials: int, n_startup_trials: int | None) -> int:
    if n_startup_trials is None:
        startup_trials = max(
            MIN_STARTUP_TRIALS, int(n_trials * EXPLORATION_STARTUP_RATIO)
        )
    else:
        startup_trials = n_startup_trials
    return min(max(1, startup_trials), n_trials)


def _resolve_ei_candidates(n_trials: int) -> int:
    return max(MIN_EI_CANDIDATES, min(MAX_EI_CANDIDATES, n_trials * 2))


class TrendOptimizer:
    """Fusion Bar 趋势优化器

    通过 Optuna 搜索 Fusion Bar 参数，最大化多窗口三重验证得分。

    Example:
        >>> from src.bars.fusion.demo import DemoBar
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
    """

    def __init__(
        self,
        fusion_bar_cls: type[FusionBarContainerBase],
        candles: np.ndarray,
        window_sizes: tuple[int, ...] = (20, 40, 60),
        step: int = 5,
        min_bar_ratio_minutes: int = 360,  # 6h
        n_top_results: int = 10,
    ):
        """初始化优化器

        Args:
            fusion_bar_cls: FusionBar 类（继承自 FusionBarContainerBase）
            candles: Jesse style 1min candles，shape=(N, 6)
            window_sizes: 评估窗口大小元组
            step: TrendValidator 滑动步长
            min_bar_ratio_minutes: 最小 bar 数量基准（分钟）
            n_top_results: 返回前 N 个结果
        """
        assert candles.ndim == 2, f"candles must be 2D, got {candles.ndim}D"
        assert candles.shape[1] == 6, (
            f"candles must have 6 columns, got {candles.shape[1]}"
        )

        self.fusion_bar_cls = fusion_bar_cls
        self.candles = candles
        self.evaluator = MultiWindowEvaluator(window_sizes, step, min_bar_ratio_minutes)
        self.n_top_results = n_top_results

        # 预计算过滤后的 candles 数量（与 FusionBar 内部一致）
        self._filtered_count = int((candles[:, 5] > 0).sum())

        # 获取 FusionBar 的有效参数名
        sig = inspect.signature(self.fusion_bar_cls.__init__)
        self._valid_params = set(sig.parameters.keys()) - {"self"}

    def optimize(
        self,
        n_trials: int = 100,
        n_startup_trials: int | None = None,
        n_jobs: int = 1,
        show_progress: bool = True,
        **param_ranges: tuple[Any, ...],
    ) -> list[TrialResult]:
        """运行优化

        Args:
            n_trials: 试验次数
            n_startup_trials: 随机采样次数（None 表示按探索比例自动计算）
            n_jobs: 并行优化的 worker 数（>=1）
            show_progress: 显示进度条
            **param_ranges: 参数搜索范围
                - float 参数: param=(min, max) 或 param=(min, max, "log")
                - int 参数: param=(min, max, "int")
                - category 参数: param=([v1, v2, ...], "category")

        Returns:
            Top-N 结果列表，按得分降序排列

        Raises:
            ValueError: 如果参数名不在 FusionBar 构造函数中
        """
        assert n_trials >= 1, f"n_trials must be >= 1, got {n_trials}"
        if n_startup_trials is not None:
            assert n_startup_trials >= 1, (
                f"n_startup_trials must be >= 1, got {n_startup_trials}"
            )
        assert n_jobs >= 1, f"n_jobs must be >= 1, got {n_jobs}"

        # 校验参数名
        self._validate_param_names(param_ranges)

        startup_trials = _resolve_startup_trials(n_trials, n_startup_trials)
        ei_candidates = _resolve_ei_candidates(n_trials)

        # 创建 study（内存存储，direction=maximize）
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=startup_trials,
                n_ei_candidates=ei_candidates,
                gamma=_exploration_gamma,
                multivariate=True,
                group=True,
                constant_liar=True,
                consider_endpoints=True,
                consider_magic_clip=False,
                warn_independent_sampling=False,
            ),
        )

        # 定义目标函数（闭包捕获 param_ranges）
        def objective(trial: optuna.Trial) -> float:
            return self._objective(trial, param_ranges)

        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
            n_jobs=n_jobs,
            catch=(Exception,),
        )

        return extract_top_n(study, self.n_top_results)

    def optimize_and_return_study(
        self,
        n_trials: int = 100,
        n_startup_trials: int | None = None,
        n_jobs: int = 1,
        show_progress: bool = True,
        **param_ranges: tuple[Any, ...],
    ) -> optuna.Study:
        """运行优化并返回 study 对象（用于分层提取）

        与 optimize() 相同，但返回 study 对象而非 top N 列表。
        调用方可使用 extract_top_n_by_tiers() 进行分层提取。

        Args:
            n_trials: 试验次数
            n_startup_trials: 随机采样次数（None 表示按探索比例自动计算）
            n_jobs: 并行优化的 worker 数（>=1）
            show_progress: 显示进度条
            **param_ranges: 参数搜索范围

        Returns:
            Optuna Study 对象
        """
        assert n_trials >= 1, f"n_trials must be >= 1, got {n_trials}"
        if n_startup_trials is not None:
            assert n_startup_trials >= 1, (
                f"n_startup_trials must be >= 1, got {n_startup_trials}"
            )
        assert n_jobs >= 1, f"n_jobs must be >= 1, got {n_jobs}"

        self._validate_param_names(param_ranges)

        startup_trials = _resolve_startup_trials(n_trials, n_startup_trials)
        ei_candidates = _resolve_ei_candidates(n_trials)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=startup_trials,
                n_ei_candidates=ei_candidates,
                gamma=_exploration_gamma,
                multivariate=True,
                group=True,
                constant_liar=True,
                consider_endpoints=True,
                consider_magic_clip=False,
                warn_independent_sampling=False,
            ),
        )

        def objective(trial: optuna.Trial) -> float:
            return self._objective(trial, param_ranges)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
            n_jobs=n_jobs,
            catch=(Exception,),
        )

        return study

    def _validate_param_names(self, param_ranges: dict[str, Any]) -> None:
        """校验参数名和范围格式是否合法（Fail-fast）"""
        for name, spec in param_ranges.items():
            # 校验参数名
            if name not in self._valid_params:
                raise ValueError(
                    f"Unknown param '{name}' for {self.fusion_bar_cls.__name__}. "
                    f"Valid params: {self._valid_params}"
                )

            # 校验 spec 格式
            if not isinstance(spec, tuple):
                raise ValueError(
                    f"Param '{name}' spec must be tuple, got {type(spec).__name__}"
                )

            if len(spec) == 2 and isinstance(spec[1], str) and spec[1] == "category":
                choices = spec[0]
                if not isinstance(choices, (list, tuple)):
                    raise ValueError(
                        f"Param '{name}' category choices must be list or tuple, got {type(choices).__name__}"
                    )
                if len(choices) < 2:
                    raise ValueError(
                        f"Param '{name}' category choices must have >= 2 values"
                    )
                continue

            if len(spec) < 2 or len(spec) > 3:
                raise ValueError(
                    f"Param '{name}' spec must have 2-3 elements, got {len(spec)}"
                )
            if not isinstance(spec[0], (int, float)) or not isinstance(
                spec[1], (int, float)
            ):
                raise ValueError(
                    f"Param '{name}' min/max must be numeric, got {type(spec[0]).__name__}, {type(spec[1]).__name__}"
                )
            if spec[0] >= spec[1]:
                raise ValueError(
                    f"Param '{name}' min({spec[0]}) must be < max({spec[1]})"
                )
            if len(spec) == 3 and spec[2] not in ("int", "log"):
                raise ValueError(
                    f"Param '{name}' type must be 'int' or 'log', got '{spec[2]}'"
                )

    def _objective(
        self,
        trial: optuna.Trial,
        param_ranges: dict[str, tuple],
    ) -> float:
        """Optuna 目标函数"""
        # 1. 采样参数
        params = {}
        for name, spec in param_ranges.items():
            if len(spec) == 2 and isinstance(spec[1], str) and spec[1] == "category":
                params[name] = trial.suggest_categorical(name, list(spec[0]))
            elif len(spec) == 3 and spec[2] == "int":
                params[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
            elif len(spec) == 3 and spec[2] == "log":
                params[name] = trial.suggest_float(name, spec[0], spec[1], log=True)
            else:
                params[name] = trial.suggest_float(name, spec[0], spec[1])

        # 2. 创建 FusionBar 并生成 bars
        try:
            container = self.fusion_bar_cls(**params)
            container.update_with_candles(self.candles)
            fusion_bars = container.get_fusion_bars()
        except Exception as e:
            trial.set_user_attr("error", str(e))
            trial.set_user_attr("constraint_satisfied", False)
            trial.set_user_attr("constraint_reason", f"exception: {e}")
            return PENALTY_SCORE

        fusion_bar_count = 0 if fusion_bars is None else len(fusion_bars)
        print(f"\ntrial={trial.number} bars={fusion_bar_count}", flush=True)

        # 3. 检查 bars 是否有效
        if fusion_bar_count == 0:
            trial.set_user_attr("constraint_reason", "no bars generated")
            trial.set_user_attr("constraint_satisfied", False)
            trial.set_user_attr("fusion_bar_count", 0)
            return PENALTY_SCORE

        trial.set_user_attr("fusion_bar_count", fusion_bar_count)

        # 4. 检查约束
        satisfied, reason = self.evaluator.check_constraints(
            fusion_bar_count, self._filtered_count
        )
        trial.set_user_attr("constraint_satisfied", satisfied)
        trial.set_user_attr("constraint_reason", reason)

        if not satisfied:
            return PENALTY_SCORE

        # 5. 多窗口评估（捕获评估异常）
        try:
            eval_result = self.evaluator.evaluate(fusion_bars)
        except Exception as e:
            trial.set_user_attr("error", f"evaluation: {e}")
            trial.set_user_attr("constraint_satisfied", False)
            trial.set_user_attr("constraint_reason", f"evaluation exception: {e}")
            return PENALTY_SCORE

        trial.set_user_attr("window_scores", eval_result["window_scores"])

        return eval_result["final_score"]
