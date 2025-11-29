import numpy as np

from src.bars.fusion.base import FusionBarContainerBase


class DemoBar(FusionBarContainerBase):
    """
    abs(close - close_lag1) * (high - low) / close

    Parameters:
    -----------
    clip_r : float
        小于此阈值的波动将被压缩为0，用于过滤噪声。默认为0（不过滤）。
    max_bars : int
        最大bar数量，-1表示不限制。
    threshold : float
        累积阈值，达到此值时生成新bar。
    """

    def __init__(
        self,
        clip_r: float = 0.012,
        max_bars: int = -1,
        threshold: float = 1.399,
    ):
        super().__init__(max_bars, threshold)
        self.clip_r = clip_r

    @property
    def max_lookback(self) -> int:
        return 1

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        close_arr = candles[:, 2]
        high_arr = candles[:, 3]
        low_arr = candles[:, 4]
        res = (
            np.abs(close_arr[1:] - close_arr[:-1])
            * (high_arr[1:] - low_arr[1:])
            / close_arr[1:]
        )
        # 根据 clip_r 过滤：小于 clip_r 的值设为 0
        if self.clip_r > 0:
            res[res < self.clip_r] = 0
        return res


if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    import optuna

    candles = np.load("../../../data/btc_1m.npy")
    print(f"加载了 {len(candles)} 根1分钟K线")
    print(f"4小时K线理论数量: {len(candles) // 240}")

    def objective(trial):
        bar_container = DemoBar(max_bars=-1)
        bar_container.THRESHOLD = trial.suggest_float("THRESHOLD", 0.2, 10)
        bar_container.update_with_candles(candles)
        fusion_bar = bar_container.get_fusion_bars()

        # 计算4小时K线的理论数量
        # 1分钟K线转换为4小时K线，4小时 = 240分钟
        four_hour_candle_count = len(candles) // 240

        # 检查fusion bar数量是否少于4小时K线数量
        fusion_bar_count = len(fusion_bar)
        if fusion_bar_count < four_hour_candle_count:
            # 返回一个极大的kurtosis值作为惩罚
            return 1e10

        # 求lag 4的log return
        ret = np.log(fusion_bar[5:, 2]) - np.log(fusion_bar[:-5, 2])
        standard = (ret - ret.mean()) / ret.std()
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")

        # 存储额外信息供最终结果使用
        trial.set_user_attr("fusion_bar_count", fusion_bar_count)
        trial.set_user_attr("four_hour_candle_count", four_hour_candle_count)
        trial.set_user_attr("kurtosis", kurtosis)

        return kurtosis

    # 创建进度回调函数
    def show_progress(study, trial):
        n = trial.number
        if n > 0 and n % 100 == 0:
            # 每100次试验输出一次中间结果
            valid_trials = [t for t in study.trials[:n] if t.value < 1e10]
            if valid_trials:
                current_best = min(t.value for t in valid_trials)
                filtered_count = len([t for t in study.trials[:n] if t.value >= 1e10])
                print(
                    f"\n[进度 {n}/1000] 当前最佳Kurtosis: {current_best:.6f}, "
                    f"有效试验: {len(valid_trials)}, 被过滤: {filtered_count}"
                )

                # 显示参数分布情况
                thresholds = [t.params["THRESHOLD"] for t in valid_trials]
                if thresholds:
                    print(
                        f"  参数分布 - Min: {min(thresholds):.3f}, "
                        f"Max: {max(thresholds):.3f}, "
                        f"Mean: {np.mean(thresholds):.3f}, "
                        f"Std: {np.std(thresholds):.3f}"
                    )

    # 使用更具探索性的配置
    print("\n开始参数优化（增强探索性）...")
    print("=" * 60)

    # 创建多种sampler策略组合以增强探索性
    study = optuna.create_study(
        direction="minimize",
        study_name="fusion_bar_optimization",
        # 使用更温和的剪枝策略，允许更多探索
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=200,  # 前200次试验不剪枝
            n_warmup_steps=10,  # 每个试验前10步不剪枝
            interval_steps=1,
        ),
        # 增强探索性的采样器配置
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=300,  # 前300次使用随机搜索以充分探索
            n_ei_candidates=100,  # 增加EI候选数量以考虑更多可能性
            gamma=lambda x: min(int(x**0.5), 25),  # 自定义gamma函数控制探索
            seed=42,  # 设置随机种子以确保可重复性
        ),
    )

    # 可选：添加手动探索点以确保覆盖整个搜索空间
    for threshold in [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        study.enqueue_trial({"THRESHOLD": threshold})

    print(f"已添加12个手动探索点以确保搜索空间覆盖")
    print(f"搜索范围: THRESHOLD ∈ [0.2, 10.0]")
    print(f"优化目标: 最小化Kurtosis（峰度）")
    print(f"约束条件: Fusion Bar数量 ≥ 4小时K线数量")
    print("=" * 60)

    # 运行优化，增加到1000次试验
    study.optimize(
        objective,
        n_trials=1000,
        callbacks=[show_progress],
        n_jobs=1,  # 单线程以保证探索的连续性
        show_progress_bar=True,  # 显示进度条
    )

    # 打印详细的最终结果
    print("\n" + "=" * 60)
    print("参数优化完成！最终结果：")
    print("=" * 60)

    best_trial = study.best_trial

    print(f"\n最优参数：")
    print(f"  - THRESHOLD: {best_trial.params['THRESHOLD']:.6f}")
    print(f"  - Kurtosis: {best_trial.value:.6f}")

    # 获取最优参数下的详细信息
    print(f"\n最优参数下的统计信息：")
    print(f"  - Fusion Bar数量: {best_trial.user_attrs.get('fusion_bar_count', 'N/A')}")
    print(
        f"  - 4小时K线理论数量: {best_trial.user_attrs.get('four_hour_candle_count', 'N/A')}"
    )
    print(
        f"  - 压缩比率: {best_trial.user_attrs.get('fusion_bar_count', 0) / len(candles):.4%}"
    )

    # 重新计算最优参数下的fusion bar以获取更多统计信息
    print(f"\n重新计算最优参数下的Fusion Bar详细统计...")
    best_bar_container = DemoBar(max_bars=-1)
    best_bar_container.THRESHOLD = best_trial.params["THRESHOLD"]
    best_bar_container.update_with_candles(candles)
    best_fusion_bar = best_bar_container.get_fusion_bars()

    # 计算fusion bar的时间间隔统计
    if len(best_fusion_bar) > 1:
        time_intervals = np.diff(best_fusion_bar[:, 0]) / (60 * 1000)  # 转换为分钟
        print(f"\nFusion Bar时间间隔统计（分钟）：")
        print(f"  - 平均间隔: {np.mean(time_intervals):.2f}")
        print(f"  - 中位数间隔: {np.median(time_intervals):.2f}")
        print(f"  - 最小间隔: {np.min(time_intervals):.2f}")
        print(f"  - 最大间隔: {np.max(time_intervals):.2f}")
        print(f"  - 标准差: {np.std(time_intervals):.2f}")

    # 计算收益率统计
    ret = np.log(best_fusion_bar[4:, 2]) - np.log(best_fusion_bar[:-4, 2])
    standard = (ret - ret.mean()) / ret.std()
    print(f"\nLag收益率统计：")
    print(f"  - 平均收益率: {ret.mean():.6f}")
    print(f"  - 收益率标准差: {ret.std():.6f}")
    print(
        f"  - 夏普比率（年化）: {ret.mean() / ret.std() * np.sqrt(252 * 24 * 60 / 4):.4f}"
    )
    print(
        f"  - 峰度 (Kurtosis): {stats.kurtosis(standard, axis=None, fisher=False, nan_policy='omit'):.6f}"
    )
    print(
        f"  - 偏度 (Skewness): {stats.skew(standard, axis=None, nan_policy='omit'):.6f}"
    )

    # 显示优化过程统计
    print(f"\n优化过程统计：")
    print(f"  - 总试验次数: {len(study.trials)}")
    print(f"  - 有效试验次数: {len([t for t in study.trials if t.value < 1e10])}")
    print(
        f"  - 被约束条件过滤的试验: {len([t for t in study.trials if t.value >= 1e10])}"
    )

    # 显示前10个最优参数（增加到10个以看到更多可能性）
    print(f"\n前10个最优参数组合：")
    sorted_trials = sorted(
        [t for t in study.trials if t.value < 1e10], key=lambda x: x.value
    )[:10]
    for i, trial in enumerate(sorted_trials, 1):
        print(
            f"  {i:2d}. THRESHOLD={trial.params['THRESHOLD']:.6f}, "
            f"Kurtosis={trial.value:.6f}, "
            f"FusionBars={trial.user_attrs.get('fusion_bar_count', 'N/A')}"
        )

    # 添加参数探索分布分析
    print(f"\n参数探索分布分析：")
    valid_trials = [t for t in study.trials if t.value < 1e10]
    if valid_trials:
        all_thresholds = [t.params["THRESHOLD"] for t in valid_trials]

        # 将搜索空间分为10个区间进行统计
        bins = np.linspace(0.2, 10.0, 11)
        hist, _ = np.histogram(all_thresholds, bins=bins)

        print(f"  搜索空间覆盖率（按区间）：")
        for i in range(len(bins) - 1):
            bar_length = int(hist[i] * 50 / max(hist))  # 归一化到50字符宽度
            bar = "█" * bar_length
            print(f"    [{bins[i]:4.1f}-{bins[i+1]:4.1f}]: {bar} ({hist[i]:3d} 次)")

        # 计算探索的均匀程度
        exploration_std = np.std(hist)
        exploration_score = 1 / (1 + exploration_std) * 100  # 转换为0-100的分数
        print(f"\n  探索均匀度评分: {exploration_score:.1f}/100 (越高越均匀)")
        print(f"  实际探索范围: [{min(all_thresholds):.3f}, {max(all_thresholds):.3f}]")

    # 显示收敛情况分析
    print(f"\n优化收敛分析：")
    checkpoints = [100, 200, 300, 500, 700, 1000]
    for checkpoint in checkpoints:
        if checkpoint <= len(study.trials):
            valid_at_checkpoint = [
                t for t in study.trials[:checkpoint] if t.value < 1e10
            ]
            if valid_at_checkpoint:
                best_at_checkpoint = min(t.value for t in valid_at_checkpoint)
                print(f"  试验{checkpoint:4d}时最佳: {best_at_checkpoint:.6f}")

    print("\n" + "=" * 60)
