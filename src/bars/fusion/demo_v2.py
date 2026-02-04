import numpy as np

from src.bars.fusion.base import FusionBarContainerBase

_EPS = 1e-12


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(float)
    csum = np.cumsum(np.insert(arr.astype(float), 0, 0.0))
    out = csum[window:] - csum[:-window]
    pad = np.full(window - 1, np.nan, dtype=float)
    return np.concatenate([pad, out])


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(float)
    return _rolling_sum(arr, window) / float(window)


def _rolling_std(arr: np.ndarray, window: int, eps: float) -> np.ndarray:
    if window <= 1:
        return np.full_like(arr, np.nan, dtype=float)
    mean = _rolling_mean(arr, window)
    mean_sq = _rolling_mean(arr * arr, window)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var + eps)


def _rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """注意：该实现为 O(N*window)，仅用于研究/调参。"""
    if window <= 1:
        return arr.astype(float)
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        out[i] = float(np.quantile(arr[i - window + 1 : i + 1], q))
    return out


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    return _rolling_quantile(arr, window, 0.5)


def _run_length(signs: np.ndarray) -> np.ndarray:
    """计算连续同向的run长度（0表示无方向）。"""
    out = np.zeros_like(signs, dtype=float)
    run = 0
    prev = 0.0
    for i, s in enumerate(signs):
        if s == 0:
            run = 0
        elif s == prev:
            run += 1
        else:
            run = 1
        out[i] = run
        if s != 0:
            prev = s
    return out


def _rolling_efficiency_ratio(
    log_ret: np.ndarray, window: int, eps: float
) -> np.ndarray:
    if window <= 1:
        return np.ones_like(log_ret, dtype=float)
    net = np.abs(_rolling_sum(log_ret, window))
    gross = _rolling_sum(np.abs(log_ret), window) + eps
    return net / gross


def _rolling_flip_rate(log_ret: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(log_ret, dtype=float)
    signs = np.sign(log_ret)
    flips = np.zeros_like(signs, dtype=float)
    flips[1:] = (signs[1:] * signs[:-1] < 0).astype(float)
    flip_count = _rolling_sum(flips, window)
    denom = float(max(window - 1, 1))
    return flip_count / denom


def _apply_clip(
    values: np.ndarray, clip_value: np.ndarray | float, style: str, tau: float
) -> np.ndarray:
    if style == "hard":
        return np.where(values < clip_value, 0.0, values)
    if style == "subtract":
        return np.maximum(0.0, values - clip_value)
    if style == "softplus":
        tau = max(tau, _EPS)
        return tau * np.log1p(np.exp((values - clip_value) / tau))
    raise ValueError(f"Unknown clip_style: {style}")


class DemoBarV2(FusionBarContainerBase):
    """
    一个“可调参数超集”的 DemoBar，用于探索更强趋势性的轴。

    核心思路：
    - 不怕短时大波动，重点压制“长时间小幅震荡”。
    - 通过方向一致性、效率比、翻转惩罚等机制，提升趋势性评分。

    设计框架（按执行顺序）：
    1) 计算基础量（固定为 DemoBar 公式）
    2) 方向/趋势权重 (ER、翻转惩罚、run boost、趋势对齐)
    3) 低波动冻结 / 区间过滤
    4) clip（固定/动态）

    注意：
    - 本类用于“寻找更好轴”的实验与调参，不建议把所有开关同时打开。
    - rolling 相关参数会提升 max_lookback，并在 get_thresholds 中自动对齐。
    - 若要实现“残差携带 / 滞回阈值 / 方向性累积（带正负阈值）”，
      需要改造 build_bar_by_cumsum 或新增构建器。

    Parameters (核心)
    -----------------
    threshold : float
        累积阈值，达到此值时生成新bar。
    clip_source : str
        clip阈值来源：
        - "none"              : 不裁剪
        - "fixed"             : 使用 clip_r
        - "rolling_std"       : clip_k * rolling_std(base, clip_window)
        - "rolling_median"    : clip_k * rolling_median(base, clip_window)
        - "rolling_quantile"  : clip_k * rolling_quantile(base, clip_window, clip_q)
    clip_style : str
        - "hard"      : 小于阈值直接置0
        - "subtract"  : res = max(0, res - clip)
        - "softplus"  : 平滑压制 (soft_clip_tau)

    directional / anti-chop 相关参数
    -------------------------------
    use_er : bool
        是否使用效率比 ER = |sum(r)| / sum(|r|) 作为方向一致性权重。
    use_flip_penalty : bool
        是否对方向翻转频率施加惩罚（越频繁权重越低）。
    use_run_boost : bool
        是否对连续同向run进行加权（run越长权重越大）。
    use_trend_align : bool
        是否对“逆趋势方向”的单根bar进行惩罚。
    use_low_vol_freeze : bool
        是否冻结低波动区间（rolling std 小于 low_vol_floor 时置0）。

    其它
    ----
    - use_volume_weight: 用成交量强化趋势段（可选）
    - min_range_ratio: 过小的高低范围直接忽略
    """

    def __init__(
        self,
        max_bars: int = -1,
        threshold: float = 2.288335,
        clip_source: str = "fixed",
        clip_style: str = "hard",
        clip_r: float = 2.072895e-04,
        clip_k: float = 1.0,
        clip_window: int = 60,
        clip_q: float = 0.5,
        soft_clip_tau: float = 1.0,
        use_er: bool = False,
        er_window: int = 20,
        er_power: float = 1.0,
        er_min: float = 0.0,
        use_flip_penalty: bool = False,
        flip_window: int = 20,
        flip_penalty: float = 2.0,
        use_run_boost: bool = False,
        run_boost: float = 0.0,
        run_cap: float = 3.0,
        use_trend_align: bool = False,
        trend_window: int = 20,
        anti_align_penalty: float = 0.5,
        use_low_vol_freeze: bool = False,
        low_vol_window: int = 60,
        low_vol_floor: float = 0.0,
        min_range_ratio: float = 0.0,
        use_volume_weight: bool = False,
        volume_window: int = 60,
        volume_power: float = 1.0,
        volume_cap: float = 5.0,
    ) -> None:
        super().__init__(max_bars, threshold)
        assert clip_source in {
            "none",
            "fixed",
            "rolling_std",
            "rolling_median",
            "rolling_quantile",
        }
        assert clip_style in {"hard", "subtract", "softplus"}
        assert clip_window >= 1
        assert er_window >= 1
        assert flip_window >= 1
        assert trend_window >= 1
        assert low_vol_window >= 1
        assert volume_window >= 1
        assert volume_cap >= 0

        self.clip_source = clip_source
        self.clip_style = clip_style
        self.clip_r = clip_r
        self.clip_k = clip_k
        self.clip_window = clip_window
        self.clip_q = clip_q
        self.soft_clip_tau = soft_clip_tau

        self.use_er = use_er
        self.er_window = er_window
        self.er_power = er_power
        self.er_min = er_min

        self.use_flip_penalty = use_flip_penalty
        self.flip_window = flip_window
        self.flip_penalty = flip_penalty

        self.use_run_boost = use_run_boost
        self.run_boost = run_boost
        self.run_cap = run_cap

        self.use_trend_align = use_trend_align
        self.trend_window = trend_window
        self.anti_align_penalty = anti_align_penalty

        self.use_low_vol_freeze = use_low_vol_freeze
        self.low_vol_window = low_vol_window
        self.low_vol_floor = low_vol_floor

        self.min_range_ratio = min_range_ratio

        self.use_volume_weight = use_volume_weight
        self.volume_window = volume_window
        self.volume_power = volume_power
        self.volume_cap = volume_cap

        self._max_lookback = self._calc_max_lookback()
        self._trim_head = max(self._max_lookback - 1, 0)

    def _calc_max_lookback(self) -> int:
        lookback = 1
        if self.clip_source in {"rolling_std", "rolling_median", "rolling_quantile"}:
            lookback = max(lookback, self.clip_window)
        if self.use_er:
            lookback = max(lookback, self.er_window)
        if self.use_flip_penalty:
            lookback = max(lookback, self.flip_window)
        if self.use_trend_align:
            lookback = max(lookback, self.trend_window)
        if self.use_low_vol_freeze:
            lookback = max(lookback, self.low_vol_window)
        if self.use_volume_weight:
            lookback = max(lookback, self.volume_window)
        return lookback

    @property
    def max_lookback(self) -> int:
        return self._max_lookback

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        assert candles.ndim == 2, f"candles must be 2D, got {candles.ndim}D"
        assert candles.shape[1] == 6, (
            f"candles must have 6 columns, got {candles.shape[1]}"
        )
        assert len(candles) > self.max_lookback, (
            "Not enough candles for configured rolling windows"
        )

        close = candles[:, 2].astype(float)
        high = candles[:, 3].astype(float)
        low = candles[:, 4].astype(float)
        volume = candles[:, 5].astype(float)

        close_prev = close[:-1]
        close_now = close[1:]
        high_now = high[1:]
        low_now = low[1:]
        volume_now = volume[1:]

        # 基础序列（return / range）
        log_ret = np.log((close_now + _EPS) / (close_prev + _EPS))
        abs_return = np.abs(close_now - close_prev)
        range_abs = high_now - low_now
        range_ratio = range_abs / (close_now + _EPS)

        # 1) 基础量定义（固定为 DemoBar 公式）
        base = abs_return * range_abs / (close_now + _EPS)

        # 2) 方向/趋势权重
        weights = np.ones_like(base, dtype=float)

        # 2.1 过小范围直接忽略（压制长时间小幅震荡）
        if self.min_range_ratio > 0:
            weights = np.where(range_ratio < self.min_range_ratio, 0.0, weights)

        # 2.2 Efficiency Ratio (ER) - 方向一致性
        if self.use_er:
            er = _rolling_efficiency_ratio(log_ret, self.er_window, _EPS)
            er = np.nan_to_num(er, nan=0.0)
            if self.er_min > 0:
                er = np.where(er < self.er_min, 0.0, er)
            weights *= np.power(er, self.er_power)

        # 2.3 翻转惩罚 - 方向频繁切换时权重降低
        if self.use_flip_penalty:
            flip_rate = _rolling_flip_rate(log_ret, self.flip_window)
            flip_rate = np.nan_to_num(flip_rate, nan=0.0)
            weights *= np.exp(-self.flip_penalty * flip_rate)

        # 2.4 连续run加权 - 同向run越长，权重越高
        if self.use_run_boost:
            run_len = _run_length(np.sign(log_ret))
            run_weight = 1.0 + self.run_boost * np.maximum(run_len - 1.0, 0.0)
            if self.run_cap > 0:
                run_weight = np.minimum(run_weight, self.run_cap)
            weights *= run_weight

        # 2.5 趋势对齐 - 逆趋势方向降低权重
        if self.use_trend_align:
            trend_dir = np.sign(_rolling_sum(log_ret, self.trend_window))
            ret_sign = np.sign(log_ret)
            align = (trend_dir == 0) | (ret_sign == trend_dir)
            align_weight = np.where(align, 1.0, self.anti_align_penalty)
            weights *= align_weight

        # 2.6 低波动冻结 - 波动极小时直接置0
        if self.use_low_vol_freeze:
            vol = _rolling_std(log_ret, self.low_vol_window, _EPS)
            weights = np.where(vol < self.low_vol_floor, 0.0, weights)

        # 2.7 成交量权重 - 低量波动更容易被压制
        if self.use_volume_weight:
            vol_scale = _rolling_median(volume_now, self.volume_window)
            vol_scale = np.nan_to_num(vol_scale, nan=0.0)
            vol_ratio = volume_now / (vol_scale + _EPS)
            vol_ratio = np.maximum(vol_ratio, 0.0)
            vol_weight = np.power(vol_ratio, self.volume_power)
            if self.volume_cap > 0:
                vol_weight = np.minimum(vol_weight, self.volume_cap)
            weights *= vol_weight

        res = base * weights

        # 3) clip（固定 / 动态）
        if self.clip_source != "none":
            if self.clip_source == "fixed":
                clip_value = float(self.clip_r)
            elif self.clip_source == "rolling_std":
                clip_value = self.clip_k * _rolling_std(base, self.clip_window, _EPS)
            elif self.clip_source == "rolling_median":
                clip_value = self.clip_k * _rolling_median(base, self.clip_window)
            elif self.clip_source == "rolling_quantile":
                clip_value = self.clip_k * _rolling_quantile(
                    base, self.clip_window, self.clip_q
                )
            else:
                raise ValueError(f"Unknown clip_source: {self.clip_source}")

            clip_value = np.nan_to_num(clip_value, nan=0.0)
            res = _apply_clip(res, clip_value, self.clip_style, self.soft_clip_tau)

        # 4) 收尾清理
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        res = np.maximum(res, 0.0)

        # 对齐滚动窗口：丢弃头部 (max_lookback - 1)
        if self._trim_head > 0:
            res = res[self._trim_head :]

        return res


if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    import optuna

    candles = np.load("../../../data/btc_1m.npy")
    print(f"加载了 {len(candles)} 根1分钟K线")
    print(f"4小时K线理论数量: {len(candles) // 240}")

    def objective(trial):
        bar_container = DemoBarV2(max_bars=-1)
        bar_container.THRESHOLD = trial.suggest_float("THRESHOLD", 0.2, 10)
        bar_container.update_with_candles(candles)
        fusion_bar = bar_container.get_fusion_bars()

        four_hour_candle_count = len(candles) // 240
        fusion_bar_count = len(fusion_bar)
        if fusion_bar_count < four_hour_candle_count:
            return 1e10

        ret = np.log(fusion_bar[5:, 2]) - np.log(fusion_bar[:-5, 2])
        standard = (ret - ret.mean()) / ret.std()
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")

        trial.set_user_attr("fusion_bar_count", fusion_bar_count)
        trial.set_user_attr("four_hour_candle_count", four_hour_candle_count)
        trial.set_user_attr("kurtosis", kurtosis)

        return kurtosis

    def show_progress(study, trial):
        n = trial.number
        if n > 0 and n % 100 == 0:
            valid_trials = [t for t in study.trials[:n] if t.value < 1e10]
            if valid_trials:
                current_best = min(t.value for t in valid_trials)
                filtered_count = len([t for t in study.trials[:n] if t.value >= 1e10])
                print(
                    f"\n[进度 {n}/1000] 当前最佳Kurtosis: {current_best:.6f}, "
                    f"有效试验: {len(valid_trials)}, 被过滤: {filtered_count}"
                )

                thresholds = [t.params["THRESHOLD"] for t in valid_trials]
                if thresholds:
                    print(
                        f"  参数分布 - Min: {min(thresholds):.3f}, "
                        f"Max: {max(thresholds):.3f}, "
                        f"Mean: {np.mean(thresholds):.3f}, "
                        f"Std: {np.std(thresholds):.3f}"
                    )

    print("\n开始参数优化（增强探索性）...")
    print("=" * 60)

    study = optuna.create_study(
        direction="minimize",
        study_name="fusion_bar_optimization",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=200,
            n_warmup_steps=10,
            interval_steps=1,
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=300,
            n_ei_candidates=100,
            gamma=lambda x: min(int(x**0.5), 25),
            seed=42,
        ),
    )

    for threshold in [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        study.enqueue_trial({"THRESHOLD": threshold})

    print("已添加12个手动探索点以确保搜索空间覆盖")
    print("搜索范围: THRESHOLD ∈ [0.2, 10.0]")
    print("优化目标: 最小化Kurtosis（峰度）")
    print("约束条件: Fusion Bar数量 ≥ 4小时K线数量")
    print("=" * 60)

    study.optimize(
        objective,
        n_trials=1000,
        callbacks=[show_progress],
        n_jobs=1,
        show_progress_bar=True,
    )

    print("\n" + "=" * 60)
    print("参数优化完成！最终结果：")
    print("=" * 60)

    best_trial = study.best_trial

    print("\n最优参数：")
    print(f"  - THRESHOLD: {best_trial.params['THRESHOLD']:.6f}")
    print(f"  - Kurtosis: {best_trial.value:.6f}")

    print("\n最优参数下的统计信息：")
    print(f"  - Fusion Bar数量: {best_trial.user_attrs.get('fusion_bar_count', 'N/A')}")
    print(
        f"  - 4小时K线理论数量: {best_trial.user_attrs.get('four_hour_candle_count', 'N/A')}"
    )
    print(
        f"  - 压缩比率: {best_trial.user_attrs.get('fusion_bar_count', 0) / len(candles):.4%}"
    )

    print("\n重新计算最优参数下的Fusion Bar详细统计...")
    best_bar_container = DemoBarV2(max_bars=-1)
    best_bar_container.THRESHOLD = best_trial.params["THRESHOLD"]
    best_bar_container.update_with_candles(candles)
    best_fusion_bar = best_bar_container.get_fusion_bars()

    if len(best_fusion_bar) > 1:
        time_intervals = np.diff(best_fusion_bar[:, 0]) / (60 * 1000)
        print("\nFusion Bar时间间隔统计（分钟）：")
        print(f"  - 平均间隔: {np.mean(time_intervals):.2f}")
        print(f"  - 中位数间隔: {np.median(time_intervals):.2f}")
        print(f"  - 最小间隔: {np.min(time_intervals):.2f}")
        print(f"  - 最大间隔: {np.max(time_intervals):.2f}")
        print(f"  - 标准差: {np.std(time_intervals):.2f}")

    ret = np.log(best_fusion_bar[4:, 2]) - np.log(best_fusion_bar[:-4, 2])
    standard = (ret - ret.mean()) / ret.std()
    print("\nLag收益率统计：")
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

    print("\n优化过程统计：")
    print(f"  - 总试验次数: {len(study.trials)}")
    print(f"  - 有效试验次数: {len([t for t in study.trials if t.value < 1e10])}")
    print(
        f"  - 被约束条件过滤的试验: {len([t for t in study.trials if t.value >= 1e10])}"
    )

    print("\n前10个最优参数组合：")
    sorted_trials = sorted(
        [t for t in study.trials if t.value < 1e10], key=lambda x: x.value
    )[:10]
    for i, trial in enumerate(sorted_trials, 1):
        print(
            f"  {i:2d}. THRESHOLD={trial.params['THRESHOLD']:.6f}, "
            f"Kurtosis={trial.value:.6f}, "
            f"FusionBars={trial.user_attrs.get('fusion_bar_count', 'N/A')}"
        )
