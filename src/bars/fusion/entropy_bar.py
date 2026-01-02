import numpy as np

from pyrs_indicators.util_entropy import (
    shannon_entropy_gaussian_rolling,
    shannon_entropy_hist_rolling,
)
from src.bars.fusion.base import FusionBarContainerBase


class EntropyBar(FusionBarContainerBase):
    """基于 Shannon self-information 的 Fusion Bar

    使用 log return 计算 per-candle 信息量，累积达到 threshold 时生成新 bar。

    Parameters
    ----------
    period : int
        滑动窗口大小（作用于 log return）。
    threshold : float
        累积阈值，达到此值时生成新 bar。
    method : str, optional
        信息量计算方法：
        - "gaussian_nll": 正态分布 NLL（包含 log σ）
        - "hist_surprisal": 直方图 surprisal
        建议最小窗口：
        - gaussian_nll: >= 50
        - hist_surprisal: >= max(200, 10 * hist_bins)
        说明：Gaussian NLL 在连续密度下可能出现负值，构建 bar 前会截断为 0。
    hist_bins : int, optional
        直方图箱数（仅 hist_surprisal 使用）。
    min_prob : float, optional
        直方图最小概率下限（仅 hist_surprisal 使用）。
    max_bars : int, optional
        最大 bar 数量，-1 表示不限制。默认为 -1。
    """

    def __init__(
        self,
        period: int,
        threshold: float,
        max_bars: int = -1,
        method: str = "gaussian_nll",
        hist_bins: int = 30,
        min_prob: float = 1e-12,
    ):
        super().__init__(max_bars, threshold)
        assert period >= 2, "period must be >= 2"
        assert hist_bins >= 2, "hist_bins must be >= 2"
        assert 0 < min_prob < 1, "min_prob must be in (0, 1)"
        if method not in ("gaussian_nll", "hist_surprisal"):
            raise ValueError(f"method must be gaussian_nll or hist_surprisal, got {method}")

        self.period = period
        self.method = method
        self.hist_bins = hist_bins
        self.min_prob = min_prob

    @property
    def max_lookback(self) -> int:
        return self.period + 1

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        close_arr = candles[:, 2].astype(np.float64)
        assert np.all(close_arr > 0), "close prices must be positive for log return"

        log_ret = np.log(close_arr[1:] / close_arr[:-1])

        if self.method == "gaussian_nll":
            info_arr = shannon_entropy_gaussian_rolling(log_ret, period=self.period)
        else:
            info_arr = shannon_entropy_hist_rolling(
                log_ret,
                period=self.period,
                bins=self.hist_bins,
                min_prob=self.min_prob,
            )

        valid_info = info_arr[self.period - 1 :]

        # NaN 填充为 0（build_bar_by_cumsum 不能正确处理 NaN）
        valid_info = np.nan_to_num(valid_info, nan=0.0)

        # 确保累计信息量非负，避免 cumsum 发生回撤
        return np.clip(valid_info, 0.0, None)
