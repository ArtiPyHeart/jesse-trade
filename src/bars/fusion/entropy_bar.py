import numpy as np
from scipy.special import erfc

from pyrs_indicators.util_entropy import (
    shannon_entropy_hist_rolling,
)
from src.bars.fusion.base import FusionBarContainerBase


def _gaussian_tail_surprisal_rolling(
    data: np.ndarray,
    period: int,
    min_prob: float = 1e-12,
) -> np.ndarray:
    """滑动窗口 Gaussian 尾部概率 surprisal

    计算每个点在窗口内高斯分布下的"惊讶度"：-log(P(|Z| >= |z|))

    Args:
        data: 输入序列（如 log returns）
        period: 滑动窗口大小
        min_prob: 最小概率下限，防止 log(0)

    Returns:
        surprisal 数组，前 (period-1) 个为 NaN
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        mu = np.mean(window)
        sigma = np.std(window, ddof=0)

        if sigma < 1e-12:
            # 窗口内所有值相同，当前值完全可预测
            result[i] = 0.0
            continue

        # 当前值的 z-score
        z = (data[i] - mu) / sigma

        # 双侧尾概率: P(|Z| >= |z|) = erfc(|z| / sqrt(2))
        p_tail = np.maximum(erfc(np.abs(z) / np.sqrt(2.0)), min_prob)

        # surprisal = -log(p_tail)
        result[i] = -np.log(p_tail)

    return result


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
        - "tail_surprisal": 尾部概率 surprisal（推荐，天然非负）
        - "hist_surprisal": 直方图 surprisal
        建议最小窗口：
        - tail_surprisal: >= 20
        - hist_surprisal: >= max(200, 10 * hist_bins)
    hist_bins : int, optional
        直方图箱数（仅 hist_surprisal 使用）。
    min_prob : float, optional
        最小概率下限，防止 log(0)。
    max_bars : int, optional
        最大 bar 数量，-1 表示不限制。默认为 -1。
    """

    def __init__(
        self,
        period: int,
        threshold: float,
        max_bars: int = -1,
        method: str = "tail_surprisal",
        hist_bins: int = 30,
        min_prob: float = 1e-12,
    ):
        super().__init__(max_bars, threshold)
        assert period >= 2, "period must be >= 2"
        assert hist_bins >= 2, "hist_bins must be >= 2"
        assert 0 < min_prob < 1, "min_prob must be in (0, 1)"
        if method not in ("tail_surprisal", "hist_surprisal"):
            raise ValueError(
                f"method must be tail_surprisal or hist_surprisal, got {method}"
            )

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

        if self.method == "tail_surprisal":
            info_arr = _gaussian_tail_surprisal_rolling(
                log_ret,
                period=self.period,
                min_prob=self.min_prob,
            )
        else:  # hist_surprisal
            info_arr = shannon_entropy_hist_rolling(
                log_ret,
                period=self.period,
                bins=self.hist_bins,
                min_prob=self.min_prob,
            )

        valid_info = info_arr[self.period - 1 :]

        # NaN 填充为 0（build_bar_by_cumsum 不能正确处理 NaN）
        return np.nan_to_num(valid_info, nan=0.0)
