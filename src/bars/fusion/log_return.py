import numpy as np

from src.bars.fusion.base import FusionBarContainerBase


class LogReturnBar(FusionBarContainerBase):
    """
    完全无量纲的趋势轴：收盘对数位移 × 高低对数振幅

    公式：
        r_t = |ln(C_t / C_{t-1})|  # 收盘对数位移
        hl_t = ln(H_t / L_t)       # 高低对数振幅
        threshold_value = r_t × hl_t

    特点：
        - 完全无量纲，便于跨资产/跨时期使用一致阈值
        - 天然更稳健

    Parameters
    ----------
    max_bars : int
        最大bar数量，-1表示不限制。
    clip_r : float
        小于此阈值的波动将被压缩为0，用于过滤噪声。
    threshold : float
        累积阈值，达到此值时生成新bar。

    Benchmark (BTC 2022-2025, 1min):
    --------------------------------
    配置: long rank 1 (Optuna score: 3.289)
    输入: 1,578,136 根 1min K线
    输出: 5,336 根 Fusion Bar
    压缩比: 295.8:1 (约 4.93 小时/根)

    评估结果:
      综合评分: 68.7/100 (B)
      趋势性: 3.29/5 | 一致性: 4.55/5 | 稳定性: 2.13/5
      Hurst均值: 0.699 | 三重共识: 35.0%

    评分分布 (窗口/均分/1分/2分/3分/4分/5分):
      20:  3.00 |  9.1% | 21.6% | 44.8% |  2.5% | 20.6%
      40:  3.38 |  4.1% | 17.8% | 42.3% |  4.9% | 30.4%
      60:  3.49 |  7.6% | 20.0% | 24.5% |  5.2% | 41.4%
    """

    def __init__(
        self,
        max_bars: int = -1,
        clip_r: float = 1.729268e-09,
        threshold: float = 2.407565e-04,
    ):
        super().__init__(max_bars, threshold)
        self.clip_r = clip_r

    @property
    def max_lookback(self) -> int:
        return 1

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        """计算阈值序列

        Args:
            candles: [timestamp, open, close, high, low, volume]

        Returns:
            阈值数组，长度为 len(candles) - 1
        """
        close_arr = candles[:, 2]
        high_arr = candles[:, 3]
        low_arr = candles[:, 4]

        # r_t = |ln(C_t / C_{t-1})| 收盘对数位移
        log_return = np.abs(np.log(close_arr[1:] / close_arr[:-1]))

        # hl_t = ln(H_t / L_t) 高低对数振幅
        # 添加小量避免 log(1) = 0 或除零
        log_range = np.log(high_arr[1:] / low_arr[1:] + 1e-10)

        # threshold_value = r_t × hl_t
        res = log_return * log_range

        # 根据 clip_r 过滤噪声
        if self.clip_r > 0:
            res[res < self.clip_r] = 0

        return res
