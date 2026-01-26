import numpy as np

from src.bars.fusion.base import FusionBarContainerBase


class RSGateBar(FusionBarContainerBase):
    """
    RS 方差钟 + 同向门控 Fusion Bar

    公式：Ξ(t) = RS_t × (1 + α × max(0, m_t))

    其中：
    - RS_t = ln(H_t/O_t) × ln(H_t/C_t) + ln(L_t/O_t) × ln(L_t/C_t)
      （Rogers-Satchell 方差代理，完全无量纲）
    - m_t = (r_t × r_{t-1}) / (|r_t| × |r_{t-1}| + ε)  ∈ [-1, 1]
      （方向一致强度）
    - r_t = ln(C_t / C_{t-1})（对数收益率）

    设计意图：
    - RS 作为底座提供"正态友好"的等方差采样
    - 同向门控(m_t > 0)时加速切分，使趋势段获得更细的时间分辨率
    - 横盘时(m_t <= 0)自然合并

    Parameters
    ----------
    max_bars : int
        最大bar数量，-1表示不限制
    alpha : float
        同向门控系数，控制同向时的加速程度
    threshold : float
        累积阈值，达到此值时生成新bar
    epsilon : float
        防止除零的小常数

    Benchmark (BTC 2022-2025, 1min):
    --------------------------------
    配置: long rank 1 (Optuna score: 3.290)
    输入: 1,578,136 根 1min K线
    输出: 6,055 根 Fusion Bar
    压缩比: 260.6:1 (约 4.34 小时/根)

    评估结果:
      综合评分: 68.6/100 (B)
      趋势性: 3.29/5 | 一致性: 4.54/5 | 稳定性: 2.12/5
      Hurst均值: 0.707 | 三重共识: 34.4%

    评分分布 (窗口/均分/1分/2分/3分/4分/5分):
      20:  2.98 |  9.0% | 22.4% | 44.0% |  2.0% | 20.9%
      40:  3.43 |  3.7% | 14.9% | 44.8% |  3.7% | 32.1%
      60:  3.46 |  6.5% | 20.9% | 27.1% |  6.8% | 37.8%
    """

    def __init__(
        self,
        max_bars: int = -1,
        alpha: float = 1.254463,
        threshold: float = 2.262993e-04,
        epsilon: float = 1e-10,
    ):
        super().__init__(max_bars, threshold)
        self.alpha = alpha
        self.epsilon = epsilon

    @property
    def max_lookback(self) -> int:
        # 需要 t, t-1, t-2 三个时间点的数据来计算 r_t, r_{t-1}
        return 2

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        """
        计算每根K线的阈值贡献

        Parameters
        ----------
        candles : np.ndarray
            Jesse K线数据，格式 [timestamp, open, close, high, low, volume]

        Returns
        -------
        np.ndarray
            每根K线的阈值贡献（长度为 len(candles) - max_lookback）
        """
        # 提取 OHLC
        open_arr = candles[:, 1]
        close_arr = candles[:, 2]
        high_arr = candles[:, 3]
        low_arr = candles[:, 4]

        eps = self.epsilon

        # 计算对数收益率 r_t = ln(C_t / C_{t-1})
        # r[i] 对应 candles[i+1] 的收益率
        log_returns = np.log(close_arr[1:] / (close_arr[:-1] + eps) + eps)

        # r_t 和 r_{t-1}
        r_t = log_returns[1:]  # 从 index 2 开始
        r_t_1 = log_returns[:-1]  # 从 index 1 开始

        # 计算方向一致强度 m_t
        # m_t = (r_t × r_{t-1}) / (|r_t| × |r_{t-1}| + ε)
        m_t = (r_t * r_t_1) / (np.abs(r_t) * np.abs(r_t_1) + eps)

        # 计算 Rogers-Satchell 方差代理 RS_t
        # RS_t = ln(H_t/O_t) × ln(H_t/C_t) + ln(L_t/O_t) × ln(L_t/C_t)
        # 对应 candles[2:] 的数据
        h = high_arr[2:]
        l = low_arr[2:]
        o = open_arr[2:]
        c = close_arr[2:]

        ln_h_o = np.log(h / (o + eps) + eps)
        ln_h_c = np.log(h / (c + eps) + eps)
        ln_l_o = np.log(l / (o + eps) + eps)
        ln_l_c = np.log(l / (c + eps) + eps)

        rs_t = ln_h_o * ln_h_c + ln_l_o * ln_l_c

        # 计算最终阈值
        # Ξ(t) = RS_t × (1 + α × max(0, m_t))
        gate = 1.0 + self.alpha * np.maximum(0.0, m_t)
        xi = rs_t * gate

        # RS_t 理论上应该 >= 0，但数值误差可能导致微小负值
        # 确保阈值非负
        xi = np.maximum(xi, 0.0)

        return xi


if __name__ == "__main__":
    import numpy as np

    # 加载数据
    candles = np.load("data/btc_1m.npy")
    print(f"加载了 {len(candles):,} 根 1 分钟 K 线")

    # 测试 get_thresholds
    bar = RSGateBar(alpha=0.4, threshold=1.0)
    thresholds = bar.get_thresholds(candles[:1000])
    print(f"\n阈值计算测试（前 1000 根 K 线）:")
    print(f"  输出长度: {len(thresholds)} (预期 {1000 - bar.max_lookback})")
    print(f"  min:    {np.min(thresholds):.2e}")
    print(f"  max:    {np.max(thresholds):.2e}")
    print(f"  mean:   {np.mean(thresholds):.2e}")
    print(f"  median: {np.median(thresholds):.2e}")

    # 测试完整流程
    print(f"\n完整流程测试:")
    bar = RSGateBar(alpha=0.4, threshold=1e-6)  # 用小阈值测试
    bar.update_with_candles(candles[:10000])
    fusion_bars = bar.get_fusion_bars()
    print(f"  输入 K 线: 10,000")
    print(f"  生成 Fusion Bar: {len(fusion_bars):,}")
    print(f"  压缩比: {10000 / len(fusion_bars):.1f}:1")
