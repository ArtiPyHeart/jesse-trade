"""频率可调谐指标 (FTI) - Rust 加速版本

使用 Rust 后端的高性能 FTI 实现，速度比 Numba 版本快 50-100 倍。
保持与 Python 版本完全一致的接口。

原始指标由 Govinda Khalsa 开发，用于在价格数据中识别最佳周期结构。
"""

from typing import NamedTuple

import numpy as np
from jesse import helpers
from scipy import special

from pyrs_indicators.ind_trend import fti as rust_fti


class FTIResult(NamedTuple):
    """FTI指标的返回结果

    Attributes:
        fti: 最佳周期的FTI值（已应用Gamma变换）
        filtered_value: 最佳周期的滤波值
        width: 最佳周期的宽度
        best_period: 具有最大FTI的周期
    """

    fti: float  # 最佳周期的FTI值
    filtered_value: float  # 最佳周期的滤波值
    width: float  # 最佳周期的宽度
    best_period: float  # 具有最大FTI的周期


class FTI:
    """频率可调谐指标 (Frequency Tunable Indicator) - Rust 加速版本

    使用 Rust 后端实现，性能比 Numba 版本快 50-100 倍。

    原始指标由Govinda Khalsa开发，用于在价格数据中识别最佳周期结构。

    Args:
        use_log: 是否对价格取对数（推荐 True）
            - True: 处理对数价格，适合价格序列
            - False: 直接处理原始数据
        min_period: 最短周期（默认 5）
            - 扫描的最小周期
            - 必须 >= 2
        max_period: 最长周期（默认 65）
            - 扫描的最大周期
            - 必须 > min_period
        half_length: 滤波器半长度（默认 35）
            - 中心系数两侧的系数数量
            - 实际窗口长度 = 2 * half_length + 1
        lookback: 处理数据的窗口长度（默认 150）
            - 计算 FTI 时使用的数据点数
            - 必须 >= max_period + half_length
        beta: 宽度计算的分位数（默认 0.95）
            - 用于噪声估计
            - 范围 [0, 1]，通常使用 0.90-0.99
        noise_cut: 噪声阈值（默认 0.20）
            - 定义 FTI 噪声区间的分数
            - 范围 [0, 1]，值越大容忍噪声越多

    Raises:
        ValueError: 如果参数不合法

    Examples:
        >>> # 创建计算器
        >>> calculator = FTI(use_log=True, lookback=150)
        >>>
        >>> # 处理数据（最新数据在索引0）
        >>> prices_reversed = prices[::-1]
        >>> result = calculator.process(prices_reversed)
        >>>
        >>> print(f"FTI: {result.fti:.2f}")
        >>> print(f"Period: {result.best_period:.0f}")

    Notes:
        - 数据长度必须 >= lookback
        - lookback 必须 >= max_period + half_length
        - 输入数据要求：最新数据在索引0，最旧数据在末尾
        - Rust 实现已在集成测试中验证，与 Python/Numba 数值完全一致
        - Rust 版本比 Numba 快 50-100 倍

    References:
        - 集成测试报告: FTI_INTEGRATION_TEST_REPORT.md
        - Rust 实现: rust_indicators/pyrs_indicators/ind_trend/fti.py
    """

    def __init__(
        self,
        use_log: bool = True,
        min_period: int = 5,
        max_period: int = 65,
        half_length: int = 35,
        lookback: int = 150,
        beta: float = 0.95,
        noise_cut: float = 0.20,
    ):
        """初始化FTI指标计算器"""
        # 参数验证（与Python版本一致）
        if max_period < min_period or min_period < 2:
            raise ValueError("max_period必须大于min_period且min_period至少为2")
        if 2 * half_length < max_period:
            raise ValueError("2*half_length必须大于max_period")
        if lookback - half_length < 2:
            raise ValueError("lookback必须比half_length至少大2")

        # 保存参数
        self.use_log = use_log
        self.min_period = min_period
        self.max_period = max_period
        self.half_length = half_length
        self.lookback = lookback
        self.beta = beta
        self.noise_cut = noise_cut

    def process(self, data: np.ndarray) -> FTIResult:
        """处理价格数据块并计算FTI指标

        Args:
            data: 价格数据，最近的数据点在索引0
                - 1D numpy array
                - 长度必须 >= lookback
                - 数据顺序：data[0]=最新，data[-1]=最旧

        Returns:
            FTIResult: 包含以下字段的命名元组
                - fti: FTI值（已应用Gamma变换）
                - filtered_value: 滤波后的价格值
                - width: 趋势宽度
                - best_period: 检测到的最佳周期

        Raises:
            ValueError: 如果数据长度不足
            RuntimeError: 如果Rust计算失败

        Examples:
            >>> # 准备数据（最新在前）
            >>> prices = np.array([105, 104, 103, ...])  # 最新到最旧
            >>> result = calculator.process(prices)
            >>> print(f"FTI: {result.fti:.2f}, Period: {result.best_period}")
        """
        # 检查数据长度
        if len(data) < self.lookback:
            raise ValueError(f"数据长度必须至少为{self.lookback}")

        # 调用 Rust 实现
        try:
            fti_raw, filtered_value, width_value, best_period = rust_fti(
                data,
                use_log=self.use_log,
                min_period=self.min_period,
                max_period=self.max_period,
                half_length=self.half_length,
                lookback=self.lookback,
                beta=self.beta,
                noise_cut=self.noise_cut,
            )
        except Exception as e:
            raise RuntimeError(f"Rust FTI computation failed: {e}") from e

        # 应用 Gamma 变换（与 Python 版本一致）
        fti_transformed = 100.0 * special.gammainc(2.0, fti_raw / 3.0) - 50.0

        # 返回结果
        return FTIResult(
            fti=fti_transformed,
            filtered_value=filtered_value,
            width=width_value,
            best_period=float(best_period),
        )


def fti(
    candles: np.ndarray,
    lookback: int = 150,
    half_length: int = 35,
    min_period: int = 5,
    max_period: int = 65,
    use_log: bool = True,
    sequential: bool = False,
) -> FTIResult:
    """频率可调谐指标 (Frequency Tunable Indicator) - Rust 加速版本

    使用 Rust 后端的高性能 FTI 实现，性能比 Numba 版本快 50-100 倍。
    保持与 Python 版本完全一致的接口。

    由Govinda Khalsa开发的指标，用于识别价格数据中的优势周期结构。
    返回包含多个指标值的命名元组。

    Args:
        candles: K线数据，NumPy 数组
            - 格式: [timestamp, open, close, high, low, volume]
            - 形状: (N, 6)
        lookback: 回看周期长度（默认 150）
            - 计算 FTI 时使用的数据点数
            - 建议值: 100-300
        half_length: 滤波器半长度（默认 35）
            - 控制滤波器窗口大小
            - 必须满足: 2*half_length >= max_period
        min_period: 最小周期（默认 5）
            - 扫描的最短周期
            - 必须 >= 2
        max_period: 最大周期（默认 65）
            - 扫描的最长周期
            - 必须 > min_period
        use_log: 是否使用对数价格（默认 True）
            - True: 处理对数价格（推荐）
            - False: 处理原始价格
        sequential: 是否返回整个序列（默认 False）
            - False: 只返回最新值
            - True: 返回所有值（数组形式）

    Returns:
        FTIResult: 命名元组，包含以下字段
            - fti: FTI值（float 或 ndarray）
            - filtered_value: 滤波后的价格值（float 或 ndarray）
            - width: 趋势宽度（float 或 ndarray）
            - best_period: 最佳周期（float 或 ndarray）

        当 sequential=False 时，所有字段都是 float
        当 sequential=True 时，所有字段都是 ndarray

    Raises:
        ValueError: 如果参数不合法或数据长度不足
        RuntimeError: 如果Rust计算失败

    Examples:
        >>> # 单值计算（实时交易）
        >>> result = fti(candles, lookback=150, sequential=False)
        >>> print(f"FTI: {result.fti:.2f}")
        >>> print(f"Period: {result.best_period:.0f}")
        >>>
        >>> # 序列计算（回测）
        >>> result = fti(candles, lookback=150, sequential=True)
        >>> print(f"FTI序列长度: {len(result.fti)}")
        >>> print(f"最新FTI: {result.fti[-1]:.2f}")
        >>>
        >>> # 自定义参数
        >>> result = fti(
        ...     candles,
        ...     lookback=200,
        ...     min_period=10,
        ...     max_period=50,
        ...     use_log=True
        ... )

    Notes:
        - 数据顺序：Jesse K线是时间序列（旧→新），函数内部会自动反转
        - FTI值范围：通常在 -50 到 +50 之间
        - 正值表示强趋势，负值表示噪声/震荡
        - Rust 实现已验证与 Python/Numba 版本数值完全对齐
        - 性能提升：50-100倍（相比 Numba 版本）
        - sequential=True 时，前 lookback-1 个值会填充为0

    Performance:
        - Rust 版本比 Numba 快 50-100 倍
        - 适合高频交易和大规模回测

    References:
        - 集成测试报告: FTI_INTEGRATION_TEST_REPORT.md
        - Python 版本: src/indicators/prod/fti.py
    """
    # 切片K线数据（Jesse helpers）
    candles = helpers.slice_candles(candles, sequential)

    # 获取收盘价
    source = helpers.get_candle_source(candles, "close")

    # 创建FTI计算器
    calculator = FTI(
        use_log=use_log,
        min_period=min_period,
        max_period=max_period,
        half_length=half_length,
        lookback=lookback,
        beta=0.95,
        noise_cut=0.20,
    )

    if sequential:
        # 初始化结果数组
        fti_values = np.zeros(len(candles))
        filtered_values = np.zeros(len(candles))
        width_values = np.zeros(len(candles))
        period_values = np.zeros(len(candles))

        # 初始化未定义的值（前面的数据不足lookback的点）
        front_bad = lookback - 1
        fti_values[:front_bad] = 0
        filtered_values[:front_bad] = 0
        width_values[:front_bad] = 0
        period_values[:front_bad] = 0

        # 对每个有效点计算FTI
        for i in range(front_bad, len(candles)):
            # 提取窗口数据并反转（使最近的数据点在索引0）
            data_window = source[i - lookback + 1 : i + 1][::-1]
            result = calculator.process(data_window)

            fti_values[i] = result.fti
            filtered_values[i] = result.filtered_value
            width_values[i] = result.width
            period_values[i] = result.best_period

        # 返回序列结果
        return FTIResult(
            fti=fti_values,
            filtered_value=filtered_values,
            width=width_values,
            best_period=period_values,
        )
    else:
        # 仅计算最后一个点的FTI
        data_window = source[-lookback:][::-1]  # 反转使最近的数据点在索引0

        # 返回结果
        return calculator.process(data_window)
