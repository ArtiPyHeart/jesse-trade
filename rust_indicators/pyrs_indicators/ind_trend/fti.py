"""Fast Trend Indicator (FTI) - 快速趋势指标

基于 Rust 的高性能 FTI 实现，速度比纯 Python 实现快 50-100 倍。

FTI 是一种自适应趋势检测算法，通过多尺度分析和统计滤波来识别价格趋势。
它结合了周期检测、滤波器设计和噪声抑制等多种技术。
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt

from .._core import _rust_fti


def fti(
    data: npt.NDArray[np.float64],
    use_log: bool = True,
    min_period: int = 5,
    max_period: int = 65,
    half_length: int = 35,
    lookback: int = 150,
    beta: float = 0.95,
    noise_cut: float = 0.20,
) -> Tuple[float, float, float, float]:
    """计算快速趋势指标

    Args:
        data: 价格数据，1D 数组，**最新数据在索引0**，最旧数据在末尾
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
            - 控制滤波器窗口大小
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

    Returns:
        (fti_value, filtered_value, width, best_period): 四元组
        - fti_value: FTI 指标值（>= 0，无上界）
            接近 0: 强噪声/无趋势
            数值越大: 趋势越强
        - filtered_value: 滤波后的价格值
        - width: 趋势宽度（振幅）
        - best_period: 检测到的最佳周期

    Raises:
        ValueError: 如果输入参数不合法
        RuntimeError: 如果 FTI 计算失败或产生 NaN/Inf

    Examples:
        >>> import numpy as np
        >>> from pyrs_indicators.ind_trend import fti
        >>>
        >>> # 创建趋势价格序列
        >>> np.random.seed(42)
        >>> prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        >>>
        >>> # 计算 FTI
        >>> fti_val, filtered, width, period = fti(prices)
        >>> print(f"FTI: {fti_val:.2f}, Period: {period:.0f}")
        FTI: 45.23, Period: 12
        >>>
        >>> # 使用自定义参数
        >>> fti_val, filtered, width, period = fti(
        ...     prices,
        ...     use_log=True,
        ...     min_period=10,
        ...     max_period=50,
        ...     lookback=100
        ... )

    Notes:
        - 数据长度必须 >= lookback
        - lookback 必须 >= max_period + half_length
        - FTI 值接近 0 表示价格处于噪声状态
        - FTI 值越大表示趋势越强（根据 mean_move/width 计算）
        - best_period 是自动检测的主导周期
        - Rust 实现比 Python/Numba 快 50-100 倍

    References:
        Ehlers, J. F. (2013). Cycle Analytics for Traders.
        John Wiley & Sons.
    """
    # ========== 参数验证（Fail Fast） ==========

    # 验证 data
    if not isinstance(data, np.ndarray):
        raise ValueError(f"data must be a numpy array, got {type(data)}")

    if data.ndim != 1:
        raise ValueError(f"data must be 1D array, got {data.ndim}D")

    if len(data) < 10:
        raise ValueError(f"data length must be >= 10, got {len(data)}")

    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f"data must be float array, got {data.dtype}")

    # 验证 use_log
    if not isinstance(use_log, bool):
        raise ValueError(f"use_log must be boolean, got {type(use_log)}")

    # 验证 min_period
    if not isinstance(min_period, int) or min_period < 2:
        raise ValueError(f"min_period must be integer >= 2, got {min_period}")

    # 验证 max_period
    if not isinstance(max_period, int) or max_period <= min_period:
        raise ValueError(
            f"max_period must be integer > min_period ({min_period}), got {max_period}"
        )

    # 验证 half_length
    if not isinstance(half_length, int) or half_length < 1:
        raise ValueError(f"half_length must be positive integer, got {half_length}")

    # 验证 lookback
    if not isinstance(lookback, int) or lookback < 10:
        raise ValueError(f"lookback must be integer >= 10, got {lookback}")

    min_lookback = max_period + half_length
    if lookback < min_lookback:
        raise ValueError(
            f"lookback must be >= max_period + half_length ({min_lookback}), "
            f"got {lookback}"
        )

    if len(data) < lookback:
        raise ValueError(
            f"data length ({len(data)}) must be >= lookback ({lookback})"
        )

    # 验证 beta
    if not isinstance(beta, (int, float)) or not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got {beta}")

    # 验证 noise_cut
    if not isinstance(noise_cut, (int, float)) or not (0 <= noise_cut <= 1):
        raise ValueError(f"noise_cut must be in [0, 1], got {noise_cut}")

    # ========== 调用 Rust 实现 ==========

    try:
        fti_value, filtered_value, width, best_period = _rust_fti(
            data,
            use_log=bool(use_log),
            min_period=int(min_period),
            max_period=int(max_period),
            half_length=int(half_length),
            lookback=int(lookback),
            beta=float(beta),
            noise_cut=float(noise_cut),
        )
    except Exception as e:
        raise RuntimeError(f"FTI computation failed: {e}") from e

    # ========== 结果验证 ==========

    # 检查 NaN/Inf
    if (
        np.isnan(fti_value)
        or np.isinf(fti_value)
        or np.isnan(filtered_value)
        or np.isinf(filtered_value)
        or np.isnan(width)
        or np.isinf(width)
        or np.isnan(best_period)
        or np.isinf(best_period)
    ):
        raise RuntimeError(
            "FTI produced NaN or Inf values. "
            "Try adjusting parameters or check input data for NaN/Inf."
        )

    # 检查 FTI 值域（根据公式 mean_move/width，应该 >= 0）
    if fti_value < 0:
        raise RuntimeError(
            f"FTI value must be non-negative: got {fti_value}"
        )

    # 检查 best_period 范围
    if not (min_period <= best_period <= max_period):
        raise RuntimeError(
            f"best_period ({best_period}) out of range "
            f"[{min_period}, {max_period}]"
        )

    # 检查 width 合理性
    if width < 0:
        raise RuntimeError(f"width must be non-negative, got {width}")

    # ========== 返回结果 ==========

    return fti_value, filtered_value, width, best_period
