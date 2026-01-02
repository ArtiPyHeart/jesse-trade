"""Entropy 工具函数 - Approximate Entropy (ApEn) 和 Sample Entropy (SampEn)

提供时间序列复杂度度量的高性能 Rust 实现。
"""

import numpy as np
import numpy.typing as npt

from .._core import (
    _rust_approximate_entropy,
    _rust_approximate_entropy_rolling,
    _rust_sample_entropy,
    _rust_sample_entropy_rolling,
)


def approximate_entropy(
    x: npt.NDArray[np.float64],
    m: int = 2,
    r_ratio: float = 0.3,
    mode: str = "range",
) -> float:
    """Approximate Entropy (ApEn)

    测量时间序列的规律性和不可预测性。较低的值表示更规则的序列。

    算法:
        ApEn(m, r) = phi(m, r) - phi(m+1, r)
        其中 phi(m, r) = mean(log(C_i^m(r)))
        C_i^m(r) 包含自匹配

    Args:
        x: 输入时间序列（1D numpy array）
        m: 嵌入维度，通常为 2（default: 2）
        r_ratio: 容忍度比例，通常为 0.2-0.3（default: 0.3）
        mode: 范围计算方式
            - "range": r = r_ratio * (max - min)
            - "std": r = r_ratio * std(x)

    Returns:
        ApEn 值。较低的值表示更规则的序列。

    Raises:
        ValueError: 如果输入参数不合法

    Example:
        >>> import numpy as np
        >>> from pyrs_indicators.util_entropy import approximate_entropy
        >>> x = np.sin(np.linspace(0, 4*np.pi, 100))  # 正弦波
        >>> apen = approximate_entropy(x, m=2, r_ratio=0.3)
        >>> print(f"ApEn: {apen:.4f}")  # 周期信号，较低的 ApEn
    """
    # 参数验证 (Fail Fast)
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got {x.ndim}D")
    if len(x) < m + 2:
        raise ValueError(f"x must have at least {m + 2} elements, got {len(x)}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not 0 < r_ratio < 1:
        raise ValueError(f"r_ratio must be in (0, 1), got {r_ratio}")
    if mode not in ("range", "std"):
        raise ValueError(f"mode must be 'range' or 'std', got '{mode}'")

    use_std = mode == "std"
    return _rust_approximate_entropy(
        x.astype(np.float64, copy=False), m, r_ratio, use_std
    )


def sample_entropy(
    x: npt.NDArray[np.float64],
    m: int = 2,
    r_ratio: float = 0.3,
    mode: str = "range",
) -> float:
    """Sample Entropy (SampEn)

    测量时间序列的复杂度，是 ApEn 的改进版本（无偏估计）。

    算法:
        SampEn(m, r) = -ln(A / B)
        其中:
        - B: m 维模板匹配数（不含自匹配）
        - A: (m+1) 维模板匹配数（不含自匹配）

    与 ApEn 的区别:
        - SampEn 排除自匹配，减少偏差
        - SampEn 对序列长度的依赖性更小

    Args:
        x: 输入时间序列（1D numpy array）
        m: 嵌入维度，通常为 2（default: 2）
        r_ratio: 容忍度比例，通常为 0.2-0.3（default: 0.3）
        mode: 范围计算方式
            - "range": r = r_ratio * (max - min)
            - "std": r = r_ratio * std(x)

    Returns:
        SampEn 值。较低的值表示更规则的序列。
        如果无法计算（A=0 或 B=0），返回 NaN。

    Raises:
        ValueError: 如果输入参数不合法

    Example:
        >>> import numpy as np
        >>> from pyrs_indicators.util_entropy import sample_entropy
        >>> x = np.random.randn(200)  # 随机序列
        >>> sampen = sample_entropy(x, m=2, r_ratio=0.3)
        >>> print(f"SampEn: {sampen:.4f}")  # 随机信号，较高的 SampEn
    """
    # 参数验证 (Fail Fast)
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got {x.ndim}D")
    if len(x) < m + 2:
        raise ValueError(f"x must have at least {m + 2} elements, got {len(x)}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not 0 < r_ratio < 1:
        raise ValueError(f"r_ratio must be in (0, 1), got {r_ratio}")
    if mode not in ("range", "std"):
        raise ValueError(f"mode must be 'range' or 'std', got '{mode}'")

    use_std = mode == "std"
    return _rust_sample_entropy(
        x.astype(np.float64, copy=False), m, r_ratio, use_std
    )


def approximate_entropy_rolling(
    data: npt.NDArray[np.float64],
    period: int,
    m: int = 2,
    r_ratio: float = 0.3,
    mode: str = "range",
) -> npt.NDArray[np.float64]:
    """滑动窗口 Approximate Entropy (ApEn) 计算

    对输入序列进行纯粹的滑动窗口 entropy 计算，使用 Rust Rayon 并行。

    Args:
        data: 输入序列（1D numpy array，任意数据如 log returns）
        period: 滑动窗口大小
        m: 嵌入维度，通常为 2（default: 2）
        r_ratio: 容忍度比例，通常为 0.2-0.3（default: 0.3）
        mode: 范围计算方式
            - "range": r = r_ratio * (max - min)
            - "std": r = r_ratio * std(x)

    Returns:
        熵值数组，前 (period-1) 个位置为 NaN

    Raises:
        ValueError: 如果输入参数不合法

    Example:
        >>> import numpy as np
        >>> from pyrs_indicators.util_entropy import approximate_entropy_rolling
        >>> data = np.random.randn(1000)  # 任意序列
        >>> apen = approximate_entropy_rolling(data, period=32)
        >>> print(f"Shape: {apen.shape}, Non-NaN: {np.sum(~np.isnan(apen))}")
    """
    # 参数验证 (Fail Fast)
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")
    if data.ndim != 1:
        raise ValueError(f"data must be 1D array, got {data.ndim}D")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not 0 < r_ratio < 1:
        raise ValueError(f"r_ratio must be in (0, 1), got {r_ratio}")
    if mode not in ("range", "std"):
        raise ValueError(f"mode must be 'range' or 'std', got '{mode}'")

    use_std = mode == "std"
    return _rust_approximate_entropy_rolling(
        data.astype(np.float64, copy=False),
        period,
        m,
        r_ratio,
        use_std,
    )


def sample_entropy_rolling(
    data: npt.NDArray[np.float64],
    period: int,
    m: int = 2,
    r_ratio: float = 0.3,
    mode: str = "range",
) -> npt.NDArray[np.float64]:
    """滑动窗口 Sample Entropy (SampEn) 计算

    对输入序列进行纯粹的滑动窗口 entropy 计算，使用 Rust Rayon 并行。

    Args:
        data: 输入序列（1D numpy array，任意数据如 log returns）
        period: 滑动窗口大小
        m: 嵌入维度，通常为 2（default: 2）
        r_ratio: 容忍度比例，通常为 0.2-0.3（default: 0.3）
        mode: 范围计算方式
            - "range": r = r_ratio * (max - min)
            - "std": r = r_ratio * std(x)

    Returns:
        熵值数组，前 (period-1) 个位置为 NaN

    Raises:
        ValueError: 如果输入参数不合法

    Example:
        >>> import numpy as np
        >>> from pyrs_indicators.util_entropy import sample_entropy_rolling
        >>> data = np.random.randn(1000)  # 任意序列
        >>> sampen = sample_entropy_rolling(data, period=32)
        >>> print(f"Shape: {sampen.shape}, Non-NaN: {np.sum(~np.isnan(sampen))}")
    """
    # 参数验证 (Fail Fast)
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")
    if data.ndim != 1:
        raise ValueError(f"data must be 1D array, got {data.ndim}D")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not 0 < r_ratio < 1:
        raise ValueError(f"r_ratio must be in (0, 1), got {r_ratio}")
    if mode not in ("range", "std"):
        raise ValueError(f"mode must be 'range' or 'std', got '{mode}'")

    use_std = mode == "std"
    return _rust_sample_entropy_rolling(
        data.astype(np.float64, copy=False),
        period,
        m,
        r_ratio,
        use_std,
    )
