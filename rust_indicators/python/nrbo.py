"""
NRBO (Newton-Raphson Boundary Optimization) - Rust 加速版本

用于改善 IMF (Intrinsic Mode Function) 边界效应的优化算法。
"""

import numpy as np

# 尝试导入 Rust 实现
try:
    from rust_indicators._nrbo import nrbo_py as _nrbo_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _nrbo_rust = None


def nrbo(
    imf: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Newton-Raphson Boundary Optimization

    使用 Newton-Raphson 方法优化 IMF 的边界点，以改善边界效应。

    Parameters
    ----------
    imf : np.ndarray
        输入的 IMF (本征模态函数), 1D 数组
    max_iter : int, default=10
        最大迭代次数
    tol : float, default=1e-6
        收敛容差

    Returns
    -------
    np.ndarray
        优化后的 IMF

    Notes
    -----
    该算法通过迭代调整边界极值点的值来减少边界效应。
    使用 Newton-Raphson 方法基于一阶和二阶导数来更新边界点。

    Examples
    --------
    >>> import numpy as np
    >>> from rust_indicators.python.nrbo import nrbo
    >>>
    >>> # 创建测试 IMF
    >>> imf = np.sin(np.linspace(0, 10, 100))
    >>>
    >>> # 优化边界
    >>> imf_optimized = nrbo(imf, max_iter=10, tol=1e-6)
    """
    if not RUST_AVAILABLE:
        raise ImportError(
            "Rust NRBO backend not available. "
            "Please compile the Rust extension by running: "
            "cd rust_indicators && maturin develop --release"
        )

    # 输入验证
    if not isinstance(imf, np.ndarray):
        imf = np.array(imf, dtype=np.float64)

    if imf.ndim != 1:
        raise ValueError(f"IMF must be 1D array, got shape {imf.shape}")

    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")

    # 调用 Rust 实现
    result = _nrbo_rust(imf, max_iter, tol)

    return result
