"""Variational Mode Decomposition (VMD) - 变分模态分解

基于 Rust 的高性能 VMD 实现，速度比纯 Python 实现快 50-100 倍。

VMD 是一种自适应信号分解方法，可以将复杂信号分解为多个本征模态函数（IMF）。
相比 EMD/EEMD，VMD 具有更好的数学基础和抗噪性能。
"""

from typing import List, Tuple, Union
import numpy as np
import numpy.typing as npt

from .._core import _rust_vmd, _rust_vmd_batch


def vmd(
    signal: npt.NDArray[np.float64],
    alpha: float = 2000.0,
    tau: float = 0.0,
    K: int = 5,
    DC: bool = False,
    init: int = 1,
    tol: float = 1e-7,
    *,
    return_full: bool = False,
) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], npt.NDArray[np.float64]]]:
    """执行 VMD 分解

    Args:
        signal: 输入信号，1D 数组
        alpha: 数据保真度约束参数，控制模态带宽 (推荐值 2000-5000)
            - 值越大，模态带宽越窄，分解越精细
            - 值越小，模态带宽越宽，分解越粗糙
        tau: 对偶上升时间步长，噪声容忍度 (通常为 0)
            - 0 表示无噪声假设
            - 大于 0 表示对噪声的容忍度
        K: 模态数量 (推荐值 3-10)
            - 过小：无法充分分解信号
            - 过大：可能产生虚假模态
        DC: 第一个模态是否固定在直流分量 (通常为 False)
        init: omega 初始化方式
            - 0: 全部初始化为 0
            - 1: 均匀分布初始化 (推荐)
            - 2: 随机初始化
        tol: 收敛容差 (推荐值 1e-7)
        return_full: 是否返回完整结果 (u, u_hat, omega)，默认只返回 u

    Returns:
        如果 return_full=False (默认):
            modes: 分解后的模态 (K, N)，每行是一个 IMF

        如果 return_full=True:
            (modes, spectrum, omega): 完整结果元组
            - modes: 分解后的模态 (K, N)
            - spectrum: 模态的频谱 (N, K)，复数数组
            - omega: 中心频率演化历史 (Niter, K)

    Raises:
        ValueError: 如果输入参数不合法
        RuntimeError: 如果 VMD 计算失败或产生 NaN/Inf

    Examples:
        >>> import numpy as np
        >>> from pyrs_indicators.ind_decomposition import vmd
        >>>
        >>> # 创建测试信号：两个正弦波叠加
        >>> t = np.linspace(0, 1, 200)
        >>> signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        >>>
        >>> # 分解为 3 个模态
        >>> modes = vmd(signal, alpha=2000.0, K=3)
        >>> modes.shape
        (3, 200)
        >>>
        >>> # 获取完整结果
        >>> modes, spectrum, omega = vmd(signal, alpha=2000.0, K=3, return_full=True)
        >>> spectrum.dtype
        dtype('complex128')

    Notes:
        - 输入信号长度建议 > 50，过短的信号可能无法有效分解
        - alpha 参数对结果影响最大，建议根据信号特性调整
        - 重构信号：reconstructed = np.sum(modes, axis=0)
        - 能量守恒：np.sum(signal**2) ≈ np.sum(modes**2)
        - Rust 实现比 Python/Numba 快 50-100 倍

    References:
        Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition.
        IEEE Transactions on Signal Processing, 62(3), 531-544.
    """
    # ========== 参数验证（Fail Fast） ==========

    # 验证 signal
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"signal must be a numpy array, got {type(signal)}")

    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D array, got {signal.ndim}D")

    if len(signal) < 10:
        raise ValueError(f"signal length must be >= 10, got {len(signal)}")

    if not np.issubdtype(signal.dtype, np.floating):
        raise ValueError(f"signal must be float array, got {signal.dtype}")

    # 验证 alpha
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"alpha must be positive number, got {alpha}")

    # 验证 tau
    if not isinstance(tau, (int, float)) or tau < 0:
        raise ValueError(f"tau must be non-negative number, got {tau}")

    # 验证 K
    if not isinstance(K, int) or K < 1:
        raise ValueError(f"K must be positive integer, got {K}")

    max_k = len(signal) // 2
    if K > max_k:
        raise ValueError(f"K must be <= signal_length/2 ({max_k}), got {K}")

    # 验证 DC
    if not isinstance(DC, bool):
        raise ValueError(f"DC must be boolean, got {type(DC)}")

    # 验证 init
    if not isinstance(init, int) or init not in [0, 1, 2]:
        raise ValueError(f"init must be 0, 1, or 2, got {init}")

    # 验证 tol
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError(f"tol must be positive number, got {tol}")

    # ========== 调用 Rust 实现 ==========

    try:
        u, u_hat, omega = _rust_vmd(
            signal,
            alpha=float(alpha),
            tau=float(tau),
            k=int(K),
            dc=bool(DC),
            init=int(init),
            tol=float(tol),
        )
    except Exception as e:
        raise RuntimeError(f"VMD computation failed: {e}") from e

    # ========== 结果验证 ==========

    # 检查 NaN/Inf
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        raise RuntimeError(
            "VMD produced NaN or Inf values. "
            "Try adjusting parameters (alpha, K, tol) or check input signal."
        )

    # 检查输出形状
    if u.shape[0] != K or u.shape[1] != len(signal):
        raise RuntimeError(
            f"VMD output shape mismatch: expected ({K}, {len(signal)}), "
            f"got {u.shape}"
        )

    # ========== 返回结果 ==========

    if return_full:
        return u, u_hat, omega
    else:
        return u


def vmd_batch(
    signals: List[npt.NDArray[np.float64]],
    alpha: float = 2000.0,
    tau: float = 0.0,
    K: int = 5,
    DC: bool = False,
    init: int = 1,
    tol: float = 1e-7,
) -> List[npt.NDArray[np.float64]]:
    """批量执行 VMD 分解（Rayon 多线程并行加速）

    一次调用处理多个信号窗口，使用 Rust Rayon 多线程并行处理，
    避免 Python 循环调用的开销。适用于滚动窗口特征计算场景。

    Args:
        signals: 输入信号列表，每个元素为 1D 数组
        alpha: 数据保真度约束参数，控制模态带宽 (推荐值 2000-5000)
        tau: 对偶上升时间步长，噪声容忍度 (通常为 0)
        K: 模态数量 (推荐值 3-10)
        DC: 第一个模态是否固定在直流分量 (通常为 False)
        init: omega 初始化方式 (0: 全零, 1: 均匀分布, 2: 随机)
        tol: 收敛容差 (推荐值 1e-7)

    Returns:
        modes_list: 分解后的模态列表，每个元素形状为 (K, N)

    Raises:
        ValueError: 如果输入参数不合法
        RuntimeError: 如果 VMD 计算失败

    Examples:
        >>> import numpy as np
        >>> from pyrs_indicators.ind_decomposition import vmd_batch
        >>>
        >>> # 创建多个测试信号
        >>> t = np.linspace(0, 1, 200)
        >>> signals = [
        ...     np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t),
        ...     np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 15 * t),
        ... ]
        >>>
        >>> # 批量分解
        >>> modes_list = vmd_batch(signals, alpha=2000.0, K=3)
        >>> len(modes_list)
        2
        >>> modes_list[0].shape
        (3, 200)

    Notes:
        - 信号长度可以不同，但建议保持一致以获得最佳并行效率
        - 内部使用全局 FFT Plan 缓存，跨调用复用
        - 使用 Rayon 多线程并行，绕过 Python GIL 限制
        - 相比循环调用 vmd()，可获得 3-8x 加速（取决于 CPU 核心数）
    """
    # ========== 参数验证（Fail Fast） ==========

    if not isinstance(signals, (list, tuple)):
        raise ValueError(f"signals must be a list or tuple, got {type(signals)}")

    if len(signals) == 0:
        return []

    # 验证每个信号
    for i, signal in enumerate(signals):
        if not isinstance(signal, np.ndarray):
            raise ValueError(f"signals[{i}] must be a numpy array, got {type(signal)}")

        if signal.ndim != 1:
            raise ValueError(f"signals[{i}] must be 1D array, got {signal.ndim}D")

        if len(signal) < 10:
            raise ValueError(f"signals[{i}] length must be >= 10, got {len(signal)}")

        if not np.issubdtype(signal.dtype, np.floating):
            raise ValueError(f"signals[{i}] must be float array, got {signal.dtype}")

        max_k = len(signal) // 2
        if K > max_k:
            raise ValueError(
                f"K must be <= signal_length/2 for signals[{i}] ({max_k}), got {K}"
            )

    # 验证 alpha
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"alpha must be positive number, got {alpha}")

    # 验证 tau
    if not isinstance(tau, (int, float)) or tau < 0:
        raise ValueError(f"tau must be non-negative number, got {tau}")

    # 验证 K
    if not isinstance(K, int) or K < 1:
        raise ValueError(f"K must be positive integer, got {K}")

    # 验证 DC
    if not isinstance(DC, bool):
        raise ValueError(f"DC must be boolean, got {type(DC)}")

    # 验证 init
    if not isinstance(init, int) or init not in [0, 1, 2]:
        raise ValueError(f"init must be 0, 1, or 2, got {init}")

    # 验证 tol
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError(f"tol must be positive number, got {tol}")

    # ========== 调用 Rust 批量处理 ==========

    try:
        results = _rust_vmd_batch(
            signals,
            alpha=float(alpha),
            tau=float(tau),
            k=int(K),
            dc=bool(DC),
            init=int(init),
            tol=float(tol),
        )
    except Exception as e:
        raise RuntimeError(f"VMD batch computation failed: {e}") from e

    # ========== 结果验证 ==========

    for i, u in enumerate(results):
        # 检查 NaN/Inf
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            raise RuntimeError(
                f"VMD produced NaN or Inf values for signals[{i}]. "
                "Try adjusting parameters (alpha, K, tol) or check input signal."
            )

        # 检查输出形状
        expected_n = len(signals[i])
        if u.shape[0] != K or u.shape[1] != expected_n:
            raise RuntimeError(
                f"VMD output shape mismatch for signals[{i}]: "
                f"expected ({K}, {expected_n}), got {u.shape}"
            )

    return results
