"""Continuous Wavelet Transform (CWT) - 连续小波变换

基于 Rust 的高性能 CWT 实现，速度比纯 Python 实现快 50-100 倍。

CWT 是一种时频分析方法，可以将信号分解为不同尺度和时间位置的小波系数。
相比 STFT，CWT 具有自适应的时频分辨率。
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt

from .._core import _rust_cwt


def cwt(
    signal: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    wavelet: str = "cmor1.5-1.0",
    sampling_period: float = 1.0,
    precision: int = 12,
    pad_width: int = 0,
    verbose: bool = False,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """计算连续小波变换

    Args:
        signal: 输入信号，1D 数组
        scales: 尺度数组，1D 数组，通常使用对数间隔
            例如: np.logspace(np.log2(8), np.log2(128), num=64, base=2)
        wavelet: 小波类型，默认 "cmor1.5-1.0" (Complex Morlet)
            格式: "cmor<bandwidth>-<center_freq>"
            - bandwidth: 带宽参数（推荐 1.5）
            - center_freq: 中心频率（推荐 1.0）
        sampling_period: 采样周期（秒），默认 1.0
            - 用于将尺度转换为物理频率
            - 如果采样率是 fs Hz，则 sampling_period = 1/fs
        precision: 小波精度（默认 12）
            - 控制小波窗口大小
            - 值越大，精度越高，但计算越慢
        pad_width: 对称填充宽度（默认 0，不填充）
            - 用于减少边界效应
            - 推荐使用 int(max(scales))
        verbose: 是否打印调试信息（默认 False）

    Returns:
        (coefficients, frequencies): 元组
        - coefficients: CWT 系数（dB 尺度），形状 (signal_len, num_scales)
            第 i 行对应信号第 i 个时间点在所有尺度上的系数
        - frequencies: 对应的频率数组，形状 (num_scales,)
            frequencies[j] 是第 j 个尺度对应的频率

    Raises:
        ValueError: 如果输入参数不合法
        RuntimeError: 如果 CWT 计算失败或产生 NaN/Inf

    Examples:
        >>> import numpy as np
        >>> from pyrs_indicators.ind_wavelets import cwt
        >>>
        >>> # 创建测试信号
        >>> t = np.linspace(0, 1, 200)
        >>> signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz 正弦波
        >>>
        >>> # 计算 CWT
        >>> scales = np.logspace(np.log2(8), np.log2(128), num=32, base=2)
        >>> coef, freqs = cwt(signal, scales, sampling_period=0.5)
        >>>
        >>> coef.shape
        (200, 32)
        >>> freqs.shape
        (32,)

        >>> # 使用填充减少边界效应
        >>> pad = int(max(scales))
        >>> coef, freqs = cwt(signal, scales, pad_width=pad)

    Notes:
        - 尺度和频率的关系：frequency = center_freq / (scale * sampling_period)
        - 输出已经转换为 dB 尺度：20 * log10(abs(coef) + epsilon)
        - 频率数组按降序排列（大尺度 -> 低频，小尺度 -> 高频）
        - 填充会在计算后自动移除，不影响输出长度
        - Rust 实现比 PyWavelets 快 50-100 倍

    References:
        Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis.
        Bulletin of the American Meteorological Society, 79(1), 61-78.
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

    # 验证 scales
    if not isinstance(scales, np.ndarray):
        raise ValueError(f"scales must be a numpy array, got {type(scales)}")

    if scales.ndim != 1:
        raise ValueError(f"scales must be 1D array, got {scales.ndim}D")

    if len(scales) < 1:
        raise ValueError("scales array must not be empty")

    if not np.all(scales > 0):
        raise ValueError("all scales must be positive")

    # 验证 wavelet
    if not isinstance(wavelet, str):
        raise ValueError(f"wavelet must be string, got {type(wavelet)}")

    if not wavelet.startswith("cmor"):
        raise ValueError(
            f"Currently only Complex Morlet wavelets are supported (cmor*), got {wavelet}"
        )

    # 验证 sampling_period
    if not isinstance(sampling_period, (int, float)) or sampling_period <= 0:
        raise ValueError(f"sampling_period must be positive number, got {sampling_period}")

    # 验证 precision
    if not isinstance(precision, int) or precision < 1:
        raise ValueError(f"precision must be positive integer, got {precision}")

    if precision > 20:
        raise ValueError(f"precision too large (max 20), got {precision}")

    # 验证 pad_width
    if not isinstance(pad_width, int) or pad_width < 0:
        raise ValueError(f"pad_width must be non-negative integer, got {pad_width}")

    if pad_width > len(signal):
        raise ValueError(
            f"pad_width ({pad_width}) must be <= signal length ({len(signal)})"
        )

    # 验证 verbose
    if not isinstance(verbose, bool):
        raise ValueError(f"verbose must be boolean, got {type(verbose)}")

    # ========== 调用 Rust 实现 ==========

    try:
        coef, freqs = _rust_cwt(
            signal,
            scales,
            wavelet,
            sampling_period=float(sampling_period),
            precision=int(precision),
            pad_width=int(pad_width),
            verbose=bool(verbose),
        )
    except Exception as e:
        raise RuntimeError(f"CWT computation failed: {e}") from e

    # ========== 结果验证 ==========

    # 检查 NaN/Inf
    if np.any(np.isnan(coef)) or np.any(np.isinf(coef)):
        raise RuntimeError(
            "CWT produced NaN or Inf values. "
            "Try adjusting parameters (scales, precision) or check input signal."
        )

    # 检查输出形状
    if coef.shape != (len(signal), len(scales)):
        raise RuntimeError(
            f"CWT output shape mismatch: expected ({len(signal)}, {len(scales)}), "
            f"got {coef.shape}"
        )

    if freqs.shape != (len(scales),):
        raise RuntimeError(
            f"Frequencies shape mismatch: expected ({len(scales)},), "
            f"got {freqs.shape}"
        )

    # 验证频率递减（大尺度 -> 低频）
    if not np.all(freqs[:-1] >= freqs[1:]):
        raise RuntimeError("Frequencies must be in descending order")

    # ========== 返回结果 ==========

    return coef, freqs
