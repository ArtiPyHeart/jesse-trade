"""util_entropy - 熵计算工具函数

提供 Approximate Entropy (ApEn)、Sample Entropy (SampEn) 与 Shannon
entropy/self-information 的高性能实现。

使用示例:
    >>> from pyrs_indicators.util_entropy import approximate_entropy, sample_entropy
    >>> import numpy as np
    >>>
    >>> x = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> apen = approximate_entropy(x, m=2, r_ratio=0.3)
    >>> sampen = sample_entropy(x, m=2, r_ratio=0.3)

滑动窗口示例:
    >>> from pyrs_indicators.util_entropy import approximate_entropy_rolling
    >>> prices = np.cumsum(np.random.randn(1000)) + 100
    >>> apen_rolling = approximate_entropy_rolling(prices, period=32)
"""

from .entropy import (
    approximate_entropy,
    approximate_entropy_rolling,
    sample_entropy,
    sample_entropy_rolling,
    shannon_entropy_gaussian,
    shannon_entropy_gaussian_rolling,
    shannon_entropy_hist,
    shannon_entropy_hist_rolling,
)

__all__ = [
    "approximate_entropy",
    "approximate_entropy_rolling",
    "sample_entropy",
    "sample_entropy_rolling",
]
