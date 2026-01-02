"""util_entropy - 熵计算工具函数

提供 Approximate Entropy (ApEn) 和 Sample Entropy (SampEn) 的高性能实现。

使用示例:
    >>> from pyrs_indicators.util_entropy import approximate_entropy, sample_entropy
    >>> import numpy as np
    >>>
    >>> x = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> apen = approximate_entropy(x, m=2, r_ratio=0.3)
    >>> sampen = sample_entropy(x, m=2, r_ratio=0.3)
"""

from .entropy import approximate_entropy, sample_entropy

__all__ = ["approximate_entropy", "sample_entropy"]
