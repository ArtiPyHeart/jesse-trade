"""
Rust Indicators - High-performance indicators for Jesse Trade

This package provides Rust-accelerated implementations of computationally
intensive indicators like VMD (Variational Mode Decomposition) and NRBO
(Newton-Raphson Boundary Optimization).

Performance improvements: 10-20x faster than Python/Numba implementations.
"""

__version__ = "0.1.0"

# 导出主要接口
from .vmd import vmd
from .nrbo import nrbo

__all__ = ["vmd", "nrbo"]
