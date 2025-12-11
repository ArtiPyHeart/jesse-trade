"""信号分解指标子包

提供基于 Rust 的高性能信号分解算法。
"""

from .vmd import vmd, vmd_batch

__all__ = ["vmd", "vmd_batch"]
