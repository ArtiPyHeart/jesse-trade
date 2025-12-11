"""pyrs_indicators - Rust-powered high-performance technical indicators

这是一个基于 Rust 实现的高性能技术指标库，为量化交易提供 50-100 倍的速度提升。

组织结构
--------
- ind_wavelets: 小波分析指标（CWT等）
- ind_decomposition: 信号分解指标（VMD等）
- ind_trend: 趋势分析指标（FTI等）

使用示例
--------
    >>> # 导入指标
    >>> from pyrs_indicators.ind_decomposition import vmd
    >>> from pyrs_indicators.ind_wavelets import cwt
    >>> from pyrs_indicators.ind_trend import fti

    >>> # 使用 VMD 分解信号
    >>> import numpy as np
    >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200))
    >>> modes = vmd(signal, alpha=2000.0, K=3)
    >>> modes.shape
    (3, 200)

注意事项
--------
- 所有指标都使用 Rust 实现，需要先编译扩展模块
- 编译命令: cargo clean && maturin develop --release
- 确保安装了 Rust 工具链和 Maturin

版本信息
--------
Version: 0.6.1
Dependencies: PyO3 0.27.1, numpy 0.27.0
Python Support: 3.8+
"""

# 导入子包（按用户选择：仅支持分类导入）
from . import ind_wavelets
from . import ind_decomposition
from . import ind_trend

# 导入工具
from ._core import HAS_RUST

__version__ = "0.6.2"
__all__ = [
    # 子包
    "ind_wavelets",
    "ind_decomposition",
    "ind_trend",
    # 工具
    "HAS_RUST",
]
