"""核心模块：从 Rust 扩展导入底层函数

此模块负责导入 _rust_indicators 扩展模块中的所有函数。
如果 Rust 扩展不可用，会提供有用的错误提示。

注意：
- 带下划线前缀的函数（如 _rust_nrbo）表示内部使用，不应直接暴露给用户
- HAS_RUST 标志可用于检查 Rust 扩展是否可用
"""

try:
    from pyrs_indicators._rust_indicators import (
        vmd_py as _rust_vmd,
        vmd_batch_py as _rust_vmd_batch,  # 批量处理 API（Rayon 并行）
        cwt_py as _rust_cwt,
        fti_process_py as _rust_fti,
        nrbo_py as _rust_nrbo,  # 内部使用，不导出到公开 API
        nrbo_batch_py as _rust_nrbo_batch,  # NRBO 批量处理 API（Rayon 并行）
    )
    HAS_RUST = True
    _IMPORT_ERROR = None
except ImportError as e:
    HAS_RUST = False
    _IMPORT_ERROR = e

    def _raise_import_error(*args, **kwargs):
        """当 Rust 扩展不可用时抛出有用的错误信息"""
        raise ImportError(
            f"Rust indicators extension module not available.\n"
            f"Original error: {_IMPORT_ERROR}\n\n"
            f"To build the extension, run:\n"
            f"  cd rust_indicators\n"
            f"  cargo clean && maturin develop --release\n\n"
            f"Make sure you have Rust and Maturin installed:\n"
            f"  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n"
            f"  pip install maturin"
        ) from e

    # 创建占位函数，调用时会抛出有用的错误
    _rust_vmd = _raise_import_error
    _rust_vmd_batch = _raise_import_error
    _rust_cwt = _raise_import_error
    _rust_fti = _raise_import_error
    _rust_nrbo = _raise_import_error
    _rust_nrbo_batch = _raise_import_error


__all__ = [
    "HAS_RUST",
    "_rust_vmd",
    "_rust_vmd_batch",
    "_rust_cwt",
    "_rust_fti",
    # 注意：_rust_nrbo 不在 __all__ 中，表示仅供内部使用
]
