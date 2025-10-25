//! # Rust Indicators - High-performance indicators for Jesse Trade
//!
//! This library provides Rust implementations of computationally intensive
//! indicators like VMD, NRBO, CWT, and FTI, with 50-100x performance improvements.

pub mod vmd;
pub mod nrbo;
pub mod cwt;
pub mod fti;

// Ripser 模块（独立开发，暂不导出到 Python）
// 用于持久同调计算，处于迭代开发阶段
pub mod ripser;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn _rust_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 VMD 函数
    m.add_function(wrap_pyfunction!(vmd::vmd_py, m)?)?;

    // 注册 NRBO 函数
    m.add_function(wrap_pyfunction!(nrbo::nrbo_py, m)?)?;

    // 注册 CWT 函数
    m.add_function(wrap_pyfunction!(cwt::cwt_py, m)?)?;

    // 注册 FTI 函数
    m.add_function(wrap_pyfunction!(fti::fti_process_py, m)?)?;

    // 注册 Ripser 函数
    ripser::ffi::register_ripser_functions(m)?;

    Ok(())
}
