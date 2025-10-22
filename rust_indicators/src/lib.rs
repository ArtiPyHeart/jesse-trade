//! # Rust Indicators - High-performance indicators for Jesse Trade
//!
//! This library provides Rust implementations of computationally intensive
//! indicators like VMD, NRBO, and CWT, with 50-100x performance improvements.

pub mod vmd;
pub mod nrbo;
pub mod cwt;

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

    Ok(())
}
