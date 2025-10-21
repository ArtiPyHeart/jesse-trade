//! # Rust Indicators - High-performance indicators for Jesse Trade
//!
//! This library provides Rust implementations of computationally intensive
//! indicators like VMD and NRBO, with 10-20x performance improvements.

pub mod vmd;
pub mod nrbo;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn _rust_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 VMD 函数
    m.add_function(wrap_pyfunction!(vmd::vmd_py, m)?)?;

    // 注册 NRBO 函数
    m.add_function(wrap_pyfunction!(nrbo::nrbo_py, m)?)?;

    Ok(())
}
