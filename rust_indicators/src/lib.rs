//! # Rust Indicators - High-performance indicators for Jesse Trade
//!
//! This library provides Rust implementations of computationally intensive
//! indicators like VMD, NRBO, CWT, and FTI, with 50-100x performance improvements.

pub mod vmd;
pub mod nrbo;
pub mod cwt;
pub mod fti;
pub mod entropy;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn _rust_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册 VMD 函数
    m.add_function(wrap_pyfunction!(vmd::vmd_py, m)?)?;
    m.add_function(wrap_pyfunction!(vmd::vmd_batch_py, m)?)?;

    // 注册 NRBO 函数
    m.add_function(wrap_pyfunction!(nrbo::nrbo_py, m)?)?;
    m.add_function(wrap_pyfunction!(nrbo::nrbo_batch_py, m)?)?;

    // 注册 CWT 函数
    m.add_function(wrap_pyfunction!(cwt::cwt_py, m)?)?;

    // 注册 FTI 函数
    m.add_function(wrap_pyfunction!(fti::fti_process_py, m)?)?;

    // 注册 Entropy 函数
    m.add_function(wrap_pyfunction!(entropy::approximate_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::sample_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::approximate_entropy_rolling_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::sample_entropy_rolling_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::shannon_entropy_gaussian_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::shannon_entropy_gaussian_rolling_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::shannon_entropy_hist_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::shannon_entropy_hist_rolling_py, m)?)?;

    Ok(())
}
