//! Entropy Python FFI bindings

use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use super::core;

/// Approximate Entropy (ApEn) Python 接口
///
/// # Arguments
/// * `x` - 输入时间序列（1D numpy array）
/// * `m` - 嵌入维度 (default: 2)
/// * `r_ratio` - 容忍度比例 (default: 0.3)
/// * `use_std` - 是否使用标准差计算容忍度 (default: false, 使用 range)
///
/// # Returns
/// ApEn 值
#[pyfunction]
#[pyo3(signature = (x, m=2, r_ratio=0.3, use_std=false))]
pub fn approximate_entropy_py(
    x: PyReadonlyArray1<f64>,
    m: usize,
    r_ratio: f64,
    use_std: bool,
) -> f64 {
    let x_array = Array1::from_iter(x.as_array().iter().copied());
    core::approximate_entropy(&x_array, m, r_ratio, use_std)
}

/// Sample Entropy (SampEn) Python 接口
///
/// # Arguments
/// * `x` - 输入时间序列（1D numpy array）
/// * `m` - 嵌入维度 (default: 2)
/// * `r_ratio` - 容忍度比例 (default: 0.3)
/// * `use_std` - 是否使用标准差计算容忍度 (default: false, 使用 range)
///
/// # Returns
/// SampEn 值，如果无法计算则返回 NaN
#[pyfunction]
#[pyo3(signature = (x, m=2, r_ratio=0.3, use_std=false))]
pub fn sample_entropy_py(
    x: PyReadonlyArray1<f64>,
    m: usize,
    r_ratio: f64,
    use_std: bool,
) -> f64 {
    let x_array = Array1::from_iter(x.as_array().iter().copied());
    core::sample_entropy(&x_array, m, r_ratio, use_std)
}
