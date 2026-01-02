//! Entropy Python FFI bindings

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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

/// Shannon entropy (Gaussian NLL, 包含 log σ) Python 接口
///
/// # Arguments
/// * `x` - 输入时间序列（1D numpy array）
///
/// # Returns
/// 平均 NLL（nats）
#[pyfunction]
#[pyo3(signature = (x))]
pub fn shannon_entropy_gaussian_py(x: PyReadonlyArray1<f64>) -> f64 {
    let x_array = Array1::from_iter(x.as_array().iter().copied());
    core::shannon_entropy_gaussian(&x_array)
}

/// Shannon entropy (Histogram 插件估计) Python 接口
///
/// # Arguments
/// * `x` - 输入时间序列（1D numpy array）
/// * `bins` - 直方图箱数 (default: 30)
///
/// # Returns
/// Shannon entropy（nats）
#[pyfunction]
#[pyo3(signature = (x, bins=30))]
pub fn shannon_entropy_hist_py(x: PyReadonlyArray1<f64>, bins: usize) -> f64 {
    let x_array = Array1::from_iter(x.as_array().iter().copied());
    core::shannon_entropy_hist(&x_array, bins)
}

/// 滑动窗口 Approximate Entropy (ApEn) Python 接口
///
/// 对输入序列进行纯粹的滑动窗口 entropy 计算，使用 Rayon 并行。
///
/// # Arguments
/// * `data` - 输入序列（1D numpy array，任意数据如 log returns）
/// * `period` - 滑动窗口大小
/// * `m` - 嵌入维度 (default: 2)
/// * `r_ratio` - 容忍度比例 (default: 0.3)
/// * `use_std` - 是否使用标准差计算容忍度 (default: false)
///
/// # Returns
/// 熵值数组，前 (period-1) 个位置为 NaN
#[pyfunction]
#[pyo3(signature = (data, period, m=2, r_ratio=0.3, use_std=false))]
pub fn approximate_entropy_rolling_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
    m: usize,
    r_ratio: f64,
    use_std: bool,
) -> Bound<'py, PyArray1<f64>> {
    let data_array = Array1::from_iter(data.as_array().iter().copied());

    // 释放 GIL 进行并行计算
    let result = py.detach(|| {
        core::approximate_entropy_rolling(&data_array, period, m, r_ratio, use_std)
    });

    result.into_pyarray(py)
}

/// 滑动窗口 Sample Entropy (SampEn) Python 接口
///
/// 对输入序列进行纯粹的滑动窗口 entropy 计算，使用 Rayon 并行。
///
/// # Arguments
/// * `data` - 输入序列（1D numpy array，任意数据如 log returns）
/// * `period` - 滑动窗口大小
/// * `m` - 嵌入维度 (default: 2)
/// * `r_ratio` - 容忍度比例 (default: 0.3)
/// * `use_std` - 是否使用标准差计算容忍度 (default: false)
///
/// # Returns
/// 熵值数组，前 (period-1) 个位置为 NaN
#[pyfunction]
#[pyo3(signature = (data, period, m=2, r_ratio=0.3, use_std=false))]
pub fn sample_entropy_rolling_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
    m: usize,
    r_ratio: f64,
    use_std: bool,
) -> Bound<'py, PyArray1<f64>> {
    let data_array = Array1::from_iter(data.as_array().iter().copied());

    // 释放 GIL 进行并行计算
    let result = py.detach(|| {
        core::sample_entropy_rolling(&data_array, period, m, r_ratio, use_std)
    });

    result.into_pyarray(py)
}

/// 滑动窗口 Shannon entropy（Gaussian NLL, 包含 log σ）Python 接口
///
/// 返回每个窗口最后一个样本的 self-information（nats）。
///
/// # Arguments
/// * `data` - 输入序列（1D numpy array，任意数据如 log returns）
/// * `period` - 滑动窗口大小
///
/// # Returns
/// self-information 数组，前 (period-1) 个位置为 NaN
#[pyfunction]
#[pyo3(signature = (data, period))]
pub fn shannon_entropy_gaussian_rolling_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_array = Array1::from_iter(data.as_array().iter().copied());

    let result = py.detach(|| core::shannon_entropy_gaussian_rolling(&data_array, period));

    result.into_pyarray(py)
}

/// 滑动窗口 Shannon entropy（Histogram surprisal）Python 接口
///
/// 返回每个窗口最后一个样本的 self-information（nats）。
///
/// # Arguments
/// * `data` - 输入序列（1D numpy array，任意数据如 log returns）
/// * `period` - 滑动窗口大小
/// * `bins` - 直方图箱数 (default: 30)
/// * `min_prob` - 最小概率下限 (default: 1e-12)
///
/// # Returns
/// self-information 数组，前 (period-1) 个位置为 NaN
#[pyfunction]
#[pyo3(signature = (data, period, bins=30, min_prob=1e-12))]
pub fn shannon_entropy_hist_rolling_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
    bins: usize,
    min_prob: f64,
) -> Bound<'py, PyArray1<f64>> {
    let data_array = Array1::from_iter(data.as_array().iter().copied());

    let result = py.detach(|| {
        core::shannon_entropy_hist_rolling(&data_array, period, bins, min_prob)
    });

    result.into_pyarray(py)
}
