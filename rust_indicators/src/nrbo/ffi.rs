//! NRBO Python FFI bindings

use pyo3::prelude::*;
use pyo3::Bound;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array1;

use super::core;

/// NRBO Python 接口
#[pyfunction]
#[pyo3(signature = (imf, max_iter=10, tol=1e-6))]
pub fn nrbo_py<'py>(
    py: Python<'py>,
    imf: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let imf_array = imf.as_array();
    let imf_owned = Array1::from_iter(imf_array.iter().copied());

    let result = core::nrbo(&imf_owned, max_iter, tol);

    Ok(result.into_pyarray(py))
}

/// NRBO 批量处理 Python 接口（Rayon 并行）
///
/// 一次调用处理多个 IMF（2D 数组，每行是一个 IMF），使用多线程并行加速。
///
/// # Arguments
/// * `imfs` - 输入 IMF 数组 (K, N)，每行是一个 IMF
/// * `max_iter` - 最大迭代次数 (default: 10)
/// * `tol` - 收敛容差 (default: 1e-6)
///
/// # Returns
/// 优化后的 IMF 数组 (K, N)
#[pyfunction]
#[pyo3(signature = (imfs, max_iter=10, tol=1e-6))]
pub fn nrbo_batch_py<'py>(
    py: Python<'py>,
    imfs: PyReadonlyArray2<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let imfs_array = imfs.as_array();
    let k = imfs_array.nrows();
    let n = imfs_array.ncols();

    // 转换为 Vec<Array1<f64>>
    let imfs_owned: Vec<Array1<f64>> = (0..k)
        .map(|i| Array1::from_iter(imfs_array.row(i).iter().copied()))
        .collect();

    // 释放 GIL 并行处理
    let results = py.detach(|| {
        core::nrbo_batch(&imfs_owned, max_iter, tol)
    });

    // 转换回 2D numpy array
    let mut output = ndarray::Array2::zeros((k, n));
    for (i, result) in results.into_iter().enumerate() {
        for (j, &val) in result.iter().enumerate() {
            output[[i, j]] = val;
        }
    }

    Ok(output.into_pyarray(py))
}
