/*!
 * Ripser FFI - Python 绑定
 *
 * 这个模块实现 Rust Ripser 到 Python 的 FFI 绑定。
 *
 * # 接口设计
 *
 * Python 端调用：
 * ```python
 * result = ripser_compute(
 *     points,           # numpy array (n, d)
 *     max_dim=2,        # 最大维度
 *     threshold=None,   # 距离阈值
 *     metric='euclidean' # 距离度量
 * )
 * # result = {
 * #     'persistence': [dim0_pairs, dim1_pairs, ...],
 * #     'num_points': n,
 * #     'max_dim': 2,
 * #     'threshold': float
 * # }
 * ```
 */

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::core::algorithm::{compute_ripser, DistanceInput};

/// Ripser 计算（从点云）
///
/// # Arguments
///
/// * `py` - Python 解释器 GIL
/// * `points` - 点云数据 (n, d) numpy 数组
/// * `max_dim` - 计算的最大维度
/// * `threshold` - 距离阈值（None 表示无限制）
/// * `metric` - 距离度量
///
/// # Returns
///
/// Python 字典包含：
/// - 'persistence': 各维度的持久性对列表
/// - 'num_points': 点数量
/// - 'max_dim': 最大维度
/// - 'threshold': 使用的阈值
#[pyfunction]
#[pyo3(signature = (points, max_dim=1, threshold=None, metric="euclidean"))]
pub fn ripser_compute(
    py: Python,
    points: PyReadonlyArray2<f64>,
    max_dim: usize,
    threshold: Option<f64>,
    metric: &str,
) -> PyResult<Py<PyAny>> {
    // 转换 numpy 数组为 Vec<Vec<f64>>
    let points_array = points.as_array();
    let n = points_array.nrows();
    let d = points_array.ncols();

    let mut points_vec = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(d);
        for j in 0..d {
            row.push(points_array[[i, j]]);
        }
        points_vec.push(row);
    }

    // 调用 Rust 核心算法
    let result = compute_ripser(
        DistanceInput::Points(&points_vec),
        max_dim,
        threshold.map(|t| t as f32),
        metric,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // 转换结果为 Python 字典
    let dict = PyDict::new(py);

    // 添加持久性对
    let persistence_list = PyList::empty(py);
    for dim in 0..=result.max_dim {
        if let Some(pairs) = result.get_persistence(dim) {
            // 转换为 Vec<Vec<f64>>
            let pairs_vec: Vec<Vec<f64>> = pairs
                .iter()
                .map(|(birth, death)| vec![*birth as f64, *death as f64])
                .collect();

            // 转换为 ndarray，然后转为 PyArray
            let n = pairs_vec.len();
            let flat: Vec<f64> = pairs_vec.into_iter().flatten().collect();
            let array = ndarray::Array2::from_shape_vec((n, 2), flat)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            let py_array = array.into_pyarray(py);
            persistence_list.append(py_array)?;
        } else {
            // 空数组
            let empty = ndarray::Array2::<f64>::zeros((0, 2));
            let py_array = empty.into_pyarray(py);
            persistence_list.append(py_array)?;
        }
    }
    dict.set_item("persistence", persistence_list)?;

    // 添加元数据
    dict.set_item("num_points", result.num_points)?;
    dict.set_item("max_dim", result.max_dim)?;
    dict.set_item("threshold", result.threshold as f64)?;

    Ok(dict.into())
}

/// Ripser 计算（从距离矩阵）
///
/// # Arguments
///
/// * `py` - Python 解释器 GIL
/// * `distances` - 压缩距离矩阵（下三角，1D numpy 数组）
/// * `max_dim` - 计算的最大维度
/// * `threshold` - 距离阈值（None 表示无限制）
///
/// # Returns
///
/// Python 字典（同 ripser_compute）
#[pyfunction]
#[pyo3(signature = (distances, max_dim=1, threshold=None))]
pub fn ripser_compute_from_distance_matrix(
    py: Python,
    distances: PyReadonlyArray1<f64>,
    max_dim: usize,
    threshold: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // 转换为 Vec<f64>
    let distances_vec: Vec<f64> = distances.as_slice()?.to_vec();

    // 调用 Rust 核心算法
    let result = compute_ripser(
        DistanceInput::Compressed(&distances_vec),
        max_dim,
        threshold.map(|t| t as f32),
        "euclidean", // metric 对距离矩阵无影响
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // 转换结果为 Python 字典（同上）
    let dict = PyDict::new(py);

    let persistence_list = PyList::empty(py);
    for dim in 0..=result.max_dim {
        if let Some(pairs) = result.get_persistence(dim) {
            let pairs_vec: Vec<Vec<f64>> = pairs
                .iter()
                .map(|(birth, death)| vec![*birth as f64, *death as f64])
                .collect();

            let n = pairs_vec.len();
            let flat: Vec<f64> = pairs_vec.into_iter().flatten().collect();
            let array = ndarray::Array2::from_shape_vec((n, 2), flat)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            let py_array = array.into_pyarray(py);
            persistence_list.append(py_array)?;
        } else {
            let empty = ndarray::Array2::<f64>::zeros((0, 2));
            let py_array = empty.into_pyarray(py);
            persistence_list.append(py_array)?;
        }
    }
    dict.set_item("persistence", persistence_list)?;

    dict.set_item("num_points", result.num_points)?;
    dict.set_item("max_dim", result.max_dim)?;
    dict.set_item("threshold", result.threshold as f64)?;

    Ok(dict.into())
}

/// 注册 Ripser 函数到 Python 模块
pub fn register_ripser_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ripser_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ripser_compute_from_distance_matrix, m)?)?;
    Ok(())
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    // FFI 测试需要在 Python 环境中进行
    // 这里只测试编译通过
}
