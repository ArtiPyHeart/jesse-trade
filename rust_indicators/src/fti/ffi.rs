//! FTI Python 绑定层
//!
//! 通过 PyO3 暴露给 Python

use super::core::FTI;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// FTI计算 - Python接口
///
/// 对应 Python 中的 FTI.process() 方法
///
/// # Arguments
/// * `data` - 价格数据，最近的数据点在索引0
/// * `use_log` - 是否对价格取对数
/// * `min_period` - 最短周期
/// * `max_period` - 最长周期
/// * `half_length` - 中心系数两侧的系数数量
/// * `lookback` - 处理数据的块长度
/// * `beta` - 宽度计算的分位数
/// * `noise_cut` - 定义FTI噪声的最长内部移动的分数
///
/// # Returns
/// (fti, filtered_value, width, best_period)
#[pyfunction]
pub fn fti_process_py<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    use_log: bool,
    min_period: usize,
    max_period: usize,
    half_length: usize,
    lookback: usize,
    beta: f64,
    noise_cut: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    // 转换输入数据
    let data_array = data.as_array();

    // 创建 FTI 计算器
    let mut fti = FTI::new(
        use_log,
        min_period,
        max_period,
        half_length,
        lookback,
        beta,
        noise_cut,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // 处理数据
    let result = fti
        .process(&data_array)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // 返回结果
    Ok((
        result.fti,
        result.filtered_value,
        result.width,
        result.best_period,
    ))
}
