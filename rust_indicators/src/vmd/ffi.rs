//! Python FFI 绑定
//!
//! 通过 PyO3 提供 Python 可调用的接口

use super::core;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::Bound;

/// VMD Python 接口
///
/// # Arguments
/// * `signal` - 输入信号 (numpy array)
/// * `alpha` - 数据保真度约束参数 (default: 2000)
/// * `tau` - 对偶上升时间步长 (default: 0.0)
/// * `k` - 模态数量 (default: 5)
/// * `dc` - 第一个模态是否固定在 DC (default: False)
/// * `init` - omega 初始化方式 (default: 1)
/// * `tol` - 收敛容差 (default: 1e-7)
///
/// # Returns
/// Tuple of (u, u_hat, omega)
/// - u: 分解后的模态 (K, N)
/// - u_hat: 模态的频谱 (N, K) - 复数数组
/// - omega: 中心频率历史 (Niter, K)
#[pyfunction]
#[pyo3(signature = (signal, alpha=2000.0, tau=0.0, k=5, dc=false, init=1, tol=1e-7))]
pub fn vmd_py<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    alpha: f64,
    tau: f64,
    k: usize,
    dc: bool,
    init: usize,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    // 转换 numpy array 到 ndarray
    let signal_array = signal.as_array();
    let signal_owned = Array1::from_iter(signal_array.iter().copied());

    // 调用 Rust 实现
    let result = core::vmd(&signal_owned, alpha, tau, k, dc, init, tol)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("VMD error: {}", e)))?;

    // 转换回 numpy arrays
    let u_py = result.u.into_pyarray(py);
    let u_hat_py = result.u_hat.into_pyarray(py);
    let omega_py = result.omega.into_pyarray(py);

    Ok((u_py, u_hat_py, omega_py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_module_compiles() {
        // 编译测试
        assert!(true);
    }
}
