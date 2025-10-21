//! NRBO Python FFI bindings

use pyo3::prelude::*;
use pyo3::Bound;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
