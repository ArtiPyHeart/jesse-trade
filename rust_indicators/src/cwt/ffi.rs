//! Python FFI bindings for CWT
//!
//! Provides Python-callable functions using PyO3.

use ndarray::s;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use super::core::cwt;
use super::utils::{symmetric_pad, to_db};

/// Python-callable CWT function
///
/// Implements the same interface as pywt.cwt but with Rust performance.
///
/// # Arguments
/// * `signal` - Input signal (1D numpy array)
/// * `scales` - Scale values (1D numpy array)
/// * `wavelet` - Wavelet name (e.g., "cmor1.5-1.0")
/// * `sampling_period` - Sampling period (default: 1.0)
/// * `precision` - Wavelet precision (default: 12)
/// * `pad_width` - Padding width for symmetric padding (default: 0, no padding)
/// * `verbose` - Print progress updates for dev/debugging (default: false)
///
/// # Returns
/// Tuple of (cwt_db, frequencies):
/// - cwt_db: CWT coefficients in dB scale (2D array, transposed: [signal_len, num_scales])
/// - frequencies: Corresponding frequencies (1D array)
///
/// # Example (Python)
/// ```python
/// import _rust_indicators
/// import numpy as np
///
/// signal = np.sin(np.linspace(0, 10, 100))
/// scales = np.logspace(np.log2(8), np.log2(128), 64, base=2)
/// cwt_db, freqs = _rust_indicators.cwt_py(signal, scales, "cmor1.5-1.0", 0.5)
/// ```
#[pyfunction]
#[pyo3(signature = (signal, scales, wavelet, sampling_period=1.0, precision=12, pad_width=0, verbose=false))]
pub fn cwt_py<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
    wavelet: &str,
    sampling_period: f64,
    precision: usize,
    pad_width: usize,
    verbose: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    // Convert numpy arrays to ndarray
    let signal_arr = signal.as_array().to_owned();
    let scales = scales.as_array().to_owned();

    // Step 1: Apply symmetric padding if requested
    let (signal_to_process, original_len) = if pad_width > 0 {
        let padded = symmetric_pad(&signal_arr, pad_width);
        let orig_len = signal_arr.len();
        (padded, orig_len)
    } else {
        let orig_len = signal_arr.len();
        (signal_arr, orig_len)
    };

    // Step 2: Compute CWT
    let output = cwt(&signal_to_process, &scales, wavelet, sampling_period, precision, verbose)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Step 3: Remove padding from coefficients
    let coefs_cropped = if pad_width > 0 {
        // Crop to original signal length
        output.coefs.slice(s![.., pad_width..pad_width + original_len]).to_owned()
    } else {
        output.coefs
    };

    // Step 4: Convert to dB and transpose to match Python output format
    // Python expects shape (signal_length, num_scales), i.e., transposed
    let cwt_db = to_db(&coefs_cropped, 1e-12);
    let cwt_db_t = cwt_db.t().to_owned();

    // Step 5: Convert back to numpy arrays
    let cwt_db_py = cwt_db_t.into_pyarray(py);
    let freqs_py = output.frequencies.into_pyarray(py);

    Ok((cwt_db_py, freqs_py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // This test just ensures the module compiles
        // Actual functionality tests will be in Python
    }
}
