//! Core CWT algorithm implementation
//!
//! Implements continuous wavelet transform using FFT-based convolution.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::wavelets::{cmor_wavelet, integrate_wavelet, CmorWavelet};
use super::utils::next_fast_len;

/// Thread-local reusable buffers for CWT computation
/// Avoids repeated heap allocations in parallel workers
#[derive(Default)]
struct CwtBuffers {
    signal_padded: Vec<Complex<f64>>,
    psi_padded: Vec<Complex<f64>>,
    j_indices: Vec<usize>,
    scratch_fwd: Vec<Complex<f64>>,
    scratch_inv: Vec<Complex<f64>>,
}

/// CWT computation method
#[derive(Debug, Clone, Copy)]
pub enum CwtMethod {
    /// Use FFT-based convolution (fast for large signals)
    Fft,
    /// Use direct convolution (simple but slower)
    Conv,
}

/// CWT output structure
#[derive(Debug, Clone)]
pub struct CwtOutput {
    /// CWT coefficients matrix (num_scales, signal_length)
    pub coefs: Array2<Complex64>,
    /// Frequencies corresponding to scales
    pub frequencies: Array1<f64>,
}

/// FFT plan cache for reusing FFT plans across scales
pub type FftPlanCache = Arc<HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>>;

/// Compute CWT for a single scale with buffer reuse
///
/// Corresponds to one iteration of the scale loop in pywt._cwt.cwt()
///
/// # Arguments
/// * `signal` - Input signal
/// * `scale` - Current scale value
/// * `int_psi` - Integrated wavelet function
/// * `x` - Wavelet domain points
/// * `method` - Computation method (FFT or Conv)
/// * `fft_cache` - Pre-computed FFT plans (optional, for performance)
/// * `buffers` - Reusable buffers to avoid heap allocations (optional)
///
/// # Algorithm
/// 1. Scale the wavelet: sample int_psi at scaled positions and reverse
/// 2. Convolve with signal using FFT
/// 3. Compute -sqrt(scale) * diff(conv)
/// 4. Crop to original signal length
///
/// # References
/// - pywt/_cwt.py lines 147-196
fn cwt_single_scale_with_buffers(
    signal: &Array1<f64>,
    scale: f64,
    int_psi: &Array1<Complex64>,
    x: &Array1<f64>,
    method: CwtMethod,
    fft_cache: Option<&FftPlanCache>,
    buffers: &mut CwtBuffers,
) -> Array1<Complex64> {
    let n = signal.len();

    // Step 1: Scale the wavelet (pywt lines 148-153)
    let step = x[1] - x[0];
    let x_range = x[x.len() - 1] - x[0];
    let num_samples = (scale * x_range + 1.0) as usize + 1;

    // Reuse j_indices buffer
    buffers.j_indices.clear();
    let inv_scale_step = 1.0 / (scale * step);
    for i in 0..num_samples {
        let j = ((i as f64) * inv_scale_step) as usize;
        if j < int_psi.len() {
            buffers.j_indices.push(j);
        }
    }

    let psi_len = buffers.j_indices.len();

    // Step 2: Convolution (pywt lines 166-180)
    let conv = match method {
        CwtMethod::Fft => {
            let conv_len = n + psi_len - 1;
            let fft_size = next_fast_len(conv_len);

            // Reuse signal_padded buffer
            buffers.signal_padded.clear();
            buffers.signal_padded.reserve(fft_size);
            for &x in signal.iter() {
                buffers.signal_padded.push(Complex::new(x, 0.0));
            }
            buffers.signal_padded.resize(fft_size, Complex::new(0.0, 0.0));

            // Reuse psi_padded buffer (sample int_psi at scaled indices, reversed)
            buffers.psi_padded.clear();
            buffers.psi_padded.reserve(fft_size);
            for &idx in buffers.j_indices.iter().rev() {
                let c = int_psi[idx];
                buffers.psi_padded.push(Complex::new(c.re, c.im));
            }
            buffers.psi_padded.resize(fft_size, Complex::new(0.0, 0.0));

            // Get FFT plans (from cache or create new)
            let (fft, ifft) = if let Some(cache) = fft_cache {
                if let Some((fft_plan, ifft_plan)) = cache.get(&fft_size) {
                    (Arc::clone(fft_plan), Arc::clone(ifft_plan))
                } else {
                    let mut planner = FftPlanner::new();
                    (planner.plan_fft_forward(fft_size), planner.plan_fft_inverse(fft_size))
                }
            } else {
                let mut planner = FftPlanner::new();
                (planner.plan_fft_forward(fft_size), planner.plan_fft_inverse(fft_size))
            };

            // Ensure scratch buffers are large enough
            let scratch_fwd_len = fft.get_inplace_scratch_len();
            let scratch_inv_len = ifft.get_inplace_scratch_len();
            if buffers.scratch_fwd.len() < scratch_fwd_len {
                buffers.scratch_fwd.resize(scratch_fwd_len, Complex::new(0.0, 0.0));
            }
            if buffers.scratch_inv.len() < scratch_inv_len {
                buffers.scratch_inv.resize(scratch_inv_len, Complex::new(0.0, 0.0));
            }

            // FFT with scratch buffers
            fft.process_with_scratch(&mut buffers.signal_padded, &mut buffers.scratch_fwd);
            fft.process_with_scratch(&mut buffers.psi_padded, &mut buffers.scratch_fwd);

            // Multiply in frequency domain (in-place in signal_padded)
            for (s, p) in buffers.signal_padded.iter_mut().zip(buffers.psi_padded.iter()) {
                *s = *s * *p;
            }

            // IFFT with scratch buffer
            ifft.process_with_scratch(&mut buffers.signal_padded, &mut buffers.scratch_inv);

            // Normalize and extract relevant portion
            let norm = 1.0 / (fft_size as f64);
            buffers.signal_padded
                .iter()
                .take(conv_len)
                .map(|&c| Complex64::new(c.re * norm, c.im * norm))
                .collect()
        }
        CwtMethod::Conv => {
            // Direct convolution - build int_psi_scale from j_indices
            let int_psi_scale: Array1<Complex64> = buffers.j_indices
                .iter()
                .rev()
                .map(|&idx| int_psi[idx])
                .collect();
            convolve_direct(signal, &int_psi_scale)
        }
    };

    // Step 3: Differentiate and scale (pywt line 182)
    let scale_factor = -(scale.sqrt());
    let coef = (&conv.slice(s![1..]) - &conv.slice(s![..-1])) * scale_factor;

    // Step 4: Crop to signal length (pywt lines 186-188)
    let d = ((coef.len() as f64) - (n as f64)) / 2.0;
    if d > 0.0 {
        let start = d.floor() as usize;
        let end = coef.len() - d.ceil() as usize;
        coef.slice(s![start..end]).to_owned()
    } else {
        coef
    }
}

/// Compute CWT for a single scale (public API without buffer reuse)
///
/// This is kept for backward compatibility and non-parallel use cases.
pub fn cwt_single_scale(
    signal: &Array1<f64>,
    scale: f64,
    int_psi: &Array1<Complex64>,
    x: &Array1<f64>,
    method: CwtMethod,
    fft_cache: Option<&FftPlanCache>,
) -> Array1<Complex64> {
    let mut buffers = CwtBuffers::default();
    cwt_single_scale_with_buffers(signal, scale, int_psi, x, method, fft_cache, &mut buffers)
}

/// Direct convolution (for reference, slower than FFT)
fn convolve_direct(signal: &Array1<f64>, kernel: &Array1<Complex64>) -> Array1<Complex64> {
    let n = signal.len();
    let m = kernel.len();
    let out_len = n + m - 1;
    let mut result = Array1::zeros(out_len);

    for i in 0..out_len {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..m {
            let sig_idx = i as isize - j as isize;
            if sig_idx >= 0 && (sig_idx as usize) < n {
                sum += Complex64::new(signal[sig_idx as usize], 0.0) * kernel[j];
            }
        }
        result[i] = sum;
    }

    result
}

/// Compute CWT for multiple scales (parallelized)
///
/// Uses rayon to parallelize computation across scales.
/// Each scale is computed independently using cwt_single_scale().
///
/// # Arguments
/// * `signal` - Input signal
/// * `scales` - Array of scale values
/// * `wavelet` - Wavelet to use
/// * `precision` - Wavelet precision (default: 12)
/// * `method` - Computation method (FFT or Conv)
/// * `verbose` - Print progress updates (default: false)
///
/// # Returns
/// 2D array of shape (num_scales, signal_length) containing CWT coefficients
///
/// # Performance
/// Parallelization provides near-linear speedup with number of CPU cores
/// since each scale computation is independent.
/// FFT plans are pre-computed and cached for all possible FFT sizes.
///
/// # References
/// - pywt/_cwt.py lines 147-196 (scale loop)
pub fn cwt_multi_scale(
    signal: &Array1<f64>,
    scales: &Array1<f64>,
    wavelet: &CmorWavelet,
    precision: usize,
    method: CwtMethod,
    verbose: bool,
) -> Array2<Complex64> {
    // Generate wavelet function once
    let (psi, x) = cmor_wavelet(wavelet.bandwidth, wavelet.center_freq, precision);
    let int_psi = integrate_wavelet(&psi, &x);

    // Conjugate for complex wavelets (pywt line 126)
    let int_psi: Array1<Complex64> = int_psi.iter().map(|c| c.conj()).collect();

    let n_scales = scales.len();
    let signal_len = signal.len();

    // Pre-compute FFT plans for all possible sizes (optimization for FFT method)
    let fft_cache = if matches!(method, CwtMethod::Fft) {
        let step = x[1] - x[0];
        let x_range = x[x.len() - 1] - x[0];

        // Collect all unique FFT sizes needed
        let mut fft_sizes = HashSet::new();
        for &scale in scales.iter() {
            // Mimic psi_len calculation from cwt_single_scale
            let num_samples = (scale * x_range + 1.0) as usize + 1;

            // Count how many indices will actually be used
            // Optimization: pre-compute inverse to avoid repeated division
            let inv_scale_step = 1.0 / (scale * step);
            let mut psi_len = 0;
            for i in 0..num_samples {
                let j = ((i as f64) * inv_scale_step) as usize;
                if j < int_psi.len() {
                    psi_len += 1;
                }
            }

            let conv_len = signal_len + psi_len - 1;
            let fft_size = next_fast_len(conv_len);
            fft_sizes.insert(fft_size);
        }

        // Create FFT plans for all unique sizes
        let mut planner = FftPlanner::new();
        let mut cache = HashMap::new();
        for &size in fft_sizes.iter() {
            let fft = planner.plan_fft_forward(size);
            let ifft = planner.plan_fft_inverse(size);
            cache.insert(size, (fft, ifft));
        }

        Some(Arc::new(cache))
    } else {
        None
    };

    // Parallel computation across scales (preserving order)
    // Optimization: use par_iter() with thread-local buffers to reduce allocations
    let scales_vec: Vec<f64> = scales.iter().cloned().collect();

    // Use thread_local! macro for buffer reuse across iterations within same thread
    thread_local! {
        static BUFFERS: RefCell<CwtBuffers> = RefCell::new(CwtBuffers::default());
    }

    let rows: Vec<(usize, Array1<Complex64>)> = scales_vec
        .par_iter()
        .enumerate()
        .map(|(i, &scale)| {
            if verbose && i % 10 == 0 && n_scales > 20 {
                eprintln!("Computing CWT: scale {}/{}", i + 1, n_scales);
            }
            // Get thread-local buffers
            BUFFERS.with(|buffers_cell| {
                let mut buffers = buffers_cell.borrow_mut();
                let row = cwt_single_scale_with_buffers(
                    signal, scale, &int_psi, &x, method, fft_cache.as_ref(), &mut buffers
                );
                (i, row)
            })
        })
        .collect();

    // Assemble into 2D array (sorted by index)
    let mut result = Array2::zeros((n_scales, signal_len));
    for (i, row) in rows {
        result.row_mut(i).assign(&row);
    }

    result
}

/// Full CWT computation with all pre/post processing
///
/// This is the main entry point that integrates all CWT components:
/// 1. Parse wavelet parameters
/// 2. Compute multi-scale CWT
/// 3. Calculate frequencies from scales
///
/// # Arguments
/// * `signal` - Input signal (NOT padded - padding done externally)
/// * `scales` - Array of scale values
/// * `wavelet_name` - Wavelet name (e.g., "cmor1.5-1.0")
/// * `sampling_period` - Sampling period for frequency conversion
/// * `precision` - Wavelet precision (default: 12)
/// * `verbose` - Print progress updates (default: false)
///
/// # Returns
/// CwtOutput containing:
/// - coefs: 2D complex array (num_scales, signal_length)
/// - frequencies: 1D array of frequencies corresponding to scales
///
/// # References
/// - pywt/_cwt.py: cwt() function
pub fn cwt(
    signal: &Array1<f64>,
    scales: &Array1<f64>,
    wavelet_name: &str,
    sampling_period: f64,
    precision: usize,
    verbose: bool,
) -> Result<CwtOutput, String> {
    // Step 1: Parse wavelet name
    let wavelet = CmorWavelet::from_name(wavelet_name)?;

    // Step 2: Compute multi-scale CWT (parallelized)
    let coefs = cwt_multi_scale(signal, scales, &wavelet, precision, CwtMethod::Fft, verbose);

    // Step 3: Calculate frequencies from scales
    // frequency = central_frequency / scale / sampling_period
    // For cmor, central_frequency ≈ center_freq
    let central_freq = wavelet.center_freq;
    let frequencies: Array1<f64> = scales
        .iter()
        .map(|&scale| central_freq / scale / sampling_period)
        .collect();

    Ok(CwtOutput { coefs, frequencies })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cwt_output_creation() {
        let output = CwtOutput {
            coefs: Array2::zeros((10, 100)),
            frequencies: Array1::zeros(10),
        };
        assert_eq!(output.coefs.shape(), &[10, 100]);
        assert_eq!(output.frequencies.len(), 10);
    }
}
