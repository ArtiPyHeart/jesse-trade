//! Core CWT algorithm implementation
//!
//! Implements continuous wavelet transform using FFT-based convolution.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

use super::wavelets::{cmor_wavelet, integrate_wavelet, CmorWavelet};
use super::utils::next_fast_len;

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

/// Compute CWT for a single scale
///
/// Corresponds to one iteration of the scale loop in pywt._cwt.cwt()
///
/// # Arguments
/// * `signal` - Input signal
/// * `scale` - Current scale value
/// * `int_psi` - Integrated wavelet function
/// * `x` - Wavelet domain points
/// * `method` - Computation method (FFT or Conv)
///
/// # Algorithm
/// 1. Scale the wavelet: sample int_psi at scaled positions and reverse
/// 2. Convolve with signal using FFT
/// 3. Compute -sqrt(scale) * diff(conv)
/// 4. Crop to original signal length
///
/// # References
/// - pywt/_cwt.py lines 147-196
pub fn cwt_single_scale(
    signal: &Array1<f64>,
    scale: f64,
    int_psi: &Array1<Complex64>,
    x: &Array1<f64>,
    method: CwtMethod,
) -> Array1<Complex64> {
    let n = signal.len();

    // Step 1: Scale the wavelet (pywt lines 148-153)
    let step = x[1] - x[0];
    let x_range = x[x.len() - 1] - x[0];
    // Python: np.arange(scale * x_range + 1) generates [0, 1, ..., floor(scale * x_range)]
    // which is floor(scale * x_range) + 1 elements
    let num_samples = (scale * x_range + 1.0) as usize + 1;

    // Generate scaled indices: j = arange(...) / (scale * step)
    let mut j_indices: Vec<usize> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let j_float = (i as f64) / (scale * step);
        let j = j_float.floor() as usize;
        if j < int_psi.len() {
            j_indices.push(j);
        }
    }

    // Sample int_psi at scaled indices and reverse
    let int_psi_scale: Array1<Complex64> = j_indices
        .iter()
        .rev()
        .map(|&idx| int_psi[idx])
        .collect();

    let psi_len = int_psi_scale.len();

    // Step 2: Convolution (pywt lines 166-180)
    let conv = match method {
        CwtMethod::Fft => {
            // FFT-based convolution
            let conv_len = n + psi_len - 1;
            let fft_size = next_fast_len(conv_len);

            // Convert signal to complex and pad
            let mut signal_padded: Vec<Complex<f64>> = signal
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            signal_padded.resize(fft_size, Complex::new(0.0, 0.0));

            // Pad wavelet
            let mut psi_padded: Vec<Complex<f64>> = int_psi_scale
                .iter()
                .map(|&c| Complex::new(c.re, c.im))
                .collect();
            psi_padded.resize(fft_size, Complex::new(0.0, 0.0));

            // FFT
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(fft_size);

            fft.process(&mut signal_padded);
            fft.process(&mut psi_padded);

            // Multiply in frequency domain
            let mut product: Vec<Complex<f64>> = signal_padded
                .iter()
                .zip(psi_padded.iter())
                .map(|(&s, &p)| s * p)
                .collect();

            // IFFT
            let ifft = planner.plan_fft_inverse(fft_size);
            ifft.process(&mut product);

            // Normalize and extract relevant portion
            let norm = 1.0 / (fft_size as f64);
            product
                .iter()
                .take(conv_len)
                .map(|&c| Complex64::new(c.re * norm, c.im * norm))
                .collect()
        }
        CwtMethod::Conv => {
            // Direct convolution (simpler but slower)
            convolve_direct(signal, &int_psi_scale)
        }
    };

    // Step 3: Differentiate and scale (pywt line 182)
    // coef = -sqrt(scale) * diff(conv)
    let scale_factor = -(scale.sqrt());
    let mut coef = Array1::zeros(conv.len() - 1);
    for i in 0..coef.len() {
        coef[i] = scale_factor * (conv[i + 1] - conv[i]);
    }

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

    // Parallel computation across scales (preserving order)
    let rows: Vec<(usize, Array1<Complex64>)> = scales
        .iter()
        .enumerate()
        .par_bridge()
        .map(|(i, &scale)| {
            if verbose && i % 10 == 0 && n_scales > 20 {
                // Progress indication for large jobs (dev mode only)
                eprintln!("Computing CWT: scale {}/{}", i + 1, n_scales);
            }
            let row = cwt_single_scale(signal, scale, &int_psi, &x, method);
            (i, row)  // Return index with result to preserve order
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
