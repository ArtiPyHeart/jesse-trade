//! Wavelet function generation
//!
//! This module implements various continuous wavelet functions,
//! with primary focus on Complex Morlet (cmor) wavelet.

use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Complex Morlet wavelet parameters
#[derive(Debug, Clone)]
pub struct CmorWavelet {
    /// Bandwidth parameter (e.g., 1.5)
    pub bandwidth: f64,
    /// Center frequency (e.g., 1.0)
    pub center_freq: f64,
}

impl CmorWavelet {
    /// Create a new Complex Morlet wavelet
    pub fn new(bandwidth: f64, center_freq: f64) -> Self {
        Self {
            bandwidth,
            center_freq,
        }
    }

    /// Parse from wavelet name string (e.g., "cmor1.5-1.0")
    pub fn from_name(name: &str) -> Result<Self, String> {
        if !name.starts_with("cmor") {
            return Err(format!("Invalid wavelet name: {}", name));
        }

        let parts: Vec<&str> = name[4..].split('-').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid cmor format: {}", name));
        }

        let bandwidth = parts[0]
            .parse::<f64>()
            .map_err(|_| format!("Invalid bandwidth: {}", parts[0]))?;
        let center_freq = parts[1]
            .parse::<f64>()
            .map_err(|_| format!("Invalid center frequency: {}", parts[1]))?;

        Ok(Self::new(bandwidth, center_freq))
    }
}

/// Generate Complex Morlet wavelet function
///
/// Returns (psi, x) where:
/// - psi: Complex wavelet function values
/// - x: Time/space domain points
///
/// # Arguments
/// * `bandwidth` - Bandwidth parameter (FB in pywt, e.g., 1.5)
/// * `center_freq` - Center frequency (FC in pywt, e.g., 1.0)
/// * `precision` - Precision level (default: 12, meaning 2^12 = 4096 points)
///
/// # Formula
/// Based on pywt's C implementation (c/cwt.c):
/// ```text
/// psi_r(x) = cos(2*pi*FC*x) * exp(-x^2/FB) / sqrt(pi*FB)
/// psi_i(x) = sin(2*pi*FC*x) * exp(-x^2/FB) / sqrt(pi*FB)
/// ```
///
/// # References
/// - pywt/c/cwt.c: `_cmor()` function
/// - pywt/_functions.py: `integrate_wavelet()`
pub fn cmor_wavelet(
    bandwidth: f64,
    center_freq: f64,
    precision: usize,
) -> (Array1<Complex64>, Array1<f64>) {
    // Number of points: 2^precision
    let n = 2_usize.pow(precision as u32);

    // Generate x domain (same range as pywt)
    // PyWavelets uses a domain range that depends on the wavelet
    // For cmor, typical range is around [-8, 8]
    let x = Array1::linspace(-8.0, 8.0, n);

    // Pre-compute constants
    let fb = bandwidth;
    let fc = center_freq;
    let norm_factor = 1.0 / (PI * fb).sqrt();
    let two_pi_fc = 2.0 * PI * fc;

    // Compute complex wavelet: psi(x) = psi_r(x) + i*psi_i(x)
    let psi = x.mapv(|xi: f64| {
        let exp_term = (-xi * xi / fb).exp() * norm_factor;
        let angle = two_pi_fc * xi;
        let psi_r = angle.cos() * exp_term;
        let psi_i = angle.sin() * exp_term;
        Complex64::new(psi_r, psi_i)
    });

    (psi, x)
}

/// Integrate wavelet function (cumulative sum with step)
///
/// Corresponds to pywt._functions.integrate_wavelet
///
/// # Formula
/// ```text
/// integral[i] = sum(psi[0..i]) * step
/// ```
/// where step = x[1] - x[0]
pub fn integrate_wavelet(
    psi: &Array1<Complex64>,
    x: &Array1<f64>,
) -> Array1<Complex64> {
    let step = x[1] - x[0];
    let mut integral = Array1::zeros(psi.len());
    let mut cumsum = Complex64::new(0.0, 0.0);

    for i in 0..psi.len() {
        cumsum += psi[i];
        integral[i] = cumsum * step;
    }

    integral
}

/// Get central frequency of the wavelet
///
/// For cmor wavelet, this corresponds to pywt.scale2frequency
/// The central frequency is directly related to the center_freq parameter.
///
/// # Formula
/// For cmor wavelet: central_frequency ≈ center_freq
pub fn central_frequency(wavelet: &CmorWavelet, _precision: usize) -> f64 {
    // For Complex Morlet, the central frequency is approximately
    // equal to the center_freq parameter
    // This is a simplified version; pywt uses FFT to find the peak
    wavelet.center_freq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmor_parse() {
        let wavelet = CmorWavelet::from_name("cmor1.5-1.0").unwrap();
        assert_eq!(wavelet.bandwidth, 1.5);
        assert_eq!(wavelet.center_freq, 1.0);
    }

    #[test]
    fn test_cmor_invalid() {
        assert!(CmorWavelet::from_name("invalid").is_err());
        assert!(CmorWavelet::from_name("cmor1.5").is_err());
    }
}
