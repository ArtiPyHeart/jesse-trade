//! Utility functions for CWT
//!
//! Includes padding, dB conversion, and other helper functions.

use ndarray::{s, Array1};
use num_complex::Complex64;

/// Symmetric padding (corresponds to pywt.pad with mode='symmetric')
///
/// Implements the same algorithm as numpy.pad with mode='symmetric'.
///
/// The symmetric mode mirrors the signal at the edges, including the edge value.
///
/// # Algorithm
/// - Left padding: Take signal[0:pad_width], reverse it
/// - Right padding: Take signal[-pad_width:], reverse it
/// - If pad_width > len(signal), repeat the mirroring
///
/// # Examples
/// ```text
/// signal: [1, 2, 3, 4, 5], pad_width: 3
/// left pad: [3, 2, 1] (signal[2], signal[1], signal[0])
/// result: [3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3]
/// ```
///
/// # References
/// - numpy.pad documentation
/// - pywt.pad with mode='symmetric'
pub fn symmetric_pad(signal: &Array1<f64>, pad_width: usize) -> Array1<f64> {
    if pad_width == 0 {
        return signal.clone();
    }

    let n = signal.len();

    // Handle single element case
    if n == 1 {
        return Array1::from_elem(1 + 2 * pad_width, signal[0]);
    }

    let total_len = n + 2 * pad_width;
    let mut padded = Array1::zeros(total_len);

    // Copy original signal to center
    padded.slice_mut(s![pad_width..pad_width + n]).assign(signal);

    // Left padding: mirror signal[0:pad_width] in reverse
    // For pad_width >= n, we need to repeat the mirroring pattern
    for i in 0..pad_width {
        // Position in padded array
        let pad_pos = pad_width - 1 - i;

        // Map to original signal using reflection
        // The pattern repeats every (2*n - 2) steps
        let cycle_len = 2 * n - 2;
        let offset = i % cycle_len;

        let src_idx = if offset < n {
            offset
        } else {
            2 * n - 2 - offset
        };

        padded[pad_pos] = signal[src_idx];
    }

    // Right padding: mirror signal[-pad_width:] in reverse
    for i in 0..pad_width {
        // Position in padded array
        let pad_pos = pad_width + n + i;

        // Map to original signal using reflection
        let cycle_len = 2 * n - 2;
        let offset = i % cycle_len;

        let src_idx = if offset < n {
            n - 1 - offset
        } else {
            offset - n + 1
        };

        padded[pad_pos] = signal[src_idx];
    }

    padded
}

/// Convert complex CWT coefficients to dB scale
///
/// dB = 20 * log10(abs(cwt_coef) + epsilon)
pub fn to_db(cwtmat: &ndarray::Array2<Complex64>, epsilon: f64) -> ndarray::Array2<f64> {
    cwtmat.mapv(|c| 20.0 * (c.norm() + epsilon).log10())
}

/// Compute next power of 2 (for FFT optimization)
pub fn next_fast_len(n: usize) -> usize {
    let log2 = (n as f64).log2().ceil() as u32;
    2_usize.pow(log2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_fast_len() {
        assert_eq!(next_fast_len(100), 128);
        assert_eq!(next_fast_len(128), 128);
        assert_eq!(next_fast_len(129), 256);
        assert_eq!(next_fast_len(1000), 1024);
    }

    #[test]
    fn test_to_db() {
        use ndarray::arr2;
        let cwtmat = arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(10.0, 0.0)],
            [Complex64::new(100.0, 0.0), Complex64::new(1000.0, 0.0)],
        ]);
        let db = to_db(&cwtmat, 1e-12);

        assert!((db[[0, 0]] - 0.0).abs() < 1e-10);    // 20*log10(1) = 0
        assert!((db[[0, 1]] - 20.0).abs() < 1e-10);   // 20*log10(10) = 20
        assert!((db[[1, 0]] - 40.0).abs() < 1e-10);   // 20*log10(100) = 40
        assert!((db[[1, 1]] - 60.0).abs() < 1e-10);   // 20*log10(1000) = 60
    }

    #[test]
    fn test_symmetric_pad_basic() {
        use ndarray::arr1;

        // Basic test
        let signal = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let padded = symmetric_pad(&signal, 2);

        assert_eq!(padded.len(), 9);
        assert_eq!(padded[0], 2.0);
        assert_eq!(padded[1], 1.0);
        assert_eq!(padded[2], 1.0); // signal[0]
        assert_eq!(padded[6], 5.0); // signal[-1]
        assert_eq!(padded[7], 5.0);
        assert_eq!(padded[8], 4.0);
    }

    #[test]
    fn test_symmetric_pad_zero_width() {
        use ndarray::arr1;

        let signal = arr1(&[1.0, 2.0, 3.0]);
        let padded = symmetric_pad(&signal, 0);

        assert_eq!(padded.len(), 3);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[1], 2.0);
        assert_eq!(padded[2], 3.0);
    }
}
