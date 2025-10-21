//! 工具函数
//!
//! FFT/IFFT 辅助函数和数组操作

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// FFT Shift (对应 numpy.fft.fftshift)
///
/// 将零频率分量移到频谱中心
///
/// 对于偶数长度 n: 移动 n/2 个位置
/// 对于奇数长度 n: 移动 (n+1)/2 个位置
pub fn fftshift<T: Clone>(arr: &ArrayView1<T>) -> Array1<T> {
    let n = arr.len();
    let mid = (n + 1) / 2;

    let mut result = Array1::uninit(n);

    // 复制后半部分到前面
    for i in 0..(n - mid) {
        result[i].write(arr[mid + i].clone());
    }

    // 复制前半部分到后面
    for i in 0..mid {
        result[n - mid + i].write(arr[i].clone());
    }

    unsafe { result.assume_init() }
}

/// IFFT Shift (对应 numpy.fft.ifftshift)
///
/// fftshift 的逆操作
///
/// 对于偶数长度 n: 移动 n/2 个位置
/// 对于奇数长度 n: 移动 n/2 个位置
pub fn ifftshift<T: Clone>(arr: &ArrayView1<T>) -> Array1<T> {
    let n = arr.len();
    let mid = n / 2;

    let mut result = Array1::uninit(n);

    // 复制后半部分到前面
    for i in 0..(n - mid) {
        result[i].write(arr[mid + i].clone());
    }

    // 复制前半部分到后面
    for i in 0..mid {
        result[n - mid + i].write(arr[i].clone());
    }

    unsafe { result.assume_init() }
}

/// FFT 包装函数
pub fn fft(input: &Array1<Complex64>) -> Array1<Complex64> {
    let n = input.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer = input.to_vec();
    fft.process(&mut buffer);

    Array1::from_vec(buffer)
}

/// IFFT 包装函数
pub fn ifft(input: &Array1<Complex64>) -> Array1<Complex64> {
    let n = input.len();
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);

    let mut buffer = input.to_vec();
    ifft.process(&mut buffer);

    // 归一化
    for x in &mut buffer {
        *x /= n as f64;
    }

    Array1::from_vec(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fftshift_even() {
        // 偶数长度: [0, 1, 2, 3, 4, 5] -> [3, 4, 5, 0, 1, 2]
        let arr = Array1::from_vec(vec![0, 1, 2, 3, 4, 5]);
        let shifted = fftshift(&arr.view());
        assert_eq!(shifted.to_vec(), vec![3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn test_fftshift_odd() {
        // 奇数长度: [0, 1, 2, 3, 4] -> [3, 4, 0, 1, 2]
        let arr = Array1::from_vec(vec![0, 1, 2, 3, 4]);
        let shifted = fftshift(&arr.view());
        assert_eq!(shifted.to_vec(), vec![3, 4, 0, 1, 2]);
    }

    #[test]
    fn test_fftshift_ifftshift_inverse() {
        // fftshift 和 ifftshift 应该互为逆操作
        let arr = Array1::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let shifted = fftshift(&arr.view());
        let back = ifftshift(&shifted.view());
        assert_eq!(arr.to_vec(), back.to_vec());
    }

    #[test]
    fn test_fft_ifft_inverse() {
        let input = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);

        let freq = fft(&input);
        let reconstructed = ifft(&freq);

        for i in 0..input.len() {
            assert_relative_eq!(reconstructed[i].re, input[i].re, epsilon = 1e-10);
            assert_relative_eq!(reconstructed[i].im, input[i].im, epsilon = 1e-10);
        }
    }
}
