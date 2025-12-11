//! 工具函数
//!
//! FFT/IFFT 辅助函数和数组操作

use lazy_static::lazy_static;
use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// FFT Plan Cache type
/// Arc allows zero-cost sharing across threads
pub type FftPlanCache = Arc<HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>>;

// ============================================================
// 全局 FFT Plan 缓存（跨调用复用，避免重复规划）
// ============================================================

lazy_static! {
    /// 全局 FFT Plan 缓存
    /// - Key: FFT 大小
    /// - Value: (forward_plan, inverse_plan)
    static ref GLOBAL_FFT_CACHE: RwLock<HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>>
        = RwLock::new(HashMap::new());
}

/// 获取或创建指定大小的 FFT Plan（全局缓存）
///
/// 使用读写锁实现高效并发：
/// - 缓存命中时仅需读锁（多读者并发）
/// - 缓存未命中时升级为写锁创建
pub fn get_global_fft_plan(size: usize) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>) {
    // 先尝试读锁（快速路径）
    {
        let cache = GLOBAL_FFT_CACHE.read().unwrap();
        if let Some(plans) = cache.get(&size) {
            return (Arc::clone(&plans.0), Arc::clone(&plans.1));
        }
    }

    // 缓存未命中，获取写锁创建
    let mut cache = GLOBAL_FFT_CACHE.write().unwrap();

    // 双重检查（另一个线程可能已经创建）
    if let Some(plans) = cache.get(&size) {
        return (Arc::clone(&plans.0), Arc::clone(&plans.1));
    }

    // 创建新的 FFT Plan
    let mut planner = FftPlanner::new();
    let forward = planner.plan_fft_forward(size);
    let inverse = planner.plan_fft_inverse(size);
    let plans = (forward, inverse);

    cache.insert(size, (Arc::clone(&plans.0), Arc::clone(&plans.1)));
    plans
}

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
///
/// # Arguments
/// * `input` - Input array
/// * `fft_cache` - Optional FFT plan cache for performance (falls back to global cache)
pub fn fft(input: &Array1<Complex64>, fft_cache: Option<&FftPlanCache>) -> Array1<Complex64> {
    let n = input.len();

    // Get FFT plan (from local cache → global cache)
    let fft_plan = if let Some(cache) = fft_cache {
        if let Some((fft_plan, _)) = cache.get(&n) {
            Arc::clone(fft_plan)
        } else {
            // Fallback to global cache
            get_global_fft_plan(n).0
        }
    } else {
        // Use global cache
        get_global_fft_plan(n).0
    };

    let mut buffer = input.to_vec();
    fft_plan.process(&mut buffer);

    Array1::from_vec(buffer)
}

/// IFFT 包装函数
///
/// # Arguments
/// * `input` - Input array
/// * `fft_cache` - Optional FFT plan cache for performance (falls back to global cache)
pub fn ifft(input: &Array1<Complex64>, fft_cache: Option<&FftPlanCache>) -> Array1<Complex64> {
    let n = input.len();

    // Get IFFT plan (from local cache → global cache)
    let ifft_plan = if let Some(cache) = fft_cache {
        if let Some((_, ifft_plan)) = cache.get(&n) {
            Arc::clone(ifft_plan)
        } else {
            // Fallback to global cache
            get_global_fft_plan(n).1
        }
    } else {
        // Use global cache
        get_global_fft_plan(n).1
    };

    let mut buffer = input.to_vec();
    ifft_plan.process(&mut buffer);

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

        let freq = fft(&input, None);
        let reconstructed = ifft(&freq, None);

        for i in 0..input.len() {
            assert_relative_eq!(reconstructed[i].re, input[i].re, epsilon = 1e-10);
            assert_relative_eq!(reconstructed[i].im, input[i].im, epsilon = 1e-10);
        }
    }
}
