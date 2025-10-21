//! NRBO 核心算法实现
//!
//! 对应 Python nrbo.py 的实现

use ndarray::{Array1, ArrayView1};

/// 查找极值点的索引
///
/// 对应 Python 中的 scipy.signal.find_peaks
fn find_peaks(signal: &ArrayView1<f64>) -> Vec<usize> {
    let n = signal.len();
    if n < 3 {
        return Vec::new();
    }

    let mut peaks = Vec::new();

    for i in 1..(n - 1) {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

/// 查找波谷的索引
fn find_valleys(signal: &ArrayView1<f64>) -> Vec<usize> {
    let n = signal.len();
    if n < 3 {
        return Vec::new();
    }

    let mut valleys = Vec::new();

    for i in 1..(n - 1) {
        if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
            valleys.push(i);
        }
    }

    valleys
}

/// 合并并排序极值点索引
///
/// 对应 Python 中的 `_find_extrema_indices`
fn find_extrema_indices(peaks: &[usize], valleys: &[usize]) -> Vec<usize> {
    let mut extrema = Vec::with_capacity(peaks.len() + valleys.len());
    extrema.extend_from_slice(peaks);
    extrema.extend_from_slice(valleys);
    extrema.sort_unstable();
    extrema
}

/// 更新边界点
///
/// 对应 Python 中的 `_update_boundary_points`
///
/// # Returns
/// (new_left, new_right, converged)
fn update_boundary_points(
    imf: &ArrayView1<f64>,
    left_idx: usize,
    right_idx: usize,
    tol: f64,
) -> (f64, f64, bool) {
    let n = imf.len();

    // 边界检查
    if left_idx == 0 || left_idx >= n - 1 || right_idx == 0 || right_idx >= n - 1 {
        return (imf[left_idx], imf[right_idx], false);
    }

    // 一阶导数（中心差分）
    let df_left = (imf[left_idx + 1] - imf[left_idx - 1]) * 0.5;
    let df_right = (imf[right_idx + 1] - imf[right_idx - 1]) * 0.5;

    // 二阶导数（中心差分）
    let d2f_left = imf[left_idx + 1] - 2.0 * imf[left_idx] + imf[left_idx - 1];
    let d2f_right = imf[right_idx + 1] - 2.0 * imf[right_idx] + imf[right_idx - 1];

    // 避免除零错误
    if d2f_left.abs() < 1e-10 || d2f_right.abs() < 1e-10 {
        return (imf[left_idx], imf[right_idx], false);
    }

    // Newton-Raphson 更新
    let new_left = imf[left_idx] - df_left / d2f_left;
    let new_right = imf[right_idx] - df_right / d2f_right;

    // 检查收敛
    let converged = (new_left - imf[left_idx]).abs() < tol && (new_right - imf[right_idx]).abs() < tol;

    (new_left, new_right, converged)
}

/// Newton-Raphson Boundary Optimization
///
/// 完全对应 Python nrbo.nrbo 的实现
///
/// # Arguments
/// * `imf` - 输入的 IMF (本征模态函数)
/// * `max_iter` - 最大迭代次数 (default: 10)
/// * `tol` - 收敛容差 (default: 1e-6)
///
/// # Returns
/// 优化后的 IMF
pub fn nrbo(imf: &Array1<f64>, max_iter: usize, tol: f64) -> Array1<f64> {
    // 输入验证
    if imf.len() < 3 {
        return imf.clone();
    }

    // 创建副本以避免修改原始数据
    let mut imf_copy = imf.clone();

    for _iteration in 0..max_iter {
        // 查找极值点
        let peaks = find_peaks(&imf_copy.view());
        let valleys = find_valleys(&imf_copy.view());

        // 合并极值点
        let extrema = find_extrema_indices(&peaks, &valleys);

        if extrema.len() < 2 {
            break;
        }

        // 获取边界极值点
        let left_extrema = extrema[0];
        let right_extrema = extrema[extrema.len() - 1];

        // 更新边界点
        let (new_left, new_right, converged) =
            update_boundary_points(&imf_copy.view(), left_extrema, right_extrema, tol);

        if converged {
            break;
        }

        // 更新边界点的值
        imf_copy[left_extrema] = new_left;
        imf_copy[right_extrema] = new_right;
    }

    imf_copy
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_find_peaks() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
        let peaks = find_peaks(&signal.view());

        assert_eq!(peaks, vec![1, 3]);
    }

    #[test]
    fn test_find_valleys() {
        let signal = Array1::from_vec(vec![3.0, 1.0, 2.0, 0.5, 3.0]);
        let valleys = find_valleys(&signal.view());

        assert_eq!(valleys, vec![1, 3]);
    }

    #[test]
    fn test_nrbo_basic() {
        let imf = Array1::from_vec(vec![1.0, 2.0, 1.0, 3.0, 1.0]);
        let result = nrbo(&imf, 10, 1e-6);

        // 应该返回相同长度的数组
        assert_eq!(result.len(), imf.len());
    }

    #[test]
    fn test_nrbo_short_input() {
        let imf = Array1::from_vec(vec![1.0, 2.0]);
        let result = nrbo(&imf, 10, 1e-6);

        // 短输入应该直接返回
        assert_eq!(result.to_vec(), imf.to_vec());
    }
}
