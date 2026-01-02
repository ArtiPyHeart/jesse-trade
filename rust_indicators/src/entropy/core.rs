//! Entropy 核心算法实现
//!
//! 提供 Approximate Entropy (ApEn) 和 Sample Entropy (SampEn) 的高性能实现。

use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

/// 计算数据范围（range 或 std）
///
/// # Arguments
/// * `x` - 输入数据
/// * `use_std` - true: 使用标准差, false: 使用 max-min
fn data_range(x: &ArrayView1<f64>, use_std: bool) -> f64 {
    if x.is_empty() {
        return 0.0;
    }

    if use_std {
        // 总体标准差 (ddof=0)
        let n = x.len() as f64;
        let mean = x.sum() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    } else {
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }
}

/// 切比雪夫距离（最大绝对差）
#[inline]
fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// ApEn 的 phi 函数
///
/// phi(m, r) = mean(log(C_i^m(r)))
/// C_i^m(r) = count(dist <= r) / (N - m + 1), 包含自匹配
fn phi(x: &[f64], m: usize, r: f64) -> f64 {
    let n = x.len();
    if n < m + 1 {
        return 0.0;
    }

    let count = n - m + 1;
    let mut sum = 0.0;

    for i in 0..count {
        let template = &x[i..i + m];
        let mut matches = 0usize;

        // 包含自匹配 (j == i 时也算)
        for j in 0..count {
            if chebyshev_distance(template, &x[j..j + m]) <= r {
                matches += 1;
            }
        }

        // matches 至少为 1（自匹配），所以 log 不会是 -inf
        if matches > 0 {
            sum += (matches as f64 / count as f64).ln();
        }
    }

    sum / count as f64
}

/// Approximate Entropy (ApEn)
///
/// ApEn(m, r) = phi(m, r) - phi(m+1, r)
///
/// # Arguments
/// * `x` - 输入时间序列
/// * `m` - 嵌入维度 (通常为 2)
/// * `r_ratio` - 容忍度比例 (通常为 0.2-0.3)
/// * `use_std` - true: r = r_ratio * std(x), false: r = r_ratio * (max-min)
///
/// # Returns
/// ApEn 值
pub fn approximate_entropy(x: &Array1<f64>, m: usize, r_ratio: f64, use_std: bool) -> f64 {
    let r = r_ratio * data_range(&x.view(), use_std);
    if r <= 0.0 {
        return f64::NAN;
    }

    let x_slice = x.as_slice().unwrap();
    phi(x_slice, m, r) - phi(x_slice, m + 1, r)
}

/// SampEn 的配对计数（上三角，不含自匹配）
///
/// 计算满足 dist(X[i], X[j]) <= r 的 (i, j) 对数量，其中 i < j
fn count_pairs(embeddings: &[Vec<f64>], r: f64) -> usize {
    let n = embeddings.len();
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            if chebyshev_distance(&embeddings[i], &embeddings[j]) <= r {
                count += 1;
            }
        }
    }

    count
}

/// Sample Entropy (SampEn)
///
/// SampEn(m, r) = -ln(A / B)
/// 其中:
/// - B: m 维模板匹配数（不含自匹配）
/// - A: (m+1) 维模板匹配数（不含自匹配）
///
/// # Arguments
/// * `x` - 输入时间序列
/// * `m` - 嵌入维度 (通常为 2)
/// * `r_ratio` - 容忍度比例 (通常为 0.2-0.3)
/// * `use_std` - true: r = r_ratio * std(x), false: r = r_ratio * (max-min)
///
/// # Returns
/// SampEn 值，如果 A=0 或 B=0 则返回 NaN
pub fn sample_entropy(x: &Array1<f64>, m: usize, r_ratio: f64, use_std: bool) -> f64 {
    let n = x.len();
    if n < m + 2 {
        return f64::NAN;
    }

    let r = r_ratio * data_range(&x.view(), use_std);
    if r <= 0.0 {
        return f64::NAN;
    }

    let x_slice = x.as_slice().unwrap();

    // 按照 Richman & Moorman 定义，m 维和 m+1 维使用相同的模板索引范围
    // 构建 m 维嵌入向量 (N-m 个)
    let xm: Vec<Vec<f64>> = (0..n - m)
        .map(|i| x_slice[i..i + m].to_vec())
        .collect();

    // 构建 (m+1) 维嵌入向量 (N-m 个，与 m 维相同数量)
    let xm1: Vec<Vec<f64>> = (0..n - m)
        .map(|i| x_slice[i..i + m + 1].to_vec())
        .collect();

    let bm = count_pairs(&xm, r);
    let am = count_pairs(&xm1, r);

    if am == 0 || bm == 0 {
        f64::NAN
    } else {
        -((am as f64) / (bm as f64)).ln()
    }
}

// ============================================================================
// 内部切片版本（用于 rolling 计算，避免 Array1 分配开销）
// ============================================================================

/// 计算切片的数据范围
fn data_range_slice(x: &[f64], use_std: bool) -> f64 {
    if x.is_empty() {
        return 0.0;
    }

    if use_std {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    } else {
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }
}

/// 切片版本的 ApEn（用于 rolling）
fn approximate_entropy_slice(x: &[f64], m: usize, r_ratio: f64, use_std: bool) -> f64 {
    let r = r_ratio * data_range_slice(x, use_std);
    if r <= 0.0 {
        return f64::NAN;
    }
    phi(x, m, r) - phi(x, m + 1, r)
}

/// 切片版本的 SampEn（用于 rolling）
fn sample_entropy_slice(x: &[f64], m: usize, r_ratio: f64, use_std: bool) -> f64 {
    let n = x.len();
    if n < m + 2 {
        return f64::NAN;
    }

    let r = r_ratio * data_range_slice(x, use_std);
    if r <= 0.0 {
        return f64::NAN;
    }

    // 构建嵌入向量
    let xm: Vec<Vec<f64>> = (0..n - m).map(|i| x[i..i + m].to_vec()).collect();
    let xm1: Vec<Vec<f64>> = (0..n - m).map(|i| x[i..i + m + 1].to_vec()).collect();

    let bm = count_pairs(&xm, r);
    let am = count_pairs(&xm1, r);

    if am == 0 || bm == 0 {
        f64::NAN
    } else {
        -((am as f64) / (bm as f64)).ln()
    }
}

// ============================================================================
// 滑动窗口并行计算（Rayon）
// ============================================================================

/// 滑动窗口 Approximate Entropy 计算（Rayon 并行）
///
/// 对输入序列进行纯粹的滑动窗口 entropy 计算，不包含任何数据预处理。
///
/// # Arguments
/// * `data` - 输入序列（任意数据，如 log returns）
/// * `period` - 滑动窗口大小
/// * `m` - 嵌入维度
/// * `r_ratio` - 容忍度比例
/// * `use_std` - true: 使用 std 模式, false: 使用 range 模式
///
/// # Returns
/// 熵值数组，前 (period-1) 个位置为 NaN
pub fn approximate_entropy_rolling(
    data: &Array1<f64>,
    period: usize,
    m: usize,
    r_ratio: f64,
    use_std: bool,
) -> Array1<f64> {
    let n = data.len();
    let data_slice = data.as_slice().unwrap();

    // 边界检查：需要至少 period 个数据点
    if n < period {
        return Array1::from_elem(n, f64::NAN);
    }

    // 并行计算：从第 (period-1) 个位置开始，每个窗口包含 period 个点
    let results: Vec<f64> = (period - 1..n)
        .into_par_iter()
        .map(|end_idx| {
            let start_idx = end_idx + 1 - period;
            let window = &data_slice[start_idx..=end_idx];
            approximate_entropy_slice(window, m, r_ratio, use_std)
        })
        .collect();

    // 构建结果数组（前 period-1 个为 NaN）
    let mut output = Array1::from_elem(n, f64::NAN);
    for (i, &val) in results.iter().enumerate() {
        output[period - 1 + i] = val;
    }
    output
}

/// 滑动窗口 Sample Entropy 计算（Rayon 并行）
///
/// 对输入序列进行纯粹的滑动窗口 entropy 计算，不包含任何数据预处理。
///
/// # Arguments
/// * `data` - 输入序列（任意数据，如 log returns）
/// * `period` - 滑动窗口大小
/// * `m` - 嵌入维度
/// * `r_ratio` - 容忍度比例
/// * `use_std` - true: 使用 std 模式, false: 使用 range 模式
///
/// # Returns
/// 熵值数组，前 (period-1) 个位置为 NaN
pub fn sample_entropy_rolling(
    data: &Array1<f64>,
    period: usize,
    m: usize,
    r_ratio: f64,
    use_std: bool,
) -> Array1<f64> {
    let n = data.len();
    let data_slice = data.as_slice().unwrap();

    // 边界检查：需要至少 period 个数据点
    if n < period {
        return Array1::from_elem(n, f64::NAN);
    }

    // 并行计算：从第 (period-1) 个位置开始，每个窗口包含 period 个点
    let results: Vec<f64> = (period - 1..n)
        .into_par_iter()
        .map(|end_idx| {
            let start_idx = end_idx + 1 - period;
            let window = &data_slice[start_idx..=end_idx];
            sample_entropy_slice(window, m, r_ratio, use_std)
        })
        .collect();

    // 构建结果数组（前 period-1 个为 NaN）
    let mut output = Array1::from_elem(n, f64::NAN);
    for (i, &val) in results.iter().enumerate() {
        output[period - 1 + i] = val;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_data_range_range_mode() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let range = data_range(&x.view(), false);
        assert_relative_eq!(range, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_data_range_std_mode() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let std_val = data_range(&x.view(), true);
        // std([1,2,3,4,5], ddof=0) = sqrt(2) ≈ 1.4142
        assert_relative_eq!(std_val, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.5, 5.0];
        let dist = chebyshev_distance(&a, &b);
        assert_relative_eq!(dist, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_approximate_entropy_basic() {
        let x = Array1::from_vec(vec![
            0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
            1.0, 2.0, 1.0,
        ]);
        let apen = approximate_entropy(&x, 2, 0.3, false);
        // 周期序列应有较低的 ApEn
        assert!(apen.is_finite());
        assert!(apen >= 0.0);
    }

    #[test]
    fn test_sample_entropy_basic() {
        let x = Array1::from_vec(vec![
            0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
            1.0, 2.0, 1.0,
        ]);
        let sampen = sample_entropy(&x, 2, 0.3, false);
        // 周期序列应有较低的 SampEn
        assert!(sampen.is_finite());
    }

    #[test]
    fn test_short_input() {
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let apen = approximate_entropy(&x, 2, 0.3, false);
        let sampen = sample_entropy(&x, 2, 0.3, false);
        // 短序列应返回 NaN 或 0
        assert!(apen.is_nan() || apen == 0.0);
        assert!(sampen.is_nan());
    }
}
