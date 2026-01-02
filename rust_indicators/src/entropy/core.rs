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
// Shannon entropy / self-information helpers
// ============================================================================

/// 计算样本均值和标准差（ddof=0）
fn mean_and_std(x: &[f64]) -> Option<(f64, f64)> {
    if x.is_empty() {
        return None;
    }

    let mut sum = 0.0;
    for &v in x {
        if !v.is_finite() {
            return None;
        }
        sum += v;
    }
    let n = x.len() as f64;
    let mean = sum / n;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if variance <= 0.0 {
        return None;
    }
    Some((mean, variance.sqrt()))
}

/// Gaussian NLL (包含 log σ) 的单点 self-information
fn gaussian_surprisal(value: f64, mean: f64, std: f64) -> f64 {
    let z = (value - mean) / std;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    0.5 * (log_2pi + z * z) + std.ln()
}

/// Shannon entropy (Gaussian NLL, 包含 log σ)
///
/// 返回整段序列的平均 NLL（nats）。如果 std == 0，则返回 NaN。
pub fn shannon_entropy_gaussian(x: &Array1<f64>) -> f64 {
    let x_slice = x.as_slice().unwrap();
    let (mean, std) = match mean_and_std(x_slice) {
        Some(res) => res,
        None => return f64::NAN,
    };

    let mut nll_sum = 0.0;
    for &v in x_slice {
        nll_sum += gaussian_surprisal(v, mean, std);
    }
    nll_sum / x_slice.len() as f64
}

/// Shannon entropy (Histogram 插件估计)
///
/// 使用区间 [min, max] 均分 bins 来估计离散分布。
pub fn shannon_entropy_hist(x: &Array1<f64>, bins: usize) -> f64 {
    if bins == 0 {
        return f64::NAN;
    }

    let x_slice = x.as_slice().unwrap();
    if x_slice.is_empty() {
        return f64::NAN;
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in x_slice {
        if !v.is_finite() {
            return f64::NAN;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let range = max - min;
    if range <= 0.0 {
        return 0.0;
    }

    let bin_width = range / bins as f64;
    if bin_width <= 0.0 {
        return f64::NAN;
    }

    let mut counts = vec![0usize; bins];
    for &v in x_slice {
        let mut idx = ((v - min) / bin_width).floor() as isize;
        if idx < 0 {
            idx = 0;
        } else if idx as usize >= bins {
            idx = bins as isize - 1;
        }
        counts[idx as usize] += 1;
    }

    let n = x_slice.len() as f64;
    let mut entropy = 0.0;
    for count in counts {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }
    entropy
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

/// 切片版本的 Gaussian NLL self-information（用于 rolling）
fn shannon_entropy_gaussian_slice(x: &[f64]) -> f64 {
    let (mean, std) = match mean_and_std(x) {
        Some(res) => res,
        None => return f64::NAN,
    };
    gaussian_surprisal(x[x.len() - 1], mean, std)
}

/// 切片版本的 Histogram surprisal（用于 rolling）
fn shannon_entropy_hist_slice(x: &[f64], bins: usize, min_prob: f64) -> f64 {
    if bins == 0 || !(0.0 < min_prob && min_prob < 1.0) {
        return f64::NAN;
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in x {
        if !v.is_finite() {
            return f64::NAN;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let range = max - min;
    if range <= 0.0 {
        return 0.0;
    }

    let bin_width = range / bins as f64;
    if bin_width <= 0.0 {
        return f64::NAN;
    }

    let mut counts = vec![0usize; bins];
    for &v in x {
        let mut idx = ((v - min) / bin_width).floor() as isize;
        if idx < 0 {
            idx = 0;
        } else if idx as usize >= bins {
            idx = bins as isize - 1;
        }
        counts[idx as usize] += 1;
    }

    let last = x[x.len() - 1];
    let mut last_idx = ((last - min) / bin_width).floor() as isize;
    if last_idx < 0 {
        last_idx = 0;
    } else if last_idx as usize >= bins {
        last_idx = bins as isize - 1;
    }

    let p = counts[last_idx as usize] as f64 / x.len() as f64;
    let p = if p < min_prob { min_prob } else { p };
    -p.ln()
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

/// 滑动窗口 Shannon entropy（Gaussian NLL, 包含 log σ）
///
/// 返回每个窗口最后一个样本的 self-information（nats）。
pub fn shannon_entropy_gaussian_rolling(data: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let data_slice = data.as_slice().unwrap();

    if n < period {
        return Array1::from_elem(n, f64::NAN);
    }

    let results: Vec<f64> = (period - 1..n)
        .into_par_iter()
        .map(|end_idx| {
            let start_idx = end_idx + 1 - period;
            let window = &data_slice[start_idx..=end_idx];
            shannon_entropy_gaussian_slice(window)
        })
        .collect();

    let mut output = Array1::from_elem(n, f64::NAN);
    for (i, &val) in results.iter().enumerate() {
        output[period - 1 + i] = val;
    }
    output
}

/// 滑动窗口 Shannon entropy（Histogram surprisal）
///
/// 返回每个窗口最后一个样本的 self-information（nats）。
pub fn shannon_entropy_hist_rolling(
    data: &Array1<f64>,
    period: usize,
    bins: usize,
    min_prob: f64,
) -> Array1<f64> {
    let n = data.len();
    let data_slice = data.as_slice().unwrap();

    if n < period {
        return Array1::from_elem(n, f64::NAN);
    }

    let results: Vec<f64> = (period - 1..n)
        .into_par_iter()
        .map(|end_idx| {
            let start_idx = end_idx + 1 - period;
            let window = &data_slice[start_idx..=end_idx];
            shannon_entropy_hist_slice(window, bins, min_prob)
        })
        .collect();

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

    #[test]
    fn test_shannon_entropy_gaussian_basic() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let entropy = shannon_entropy_gaussian(&x);

        let std = 1.25_f64.sqrt();
        let expected = 0.5 * (2.0 * std::f64::consts::PI).ln() + std.ln() + 0.5;
        assert_relative_eq!(entropy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_gaussian_constant() {
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let entropy = shannon_entropy_gaussian(&x);
        assert!(entropy.is_nan());
    }

    #[test]
    fn test_shannon_entropy_hist_uniform() {
        let x = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let entropy = shannon_entropy_hist(&x, 2);
        assert_relative_eq!(entropy, std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_gaussian_rolling() {
        let x = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let res = shannon_entropy_gaussian_rolling(&x, 3);

        assert!(res[0].is_nan());
        assert!(res[1].is_nan());
        assert!(res[2].is_finite());
        assert!(res[3].is_finite());
    }

    #[test]
    fn test_shannon_entropy_hist_rolling() {
        let x = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let res = shannon_entropy_hist_rolling(&x, 2, 2, 1e-12);

        assert!(res[0].is_nan());
        assert_relative_eq!(res[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(res[2], std::f64::consts::LN_2, epsilon = 1e-10);
        assert_relative_eq!(res[3], 0.0, epsilon = 1e-10);
    }
}
