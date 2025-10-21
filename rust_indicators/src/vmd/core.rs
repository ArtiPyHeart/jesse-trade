//! VMD 核心算法实现
//!
//! 此模块包含 VMD 算法的所有核心函数，与 Python vmdpy.py 保持数值对齐

use ndarray::{s, Array1, Array2, Array3, ArrayView1};
use num_complex::Complex64;

/// VMD 算法错误类型
#[derive(Debug, thiserror::Error)]
pub enum VmdError {
    #[error("Invalid input length: {0}")]
    InvalidLength(usize),

    #[error("Invalid parameter: {name} = {value}")]
    InvalidParameter { name: String, value: String },

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
}

/// VMD 算法输出
#[derive(Debug, Clone)]
pub struct VmdOutput {
    /// 分解后的模态 (K, N)
    pub u: Array2<f64>,
    /// 模态的频谱 (N, K)
    pub u_hat: Array2<Complex64>,
    /// 中心频率历史 (Niter, K)
    pub omega: Array2<f64>,
}

/// 计算中心频率 omega_plus
///
/// 对应 Python 中的 `_compute_omega_plus`
///
/// # Arguments
/// * `freqs` - 频率数组
/// * `u_hat_plus` - 正频率谱 (Niter, T, K)
/// * `t` - 信号长度
/// * `k_max` - 要计算的模态数
/// * `n` - 当前迭代索引
///
/// # Returns
/// 中心频率数组 (k_max,)
pub fn compute_omega_plus(
    freqs: &ArrayView1<f64>,
    u_hat_plus: &Array3<Complex64>,
    t: usize,
    k_max: usize,
    n: usize,
) -> Array1<f64> {
    let mut omega_new = Array1::zeros(k_max);

    for k in 0..k_max {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in (t / 2)..t {
            let abs_val_sq = u_hat_plus[[n + 1, i, k]].norm_sqr();
            numerator += freqs[i] * abs_val_sq;
            denominator += abs_val_sq;
        }

        if denominator > 0.0 {
            omega_new[k] = numerator / denominator;
        }
    }

    omega_new
}

/// 更新单个模态的频谱
///
/// 对应 Python 中的 `_update_u_hat_plus`
///
/// # Arguments
/// * `f_hat_plus` - 信号频谱
/// * `sum_uk` - 其他模态之和
/// * `lambda_hat` - 拉格朗日乘子
/// * `alpha` - 正则化参数
/// * `freqs` - 频率数组
/// * `omega_plus` - 中心频率数组
/// * `k` - 当前模态索引
/// * `n` - 当前迭代索引
pub fn update_u_hat_plus(
    f_hat_plus: &ArrayView1<Complex64>,
    sum_uk: &ArrayView1<Complex64>,
    lambda_hat: &Array2<Complex64>,
    alpha: f64,
    freqs: &ArrayView1<f64>,
    omega_plus: &Array2<f64>,
    k: usize,
    n: usize,
) -> Array1<Complex64> {
    let t = freqs.len();
    let mut u_hat_new = Array1::zeros(t);

    for i in 0..t {
        let numerator = f_hat_plus[i] - sum_uk[i] - lambda_hat[[n, i]] / 2.0;
        let denominator = 1.0 + alpha * (freqs[i] - omega_plus[[n, k]]).powi(2);
        u_hat_new[i] = numerator / denominator;
    }

    u_hat_new
}

/// 计算收敛性指标
///
/// 对应 Python 中的 `_compute_convergence`
pub fn compute_convergence(
    u_hat_plus: &Array3<Complex64>,
    n: usize,
    k: usize,
    t: usize,
) -> f64 {
    let mut u_diff = f64::EPSILON;

    for i in 0..k {
        let mut diff_sum = 0.0;
        for j in 0..t {
            let diff = u_hat_plus[[n, j, i]] - u_hat_plus[[n - 1, j, i]];
            diff_sum += diff.norm_sqr();
        }
        u_diff += (1.0 / t as f64) * diff_sum;
    }

    u_diff.abs()
}

/// VMD 核心循环
///
/// 对应 Python 中的 `_vmd_core_loop`
///
/// # Returns
/// (实际迭代次数, omega_plus, lambda_hat, u_hat_plus)
#[allow(clippy::too_many_arguments)]
pub fn vmd_core_loop(
    f_hat_plus: &ArrayView1<Complex64>,
    freqs: &ArrayView1<f64>,
    alpha: &Array1<f64>,
    mut omega_plus: Array2<f64>,
    mut lambda_hat: Array2<Complex64>,
    mut u_hat_plus: Array3<Complex64>,
    k: usize,
    t: usize,
    dc: bool,
    tau: f64,
    tol: f64,
    niter: usize,
) -> (usize, Array2<f64>, Array2<Complex64>, Array3<Complex64>) {
    let mut n = 0;
    let mut u_diff = tol + f64::EPSILON;

    while (u_diff > tol) && (n < niter - 1) {
        // 预计算所有模态的总和（使用上一轮迭代的值）
        let mut total_sum: Array1<Complex64> = Array1::zeros(t);
        for i in 0..t {
            for j in 0..k {
                total_sum[i] += u_hat_plus[[n, i, j]];
            }
        }

        // 更新第一个模态
        let mut sum_uk = Array1::zeros(t);
        for i in 0..t {
            // sum_uk = total_sum - u_hat_plus[n][i][0]
            sum_uk[i] = total_sum[i] - u_hat_plus[[n, i, 0]];
        }

        let u_hat_new = update_u_hat_plus(
            f_hat_plus,
            &sum_uk.view(),
            &lambda_hat,
            alpha[0],
            freqs,
            &omega_plus,
            0,
            n,
        );
        u_hat_plus.slice_mut(s![n + 1, .., 0]).assign(&u_hat_new);

        // 更新第一个 omega（如果不是 DC 模式）
        if !dc {
            let omega_new = compute_omega_plus(freqs, &u_hat_plus, t, 1, n);
            omega_plus[[n + 1, 0]] = omega_new[0];
        }

        // 更新 total_sum：替换第一个模态的值从 n 到 n+1
        for i in 0..t {
            total_sum[i] = total_sum[i] - u_hat_plus[[n, i, 0]] + u_hat_plus[[n + 1, i, 0]];
        }

        // 更新其他模态
        for k_idx in 1..k {
            // 使用增量更新：sum_uk = total_sum - u_hat_plus[n][i][k_idx]
            for i in 0..t {
                sum_uk[i] = total_sum[i] - u_hat_plus[[n, i, k_idx]];
            }

            let u_hat_new = update_u_hat_plus(
                f_hat_plus,
                &sum_uk.view(),
                &lambda_hat,
                alpha[k_idx],
                freqs,
                &omega_plus,
                k_idx,
                n,
            );
            u_hat_plus.slice_mut(s![n + 1, .., k_idx]).assign(&u_hat_new);

            // 更新中心频率
            let omega_new = compute_omega_plus(freqs, &u_hat_plus, t, k_idx + 1, n);
            omega_plus[[n + 1, k_idx]] = omega_new[k_idx];

            // 更新 total_sum：替换当前模态的值从 n 到 n+1
            for i in 0..t {
                total_sum[i] = total_sum[i] - u_hat_plus[[n, i, k_idx]] + u_hat_plus[[n + 1, i, k_idx]];
            }
        }

        // 对偶上升（直接使用 total_sum，它已经包含所有 n+1 时刻的模态和）
        for i in 0..t {
            lambda_hat[[n + 1, i]] = lambda_hat[[n, i]] + tau * (total_sum[i] - f_hat_plus[i]);
        }

        n += 1;

        // 检查收敛性
        if n > 0 {
            u_diff = compute_convergence(&u_hat_plus, n, k, t);
        }
    }

    (n, omega_plus, lambda_hat, u_hat_plus)
}

/// VMD 主函数
///
/// 完全对应 Python vmdpy.VMD 的实现
///
/// # Arguments
/// * `f` - 输入信号
/// * `alpha` - 数据保真度约束参数
/// * `tau` - 对偶上升时间步长
/// * `k` - 模态数量
/// * `dc` - 第一个模态是否固定在 DC (0频率)
/// * `init` - omega 初始化方式 (0=全0, 1=均匀分布, 2=随机)
/// * `tol` - 收敛容差
///
/// # Returns
/// `VmdOutput` 包含分解结果
pub fn vmd(
    f: &Array1<f64>,
    alpha: f64,
    tau: f64,
    k: usize,
    dc: bool,
    init: usize,
    tol: f64,
) -> Result<VmdOutput, VmdError> {
    if k == 0 {
        return Err(VmdError::InvalidParameter {
            name: "k".to_string(),
            value: k.to_string(),
        });
    }

    let mut n = f.len();
    let mut f_vec = f.to_vec();

    // 如果长度为奇数，补齐到偶数
    if n % 2 == 1 {
        f_vec.push(f_vec[n - 1]);
        n += 1;
    }

    // 镜像扩展信号
    let l_temp = n / 2;
    let mut f_mirr = Vec::with_capacity(n * 2);

    // 左镜像
    for i in (0..l_temp).rev() {
        f_mirr.push(f_vec[i]);
    }
    // 原始信号
    f_mirr.extend_from_slice(&f_vec);
    // 右镜像
    for i in (n - l_temp..n).rev() {
        f_mirr.push(f_vec[i]);
    }

    let t = f_mirr.len();

    // 频率域离散化 (对应 Python: t = arange(1, T+1) / T - 0.5 - 1/T)
    let freqs: Array1<f64> = Array1::from_shape_fn(t, |i| {
        ((i + 1) as f64 / t as f64) - 0.5 - (1.0 / t as f64)
    });

    // FFT
    let f_complex_input: Array1<Complex64> = Array1::from_shape_fn(t, |i| {
        Complex64::new(f_mirr[i], 0.0)
    });

    let f_hat_raw = super::utils::fft(&f_complex_input);

    // fftshift
    let f_hat = super::utils::fftshift(&f_hat_raw.view());

    // 构建正频率部分
    let mut f_hat_plus = f_hat.clone();
    for i in 0..(t / 2) {
        f_hat_plus[i] = Complex64::new(0.0, 0.0);
    }

    // 初始化 omega_k
    let niter = 500;
    let alpha_vec = Array1::from_elem(k, alpha);
    let mut omega_plus = Array2::zeros((niter, k));

    match init {
        1 => {
            // 均匀分布初始化
            for i in 0..k {
                omega_plus[[0, i]] = (0.5 / k as f64) * i as f64;
            }
        }
        2 => {
            // 随机初始化（在 fs 到 0.5 之间）
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let fs = 1.0 / n as f64;

            let mut random_omegas: Vec<f64> = (0..k)
                .map(|_| {
                    let log_fs = fs.ln();
                    let log_half = 0.5_f64.ln();
                    let rand_val: f64 = rng.gen(); // [0, 1)
                    (log_fs + (log_half - log_fs) * rand_val).exp()
                })
                .collect();

            // 排序
            random_omegas.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for i in 0..k {
                omega_plus[[0, i]] = random_omegas[i];
            }
        }
        _ => {
            // omega_plus 已经初始化为全 0
        }
    }

    if dc {
        omega_plus[[0, 0]] = 0.0;
    }

    let lambda_hat = Array2::zeros((niter, t));
    let u_hat_plus = Array3::zeros((niter, t, k));

    // 主循环
    let (actual_niter, omega_final, _lambda_final, u_hat_plus_final) = vmd_core_loop(
        &f_hat_plus.view(),
        &freqs.view(),
        &alpha_vec,
        omega_plus,
        lambda_hat,
        u_hat_plus,
        k,
        t,
        dc,
        tau,
        tol,
        niter,
    );

    // ========== 后处理和信号重建 ==========

    // 截取实际使用的迭代次数
    let actual_niter = (actual_niter + 1).min(niter);
    let omega = omega_final.slice(s![..actual_niter, ..]).to_owned();

    // 信号重建
    let mut u_hat_reconstructed = Array2::zeros((t, k));

    // 填充正频率部分
    for i in (t / 2)..t {
        for k_idx in 0..k {
            u_hat_reconstructed[[i, k_idx]] = u_hat_plus_final[[actual_niter - 1, i, k_idx]];
        }
    }

    // 填充负频率部分（共轭对称）
    // 对应 Python: idxs = flip(arange(1, T//2+1))
    //              u_hat[idxs] = conj(u_hat_plus[T//2 : T])
    // 即: u_hat[T//2], u_hat[T//2-1], ..., u_hat[1] = conj(u_hat_plus[T//2:T])
    for i in 0..(t / 2) {
        let src_idx = t / 2 + i;  // T//2, T//2+1, ..., T-1
        let dst_idx = t / 2 - i;  // T//2, T//2-1, ..., 1
        for k_idx in 0..k {
            u_hat_reconstructed[[dst_idx, k_idx]] =
                u_hat_plus_final[[actual_niter - 1, src_idx, k_idx]].conj();
        }
    }

    // DC 分量
    for k_idx in 0..k {
        u_hat_reconstructed[[0, k_idx]] = u_hat_reconstructed[[t - 1, k_idx]].conj();
    }

    // IFFT 重建时域信号
    let mut u_time = Array2::zeros((k, t));
    for k_idx in 0..k {
        // 提取单个模态的频谱
        let u_hat_k = u_hat_reconstructed.column(k_idx).to_owned();

        // ifftshift
        let u_hat_k_shifted = super::utils::ifftshift(&u_hat_k.view());

        // IFFT
        let u_k_complex = super::utils::ifft(&u_hat_k_shifted);

        // 取实部
        for i in 0..t {
            u_time[[k_idx, i]] = u_k_complex[i].re;
        }
    }

    // 去镜像：提取中间部分
    let start = t / 4;
    let end = 3 * t / 4;
    let mut u_final = Array2::zeros((k, end - start));
    for k_idx in 0..k {
        for i in 0..(end - start) {
            u_final[[k_idx, i]] = u_time[[k_idx, start + i]];
        }
    }

    // 截取到原始信号长度
    let original_n = f.len();
    let mut u_output = Array2::zeros((k, original_n));
    for k_idx in 0..k {
        for i in 0..original_n {
            u_output[[k_idx, i]] = u_final[[k_idx, i]];
        }
    }

    // 重新计算频谱（用于输出）
    // 注意: Python 返回基于去镜像后长度的 u_hat，而不是原始长度
    let u_hat_len = end - start;  // 去镜像后的长度
    let mut u_hat_output = Array2::zeros((u_hat_len, k));
    for k_idx in 0..k {
        let u_k = u_final.row(k_idx).to_owned();
        let u_k_complex = Array1::from_shape_fn(u_hat_len, |i| Complex64::new(u_k[i], 0.0));

        let u_hat_k_raw = super::utils::fft(&u_k_complex);
        let u_hat_k_shifted = super::utils::fftshift(&u_hat_k_raw.view());

        for i in 0..u_hat_len {
            u_hat_output[[i, k_idx]] = u_hat_k_shifted[i];
        }
    }

    Ok(VmdOutput {
        u: u_output,
        u_hat: u_hat_output,
        omega,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_compute_omega_plus_basic() {
        // 简单测试：验证函数可以运行
        let freqs = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3]);
        let u_hat_plus = Array3::zeros((10, 4, 2));

        let result = compute_omega_plus(&freqs.view(), &u_hat_plus, 4, 2, 0);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_update_u_hat_plus_basic() {
        let t = 4;
        let f_hat_plus = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);
        let sum_uk = Array1::zeros(t);
        let lambda_hat = Array2::zeros((10, t));
        let alpha = 2000.0;
        let freqs = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3]);
        let omega_plus = Array2::zeros((10, 2));

        let result = update_u_hat_plus(
            &f_hat_plus.view(),
            &sum_uk.view(),
            &lambda_hat,
            alpha,
            &freqs.view(),
            &omega_plus,
            0,
            0,
        );

        assert_eq!(result.len(), t);
    }
}
