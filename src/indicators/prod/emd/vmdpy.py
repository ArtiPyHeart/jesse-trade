import time

import numpy as np
from numba import jit


@jit(nopython=True)
def _compute_omega_plus(freqs, u_hat_plus, T, K, n):
    """计算中心频率omega_plus"""
    omega_new = np.zeros(K)
    for k in range(K):
        numerator = 0.0
        denominator = 0.0
        for i in range(T // 2, T):
            abs_val_sq = np.abs(u_hat_plus[n + 1, i, k]) ** 2
            numerator += freqs[i] * abs_val_sq
            denominator += abs_val_sq
        if denominator > 0:
            omega_new[k] = numerator / denominator
    return omega_new


@jit(nopython=True)
def _update_u_hat_plus(f_hat_plus, sum_uk, lambda_hat, Alpha, freqs, omega_plus, k, n):
    """更新u_hat_plus的单个模态"""
    T = len(freqs)
    u_hat_new = np.zeros(T, dtype=np.complex128)
    for i in range(T):
        u_hat_new[i] = (f_hat_plus[i] - sum_uk[i] - lambda_hat[n, i] / 2) / (
            1.0 + Alpha[k] * (freqs[i] - omega_plus[n, k]) ** 2
        )
    return u_hat_new


@jit(nopython=True)
def _compute_convergence(u_hat_plus, n, K, T):
    """计算收敛性指标"""
    uDiff = np.spacing(1)
    for i in range(K):
        diff_sum = 0.0
        for j in range(T):
            diff = u_hat_plus[n, j, i] - u_hat_plus[n - 1, j, i]
            diff_sum += np.real(diff * np.conj(diff))
        uDiff += (1 / T) * diff_sum
    return np.abs(uDiff)


@jit(nopython=True)
def _vmd_core_loop(
    f_hat_plus,
    freqs,
    Alpha,
    omega_plus,
    lambda_hat,
    u_hat_plus,
    K,
    T,
    DC,
    tau,
    tol,
    Niter,
):
    """VMD算法的核心循环，使用numba加速"""
    n = 0
    uDiff = tol + np.spacing(1)

    while (uDiff > tol) and (n < Niter - 1):
        # 更新第一个模态
        k = 0
        sum_uk = np.zeros(T, dtype=np.complex128)

        # 初始化sum_uk
        for i in range(T):
            for j in range(1, K):
                sum_uk[i] += u_hat_plus[n, i, j]

        # 更新第一个模态的频谱
        u_hat_plus[n + 1, :, k] = _update_u_hat_plus(
            f_hat_plus, sum_uk, lambda_hat, Alpha, freqs, omega_plus, k, n
        )

        # 更新第一个omega（如果不是DC模式）
        if not DC:
            omega_plus[n + 1, k] = _compute_omega_plus(freqs, u_hat_plus, T, 1, n)[0]

        # 更新其他模态
        for k in range(1, K):
            # 重新计算sum_uk
            sum_uk = np.zeros(T, dtype=np.complex128)
            for i in range(T):
                for j in range(K):
                    if j != k:
                        if j < k:
                            sum_uk[i] += u_hat_plus[n + 1, i, j]
                        else:
                            sum_uk[i] += u_hat_plus[n, i, j]

            # 模态频谱
            u_hat_plus[n + 1, :, k] = _update_u_hat_plus(
                f_hat_plus, sum_uk, lambda_hat, Alpha, freqs, omega_plus, k, n
            )

            # 中心频率
            omega_plus[n + 1, k] = _compute_omega_plus(freqs, u_hat_plus, T, k + 1, n)[
                k
            ]

        # 对偶上升
        for i in range(T):
            sum_modes = np.complex128(0)
            for k in range(K):
                sum_modes += u_hat_plus[n + 1, i, k]
            lambda_hat[n + 1, i] = lambda_hat[n, i] + tau * (sum_modes - f_hat_plus[i])

        n = n + 1

        # 检查收敛性
        if n > 0:
            uDiff = _compute_convergence(u_hat_plus, n, K, T)

    return n, omega_plus, lambda_hat, u_hat_plus


def VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) 'Variational Mode Decomposition',
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.


    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """

    N = len(f)
    if N % 2:
        f = np.append(f, f[-1])  # Append the last element to make it even

    # Period and sampling frequency of input signal
    fs = 1.0 / N

    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - (1 / T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = np.copy(f_hat)  # copy f_hat
    f_hat_plus[: T // 2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init == 2:
        omega_plus[0, :] = np.sort(
            np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K))
        )
    else:
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=np.complex128)

    # other inits
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=np.complex128)

    # *** Main loop for iterative updates***
    n, omega_plus, lambda_hat, u_hat_plus = _vmd_core_loop(
        f_hat_plus,
        freqs,
        Alpha,
        omega_plus,
        lambda_hat,
        u_hat_plus,
        K,
        T,
        DC,
        tau,
        tol,
        Niter,
    )

    # Postprocessing and cleanup
    Niter = min(Niter, n + 1)
    omega = omega_plus[:Niter, :]

    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    # Signal reconstruction
    u_hat = np.zeros([T, K], dtype=np.complex128)
    u_hat[T // 2 : T, :] = u_hat_plus[Niter - 1, T // 2 : T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2 : T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4 : 3 * T // 4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=np.complex128)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

    return u[:, :N], u_hat, omega  # Ensure the output length matches the input length


if __name__ == "__main__":
    import unittest

    class TestVMD(unittest.TestCase):
        def setUp(self):
            """设置测试数据"""
            # 创建测试信号：两个正弦波的叠加
            self.fs = 1000  # 采样频率
            self.t = np.arange(0, 1, 1 / self.fs)
            self.f1 = 10  # 第一个分量频率
            self.f2 = 50  # 第二个分量频率
            self.signal = np.sin(2 * np.pi * self.f1 * self.t) + np.sin(
                2 * np.pi * self.f2 * self.t
            )

            # VMD参数
            self.alpha = 2000
            self.tau = 0
            self.K = 2
            self.DC = False
            self.init = 1
            self.tol = 1e-6

        def test_vmd_output_shape(self):
            """测试输出形状是否正确"""
            u, u_hat, omega = VMD(
                self.signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol
            )

            # 检查输出形状
            self.assertEqual(u.shape[0], self.K)
            self.assertEqual(u.shape[1], len(self.signal))
            self.assertEqual(u_hat.shape[1], self.K)
            self.assertTrue(omega.shape[1] == self.K)

        def test_vmd_reconstruction(self):
            """测试信号重构误差"""
            u, u_hat, omega = VMD(
                self.signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol
            )

            # 重构信号
            reconstructed = np.sum(u, axis=0)

            # 计算重构误差
            error = np.mean(np.abs(self.signal - reconstructed))
            self.assertLess(error, 0.1)  # 误差应该很小

        def test_vmd_frequency_separation(self):
            """测试频率分离效果"""
            u, u_hat, omega = VMD(
                self.signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol
            )

            # 获取最终的中心频率
            final_omega = omega[-1, :]

            # 检查是否正确分离了两个频率分量
            # 注意：omega是归一化频率，需要转换
            estimated_freqs = np.sort(final_omega * self.fs)

            # 允许一定的误差范围
            self.assertAlmostEqual(estimated_freqs[0], self.f1, delta=5)
            self.assertAlmostEqual(estimated_freqs[1], self.f2, delta=5)

        def test_performance_improvement(self):
            """测试性能提升"""
            # 创建一个较大的测试信号
            t_long = np.arange(0, 5, 1 / self.fs)
            signal_long = (
                np.sin(2 * np.pi * self.f1 * t_long)
                + np.sin(2 * np.pi * self.f2 * t_long)
                + 0.1 * np.random.randn(len(t_long))
            )

            # 测试优化后的版本
            start_time = time.time()
            u, u_hat, omega = VMD(
                signal_long, self.alpha, self.tau, self.K, self.DC, self.init, self.tol
            )
            optimized_time = time.time() - start_time

            print(f"\n优化后的VMD执行时间: {optimized_time:.4f} 秒")
            print(f"信号长度: {len(signal_long)} 样本")
            print(f"每样本处理时间: {optimized_time/len(signal_long)*1000:.4f} 毫秒")

            # 验证结果的有效性
            self.assertIsNotNone(u)
            self.assertIsNotNone(u_hat)
            self.assertIsNotNone(omega)

        def test_edge_cases(self):
            """测试边界情况"""
            # 测试奇数长度信号
            odd_signal = self.signal[:-1]
            u, u_hat, omega = VMD(
                odd_signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol
            )
            self.assertEqual(u.shape[1], len(odd_signal))

            # 测试单一模态
            u_single, _, _ = VMD(
                self.signal, self.alpha, self.tau, 1, self.DC, self.init, self.tol
            )
            self.assertEqual(u_single.shape[0], 1)

            # 测试DC模式
            u_dc, _, omega_dc = VMD(
                self.signal, self.alpha, self.tau, self.K, True, self.init, self.tol
            )
            # 第一个模态的中心频率应该是0
            self.assertAlmostEqual(omega_dc[-1, 0], 0, delta=1e-6)

    # 运行测试
    unittest.main(argv=[""], exit=False)
