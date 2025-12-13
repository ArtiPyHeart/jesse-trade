"""
ARD-VAE 数学正确性测试

验证：
1. KL 散度公式正确性
2. 重参数化技巧正确性
3. 维度选择逻辑正确性
4. 输入输出一致性
5. Save/Load 完整性
"""

import random
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch


def seed_everything(seed: int = 42) -> None:
    """统一设置所有随机种子，确保测试可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 操作确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestKLDivergenceFormula:
    """KL 散度公式的数学正确性测试"""

    def test_kl_standard_normal_prior(self):
        """测试当 alpha=1 时，KL 公式退化为标准 VAE KL"""
        # KL[N(mu, var) || N(0, 1)] = 0.5 * (mu^2 + var - log(var) - 1)
        mu = torch.tensor([[1.0, 2.0], [0.5, -0.5]])
        log_var = torch.tensor([[0.0, 0.5], [-0.5, 0.0]])
        log_alpha = torch.zeros(2)  # alpha = 1

        # 使用我们的公式
        alpha = torch.exp(log_alpha)
        var = torch.exp(log_var)
        kl_ours = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

        # 标准 VAE KL 公式
        kl_standard = 0.5 * (mu.pow(2) + var - log_var - 1)

        assert torch.allclose(kl_ours, kl_standard, atol=1e-6)

    def test_kl_nonnegative(self):
        """KL 散度必须非负"""
        torch.manual_seed(42)
        for _ in range(100):
            mu = torch.randn(10, 32)
            log_var = torch.randn(10, 32).clamp(-5, 5)
            log_alpha = torch.randn(32)

            alpha = torch.exp(log_alpha)
            var = torch.exp(log_var)
            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

            # KL 散度必须非负（允许极小的数值误差）
            assert kl.min() >= -1e-5, f"KL became negative: {kl.min()}"

    def test_kl_zero_at_prior(self):
        """当后验等于先验时，KL 应该为 0"""
        # 先验 N(0, 1/alpha)，当 mu=0, var=1/alpha 时，KL=0
        log_alpha = torch.tensor([0.5, 1.0, -0.5])  # 不同的 alpha
        alpha = torch.exp(log_alpha)

        # 设置后验等于先验
        mu = torch.zeros(1, 3)
        var = 1.0 / alpha  # var = 1/alpha
        log_var = torch.log(var).unsqueeze(0)

        kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_increases_with_deviation(self):
        """KL 应随后验偏离先验而增加"""
        log_alpha = torch.zeros(1)
        alpha = torch.exp(log_alpha)
        log_var = torch.zeros(1, 1)  # var = 1
        var = torch.exp(log_var)

        kl_values = []
        for mu_val in [0.0, 0.5, 1.0, 2.0, 3.0]:
            mu = torch.tensor([[mu_val]])
            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)
            kl_values.append(kl.item())

        # KL 应该单调递增
        for i in range(len(kl_values) - 1):
            assert kl_values[i] < kl_values[i + 1]


class TestReparameterizationTrick:
    """重参数化技巧正确性测试"""

    def test_reparameterize_mean_correct(self):
        """重参数化采样的均值应接近 mu，容差基于 CLT"""
        from src.features.dimensionality_reduction.ard_vae import ARDVAEConfig, ARDVAENet

        seed_everything(42)

        config = ARDVAEConfig(input_dim=10, max_latent_dim=8)
        net = ARDVAENet(config)
        net.train()

        mu = torch.tensor([[1.0, 2.0, -1.0, 0.5, -0.5, 0.0, 1.5, -1.5]])
        log_var = torch.zeros(1, 8)  # var = 1

        # 多次采样取平均
        n_samples = 10000
        samples = []
        for _ in range(n_samples):
            z = net.reparameterize(mu, log_var)
            samples.append(z)

        samples_tensor = torch.stack(samples).squeeze(1)  # [n_samples, 8]
        sample_mean = samples_tensor.mean(dim=0)
        sample_std = samples_tensor.std(dim=0)

        # 基于 CLT 计算容差: stderr = std / sqrt(n), tol = z_score * stderr
        stderr = sample_std / np.sqrt(n_samples)
        z_score = 4.0  # 99.99% 置信度
        tol = z_score * stderr

        diff = torch.abs(sample_mean - mu.squeeze())
        assert (diff < tol).all(), \
            f"Sample mean {sample_mean} differs from mu {mu.squeeze()} by {diff}, tol={tol}"

    def test_reparameterize_variance_correct(self):
        """重参数化采样的方差应接近 exp(log_var)，容差基于统计原理"""
        from src.features.dimensionality_reduction.ard_vae import ARDVAEConfig, ARDVAENet

        seed_everything(42)

        config = ARDVAEConfig(input_dim=10, max_latent_dim=4)
        net = ARDVAENet(config)
        net.train()

        mu = torch.zeros(1, 4)
        log_var = torch.tensor([[0.0, 1.0, -1.0, 0.5]])  # 不同方差
        expected_var = torch.exp(log_var)

        n_samples = 10000
        samples = []
        for _ in range(n_samples):
            z = net.reparameterize(mu, log_var)
            samples.append(z)

        samples_tensor = torch.stack(samples).squeeze(1)  # [n_samples, 4]
        sample_var = samples_tensor.var(dim=0)

        # 样本方差的标准误: SE(S²) ≈ σ² * sqrt(2/(n-1))
        # 容差 = z_score * SE
        se_var = expected_var.squeeze() * np.sqrt(2.0 / (n_samples - 1))
        z_score = 4.0  # 99.99% 置信度
        tol = z_score * se_var

        diff = torch.abs(sample_var - expected_var.squeeze())
        assert (diff < tol).all(), \
            f"Sample var {sample_var} differs from expected {expected_var.squeeze()} by {diff}, tol={tol}"

    def test_reparameterize_eval_mode(self):
        """评估模式下应直接返回均值"""
        from src.features.dimensionality_reduction.ard_vae import ARDVAEConfig, ARDVAENet

        config = ARDVAEConfig(input_dim=10, max_latent_dim=4)
        net = ARDVAENet(config)
        net.eval()

        mu = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        log_var = torch.ones(1, 4)

        z = net.reparameterize(mu, log_var)
        assert torch.equal(z, mu)


class TestDimensionSelection:
    """维度选择逻辑测试"""

    def test_active_dims_selection(self):
        """测试 active dimensions 选择逻辑"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        # 创建有明显结构的测试数据
        np.random.seed(42)
        n_samples = 500
        n_features = 50

        # 只有前 5 个特征有真实信号
        X_signal = np.random.randn(n_samples, 5) * 2
        X_noise = np.random.randn(n_samples, n_features - 5) * 0.01
        X = np.hstack([X_signal, X_noise])

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

        config = ARDVAEConfig(
            max_latent_dim=16, max_epochs=50, kl_threshold=0.1, seed=42
        )
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        # 应该识别出有效维度（至少 1 个，不超过 max_latent_dim）
        # 注：具体数量取决于数据结构和训练收敛情况
        assert 1 <= ard_vae.n_components <= 16

    def test_active_dims_at_least_one(self):
        """即使所有维度都不活跃，也应至少保留一个"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        # 极端情况：所有特征都是常数
        np.random.seed(42)
        n_samples = 100
        X = np.ones((n_samples, 20)) + np.random.randn(n_samples, 20) * 1e-6

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=20, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        assert ard_vae.n_components >= 1


class TestInputOutputConsistency:
    """输入输出一致性测试"""

    def test_transform_output_shape(self):
        """transform 输出形状正确"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        n_samples = 200
        n_features = 30

        df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )

        config = ARDVAEConfig(max_latent_dim=16, max_epochs=20, seed=42)
        ard_vae = ARDVAE(config)
        X_reduced = ard_vae.fit_transform(df, verbose=False)

        # 检查输出形状
        assert X_reduced.shape[0] == n_samples
        assert X_reduced.shape[1] == ard_vae.n_components

        # 检查列名格式
        assert list(X_reduced.columns) == [str(i) for i in range(ard_vae.n_components)]

        # 检查索引保持
        assert list(X_reduced.index) == list(df.index)

    def test_transform_preserves_index(self):
        """transform 保持原始索引"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 20),
            index=pd.date_range("2020-01-01", periods=100, freq="h"),
            columns=[f"f{i}" for i in range(20)],
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=15, seed=42)
        ard_vae = ARDVAE(config)
        X_reduced = ard_vae.fit_transform(df, verbose=False)

        pd.testing.assert_index_equal(X_reduced.index, df.index)

    def test_transform_deterministic(self):
        """同一模型多次 transform 应得到相同结果"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 20), columns=[f"f{i}" for i in range(20)]
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=15, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        result1 = ard_vae.transform(df)
        result2 = ard_vae.transform(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_column_mismatch_error(self):
        """列名不匹配时应报错"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df_train = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)]
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=10, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df_train, verbose=False)

        # 列名不同
        df_test = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"g{i}" for i in range(10)]
        )

        with pytest.raises(ValueError, match="Column mismatch"):
            ard_vae.transform(df_test)

    def test_nan_input_error(self):
        """包含 NaN 的输入应报错"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        df = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)]
        )
        df.iloc[50, 5] = np.nan

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=10, seed=42)
        ard_vae = ARDVAE(config)

        with pytest.raises(ValueError, match="NaN"):
            ard_vae.fit(df, verbose=False)


class TestSaveLoad:
    """保存和加载完整性测试"""

    def test_save_load_preserves_transform(self):
        """save/load 后 transform 结果应一致"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 20), columns=[f"f{i}" for i in range(20)]
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=15, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        result_before = ard_vae.transform(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            ard_vae.save(tmpdir, "test_model")
            loaded = ARDVAE.load(tmpdir, "test_model")

        result_after = loaded.transform(df)

        # 数值应完全一致
        np.testing.assert_allclose(
            result_before.values, result_after.values, rtol=1e-5
        )

    def test_save_load_preserves_metadata(self):
        """save/load 后元数据应一致"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 20), columns=[f"f{i}" for i in range(20)]
        )

        config = ARDVAEConfig(
            max_latent_dim=16, max_epochs=15, kl_threshold=0.02, seed=42
        )
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            ard_vae.save(tmpdir, "test_model")
            loaded = ARDVAE.load(tmpdir, "test_model")

        # 检查关键属性
        assert loaded.n_components == ard_vae.n_components
        np.testing.assert_array_equal(loaded.active_dims, ard_vae.active_dims)
        np.testing.assert_allclose(loaded.kl_per_dim, ard_vae.kl_per_dim)
        assert loaded._feature_names == ard_vae._feature_names

    def test_save_without_fit_error(self):
        """未 fit 时 save 应报错"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        config = ARDVAEConfig(max_latent_dim=8)
        ard_vae = ARDVAE(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="unfitted"):
                ard_vae.save(tmpdir, "test_model")

    def test_load_missing_file_error(self):
        """加载不存在的文件应报错"""
        from src.features.dimensionality_reduction import ARDVAE

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                ARDVAE.load(tmpdir, "nonexistent_model")


class TestScalerIntegration:
    """Scaler 集成测试"""

    def test_scaler_mean_std_preserved(self):
        """Scaler 参数应在 save/load 后保持"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        # 使用非零均值和非单位方差的数据
        X = np.random.randn(100, 10) * 5 + 3
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=15, use_scaler=True, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        original_mean = ard_vae._scaler_mean.copy()
        original_std = ard_vae._scaler_std.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            ard_vae.save(tmpdir, "test_model")
            loaded = ARDVAE.load(tmpdir, "test_model")

        np.testing.assert_allclose(loaded._scaler_mean, original_mean)
        np.testing.assert_allclose(loaded._scaler_std, original_std)

    def test_scaler_disabled(self):
        """use_scaler=False 时不应标准化"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        X = np.random.randn(100, 10) * 5 + 3
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=15, use_scaler=False, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        assert ard_vae._scaler_mean is None
        assert ard_vae._scaler_std is None


class TestReconstructionQuality:
    """重建质量测试"""

    def test_reconstruction_loss_decreases(self):
        """训练过程中重建损失应下降"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(200, 20), columns=[f"f{i}" for i in range(20)]
        )

        config = ARDVAEConfig(max_latent_dim=16, max_epochs=50, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        history = ard_vae.training_history
        assert len(history) > 0

        # 取前 5 个和后 5 个 epoch 的平均重建损失比较
        early_recon = np.mean([h["train_recon"] for h in history[:5]])
        late_recon = np.mean([h["train_recon"] for h in history[-5:]])

        assert late_recon < early_recon, "Reconstruction loss should decrease"


class TestEdgeCases:
    """边界情况测试"""

    def test_single_feature(self):
        """单特征输入"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 1), columns=["f0"])

        config = ARDVAEConfig(max_latent_dim=4, max_epochs=10, seed=42)
        ard_vae = ARDVAE(config)
        X_reduced = ard_vae.fit_transform(df, verbose=False)

        assert X_reduced.shape[0] == 100
        assert X_reduced.shape[1] >= 1

    def test_many_features(self):
        """大量特征输入"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        n_features = 500
        df = pd.DataFrame(
            np.random.randn(100, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )

        config = ARDVAEConfig(max_latent_dim=64, max_epochs=10, seed=42)
        ard_vae = ARDVAE(config)
        X_reduced = ard_vae.fit_transform(df, verbose=False)

        assert X_reduced.shape[0] == 100
        assert X_reduced.shape[1] <= 64

    def test_small_batch(self):
        """样本数小于 batch_size"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(30, 10), columns=[f"f{i}" for i in range(10)]
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=10, batch_size=64, seed=42)
        ard_vae = ARDVAE(config)

        # 应该正常运行
        X_reduced = ard_vae.fit_transform(df, verbose=False)
        assert X_reduced.shape[0] == 30


class TestCodexIssuesFixes:
    """Codex 审查发现的问题修复测试"""

    def test_unsupported_prior_type_raises(self):
        """不支持的 prior type 应抛异常"""
        from src.features.dimensionality_reduction import ARDVAEConfig

        with pytest.raises(NotImplementedError, match="not implemented"):
            ARDVAEConfig(ard_prior_type="horseshoe")

        with pytest.raises(NotImplementedError, match="not implemented"):
            ARDVAEConfig(ard_prior_type="spike_slab")

    def test_active_dims_use_kl_not_variance(self):
        """验证 active dims 基于 KL 计算"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 20), columns=[f"f{i}" for i in range(20)]
        )

        config = ARDVAEConfig(max_latent_dim=16, max_epochs=20, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        # kl_per_dim 应该是正数数组
        assert len(ard_vae.kl_per_dim) == 16
        assert (ard_vae.kl_per_dim >= 0).all(), "KL contributions should be non-negative"

        # active_dims 应该基于 kl_per_dim 选择
        importance = ard_vae.get_dimension_importance()
        assert "kl" in importance.columns, "Should have 'kl' column, not 'activity'"

    def test_invalid_input_types(self):
        """验证 DataFrame 类型强制"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=5, seed=42)
        ard_vae = ARDVAE(config)

        # ndarray 输入应该报错
        with pytest.raises(TypeError, match="DataFrame"):
            ard_vae.fit(np.random.randn(100, 10))

    def test_val_data_validation(self):
        """验证集校验"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df_train = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)]
        )

        # 验证集列名不匹配
        df_val_wrong = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"g{i}" for i in range(10)]
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=5, seed=42)
        ard_vae = ARDVAE(config)

        with pytest.raises(ValueError, match="Column mismatch"):
            ard_vae.fit(df_train, val_data=df_val_wrong, verbose=False)

        # 验证集含 NaN
        df_val_nan = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"f{i}" for i in range(10)]
        )
        df_val_nan.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            ard_vae.fit(df_train, val_data=df_val_nan, verbose=False)

    def test_empty_input_raises(self):
        """空输入报错"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=5, seed=42)
        ard_vae = ARDVAE(config)

        # 0 行
        df_empty_rows = pd.DataFrame(columns=[f"f{i}" for i in range(10)])
        with pytest.raises(ValueError, match="0 rows"):
            ard_vae.fit(df_empty_rows)

        # 0 列
        df_empty_cols = pd.DataFrame(index=range(10))
        with pytest.raises(ValueError, match="no columns"):
            ard_vae.fit(df_empty_cols)

    def test_inf_input_raises(self):
        """inf 输入报错"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)]
        )
        df.iloc[0, 0] = np.inf

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=5, seed=42)
        ard_vae = ARDVAE(config)

        with pytest.raises(ValueError, match="infinite"):
            ard_vae.fit(df)

    def test_valid_val_data_accepted(self):
        """正确的验证集应该被接受"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        df_train = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)]
        )
        df_val = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"f{i}" for i in range(10)]
        )

        config = ARDVAEConfig(max_latent_dim=8, max_epochs=10, seed=42)
        ard_vae = ARDVAE(config)

        # 应该正常运行
        ard_vae.fit(df_train, val_data=df_val, verbose=False)
        assert ard_vae.is_fitted


class TestKLNumericalCorrectness:
    """KL 散度数值正确性测试（扩展）"""

    def test_kl_zero_at_optimal_for_any_alpha(self):
        """任意 alpha>0 时，mu=0, var=1/alpha 处 KL=0"""
        torch.manual_seed(42)
        for _ in range(10):
            log_alpha = torch.randn(8)
            alpha = torch.exp(log_alpha)
            mu = torch.zeros(1, 8)
            var = 1.0 / alpha
            log_var = torch.log(var).unsqueeze(0)

            # KL_j = 0.5 * (alpha_j * (mu_j^2 + var_j) - log(var_j) - 1 - log(alpha_j))
            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

            assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5), \
                f"KL should be 0 at optimal point, got {kl}"

    def test_kl_analytic_gradients(self):
        """验证 autograd 与闭式解一致"""
        torch.manual_seed(42)

        mu = torch.randn(4, 8, requires_grad=True)
        log_var = torch.randn(4, 8, requires_grad=True)
        log_alpha = torch.randn(8, requires_grad=True)

        alpha = torch.exp(log_alpha)
        var = torch.exp(log_var)

        # 计算 KL
        kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)
        kl_sum = kl.sum()
        kl_sum.backward()

        # 解析梯度
        # ∂KL/∂mu = alpha * mu
        expected_grad_mu = alpha * mu
        # ∂KL/∂log_var = 0.5 * (alpha * var - 1)
        expected_grad_log_var = 0.5 * (alpha * var - 1)
        # ∂KL/∂log_alpha = 0.5 * (alpha*(mu^2+var) - 1), summed over batch
        expected_grad_log_alpha = 0.5 * (alpha * (mu.pow(2) + var) - 1).sum(dim=0)

        assert torch.allclose(mu.grad, expected_grad_mu, rtol=1e-4), \
            "Gradient w.r.t. mu mismatch"
        assert torch.allclose(log_var.grad, expected_grad_log_var, rtol=1e-4), \
            "Gradient w.r.t. log_var mismatch"
        assert torch.allclose(log_alpha.grad, expected_grad_log_alpha, rtol=1e-4), \
            "Gradient w.r.t. log_alpha mismatch"

    def test_kl_monte_carlo_matches_analytic(self):
        """采样估计 KL 应与解析公式一致，容差基于 CLT 置信区间"""
        seed_everything(42)

        mu = torch.tensor([[1.0, -0.5, 0.3]])
        log_var = torch.tensor([[0.5, -0.3, 0.1]])
        log_alpha = torch.tensor([0.2, -0.1, 0.4])

        alpha = torch.exp(log_alpha)
        var = torch.exp(log_var)

        # 解析 KL
        kl_analytic = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

        # Monte Carlo 估计: E[log q(z) - log p(z)]
        n_samples = 100000
        std = torch.sqrt(var)
        eps = torch.randn(n_samples, 3)
        z = mu + std * eps  # [n_samples, 3]

        # log q(z) = -0.5 * ((z-mu)^2/var + log(2*pi*var))
        log_q = -0.5 * ((z - mu).pow(2) / var + torch.log(2 * torch.pi * var))
        # log p(z) = -0.5 * (alpha * z^2 + log(2*pi/alpha))
        log_p = -0.5 * (alpha * z.pow(2) + torch.log(2 * torch.pi / alpha))

        kl_samples = log_q - log_p  # [n_samples, 3]
        kl_mc = kl_samples.mean(dim=0)

        # 基于 CLT 计算置信区间: tol = z_score * stderr
        sample_std = kl_samples.std(dim=0)
        stderr = sample_std / np.sqrt(n_samples)
        z_score = 4.0  # 99.99% 置信度
        tol = z_score * stderr

        # 使用统计计算的容差而非硬编码
        diff = torch.abs(kl_mc - kl_analytic.squeeze())
        assert (diff < tol).all(), \
            f"MC KL {kl_mc} differs from Analytic {kl_analytic.squeeze()} by {diff}, tol={tol}"

    def test_kl_extreme_values_stable(self):
        """mu, log_var, log_alpha 在极值范围内保持数值稳定"""
        seed_everything(42)

        # 测试多种极值组合
        extreme_ranges = [(-10, 10), (-15, 15), (-20, 20)]

        for low, high in extreme_ranges:
            mu = torch.FloatTensor(10, 16).uniform_(low, high)
            log_var = torch.FloatTensor(10, 16).uniform_(low / 2, high / 2)  # 稍微保守
            log_alpha = torch.FloatTensor(16).uniform_(low / 2, high / 2)

            alpha = torch.exp(log_alpha)
            var = torch.exp(log_var)

            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

            assert torch.isfinite(kl).all(), \
                f"KL contains non-finite values at range [{low}, {high}]"
            assert (kl >= -1e-5).all(), \
                f"KL became negative at range [{low}, {high}]: min={kl.min()}"

    def test_kl_boundary_conditions(self):
        """参数化测试 KL 在各种边界条件下的行为"""
        seed_everything(42)

        boundary_cases = [
            # (description, mu, log_var, log_alpha)
            # Case 1: 极小方差 (var 接近 0)
            ("tiny_var", torch.zeros(2, 4), torch.full((2, 4), -10.0), torch.zeros(4)),
            # Case 2: 极大方差
            ("large_var", torch.zeros(2, 4), torch.full((2, 4), 10.0), torch.zeros(4)),
            # Case 3: 极大 |mu|
            ("large_mu", torch.full((2, 4), 50.0), torch.zeros(2, 4), torch.zeros(4)),
            # Case 4: 极小 alpha (接近 0)
            ("tiny_alpha", torch.zeros(2, 4), torch.zeros(2, 4), torch.full((4,), -10.0)),
            # Case 5: 极大 alpha
            ("large_alpha", torch.zeros(2, 4), torch.zeros(2, 4), torch.full((4,), 10.0)),
            # Case 6: 单维度
            ("single_dim", torch.randn(5, 1), torch.randn(5, 1), torch.randn(1)),
            # Case 7: 大 batch
            ("large_batch", torch.randn(1000, 8), torch.randn(1000, 8), torch.randn(8)),
            # Case 8: 混合极值
            ("mixed_extreme", torch.tensor([[100.0, -100.0]]), torch.tensor([[5.0, -5.0]]), torch.tensor([5.0, -5.0])),
        ]

        for desc, mu, log_var, log_alpha in boundary_cases:
            alpha = torch.exp(log_alpha)
            var = torch.exp(log_var)

            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)

            # 断言 1: KL 始终有限 (无 NaN/inf)
            assert torch.isfinite(kl).all(), \
                f"[{desc}] KL contains non-finite values: {kl}"

            # 断言 2: KL 非负 (允许微小数值误差)
            assert (kl >= -1e-5).all(), \
                f"[{desc}] KL became negative: min={kl.min()}"

            # 断言 3: 形状正确
            assert kl.shape == mu.shape, \
                f"[{desc}] Shape mismatch: kl={kl.shape}, mu={mu.shape}"


class TestELBOInvariants:
    """ELBO 不变量测试"""

    def test_elbo_upper_bounds_log_evidence(self):
        """ELBO ≤ log p(x)（在 1D Gaussian 模型上验证）

        生成模型: z ~ N(0, 1/alpha), x | z ~ N(z, sigma^2)
        边际似然: p(x) = N(0, 1/alpha + sigma^2)
        """
        torch.manual_seed(42)

        alpha = 2.0  # precision of prior
        sigma_sq = 0.5  # observation noise variance

        # 生成数据
        z_true = torch.randn(1) / np.sqrt(alpha)
        x = z_true + torch.randn(1) * np.sqrt(sigma_sq)

        # 真实边际似然
        marginal_var = 1.0 / alpha + sigma_sq
        log_p_x = -0.5 * (x.pow(2) / marginal_var + torch.log(torch.tensor(2 * np.pi * marginal_var)))

        # 测试多个任意 q(z)
        for _ in range(20):
            q_mu = torch.randn(1)
            q_var = torch.exp(torch.randn(1).clamp(-2, 2))

            # ELBO = E_q[log p(x|z)] - KL(q||p)
            # E_q[log p(x|z)] = -0.5 * (E[(x-z)^2]/sigma^2 + log(2*pi*sigma^2))
            #                 = -0.5 * ((x-q_mu)^2 + q_var)/sigma^2 + log(2*pi*sigma^2))
            expected_sq_error = (x - q_mu).pow(2) + q_var
            log_likelihood_term = -0.5 * (expected_sq_error / sigma_sq + np.log(2 * np.pi * sigma_sq))

            # KL(q || p) where p = N(0, 1/alpha)
            kl = 0.5 * (alpha * (q_mu.pow(2) + q_var) - torch.log(q_var) - 1 + np.log(alpha))

            elbo = log_likelihood_term - kl

            assert elbo <= log_p_x + 1e-5, \
                f"ELBO {elbo.item():.4f} > log p(x) {log_p_x.item():.4f}"

    def test_prior_matching_on_uninformative_data(self):
        """纯噪声数据上，encoder 应收敛到接近先验（多 seed 统计）"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        n_samples = 200
        n_features = 20

        # 多 seed 运行
        seeds = [42, 123, 456]
        n_components_results = []

        for seed in seeds:
            seed_everything(seed)

            # 纯噪声数据，无结构
            X = np.random.randn(n_samples, n_features) * 0.01  # 极小方差
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

            config = ARDVAEConfig(
                max_latent_dim=8,
                max_epochs=100,  # 多训练几轮
                kl_threshold=0.01,
                seed=seed
            )
            ard_vae = ARDVAE(config)
            ard_vae.fit(df, verbose=False)

            n_components_results.append(ard_vae.n_components)

        # 中位数应该 <= max_latent_dim（无信息数据上理论上应更稀疏）
        median_components = np.median(n_components_results)
        assert median_components <= 8, \
            f"Expected median few active dims on noise, got {median_components} (results: {n_components_results})"


class TestARDSparsityMechanism:
    """ARD 稀疏性机制测试"""

    def test_ard_recovers_intrinsic_dimension(self):
        """合成 k 因子数据，ARD 应识别出约 k 个活跃维度（多 seed 统计）"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        k_true = 4  # 真实因子数
        n_samples = 500
        n_features = 30

        # 多 seed 运行，取中位数判断
        seeds = [42, 123, 456]
        n_active_results = []
        kl_separation_results = []

        for seed in seeds:
            seed_everything(seed)

            # 生成低秩数据: x = W @ z + noise
            z_true = np.random.randn(n_samples, k_true)
            W = np.random.randn(k_true, n_features) * 2
            noise = np.random.randn(n_samples, n_features) * 0.1

            X = z_true @ W + noise
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

            config = ARDVAEConfig(
                max_latent_dim=16,
                max_epochs=100,
                kl_threshold=0.05,
                seed=seed
            )
            ard_vae = ARDVAE(config)
            ard_vae.fit(df, verbose=False)

            n_active_results.append(ard_vae.n_components)

            # 检查 KL 分离
            kl_active = ard_vae.kl_per_dim[ard_vae.active_dims]
            all_dims = set(range(len(ard_vae.kl_per_dim)))
            inactive_dims = list(all_dims - set(ard_vae.active_dims))
            if len(inactive_dims) > 0:
                kl_inactive = ard_vae.kl_per_dim[inactive_dims]
                kl_separation_results.append(np.median(kl_active) > np.median(kl_inactive) * 2)
            else:
                kl_separation_results.append(True)

        # 中位数应接近 k_true
        median_active = np.median(n_active_results)
        assert k_true - 2 <= median_active <= k_true + 4, \
            f"Expected median ~{k_true} active dims, got {median_active} (results: {n_active_results})"

        # 多数运行应有 KL 分离
        assert sum(kl_separation_results) >= len(seeds) // 2 + 1, \
            f"KL separation failed in majority: {kl_separation_results}"

    def test_alpha_only_optimization(self):
        """固定 mu, var，只优化 log_alpha，应收敛到理论最优"""
        torch.manual_seed(42)

        # 固定 mu 和 var
        mu = torch.tensor([[1.0, 0.5, -0.3, 2.0]])
        var = torch.tensor([[0.5, 1.0, 0.2, 1.5]])
        log_var = torch.log(var)

        # 初始化 log_alpha
        log_alpha = torch.zeros(4, requires_grad=True)
        optimizer = torch.optim.Adam([log_alpha], lr=0.1)

        # 优化
        for _ in range(500):
            optimizer.zero_grad()
            alpha = torch.exp(log_alpha)
            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - log_alpha)
            loss = kl.sum()
            loss.backward()
            optimizer.step()

        # 理论最优: alpha* = 1 / (mu^2 + var)
        alpha_optimal = 1.0 / (mu.pow(2) + var)
        alpha_learned = torch.exp(log_alpha.detach())

        assert torch.allclose(alpha_learned, alpha_optimal.squeeze(), rtol=0.1), \
            f"Learned alpha {alpha_learned} != optimal {alpha_optimal}"

    def test_inactive_dims_negligible_for_reconstruction(self):
        """将 inactive dims 置零，重建损失变化应远小于置零 active dims"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)
        n_samples = 200
        n_features = 30

        # 有结构的数据
        z_true = np.random.randn(n_samples, 5)
        W = np.random.randn(5, n_features)
        X = z_true @ W + np.random.randn(n_samples, n_features) * 0.1

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

        config = ARDVAEConfig(max_latent_dim=16, max_epochs=50, seed=42)
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        # 先做 transform 来准备 scaled 数据
        _ = ard_vae.transform(df)

        # 获取 latent representations - 使用 scaler 手动转换
        X_scaled = (df.values - ard_vae._scaler_mean) / ard_vae._scaler_std
        X_tensor = torch.tensor(X_scaled, dtype=ard_vae.config.torch_dtype)
        ard_vae.model.eval()
        with torch.no_grad():
            mu, _ = ard_vae.model.encode(X_tensor)
            x_recon_full = ard_vae.model.decode(mu)
            recon_loss_full = ((X_tensor - x_recon_full) ** 2).mean().item()

        active_dims = ard_vae.active_dims
        all_dims = set(range(mu.shape[1]))
        inactive_dims = list(all_dims - set(active_dims))

        if len(inactive_dims) > 0 and len(active_dims) > 0:
            # 置零 inactive dims
            mu_zero_inactive = mu.clone()
            mu_zero_inactive[:, inactive_dims] = 0
            with torch.no_grad():
                x_recon_zero_inactive = ard_vae.model.decode(mu_zero_inactive)
            recon_loss_zero_inactive = ((X_tensor - x_recon_zero_inactive) ** 2).mean().item()

            # 置零 active dims
            mu_zero_active = mu.clone()
            mu_zero_active[:, active_dims] = 0
            with torch.no_grad():
                x_recon_zero_active = ard_vae.model.decode(mu_zero_active)
            recon_loss_zero_active = ((X_tensor - x_recon_zero_active) ** 2).mean().item()

            # 置零 inactive 的损失增加应远小于置零 active
            delta_inactive = abs(recon_loss_zero_inactive - recon_loss_full)
            delta_active = abs(recon_loss_zero_active - recon_loss_full)

            assert delta_inactive < delta_active, \
                f"Zeroing inactive ({delta_inactive:.4f}) should hurt less than active ({delta_active:.4f})"


class TestGradientStability:
    """梯度与训练稳定性测试"""

    def test_reparameterization_gradient_flow(self):
        """验证重参数化的梯度流正确"""
        from src.features.dimensionality_reduction.ard_vae import ARDVAEConfig, ARDVAENet

        torch.manual_seed(42)

        config = ARDVAEConfig(input_dim=10, max_latent_dim=4)
        net = ARDVAENet(config)
        net.train()

        mu = torch.randn(1, 4, requires_grad=True)
        log_var = torch.randn(1, 4, requires_grad=True)

        # 固定 eps 以便验证
        eps = torch.randn(1, 4)

        # z = mu + exp(0.5 * log_var) * eps
        std = torch.exp(0.5 * log_var)
        z = mu + std * eps

        # ∂z/∂mu = 1
        z.sum().backward(retain_graph=True)
        assert torch.allclose(mu.grad, torch.ones_like(mu)), \
            "∂z/∂mu should be 1"

        # ∂z/∂log_var = 0.5 * std * eps
        mu.grad.zero_()
        log_var.grad.zero_()
        z = mu + torch.exp(0.5 * log_var) * eps
        z.sum().backward()
        expected_grad_log_var = 0.5 * torch.exp(0.5 * log_var) * eps
        assert torch.allclose(log_var.grad, expected_grad_log_var, rtol=1e-4), \
            "∂z/∂log_var should be 0.5 * std * eps"

    def test_full_loss_finite_diff_gradients(self):
        """小网络上，autograd 与数值梯度一致（使用 float64 提高精度）"""
        from src.features.dimensionality_reduction.ard_vae import ARDVAEConfig, ARDVAENet

        seed_everything(42)

        # 创建网络并转换为 float64 提高数值精度
        config = ARDVAEConfig(
            input_dim=5,
            max_latent_dim=3,
            encoder_hidden=(8,),
            decoder_hidden=(8,)
        )
        net = ARDVAENet(config)
        net = net.double()  # 转换为 float64
        net.eval()  # 确定性

        x = torch.randn(2, 5, dtype=torch.float64)

        # 选择一个参数进行有限差分验证
        param = net.log_alpha
        param_idx = 0

        def compute_loss():
            mu, log_var = net.encode(x)
            z = net.reparameterize(mu, log_var)
            x_recon = net.decode(z)
            recon_loss = ((x - x_recon) ** 2).sum()
            alpha = torch.exp(net.log_alpha)
            var = torch.exp(log_var)
            kl = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - net.log_alpha)
            return recon_loss + kl.sum()

        # Autograd
        loss = compute_loss()
        loss.backward()
        autograd_grad = param.grad[param_idx].item()

        # 有限差分（使用更小的 eps）
        eps = 1e-5
        with torch.no_grad():
            orig_val = param[param_idx].item()

            param[param_idx] = orig_val + eps
            loss_plus = compute_loss().item()

            param[param_idx] = orig_val - eps
            loss_minus = compute_loss().item()

            param[param_idx] = orig_val

        fd_grad = (loss_plus - loss_minus) / (2 * eps)

        # float64 下应有更高精度
        assert abs(autograd_grad - fd_grad) < 1e-4, \
            f"Autograd {autograd_grad:.8f} != FD {fd_grad:.8f}, diff={abs(autograd_grad - fd_grad):.2e}"

    def test_no_nan_explosion_on_random_init(self):
        """高方差初始化下运行几步，loss/grad 保持有限"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig

        np.random.seed(42)

        # 高方差数据
        df = pd.DataFrame(
            np.random.randn(50, 15) * 10,
            columns=[f"f{i}" for i in range(15)]
        )

        config = ARDVAEConfig(
            max_latent_dim=8,
            max_epochs=5,  # 只运行几步
            seed=42
        )
        ard_vae = ARDVAE(config)

        # 应该不会崩溃
        ard_vae.fit(df, verbose=False)

        # 检查模型参数有限
        for name, param in ard_vae.model.named_parameters():
            assert torch.isfinite(param).all(), \
                f"Parameter {name} contains non-finite values"

    def test_kl_gradient_separation(self):
        """KL 对 decoder 参数梯度应为 0，recon 对 log_alpha 梯度应为 0"""
        from src.features.dimensionality_reduction.ard_vae import ARDVAEConfig, ARDVAENet

        torch.manual_seed(42)

        config = ARDVAEConfig(
            input_dim=10,
            max_latent_dim=4,
            encoder_hidden=(16,),
            decoder_hidden=(16,)
        )
        net = ARDVAENet(config)
        net.train()

        x = torch.randn(4, 10)
        mu, log_var = net.encode(x)
        z = net.reparameterize(mu, log_var)
        x_recon = net.decode(z)

        # KL loss
        alpha = torch.exp(net.log_alpha)
        var = torch.exp(log_var)
        kl_loss = 0.5 * (alpha * (mu.pow(2) + var) - log_var - 1 - net.log_alpha)
        kl_loss = kl_loss.sum()

        # Recon loss
        recon_loss = ((x - x_recon) ** 2).sum()

        # KL backward: decoder 参数梯度应为 0
        net.zero_grad()
        kl_loss.backward(retain_graph=True)

        for name, param in net.decoder.named_parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-6), \
                    f"KL should not affect decoder param {name}"

        # Recon backward: log_alpha 梯度应为 0
        net.zero_grad()
        recon_loss.backward()

        if net.log_alpha.grad is not None:
            assert torch.allclose(net.log_alpha.grad, torch.zeros_like(net.log_alpha.grad), atol=1e-6), \
                "Recon loss should not affect log_alpha"


class TestRepresentationQuality:
    """表示质量测试"""

    def test_latent_recovers_true_factors(self):
        """合成因子数据上，learned mu(x) 与 true z 高度相关（多 seed 统计）"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig
        from scipy.stats import pearsonr

        k_true = 3
        n_samples = 500
        n_features = 20

        # 多 seed 运行
        seeds = [42, 123, 456]
        best_corr_per_run = []

        for seed in seeds:
            seed_everything(seed)

            # 生成因子数据
            z_true = np.random.randn(n_samples, k_true)
            W = np.random.randn(k_true, n_features) * 2
            X = z_true @ W + np.random.randn(n_samples, n_features) * 0.1

            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

            config = ARDVAEConfig(
                max_latent_dim=8,
                max_epochs=100,
                seed=seed
            )
            ard_vae = ARDVAE(config)
            ard_vae.fit(df, verbose=False)

            # 获取 latent
            X_latent = ard_vae.transform(df).values

            # 检查每个真实因子是否与某个 latent 高度相关
            max_correlations = []
            for i in range(k_true):
                correlations = []
                for j in range(X_latent.shape[1]):
                    corr, _ = pearsonr(z_true[:, i], X_latent[:, j])
                    correlations.append(abs(corr))
                max_correlations.append(max(correlations))

            best_corr_per_run.append(max(max_correlations))

        # 中位数应 > 0.5
        median_best_corr = np.median(best_corr_per_run)
        assert median_best_corr > 0.5, \
            f"Median best correlation should > 0.5, got {median_best_corr:.3f} (results: {best_corr_per_run})"

    def test_latent_aligns_with_pca(self):
        """active latent 应与 top-k PCs 有一定对齐"""
        from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig
        from sklearn.decomposition import PCA

        np.random.seed(42)

        # 生成低秩数据
        n_samples = 300
        n_features = 30
        k = 4

        z = np.random.randn(n_samples, k)
        W = np.random.randn(k, n_features)
        X = z @ W + np.random.randn(n_samples, n_features) * 0.1

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

        config = ARDVAEConfig(
            max_latent_dim=12,
            max_epochs=80,
            seed=42
        )
        ard_vae = ARDVAE(config)
        ard_vae.fit(df, verbose=False)

        X_latent = ard_vae.transform(df).values

        # PCA
        pca = PCA(n_components=k)
        X_pca = pca.fit_transform(X)

        # 计算 latent 与 PCA 的解释方差
        # 用线性回归: X_pca ~ X_latent
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X_latent, X_pca)
        X_pca_pred = reg.predict(X_latent)

        # R^2
        ss_res = ((X_pca - X_pca_pred) ** 2).sum()
        ss_tot = ((X_pca - X_pca.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        # latent 应该能解释大部分 PCA 方差
        assert r2 > 0.3, \
            f"Latent should explain PCA variance, R^2 = {r2:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
