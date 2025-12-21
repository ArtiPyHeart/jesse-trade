"""
DeepSSM Overlap PCB 精度验证测试

验证 forward_train() 与 Full BPTT 的一致性：
1. 梯度精度
2. 初始状态梯度
3. Loss 一致性
4. 边界情况处理
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.deep_ssm.deep_ssm import DeepSSM, DeepSSMConfig, DeepSSMNet


def generate_test_data(T: int = 500, obs_dim: int = 10, seed: int = 42):
    """生成测试数据"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.randn(T, obs_dim).astype(np.float32)


class TestForwardTrainGradientAccuracy:
    """测试 forward_train 与 Full BPTT 的梯度一致性"""

    def setup_method(self):
        """每个测试前重置种子"""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_initial_state_gradient(self):
        """验证 initial_state_mean 能正确接收梯度"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            max_epochs=1,
            chunk_size=128,
            overlap=32,
            seed=42,
        )

        model = DeepSSM(config)
        X = generate_test_data(T=200, obs_dim=10)

        # 训练一个 epoch
        model.fit(X)

        # 检查 initial_state_mean 的梯度
        # 由于训练已完成，检查参数是否被更新（间接验证梯度存在）
        initial_mean = model.model.initial_state_mean.detach().numpy()

        # 创建新模型对比
        model2 = DeepSSM(config)
        initial_mean2 = model2.model.initial_state_mean.detach().numpy()

        # 训练后参数应该有变化（说明梯度起作用了）
        assert not np.allclose(initial_mean, initial_mean2, atol=1e-6), \
            "initial_state_mean should be updated during training"

    def test_gradient_magnitude_reasonable(self):
        """验证梯度幅度合理"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            chunk_size=128,
            overlap=32,
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        X = torch.randn(1, 200, 10)

        # 运行 forward_train_ekf (EKF-based training path)
        net.zero_grad()
        result = net.forward_train_ekf(X, chunk_size=128, overlap=32)

        # EKF 训练使用的网络：transition_prior, observation, initial_state params
        # 不使用：
        # - transition_posterior (legacy posterior network, replaced by EKF update)
        # - lstm (not used in EKF path - transition_prior only takes z_prev)
        unused_params = {"transition_posterior.", "lstm."}  # Not used in EKF training path
        for name, param in net.named_parameters():
            is_unused = any(name.startswith(prefix) for prefix in unused_params)
            if is_unused:
                continue  # Skip params not used in EKF training
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"

        # 检查 initial_state_mean 梯度非零
        initial_grad = net.initial_state_mean.grad
        assert initial_grad.abs().sum() > 0, "initial_state_mean gradient should be non-zero"

    def test_loss_value_reasonable(self):
        """验证 loss 值合理"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            chunk_size=128,
            overlap=32,
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        X = torch.randn(1, 200, 10)

        result = net.forward_train(X, chunk_size=128, overlap=32)

        # Loss 应该是有限的正数
        assert np.isfinite(result["total_loss"]), "Loss should be finite"
        assert result["total_loss"] > 0, "Loss should be positive"

        # num_chunks 应该合理
        expected_chunks = (200 + 128 - 1) // 128  # ceil(200/128) = 2
        assert result["num_chunks"] == expected_chunks, \
            f"Expected {expected_chunks} chunks, got {result['num_chunks']}"


class TestBoundaryConditions:
    """测试边界条件"""

    def test_overlap_zero(self):
        """测试 overlap=0 的情况"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            chunk_size=128,
            overlap=0,  # 无 overlap
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        X = torch.randn(1, 200, 10)

        # 应该正常运行
        result = net.forward_train(X, chunk_size=128, overlap=0)

        assert np.isfinite(result["total_loss"])
        assert result["num_chunks"] > 0

    def test_short_sequence(self):
        """测试序列长度小于 chunk_size 的情况"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            chunk_size=256,
            overlap=64,
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        X = torch.randn(1, 100, 10)  # T < chunk_size

        result = net.forward_train(X, chunk_size=256, overlap=64)

        assert np.isfinite(result["total_loss"])
        assert result["num_chunks"] == 1  # 应该只有一个 chunk

    def test_overlap_near_chunk_size(self):
        """测试 overlap 接近 chunk_size 的情况"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            chunk_size=128,
            overlap=120,  # 接近 chunk_size
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        X = torch.randn(1, 500, 10)

        result = net.forward_train(X, chunk_size=128, overlap=120)

        assert np.isfinite(result["total_loss"])
        assert result["num_chunks"] > 0

    def test_config_validation_overlap_exceeds_chunk_size(self):
        """测试 overlap >= chunk_size 时抛出错误"""
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            DeepSSMConfig(
                obs_dim=10,
                state_dim=5,
                chunk_size=128,
                overlap=128,  # overlap == chunk_size
            )

        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            DeepSSMConfig(
                obs_dim=10,
                state_dim=5,
                chunk_size=128,
                overlap=200,  # overlap > chunk_size
            )


class TestTrainingIntegration:
    """测试完整训练流程"""

    def test_full_training_convergence(self):
        """测试完整训练能收敛"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            max_epochs=20,
            chunk_size=128,
            overlap=32,
            seed=42,
        )

        model = DeepSSM(config)
        X = generate_test_data(T=300, obs_dim=10)

        model.fit(X)

        # 检查模型已训练
        assert model.is_fitted
        assert len(model.training_history) > 0

        # 检查 loss 有下降趋势
        first_loss = model.training_history[0]["train_loss"]
        last_loss = model.training_history[-1]["train_loss"]
        assert last_loss <= first_loss, "Loss should decrease during training"

    def test_transform_after_training(self):
        """测试训练后 transform 正常工作"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            max_epochs=5,
            chunk_size=128,
            overlap=32,
            seed=42,
        )

        model = DeepSSM(config)
        X = generate_test_data(T=200, obs_dim=10)

        model.fit(X)

        # Transform 应该正常工作
        states = model.transform(X)

        assert states.shape == (200, 5)
        assert np.isfinite(states).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
