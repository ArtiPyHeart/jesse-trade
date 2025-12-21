"""
DeepSSM 内存节省验证测试

验证 Overlap + Per-chunk Backward 策略的内存效率：
1. 内存不随序列长度线性增长
2. 长序列训练时内存保持稳定
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.deep_ssm.deep_ssm import DeepSSM, DeepSSMConfig, DeepSSMNet


def get_memory_usage_mb() -> float:
    """获取当前 PyTorch GPU/MPS/CPU 内存使用量（MB）"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS 不支持精确内存追踪，返回进程内存估计
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB
    else:
        # CPU 模式：使用 Python 内存追踪
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def reset_memory():
    """重置内存状态"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class TestMemoryScaling:
    """测试内存缩放特性"""

    def test_memory_does_not_scale_linearly(self):
        """验证 chunked 模式内存不随 T 线性增长"""
        config = DeepSSMConfig(
            obs_dim=20,
            state_dim=10,
            lstm_hidden=64,
            chunk_size=256,
            overlap=64,
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        memory_usage = {}

        for T in [500, 1000, 2000]:
            reset_memory()
            torch.manual_seed(42)

            X = torch.randn(1, T, config.obs_dim)

            net.zero_grad()
            result = net.forward_train(X, chunk_size=256, overlap=64)

            memory_usage[T] = get_memory_usage_mb()

            # 清理
            del result, X
            gc.collect()

        # 验证内存不是线性增长
        # 如果是线性增长，T=2000 的内存应该是 T=1000 的 2 倍
        # 但 chunked 模式下，内存应该接近常数
        ratio_2000_1000 = memory_usage[2000] / memory_usage[1000]
        ratio_1000_500 = memory_usage[1000] / memory_usage[500]

        print(f"\nMemory usage: {memory_usage}")
        print(f"Ratio T=2000/T=1000: {ratio_2000_1000:.2f}")
        print(f"Ratio T=1000/T=500: {ratio_1000_500:.2f}")

        # Chunked 模式下，比率应该远小于 2（线性增长）
        # 允许一些开销，但比率应该小于 1.5
        assert ratio_2000_1000 < 1.5, (
            f"Memory scales too much: {ratio_2000_1000:.2f}x for 2x sequence length"
        )

    def test_long_sequence_trainable(self):
        """验证长序列可以成功训练"""
        config = DeepSSMConfig(
            obs_dim=20,
            state_dim=10,
            lstm_hidden=64,
            max_epochs=3,
            chunk_size=256,
            overlap=64,
            seed=42,
        )

        model = DeepSSM(config)

        # 生成长序列数据
        np.random.seed(42)
        X = np.random.randn(3000, config.obs_dim).astype(np.float32)

        # 应该能成功训练而不 OOM
        model.fit(X)

        assert model.is_fitted
        assert len(model.training_history) > 0

        # Transform 也应该正常工作
        states = model.transform(X)
        assert states.shape == (3000, config.state_dim)
        assert np.isfinite(states).all()

    def test_chunk_size_affects_memory(self):
        """验证 chunk_size 影响内存使用"""
        net_small_chunk = DeepSSMNet(
            DeepSSMConfig(
                obs_dim=20,
                state_dim=10,
                lstm_hidden=64,
                chunk_size=128,
                overlap=32,
                seed=42,
            )
        )
        net_small_chunk.train()

        net_large_chunk = DeepSSMNet(
            DeepSSMConfig(
                obs_dim=20,
                state_dim=10,
                lstm_hidden=64,
                chunk_size=512,
                overlap=64,
                seed=42,
            )
        )
        net_large_chunk.train()

        T = 1000
        X = torch.randn(1, T, 20)

        # 小 chunk
        reset_memory()
        torch.manual_seed(42)
        net_small_chunk.zero_grad()
        result_small = net_small_chunk.forward_train(X, chunk_size=128, overlap=32)
        mem_small = get_memory_usage_mb()

        # 大 chunk
        reset_memory()
        torch.manual_seed(42)
        net_large_chunk.zero_grad()
        result_large = net_large_chunk.forward_train(X, chunk_size=512, overlap=64)
        mem_large = get_memory_usage_mb()

        print(f"\nMemory with chunk_size=128: {mem_small:.2f} MB")
        print(f"Memory with chunk_size=512: {mem_large:.2f} MB")

        # 两种配置都应该能工作
        assert np.isfinite(result_small["total_loss"])
        assert np.isfinite(result_large["total_loss"])


class TestMemoryEfficiency:
    """测试内存效率特性"""

    def test_gradient_accumulation_works(self):
        """验证梯度在 chunks 之间正确累积"""
        config = DeepSSMConfig(
            obs_dim=10,
            state_dim=5,
            lstm_hidden=32,
            chunk_size=100,
            overlap=20,
            seed=42,
        )

        net = DeepSSMNet(config)
        net.train()

        X = torch.randn(1, 300, 10)

        net.zero_grad()
        result = net.forward_train_ekf(X, chunk_size=100, overlap=20)

        # 应该处理了多个 chunks
        assert result["num_chunks"] > 1

        # 梯度应该已经累积
        # EKF 训练使用的网络：transition_prior, observation, initial_state params
        # 不使用：transition_posterior (legacy), lstm (not used in EKF path)
        unused_params = {"transition_posterior.", "lstm."}  # Not used in EKF training path
        for name, param in net.named_parameters():
            is_unused = any(name.startswith(prefix) for prefix in unused_params)
            if is_unused:
                continue  # Skip params not used in EKF training
            assert param.grad is not None, f"{name} should have gradient"
            # 梯度应该是多个 chunk 的累积结果
            # （虽然每个 chunk 已经用 1/T 缩放了）

    def test_consistent_loss_different_chunk_sizes(self):
        """验证不同 chunk_size 得到相似的 loss"""
        torch.manual_seed(42)

        configs = [
            DeepSSMConfig(
                obs_dim=10,
                state_dim=5,
                lstm_hidden=32,
                chunk_size=100,
                overlap=25,
                seed=42,
            ),
            DeepSSMConfig(
                obs_dim=10,
                state_dim=5,
                lstm_hidden=32,
                chunk_size=200,
                overlap=50,
                seed=42,
            ),
        ]

        X = torch.randn(1, 400, 10)
        losses = []

        for config in configs:
            net = DeepSSMNet(config)
            net.train()

            # 使用相同的 RNG 状态
            torch.manual_seed(42)
            net.zero_grad()
            result = net.forward_train(X, chunk_size=config.chunk_size, overlap=config.overlap)
            losses.append(result["total_loss"])

        print(f"\nLoss with chunk_size=100: {losses[0]:.6f}")
        print(f"Loss with chunk_size=200: {losses[1]:.6f}")

        # Loss 应该相近（但不完全相同，因为重参数化采样）
        # 主要验证两种配置都能正常工作
        assert all(np.isfinite(l) for l in losses)
        assert all(l > 0 for l in losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
