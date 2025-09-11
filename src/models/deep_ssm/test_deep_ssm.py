"""
Test script for DeepSSM JAX implementation
DeepSSM JAX实现的测试脚本
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.deep_ssm import deep_ssm_kalman_filter, train_deep_ssm
from src.models.deep_ssm.model import create_model, init_model_params
from src.models.deep_ssm.training import (
    load_data,
    generate_features,
    save_model,
    load_model,
)
from src.models.deep_ssm.inference import create_realtime_processor


def test_model_creation():
    """测试模型创建"""
    print("\n测试1: 模型创建")
    print("-" * 50)

    obs_dim = 77
    state_dim = 5
    lstm_hidden = 64

    model = create_model(obs_dim, state_dim, lstm_hidden)
    print(f"✓ 模型创建成功")
    print(f"  观测维度: {model.obs_dim}")
    print(f"  状态维度: {model.state_dim}")
    print(f"  LSTM隐藏层: {model.lstm_hidden}")

    # 初始化参数
    key = random.PRNGKey(42)
    sample_input = jnp.zeros((1, 10, obs_dim))
    params = init_model_params(model, key, sample_input)
    print(f"✓ 参数初始化成功")
    print(f"  参数键: {list(params.keys())}")

    return model, params


def test_kalman_filter(model, params):
    """测试卡尔曼滤波器"""
    print("\n测试2: 卡尔曼滤波器")
    print("-" * 50)

    # 创建测试数据
    T = 100
    obs_dim = model.obs_dim
    y_seq = jnp.array(np.random.randn(T, obs_dim), dtype=jnp.float32)

    # 运行卡尔曼滤波
    states, P = deep_ssm_kalman_filter(y_seq, model, params)

    print(f"✓ 卡尔曼滤波成功")
    print(f"  输入形状: {y_seq.shape}")
    print(f"  状态序列形状: {states.shape}")
    print(f"  协方差矩阵形状: {P.shape}")

    # 检查输出维度
    assert states.shape == (T, model.state_dim), "状态序列维度错误"
    assert P.shape == (model.state_dim, model.state_dim), "协方差矩阵维度错误"
    print(f"✓ 输出维度验证通过")

    return states


def test_training():
    """测试训练流程"""
    print("\n测试3: 训练流程")
    print("-" * 50)

    # 创建合成数据
    T = 500
    obs_dim = 10
    state_dim = 3

    # 生成合成时间序列
    t = np.linspace(0, 4 * np.pi, T)
    data = np.column_stack(
        [np.sin(t + i * 0.5) + 0.1 * np.random.randn(T) for i in range(obs_dim)]
    )
    y_data = jnp.array(data, dtype=jnp.float32)

    # 训练模型
    model, params, losses = train_deep_ssm(
        y_data=y_data,
        obs_dim=obs_dim,
        state_dim=state_dim,
        lstm_hidden=32,
        max_epochs=20,  # 减少轮数用于测试
        patience=5,
        learning_rate=0.001,
    )

    print(f"✓ 训练完成")
    print(f"  训练轮数: {len(losses)}")
    print(f"  最终损失: {losses[-1]:.4f}")
    print(f"  损失下降: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # 生成特征
    features = generate_features(y_data, model, params)
    print(f"✓ 特征生成成功")
    print(f"  特征形状: {features.shape}")

    assert features.shape == (T, state_dim), "特征维度错误"

    return model, params


def test_realtime_inference():
    """测试实时推理"""
    print("\n测试4: 实时推理")
    print("-" * 50)

    obs_dim = 10
    state_dim = 3

    # 创建实时处理器
    processor = create_realtime_processor(
        obs_dim=obs_dim, state_dim=state_dim, lstm_hidden=32
    )

    print(f"✓ 实时处理器创建成功")

    # 测试单步处理
    features = []
    for i in range(10):
        new_data = np.random.randn(obs_dim)
        feature = processor.process(new_data)
        features.append(feature)

        if i == 0:
            print(f"✓ 单步处理成功")
            print(f"  输入形状: {new_data.shape}")
            print(f"  输出形状: {feature.shape}")

    # 测试批处理
    batch_data = np.random.randn(5, obs_dim)
    processor.reset()  # 重置状态
    batch_features = processor.process_batch(batch_data)

    print(f"✓ 批处理成功")
    print(f"  批输入形状: {batch_data.shape}")
    print(f"  批输出形状: {batch_features.shape}")

    # 测试状态管理
    state = processor.get_state()
    print(f"✓ 状态获取成功")
    print(f"  状态键: {list(state.keys())}")

    processor.set_state(state)
    print(f"✓ 状态设置成功")

    processor.reset()
    print(f"✓ 状态重置成功")

    return processor


def test_save_load():
    """测试模型保存和加载"""
    print("\n测试5: 模型保存/加载")
    print("-" * 50)

    # 创建模型
    obs_dim = 10
    state_dim = 3
    model = create_model(obs_dim, state_dim, lstm_hidden=32)

    # 初始化参数
    key = random.PRNGKey(42)
    sample_input = jnp.zeros((1, 10, obs_dim))
    params = init_model_params(model, key, sample_input)

    # 创建标准化器（模拟）
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, obs_dim))

    # 保存模型
    save_path = "test_model.pkl"
    save_model(model, params, scaler, save_path)
    print(f"✓ 模型保存成功: {save_path}")

    # 加载模型
    loaded_model, loaded_params, loaded_scaler = load_model(save_path)
    print(f"✓ 模型加载成功")
    print(
        f"  加载的模型维度: obs={loaded_model.obs_dim}, state={loaded_model.state_dim}"
    )

    # 清理测试文件
    os.remove(save_path)
    print(f"✓ 测试文件已清理")

    return loaded_model, loaded_params


def test_with_real_data():
    """使用真实数据测试（如果存在）"""
    print("\n测试6: 真实数据测试")
    print("-" * 50)

    csv_path = "extern/DeepSSM/np_fracdiff_features.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"⚠ 真实数据文件不存在: {csv_path}")
        print("  跳过真实数据测试")
        return

    # 加载数据
    y_data, scaler = load_data(csv_path)
    T, obs_dim = y_data.shape
    print(f"✓ 数据加载成功")
    print(f"  数据形状: {y_data.shape}")

    # 使用小批量进行快速测试
    y_small = y_data[:200]  # 只使用前200个样本

    # 训练模型
    print("  训练小模型进行测试...")
    model, params, losses = train_deep_ssm(
        y_data=y_small,
        obs_dim=obs_dim,
        state_dim=5,
        lstm_hidden=64,
        max_epochs=10,  # 快速测试
        patience=3,
        learning_rate=0.001,
    )

    print(f"✓ 真实数据训练成功")
    print(f"  最终损失: {losses[-1]:.4f}")

    # 生成特征
    features = generate_features(y_small, model, params)
    print(f"✓ 特征生成成功")
    print(f"  特征形状: {features.shape}")
    print(f"  特征范围: [{np.min(features):.4f}, {np.max(features):.4f}]")

    return model, params, features


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("DeepSSM JAX实现测试套件")
    print("=" * 60)

    try:
        # 测试1: 模型创建
        model, params = test_model_creation()

        # 测试2: 卡尔曼滤波
        states = test_kalman_filter(model, params)

        # 测试3: 训练流程
        trained_model, trained_params = test_training()

        # 测试4: 实时推理
        processor = test_realtime_inference()

        # 测试5: 保存/加载
        loaded_model, loaded_params = test_save_load()

        # 测试6: 真实数据（可选）
        test_with_real_data()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    # 设置JAX配置
    # 启用64位精度（可选）
    # jax.config.update("jax_enable_x64", True)

    # 运行测试
    run_all_tests()
