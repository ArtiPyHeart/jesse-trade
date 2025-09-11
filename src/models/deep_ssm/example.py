"""
DeepSSM使用示例
"""

import numpy as np
import jax.numpy as jnp
from jax import random
import pandas as pd
from pathlib import Path

from .model import create_model, init_model_params
from .kalman_filter import deep_ssm_kalman_filter
from .training import save_model, load_model, save_model_npz, load_model_npz
from .inference import DeepSSMRealTime


def normalize_data(data: np.ndarray) -> np.ndarray:
    """简单的数据标准化（如需要）"""
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8  # 避免除零
    return (data - mean) / std


def main():
    """基本使用示例"""
    
    # 1. 准备数据
    try:
        # 尝试加载真实数据
        data_path = Path(__file__).parent.parent.parent.parent / "data" / "test.csv"
        data = pd.read_csv(data_path)
        print(f"加载数据: {data.shape}")
    except:
        # 创建示例数据
        print("创建示例数据用于演示")
        data = pd.DataFrame(np.random.randn(100, 77))  # 77维特征
        print(f"数据形状: {data.shape}")
    
    # 2. 数据预处理
    data_array = data.values.astype(np.float32)
    
    # 用户可以自行决定是否需要标准化
    # 例如使用sklearn的StandardScaler：
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # data_array = scaler.fit_transform(data_array)
    # 或者使用简单的标准化：
    # data_array = normalize_data(data_array)
    
    # 3. 模型配置
    obs_dim = data.shape[1]
    state_dim = 5  # 潜在状态维度（降维目标）
    lstm_hidden = 64  # LSTM隐藏层大小
    
    print(f"\n模型配置:")
    print(f"  观测维度: {obs_dim}")
    print(f"  状态维度: {state_dim}")
    print(f"  LSTM隐藏层: {lstm_hidden}")
    
    # 4. 创建和初始化模型
    model = create_model(obs_dim, state_dim, lstm_hidden)
    key = random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 10, obs_dim))
    params = init_model_params(model, key, dummy_input)
    
    # 5. 运行卡尔曼滤波生成状态特征
    print("\n运行卡尔曼滤波...")
    states, P = deep_ssm_kalman_filter(
        jnp.array(data_array),
        model,
        params
    )
    
    print(f"生成状态特征:")
    print(f"  形状: {states.shape}")
    print(f"  范围: [{states.min():.4f}, {states.max():.4f}]")
    
    # 6. 保存结果
    features_df = pd.DataFrame(
        np.array(states),
        columns=[f'state_{i}' for i in range(state_dim)]
    )
    features_df.to_csv("deep_ssm_features.csv", index=False)
    print(f"\n特征已保存到 deep_ssm_features.csv")
    
    # 7. 保存模型（支持JSON和NPZ格式）
    # JSON格式（人类可读，支持元数据）
    save_model(model, params, "deep_ssm_model.json", 
              metadata={
                  "created": "2024", 
                  "version": "1.0",
                  "description": "DeepSSM特征提取模型"
              })
    print(f"模型已保存到 deep_ssm_model.json（人类可读）")
    
    # NPZ格式（二进制，更紧凑高效）
    save_model_npz(model, params, "deep_ssm_model.npz")
    print(f"模型已保存到 deep_ssm_model.npz（二进制格式）")
    
    # 8. 演示模型加载
    print("\n演示模型加载:")
    
    # 从JSON加载
    loaded_model, loaded_params, metadata = load_model("deep_ssm_model.json")
    print(f"从JSON加载成功，元数据: {metadata}")
    
    # 从NPZ加载
    loaded_model_npz, loaded_params_npz = load_model_npz("deep_ssm_model.npz")
    print(f"从NPZ加载成功")
    
    # 9. 演示实时推理
    print("\n演示实时推理:")
    realtime_processor = DeepSSMRealTime(
        model=model,
        model_params=params
    )
    
    # 处理单条数据
    new_observation = data_array[0]
    feature = realtime_processor.process(new_observation)
    print(f"实时特征提取: {feature[:3]}...（前3维）")
    
    return states


def demo_workflow():
    """完整工作流程演示"""
    print("=" * 60)
    print("DeepSSM完整工作流程演示")
    print("=" * 60)
    
    # 步骤1: 数据准备
    print("\n步骤1: 数据准备")
    data = pd.DataFrame(np.random.randn(200, 77))
    train_data = data.iloc[:150].values.astype(np.float32)
    test_data = data.iloc[150:].values.astype(np.float32)
    print(f"训练数据: {train_data.shape}")
    print(f"测试数据: {test_data.shape}")
    
    # 步骤2: 创建和初始化模型
    print("\n步骤2: 模型初始化（使用PyTorch风格初始化）")
    model = create_model(obs_dim=77, state_dim=5, lstm_hidden=64)
    key = random.PRNGKey(42)
    params = init_model_params(model, key, jnp.zeros((1, 10, 77)))
    
    # 步骤3: 批量特征提取
    print("\n步骤3: 批量特征提取")
    train_features, _ = deep_ssm_kalman_filter(
        jnp.array(train_data), model, params
    )
    print(f"提取的特征: {train_features.shape}")
    
    # 步骤4: 实时推理
    print("\n步骤4: 实时推理模式")
    processor = DeepSSMRealTime(model=model, model_params=params)
    
    realtime_features = []
    for i in range(5):
        feature = processor.process(test_data[i])
        realtime_features.append(feature)
        print(f"  第{i+1}条: {feature[:3].round(4)}...")
    
    print("\n工作流程演示完成！")


if __name__ == "__main__":
    # 运行基本示例
    print("运行基本示例...")
    states = main()
    
    print("\n" + "=" * 60)
    
    # 运行完整工作流程
    demo_workflow()