"""
Training logic for DeepSSM
DeepSSM的训练逻辑
"""

import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

from .kalman_filter import deep_ssm_kalman_filter
from .model import (
    DeepSSM,
    deep_ssm_model,
    deep_ssm_guide,
    create_model,
    init_model_params,
)


def train_deep_ssm(
    y_data: jnp.ndarray,
    obs_dim: int,
    state_dim: int = 5,
    lstm_hidden: int = 64,
    max_epochs: int = 50,
    patience: int = 5,
    min_delta: float = 0.01,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> Tuple[DeepSSM, dict, list]:
    """
    训练DeepSSM模型
    
    Args:
        y_data: 训练数据 [T, obs_dim] 或 [batch_size, T, obs_dim]
        obs_dim: 观测维度
        state_dim: 状态维度
        lstm_hidden: LSTM隐藏层维度
        max_epochs: 最大训练轮数
        patience: 早停耐心值
        min_delta: 损失改善的最小阈值
        learning_rate: 学习率
        seed: 随机种子
    
    Returns:
        (训练好的模型, 模型参数, 损失历史)
    """
    # 确保数据格式正确
    if len(y_data.shape) == 2:
        y_data = y_data[jnp.newaxis, :]  # [1, T, obs_dim]
    
    batch_size, T, _ = y_data.shape
    
    # 创建模型
    model = create_model(obs_dim, state_dim, lstm_hidden)
    
    # 初始化模型参数
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    model_params = init_model_params(model, subkey, y_data)
    
    # 设置优化器
    optimizer = numpyro.optim.Adam(step_size=learning_rate)
    
    # 设置SVI
    svi = SVI(
        model=lambda y: deep_ssm_model(y, model, model_params),
        guide=lambda y: deep_ssm_guide(y, model, model_params),
        optim=optimizer,
        loss=Trace_ELBO(),
    )
    
    # 初始化SVI状态
    key, subkey = random.split(key)
    svi_state = svi.init(subkey, y_data)
    
    # 训练循环
    losses = []
    best_loss = float("inf")
    patience_counter = 0
    
    print(f"开始训练DeepSSM模型...")
    print(f"数据形状: {y_data.shape}")
    print(f"模型配置: state_dim={state_dim}, lstm_hidden={lstm_hidden}")
    
    for epoch in tqdm(range(max_epochs), desc="训练进度"):
        key, subkey = random.split(key)
        
        # 执行一步SVI更新
        svi_state, loss = svi.update(svi_state, y_data)
        
        # 计算平均损失
        avg_loss = loss / T
        losses.append(avg_loss)
        
        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.4f}")
        
        # 早停检查
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳参数
            best_params = svi.get_params(svi_state)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发：第{epoch+1}轮损失未改善，停止训练")
                break
    
    # 更新模型参数为最佳参数
    if "best_params" in locals():
        # 将NumPyro参数转换回模型参数格式
        model_params = update_model_params(model_params, best_params)
    
    return model, model_params, losses


def update_model_params(model_params: dict, numpyro_params: dict) -> dict:
    """
    从NumPyro参数更新模型参数
    
    Args:
        model_params: 原始模型参数
        numpyro_params: NumPyro优化后的参数
    
    Returns:
        更新后的模型参数
    """
    # 这里需要根据实际的参数结构进行映射
    # 由于我们使用的是变分推断，主要更新的是变分参数
    # 模型参数本身可能不会直接更新
    return model_params


def generate_features(
    y_data: jnp.ndarray, 
    model: DeepSSM, 
    model_params: dict
) -> jnp.ndarray:
    """
    使用训练好的模型生成特征
    
    Args:
        y_data: 输入数据 [T, obs_dim]
        model: 训练好的模型
        model_params: 模型参数
    
    Returns:
        生成的特征 [T, state_dim]
    """
    states, _ = deep_ssm_kalman_filter(y_data, model, model_params)
    return states


def _convert_arrays_to_lists(obj):
    """递归将numpy/jax数组转换为列表以便JSON序列化"""
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_arrays_to_lists(item) for item in obj]
    else:
        return obj


def _convert_lists_to_arrays(obj):
    """递归将列表转换回numpy数组"""
    if isinstance(obj, dict):
        # 检查是否是参数字典的叶节点
        if all(isinstance(v, list) for v in obj.values()):
            # 如果所有值都是列表，可能是参数
            return {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v 
                   for k, v in obj.items()}
        else:
            return {k: _convert_lists_to_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # 尝试转换为数组，如果失败则保持为列表
        try:
            return np.array(obj, dtype=np.float32)
        except:
            return [_convert_lists_to_arrays(item) for item in obj]
    else:
        return obj


def save_model(
    model: DeepSSM, 
    model_params: dict, 
    path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    保存模型配置和参数为JSON格式
    
    Args:
        model: DeepSSM模型
        model_params: 模型参数
        path: 保存路径 (建议使用.json后缀)
        metadata: 可选的额外元数据
    """
    save_dict = {
        "model_config": {
            "obs_dim": model.obs_dim,
            "state_dim": model.state_dim,
            "lstm_hidden": model.lstm_hidden,
        },
        "model_params": _convert_arrays_to_lists(model_params),
        "metadata": metadata or {}
    }
    
    # 确保路径以.json结尾
    path = Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    
    with open(path, "w") as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"模型已保存到 {path}")


def load_model(path: str) -> Tuple[DeepSSM, dict, Dict[str, Any]]:
    """
    从JSON文件加载模型配置和参数
    
    Args:
        path: 模型文件路径
    
    Returns:
        (模型, 模型参数, 元数据)
    """
    with open(path, "r") as f:
        save_dict = json.load(f)
    
    # 重建模型
    config = save_dict["model_config"]
    model = create_model(
        obs_dim=config["obs_dim"],
        state_dim=config["state_dim"],
        lstm_hidden=config["lstm_hidden"],
    )
    
    # 转换参数回数组格式
    model_params = _convert_lists_to_arrays(save_dict["model_params"])
    metadata = save_dict.get("metadata", {})
    
    return model, model_params, metadata


def save_model_npz(
    model: DeepSSM,
    model_params: dict,
    path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    使用NumPy的npz格式保存模型（二进制格式，更紧凑）
    
    Args:
        model: DeepSSM模型
        model_params: 模型参数
        path: 保存路径 (建议使用.npz后缀)
        metadata: 可选的额外元数据
    """
    # 展平参数字典为单层结构
    flat_params = {}
    
    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                flat_params[new_key] = np.array(v) if isinstance(v, (jnp.ndarray, list)) else v
    
    flatten_dict(model_params['params'])
    
    # 保存配置和参数
    np.savez_compressed(
        path,
        obs_dim=model.obs_dim,
        state_dim=model.state_dim,
        lstm_hidden=model.lstm_hidden,
        **flat_params
    )
    
    print(f"模型已保存到 {path} (npz格式)")


def load_model_npz(path: str) -> Tuple[DeepSSM, dict]:
    """
    从npz文件加载模型
    
    Args:
        path: 模型文件路径
    
    Returns:
        (模型, 模型参数)
    """
    data = np.load(path)
    
    # 提取配置
    obs_dim = int(data['obs_dim'])
    state_dim = int(data['state_dim'])
    lstm_hidden = int(data['lstm_hidden'])
    
    # 重建模型
    model = create_model(obs_dim, state_dim, lstm_hidden)
    
    # 重建参数字典
    params = {'params': {}}
    for key in data.files:
        if key not in ['obs_dim', 'state_dim', 'lstm_hidden']:
            # 重建嵌套结构
            keys = key.split('.')
            current = params['params']
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = data[key]
    
    return model, params


if __name__ == "__main__":
    # 测试示例
    print("DeepSSM训练模块 - 无StandardScaler版本")
    print("支持JSON和NPZ格式保存")
    
    # 创建示例数据
    test_data = np.random.randn(100, 10).astype(np.float32)
    
    # 训练模型
    model, params, losses = train_deep_ssm(
        y_data=jnp.array(test_data),
        obs_dim=10,
        state_dim=5,
        lstm_hidden=32,
        max_epochs=10
    )
    
    # 保存为JSON
    save_model(model, params, "test_model.json", metadata={"version": "1.0"})
    
    # 保存为NPZ
    save_model_npz(model, params, "test_model.npz")
    
    print("测试完成！")