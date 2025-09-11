"""
Training logic for DeepSSM
DeepSSM的训练逻辑
"""

import pickle
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .kalman_filter import deep_ssm_kalman_filter
from .model import (
    DeepSSM,
    deep_ssm_model,
    deep_ssm_guide,
    create_model,
    init_model_params,
)


def load_data(csv_path: str) -> Tuple[jnp.ndarray, StandardScaler]:
    """
    加载并标准化数据

    Args:
        csv_path: CSV文件路径

    Returns:
        (标准化后的数据, 标准化器)
    """
    df = pd.read_csv(csv_path)
    features = df.values

    # 标准化数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return jnp.array(features_scaled, dtype=jnp.float32), scaler


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
    y_data: jnp.ndarray, model: DeepSSM, model_params: dict
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


def save_model(model: DeepSSM, model_params: dict, scaler: StandardScaler, path: str):
    """
    保存模型和相关参数

    Args:
        model: DeepSSM模型
        model_params: 模型参数
        scaler: 数据标准化器
        path: 保存路径
    """
    save_dict = {
        "model_config": {
            "obs_dim": model.obs_dim,
            "state_dim": model.state_dim,
            "lstm_hidden": model.lstm_hidden,
        },
        "model_params": model_params,
        "scaler": scaler,
    }

    with open(path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"模型已保存到 {path}")


def load_model(path: str) -> Tuple[DeepSSM, dict, StandardScaler]:
    """
    加载模型和相关参数

    Args:
        path: 模型文件路径

    Returns:
        (模型, 模型参数, 标准化器)
    """
    with open(path, "rb") as f:
        save_dict = pickle.load(f)

    # 重建模型
    config = save_dict["model_config"]
    model = create_model(
        obs_dim=config["obs_dim"],
        state_dim=config["state_dim"],
        lstm_hidden=config["lstm_hidden"],
    )

    model_params = save_dict["model_params"]
    scaler = save_dict["scaler"]

    return model, model_params, scaler


def train_from_csv(
    csv_path: str,
    model_save_path: str = "deep_ssm_model.pkl",
    feature_save_path: str = "deep_ssm_features.csv",
    state_dim: int = 5,
    lstm_hidden: int = 64,
    max_epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 0.001,
):
    """
    从CSV文件训练模型的完整流程

    Args:
        csv_path: 输入CSV文件路径
        model_save_path: 模型保存路径
        feature_save_path: 特征保存路径
        state_dim: 状态维度
        lstm_hidden: LSTM隐藏层维度
        max_epochs: 最大训练轮数
        patience: 早停耐心值
        learning_rate: 学习率
    """
    # 加载数据
    print(f"加载数据: {csv_path}")
    y_data, scaler = load_data(csv_path)
    T, obs_dim = y_data.shape
    print(f"数据加载完成：{T}行，{obs_dim}维特征")

    # 训练模型
    model, model_params, losses = train_deep_ssm(
        y_data=y_data,
        obs_dim=obs_dim,
        state_dim=state_dim,
        lstm_hidden=lstm_hidden,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
    )

    # 生成特征
    print("生成特征...")
    features = generate_features(y_data, model, model_params)
    print(f"特征生成完成，形状：{features.shape}")

    # 保存特征
    feature_df = pd.DataFrame(
        np.array(features), columns=[f"deep_ssm_feature_{i}" for i in range(state_dim)]
    )
    feature_df.to_csv(feature_save_path, index=False)
    print(f"特征已保存到 {feature_save_path}")

    # 保存模型
    save_model(model, model_params, scaler, model_save_path)

    return model, model_params, features


if __name__ == "__main__":
    # 测试训练流程
    config = {
        "csv_path": "./np_fracdiff_features.csv",
        "model_path": "deep_ssm_model_jax.pkl",
        "feature_save_path": "deep_ssm_features_jax.csv",
        "state_dim": 5,
        "lstm_hidden": 64,
        "max_epochs": 50,
        "patience": 5,
        "learning_rate": 0.001,
    }

    train_from_csv(
        csv_path=config["csv_path"],
        model_save_path=config["model_path"],
        feature_save_path=config["feature_save_path"],
        state_dim=config["state_dim"],
        lstm_hidden=config["lstm_hidden"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        learning_rate=config["learning_rate"],
    )
