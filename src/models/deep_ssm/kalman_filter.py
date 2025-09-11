"""
Extended Kalman Filter for DeepSSM
扩展卡尔曼滤波器的JAX实现
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit

from .model import DeepSSM


def ekf_predict(
    z: jnp.ndarray,
    P: jnp.ndarray,
    lstm_out: jnp.ndarray,
    model: DeepSSM,
    model_params: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    扩展卡尔曼滤波预测步

    Args:
        z: 当前状态估计 [1, state_dim]
        P: 当前协方差矩阵 [state_dim, state_dim]
        lstm_out: LSTM输出特征 [1, lstm_hidden]
        model: DeepSSM模型
        model_params: 模型参数

    Returns:
        (预测状态, 预测协方差)
    """
    # 计算状态转移
    transition_mean, transition_log_var = model.apply(
        model_params, lstm_out, z, method=model.get_transition_dist
    )

    # 预测状态
    z_pred = transition_mean

    # 过程噪声协方差（与原始PyTorch版本对齐：直接使用预测方差）
    Q = jnp.diag(jnp.exp(transition_log_var.squeeze(0)))

    # 预测协方差：与原版保持一致（不累加上一时刻P）
    P_pred = Q

    return z_pred, P_pred


def ekf_update(
    z_pred: jnp.ndarray,
    P_pred: jnp.ndarray,
    y_obs: jnp.ndarray,
    model: DeepSSM,
    model_params: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    扩展卡尔曼滤波更新步

    Args:
        z_pred: 预测状态 [1, state_dim]
        P_pred: 预测协方差 [state_dim, state_dim]
        y_obs: 观测值 [1, obs_dim]
        model: DeepSSM模型
        model_params: 模型参数

    Returns:
        (更新状态, 更新协方差)
    """
    state_dim = z_pred.shape[-1]

    # 计算观测预测
    obs_mean, obs_log_var = model.apply(
        model_params, z_pred, method=model.get_observation_dist
    )

    # 观测函数的雅可比矩阵：确保函数输入是一维(state_dim,)
    def obs_func(z_vec):
        z_in = z_vec.reshape(1, -1)
        mean, _ = model.apply(model_params, z_in, method=model.get_observation_dist)
        return mean.squeeze(0)

    H = jax.jacfwd(obs_func)(z_pred.squeeze(0))
    if H.ndim == 1:
        H = H.reshape(1, -1)

    # 观测噪声协方差
    obs_var = jnp.exp(obs_log_var.squeeze(0))
    R = jnp.diag(obs_var)

    # 卡尔曼增益
    S = H @ P_pred @ H.T + R
    # 添加小的正则化项确保数值稳定
    S = S + jnp.eye(S.shape[0]) * 1e-6
    K = P_pred @ H.T @ jnp.linalg.inv(S)

    # 状态更新
    innovation = (y_obs - obs_mean).T
    z_new = z_pred.T + K @ innovation
    z_new = z_new.T

    # 协方差更新（简单形式；如需更稳定可用Joseph形式）
    P_new = (jnp.eye(state_dim) - K @ H) @ P_pred

    return z_new, P_new


def deep_ssm_kalman_filter(
    y_seq: jnp.ndarray, model: DeepSSM, model_params: dict
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    对整个序列运行扩展卡尔曼滤波

    Args:
        y_seq: 观测序列 [T, obs_dim]
        model: DeepSSM模型
        model_params: 模型参数

    Returns:
        (状态序列 [T, state_dim], 最终协方差矩阵)
    """
    T = len(y_seq)
    state_dim = model.state_dim
    obs_dim = model.obs_dim

    # 扩展输入维度以适配模型
    y_batch = y_seq.reshape(1, T, obs_dim)

    # 获取LSTM特征（整段前向，与逐步一致，因为carry从零开始）
    lstm_features = model.apply(model_params, y_batch)  # [1, T, lstm_hidden]
    lstm_features = lstm_features.squeeze(0)  # [T, lstm_hidden]

    # 初始化状态和协方差
    z = model_params["params"]["initial_state_mean"].reshape(1, -1)
    P = jnp.diag(jnp.exp(model_params["params"]["initial_state_log_var"]))

    states = [z.squeeze(0)]

    # 逐时间步滤波
    for t in range(1, T):
        # 预测步
        lstm_out_t = lstm_features[t : t + 1]
        z_pred, P_pred = ekf_predict(z, P, lstm_out_t, model, model_params)

        # 更新步
        y_t = y_seq[t : t + 1]
        z, P = ekf_update(z_pred, P_pred, y_t, model, model_params)

        states.append(z.squeeze(0))

    return jnp.stack(states), P


def kalman_filter_step(
    z: jnp.ndarray,
    P: jnp.ndarray,
    lstm_hidden: jnp.ndarray,
    lstm_cell_state: jnp.ndarray,
    y_new: jnp.ndarray,
    model: DeepSSM,
    model_params: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    单步卡尔曼滤波（用于实时推理）

    Args:
        z: 当前状态 [1, state_dim]
        P: 当前协方差 [state_dim, state_dim]
        lstm_hidden: LSTM隐藏状态
        lstm_cell_state: LSTM细胞状态
        y_new: 新观测 [obs_dim]
        model: DeepSSM模型
        model_params: 模型参数
        lstm_cell: LSTM单元

    Returns:
        (新状态, 新协方差, 新LSTM隐藏状态, 新LSTM细胞状态)
    """
    # LSTM处理新观测（复用模型内部LSTM权重）
    carry = (lstm_hidden, lstm_cell_state)
    carry, lstm_out = model.apply(
        model_params, carry, y_new.reshape(1, -1), method=model.lstm_step
    )
    lstm_hidden_new, lstm_cell_new = carry

    # 扩展卡尔曼滤波
    z_pred, P_pred = ekf_predict(z, P, lstm_out.reshape(1, -1), model, model_params)
    z_new, P_new = ekf_update(z_pred, P_pred, y_new.reshape(1, -1), model, model_params)

    return z_new, P_new, lstm_hidden_new, lstm_cell_new


@jit
def batch_kalman_filter(
    y_batch: jnp.ndarray, model: DeepSSM, model_params: dict
) -> jnp.ndarray:
    """
    批量卡尔曼滤波（使用vmap优化）

    Args:
        y_batch: 批量观测 [batch_size, T, obs_dim]
        model: DeepSSM模型
        model_params: 模型参数

    Returns:
        批量状态序列 [batch_size, T, state_dim]
    """

    def single_sequence_filter(y_seq):
        states, _ = deep_ssm_kalman_filter(y_seq, model, model_params)
        return states

    # 使用vmap并行处理批次
    return jax.vmap(single_sequence_filter)(y_batch)
