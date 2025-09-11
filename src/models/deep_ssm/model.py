"""
DeepSSM Model implemented in JAX/Flax
深度状态空间模型的JAX实现
"""

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flax.linen import LSTMCell
from jax import random


class DeepSSM(nn.Module):
    """
    深度状态空间模型 - 结合LSTM和状态空间模型

    Attributes:
        obs_dim: 观测数据维度
        state_dim: 潜在状态维度
        lstm_hidden: LSTM隐藏层维度
    """

    obs_dim: int
    state_dim: int = 5
    lstm_hidden: int = 64

    def setup(self):
        """初始化网络层"""
        # LSTM层用于提取时序特征
        self.lstm_cell = LSTMCell(features=self.lstm_hidden)

        # 状态转移网络
        self.transition = nn.Sequential(
            [nn.Dense(128), nn.tanh, nn.Dense(2 * self.state_dim)]  # 输出均值和对数方差
        )

        # 观测网络
        self.observation = nn.Sequential(
            [nn.Dense(128), nn.tanh, nn.Dense(2 * self.obs_dim)]  # 输出均值和对数方差
        )

        # 初始状态参数
        self.initial_state_mean = self.param(
            "initial_state_mean", nn.initializers.zeros, (self.state_dim,)
        )
        self.initial_state_log_var = self.param(
            "initial_state_log_var", nn.initializers.zeros, (self.state_dim,)
        )

    def lstm_init_carry(self, sample_input: jnp.ndarray):
        """根据输入样本初始化LSTM状态(carry)。

        兼容不同Flax版本的initialize_carry签名：
        - 新API: initialize_carry(key, sample_input)
        - 旧API: initialize_carry(key, batch_dims, size)
        """
        try:
            # Newer API: pass a sample input (without time dimension)
            return self.lstm_cell.initialize_carry(random.PRNGKey(0), sample_input)
        except TypeError:
            # Fallback to legacy API: batch_dims=(), size=features
            return self.lstm_cell.initialize_carry(
                random.PRNGKey(0), (), self.lstm_hidden
            )

    def get_transition_dist(
        self, lstm_out: jnp.ndarray, z_prev: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        计算状态转移分布的参数

        Args:
            lstm_out: LSTM输出特征
            z_prev: 前一时刻的状态

        Returns:
            (均值, 对数方差)
        """
        # 拼接LSTM输出和前一状态
        input_feat = jnp.concatenate([lstm_out, z_prev], axis=-1)
        out = self.transition(input_feat)

        # 分割为均值和对数方差
        mean, log_var = jnp.split(out, 2, axis=-1)
        # 限制对数方差范围以保证数值稳定性
        log_var = jnp.clip(log_var, -10, 10)

        return mean, log_var

    def get_observation_dist(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        计算观测分布的参数

        Args:
            z: 当前状态

        Returns:
            (均值, 对数方差)
        """
        out = self.observation(z)

        # 分割为均值和对数方差
        mean, log_var = jnp.split(out, 2, axis=-1)
        # 限制对数方差范围
        log_var = jnp.clip(log_var, -10, 10)

        return mean, log_var

    def lstm_sequence(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        处理整个序列通过LSTM

        Args:
            y: 观测序列 [T, obs_dim]

        Returns:
            LSTM输出序列 [T, lstm_hidden]
        """
        T = y.shape[0]

        # 初始化LSTM状态（使用首个时间步的形状做参考）
        sample_input = y[0]
        carry = self.lstm_init_carry(sample_input)

        def step(c, x):
            c_new, h = self.lstm_cell(c, x)
            return c_new, h

        # 使用lax.scan以获得更好的性能和jit兼容性
        carry, outputs = jax.lax.scan(step, carry, y)
        return outputs

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        前向传播（用于特征提取）

        Args:
            y: 观测序列 [batch_size, T, obs_dim]

        Returns:
            LSTM特征 [batch_size, T, lstm_hidden]
        """
        # 批处理LSTM
        batch_size, T, _ = y.shape

        # 使用vmap处理批次
        lstm_batch = jax.vmap(self.lstm_sequence)
        return lstm_batch(y)

    def lstm_step(self, carry, x: jnp.ndarray):
        """单步LSTM前向（用于实时推理，复用模型内参数）。"""
        return self.lstm_cell(carry, x)


def deep_ssm_model(y: jnp.ndarray, model: DeepSSM, model_params: dict):
    """
    DeepSSM的概率生成模型（用于NumPyro）

    Args:
        y: 观测数据 [batch_size, T, obs_dim]
        model: DeepSSM模型实例
        model_params: 模型参数
    """
    batch_size, T, obs_dim = y.shape
    state_dim = model.state_dim

    # 使用模型参数处理LSTM
    lstm_out = model.apply(model_params, y)  # [batch_size, T, lstm_hidden]

    # 使用plate明确批维度
    with numpyro.plate("batch", batch_size):
        # 初始状态先验
        z0_mean = model_params["params"]["initial_state_mean"]
        z0_log_var = model_params["params"]["initial_state_log_var"]
        z = numpyro.sample(
            "z0", dist.Normal(z0_mean, jnp.exp(0.5 * z0_log_var)).to_event(1)
        )

        # 按时间步（从t=1起）
        for t in range(1, T):
            transition_mean, transition_log_var = model.apply(
                model_params,
                lstm_out[:, t, :],
                z,
                method=model.get_transition_dist,
            )
            z = numpyro.sample(
                f"z{t}",
                dist.Normal(
                    transition_mean, jnp.exp(0.5 * transition_log_var)
                ).to_event(1),
            )

            obs_mean, obs_log_var = model.apply(
                model_params, z, method=model.get_observation_dist
            )
            numpyro.sample(
                f"y{t}",
                dist.Normal(obs_mean, jnp.exp(0.5 * obs_log_var)).to_event(1),
                obs=y[:, t, :],
            )


def deep_ssm_guide(y: jnp.ndarray, model: DeepSSM, model_params: dict):
    """
    DeepSSM的变分指导函数（近似后验）

    Args:
        y: 观测数据 [batch_size, T, obs_dim]
        model: DeepSSM模型实例
        model_params: 模型参数
    """
    batch_size, T, _ = y.shape
    state_dim = model.state_dim

    # 定义变分参数
    z_loc = numpyro.param("z_loc", jnp.zeros((batch_size, T, state_dim)))
    z_scale = numpyro.param(
        "z_scale",
        jnp.ones((batch_size, T, state_dim)) * 0.1,
        constraint=dist.constraints.positive,
    )

    # 按时间步定义变分分布
    with numpyro.plate("batch", batch_size):
        for t in range(T):
            numpyro.sample(
                f"z{t}", dist.Normal(z_loc[:, t, :], z_scale[:, t, :]).to_event(1)
            )


def compute_jacobian_numerical(f, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    数值法计算雅可比矩阵（用于扩展卡尔曼滤波）

    Args:
        f: 函数
        x: 输入点
        eps: 数值微分步长

    Returns:
        雅可比矩阵
    """
    # 使用JAX的自动微分更高效
    return jax.jacfwd(f)(x)


def create_model(obs_dim: int, state_dim: int = 5, lstm_hidden: int = 64) -> DeepSSM:
    """
    创建DeepSSM模型实例

    Args:
        obs_dim: 观测维度
        state_dim: 状态维度
        lstm_hidden: LSTM隐藏层维度

    Returns:
        DeepSSM模型实例
    """
    return DeepSSM(obs_dim=obs_dim, state_dim=state_dim, lstm_hidden=lstm_hidden)


def init_model_params(
    model: DeepSSM, key: random.PRNGKey, sample_input: jnp.ndarray
) -> dict:
    """
    初始化模型参数

    Args:
        model: DeepSSM模型
        key: 随机密钥
        sample_input: 样本输入用于形状推断

    Returns:
        初始化的参数字典
    """
    return model.init(key, sample_input)
