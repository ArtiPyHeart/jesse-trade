"""
Real-time inference for DeepSSM
DeepSSM的实时推理模块
"""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from .kalman_filter import kalman_filter_step
from .model import DeepSSM, create_model
from .training import load_model, load_model_npz


class DeepSSMRealTime:
    """
    DeepSSM实时推理类
    用于逐步处理新数据并生成特征
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[DeepSSM] = None,
        model_params: Optional[dict] = None,
    ):
        """
        初始化实时推理器
        
        Args:
            model_path: 模型文件路径（支持.json和.npz格式）
            model: 预加载的模型（可选）
            model_params: 预加载的模型参数（可选）
        """
        if model_path is not None:
            # 从文件加载模型
            path = Path(model_path)
            if path.suffix == '.json':
                self.model, self.model_params, self.metadata = load_model(model_path)
            elif path.suffix == '.npz':
                self.model, self.model_params = load_model_npz(model_path)
                self.metadata = {}
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
        else:
            # 使用提供的模型
            assert (
                model is not None and model_params is not None
            ), "必须提供model_path或(model, model_params)"
            self.model = model
            self.model_params = model_params
            self.metadata = {}
        
        # 获取模型配置
        self.state_dim = self.model.state_dim
        self.obs_dim = self.model.obs_dim
        self.lstm_hidden_size = self.model.lstm_hidden
        
        # 初始化状态变量
        self._reset_state()
        
        # JIT编译处理函数以提高性能
        self._process_jit = jit(self._process_step)
    
    def _reset_state(self):
        """重置内部状态"""
        # 状态估计
        self.z = self.model_params["params"]["initial_state_mean"].reshape(1, -1)
        
        # 协方差矩阵
        self.P = jnp.diag(jnp.exp(self.model_params["params"]["initial_state_log_var"]))
        
        # LSTM状态（c, h）
        c = jnp.zeros((1, self.lstm_hidden_size))
        h = jnp.zeros((1, self.lstm_hidden_size))
        self.lstm_carry = (c, h)
    
    def _process_step(
        self,
        z: jnp.ndarray,
        P: jnp.ndarray,
        lstm_carry: Tuple[jnp.ndarray, jnp.ndarray],
        new_data: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        单步处理（内部JIT编译函数）
        
        Args:
            z: 当前状态
            P: 当前协方差
            lstm_carry: LSTM状态 (c, h)
            new_data: 新数据
        
        Returns:
            (新状态, 新协方差, 新LSTM状态)
        """
        # 注意：kalman_filter_step需要的是(h, c)而不是(c, h)
        lstm_hidden = lstm_carry[1]
        lstm_cell_state = lstm_carry[0]
        
        # 执行卡尔曼滤波步
        z_new, P_new, lstm_hidden_new, lstm_cell_new = kalman_filter_step(
            z=z,
            P=P,
            lstm_hidden=lstm_hidden,
            lstm_cell_state=lstm_cell_state,
            y_new=new_data,
            model=self.model,
            model_params=self.model_params,
        )
        
        return z_new, P_new, (lstm_cell_new, lstm_hidden_new)
    
    def process(self, new_data: np.ndarray) -> np.ndarray:
        """
        处理新数据并返回特征
        
        Args:
            new_data: 新的观测数据 [obs_dim]
        
        Returns:
            生成的特征 [state_dim]
        """
        # 确保输入维度正确
        assert (
            new_data.shape[0] == self.obs_dim
        ), f"输入维度不匹配：期望{self.obs_dim}，得到{new_data.shape[0]}"
        
        # 转换为JAX数组
        new_data_jax = jnp.array(new_data, dtype=jnp.float32)
        
        # 执行处理步骤
        self.z, self.P, self.lstm_carry = self._process_jit(
            self.z, self.P, self.lstm_carry, new_data_jax
        )
        
        # 返回状态估计作为特征
        return np.array(self.z.flatten())
    
    def process_batch(self, data_batch: np.ndarray) -> np.ndarray:
        """
        批量处理数据
        
        Args:
            data_batch: 批量数据 [batch_size, obs_dim]
        
        Returns:
            批量特征 [batch_size, state_dim]
        """
        features = []
        for data in data_batch:
            feature = self.process(data)
            features.append(feature)
        return np.array(features)
    
    def reset(self):
        """重置推理器状态"""
        self._reset_state()
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        获取当前内部状态
        
        Returns:
            包含当前状态信息的字典
        """
        return {
            "z": np.array(self.z),
            "P": np.array(self.P),
            "lstm_cell": np.array(self.lstm_carry[0]),
            "lstm_hidden": np.array(self.lstm_carry[1]),
        }
    
    def set_state(self, state_dict: Dict[str, np.ndarray]):
        """
        设置内部状态
        
        Args:
            state_dict: 状态字典
        """
        self.z = jnp.array(state_dict["z"])
        self.P = jnp.array(state_dict["P"])
        lstm_cell = jnp.array(state_dict["lstm_cell"])
        lstm_hidden = jnp.array(state_dict["lstm_hidden"])
        self.lstm_carry = (lstm_cell, lstm_hidden)
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取模型元数据"""
        return self.metadata


def create_realtime_processor(
    obs_dim: int,
    state_dim: int = 5,
    lstm_hidden: int = 64,
    model_params: Optional[dict] = None,
) -> DeepSSMRealTime:
    """
    创建实时处理器（不从文件加载）
    
    Args:
        obs_dim: 观测维度
        state_dim: 状态维度
        lstm_hidden: LSTM隐藏层维度
        model_params: 模型参数（可选）
    
    Returns:
        实时处理器实例
    """
    model = create_model(obs_dim, state_dim, lstm_hidden)
    
    if model_params is None:
        # 使用随机初始化
        key = jax.random.PRNGKey(0)
        sample_input = jnp.zeros((1, 10, obs_dim))
        model_params = model.init(key, sample_input)
    
    return DeepSSMRealTime(model=model, model_params=model_params)


def demo_realtime_inference():
    """
    演示实时推理功能
    """
    print("DeepSSM实时推理演示")
    print("-" * 50)
    
    # 创建模拟的实时处理器
    obs_dim = 77  # 与原始特征维度一致
    state_dim = 5
    
    processor = create_realtime_processor(obs_dim=obs_dim, state_dim=state_dim)
    
    print(f"实时处理器初始化完成")
    print(f"输入维度: {obs_dim}")
    print(f"输出特征维度: {state_dim}")
    print()
    
    # 模拟实时数据流
    print("模拟实时数据处理:")
    for i in range(5):
        # 生成随机数据（实际使用时应该是标准化后的数据）
        new_data = np.random.randn(obs_dim) * 0.1
        
        # 处理数据
        feature = processor.process(new_data)
        
        print(f"第{i+1}条数据 -> 特征: {feature[:3].round(4)}...")
    
    print()
    print("获取当前状态:")
    state = processor.get_state()
    print(f"状态估计形状: {state['z'].shape}")
    print(f"协方差矩阵形状: {state['P'].shape}")
    
    print()
    print("重置处理器...")
    processor.reset()
    print("处理器已重置")
    
    # 演示从文件加载
    print("\n" + "-" * 50)
    print("从文件加载模型演示:")
    print("支持格式: .json (人类可读) 或 .npz (二进制)")
    print("示例: processor = DeepSSMRealTime('model.json')")


if __name__ == "__main__":
    # 运行演示
    demo_realtime_inference()