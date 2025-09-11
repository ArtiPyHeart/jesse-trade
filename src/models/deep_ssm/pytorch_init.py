"""
PyTorch风格的初始化器，用于JAX/Flax模型
确保与PyTorch的默认初始化行为一致
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple, Optional
import numpy as np


def pytorch_uniform_init(scale: float = None, fan_in: int = None, fan_out: int = None):
    """
    创建PyTorch风格的uniform初始化器
    
    PyTorch的默认初始化：
    - Linear层: uniform(-sqrt(1/fan_in), sqrt(1/fan_in))
    - LSTM: uniform(-sqrt(1/hidden_size), sqrt(1/hidden_size))
    
    Args:
        scale: 直接指定范围，如果为None则根据fan_in/fan_out计算
        fan_in: 输入维度
        fan_out: 输出维度（用于LSTM的hidden_size）
    """
    def init(key, shape, dtype=jnp.float32):
        if scale is not None:
            bound = scale
        elif fan_out is not None:
            # LSTM使用hidden_size (fan_out)
            bound = 1.0 / np.sqrt(fan_out)
        elif fan_in is not None:
            # Linear层使用fan_in
            bound = 1.0 / np.sqrt(fan_in)
        else:
            # 从shape推断
            if len(shape) >= 2:
                bound = 1.0 / np.sqrt(shape[0])
            else:
                bound = 1.0 / np.sqrt(shape[-1])
        
        return random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    
    return init


def pytorch_zeros_init():
    """PyTorch风格的零初始化（用于偏置）"""
    return nn.initializers.zeros


def pytorch_lstm_init(hidden_size: int):
    """
    为LSTM创建PyTorch风格的初始化器
    
    PyTorch LSTM使用 uniform(-sqrt(1/hidden_size), sqrt(1/hidden_size))
    """
    bound = 1.0 / np.sqrt(hidden_size)
    return pytorch_uniform_init(scale=bound)


def pytorch_linear_init(fan_in: int):
    """
    为Linear层创建PyTorch风格的初始化器
    
    PyTorch Linear使用 uniform(-sqrt(1/fan_in), sqrt(1/fan_in))
    """
    bound = 1.0 / np.sqrt(fan_in)
    return pytorch_uniform_init(scale=bound)


class PyTorchLSTMCell(nn.Module):
    """
    使用PyTorch风格初始化的LSTM Cell
    """
    features: int
    
    def setup(self):
        # 使用PyTorch风格的初始化
        lstm_init = pytorch_lstm_init(self.features)
        bias_init = pytorch_zeros_init()
        
        # Input weights (对应PyTorch的weight_ih)
        self.ii = nn.Dense(self.features, use_bias=False, 
                          kernel_init=lstm_init, name='ii')
        self.if_ = nn.Dense(self.features, use_bias=False,
                           kernel_init=lstm_init, name='if')
        self.ig = nn.Dense(self.features, use_bias=False,
                          kernel_init=lstm_init, name='ig')
        self.io = nn.Dense(self.features, use_bias=False,
                          kernel_init=lstm_init, name='io')
        
        # Hidden weights (对应PyTorch的weight_hh)
        self.hi = nn.Dense(self.features, use_bias=True,
                          kernel_init=lstm_init, bias_init=bias_init, name='hi')
        self.hf = nn.Dense(self.features, use_bias=True,
                          kernel_init=lstm_init, bias_init=bias_init, name='hf')
        self.hg = nn.Dense(self.features, use_bias=True,
                          kernel_init=lstm_init, bias_init=bias_init, name='hg')
        self.ho = nn.Dense(self.features, use_bias=True,
                          kernel_init=lstm_init, bias_init=bias_init, name='ho')
        
        # Input biases (对应PyTorch的bias_ih)
        # 注意：PyTorch有两组bias，我们这里用参数来模拟
        self.bi = self.param('bi', bias_init, (self.features,))
        self.bf = self.param('bf', bias_init, (self.features,))
        self.bg = self.param('bg', bias_init, (self.features,))
        self.bo = self.param('bo', bias_init, (self.features,))
    
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], 
                 x: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        LSTM Cell前向传播
        
        Args:
            carry: (c, h) 前一时刻的细胞状态和隐藏状态
            x: 当前输入
            
        Returns:
            new_carry: (new_c, new_h)
            new_h: 输出（与new_h相同）
        """
        c, h = carry
        
        # 计算门（使用PyTorch的公式）
        # i = sigmoid(W_ii @ x + b_ii + W_hi @ h + b_hi)
        i = nn.sigmoid(self.ii(x) + self.bi + self.hi(h))
        f = nn.sigmoid(self.if_(x) + self.bf + self.hf(h))
        g = nn.tanh(self.ig(x) + self.bg + self.hg(h))
        o = nn.sigmoid(self.io(x) + self.bo + self.ho(h))
        
        # 更新状态
        new_c = f * c + i * g
        new_h = o * nn.tanh(new_c)
        
        return (new_c, new_h), new_h
    
    def initialize_carry(self, rng: random.PRNGKey, 
                        input_shape: Tuple[int, ...]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """初始化carry (c, h)"""
        batch_dims = input_shape[:-1]
        c = jnp.zeros(batch_dims + (self.features,))
        h = jnp.zeros(batch_dims + (self.features,))
        return (c, h)


def create_pytorch_style_sequential(layers_specs):
    """
    创建PyTorch风格初始化的Sequential
    
    Args:
        layers_specs: 层规格列表，例如 [(128, 'tanh'), (10, None)]
    """
    layers = []
    prev_size = None
    
    for i, spec in enumerate(layers_specs):
        if isinstance(spec, tuple):
            size, activation = spec
            if prev_size is not None:
                # 使用前一层的输出作为fan_in
                init = pytorch_linear_init(prev_size)
            else:
                # 第一层，将在运行时推断
                init = None
            
            # 添加Dense层
            if init is not None:
                layers.append(nn.Dense(size, kernel_init=init, 
                                      bias_init=pytorch_zeros_init()))
            else:
                layers.append(nn.Dense(size, bias_init=pytorch_zeros_init()))
            
            # 添加激活函数
            if activation == 'tanh':
                layers.append(nn.activation.tanh)
            elif activation == 'relu':
                layers.append(nn.relu)
            elif activation == 'sigmoid':
                layers.append(nn.sigmoid)
            # None表示没有激活函数
            
            prev_size = size
        else:
            # 直接是激活函数
            layers.append(spec)
    
    return nn.Sequential(layers)


def sync_pytorch_weights_to_jax(torch_lstm, hidden_size):
    """
    将PyTorch LSTM权重转换为JAX格式
    
    Args:
        torch_lstm: PyTorch的LSTM模块
        hidden_size: 隐藏层大小
        
    Returns:
        适用于PyTorchLSTMCell的参数字典
    """
    import torch
    
    # 提取PyTorch权重
    weight_ih = torch_lstm.weight_ih_l0.detach().numpy()
    weight_hh = torch_lstm.weight_hh_l0.detach().numpy()
    bias_ih = torch_lstm.bias_ih_l0.detach().numpy()
    bias_hh = torch_lstm.bias_hh_l0.detach().numpy()
    
    # 分割权重（i, f, g, o顺序）
    W_ii, W_if, W_ig, W_io = np.split(weight_ih, 4, axis=0)
    W_hi, W_hf, W_hg, W_ho = np.split(weight_hh, 4, axis=0)
    b_ii, b_if, b_ig, b_io = np.split(bias_ih, 4)
    b_hi, b_hf, b_hg, b_ho = np.split(bias_hh, 4)
    
    # 构建JAX参数字典
    params = {
        'ii': {'kernel': W_ii.T},
        'if': {'kernel': W_if.T},
        'ig': {'kernel': W_ig.T},
        'io': {'kernel': W_io.T},
        'hi': {'kernel': W_hi.T, 'bias': b_hi},
        'hf': {'kernel': W_hf.T, 'bias': b_hf},
        'hg': {'kernel': W_hg.T, 'bias': b_hg},
        'ho': {'kernel': W_ho.T, 'bias': b_ho},
        'bi': b_ii,
        'bf': b_if,
        'bg': b_ig,
        'bo': b_io,
    }
    
    return params