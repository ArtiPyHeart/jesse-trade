"""
权重同步工具：在PyTorch和JAX DeepSSM之间同步权重
"""

import numpy as np
import torch
import jax.numpy as jnp


def sync_pytorch_to_jax_deepsm(torch_model, jax_params):
    """
    将PyTorch DeepSSM的权重完全同步到JAX版本
    
    Args:
        torch_model: PyTorch的DeepSSM模型
        jax_params: JAX模型的参数字典
        
    Returns:
        更新后的JAX参数字典
    """
    import copy
    updated_params = copy.deepcopy(jax_params)
    
    # 1. 同步LSTM权重
    if hasattr(torch_model, 'lstm'):
        hidden_size = torch_model.lstm.hidden_size
        
        # 提取PyTorch LSTM权重
        weight_ih = torch_model.lstm.weight_ih_l0.detach().numpy()
        weight_hh = torch_model.lstm.weight_hh_l0.detach().numpy()
        bias_ih = torch_model.lstm.bias_ih_l0.detach().numpy()
        bias_hh = torch_model.lstm.bias_hh_l0.detach().numpy()
        
        # 分割权重到各个门 (i, f, g, o)
        W_ii, W_if, W_ig, W_io = np.split(weight_ih, 4, axis=0)
        W_hi, W_hf, W_hg, W_ho = np.split(weight_hh, 4, axis=0)
        b_ii, b_if, b_ig, b_io = np.split(bias_ih, 4)
        b_hi, b_hf, b_hg, b_ho = np.split(bias_hh, 4)
        
        # 更新JAX LSTM参数
        if 'lstm_cell' in updated_params['params']:
            lstm_params = updated_params['params']['lstm_cell']
            
            # 更新输入权重
            lstm_params['ii'] = {'kernel': W_ii.T}
            lstm_params['if'] = {'kernel': W_if.T}
            lstm_params['ig'] = {'kernel': W_ig.T}
            lstm_params['io'] = {'kernel': W_io.T}
            
            # 更新隐藏权重和偏置
            lstm_params['hi'] = {'kernel': W_hi.T, 'bias': b_hi}
            lstm_params['hf'] = {'kernel': W_hf.T, 'bias': b_hf}
            lstm_params['hg'] = {'kernel': W_hg.T, 'bias': b_hg}
            lstm_params['ho'] = {'kernel': W_ho.T, 'bias': b_ho}
            
            # 更新输入偏置
            lstm_params['bi'] = b_ii
            lstm_params['bf'] = b_if
            lstm_params['bg'] = b_ig
            lstm_params['bo'] = b_io
    
    # 2. 同步transition网络权重
    if hasattr(torch_model, 'transition'):
        transition_params = {}
        
        # 第一层
        transition_params['layers_0'] = {
            'kernel': torch_model.transition[0].weight.detach().numpy().T,
            'bias': torch_model.transition[0].bias.detach().numpy()
        }
        
        # 第二层（跳过激活函数）
        transition_params['layers_2'] = {
            'kernel': torch_model.transition[2].weight.detach().numpy().T,
            'bias': torch_model.transition[2].bias.detach().numpy()
        }
        
        updated_params['params']['transition'] = transition_params
    
    # 3. 同步observation网络权重
    if hasattr(torch_model, 'observation'):
        observation_params = {}
        
        # 第一层
        observation_params['layers_0'] = {
            'kernel': torch_model.observation[0].weight.detach().numpy().T,
            'bias': torch_model.observation[0].bias.detach().numpy()
        }
        
        # 第二层
        observation_params['layers_2'] = {
            'kernel': torch_model.observation[2].weight.detach().numpy().T,
            'bias': torch_model.observation[2].bias.detach().numpy()
        }
        
        updated_params['params']['observation'] = observation_params
    
    # 4. 同步初始状态参数
    if hasattr(torch_model, 'initial_state_mean'):
        updated_params['params']['initial_state_mean'] = \
            torch_model.initial_state_mean.detach().numpy()
        updated_params['params']['initial_state_log_var'] = \
            torch_model.initial_state_log_var.detach().numpy()
    
    return updated_params


def compare_model_outputs(torch_model, jax_model, jax_params, test_input):
    """
    比较PyTorch和JAX模型在相同输入下的输出
    
    Args:
        torch_model: PyTorch模型
        jax_model: JAX模型
        jax_params: JAX参数
        test_input: 测试输入 [batch, seq, features]
        
    Returns:
        差异统计字典
    """
    import torch
    
    # PyTorch前向
    torch_model.eval()
    with torch.no_grad():
        torch_input = torch.tensor(test_input)
        torch_out, (h_torch, c_torch) = torch_model.lstm(torch_input)
    
    # JAX前向
    jax_input = jnp.array(test_input)
    jax_out = jax_model.apply(jax_params, jax_input)
    
    # 计算差异
    diff = np.abs(torch_out.numpy() - np.array(jax_out))
    
    stats = {
        'max_diff': diff.max(),
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        'torch_shape': torch_out.shape,
        'jax_shape': jax_out.shape,
        'torch_range': (torch_out.min().item(), torch_out.max().item()),
        'jax_range': (float(jax_out.min()), float(jax_out.max()))
    }
    
    return stats


def create_matched_models(obs_dim, state_dim, lstm_hidden, seed=42):
    """
    创建具有相同初始权重的PyTorch和JAX模型
    
    Args:
        obs_dim: 观测维度
        state_dim: 状态维度
        lstm_hidden: LSTM隐藏维度
        seed: 随机种子
        
    Returns:
        torch_model, jax_model, jax_params
    """
    import torch
    from jax import random
    from src.models.deep_ssm.model import create_model, init_model_params
    
    # 先导入PyTorch模型定义
    import torch.nn as nn
    
    class TorchDeepSSM(nn.Module):
        def __init__(self, obs_dim, state_dim=5, lstm_hidden=64):
            super().__init__()
            self.obs_dim = obs_dim
            self.state_dim = state_dim
            self.lstm = nn.LSTM(
                input_size=obs_dim,
                hidden_size=lstm_hidden,
                batch_first=True,
                num_layers=1,
            )
            self.transition = nn.Sequential(
                nn.Linear(lstm_hidden + state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 2 * state_dim),
            )
            self.observation = nn.Sequential(
                nn.Linear(state_dim, 128), 
                nn.Tanh(), 
                nn.Linear(128, 2 * obs_dim)
            )
            self.initial_state_mean = nn.Parameter(torch.zeros(state_dim))
            self.initial_state_log_var = nn.Parameter(torch.zeros(state_dim))
    
    # 创建PyTorch模型
    torch.manual_seed(seed)
    torch_model = TorchDeepSSM(obs_dim, state_dim, lstm_hidden)
    
    # 创建JAX模型
    key = random.PRNGKey(seed)
    jax_model = create_model(obs_dim, state_dim, lstm_hidden)
    dummy_input = jnp.zeros((1, 10, obs_dim))
    jax_params = init_model_params(jax_model, key, dummy_input)
    
    # 同步权重
    jax_params = sync_pytorch_to_jax_deepsm(torch_model, jax_params)
    
    return torch_model, jax_model, jax_params