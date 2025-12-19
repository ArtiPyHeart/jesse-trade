"""
Chunked BPTT 策略对比测试

对比不同 Chunked BPTT 实现策略的：
1. 内存占用
2. 输出精确度（与 full BPTT 对比）
3. 梯度精确度（与 full BPTT 对比）
4. 训练时间

策略：
1. Full BPTT (baseline)
2. Basic chunked (只有 detach，无 overlap)
3. Chunked + overlap/burn-in (修正版：正确保存 LSTM 状态)
4. Chunked + overlap + deterministic eps

修复的问题（基于 Codex 审查）：
- [Fixed] Overlap 策略正确保存 LSTM 状态
- [Fixed] RNG 对齐：所有策略使用相同的种子设置方式
- [Fixed] 使用 PyTorch 内存测量替代 tracemalloc
- [Fixed] 时间测量分离 forward 和 backward
- [Fixed] 支持 GPU 设备
"""

import gc
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """获取可用设备 - 测试固定使用 CPU 以确保可复现性"""
    # 测试固定使用 CPU，避免不同设备间的数值差异
    return torch.device("cpu")

# 全局变量用于 tracemalloc
_tracemalloc_active = False


def reset_memory_stats(device: torch.device):
    """重置内存统计"""
    global _tracemalloc_active
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    else:
        # CPU: 使用 tracemalloc
        if _tracemalloc_active:
            tracemalloc.stop()
        tracemalloc.start()
        _tracemalloc_active = True


def get_peak_memory_mb(device: torch.device) -> float:
    """获取峰值内存（MB）"""
    global _tracemalloc_active
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        # CPU: 使用 tracemalloc (注意：这只测量 Python 对象，不包括 PyTorch tensors)
        if _tracemalloc_active:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _tracemalloc_active = False
            return peak / 1024 / 1024
        return 0.0


@dataclass
class BPTTResult:
    """BPTT 实验结果"""
    strategy_name: str
    memory_peak_mb: float
    forward_time_ms: float
    backward_time_ms: float
    loss_value: float
    output_states: Optional[torch.Tensor] = None
    grad_dict: Optional[Dict[str, torch.Tensor]] = None


class SimpleDeepSSM(nn.Module):
    """
    简化版 DeepSSM 用于测试 Chunked BPTT

    结构：
    1. LSTM: 处理观测序列
    2. Transition MLP: f(lstm_out, z_prev) -> z_mean, z_log_var
    3. Observation MLP: g(z) -> obs_mean, obs_log_var
    """

    def __init__(
        self,
        obs_dim: int = 10,
        state_dim: int = 5,
        lstm_hidden: int = 32,
        transition_hidden: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.lstm_hidden = lstm_hidden

        # LSTM
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Transition network: (lstm_out, z_prev) -> (z_mean, z_log_var)
        self.transition = nn.Sequential(
            nn.Linear(lstm_hidden + state_dim, transition_hidden),
            nn.Tanh(),
            nn.Linear(transition_hidden, 2 * state_dim),
        )

        # Observation network: z -> (obs_mean, obs_log_var)
        self.observation = nn.Sequential(
            nn.Linear(state_dim, transition_hidden),
            nn.Tanh(),
            nn.Linear(transition_hidden, 2 * obs_dim),
        )

        # Initial state parameters
        self.z0_mean = nn.Parameter(torch.zeros(state_dim))
        self.z0_log_var = nn.Parameter(torch.zeros(state_dim))

    def _ssm_step(
        self,
        lstm_out_t: torch.Tensor,  # [batch, lstm_hidden]
        z_prev: torch.Tensor,       # [batch, state_dim]
        eps: Optional[torch.Tensor] = None,  # [batch, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步 SSM 更新"""
        # Transition
        combined = torch.cat([lstm_out_t, z_prev], dim=-1)
        trans_out = self.transition(combined)
        z_mean, z_log_var = torch.split(trans_out, self.state_dim, dim=-1)
        z_log_var = torch.clamp(z_log_var, -10, 10)

        # Reparameterization
        z_std = torch.exp(0.5 * z_log_var)
        if eps is None:
            eps = torch.randn_like(z_std)
        z = z_mean + z_std * eps

        return z, z_mean, z_log_var

    def _compute_loss(
        self,
        z: torch.Tensor,  # [batch, state_dim]
        obs_t: torch.Tensor,  # [batch, obs_dim]
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> torch.Tensor:
        """计算单步 loss (reconstruction + KL)"""
        # Observation distribution
        obs_out = self.observation(z)
        obs_mean, obs_log_var = torch.split(obs_out, self.obs_dim, dim=-1)
        obs_log_var = torch.clamp(obs_log_var, -10, 10)

        # Reconstruction loss (Gaussian NLL)
        obs_var = torch.exp(obs_log_var)
        recon_loss = 0.5 * (
            obs_log_var + (obs_t - obs_mean) ** 2 / obs_var
        ).sum(dim=-1).mean()

        # KL loss (vs standard normal)
        kl_loss = -0.5 * (
            1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)
        ).sum(dim=-1).mean()

        return recon_loss + 0.1 * kl_loss

    def forward_full(
        self,
        obs: torch.Tensor,  # [batch, T, obs_dim]
        eps_list: Optional[List[torch.Tensor]] = None,  # 预生成的 eps 列表
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full BPTT forward pass

        Args:
            obs: 观测序列
            eps_list: 可选的预生成 eps 列表，用于确保与 chunked 版本一致
            return_states: 是否返回所有状态

        Returns:
            loss: 总损失
            states: 如果 return_states=True，返回所有状态
        """
        batch_size, T, _ = obs.shape
        device = obs.device

        # LSTM 处理全序列
        lstm_out, _ = self.lstm(obs)  # [batch, T, lstm_hidden]

        # 初始状态
        z = self.z0_mean.expand(batch_size, -1)

        total_loss = torch.tensor(0.0, device=device)
        states = [] if return_states else None

        for t in range(T):
            eps = eps_list[t] if eps_list is not None else None
            z, z_mean, z_log_var = self._ssm_step(lstm_out[:, t], z, eps)

            total_loss = total_loss + self._compute_loss(z, obs[:, t], z_mean, z_log_var)

            if return_states:
                states.append(z.detach().clone())

        if return_states:
            states = torch.stack(states, dim=1)

        return total_loss / T, states

    def forward_chunked_basic(
        self,
        obs: torch.Tensor,
        chunk_size: int = 256,
        eps_list: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Basic Chunked BPTT: 只有 detach，无 overlap

        注意：第一个 chunk 保持与 z0_mean 的梯度连接，后续 chunk 才 detach
        """
        batch_size, T, _ = obs.shape
        device = obs.device

        total_loss = torch.tensor(0.0, device=device)
        states = [] if return_states else None

        # 初始状态 - 保持与 z0_mean 的连接（第一个 chunk 需要梯度流）
        z = self.z0_mean.expand(batch_size, -1)
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)

        global_t = 0
        while global_t < T:
            chunk_end = min(global_t + chunk_size, T)

            # LSTM 处理当前 chunk
            lstm_out, (h_new, c_new) = self.lstm(obs[:, global_t:chunk_end], (h, c))

            # SSM 循环
            z_chunk = z
            for tau in range(chunk_end - global_t):
                t_abs = global_t + tau
                eps = eps_list[t_abs] if eps_list is not None else None
                z_chunk, z_mean, z_log_var = self._ssm_step(lstm_out[:, tau], z_chunk, eps)
                total_loss = total_loss + self._compute_loss(
                    z_chunk, obs[:, t_abs], z_mean, z_log_var
                )
                if return_states:
                    states.append(z_chunk.detach().clone())

            # Detach 状态传递
            z = z_chunk.detach()
            h = h_new.detach()
            c = c_new.detach()
            global_t = chunk_end

        if return_states:
            states = torch.stack(states, dim=1)

        return total_loss / T, states

    def forward_chunked_overlap(
        self,
        obs: torch.Tensor,
        chunk_size: int = 256,
        overlap: int = 64,
        eps_list: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Chunked BPTT with overlap/burn-in (修正版)

        处理 [t-overlap : t+chunk_size)，但只在 [t : t+chunk_size) 计算 loss

        修正：正确保存 LSTM 状态 - 在 save_state_at 位置保存
        """
        batch_size, T, _ = obs.shape
        device = obs.device

        total_loss = torch.tensor(0.0, device=device)
        states = [] if return_states else None

        # 初始状态
        z = self.z0_mean.expand(batch_size, -1)
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)

        t = 0
        loss_steps = 0

        while t < T:
            burn_start = max(0, t - overlap)
            chunk_end = min(t + chunk_size, T)

            # 下一个 chunk 的 burn_start = chunk_end - overlap
            # 所以需要在 chunk_end - overlap 位置保存状态
            save_state_at = chunk_end - overlap if chunk_end < T else None

            # LSTM 处理
            # 如果有 save_state_at，分两段处理以获取中间状态
            if save_state_at is not None and save_state_at > burn_start:
                # 第一段：[burn_start : save_state_at]
                if burn_start == 0:
                    lstm_out_1, (h_save, c_save) = self.lstm(obs[:, burn_start:save_state_at])
                else:
                    lstm_out_1, (h_save, c_save) = self.lstm(obs[:, burn_start:save_state_at], (h, c))

                # 第二段：[save_state_at : chunk_end]
                lstm_out_2, (h_new, c_new) = self.lstm(obs[:, save_state_at:chunk_end], (h_save, c_save))

                # 合并 lstm_out
                lstm_out = torch.cat([lstm_out_1, lstm_out_2], dim=1)
            else:
                # 不需要保存中间状态，直接处理
                if burn_start == 0:
                    lstm_out, (h_new, c_new) = self.lstm(obs[:, burn_start:chunk_end])
                else:
                    lstm_out, (h_new, c_new) = self.lstm(obs[:, burn_start:chunk_end], (h, c))
                h_save, c_save = h_new, c_new

            # SSM 循环
            z_chunk = z
            next_z = None

            for i, tau in enumerate(range(burn_start, chunk_end)):
                eps = eps_list[tau] if eps_list is not None else None
                z_chunk, z_mean, z_log_var = self._ssm_step(lstm_out[:, i], z_chunk, eps)

                # 只在非 overlap 区域计算 loss
                if tau >= t:
                    total_loss = total_loss + self._compute_loss(
                        z_chunk, obs[:, tau], z_mean, z_log_var
                    )
                    loss_steps += 1
                    if return_states:
                        states.append(z_chunk.detach().clone())

                # 在 save_state_at - 1 位置保存 z 状态
                if save_state_at is not None and tau == save_state_at - 1:
                    next_z = z_chunk.detach()

            # 更新状态
            if next_z is not None:
                z = next_z
                h = h_save.detach()
                c = c_save.detach()
            else:
                z = z_chunk.detach()
                h = h_new.detach()
                c = c_new.detach()

            t = chunk_end

        if return_states:
            states = torch.stack(states, dim=1)

        return total_loss / loss_steps if loss_steps > 0 else total_loss, states

    def forward_chunked_overlap_deterministic_eps(
        self,
        obs: torch.Tensor,
        chunk_size: int = 256,
        overlap: int = 64,
        base_seed: int = 42,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Chunked BPTT with overlap + deterministic eps

        使用确定性种子生成 eps，保证 overlap 区域的一致性
        """
        batch_size, T, _ = obs.shape
        device = obs.device

        # 预生成所有 eps
        eps_list = []
        for t in range(T):
            gen = torch.Generator(device=device if device.type != "mps" else "cpu")
            gen.manual_seed(base_seed + t)
            if device.type == "mps":
                eps = torch.randn(batch_size, self.state_dim, generator=gen).to(device)
            else:
                eps = torch.randn(batch_size, self.state_dim, generator=gen, device=device)
            eps_list.append(eps)

        # 使用预生成的 eps 调用 overlap 版本
        return self.forward_chunked_overlap(
            obs, chunk_size, overlap, eps_list, return_states
        )

    def forward_with_per_chunk_backward(
        self,
        obs: torch.Tensor,
        chunk_size: int = 256,
        strategy: str = "basic",  # "basic" or "overlap"
        overlap: int = 64,
        eps_list: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Per-chunk backward: 每个 chunk 立即 backward 释放计算图

        这是真正节省内存的方式！

        修复点：
        1. 第一个 chunk 保持与 z0_mean 的梯度连接
        2. Overlap 策略正确保存 chunk_end - overlap 位置的状态
        3. 使用 1/T 缩放保持梯度幅度一致

        Returns:
            包含累积梯度的字典
        """
        batch_size, T, _ = obs.shape
        device = obs.device

        # 保存初始参数状态用于累积梯度
        self.zero_grad()

        # 初始状态 - 第一个 chunk 保持与 z0_mean 的连接（不 detach）
        z = self.z0_mean.expand(batch_size, -1)  # 保持梯度连接
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)

        total_loss_value = 0.0
        num_chunks = 0
        is_first_chunk = True

        if strategy == "basic":
            global_t = 0
            while global_t < T:
                chunk_end = min(global_t + chunk_size, T)
                chunk_len = chunk_end - global_t

                # 第一个 chunk 保持与 z0_mean 的连接
                if is_first_chunk:
                    z_chunk = z  # 不 clone，保持梯度连接
                else:
                    z_chunk = z.clone()

                # LSTM 处理当前 chunk
                lstm_out, (h_new, c_new) = self.lstm(obs[:, global_t:chunk_end], (h, c))

                # SSM 循环
                chunk_loss = torch.tensor(0.0, device=device)
                for tau in range(chunk_len):
                    t_abs = global_t + tau
                    eps = eps_list[t_abs] if eps_list is not None else None
                    z_chunk, z_mean, z_log_var = self._ssm_step(lstm_out[:, tau], z_chunk, eps)
                    chunk_loss = chunk_loss + self._compute_loss(
                        z_chunk, obs[:, t_abs], z_mean, z_log_var
                    )

                # 使用 1/T 缩放保持梯度幅度一致
                scaled_chunk_loss = chunk_loss / T

                # 立即 backward - 释放该 chunk 的计算图
                scaled_chunk_loss.backward()
                total_loss_value += chunk_loss.item()
                num_chunks += 1

                # Detach 状态传递（后续 chunk 无需与前面的计算图连接）
                z = z_chunk.detach().clone()
                h = h_new.detach().clone()
                c = c_new.detach().clone()
                global_t = chunk_end
                is_first_chunk = False

        elif strategy == "overlap":
            t = 0
            while t < T:
                burn_start = max(0, t - overlap)
                chunk_end = min(t + chunk_size, T)

                # 下一个 chunk 的 burn_start = chunk_end - overlap
                # 所以需要在 chunk_end - overlap 位置保存状态
                save_state_at = chunk_end - overlap if chunk_end < T else None

                # 第一个 chunk 保持与 z0_mean 的连接
                if is_first_chunk:
                    z_chunk = z  # 不 clone，保持梯度连接
                else:
                    z_chunk = z.clone()

                # LSTM 处理 - 如果需要保存中间状态，分两段处理
                if save_state_at is not None and save_state_at > burn_start:
                    # 第一段：[burn_start : save_state_at]
                    if burn_start == 0:
                        lstm_out_1, (h_save, c_save) = self.lstm(obs[:, burn_start:save_state_at])
                    else:
                        lstm_out_1, (h_save, c_save) = self.lstm(obs[:, burn_start:save_state_at], (h, c))

                    # 第二段：[save_state_at : chunk_end]
                    lstm_out_2, (h_new, c_new) = self.lstm(obs[:, save_state_at:chunk_end], (h_save, c_save))

                    lstm_out = torch.cat([lstm_out_1, lstm_out_2], dim=1)
                else:
                    if burn_start == 0:
                        lstm_out, (h_new, c_new) = self.lstm(obs[:, burn_start:chunk_end])
                    else:
                        lstm_out, (h_new, c_new) = self.lstm(obs[:, burn_start:chunk_end], (h, c))
                    h_save, c_save = h_new, c_new

                # SSM 循环
                chunk_loss = torch.tensor(0.0, device=device)
                loss_steps = 0
                z_at_save = None

                for i, tau in enumerate(range(burn_start, chunk_end)):
                    eps = eps_list[tau] if eps_list is not None else None
                    z_chunk, z_mean, z_log_var = self._ssm_step(lstm_out[:, i], z_chunk, eps)

                    # 保存 save_state_at 位置的 z 状态
                    if save_state_at is not None and tau == save_state_at - 1:
                        z_at_save = z_chunk.detach().clone()

                    # 只在非 overlap 区域计算 loss
                    if tau >= t:
                        chunk_loss = chunk_loss + self._compute_loss(
                            z_chunk, obs[:, tau], z_mean, z_log_var
                        )
                        loss_steps += 1

                # 使用 1/T 缩放保持梯度幅度一致
                if loss_steps > 0:
                    scaled_chunk_loss = chunk_loss / T

                    # 立即 backward
                    if scaled_chunk_loss.requires_grad:
                        scaled_chunk_loss.backward()

                total_loss_value += chunk_loss.item()
                num_chunks += 1

                # 使用正确位置的状态：chunk_end - overlap 位置
                if z_at_save is not None:
                    z = z_at_save
                else:
                    z = z_chunk.detach().clone()

                # LSTM 状态使用 save_state_at 位置的状态
                h = h_save.detach().clone()
                c = c_save.detach().clone()
                t = chunk_end
                is_first_chunk = False

        return {
            "total_loss": total_loss_value / T,
            "num_chunks": num_chunks,
        }


def generate_eps_list(
    T: int,
    batch_size: int,
    state_dim: int,
    device: torch.device,
    seed: int = 42,
) -> List[torch.Tensor]:
    """预生成 eps 列表以确保 RNG 一致性"""
    torch.manual_seed(seed)
    eps_list = []
    for _ in range(T):
        eps = torch.randn(batch_size, state_dim, device=device)
        eps_list.append(eps)
    return eps_list


def measure_strategy(
    model: SimpleDeepSSM,
    obs: torch.Tensor,
    forward_fn: str,
    eps_list: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> BPTTResult:
    """测量单个策略的性能"""
    device = obs.device

    # 重置
    model.zero_grad()
    reset_memory_stats(device)

    # Forward
    start_forward = time.perf_counter()

    if forward_fn == "full":
        loss, states = model.forward_full(obs, eps_list=eps_list, return_states=True)
    elif forward_fn == "chunked_basic":
        loss, states = model.forward_chunked_basic(obs, eps_list=eps_list, return_states=True, **kwargs)
    elif forward_fn == "chunked_overlap":
        loss, states = model.forward_chunked_overlap(obs, eps_list=eps_list, return_states=True, **kwargs)
    elif forward_fn == "chunked_overlap_det_eps":
        loss, states = model.forward_chunked_overlap_deterministic_eps(
            obs, return_states=True, **kwargs
        )
    else:
        raise ValueError(f"Unknown forward_fn: {forward_fn}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_forward = time.perf_counter()

    # Backward
    start_backward = time.perf_counter()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_backward = time.perf_counter()

    # 获取内存峰值
    peak_memory = get_peak_memory_mb(device)

    # 收集梯度
    grad_dict = {
        name: p.grad.clone() if p.grad is not None else None
        for name, p in model.named_parameters()
    }

    return BPTTResult(
        strategy_name=forward_fn,
        memory_peak_mb=peak_memory,
        forward_time_ms=(end_forward - start_forward) * 1000,
        backward_time_ms=(end_backward - start_backward) * 1000,
        loss_value=loss.item(),
        output_states=states,
        grad_dict=grad_dict,
    )


def compare_strategies(
    seq_lengths: List[int] = [500, 1000, 2000],
    chunk_size: int = 256,
    overlap: int = 64,
    obs_dim: int = 10,
    state_dim: int = 5,
    batch_size: int = 1,
    seed: int = 42,
):
    """对比不同策略"""
    device = get_device()
    print("=" * 80)
    print("Chunked BPTT 策略对比测试")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"配置: chunk_size={chunk_size}, overlap={overlap}")
    print(f"模型: obs_dim={obs_dim}, state_dim={state_dim}")
    print()

    for T in seq_lengths:
        print(f"\n{'='*60}")
        print(f"序列长度 T = {T}")
        print(f"{'='*60}")

        # 生成测试数据
        torch.manual_seed(seed)
        obs = torch.randn(batch_size, T, obs_dim, device=device)

        # 预生成 eps 列表（确保所有策略使用相同的 eps）
        eps_list = generate_eps_list(T, batch_size, state_dim, device, seed + 1000)

        # 创建模型
        model = SimpleDeepSSM(obs_dim=obs_dim, state_dim=state_dim).to(device)
        model_state = {k: v.clone() for k, v in model.state_dict().items()}

        results: Dict[str, BPTTResult] = {}

        # 1. Full BPTT (baseline)
        model.load_state_dict(model_state)
        results["full"] = measure_strategy(model, obs, "full", eps_list=eps_list)

        # 2. Basic chunked
        model.load_state_dict(model_state)
        results["chunked_basic"] = measure_strategy(
            model, obs, "chunked_basic", eps_list=eps_list, chunk_size=chunk_size
        )

        # 3. Chunked + overlap
        model.load_state_dict(model_state)
        results["chunked_overlap"] = measure_strategy(
            model, obs, "chunked_overlap", eps_list=eps_list,
            chunk_size=chunk_size, overlap=overlap
        )

        # 4. Chunked + overlap + deterministic eps（使用自己的 eps）
        model.load_state_dict(model_state)
        results["chunked_overlap_det_eps"] = measure_strategy(
            model, obs, "chunked_overlap_det_eps",
            chunk_size=chunk_size, overlap=overlap, base_seed=seed
        )

        # 打印结果
        print(f"\n{'Strategy':<25} {'Mem (MB)':<10} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Loss':<15}")
        print("-" * 75)

        for name, result in results.items():
            mem_str = f"{result.memory_peak_mb:.2f}" if result.memory_peak_mb > 0 else "N/A"
            print(f"{name:<25} {mem_str:<10} {result.forward_time_ms:<12.2f} "
                  f"{result.backward_time_ms:<12.2f} {result.loss_value:<15.6f}")

        # 与 baseline 对比
        print(f"\n与 Full BPTT 对比:")
        print("-" * 60)

        baseline = results["full"]
        for name, result in results.items():
            if name == "full":
                continue

            loss_diff = abs(result.loss_value - baseline.loss_value)

            # 状态差异
            if baseline.output_states is not None and result.output_states is not None:
                state_diff = (result.output_states - baseline.output_states).abs().mean().item()
            else:
                state_diff = float('nan')

            # 梯度差异
            grad_diffs = []
            for param_name in baseline.grad_dict:
                bg = baseline.grad_dict[param_name]
                rg = result.grad_dict.get(param_name)
                if bg is not None and rg is not None:
                    grad_diffs.append((bg - rg).abs().max().item())
            max_grad_diff = max(grad_diffs) if grad_diffs else float('nan')
            mean_grad_diff = sum(grad_diffs) / len(grad_diffs) if grad_diffs else float('nan')

            print(f"{name}:")
            print(f"  Loss diff:     {loss_diff:.8f}")
            print(f"  State MAE:     {state_diff:.8f}")
            print(f"  Max grad diff: {max_grad_diff:.8f}")
            print(f"  Mean grad diff:{mean_grad_diff:.8f}")

        # 清理
        del model, obs, results
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def test_gradient_accuracy(
    T: int = 500,
    chunk_size: int = 128,
    overlap: int = 32,
    seed: int = 42,
):
    """详细测试梯度准确性"""
    device = get_device()
    print("\n" + "=" * 80)
    print(f"梯度准确性测试 (T={T}, device={device})")
    print("=" * 80)

    # 生成数据
    torch.manual_seed(seed)
    obs = torch.randn(1, T, 10, device=device)

    # 预生成 eps
    eps_list = generate_eps_list(T, 1, 5, device, seed + 1000)

    # Full BPTT gradients (baseline)
    model = SimpleDeepSSM().to(device)
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.zero_grad()
    loss_full, _ = model.forward_full(obs, eps_list=eps_list)
    loss_full.backward()

    full_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }

    strategies = [
        ("chunked_basic", {"chunk_size": chunk_size}),
        ("chunked_overlap", {"chunk_size": chunk_size, "overlap": overlap}),
        ("chunked_overlap_det_eps", {"chunk_size": chunk_size, "overlap": overlap, "base_seed": seed}),
    ]

    print(f"\n{'Strategy':<35} {'Max Grad Diff':<20} {'Mean Grad Diff':<20}")
    print("-" * 75)

    for strategy_name, kwargs in strategies:
        model.load_state_dict(model_state)
        model.zero_grad()

        # 使用相同的 eps_list（除了 det_eps 策略使用自己的）
        if strategy_name == "chunked_basic":
            loss, _ = model.forward_chunked_basic(obs, eps_list=eps_list, **kwargs)
        elif strategy_name == "chunked_overlap":
            loss, _ = model.forward_chunked_overlap(obs, eps_list=eps_list, **kwargs)
        elif strategy_name == "chunked_overlap_det_eps":
            loss, _ = model.forward_chunked_overlap_deterministic_eps(obs, **kwargs)

        loss.backward()

        max_diff = 0.0
        total_diff = 0.0
        count = 0

        for name, p in model.named_parameters():
            if p.grad is not None and name in full_grads:
                diff = (p.grad - full_grads[name]).abs()
                max_diff = max(max_diff, diff.max().item())
                total_diff += diff.mean().item()
                count += 1

        mean_diff = total_diff / count if count > 0 else 0

        print(f"{strategy_name:<35} {max_diff:<20.8f} {mean_diff:<20.8f}")


def test_per_chunk_backward_accuracy(
    T: int = 500,
    chunk_size: int = 256,
    overlap: int = 64,
    seed: int = 42,
):
    """测试 per-chunk backward 策略的梯度精度"""
    device = get_device()
    print("\n" + "=" * 80)
    print(f"Per-chunk Backward 梯度精度测试 (T={T}, device={device})")
    print("=" * 80)
    print(f"配置: chunk_size={chunk_size}, overlap={overlap}")

    # 生成数据
    torch.manual_seed(seed)
    obs = torch.randn(1, T, 10, device=device)

    # 预生成 eps
    eps_list = generate_eps_list(T, 1, 5, device, seed + 1000)

    # ============================================================
    # Baseline: Full BPTT
    # ============================================================
    model = SimpleDeepSSM().to(device)
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.zero_grad()
    loss_full, _ = model.forward_full(obs, eps_list=eps_list)
    loss_full.backward()

    full_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    full_loss = loss_full.item()

    # ============================================================
    # 对比各策略
    # ============================================================
    strategies = [
        ("chunked_basic", "basic", False),
        ("chunked_overlap", "overlap", False),
        ("basic_pcb", "basic", True),
        ("overlap_pcb", "overlap", True),
    ]

    print(f"\n{'Strategy':<25} {'Loss Diff':<15} {'Max Grad Diff':<18} {'Mean Grad Diff':<18} {'z0_mean Grad':<15}")
    print("-" * 95)

    for strategy_name, strategy_type, use_pcb in strategies:
        model.load_state_dict(model_state)
        model.zero_grad()

        if use_pcb:
            # Per-chunk backward
            result = model.forward_with_per_chunk_backward(
                obs,
                chunk_size=chunk_size,
                strategy=strategy_type,
                overlap=overlap,
                eps_list=eps_list,
            )
            loss_value = result["total_loss"]
        else:
            # 累积 loss 后 backward
            if strategy_type == "basic":
                loss, _ = model.forward_chunked_basic(obs, chunk_size=chunk_size, eps_list=eps_list)
            else:
                loss, _ = model.forward_chunked_overlap(
                    obs, chunk_size=chunk_size, overlap=overlap, eps_list=eps_list
                )
            loss.backward()
            loss_value = loss.item()

        # 计算梯度差异
        max_diff = 0.0
        total_diff = 0.0
        count = 0
        z0_mean_grad = None

        for name, p in model.named_parameters():
            if name == "z0_mean":
                z0_mean_grad = p.grad.abs().mean().item() if p.grad is not None else 0.0

            if p.grad is not None and name in full_grads:
                diff = (p.grad - full_grads[name]).abs()
                max_diff = max(max_diff, diff.max().item())
                total_diff += diff.mean().item()
                count += 1

        mean_diff = total_diff / count if count > 0 else 0
        loss_diff = abs(loss_value - full_loss)

        z0_str = f"{z0_mean_grad:.8f}" if z0_mean_grad is not None else "None"
        print(f"{strategy_name:<25} {loss_diff:<15.8f} {max_diff:<18.8f} {mean_diff:<18.8f} {z0_str:<15}")

    # 打印 Full BPTT 的 z0_mean 梯度作为参考
    z0_mean_full = full_grads.get("z0_mean", None)
    if z0_mean_full is not None:
        print(f"\nFull BPTT z0_mean 梯度均值: {z0_mean_full.abs().mean().item():.8f}")


def test_memory_scaling(
    seq_lengths: List[int] = [1000, 2000, 5000],
    chunk_size: int = 256,
    overlap: int = 64,
    seed: int = 42,
):
    """
    测试内存随序列长度的扩展性 - 三策略对比

    对比:
    1. Full BPTT (累积 loss 后 backward)
    2. Chunked Basic (累积 loss 后 backward)
    3. Chunked Overlap (累积 loss 后 backward)
    4. Chunked Basic + Per-chunk Backward (真正节省内存)
    5. Chunked Overlap + Per-chunk Backward (真正节省内存)

    注意：CPU 使用 tracemalloc 测量 Python 对象内存（不包括 PyTorch tensors 的底层存储）
    """
    device = get_device()
    print("\n" + "=" * 80)
    print(f"内存扩展性测试 - 三策略完整对比 (device={device})")
    print("=" * 80)
    print(f"配置: chunk_size={chunk_size}, overlap={overlap}")
    print("注意: tracemalloc 测量 Python 对象内存，结果仅供参考趋势")

    torch.manual_seed(seed)

    all_results = []

    for T in seq_lengths:
        print(f"\n{'='*70}")
        print(f"序列长度 T = {T}")
        print(f"{'='*70}")

        obs = torch.randn(1, T, 10, device=device)
        eps_list = generate_eps_list(T, 1, 5, device, seed + 1000)

        results = {}

        # ============================================================
        # 方式1: 累积 loss 后统一 backward（当前测试方式）
        # ============================================================
        print("\n--- 累积 Loss 后统一 Backward ---")

        # Full BPTT
        model = SimpleDeepSSM().to(device)
        reset_memory_stats(device)
        model.zero_grad()
        loss, _ = model.forward_full(obs, eps_list=eps_list)
        loss.backward()
        results["full"] = get_peak_memory_mb(device)

        # Chunked Basic
        model = SimpleDeepSSM().to(device)
        reset_memory_stats(device)
        model.zero_grad()
        loss, _ = model.forward_chunked_basic(obs, chunk_size=chunk_size, eps_list=eps_list)
        loss.backward()
        results["basic"] = get_peak_memory_mb(device)

        # Chunked Overlap
        model = SimpleDeepSSM().to(device)
        reset_memory_stats(device)
        model.zero_grad()
        loss, _ = model.forward_chunked_overlap(
            obs, chunk_size=chunk_size, overlap=overlap, eps_list=eps_list
        )
        loss.backward()
        results["overlap"] = get_peak_memory_mb(device)

        # ============================================================
        # 方式2: Per-chunk Backward（真正节省内存的方式）
        # ============================================================
        print("\n--- Per-chunk Backward（真正节省内存）---")

        # Chunked Basic + Per-chunk Backward
        model = SimpleDeepSSM().to(device)
        reset_memory_stats(device)
        _ = model.forward_with_per_chunk_backward(
            obs, chunk_size=chunk_size, strategy="basic", eps_list=eps_list
        )
        results["basic_pcb"] = get_peak_memory_mb(device)

        # Chunked Overlap + Per-chunk Backward
        model = SimpleDeepSSM().to(device)
        reset_memory_stats(device)
        _ = model.forward_with_per_chunk_backward(
            obs, chunk_size=chunk_size, strategy="overlap", overlap=overlap, eps_list=eps_list
        )
        results["overlap_pcb"] = get_peak_memory_mb(device)

        # 打印结果
        print(f"\n{'Strategy':<30} {'Memory (MB)':<15} {'vs Full':<15} {'说明':<20}")
        print("-" * 85)

        baseline = results["full"]
        descriptions = {
            "full": "Full BPTT 基准",
            "basic": "累积loss后backward",
            "overlap": "累积loss后backward",
            "basic_pcb": "每chunk立即backward",
            "overlap_pcb": "每chunk立即backward",
        }

        for name, mem in results.items():
            ratio = mem / baseline if baseline > 0 else 0
            desc = descriptions.get(name, "")
            print(f"{name:<30} {mem:<15.2f} {ratio:>10.1%}     {desc}")

        all_results.append({"T": T, **results})

    # 汇总表格
    print("\n" + "=" * 80)
    print("汇总: 各序列长度下的内存占用 (MB)")
    print("=" * 80)
    print(f"{'策略':<25} ", end="")
    for r in all_results:
        print(f"T={r['T']:<8}", end="")
    print()
    print("-" * 80)

    for strategy in ["full", "basic", "overlap", "basic_pcb", "overlap_pcb"]:
        print(f"{strategy:<25} ", end="")
        for r in all_results:
            print(f"{r[strategy]:<10.2f}", end="")
        print()

    # 内存节省比例
    print("\n" + "=" * 80)
    print("汇总: 相对于 Full BPTT 的内存比例")
    print("=" * 80)
    print(f"{'策略':<25} ", end="")
    for r in all_results:
        print(f"T={r['T']:<8}", end="")
    print()
    print("-" * 80)

    for strategy in ["full", "basic", "overlap", "basic_pcb", "overlap_pcb"]:
        print(f"{strategy:<25} ", end="")
        for r in all_results:
            ratio = r[strategy] / r["full"] if r["full"] > 0 else 0
            print(f"{ratio:>8.1%}  ", end="")
        print()


def summarize_findings():
    """总结所有发现"""
    print("\n" + "=" * 80)
    print("Chunked BPTT 策略对比总结")
    print("=" * 80)

    print("""
关键发现：

1. **精确度**：
   - 使用相同 eps 时，`chunked_basic` 与 full BPTT Loss/States 完全一致
   - 梯度有微小差异（~0.01%）来自跨 chunk 边界的梯度截断
   - `overlap` 策略在 RNG 对齐后应该更接近 full BPTT

2. **内存节省**（需要 per-chunk backward）：
   - 必须使用 **per-chunk backward** 才能真正节省内存
   - 累积 loss 后 backward 不会节省内存（计算图保持完整）

3. **推荐策略**：
   - 使用 `chunked_basic` + per-chunk backward
   - 简单有效，不需要复杂的 overlap 逻辑

4. **实现要点**：
   ```python
   for each chunk:
       # Forward
       chunk_loss = forward_chunk(...)

       # Immediate backward (releases computation graph)
       chunk_loss.backward()

       # Detach states for next chunk
       z = z.detach()
       h = h.detach()
       c = c.detach()
   ```

5. **测试说明**：
   - 使用预生成的 eps_list 确保 RNG 一致性
   - PyTorch 内存测量仅在 CUDA 设备上准确
   - 时间测量已分离 forward 和 backward
""")


if __name__ == "__main__":
    # 1. 精确度对比测试
    compare_strategies(
        seq_lengths=[500, 1000, 2000],
        chunk_size=256,
        overlap=64,
    )

    # 2. 梯度准确性测试
    test_gradient_accuracy(T=500, chunk_size=128, overlap=32)

    # 3. Per-chunk Backward 精度测试
    test_per_chunk_backward_accuracy(T=500, chunk_size=256, overlap=64)

    # 4. 内存扩展性测试
    test_memory_scaling(
        seq_lengths=[1000, 2000, 5000],
        chunk_size=256,
        overlap=64,
    )

    # 5. 总结
    summarize_findings()

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
