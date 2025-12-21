"""
Deep State Space Model (DeepSSM) for time series feature extraction.

This module implements a deep learning-based state space model that combines
LSTM networks with Extended Kalman Filtering for robust state estimation.
"""

import gc
import random
from typing import Optional, Dict, Tuple, Union, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Import PyTorch configuration first to ensure CPU usage
try:
    # Try importing from project root
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from pytorch_config import get_device, get_torch_dtype
except ImportError:
    # Fallback functions if pytorch_config is not available
    def get_device():
        return "cpu"

    def get_torch_dtype(dtype_str="float32"):
        import torch

        return torch.float32 if dtype_str == "float32" else torch.float64


import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .kalman_filter import ExtendedKalmanFilter


class DeepSSMConfig(BaseModel):
    """Configuration for DeepSSM model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    obs_dim: Optional[int] = None
    state_dim: int = 5
    lstm_hidden: int = 64
    lstm_layers: int = 1
    transition_hidden: int = 128
    observation_hidden: int = 128

    learning_rate: float = 0.001
    max_epochs: int = 100
    batch_size: int = 32
    patience: int = 5
    min_delta: float = 0.01

    dropout: float = 0.1
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    use_scaler: bool = True  # Whether to use StandardScaler for input data

    # Chunked BPTT 参数 (Overlap + Per-chunk Backward)
    chunk_size: int = Field(default=256, ge=64, description="每个 chunk 的长度")
    overlap: int = Field(default=64, ge=0, description="Overlap/burn-in 长度")

    # Training mode: use EKF path (same as inference) or legacy sampling path
    use_ekf_train: bool = Field(
        default=True,
        description="Use EKF-based training (recommended). If False, uses legacy sampling.",
    )

    # EKF numerical stability parameters
    ekf_jitter: float = Field(
        default=1e-4, description="Jitter added to S matrix for numerical stability"
    )
    ekf_var_clamp_min: float = Field(
        default=1e-6, description="Minimum variance for P diagonal clamping"
    )

    device: str = "cpu"  # Force CPU usage
    dtype: str = "float32"
    seed: Optional[int] = 42

    # Non-serialized computed field
    torch_dtype: torch.dtype = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def set_computed_fields(self) -> "DeepSSMConfig":
        # Always use CPU, ignore auto detection
        object.__setattr__(self, "device", get_device())  # Returns 'cpu'
        # Use helper function to get torch dtype
        object.__setattr__(self, "torch_dtype", get_torch_dtype(self.dtype))

        # 验证 overlap < chunk_size
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class DeepSSMNet(nn.Module):
    """Neural network component of DeepSSM."""

    def __init__(self, config: DeepSSMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.obs_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
        )

        # Prior network: p(z_t | z_{t-1}) - only depends on previous state
        self.transition_prior = nn.Sequential(
            nn.Linear(config.state_dim, config.transition_hidden),
            nn.LayerNorm(config.transition_hidden),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transition_hidden, config.transition_hidden // 2),
            nn.Tanh(),
            nn.Linear(config.transition_hidden // 2, 2 * config.state_dim),
        )

        # Posterior network: q(z_t | z_{t-1}, h_t) - depends on previous state and LSTM encoding
        self.transition_posterior = nn.Sequential(
            nn.Linear(config.lstm_hidden + config.state_dim, config.transition_hidden),
            nn.LayerNorm(config.transition_hidden),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transition_hidden, config.transition_hidden // 2),
            nn.Tanh(),
            nn.Linear(config.transition_hidden // 2, 2 * config.state_dim),
        )

        self.observation = nn.Sequential(
            nn.Linear(config.state_dim, config.observation_hidden),
            nn.LayerNorm(config.observation_hidden),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.observation_hidden, config.observation_hidden // 2),
            nn.Tanh(),
            nn.Linear(config.observation_hidden // 2, 2 * config.obs_dim),
        )

        self.initial_state_mean = nn.Parameter(torch.zeros(config.state_dim))
        self.initial_state_log_var = nn.Parameter(torch.zeros(config.state_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_normal_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def get_transition_prior(
        self, z_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state transition prior distribution p(z_t | z_{t-1}).

        The prior only depends on the previous state, not the current observation.
        This is used for EKF prediction and KL divergence computation.

        Args:
            z_prev: Previous state [batch, state_dim] or [state_dim]

        Returns:
            Mean and log variance of prior transition distribution
        """
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)

        output = self.transition_prior(z_prev)
        mean, log_var = torch.split(output, self.config.state_dim, dim=-1)
        log_var = torch.clamp(log_var, -10, 10)
        return mean, log_var

    def get_transition_posterior(
        self, z_prev: torch.Tensor, lstm_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state transition posterior distribution q(z_t | z_{t-1}, h_t).

        The posterior depends on both the previous state and the LSTM encoding
        of historical observations. This is used for inference/training.

        Args:
            z_prev: Previous state [batch, state_dim] or [state_dim]
            lstm_out: LSTM output encoding [batch, hidden_dim] or [hidden_dim]

        Returns:
            Mean and log variance of posterior transition distribution
        """
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)
        if lstm_out.dim() == 1:
            lstm_out = lstm_out.unsqueeze(0)

        combined = torch.cat([lstm_out, z_prev], dim=-1)
        output = self.transition_posterior(combined)
        mean, log_var = torch.split(output, self.config.state_dim, dim=-1)
        log_var = torch.clamp(log_var, -10, 10)
        return mean, log_var

    def get_transition_dist(
        self, lstm_out: torch.Tensor, z_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward-compatible alias for get_transition_posterior.

        DEPRECATED: Use get_transition_posterior instead.
        """
        return self.get_transition_posterior(z_prev, lstm_out)

    def get_observation_dist(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute observation distribution parameters.

        Args:
            z: Current state [batch, state_dim]

        Returns:
            Mean and log variance of observation distribution
        """
        output = self.observation(z)
        mean, log_var = torch.split(output, self.config.obs_dim, dim=-1)
        log_var = torch.clamp(log_var, -10, 10)
        return mean, log_var

    def compute_transition_jacobian(
        self, z_prev: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        """
        Compute per-sample Jacobian of transition prior mean w.r.t. previous state.

        Uses vmap + jacrev for efficient batched Jacobian computation.
        The Jacobian is differentiable by default (jacrev produces differentiable outputs).

        F = ∂μ_prior(z_{t-1}) / ∂z_{t-1}

        Note: This method uses eval mode internally to disable Dropout for consistent
        linearization. Callers should also compute prior_mean in eval mode for consistency.

        Args:
            z_prev: Previous state [batch, state_dim] or [state_dim]
            create_graph: Unused parameter (kept for API compatibility).
                         jacrev outputs are differentiable by default.

        Returns:
            Jacobian matrix F [state_dim, state_dim] (for batch=1)
            or [batch, state_dim, state_dim] (for batch>1)
        """
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)

        batch_size = z_prev.shape[0]

        # Detach input to prevent gradient explosion through EKF covariance chain
        # The Jacobian is used for covariance propagation P = F @ P @ F.T + Q
        # Gradients should flow through the network parameters, not through z_prev
        z_prev_detached = z_prev.detach()

        # Define per-sample function for Jacobian computation
        # This function takes a single sample [state_dim] and returns [state_dim]
        def prior_mean_fn(z: torch.Tensor) -> torch.Tensor:
            mean, _ = self.get_transition_prior(z.unsqueeze(0))
            return mean.squeeze(0)

        # Use eval mode for deterministic Jacobian (disable Dropout)
        # This ensures consistent Jacobians for EKF, as recommended by Codex
        was_training = self.transition_prior.training
        self.transition_prior.eval()

        try:
            # vmap(jacrev(fn)) computes per-sample Jacobians efficiently
            # jacrev computes Jacobian using reverse-mode autodiff
            jacobian_fn = jacrev(prior_mean_fn, argnums=0, has_aux=False)
            F = vmap(jacobian_fn)(z_prev_detached)  # [batch, state_dim, state_dim]
        finally:
            # Restore original training mode
            self.transition_prior.train(was_training)

        # Return [state_dim, state_dim] for batch_size=1 (backward compatibility)
        return F.squeeze(0) if batch_size == 1 else F

    def compute_observation_jacobian(
        self, z: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        """
        Compute per-sample Jacobian of observation mean w.r.t. state.

        Uses vmap + jacrev for efficient batched Jacobian computation.
        The Jacobian is differentiable by default (jacrev produces differentiable outputs).

        H = ∂μ_obs(z) / ∂z

        Note: This method uses eval mode internally to disable Dropout for consistent
        linearization. Callers should also compute obs_mean in eval mode for consistency.

        Args:
            z: Current state [batch, state_dim] or [state_dim]
            create_graph: Unused parameter (kept for API compatibility).
                         jacrev outputs are differentiable by default.

        Returns:
            Jacobian matrix H [obs_dim, state_dim] (for batch=1)
            or [batch, obs_dim, state_dim] (for batch>1)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        batch_size = z.shape[0]

        # Detach input to prevent gradient explosion through EKF covariance chain
        # The Jacobian is used for covariance propagation and Kalman gain
        # Gradients should flow through the network parameters, not through z
        z_detached = z.detach()

        # Define per-sample function for Jacobian computation
        # This function takes a single sample [state_dim] and returns [obs_dim]
        def obs_mean_fn(z_in: torch.Tensor) -> torch.Tensor:
            mean, _ = self.get_observation_dist(z_in.unsqueeze(0))
            return mean.squeeze(0)

        # Use eval mode for deterministic Jacobian (disable Dropout in observation network)
        was_training = self.observation.training
        self.observation.eval()

        try:
            # vmap(jacrev(fn)) computes per-sample Jacobians efficiently
            jacobian_fn = jacrev(obs_mean_fn, argnums=0, has_aux=False)
            H = vmap(jacobian_fn)(z_detached)  # [batch, obs_dim, state_dim]
        finally:
            # Restore original training mode
            self.observation.train(was_training)

        # Return [obs_dim, state_dim] for batch_size=1 (backward compatibility)
        return H.squeeze(0) if batch_size == 1 else H

    def forward(
        self, observations: torch.Tensor, return_all_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            observations: Input observations [batch, time, obs_dim]
            return_all_states: Whether to return all latent states

        Returns:
            Dictionary containing predictions and latent states
        """
        batch_size, seq_len, _ = observations.shape
        # device = observations.device

        lstm_out, _ = self.lstm(observations)

        z0_mean = self.initial_state_mean.expand(batch_size, -1)
        z0_log_var = self.initial_state_log_var.expand(batch_size, -1)
        z0_std = torch.exp(0.5 * z0_log_var)

        if self.training:
            eps = torch.randn_like(z0_std)
            z = z0_mean + z0_std * eps
        else:
            z = z0_mean

        states = [z] if return_all_states else []
        obs_means = []
        obs_log_vars = []
        trans_means = [z0_mean]
        trans_log_vars = [z0_log_var]

        for t in range(seq_len):
            if t > 0:
                trans_mean, trans_log_var = self.get_transition_dist(
                    lstm_out[:, t, :], z
                )
                trans_std = torch.exp(0.5 * trans_log_var)

                if self.training:
                    eps = torch.randn_like(trans_std)
                    z = trans_mean + trans_std * eps
                else:
                    z = trans_mean

                trans_means.append(trans_mean)
                trans_log_vars.append(trans_log_var)

            if return_all_states:
                states.append(z)

            obs_mean, obs_log_var = self.get_observation_dist(z)
            obs_means.append(obs_mean)
            obs_log_vars.append(obs_log_var)

        output = {
            "obs_means": torch.stack(obs_means, dim=1),
            "obs_log_vars": torch.stack(obs_log_vars, dim=1),
            "trans_means": torch.stack(trans_means, dim=1),
            "trans_log_vars": torch.stack(trans_log_vars, dim=1),
        }

        if return_all_states:
            output["states"] = torch.stack(states, dim=1)

        return output

    def _ssm_step(
        self,
        lstm_out_t: torch.Tensor,  # [batch, lstm_hidden]
        z_prev: torch.Tensor,  # [batch, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步 SSM 更新（重参数化采样）

        Returns:
            z: 采样后的状态
            z_mean: 转移均值
            z_log_var: 转移对数方差
        """
        z_mean, z_log_var = self.get_transition_dist(lstm_out_t, z_prev)
        z_std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(z_std)
        z = z_mean + z_std * eps
        return z, z_mean, z_log_var

    def _compute_step_loss(
        self,
        z: torch.Tensor,  # [batch, state_dim]
        obs_t: torch.Tensor,  # [batch, obs_dim]
        posterior_mean: torch.Tensor,
        posterior_log_var: torch.Tensor,
        prior_mean: Optional[torch.Tensor] = None,
        prior_log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute single-step loss (reconstruction + KL).

        If prior is provided, computes KL(posterior || prior).
        Otherwise, computes KL(posterior || N(0, I)) for backward compatibility.

        Args:
            z: Current state [batch, state_dim]
            obs_t: Current observation [batch, obs_dim]
            posterior_mean: Posterior transition mean [batch, state_dim]
            posterior_log_var: Posterior transition log variance [batch, state_dim]
            prior_mean: Prior transition mean [batch, state_dim] (optional)
            prior_log_var: Prior transition log variance [batch, state_dim] (optional)

        Returns:
            Combined loss (reconstruction + KL)
        """
        # Observation distribution
        obs_mean, obs_log_var = self.get_observation_dist(z)

        # Reconstruction loss (Gaussian NLL)
        obs_var = torch.exp(obs_log_var)
        recon_loss = (
            0.5 * (obs_log_var + (obs_t - obs_mean) ** 2 / obs_var).sum(dim=-1).mean()
        )

        # KL divergence
        if prior_mean is not None and prior_log_var is not None:
            # KL(q || p) = 0.5 * (log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1)
            # = 0.5 * (log_var_p - log_var_q + exp(log_var_q)/exp(log_var_p)
            #          + (mean_q - mean_p)²/exp(log_var_p) - 1)
            prior_var = torch.exp(prior_log_var)
            posterior_var = torch.exp(posterior_log_var)
            kl_loss = (
                0.5
                * (
                    prior_log_var
                    - posterior_log_var
                    + posterior_var / prior_var
                    + (posterior_mean - prior_mean) ** 2 / prior_var
                    - 1
                )
                .sum(dim=-1)
                .mean()
            )
        else:
            # Backward compatibility: KL vs N(0, I)
            kl_loss = (
                -0.5
                * (
                    1
                    + posterior_log_var
                    - posterior_mean**2
                    - torch.exp(posterior_log_var)
                )
                .sum(dim=-1)
                .mean()
            )

        return recon_loss + 0.1 * kl_loss

    def _kl_full_cov_to_diag(
        self,
        z_update: torch.Tensor,  # [B, D] posterior mean (from EKF update)
        P_update: torch.Tensor,  # [B, D, D] posterior full covariance
        prior_mean: torch.Tensor,  # [B, D] prior mean
        prior_log_var: torch.Tensor,  # [B, D] prior log variance (diagonal)
    ) -> torch.Tensor:
        """
        Compute KL divergence from full-covariance posterior to diagonal prior.

        KL(N(z_update, P_update) || N(prior_mean, diag(exp(prior_log_var))))

        This is used when the posterior comes from EKF update (full covariance)
        and the prior is from the transition network (diagonal covariance).

        Args:
            z_update: Posterior mean [batch, state_dim]
            P_update: Posterior full covariance [batch, state_dim, state_dim]
            prior_mean: Prior mean [batch, state_dim]
            prior_log_var: Prior log variance [batch, state_dim]

        Returns:
            KL divergence (scalar, averaged over batch)
        """
        jitter = self.config.ekf_jitter
        clamp_min = self.config.ekf_var_clamp_min
        D = P_update.size(-1)
        device = P_update.device
        dtype = P_update.dtype

        # Ensure symmetry (no jitter here - jitter only for Cholesky)
        P_sym = 0.5 * (P_update + P_update.transpose(-1, -2))

        # logdet Σq via Cholesky (add jitter for numerical stability)
        P_chol = P_sym + jitter * torch.eye(D, device=device, dtype=dtype)
        try:
            L = torch.linalg.cholesky(P_chol)  # [B, D, D]
            logdet_q = 2.0 * torch.sum(
                torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
            )
        except RuntimeError:
            # Fallback: use eigenvalues if Cholesky fails
            eigvals = torch.linalg.eigvalsh(P_sym)
            eigvals = torch.clamp(eigvals, min=clamp_min)
            logdet_q = torch.sum(torch.log(eigvals), dim=-1)

        # Prior terms (diagonal): logdet Σp = sum(prior_log_var)
        prior_log_var_clamped = prior_log_var.clamp(-30, 30)
        logdet_p = torch.sum(prior_log_var_clamped, dim=-1)

        # Σp^{-1} is diagonal: exp(-log_var)
        inv_prior_var = torch.exp(-prior_log_var_clamped)

        # trace(Σp^{-1} Σq) - use original P_diag (not jittered) with clamping
        P_diag = torch.diagonal(P_sym, dim1=-2, dim2=-1)
        P_diag_clamped = torch.clamp(P_diag, min=clamp_min)
        trace_term = torch.sum(inv_prior_var * P_diag_clamped, dim=-1)

        # Quadratic term: (μq - μp)^T Σp^{-1} (μq - μp)
        diff = z_update - prior_mean
        quad_term = torch.sum(inv_prior_var * diff * diff, dim=-1)

        # KL = 0.5 * (log|Σp| - log|Σq| - D + tr(Σp^{-1}Σq) + quad)
        kl = 0.5 * (logdet_p - logdet_q - D + trace_term + quad_term)
        return kl.mean()

    def forward_train(
        self,
        observations: torch.Tensor,  # [batch, T, obs_dim]
        chunk_size: int = 256,
        overlap: int = 64,
    ) -> Dict[str, float]:
        """
        Overlap + Per-chunk Backward 训练（默认训练方式）

        关键实现点：
        1. 第一个 chunk 保持与 initial_state_mean 的梯度连接（不 detach）
        2. Overlap 区域 [burn_start, t) 只做 warm-up，不计算 loss
        3. 在 chunk_end - overlap 位置保存 LSTM (h, c) 和 SSM (z) 状态
        4. 每个 chunk 立即 backward() 释放计算图
        5. 使用 chunk_loss / T 缩放保持梯度幅度一致

        Args:
            observations: 输入序列 [batch, T, obs_dim]
            chunk_size: 每个 chunk 的长度
            overlap: Overlap/burn-in 长度

        Returns:
            包含 total_loss 和 num_chunks 的字典
        """
        batch_size, T, _ = observations.shape
        device = observations.device

        # 初始状态 - 第一个 chunk 保持与 initial_state_mean 的梯度连接
        z = self.initial_state_mean.expand(batch_size, -1)  # 不 detach
        h = torch.zeros(
            self.config.lstm_layers,
            batch_size,
            self.config.lstm_hidden,
            device=device,
            dtype=observations.dtype,
        )
        c = torch.zeros_like(h)

        total_loss_value = 0.0
        num_chunks = 0
        is_first_chunk = True

        t = 0
        while t < T:
            burn_start = max(0, t - overlap)
            chunk_end = min(t + chunk_size, T)

            # 下一个 chunk 的 burn_start = chunk_end - overlap
            # 所以需要在 chunk_end - overlap 位置保存状态
            save_state_at = chunk_end - overlap if chunk_end < T else None

            # 第一个 chunk 保持与 initial_state_mean 的梯度连接
            if is_first_chunk:
                z_chunk = z  # 不 clone，保持梯度连接
            else:
                z_chunk = z.clone()

            # LSTM 处理 - 如果需要保存中间状态，分两段处理
            # 边界检查：save_state_at 必须严格在 (burn_start, chunk_end) 范围内才分段
            need_split = (
                save_state_at is not None
                and save_state_at > burn_start
                and save_state_at < chunk_end  # 确保第二段不为空
            )

            if need_split:
                # 第一段：[burn_start : save_state_at]
                if burn_start == 0:
                    lstm_out_1, (h_save, c_save) = self.lstm(
                        observations[:, burn_start:save_state_at]
                    )
                else:
                    lstm_out_1, (h_save, c_save) = self.lstm(
                        observations[:, burn_start:save_state_at], (h, c)
                    )

                # 第二段：[save_state_at : chunk_end]
                lstm_out_2, (h_new, c_new) = self.lstm(
                    observations[:, save_state_at:chunk_end], (h_save, c_save)
                )

                lstm_out = torch.cat([lstm_out_1, lstm_out_2], dim=1)
            else:
                # 不需要保存中间状态，直接处理
                if burn_start == 0:
                    lstm_out, (h_new, c_new) = self.lstm(
                        observations[:, burn_start:chunk_end]
                    )
                else:
                    lstm_out, (h_new, c_new) = self.lstm(
                        observations[:, burn_start:chunk_end], (h, c)
                    )
                h_save, c_save = h_new, c_new

            # SSM 循环
            chunk_loss = torch.tensor(0.0, device=device)
            loss_steps = 0
            z_at_save = None

            for i, tau in enumerate(range(burn_start, chunk_end)):
                z_chunk, z_mean, z_log_var = self._ssm_step(lstm_out[:, i], z_chunk)

                # 保存 save_state_at 位置的 z 状态
                if save_state_at is not None and tau == save_state_at - 1:
                    z_at_save = z_chunk.detach().clone()

                # 只在非 overlap 区域计算 loss
                if tau >= t:
                    chunk_loss = chunk_loss + self._compute_step_loss(
                        z_chunk, observations[:, tau], z_mean, z_log_var
                    )
                    loss_steps += 1

            # 使用 1/T 缩放保持梯度幅度一致
            if loss_steps > 0:
                scaled_chunk_loss = chunk_loss / T

                # 立即 backward - 释放该 chunk 的计算图
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
            "total_loss": total_loss_value / T if T > 0 else 0.0,
            "num_chunks": num_chunks,
        }

    def forward_train_ekf(
        self,
        observations: torch.Tensor,  # [batch, T, obs_dim]
        chunk_size: int = 256,
        overlap: int = 64,
    ) -> Dict[str, float]:
        """
        EKF-based training with proper prior/posterior separation.

        This training method mirrors the inference path (_kalman_filter):
        1. Predict: Use prior p(z_t|z_{t-1}) with transition Jacobian F
        2. Update: Use observation y_t with observation Jacobian H
        3. Compute loss = reconstruction + KL(posterior || prior)

        The LSTM is updated AFTER each step (same as inference), and its output
        from the previous step is used for the posterior distribution.

        Args:
            observations: Input sequence [batch, T, obs_dim]
            chunk_size: Length of each chunk
            overlap: Overlap/burn-in length

        Returns:
            Dictionary containing total_loss and num_chunks
        """
        batch_size, T, _ = observations.shape
        device = observations.device
        dtype = observations.dtype
        state_dim = self.config.state_dim
        obs_dim = self.config.obs_dim

        # Per-sample Jacobians are now computed using vmap + jacrev,
        # enabling batch training with correct gradients for all samples.

        # Initialize state deterministically (same as inference)
        z = self.initial_state_mean.expand(batch_size, -1)

        # Initialize EKF covariance
        P = (
            torch.diag(torch.exp(self.initial_state_log_var))
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )

        # Initialize LSTM hidden state
        lstm_hidden = torch.zeros(
            self.config.lstm_layers,
            batch_size,
            self.config.lstm_hidden,
            device=device,
            dtype=dtype,
        )
        lstm_cell = torch.zeros_like(lstm_hidden)

        # Identity matrix for EKF update
        I_state = torch.eye(state_dim, device=device, dtype=dtype)

        total_loss_value = 0.0
        num_chunks = 0

        t = 0
        while t < T:
            burn_start = max(0, t - overlap)
            chunk_end = min(t + chunk_size, T)
            save_state_at = chunk_end - overlap if chunk_end < T else None

            chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_steps = 0

            # State and covariance for this chunk
            z_chunk = z.clone() if t > 0 else z
            P_chunk = P.clone()

            # Saved states for next chunk
            z_at_save = None
            P_at_save = None
            lstm_hidden_save = None
            lstm_cell_save = None

            for tau in range(burn_start, chunk_end):
                y_tau = observations[:, tau : tau + 1, :]  # [batch, 1, obs_dim]
                obs_t = observations[:, tau]  # [batch, obs_dim]

                if tau == 0:
                    # t=0: Use initial state, compute initial loss
                    # For t=0, just compute reconstruction loss (no KL since no transition)
                    if tau >= t:  # Only compute loss in non-overlap region
                        obs_mean, obs_log_var = self.get_observation_dist(z_chunk)
                        obs_var = torch.exp(obs_log_var)
                        recon_loss = (
                            0.5
                            * (obs_log_var + (obs_t - obs_mean) ** 2 / obs_var)
                            .sum(dim=-1)
                            .mean()
                        )
                        chunk_loss = chunk_loss + recon_loss
                        loss_steps += 1
                else:
                    # Step 1: Get prior p(z_t | z_{t-1}) in eval mode
                    # Use eval mode to disable Dropout for consistent EKF linearization
                    # Both prior distribution and Jacobian must use the same mode
                    was_training_prior = self.transition_prior.training
                    self.transition_prior.eval()
                    try:
                        prior_mean, prior_log_var = self.get_transition_prior(z_chunk)
                        # Compute per-sample transition Jacobian F using vmap
                        # (compute_transition_jacobian also uses eval mode internally)
                        F_batch = self.compute_transition_jacobian(z_chunk)
                        if F_batch.dim() == 2:
                            F_batch = F_batch.unsqueeze(0)
                    finally:
                        self.transition_prior.train(was_training_prior)

                    # Step 2: EKF Predict with prior
                    z_pred = prior_mean  # [batch, state_dim]
                    Q = torch.diag_embed(
                        torch.exp(prior_log_var)
                    )  # [batch, state, state]

                    # P_pred = F @ P @ F.T + Q
                    P_pred = (
                        torch.bmm(
                            torch.bmm(F_batch, P_chunk), F_batch.transpose(-1, -2)
                        )
                        + Q
                    )

                    # Step 3: EKF Update with observation (also in eval mode for consistency)
                    was_training_obs = self.observation.training
                    self.observation.eval()
                    try:
                        # Compute per-sample observation Jacobian H using vmap
                        H_batch = self.compute_observation_jacobian(z_pred)
                        if H_batch.dim() == 2:
                            H_batch = H_batch.unsqueeze(0)
                        # Observation prediction (same eval mode as Jacobian)
                        obs_mean, obs_log_var = self.get_observation_dist(z_pred)
                    finally:
                        self.observation.train(was_training_obs)
                    R = torch.diag_embed(
                        torch.exp(obs_log_var)
                    )  # [batch, obs_dim, obs_dim]

                    # Innovation covariance S = H @ P_pred @ H.T + R
                    S = (
                        torch.bmm(torch.bmm(H_batch, P_pred), H_batch.transpose(-1, -2))
                        + R
                    )
                    # Add jitter for numerical stability
                    jitter = self.config.ekf_jitter * torch.eye(
                        obs_dim, device=device, dtype=dtype
                    )
                    S = S + jitter

                    # Kalman gain K = P_pred @ H.T @ inv(S)
                    # Use solve for stability: solve(S, H @ P_pred.T).T
                    try:
                        HP = torch.bmm(H_batch, P_pred)  # [batch, obs_dim, state_dim]
                        K = torch.linalg.solve(S, HP).transpose(
                            -1, -2
                        )  # [batch, state_dim, obs_dim]
                    except RuntimeError:
                        # Fallback to pseudo-inverse
                        K = torch.bmm(
                            torch.bmm(P_pred, H_batch.transpose(-1, -2)),
                            torch.linalg.pinv(S),
                        )

                    # State update: z_update = z_pred + K @ (y - obs_mean)
                    innovation = obs_t - obs_mean  # [batch, obs_dim]
                    z_update = z_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(
                        -1
                    )

                    # Covariance update using Joseph form for numerical stability:
                    # P_update = (I - KH) @ P_pred @ (I - KH).T + K @ R @ K.T
                    # This maintains PSD and symmetry better than simplified form
                    KH = torch.bmm(K, H_batch)  # [batch, state_dim, state_dim]
                    I_KH = I_state.unsqueeze(0) - KH
                    P_update = torch.bmm(
                        torch.bmm(I_KH, P_pred), I_KH.transpose(-1, -2)
                    )
                    P_update = P_update + torch.bmm(
                        torch.bmm(K, R), K.transpose(-1, -2)
                    )

                    # Enforce symmetry and clamp diagonal for numerical stability
                    P_update = 0.5 * (P_update + P_update.transpose(-1, -2))
                    P_diag = torch.diagonal(P_update, dim1=-2, dim2=-1)
                    P_diag_clamped = torch.clamp(
                        P_diag, min=self.config.ekf_var_clamp_min
                    )
                    # Update diagonal in-place
                    P_update = P_update.clone()
                    for i in range(state_dim):
                        P_update[:, i, i] = P_diag_clamped[:, i]

                    # Update state and covariance
                    z_chunk = z_update
                    P_chunk = P_update

                    # Step 4: Compute loss
                    if tau >= t:  # Only compute loss in non-overlap region
                        # Reconstruction loss
                        obs_var = torch.exp(obs_log_var)
                        recon_loss = (
                            0.5
                            * (obs_log_var + (obs_t - obs_mean) ** 2 / obs_var)
                            .sum(dim=-1)
                            .mean()
                        )

                        # KL loss: KL(N(z_update, P_update) || N(prior_mean, diag(prior_var)))
                        kl_loss = self._kl_full_cov_to_diag(
                            z_update, P_update, prior_mean, prior_log_var
                        )

                        step_loss = recon_loss + 0.1 * kl_loss
                        chunk_loss = chunk_loss + step_loss
                        loss_steps += 1

                # Update LSTM with y_tau (AFTER state update, same as inference)
                _, (lstm_hidden, lstm_cell) = self.lstm(y_tau, (lstm_hidden, lstm_cell))

                # Save states at checkpoint
                if save_state_at is not None and tau == save_state_at - 1:
                    z_at_save = z_chunk.detach().clone()
                    P_at_save = P_chunk.detach().clone()
                    lstm_hidden_save = lstm_hidden.detach().clone()
                    lstm_cell_save = lstm_cell.detach().clone()

            # Backward for this chunk
            if loss_steps > 0:
                scaled_chunk_loss = chunk_loss / T
                if scaled_chunk_loss.requires_grad:
                    scaled_chunk_loss.backward()

            total_loss_value += chunk_loss.item()
            num_chunks += 1

            # Update states for next chunk
            if z_at_save is not None:
                z = z_at_save
                P = P_at_save
                lstm_hidden = lstm_hidden_save
                lstm_cell = lstm_cell_save
            else:
                z = z_chunk.detach().clone()
                P = P_chunk.detach().clone()
                lstm_hidden = lstm_hidden.detach().clone()
                lstm_cell = lstm_cell.detach().clone()

            t = chunk_end

        return {
            "total_loss": total_loss_value / T if T > 0 else 0.0,
            "num_chunks": num_chunks,
        }


class DeepSSM:
    """
    Deep State Space Model for time series feature extraction.

    This model combines deep learning with state space modeling to extract
    meaningful latent features from time series data.
    """

    def __init__(
        self,
        config: Optional[DeepSSMConfig] = None,
        obs_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize DeepSSM.

        Args:
            config: Model configuration
            obs_dim: Observation dimension (if not in config)
            **kwargs: Additional config parameters
        """
        if config is None:
            config = DeepSSMConfig(obs_dim=obs_dim, **kwargs)
        elif obs_dim is not None:
            config.obs_dim = obs_dim

        if config.obs_dim is None:
            raise ValueError("obs_dim must be specified")

        self.config = config
        self.device = torch.device(config.device)

        if config.seed is not None:
            set_seed(config.seed)

        self.model = DeepSSMNet(config).to(self.device)
        self.scaler_params = None
        self.training_history = []
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Indicates whether the model has been trained."""
        return self._is_fitted

    def _standardize(
        self, data: np.ndarray, fit: bool = False, inplace: bool = False
    ) -> np.ndarray:
        """
        Standardize data using stored parameters.

        Args:
            data: Input data
            fit: Whether to fit the scaler
            inplace: If True, modify data in-place to save memory

        Returns:
            Standardized data (or original if use_scaler=False)
        """
        if not self.config.use_scaler:
            return data

        if fit:
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            std[std < 1e-8] = 1.0
            self.scaler_params = {"mean": mean, "std": std}
        elif self.scaler_params is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        mean = self.scaler_params["mean"]
        std = self.scaler_params["std"]

        if inplace:
            # In-place modification to avoid creating copies
            data -= mean
            data /= std
            return data
        else:
            return (data - mean) / std

    def _inverse_standardize(self, data: np.ndarray) -> np.ndarray:
        """Inverse standardization."""
        if not self.config.use_scaler or self.scaler_params is None:
            return data

        mean = self.scaler_params["mean"]
        std = self.scaler_params["std"]

        return data * std + mean

    def _compute_elbo_loss(
        self, observations: torch.Tensor, output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Evidence Lower Bound (ELBO) loss.

        Args:
            observations: True observations
            output: Model output dictionary

        Returns:
            ELBO loss value
        """
        obs_means = output["obs_means"]
        obs_log_vars = output["obs_log_vars"]
        trans_means = output["trans_means"]
        trans_log_vars = output["trans_log_vars"]

        obs_std = torch.exp(0.5 * obs_log_vars)
        obs_dist = torch.distributions.Normal(obs_means, obs_std)
        reconstruction_loss = -obs_dist.log_prob(observations).sum(dim=-1).mean()

        trans_std = torch.exp(0.5 * trans_log_vars)

        kl_loss = (
            0.5
            * (trans_means.pow(2) + trans_std.pow(2) - 2 * trans_log_vars - 1)
            .sum(dim=-1)
            .mean()
        )

        return reconstruction_loss + 0.1 * kl_loss

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        val_data: Optional[Union[np.ndarray, pd.DataFrame, torch.Tensor]] = None,
        seed: Optional[int] = None,
    ) -> "DeepSSM":
        """
        Train the DeepSSM model.

        Args:
            X: Training data [n_samples, n_features] or [n_timesteps, n_features]
            val_data: Validation data (optional)
            seed: Random seed for training

        Returns:
            Self
        """
        if seed is not None:
            set_seed(seed)
        elif self.config.seed is not None:
            set_seed(self.config.seed)

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X = X.astype(np.float32)

        if len(X.shape) == 2:
            X = X[np.newaxis, :]

        # In-place standardization to avoid extra copy
        # X is already a copy from astype(), so inplace is safe
        X_scaled = self._standardize(X[0], fit=True, inplace=True)
        # Use from_numpy for zero-copy tensor creation
        X_tensor = (
            torch.from_numpy(X_scaled)
            .to(dtype=self.config.torch_dtype, device=self.device)
            .unsqueeze(0)
        )

        val_tensor = None
        if val_data is not None:
            if isinstance(val_data, pd.DataFrame):
                val_data = val_data.values
            elif isinstance(val_data, torch.Tensor):
                val_data = val_data.cpu().numpy()

            val_data = val_data.astype(np.float32)
            if len(val_data.shape) == 2:
                val_data = val_data[np.newaxis, :]

            # In-place standardization for validation data too
            val_scaled = self._standardize(val_data[0], fit=False, inplace=True)
            val_tensor = (
                torch.from_numpy(val_scaled)
                .to(dtype=self.config.torch_dtype, device=self.device)
                .unsqueeze(0)
            )

        optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.config.patience // 2,
            min_lr=1e-6,
        )

        best_loss = float("inf")
        patience_counter = 0
        self.training_history = []

        self.model.train()

        for epoch in range(self.config.max_epochs):
            optimizer.zero_grad()

            # Choose training method based on config
            if self.config.use_ekf_train:
                # EKF-based training (consistent with inference)
                result = self.model.forward_train_ekf(
                    X_tensor,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.overlap,
                )
            else:
                # Legacy sampling-based training
                result = self.model.forward_train(
                    X_tensor,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.overlap,
                )
            train_loss = result["total_loss"]

            # 梯度已在 forward_train 中通过 per-chunk backward 累积
            # 无需再调用 loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            optimizer.step()

            val_loss = None
            if val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(val_tensor)
                    val_loss = self._compute_elbo_loss(val_tensor, val_output).item()
                self.model.train()

            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            scheduler.step(val_loss if val_loss is not None else train_loss)

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.config.max_epochs} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

            check_loss = val_loss if val_loss is not None else train_loss
            if check_loss < best_loss - self.config.min_delta:
                best_loss = check_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.model.eval()
        self._is_fitted = True

        # 清理训练过程中的临时变量
        del X_tensor
        if val_tensor is not None:
            del val_tensor
        gc.collect()

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        use_kalman: bool = True,
        return_final_state: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Transform data to extract latent features.

        Args:
            X: Input data [n_samples, n_features] or [n_timesteps, n_features]
            use_kalman: Whether to use Kalman filtering for state estimation
            return_final_state: Whether to return the final internal state
                               (useful for syncing with real-time processor)
                               Only works when use_kalman=True

        Returns:
            If return_final_state=False:
                Latent features [n_timesteps, state_dim]
            If return_final_state=True:
                Tuple of (states, final_state_dict) where final_state_dict contains:
                - ekf_z: Final EKF state
                - ekf_P: Final EKF covariance
                - lstm_hidden: Final LSTM hidden state
                - lstm_cell: Final LSTM cell state
                - step_count: Number of steps processed
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X = X.astype(np.float32)

        if len(X.shape) == 2:
            X_scaled = self._standardize(X, fit=False)
        else:
            X_scaled = self._standardize(X[0], fit=False)

        X_tensor = torch.tensor(X_scaled, dtype=self.config.torch_dtype).to(self.device)

        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)

        if use_kalman:
            return self._kalman_filter(X_tensor, return_final_state=return_final_state)
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(X_tensor, return_all_states=True)
                states = output["states"][0].cpu().numpy()
            if return_final_state:
                # Non-Kalman mode doesn't have state to sync
                return states, None
            return states

    def _kalman_filter(
        self,
        observations: torch.Tensor,
        return_final_state: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Apply Extended Kalman Filter for state estimation.

        The EKF follows the correct predict-then-update order:
        1. Predict: Use prior p(z_t|z_{t-1}) with transition Jacobian F
        2. Update: Use observation y_t with observation Jacobian H
        3. Update LSTM hidden state with y_t (for next step's posterior if needed)

        This avoids the "double observation" problem where y_t was used both
        in LSTM (for transition) and in EKF update.

        Args:
            observations: Observations tensor [1, time, obs_dim]
            return_final_state: Whether to return the final internal state
                               (useful for syncing with real-time processor)

        Returns:
            If return_final_state=False:
                Filtered states [time, state_dim]
            If return_final_state=True:
                Tuple of (states, final_state_dict) where final_state_dict contains:
                - ekf_z: Final EKF state
                - ekf_P: Final EKF covariance
                - lstm_hidden: Final LSTM hidden state
                - lstm_cell: Final LSTM cell state
                - step_count: Number of steps processed
        """
        seq_len = observations.shape[1]

        ekf = ExtendedKalmanFilter(
            state_dim=self.config.state_dim,
            obs_dim=self.config.obs_dim,
            device=self.device,
            dtype=self.config.torch_dtype,
        )

        ekf.reset(
            initial_state=self.model.initial_state_mean.unsqueeze(0),
            initial_covariance=torch.diag(torch.exp(self.model.initial_state_log_var)),
        )

        states = []

        # Initialize LSTM hidden state
        lstm_hidden = torch.zeros(
            self.config.lstm_layers,
            1,
            self.config.lstm_hidden,
            device=self.device,
            dtype=self.config.torch_dtype,
        )
        lstm_cell = torch.zeros_like(lstm_hidden)

        self.model.eval()
        with torch.no_grad():
            for t in range(seq_len):
                y_t = observations[:, t : t + 1, :]  # [1, 1, obs_dim]

                if t == 0:
                    # t=0: Use initial state, no prediction needed
                    z = ekf.z
                else:
                    # Step 1: EKF Predict using PRIOR (does NOT depend on y_t)
                    # Get prior distribution p(z_t | z_{t-1})
                    z_prev = ekf.z.squeeze(0) if ekf.z.dim() > 1 else ekf.z
                    prior_mean, prior_log_var = self.model.get_transition_prior(z_prev)
                    trans_cov = torch.diag(torch.exp(prior_log_var.squeeze(0)))

                    # Compute transition Jacobian F = ∂prior_mean/∂z_{t-1}
                    F = self.model.compute_transition_jacobian(z_prev)

                    # EKF predict with proper Jacobian
                    z_pred, P_pred = ekf.predict(prior_mean, trans_cov, F)

                    # Step 2: EKF Update using observation y_t
                    obs_mean, obs_log_var = self.model.get_observation_dist(z_pred)
                    obs_cov = torch.diag(torch.exp(obs_log_var.squeeze(0)))

                    # Compute observation Jacobian H = ∂obs_mean/∂z
                    H = self.model.compute_observation_jacobian(z_pred)

                    # EKF update
                    obs_input = observations[:, t, :]  # [1, obs_dim]
                    z, _ = ekf.update(
                        z_pred,
                        P_pred,
                        obs_input,
                        obs_mean,
                        obs_cov,
                        H,
                    )

                # Step 3: Update LSTM with y_t (for next iteration's posterior)
                # This happens AFTER the EKF update, so y_t is not used in predict
                _, (lstm_hidden, lstm_cell) = self.model.lstm(
                    y_t, (lstm_hidden, lstm_cell)
                )

                # Collect state
                state_np = z.cpu().numpy()
                # Ensure state_np is 1D array [state_dim]
                while state_np.ndim > 1:
                    state_np = state_np.squeeze(0)
                if state_np.ndim == 0:  # Scalar case
                    state_np = state_np.reshape(1)
                states.append(state_np)

        # Check state dimension consistency
        if states:
            state_dims = [s.shape for s in states]
            if len(set(state_dims)) > 1:
                target_dim = self.config.state_dim
                fixed_states = []
                for s in states:
                    if s.shape[0] != target_dim:
                        if s.shape[0] > target_dim:
                            s = s[:target_dim]
                        else:
                            s = np.pad(s, (0, target_dim - s.shape[0]))
                    fixed_states.append(s)
                states = fixed_states

        states_array = np.array(states)

        if return_final_state:
            final_state_dict = {
                "ekf_z": ekf.z.clone(),
                "ekf_P": ekf.P.clone(),
                "lstm_hidden": lstm_hidden.clone(),
                "lstm_cell": lstm_cell.clone(),
                "step_count": seq_len,
            }
            return states_array, final_state_dict

        return states_array

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Optional[Union[np.ndarray, pd.DataFrame, torch.Tensor]] = None,
        **fit_params,
    ) -> np.ndarray:
        """
        Fit model and transform data.

        Args:
            X: Training data
            y: Ignored
            **fit_params: Additional fit parameters

        Returns:
            Transformed features
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def save(self, path: str):
        """
        Save model to disk using SafeTensors format.

        Args:
            path: Path to save model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model. Call fit() first.")

        from safetensors.torch import save_file
        import json
        from pathlib import Path

        # Ensure path doesn't have extension
        path = str(Path(path).with_suffix(""))

        # Save model weights using SafeTensors
        state_dict = self.model.state_dict()
        save_file(state_dict, f"{path}.safetensors")

        # Prepare metadata as JSON-serializable dict
        metadata = {
            "version": "2.0.0",  # New version for SafeTensors
            "config": self.config.model_dump(),
            "is_fitted": self.is_fitted,
            "training_history": self.training_history,
        }

        # Remove non-serializable items
        metadata["config"].pop("torch_dtype", None)

        # Save scaler params as pure Python types (only if scaler is used)
        if self.config.use_scaler and self.scaler_params is not None:
            metadata["scaler_mean"] = self.scaler_params["mean"].tolist()
            metadata["scaler_std"] = self.scaler_params["std"].tolist()
        else:
            metadata["scaler_mean"] = None
            metadata["scaler_std"] = None

        # Save metadata as JSON
        with open(f"{path}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {path}.safetensors and {path}.json")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DeepSSM":
        """
        Load model from disk using SafeTensors format.

        Args:
            path: Path to load model from (without extension)
            device: Device to load model to

        Returns:
            Loaded model
        """
        from safetensors.torch import load_file
        import json
        from pathlib import Path

        # Ensure path doesn't have extension
        path = str(Path(path).with_suffix(""))

        # Load metadata from JSON
        with open(f"{path}.json", "r") as f:
            metadata = json.load(f)

        config_dict = metadata["config"]
        training_history = metadata.get("training_history", [])
        is_fitted = metadata.get("is_fitted", True)

        # Reconstruct scaler params
        if metadata.get("scaler_mean") is not None:
            scaler_params = {
                "mean": np.array(metadata["scaler_mean"], dtype=np.float32),
                "std": np.array(metadata["scaler_std"], dtype=np.float32),
            }
        else:
            scaler_params = None

        if device is not None:
            config_dict["device"] = device

        config = DeepSSMConfig(**config_dict)

        # Create model instance
        model = cls(config)

        # Load weights from SafeTensors
        state_dict = load_file(f"{path}.safetensors")
        model.model.load_state_dict(state_dict)

        # Set model attributes
        model.scaler_params = scaler_params
        model.training_history = training_history
        model._is_fitted = is_fitted

        model.model.eval()

        print(f"Model loaded from {path}.safetensors and {path}.json")
        return model

    def create_realtime_processor(self) -> "DeepSSMRealTime":
        """
        Create a real-time processor for streaming data.

        Returns:
            Real-time processor instance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return DeepSSMRealTime(self)


class DeepSSMRealTime:
    """
    Real-time processor for DeepSSM model.

    This class maintains state for processing streaming data one sample at a time.
    """

    def __init__(self, parent_model: DeepSSM):
        """
        Initialize real-time processor.

        Args:
            parent_model: Trained DeepSSM model
        """
        self.model = parent_model.model
        self.config = parent_model.config
        self.scaler_params = parent_model.scaler_params
        self.device = parent_model.device

        self.model.eval()

        self.ekf = ExtendedKalmanFilter(
            state_dim=self.config.state_dim,
            obs_dim=self.config.obs_dim,
            device=self.device,
            dtype=self.config.torch_dtype,
        )

        self.ekf.reset(
            initial_state=self.model.initial_state_mean.unsqueeze(0),
            initial_covariance=torch.diag(torch.exp(self.model.initial_state_log_var)),
        )

        self.lstm_hidden = torch.zeros(
            self.config.lstm_layers,
            1,
            self.config.lstm_hidden,
            device=self.device,
            dtype=self.config.torch_dtype,
        )
        self.lstm_cell = torch.zeros_like(self.lstm_hidden)

        self.step_count = 0

    def process_single(
        self, observation: Union[np.ndarray, List, torch.Tensor]
    ) -> np.ndarray:
        """
        Process a single observation and return latent features.

        Uses the correct order: predict → update → LSTM to avoid double observation problem.
        The prior distribution p(z_t | z_{t-1}) only depends on previous state,
        not on current observation.

        Args:
            observation: Single observation [obs_dim]

        Returns:
            Latent features [state_dim]
        """
        if isinstance(observation, (list, tuple)):
            observation = np.array(observation, dtype=np.float32)
        elif isinstance(observation, torch.Tensor):
            observation = observation.cpu().numpy()

        observation = observation.astype(np.float32).reshape(1, -1)

        if self.config.use_scaler and self.scaler_params is not None:
            mean = self.scaler_params["mean"]
            std = self.scaler_params["std"]
            obs_scaled = (observation - mean) / std
        else:
            obs_scaled = observation

        obs_tensor = torch.tensor(
            obs_scaled, dtype=self.config.torch_dtype, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self.step_count == 0:
                # t=0: Use initial state, no prediction needed
                z = self.ekf.z
            else:
                # t>0: predict → update (LSTM already updated after previous step)
                z_prev = self.ekf.z

                # Step 1: EKF Predict using PRIOR (only depends on z_prev, not y_t)
                z_prev_proc = z_prev.squeeze(0) if z_prev.dim() > 1 else z_prev
                prior_mean, prior_log_var = self.model.get_transition_prior(z_prev_proc)
                prior_var = torch.exp(prior_log_var.squeeze(0))
                trans_cov = torch.diag(prior_var)

                # Compute transition Jacobian using autograd
                F = self.model.compute_transition_jacobian(z_prev_proc)

                z_pred, P_pred = self.ekf.predict(prior_mean, trans_cov, F)

                # Step 2: EKF Update using current observation
                obs_mean, obs_log_var = self.model.get_observation_dist(z_pred)
                obs_var = torch.exp(obs_log_var.squeeze(0))
                obs_cov = torch.diag(obs_var)

                # Compute observation Jacobian using autograd
                H = self.model.compute_observation_jacobian(z_pred)

                z, _ = self.ekf.update(
                    z_pred, P_pred, obs_tensor.squeeze(0), obs_mean, obs_cov, H
                )

            # Step 3: Update LSTM AFTER EKF update (prepare for next step)
            # This ensures LSTM hidden state encodes history up to current observation
            _, (self.lstm_hidden, self.lstm_cell) = self.model.lstm(
                obs_tensor, (self.lstm_hidden, self.lstm_cell)
            )

        self.step_count += 1

        return z.squeeze(0).detach().cpu().numpy()

    def reset(self):
        """Reset the processor state."""
        self.ekf.reset(
            initial_state=self.model.initial_state_mean.unsqueeze(0),
            initial_covariance=torch.diag(torch.exp(self.model.initial_state_log_var)),
        )

        self.lstm_hidden = torch.zeros(
            self.config.lstm_layers,
            1,
            self.config.lstm_hidden,
            device=self.device,
            dtype=self.config.torch_dtype,
        )
        self.lstm_cell = torch.zeros_like(self.lstm_hidden)

        self.step_count = 0

    def sync_state(self, final_state_dict: Dict) -> None:
        """
        Synchronize internal state from batch transform results.

        This method allows syncing the real-time processor state with the
        final state from a batch transform() call, ensuring continuity
        between batch and streaming processing.

        Args:
            final_state_dict: Dictionary containing final state from transform():
                - ekf_z: Final EKF state tensor
                - ekf_P: Final EKF covariance tensor
                - lstm_hidden: Final LSTM hidden state tensor
                - lstm_cell: Final LSTM cell state tensor
                - step_count: Number of steps processed
        """
        if final_state_dict is None:
            raise ValueError("final_state_dict cannot be None")

        # Sync EKF state
        self.ekf.z = final_state_dict["ekf_z"].clone().to(self.device)
        self.ekf.P = final_state_dict["ekf_P"].clone().to(self.device)

        # Sync LSTM state
        self.lstm_hidden = final_state_dict["lstm_hidden"].clone().to(self.device)
        self.lstm_cell = final_state_dict["lstm_cell"].clone().to(self.device)

        # Sync step count
        self.step_count = final_state_dict["step_count"]


if __name__ == "__main__":
    print("DeepSSM Model Test")
    print("-" * 50)

    np.random.seed(42)
    n_timesteps = 500
    n_features = 10

    t = np.linspace(0, 10 * np.pi, n_timesteps)
    X = np.zeros((n_timesteps, n_features))
    for i in range(n_features):
        X[:, i] = np.sin(t + i * np.pi / n_features) + 0.1 * np.random.randn(
            n_timesteps
        )

    train_size = int(0.8 * n_timesteps)
    X_train = X[:train_size]
    X_val = X[train_size:]

    print(f"Data shape: {X.shape}")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print()

    config = DeepSSMConfig(
        obs_dim=n_features,
        state_dim=3,
        lstm_hidden=32,
        max_epochs=30,
        patience=5,
        learning_rate=1e-3,
        seed=42,
    )

    print("Training DeepSSM model...")
    model = DeepSSM(config)
    model.fit(X_train, val_data=X_val)

    print("\nExtracting features...")
    features = model.transform(X_train)
    print(f"Features shape: {features.shape}")
    print(f"Features mean: {features.mean():.4f}, std: {features.std():.4f}")

    print("\nTesting save/load...")
    model.save("test_deep_ssm.pt")
    loaded_model = DeepSSM.load("test_deep_ssm.pt")

    features_loaded = loaded_model.transform(X_train)
    print(f"Features match after load: {np.allclose(features, features_loaded)}")

    print("\nTesting real-time processing...")
    realtime = model.create_realtime_processor()

    realtime_features = []
    for i in range(10):
        feature = realtime.process_single(X_train[i])
        realtime_features.append(feature)
        print(f"Step {i + 1}: {feature[:3].round(4)}")

    import os

    if os.path.exists("test_deep_ssm.pt"):
        os.remove("test_deep_ssm.pt")

    print("\nTest completed successfully!")
