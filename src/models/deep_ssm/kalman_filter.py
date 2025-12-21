"""
Extended Kalman Filter implementation for DeepSSM.
"""

from typing import Callable, Tuple

import torch


def compute_jacobian_numerical(
    func: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Jacobian matrix numerically using finite differences.

    Args:
        func: Function to compute Jacobian for (input: [batch, state_dim] -> output: [batch, obs_dim])
        x: Input tensor [batch, state_dim]
        eps: Small value for finite difference

    Returns:
        Jacobian matrix [obs_dim, state_dim]
    """
    x = x.detach().clone().requires_grad_(False)
    y = func(x)

    batch_size = x.shape[0]
    state_dim = x.shape[1]
    obs_dim = y.shape[1]

    jac = torch.zeros(obs_dim, state_dim, device=x.device, dtype=x.dtype)

    for i in range(state_dim):
        x_plus = x.clone()
        x_plus[:, i] += eps

        y_plus = func(x_plus)
        grad = (y_plus - y) / eps

        if batch_size == 1:
            jac[:, i] = grad.squeeze(0)
        else:
            jac[:, i] = grad.mean(0)

    return jac


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for state estimation in nonlinear systems.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize EKF.

        Args:
            state_dim: Dimension of state vector
            obs_dim: Dimension of observation vector
            device: Torch device (CPU/GPU)
            dtype: Data type for tensors
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.z = torch.zeros(1, state_dim, device=self.device, dtype=dtype)
        self.P = torch.eye(state_dim, device=self.device, dtype=dtype) * 0.1

        self.I = torch.eye(state_dim, device=self.device, dtype=dtype)
        self.eps = 1e-6

    def reset(
        self,
        initial_state: torch.Tensor = None,
        initial_covariance: torch.Tensor = None,
    ):
        """
        Reset filter state.

        Args:
            initial_state: Initial state estimate [1, state_dim]
            initial_covariance: Initial covariance matrix [state_dim, state_dim]
        """
        if initial_state is not None:
            self.z = initial_state.to(self.device, dtype=self.dtype)
        else:
            self.z = torch.zeros(
                1, self.state_dim, device=self.device, dtype=self.dtype
            )

        if initial_covariance is not None:
            self.P = initial_covariance.to(self.device, dtype=self.dtype)
        else:
            self.P = (
                torch.eye(self.state_dim, device=self.device, dtype=self.dtype) * 0.1
            )

    def predict(
        self,
        transition_mean: torch.Tensor,
        transition_cov: torch.Tensor,
        transition_jacobian: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step of EKF.

        Args:
            transition_mean: Predicted state mean [1, state_dim]
            transition_cov: Process noise covariance [state_dim, state_dim]
            transition_jacobian: State transition Jacobian (optional) [state_dim, state_dim]

        Returns:
            Predicted state and covariance
        """
        z_pred = transition_mean

        if transition_jacobian is not None:
            F = transition_jacobian
            P_pred = F @ self.P @ F.T + transition_cov
        else:
            P_pred = self.P + transition_cov

        return z_pred, P_pred

    def update(
        self,
        z_pred: torch.Tensor,
        P_pred: torch.Tensor,
        observation: torch.Tensor,
        obs_mean: torch.Tensor,
        obs_cov: torch.Tensor,
        obs_jacobian: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update step of EKF.

        Args:
            z_pred: Predicted state [1, state_dim]
            P_pred: Predicted covariance [state_dim, state_dim]
            observation: Actual observation [1, obs_dim]
            obs_mean: Predicted observation mean [1, obs_dim]
            obs_cov: Observation noise covariance [obs_dim, obs_dim]
            obs_jacobian: Observation Jacobian [obs_dim, state_dim]

        Returns:
            Updated state and covariance
        """
        H = obs_jacobian
        R = obs_cov

        innovation = (observation - obs_mean).squeeze(0)

        S = H @ P_pred @ H.T + R
        S = S + torch.eye(self.obs_dim, device=self.device, dtype=self.dtype) * self.eps

        try:
            K = P_pred @ H.T @ torch.linalg.inv(S)
        except (RuntimeError, torch.linalg.LinAlgError):
            K = P_pred @ H.T @ torch.linalg.pinv(S)

        # 计算更新后的状态，确保维度为 [1, state_dim]
        z_update = z_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        # 确保 z_update 维度为 [1, state_dim]
        if z_update.dim() == 1:
            z_update = z_update.unsqueeze(0)
        elif z_update.dim() > 2:
            # 如果维度过多，squeeze 到 2 维
            while z_update.dim() > 2:
                z_update = z_update.squeeze(0)
        self.z = z_update

        # Covariance update using Joseph form for numerical stability:
        # P = (I - KH) @ P_pred @ (I - KH).T + K @ R @ K.T
        I_KH = self.I - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ obs_cov @ K.T

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

        return self.z, self.P

    def step(
        self,
        observation: torch.Tensor,
        transition_func: Callable,
        observation_func: Callable,
        transition_cov: torch.Tensor,
        observation_cov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step of EKF (predict + update).

        Args:
            observation: Current observation [1, obs_dim]
            transition_func: State transition function
            observation_func: Observation function
            transition_cov: Process noise covariance
            observation_cov: Observation noise covariance

        Returns:
            Updated state estimate
        """
        z_pred = transition_func(self.z)
        P_pred = self.P + transition_cov

        obs_mean = observation_func(z_pred)
        H = compute_jacobian_numerical(observation_func, z_pred)

        return self.update(z_pred, P_pred, observation, obs_mean, observation_cov, H)
