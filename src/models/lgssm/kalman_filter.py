"""
Standard Kalman Filter for Linear Gaussian State Space Models.

This module implements the standard Kalman filter equations for
linear state space models with Gaussian noise.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """Standard Kalman Filter for linear systems.
    
    Implements the prediction and update steps of the Kalman filter
    for the linear state space model:
        z_{t+1} = A @ z_t + w_t,  w_t ~ N(0, Q)
        y_t = C @ z_t + v_t,      v_t ~ N(0, R)
    """
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
    def predict(
        self,
        z: torch.Tensor,
        P: torch.Tensor,
        A: torch.Tensor,
        Q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction step of Kalman filter.
        
        Args:
            z: Current state estimate (state_dim,) or (batch, state_dim)
            P: Current error covariance (state_dim, state_dim) or (batch, state_dim, state_dim)
            A: State transition matrix (state_dim, state_dim)
            Q: Process noise covariance (state_dim, state_dim)
            
        Returns:
            z_pred: Predicted state
            P_pred: Predicted error covariance
        """
        z_pred = torch.matmul(A, z.unsqueeze(-1)).squeeze(-1) if z.dim() == 1 else torch.matmul(A, z.unsqueeze(-1)).squeeze(-1)
        P_pred = A @ P @ A.T + Q
        
        return z_pred, P_pred
    
    def update(
        self,
        z_pred: torch.Tensor,
        P_pred: torch.Tensor,
        y: torch.Tensor,
        C: torch.Tensor,
        R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update step of Kalman filter.
        
        Args:
            z_pred: Predicted state (state_dim,) or (batch, state_dim)
            P_pred: Predicted error covariance (state_dim, state_dim)
            y: Observation (obs_dim,) or (batch, obs_dim)
            C: Observation matrix (obs_dim, state_dim)
            R: Observation noise covariance (obs_dim, obs_dim)
            
        Returns:
            z: Updated state estimate
            P: Updated error covariance
            K: Kalman gain
        """
        # Innovation
        y_pred = torch.matmul(C, z_pred.unsqueeze(-1)).squeeze(-1) if z_pred.dim() == 1 else torch.matmul(C, z_pred.unsqueeze(-1)).squeeze(-1)
        innovation = y - y_pred
        
        # Innovation covariance
        S = C @ P_pred @ C.T + R
        
        # Kalman gain using solve for numerical stability
        K = torch.linalg.solve(S.T, (P_pred @ C.T).T).T
        
        # State update
        z = z_pred + torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1) if innovation.dim() == 1 else torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        # Covariance update using Joseph form for numerical stability
        I_KC = torch.eye(self.state_dim, device=self.device, dtype=self.dtype) - K @ C
        P = I_KC @ P_pred @ I_KC.T + K @ R @ K.T
        
        return z, P, K
    
    def forward(
        self,
        y: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        z0: Optional[torch.Tensor] = None,
        P0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run full Kalman filter on a sequence.
        
        Args:
            y: Observations (T, obs_dim)
            A: State transition matrix (state_dim, state_dim)
            C: Observation matrix (obs_dim, state_dim)
            Q: Process noise covariance (state_dim, state_dim)
            R: Observation noise covariance (obs_dim, obs_dim)
            z0: Initial state (state_dim,)
            P0: Initial error covariance (state_dim, state_dim)
            
        Returns:
            states: Filtered states (T, state_dim)
            covariances: Error covariances (T, state_dim, state_dim)
            log_likelihood: Log likelihood of observations
        """
        T = y.shape[0]
        device = y.device
        dtype = y.dtype
        
        # Initialize
        if z0 is None:
            z0 = torch.zeros(self.state_dim, device=device, dtype=dtype)
        if P0 is None:
            P0 = torch.eye(self.state_dim, device=device, dtype=dtype) * 0.1
            
        states = torch.zeros(T, self.state_dim, device=device, dtype=dtype)
        covariances = torch.zeros(T, self.state_dim, self.state_dim, device=device, dtype=dtype)
        
        z = z0
        P = P0
        log_likelihood = 0.0
        
        # First observation
        states[0] = z
        covariances[0] = P
        
        # Filter through sequence
        for t in range(1, T):
            # Predict
            z_pred, P_pred = self.predict(z, P, A, Q)
            
            # Update
            z, P, K = self.update(z_pred, P_pred, y[t], C, R)
            
            states[t] = z
            covariances[t] = P
            
            # Compute log likelihood contribution
            y_pred = C @ z_pred
            innovation = y[t] - y_pred
            S = C @ P_pred @ C.T + R
            
            # Log likelihood: -0.5 * (log|S| + innovation' @ S^-1 @ innovation + k*log(2π))
            log_det_S = torch.logdet(S)
            quad_form = torch.matmul(innovation.unsqueeze(0), torch.linalg.solve(S, innovation.unsqueeze(-1))).squeeze()
            log_likelihood += -0.5 * (log_det_S + quad_form + self.obs_dim * torch.log(torch.tensor(2 * torch.pi)))
            
        return states, covariances, log_likelihood
    
    def smooth(
        self,
        states: torch.Tensor,
        covariances: torch.Tensor,
        A: torch.Tensor,
        Q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rauch-Tung-Striebel smoother for offline state estimation.
        
        Args:
            states: Filtered states (T, state_dim)
            covariances: Filtered covariances (T, state_dim, state_dim)
            A: State transition matrix (state_dim, state_dim)
            Q: Process noise covariance (state_dim, state_dim)
            
        Returns:
            smoothed_states: Smoothed states (T, state_dim)
            smoothed_covariances: Smoothed covariances (T, state_dim, state_dim)
        """
        T = states.shape[0]
        device = states.device
        dtype = states.dtype
        
        smoothed_states = torch.zeros_like(states)
        smoothed_covariances = torch.zeros_like(covariances)
        
        # Initialize with final filtered estimates
        smoothed_states[-1] = states[-1]
        smoothed_covariances[-1] = covariances[-1]
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            # Predicted quantities at t+1
            z_pred = A @ states[t]
            P_pred = A @ covariances[t] @ A.T + Q
            
            # Smoother gain
            C_smooth = torch.linalg.solve(P_pred.T, (covariances[t] @ A.T).T).T
            
            # Smoothed estimates
            smoothed_states[t] = states[t] + C_smooth @ (smoothed_states[t + 1] - z_pred)
            smoothed_covariances[t] = covariances[t] + C_smooth @ (smoothed_covariances[t + 1] - P_pred) @ C_smooth.T
            
        return smoothed_states, smoothed_covariances