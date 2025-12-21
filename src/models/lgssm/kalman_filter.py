"""
Standard Kalman Filter for Linear Gaussian State Space Models.

This module implements the standard Kalman filter equations for
linear state space models with Gaussian noise.

Reference: Shumway & Stoffer, "Time Series Analysis and Its Applications"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

# Numerical jitter for stability in solve/slogdet operations
NUMERICAL_JITTER = 1e-6


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

        Single-sequence implementation (no batch dimension).

        Args:
            z: Current state estimate (state_dim,)
            P: Current error covariance (state_dim, state_dim)
            A: State transition matrix (state_dim, state_dim)
            Q: Process noise covariance (state_dim, state_dim)

        Returns:
            z_pred: Predicted state (state_dim,)
            P_pred: Predicted error covariance (state_dim, state_dim)
        """
        # State prediction: z_pred = A @ z
        z_pred = A @ z
        # Covariance prediction: P_pred = A @ P @ A^T + Q
        P_pred = A @ P @ A.T + Q

        return z_pred, P_pred
    
    def update(
        self,
        z_pred: torch.Tensor,
        P_pred: torch.Tensor,
        y: torch.Tensor,
        C: torch.Tensor,
        R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Update step of Kalman filter with NaN handling.

        Single-sequence implementation (no batch dimension).

        When observation contains NaN, skip the update step and return predicted state.
        This is the standard approach for handling missing observations in Kalman filtering.

        Args:
            z_pred: Predicted state (state_dim,)
            P_pred: Predicted error covariance (state_dim, state_dim)
            y: Observation (obs_dim,)
            C: Observation matrix (obs_dim, state_dim)
            R: Observation noise covariance (obs_dim, obs_dim)

        Returns:
            z: Updated state estimate (or predicted state if NaN)
            P: Updated error covariance (or predicted covariance if NaN)
            K: Kalman gain (or None if NaN)
        """
        # Check for NaN in observation
        if torch.isnan(y).any():
            # Skip update step when observation is missing (NaN)
            return z_pred, P_pred, None

        # Innovation (prediction error)
        y_pred = C @ z_pred  # (obs_dim,)
        innovation = y - y_pred

        # Innovation covariance with jitter for numerical stability
        S = C @ P_pred @ C.T + R
        S_stable = S + NUMERICAL_JITTER * torch.eye(
            self.obs_dim, device=self.device, dtype=self.dtype
        )

        # Kalman gain: K = P_pred @ C^T @ S^{-1}
        # Computed as solve(S^T, (P_pred @ C^T)^T)^T for stability
        K = torch.linalg.solve(S_stable.T, (P_pred @ C.T).T).T

        # State update: z = z_pred + K @ innovation
        z = z_pred + K @ innovation

        # Covariance update using Joseph form for numerical stability
        # P = (I - K @ C) @ P_pred @ (I - K @ C)^T + K @ R @ K^T
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
            # P0 = I to match prior p(z0) = N(0, I) used in ELBO
            P0 = torch.eye(self.state_dim, device=device, dtype=dtype)
            
        states = torch.zeros(T, self.state_dim, device=device, dtype=dtype)
        covariances = torch.zeros(T, self.state_dim, self.state_dim, device=device, dtype=dtype)

        z = z0
        P = P0
        log_likelihood = torch.tensor(0.0, device=device, dtype=dtype)

        # Filter through sequence - now includes first observation
        for t in range(T):
            if t == 0:
                # For the first observation, use initial state as prediction
                z_pred, P_pred = z0, P0
            else:
                # Predict from previous state
                z_pred, P_pred = self.predict(z, P, A, Q)

            # Update with current observation
            z, P, K = self.update(z_pred, P_pred, y[t], C, R)

            states[t] = z
            covariances[t] = P

            # Compute log likelihood contribution only if update was performed (K is not None)
            if K is not None:
                y_pred = C @ z_pred
                innovation = y[t] - y_pred
                S = C @ P_pred @ C.T + R
                # Add jitter for numerical stability (consistent with update step)
                S_stable = S + NUMERICAL_JITTER * torch.eye(
                    self.obs_dim, device=device, dtype=dtype
                )

                # Log likelihood: -0.5 * (log|S| + innovation' @ S^-1 @ innovation + k*log(2π))
                _, log_det_S = torch.linalg.slogdet(S_stable)
                quad_form = torch.matmul(
                    innovation.unsqueeze(0),
                    torch.linalg.solve(S_stable, innovation.unsqueeze(-1))
                ).squeeze()
                log_likelihood += -0.5 * (
                    log_det_S + quad_form
                    + self.obs_dim * torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype))
                )
            # If K is None (NaN observation), we skip the log likelihood contribution for this time step
            
        return states, covariances, log_likelihood
    
    def smooth(
        self,
        states: torch.Tensor,
        covariances: torch.Tensor,
        A: torch.Tensor,
        Q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rauch-Tung-Striebel smoother with lag-one covariance computation.

        This method computes smoothed estimates and the lag-one covariances
        required for the correct ELBO computation in variational inference.

        Reference: Shumway & Stoffer, Chapter 6

        Mathematical formulation:
            Smoother gain: J_t = P_{t|t} @ A^T @ P_{t+1|t}^{-1}
            Smoothed state: z_{t|T} = z_{t|t} + J_t @ (z_{t+1|T} - A @ z_{t|t})
            Smoothed cov: P_{t|T} = P_{t|t} + J_t @ (P_{t+1|T} - P_{t+1|t}) @ J_t^T
            Lag-one cov: P_{t+1,t|T} = P_{t+1|T} @ J_t^T

        Args:
            states: Filtered states z_{t|t} (T, state_dim)
            covariances: Filtered covariances P_{t|t} (T, state_dim, state_dim)
            A: State transition matrix (state_dim, state_dim)
            Q: Process noise covariance (state_dim, state_dim)

        Returns:
            smoothed_states: z_{t|T} (T, state_dim)
            smoothed_covariances: P_{t|T} (T, state_dim, state_dim)
            lag_one_covariances: P_{t+1,t|T} (T-1, state_dim, state_dim)
                Indexing: lag_one_covariances[t] = P_{t+1,t|T} = Cov(z_{t+1}, z_t | y_{1:T})
                Formula: P_{t+1,t|T} = P_{t+1|T} @ J_t^T (note the transpose!)
                For ELBO transition t → t+1: use lag_one_covariances[t] directly.
        """
        T = states.shape[0]
        device = states.device
        dtype = states.dtype

        # Use list-based accumulation to avoid in-place modifications
        # which would break autograd. Lists are filled in reverse order
        # and then reversed/stacked at the end.
        smoothed_states_list = []
        smoothed_cov_list = []
        J_list = []

        # Initialize: last smoothed = last filtered
        current_state = states[-1]
        current_cov = covariances[-1]
        smoothed_states_list.append(current_state)
        smoothed_cov_list.append(current_cov)

        # Backward pass: t = T-2, T-3, ..., 0
        for t in range(T - 2, -1, -1):
            # One-step prediction from t to t+1
            # P_{t+1|t} = A @ P_{t|t} @ A^T + Q
            P_pred = A @ covariances[t] @ A.T + Q

            # Add jitter for numerical stability in solve
            P_pred_stable = P_pred + NUMERICAL_JITTER * torch.eye(
                self.state_dim, device=device, dtype=dtype
            )

            # Smoother gain: J_t = P_{t|t} @ A^T @ P_{t+1|t}^{-1}
            # Computed as: solve(P_pred^T, (P_{t|t} @ A^T)^T)^T
            J_t = torch.linalg.solve(P_pred_stable.T, (covariances[t] @ A.T).T).T
            J_list.append(J_t)

            # Smoothed state: z_{t|T} = z_{t|t} + J_t @ (z_{t+1|T} - A @ z_{t|t})
            z_pred = A @ states[t]
            new_state = states[t] + J_t @ (current_state - z_pred)

            # Smoothed covariance: P_{t|T} = P_{t|t} + J_t @ (P_{t+1|T} - P_{t+1|t}) @ J_t^T
            new_cov = covariances[t] + J_t @ (current_cov - P_pred) @ J_t.T
            # Force symmetry to prevent numerical drift
            new_cov = (new_cov + new_cov.T) / 2

            smoothed_states_list.append(new_state)
            smoothed_cov_list.append(new_cov)
            current_state = new_state
            current_cov = new_cov

        # Reverse lists to get time-ordered results and stack
        smoothed_states_list.reverse()
        smoothed_cov_list.reverse()
        J_list.reverse()

        smoothed_states = torch.stack(smoothed_states_list, dim=0)
        smoothed_covariances = torch.stack(smoothed_cov_list, dim=0)
        J = torch.stack(J_list, dim=0)  # (T-1, state_dim, state_dim)

        # Compute lag-one covariances: P_{t+1,t|T} = P_{t+1|T} @ J_t^T
        # This gives Cov(z_{t+1}, z_t | y_{1:T}), the cross-covariance between
        # the state at t+1 and the state at t, conditioned on all observations.
        #
        # Derivation (Shumway & Stoffer, eq. 6.56):
        #   P_{t+1,t|T} = E[(z_{t+1} - μ_{t+1|T})(z_t - μ_{t|T})^T | y_{1:T}]
        #   From the smoothing recursion: P_{t+1,t|T} = P_{t+1|T} @ J_t^T
        #
        # This is used in ELBO for the transition term from t to t+1.
        # Vectorized: lag_one[t] = smoothed_cov[t+1] @ J[t].T
        lag_one_covariances = torch.bmm(
            smoothed_covariances[1:],  # (T-1, state_dim, state_dim)
            J.transpose(1, 2)          # (T-1, state_dim, state_dim)
        )

        return smoothed_states, smoothed_covariances, lag_one_covariances