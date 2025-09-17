"""
Linear Gaussian State Space Model (LGSSM) for time series feature extraction.

This module implements a linear state space model with variational inference
for learning latent representations of time series data.
"""

import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from .kalman_filter import KalmanFilter
except ImportError:
    from kalman_filter import KalmanFilter


@dataclass
class LGSSMConfig:
    """Configuration for LGSSM model."""

    obs_dim: int = None  # Will be set from data
    state_dim: int = 5

    # Training parameters
    learning_rate: float = 0.01
    max_epochs: int = 50
    batch_size: int = None  # Use full batch by default
    patience: int = 10
    min_delta: float = 0.001

    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Model initialization
    A_init_scale: float = 0.95  # Initialize A close to identity
    C_init_scale: float = 0.1  # Initialize C with small random values
    Q_init_scale: float = 0.1  # Initial process noise scale
    R_init_scale: float = 0.1  # Initial observation noise scale

    # Data preprocessing
    use_scaler: bool = True  # Whether to use StandardScaler for input data

    # Computation settings
    device: str = "auto"
    dtype: str = "float32"
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert string dtype to torch dtype
        self.torch_dtype = getattr(torch, self.dtype)


class LGSSM(nn.Module):
    """Linear Gaussian State Space Model with variational inference.

    State space model:
        z_{t+1} = A @ z_t + w_t,  w_t ~ N(0, Q)
        y_t = C @ z_t + v_t,      v_t ~ N(0, R)

    Parameters are learned through variational inference by maximizing the ELBO.
    """

    def __init__(self, config: LGSSMConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype

        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)

        # Model parameters (will be initialized in build())
        self.A = None  # State transition matrix
        self.C = None  # Observation matrix
        self.Q_log_diag = None  # Log diagonal of process noise covariance
        self.R_log_diag = None  # Log diagonal of observation noise covariance

        # Optional scaler parameters
        if config.use_scaler:
            self.register_buffer("scaler_mean", None)
            self.register_buffer("scaler_std", None)

        # Kalman filter
        self.kalman_filter = None

        # Training history
        self.history = {"loss": [], "val_loss": []}

    def build(self, obs_dim: int):
        """Initialize model parameters given observation dimension."""
        self.config.obs_dim = obs_dim
        state_dim = self.config.state_dim

        # Initialize parameters
        A_init = (
            torch.eye(state_dim, dtype=self.dtype, device=self.device)
            * self.config.A_init_scale
        )
        self.A = nn.Parameter(A_init)

        C_init = (
            torch.randn(obs_dim, state_dim, dtype=self.dtype, device=self.device)
            * self.config.C_init_scale
        )
        self.C = nn.Parameter(C_init)

        # Use log-scale for variance parameters to ensure positivity
        Q_log_init = torch.log(
            torch.ones(state_dim, dtype=self.dtype, device=self.device)
            * self.config.Q_init_scale
        )
        self.Q_log_diag = nn.Parameter(Q_log_init)

        R_log_init = torch.log(
            torch.ones(obs_dim, dtype=self.dtype, device=self.device)
            * self.config.R_init_scale
        )
        self.R_log_diag = nn.Parameter(R_log_init)

        # Initialize Kalman filter
        self.kalman_filter = KalmanFilter(
            state_dim=state_dim, obs_dim=obs_dim, device=self.device, dtype=self.dtype
        )

    @property
    def Q(self) -> torch.Tensor:
        """Process noise covariance matrix (diagonal)."""
        return torch.diag(torch.exp(self.Q_log_diag))

    @property
    def R(self) -> torch.Tensor:
        """Observation noise covariance matrix (diagonal)."""
        return torch.diag(torch.exp(self.R_log_diag))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input data if scaler is enabled."""
        if self.config.use_scaler and self.scaler_mean is not None:
            return (x - self.scaler_mean) / (self.scaler_std + 1e-8)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize data if scaler is enabled."""
        if self.config.use_scaler and self.scaler_mean is not None:
            return x * (self.scaler_std + 1e-8) + self.scaler_mean
        return x

    def compute_elbo(
        self,
        y: torch.Tensor,
        states: torch.Tensor,
        covariances: torch.Tensor,
        log_likelihood: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Evidence Lower Bound (ELBO) for variational inference.

        ELBO = E_q[log p(y|z)] - KL[q(z)||p(z)]

        Args:
            y: Observations (T, obs_dim)
            states: Filtered states from Kalman filter (T, state_dim)
            covariances: Filtered covariances (T, state_dim, state_dim)
            log_likelihood: Log likelihood from Kalman filter

        Returns:
            ELBO value (scalar)
        """
        T = y.shape[0]

        # The log likelihood from Kalman filter already contains p(y|z)
        # We use it directly as the reconstruction term
        reconstruction_term = log_likelihood

        # KL divergence between filtered posterior and prior
        # For linear Gaussian SSM, this can be computed analytically
        kl_divergence = 0.0

        # Initial state KL: KL[q(z0)||p(z0)]
        # Assume p(z0) = N(0, I)
        z0 = states[0]
        P0 = covariances[0]
        kl_0 = 0.5 * (
            torch.trace(P0)
            + torch.dot(z0, z0)
            - self.config.state_dim
            - torch.logdet(P0)
        )
        kl_divergence += kl_0

        # Transition KL: sum_t KL[q(z_t|z_{t-1})||p(z_t|z_{t-1})]
        for t in range(1, T):
            z_t = states[t]
            z_prev = states[t - 1]
            P_t = covariances[t]

            # Mean of prior p(z_t|z_{t-1})
            z_prior_mean = self.A @ z_prev

            # KL divergence between two Gaussians
            diff = z_t - z_prior_mean
            kl_t = 0.5 * (
                torch.trace(torch.linalg.solve(self.Q, P_t))
                + torch.dot(diff, torch.linalg.solve(self.Q, diff))
                - self.config.state_dim
                + torch.logdet(self.Q)
                - torch.logdet(P_t)
            )
            kl_divergence += kl_t

        # ELBO = likelihood - KL
        elbo = reconstruction_term - kl_divergence

        return elbo / T  # Normalize by sequence length

    def forward(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model using Kalman filtering.

        Args:
            y: Observations (T, obs_dim) or (batch, T, obs_dim)

        Returns:
            states: Filtered states
            covariances: Filtered covariances
            log_likelihood: Log likelihood of observations
        """
        # Handle batch dimension
        if y.dim() == 3:
            batch_size = y.shape[0]
            # Process each sequence in the batch
            states_list = []
            cov_list = []
            ll_list = []
            for i in range(batch_size):
                s, c, ll = self.forward(y[i])
                states_list.append(s)
                cov_list.append(c)
                ll_list.append(ll)
            return (
                torch.stack(states_list),
                torch.stack(cov_list),
                torch.stack(ll_list),
            )

        # Normalize input
        y_normalized = self.normalize(y)

        # Run Kalman filter
        states, covariances, log_likelihood = self.kalman_filter(
            y_normalized, self.A, self.C, self.Q, self.R
        )

        return states, covariances, log_likelihood

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        X_val: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        verbose: bool = True,
    ) -> "LGSSM":
        """Train the LGSSM model.

        Args:
            X_train: Training data (T, obs_dim)
            X_val: Optional validation data
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        # Convert to tensor
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).to(self.dtype)
        X_train = X_train.to(self.device)

        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(X_val, np.ndarray):
                X_val = torch.from_numpy(X_val).to(self.dtype)
            X_val = X_val.to(self.device)

        # Build model if not already built
        if self.A is None:
            self.build(X_train.shape[1])

        # Compute and store scaler parameters if enabled
        if self.config.use_scaler:
            self.scaler_mean = X_train.mean(dim=0)
            self.scaler_std = X_train.std(dim=0)

        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # Maximize ELBO
            patience=self.config.patience // 2,
            factor=0.5,
            min_lr=1e-6,
        )

        # Training loop
        best_val_elbo = -float("inf")
        patience_counter = 0
        best_state_dict = None

        for epoch in range(self.config.max_epochs):
            # Training step
            self.train()
            optimizer.zero_grad()

            # Forward pass
            states, covariances, log_likelihood = self(X_train)

            # Compute ELBO loss (negative for minimization)
            elbo = self.compute_elbo(X_train, states, covariances, log_likelihood)
            loss = -elbo  # Minimize negative ELBO

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.config.gradient_clip
                )

            optimizer.step()

            # Record training loss
            self.history["loss"].append(-loss.item())

            # Validation step
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_states, val_cov, val_ll = self(X_val)
                    val_elbo = self.compute_elbo(X_val, val_states, val_cov, val_ll)
                    self.history["val_loss"].append(val_elbo.item())

                # Learning rate scheduling
                scheduler.step(val_elbo)

                # Early stopping
                if val_elbo > best_val_elbo + self.config.min_delta:
                    best_val_elbo = val_elbo
                    patience_counter = 0
                    best_state_dict = self.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.config.max_epochs} | ELBO: {-loss.item():.4f}"
                if X_val is not None:
                    msg += f" | Val ELBO: {val_elbo.item():.4f}"
                print(msg)

        # Restore best model if validation was used
        if X_val is not None and best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        return_covariance: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate state features for input data.

        Args:
            X: Input observations (T, obs_dim)
            return_covariance: Whether to return covariance matrices

        Returns:
            states: Filtered states (T, state_dim)
            covariances: (Optional) Filtered covariances (T, state_dim, state_dim)
        """
        # Convert to tensor
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.dtype)
        X = X.to(self.device)

        # Forward pass
        self.eval()
        with torch.no_grad():
            states, covariances, _ = self(X)

        # Convert to numpy
        states_np = states.cpu().numpy()

        if return_covariance:
            covariances_np = covariances.cpu().numpy()
            return states_np, covariances_np

        return states_np

    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state and covariance for online inference.

        This method returns the same initialization used in batch processing,
        ensuring consistency between predict() and update_single().

        Returns:
            initial_state: Initial state (state_dim,) as zeros
            initial_covariance: Initial covariance (state_dim, state_dim) as 0.1 * I
        """
        if self.A is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        initial_state = np.zeros(self.config.state_dim)
        initial_covariance = np.eye(self.config.state_dim) * 0.1

        return initial_state, initial_covariance

    def update_single(
        self,
        new_observation: Union[np.ndarray, torch.Tensor],
        last_state: Optional[Union[np.ndarray, torch.Tensor]] = None,
        last_covariance: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_first_observation: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single new observation for real-time inference.

        Args:
            new_observation: New observation (obs_dim,)
            last_state: Previous state estimate (state_dim,). If None, uses initial state.
            last_covariance: Previous error covariance (state_dim, state_dim). If None, uses initial covariance.
            is_first_observation: If True, skip prediction step (for first observation in sequence)

        Returns:
            new_state: Updated state estimate (state_dim,)
            new_covariance: Updated error covariance (state_dim, state_dim)
        """
        # Use initial values if not provided
        if last_state is None or last_covariance is None:
            init_state, init_cov = self.get_initial_state()
            if last_state is None:
                last_state = init_state
            if last_covariance is None:
                last_covariance = init_cov
        # Convert to tensors
        if isinstance(new_observation, np.ndarray):
            new_observation = torch.from_numpy(new_observation).to(self.dtype)
        if isinstance(last_state, np.ndarray):
            last_state = torch.from_numpy(last_state).to(self.dtype)
        if isinstance(last_covariance, np.ndarray):
            last_covariance = torch.from_numpy(last_covariance).to(self.dtype)

        new_observation = new_observation.to(self.device)
        last_state = last_state.to(self.device)
        last_covariance = last_covariance.to(self.device)

        # Normalize observation
        new_obs_normalized = self.normalize(new_observation)

        self.eval()
        with torch.no_grad():
            if is_first_observation:
                # For first observation, use initial state as prediction (no prediction step)
                state_pred = last_state
                cov_pred = last_covariance
            else:
                # Prediction step for subsequent observations
                state_pred, cov_pred = self.kalman_filter.predict(
                    last_state, last_covariance, self.A, self.Q
                )

            # Update step with current observation
            new_state, new_covariance, _ = self.kalman_filter.update(
                state_pred, cov_pred, new_obs_normalized, self.C, self.R
            )

        return new_state.cpu().numpy(), new_covariance.cpu().numpy()

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save the model
        """
        # Save everything in state_dict format for weights_only compatibility
        state_dict = self.state_dict()

        # Add config and history as tensors to maintain weights_only compatibility
        config_dict = asdict(self.config)
        # Convert config to a format that can be saved with weights_only
        # We'll save it as a separate JSON-compatible structure

        checkpoint = {
            "model_state_dict": state_dict,
            "config_obs_dim": (
                config_dict.get("obs_dim", -1)
                if config_dict.get("obs_dim") is not None
                else -1
            ),
            "config_state_dim": config_dict["state_dim"],
            "config_learning_rate": config_dict["learning_rate"],
            "config_max_epochs": config_dict["max_epochs"],
            "config_batch_size": (
                config_dict.get("batch_size", -1)
                if config_dict.get("batch_size") is not None
                else -1
            ),
            "config_patience": config_dict["patience"],
            "config_min_delta": config_dict["min_delta"],
            "config_weight_decay": config_dict["weight_decay"],
            "config_gradient_clip": config_dict["gradient_clip"],
            "config_A_init_scale": config_dict["A_init_scale"],
            "config_C_init_scale": config_dict["C_init_scale"],
            "config_Q_init_scale": config_dict["Q_init_scale"],
            "config_R_init_scale": config_dict["R_init_scale"],
            "config_use_scaler": config_dict["use_scaler"],
            "config_device": config_dict["device"],
            "config_dtype": config_dict["dtype"],
            "config_seed": (
                config_dict.get("seed", -1)
                if config_dict.get("seed") is not None
                else -1
            ),
            "history_loss": (
                torch.tensor(self.history["loss"])
                if self.history["loss"]
                else torch.tensor([])
            ),
            "history_val_loss": (
                torch.tensor(self.history["val_loss"])
                if self.history["val_loss"]
                else torch.tensor([])
            ),
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LGSSM":
        """Load model from file.

        Args:
            path: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded LGSSM model
        """
        # Load with weights_only=True for security
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        # Recreate config from individual fields
        config_dict = {
            "obs_dim": (
                checkpoint["config_obs_dim"]
                if checkpoint["config_obs_dim"] != -1
                else None
            ),
            "state_dim": checkpoint["config_state_dim"],
            "learning_rate": checkpoint["config_learning_rate"],
            "max_epochs": checkpoint["config_max_epochs"],
            "batch_size": (
                checkpoint["config_batch_size"]
                if checkpoint["config_batch_size"] != -1
                else None
            ),
            "patience": checkpoint["config_patience"],
            "min_delta": checkpoint["config_min_delta"],
            "weight_decay": checkpoint["config_weight_decay"],
            "gradient_clip": checkpoint["config_gradient_clip"],
            "A_init_scale": checkpoint["config_A_init_scale"],
            "C_init_scale": checkpoint["config_C_init_scale"],
            "Q_init_scale": checkpoint["config_Q_init_scale"],
            "R_init_scale": checkpoint["config_R_init_scale"],
            "use_scaler": checkpoint["config_use_scaler"],
            "device": device if device is not None else checkpoint["config_device"],
            "dtype": checkpoint["config_dtype"],
            "seed": (
                checkpoint["config_seed"] if checkpoint["config_seed"] != -1 else None
            ),
        }

        config = LGSSMConfig(**config_dict)

        # Create model
        model = cls(config)

        # Build model structure
        if config.obs_dim is not None:
            model.build(config.obs_dim)

        # Load state dict
        # Filter out scaler parameters if they exist in state dict but model hasn't registered them
        state_dict = checkpoint["model_state_dict"]
        if config.use_scaler:
            # Make sure buffers are registered
            if "scaler_mean" in state_dict and model.scaler_mean is None:
                model.register_buffer("scaler_mean", state_dict["scaler_mean"])
                model.register_buffer("scaler_std", state_dict["scaler_std"])
        else:
            # Remove scaler parameters if not using scaler
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in ["scaler_mean", "scaler_std"]
            }

        model.load_state_dict(state_dict)

        # Load history
        model.history = {
            "loss": (
                checkpoint["history_loss"].tolist()
                if "history_loss" in checkpoint
                else []
            ),
            "val_loss": (
                checkpoint["history_val_loss"].tolist()
                if "history_val_loss" in checkpoint
                else []
            ),
        }

        print(f"Model loaded from {path}")
        return model
