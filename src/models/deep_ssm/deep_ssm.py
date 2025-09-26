"""
Deep State Space Model (DeepSSM) for time series feature extraction.

This module implements a deep learning-based state space model that combines
LSTM networks with Extended Kalman Filtering for robust state estimation.
"""

import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Union, List

import numpy as np
import pandas as pd

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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .kalman_filter import ExtendedKalmanFilter, compute_jacobian_numerical


@dataclass
class DeepSSMConfig:
    """Configuration for DeepSSM model."""

    obs_dim: int = None
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

    device: str = "cpu"  # Force CPU usage
    dtype: str = "float32"
    seed: Optional[int] = 42

    def __post_init__(self):
        # Always use CPU, ignore auto detection
        self.device = get_device()  # Returns 'cpu'

        # Use helper function to get torch dtype
        self.torch_dtype = get_torch_dtype(self.dtype)


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

        self.transition = nn.Sequential(
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

    def get_transition_dist(
        self, lstm_out: torch.Tensor, z_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state transition distribution parameters.

        Args:
            lstm_out: LSTM output [batch, hidden_dim] or [hidden_dim]
            z_prev: Previous state [batch, state_dim] or [state_dim]

        Returns:
            Mean and log variance of transition distribution
        """
        if lstm_out.dim() == 1:
            lstm_out = lstm_out.unsqueeze(0)
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)

        combined = torch.cat([lstm_out, z_prev], dim=-1)
        output = self.transition(combined)
        mean, log_var = torch.split(output, self.config.state_dim, dim=-1)
        log_var = torch.clamp(log_var, -10, 10)
        return mean, log_var

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
        self.is_fitted = False

    def _standardize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardize data using stored parameters.

        Args:
            data: Input data
            fit: Whether to fit the scaler

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

        X_scaled = self._standardize(X[0], fit=True)
        X_tensor = (
            torch.tensor(X_scaled, dtype=self.config.torch_dtype)
            .unsqueeze(0)
            .to(self.device)
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

            val_scaled = self._standardize(val_data[0], fit=False)
            val_tensor = (
                torch.tensor(val_scaled, dtype=self.config.torch_dtype)
                .unsqueeze(0)
                .to(self.device)
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

            output = self.model(X_tensor)
            loss = self._compute_elbo_loss(X_tensor, output)

            loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            optimizer.step()

            train_loss = loss.item()

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
                msg = f"Epoch {epoch+1}/{self.config.max_epochs} | Train Loss: {train_loss:.4f}"
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
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.eval()
        self.is_fitted = True

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor], use_kalman: bool = True
    ) -> np.ndarray:
        """
        Transform data to extract latent features.

        Args:
            X: Input data [n_samples, n_features] or [n_timesteps, n_features]
            use_kalman: Whether to use Kalman filtering for state estimation

        Returns:
            Latent features [n_timesteps, state_dim]
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
            return self._kalman_filter(X_tensor)
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(X_tensor, return_all_states=True)
                states = output["states"][0].cpu().numpy()
            return states

    def _kalman_filter(self, observations: torch.Tensor) -> np.ndarray:
        """
        Apply Extended Kalman Filter for state estimation.

        Args:
            observations: Observations tensor [1, time, obs_dim]

        Returns:
            Filtered states [time, state_dim]
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
                y_t = observations[:, t : t + 1, :]

                lstm_out, (lstm_hidden, lstm_cell) = self.model.lstm(
                    y_t, (lstm_hidden, lstm_cell)
                )
                lstm_out = lstm_out.squeeze(1)

                if t == 0:
                    z = ekf.z
                else:
                    trans_mean, trans_log_var = self.model.get_transition_dist(
                        lstm_out.squeeze(0) if lstm_out.dim() > 1 else lstm_out,
                        ekf.z.squeeze(0) if ekf.z.dim() > 1 else ekf.z,
                    )
                    trans_cov = torch.diag(torch.exp(trans_log_var.squeeze(0)))

                    z_pred, P_pred = ekf.predict(trans_mean, trans_cov)

                    obs_mean, obs_log_var = self.model.get_observation_dist(z_pred)
                    obs_cov = torch.diag(torch.exp(obs_log_var.squeeze(0)))

                    def obs_func(z):
                        return self.model.get_observation_dist(z)[0]

                    H = compute_jacobian_numerical(obs_func, z_pred)

                    z, _ = ekf.update(
                        z_pred,
                        P_pred,
                        observations[:, t, :].unsqueeze(0),
                        obs_mean,
                        obs_cov,
                        H,
                    )

                state_np = z.cpu().numpy()
                # 确保 state_np 是 1D 数组 [state_dim]
                while state_np.ndim > 1:
                    state_np = state_np.squeeze(0)
                if state_np.ndim == 0:  # 标量的情况
                    state_np = state_np.reshape(1)
                states.append(state_np)

        # 检查所有状态维度是否一致
        if states:
            state_dims = [s.shape for s in states]
            if len(set(state_dims)) > 1:
                # 如果维度不一致，尝试修复
                target_dim = self.config.state_dim
                fixed_states = []
                for s in states:
                    if s.shape[0] != target_dim:
                        # 截断或填充
                        if s.shape[0] > target_dim:
                            s = s[:target_dim]
                        else:
                            s = np.pad(s, (0, target_dim - s.shape[0]))
                    fixed_states.append(s)
                states = fixed_states

        return np.array(states)

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
            "config": asdict(self.config),
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
        model.is_fitted = is_fitted

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
            lstm_out, (self.lstm_hidden, self.lstm_cell) = self.model.lstm(
                obs_tensor, (self.lstm_hidden, self.lstm_cell)
            )
            lstm_out = lstm_out.squeeze(1)

            if self.step_count == 0:
                z = self.ekf.z
            else:
                trans_mean, trans_log_var = self.model.get_transition_dist(
                    lstm_out, self.ekf.z
                )
                trans_cov = torch.diag(torch.exp(trans_log_var.squeeze(0)))

                z_pred, P_pred = self.ekf.predict(trans_mean, trans_cov)

                obs_mean, obs_log_var = self.model.get_observation_dist(z_pred)
                obs_cov = torch.diag(torch.exp(obs_log_var.squeeze(0)))

                def obs_func(z):
                    return self.model.get_observation_dist(z)[0]

                H = compute_jacobian_numerical(obs_func, z_pred)

                z, _ = self.ekf.update(
                    z_pred, P_pred, obs_tensor.squeeze(0), obs_mean, obs_cov, H
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
        print(f"Step {i+1}: {feature[:3].round(4)}")

    import os

    if os.path.exists("test_deep_ssm.pt"):
        os.remove("test_deep_ssm.pt")

    print("\nTest completed successfully!")
