"""
ARD-VAE (Automatic Relevance Determination Variational Autoencoder)

自适应降维模块，使用 ARD prior 自动确定有效的 latent 维度数量。

核心思想：
1. 使用 over-complete latent space（如 512 维）
2. 对每个 latent 维度施加 ARD prior（可学习的精度参数）
3. 训练后，根据 KL 贡献自动识别 active dimensions
4. 只保留 active dimensions 作为降维结果

接口风格：sklearn fit/transform
保存格式：.safetensors (权重) + .json (配置+元数据+scaler参数)
"""

import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset

from pytorch_config import get_device, get_torch_dtype


class ARDVAEConfig(BaseModel):
    """Configuration for ARD-VAE dimensionality reduction model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 输入维度（训练时自动设置）
    input_dim: Optional[int] = None

    # Latent space 配置 - Over-complete 设计
    max_latent_dim: int = 512

    # ARD Prior 配置 (目前仅支持 gaussian)
    ard_prior_type: str = "gaussian"  # 仅支持 "gaussian"
    kl_threshold: float = 0.01  # 判断维度是否 active 的 KL 阈值
    warmup_epochs: int = 10  # KL annealing warmup 轮数

    # 网络架构
    encoder_hidden: Tuple[int, ...] = (512, 256, 128)
    decoder_hidden: Tuple[int, ...] = (128, 256, 512)
    activation: str = "relu"
    dropout: float = 0.1

    # 训练参数
    learning_rate: float = 1e-3
    max_epochs: int = 200
    batch_size: int = 64
    patience: int = 15
    min_delta: float = 1e-4

    # 正则化
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # 预处理
    use_scaler: bool = True

    # 计算设备
    device: str = "cpu"
    dtype: str = "float32"
    seed: Optional[int] = 42

    # 非序列化字段
    torch_dtype: torch.dtype = Field(default=None, exclude=True)

    @field_validator("encoder_hidden", "decoder_hidden", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """确保 tuple 类型。"""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("ard_prior_type")
    @classmethod
    def validate_prior_type(cls, v):
        """验证 ard_prior_type。

        TODO: 考虑实现 Horseshoe ARD prior
          - Horseshoe 提供更强的稀疏性（全局-局部收缩机制）
          - 适合需要激进维度剪枝的场景（如 100+ 特征 → 10-30 活跃维度）
          - 实现要点：hierarchical scales, non-centered parameterization, approximate KL
          - 参考：Carvalho et al. (2010) "The Horseshoe Estimator"
          - 不建议实现 Spike-and-Slab（金融时序噪声大，离散门控不稳定）
        """
        if v != "gaussian":
            raise NotImplementedError(
                f"Prior type '{v}' not implemented. "
                "Only 'gaussian' is currently supported."
            )
        return v

    @model_validator(mode="after")
    def set_computed_fields(self) -> "ARDVAEConfig":
        """设置计算字段。"""
        # 强制使用 CPU（遵循项目惯例）
        object.__setattr__(self, "device", get_device())
        object.__setattr__(self, "torch_dtype", get_torch_dtype(self.dtype))
        return self


class ARDVAENet(nn.Module):
    """ARD-VAE 神经网络组件。"""

    def __init__(self, config: ARDVAEConfig):
        super().__init__()
        self.config = config

        if config.input_dim is None:
            raise ValueError("input_dim must be set before building the network")

        # 选择激活函数
        if config.activation == "relu":
            act_fn = nn.ReLU
        elif config.activation == "tanh":
            act_fn = nn.Tanh
        elif config.activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU

        # Encoder: input_dim -> hidden layers -> (mu, log_var)
        encoder_layers = []
        in_dim = config.input_dim
        for hidden_dim in config.encoder_hidden:
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(config.encoder_hidden[-1], config.max_latent_dim)
        self.fc_log_var = nn.Linear(config.encoder_hidden[-1], config.max_latent_dim)

        # ARD prior parameters (learnable log-precision per dimension)
        # log_alpha_j 控制每个维度的稀疏性
        self.log_alpha = nn.Parameter(torch.zeros(config.max_latent_dim))

        # Decoder: latent_dim -> hidden layers -> input_dim
        decoder_layers = []
        in_dim = config.max_latent_dim
        for hidden_dim in config.decoder_hidden:
            decoder_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(config.decoder_hidden[-1], config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming 初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码得到 latent 分布参数。"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, -10, 10)  # 数值稳定性
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化采样。"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # 推理时使用均值

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码重建。"""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Returns:
            x_recon: 重建输出
            mu: latent 均值
            log_var: latent log 方差
            z: 采样的 latent representation
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z


class ARDVAE:
    """
    ARD-VAE (Automatic Relevance Determination Variational Autoencoder)
    for adaptive dimensionality reduction.

    自动确定有效降维维度的变分自编码器。使用 ARD prior 对每个 latent 维度
    施加稀疏性约束，训练后根据 KL 贡献自动识别 active dimensions。

    接口风格：sklearn fit/transform
    保存格式：.safetensors (权重) + .json (配置+元数据+scaler参数)

    Parameters
    ----------
    config : ARDVAEConfig, optional
        模型配置。若为 None，使用默认配置。

    Attributes
    ----------
    n_components : int
        有效降维维度（active latent dimensions 数量）。
    active_dims : np.ndarray
        Active latent dimension indices。
    kl_per_dim : np.ndarray
        每个 latent 维度的 KL 贡献。

    Examples
    --------
    >>> from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig
    >>> config = ARDVAEConfig(max_latent_dim=256)
    >>> ard_vae = ARDVAE(config)
    >>> X_reduced = ard_vae.fit_transform(X_train)
    >>> print(f"降维: {X_train.shape[1]} -> {ard_vae.n_components}")
    >>> ard_vae.save("/path/to/models", "my_model")
    >>> loaded = ARDVAE.load("/path/to/models", "my_model")
    """

    def __init__(self, config: Optional[ARDVAEConfig] = None):
        """
        初始化 ARD-VAE 模型。

        Args:
            config: ARDVAEConfig 配置对象。若为 None，使用默认配置。
        """
        if config is None:
            config = ARDVAEConfig()

        self.config = config

        # 模型和设备
        self.model: Optional[ARDVAENet] = None
        self.device = config.device

        # 训练状态
        self._is_fitted = False
        self.training_history: List[dict] = []

        # 特征信息
        self._feature_names: Optional[List[str]] = None

        # Active dimensions（训练后设置）
        self._active_dims: Optional[np.ndarray] = None
        self._kl_per_dim: Optional[np.ndarray] = None

        # Scaler 参数
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

        # 训练数据张量缓存（用于计算 active dims，训练后清理）
        self._train_tensor: Optional[torch.Tensor] = None

        # 设置随机种子
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练。"""
        return self._is_fitted

    @property
    def n_components(self) -> int:
        """有效降维维度（active latent dimensions 数量）。"""
        if self._active_dims is not None:
            return len(self._active_dims)
        return self.config.max_latent_dim

    @property
    def active_dims(self) -> np.ndarray:
        """Active latent dimension indices。训练后可用。"""
        if self._active_dims is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._active_dims

    @property
    def kl_per_dim(self) -> np.ndarray:
        """每个 latent 维度的 KL 贡献。训练后可用。值越大表示该维度越活跃。"""
        if self._kl_per_dim is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._kl_per_dim

    def _validate_input(self, X: pd.DataFrame, is_training: bool = False):
        """验证输入数据。"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with column names")

        # 检查空数据
        if X.shape[0] == 0:
            raise ValueError("Input DataFrame is empty (0 rows)")
        if X.shape[1] == 0:
            raise ValueError("Input DataFrame has no columns")

        if X.isnull().any().any():
            raise ValueError(
                "Input contains NaN values. Please handle missing data first."
            )

        # 检查 inf 值
        if np.isinf(X.values).any():
            raise ValueError("Input contains infinite values")

        if not is_training:
            # transform 时验证列名一致性
            if self._feature_names is None:
                raise ValueError("Model not fitted. Call fit() first.")

            if list(X.columns) != self._feature_names:
                missing = set(self._feature_names) - set(X.columns)
                extra = set(X.columns) - set(self._feature_names)
                raise ValueError(f"Column mismatch. Missing: {missing}, Extra: {extra}")

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """拟合并应用标准化。"""
        if not self.config.use_scaler:
            return X.astype(np.float32)

        self._scaler_mean = np.mean(X, axis=0, keepdims=True).astype(np.float32)
        self._scaler_std = np.std(X, axis=0, keepdims=True).astype(np.float32)
        # 防止除以零
        self._scaler_std[self._scaler_std < 1e-8] = 1.0

        return ((X - self._scaler_mean) / self._scaler_std).astype(np.float32)

    def _transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """应用已拟合的标准化。"""
        if not self.config.use_scaler:
            return X.astype(np.float32)

        if self._scaler_mean is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return ((X - self._scaler_mean) / self._scaler_std).astype(np.float32)

    def _build_network(self):
        """构建神经网络。"""
        self.model = ARDVAENet(self.config).to(self.device)

    def _compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 ARD-VAE 损失。

        Loss = Reconstruction Loss + beta * KL Divergence

        KL Divergence 使用 ARD prior：
            KL[q(z|x) || p(z)] = sum_j KL[N(mu_j, var_j) || N(0, 1/alpha_j)]
        """
        batch_size = x.shape[0]

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size

        # KL divergence with ARD prior
        # KL[N(mu, var) || N(0, 1/alpha)] =
        #   0.5 * (alpha * (mu^2 + var) - log(var) - 1 - log(alpha))
        # 推导：先验 p(z) = N(0, 1/alpha)，后验 q(z|x) = N(mu, var)
        # KL = 0.5 * [log(sigma_prior^2/var) + var/sigma_prior^2 + mu^2/sigma_prior^2 - 1]
        #    = 0.5 * [log(1/(alpha*var)) + alpha*var + alpha*mu^2 - 1]
        #    = 0.5 * [-log(alpha) - log(var) + alpha*(mu^2 + var) - 1]
        alpha = torch.exp(self.model.log_alpha)  # [latent_dim]
        var = torch.exp(log_var)  # [batch, latent_dim]

        kl_per_dim = 0.5 * (
            alpha * (mu.pow(2) + var) - log_var - 1 - self.model.log_alpha
        )  # [batch, latent_dim]

        kl_loss = kl_per_dim.mean(dim=0).sum()  # 对 batch 取均值，对维度求和

        # KL annealing (warmup)
        beta = min(1.0, epoch / max(1, self.config.warmup_epochs))

        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Tuple[float, float, float]:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for (batch_x,) in dataloader:
            batch_x = batch_x.to(self.device)

            optimizer.zero_grad()
            x_recon, mu, log_var, _ = self.model(batch_x)
            loss, recon_loss, kl_loss = self._compute_loss(
                batch_x, x_recon, mu, log_var, epoch
            )

            loss.backward()

            # 梯度裁剪
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float, float]:
        """验证一个 epoch。"""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                x_recon, mu, log_var, _ = self.model(batch_x)
                loss, recon_loss, kl_loss = self._compute_loss(
                    batch_x, x_recon, mu, log_var, epoch
                )

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    def _compute_active_dimensions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        训练后计算 active latent dimensions。

        使用 ARD KL 贡献选择 active dimensions：
        KL_j = 0.5 * (alpha_j * (mu_j^2 + var_j) - log(var_j) - 1 - log(alpha_j))

        策略：
        1. 对训练数据计算每个 latent 维度的平均 KL 贡献
        2. KL 贡献 > threshold 的维度被认为是 active
        3. 使用相对阈值：保留 KL > max(KL) * kl_threshold 的维度

        注：使用真正的 KL 贡献（包含学到的 log_alpha）进行选择，
        而非简单的 latent variance。这确保 ARD 先验的稀疏性信号被正确使用。
        """
        self.model.eval()

        # 分批计算避免 OOM
        kl_sum = torch.zeros(self.config.max_latent_dim, device=self.device)
        n_samples = 0

        # 复用训练张量，避免创建额外副本
        dataset = TensorDataset(self._train_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        with torch.no_grad():
            alpha = torch.exp(self.model.log_alpha)  # [latent_dim]

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                mu, log_var = self.model.encode(batch_x)
                var = torch.exp(log_var)

                # Per-dim KL: [batch, latent_dim]
                # KL[N(mu, var) || N(0, 1/alpha)] =
                #   0.5 * (alpha * (mu^2 + var) - log(var) - 1 - log(alpha))
                kl_batch = 0.5 * (
                    alpha * (mu.pow(2) + var) - log_var - 1 - self.model.log_alpha
                )
                kl_sum += kl_batch.sum(dim=0)
                n_samples += batch_x.shape[0]

        # 平均 KL per dimension
        kl_per_dim = (kl_sum / n_samples).cpu().numpy()

        # 使用相对阈值确定 active dimensions
        # 保留 KL > max(KL) * threshold 的维度
        max_kl = np.max(kl_per_dim)
        if max_kl > 0:
            threshold = max_kl * self.config.kl_threshold
            active_mask = kl_per_dim > threshold
        else:
            # 极端情况：所有维度 KL 为 0
            active_mask = np.zeros(len(kl_per_dim), dtype=bool)

        active_dims = np.where(active_mask)[0]

        # 如果没有 active dim，保留 KL 最大的维度
        if len(active_dims) == 0:
            # 至少保留 1 个维度
            active_dims = np.array([np.argmax(kl_per_dim)])

        return active_dims, kl_per_dim

    def fit(
        self,
        X: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> "ARDVAE":
        """
        训练 ARD-VAE 模型。

        Parameters
        ----------
        X : pd.DataFrame
            训练数据，必须是带列名的 DataFrame。形状 [n_samples, n_features]。
        val_data : pd.DataFrame, optional
            可选验证数据，列名需与训练数据一致。
        verbose : bool, default=True
            是否打印训练进度。

        Returns
        -------
        self
            返回自身，支持链式调用。

        Notes
        -----
        训练后自动计算每个 latent 维度的 KL 贡献，
        根据 kl_threshold 确定 active dimensions。
        """
        # 验证输入
        self._validate_input(X, is_training=True)
        self._feature_names = list(X.columns)

        # 标准化并创建张量（只创建一次，复用于训练和 active dim 计算）
        X_scaled = self._fit_scaler(X.values)
        self._train_tensor = torch.tensor(X_scaled, dtype=self.config.torch_dtype)
        del X_scaled  # 立即释放 numpy 副本
        gc.collect()

        # 构建网络
        self.config.input_dim = X.shape[1]
        self._build_network()

        # 准备数据（复用同一张量）
        train_dataset = TensorDataset(self._train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        val_loader = None
        if val_data is not None:
            # 验证验证集（列名需与训练数据一致）
            self._validate_input(val_data, is_training=False)
            val_scaled = self._transform_scaler(val_data.values)
            val_dataset = TensorDataset(
                torch.tensor(val_scaled, dtype=self.config.torch_dtype)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Early stopping
        best_loss = float("inf")
        patience_counter = 0

        # 训练循环
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss, train_recon, train_kl = self._train_epoch(
                train_loader, optimizer, epoch
            )

            # 验证
            if val_loader is not None:
                val_loss, val_recon, val_kl = self._validate_epoch(val_loader, epoch)
                monitor_loss = val_loss
            else:
                val_loss = val_recon = val_kl = None
                monitor_loss = train_loss

            # 更新学习率
            scheduler.step(monitor_loss)

            # 记录历史
            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_recon": train_recon,
                    "train_kl": train_kl,
                    "val_loss": val_loss,
                    "val_recon": val_recon,
                    "val_kl": val_kl,
                }
            )

            # 打印进度
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}/{self.config.max_epochs} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

            # Early stopping
            if monitor_loss < best_loss - self.config.min_delta:
                best_loss = monitor_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # 计算 active dimensions
        self._active_dims, self._kl_per_dim = self._compute_active_dimensions()

        # 清理训练数据张量
        del self._train_tensor
        self._train_tensor = None
        gc.collect()

        self._is_fitted = True

        if verbose:
            print(
                f"Training complete. Active dimensions: {self.n_components} / {self.config.max_latent_dim}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        将数据降维到 active latent space。

        Parameters
        ----------
        X : pd.DataFrame
            输入数据，必须是 DataFrame，列名需与训练时一致。

        Returns
        -------
        pd.DataFrame
            降维后的数据，列名为 "0", "1", "2", ...
            列数 = len(active_dims)

        Raises
        ------
        ValueError
            模型未训练或输入列名不匹配。
        TypeError
            输入不是 DataFrame。
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 验证输入
        self._validate_input(X, is_training=False)

        # 标准化
        X_scaled = self._transform_scaler(X.values)

        # 编码
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(
                X_scaled, dtype=self.config.torch_dtype, device=self.device
            )
            mu, _ = self.model.encode(X_tensor)

            # 提取 active dimensions
            Z = mu[:, self._active_dims].cpu().numpy()

        # 返回 DataFrame，列名为数字编号
        return pd.DataFrame(
            Z, index=X.index, columns=[str(i) for i in range(Z.shape[1])]
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """训练并转换数据。"""
        self.fit(X, val_data, verbose)
        # 直接使用 X 进行 transform，无需额外副本
        return self.transform(X)

    def get_dimension_importance(self) -> pd.DataFrame:
        """
        获取每个 latent 维度的重要性分析。

        Returns
        -------
        pd.DataFrame
            包含列 ["dim", "kl", "is_active", "rank"]
            kl 是每个维度的 KL 贡献，越大表示该维度越活跃。
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 排序索引（KL 贡献从大到小）
        rank_order = np.argsort(self._kl_per_dim)[::-1]
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(len(rank_order))

        df = pd.DataFrame(
            {
                "dim": np.arange(len(self._kl_per_dim)),
                "kl": self._kl_per_dim,
                "is_active": np.isin(
                    np.arange(len(self._kl_per_dim)), self._active_dims
                ),
                "rank": ranks,
            }
        )

        return df.sort_values("rank").reset_index(drop=True)

    def save(self, path: str, model_name: str) -> None:
        """
        保存模型到磁盘。

        Parameters
        ----------
        path : str
            保存目录路径。
        model_name : str
            模型名称字符串（如 "c_L5_N2"）。

        生成文件：
            - {path}/{model_name}.safetensors  # 模型权重
            - {path}/{model_name}.json         # 配置 + 元数据 + scaler参数
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model. Call fit() first.")

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        # 确保目录存在
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        base_path = save_dir / model_name

        # 1. 保存模型权重 (SafeTensors)
        state_dict = self.model.state_dict()
        save_file(state_dict, f"{base_path}.safetensors")

        # 2. 准备元数据
        # model_dump() 自动排除 exclude=True 的字段（如 torch_dtype）
        # 并将 tuple 转换为 list（JSON 兼容）
        config_dict = self.config.model_dump()

        metadata = {
            "version": "1.0.0",
            "config": config_dict,
            "is_fitted": self._is_fitted,
            "training_history": self.training_history,
            # Active dimensions info
            "active_dims": self._active_dims.tolist(),
            "kl_per_dim": self._kl_per_dim.tolist(),
            "n_components": int(self.n_components),
            # 输入特征信息
            "feature_names": self._feature_names,
            # Scaler 参数
            "scaler_mean": (
                self._scaler_mean.flatten().tolist()
                if self._scaler_mean is not None
                else None
            ),
            "scaler_std": (
                self._scaler_std.flatten().tolist()
                if self._scaler_std is not None
                else None
            ),
        }

        # 3. 保存 JSON
        with open(f"{base_path}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {base_path}.safetensors and {base_path}.json")

    @classmethod
    def load(cls, path: str, model_name: str) -> "ARDVAE":
        """
        从磁盘加载模型。

        Parameters
        ----------
        path : str
            模型目录路径。
        model_name : str
            模型名称字符串。

        Returns
        -------
        ARDVAE
            加载的 ARDVAE 实例。
        """
        base_path = Path(path) / model_name

        # 1. 加载元数据
        json_path = f"{base_path}.json"
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        # 2. 重建 config
        config_dict = metadata["config"]
        config = ARDVAEConfig(**config_dict)

        # 3. 创建模型实例
        model = cls(config)

        # 4. 构建网络
        model._build_network()

        # 5. 加载权重
        safetensors_path = f"{base_path}.safetensors"
        if not Path(safetensors_path).exists():
            raise FileNotFoundError(f"Weights file not found: {safetensors_path}")

        state_dict = load_file(safetensors_path)
        model.model.load_state_dict(state_dict)

        # 6. 恢复状态
        model._is_fitted = metadata["is_fitted"]
        model.training_history = metadata.get("training_history", [])
        model._active_dims = np.array(metadata["active_dims"])
        model._kl_per_dim = np.array(metadata["kl_per_dim"])
        model._feature_names = metadata["feature_names"]

        # 7. 恢复 scaler
        if metadata["scaler_mean"] is not None:
            model._scaler_mean = np.array(
                metadata["scaler_mean"], dtype=np.float32
            ).reshape(1, -1)
            model._scaler_std = np.array(
                metadata["scaler_std"], dtype=np.float32
            ).reshape(1, -1)

        model.model.eval()
        print(f"Model loaded from {base_path}.safetensors and {base_path}.json")
        print(f"Active dimensions: {model.n_components} / {config.max_latent_dim}")

        return model


if __name__ == "__main__":
    # 测试 ARD-VAE，包含内存监控
    import tempfile
    import tracemalloc

    import numpy as np
    import pandas as pd

    def get_memory_mb() -> float:
        """获取当前进程内存使用量 (MB)"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def print_memory(label: str, baseline_mb: float = 0.0):
        """打印内存使用情况"""
        current = get_memory_mb()
        delta = current - baseline_mb if baseline_mb > 0 else 0
        print(f"  [{label}] Memory: {current:.1f} MB (Δ {delta:+.1f} MB)")

    print("=" * 60)
    print("Testing ARDVAE with Memory Monitoring")
    print("=" * 60)

    # 启动内存追踪
    tracemalloc.start()
    baseline_mem = get_memory_mb()
    print_memory("Baseline", 0)

    # ==================== 小规模测试 ====================
    print("\n[Test 1] Small scale: 1000 samples × 100 features")
    np.random.seed(42)
    n_samples, n_features = 1000, 100

    X_informative = np.random.randn(n_samples, 10)
    X_noise = np.random.randn(n_samples, n_features - 10) * 0.1
    X = np.hstack([X_informative, X_noise]).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    print_memory("After data creation", baseline_mem)

    config = ARDVAEConfig(max_latent_dim=32, max_epochs=50, kl_threshold=0.01)
    ard_vae = ARDVAE(config)
    X_reduced = ard_vae.fit_transform(df, verbose=False)
    print_memory("After fit_transform", baseline_mem)

    print(f"  Shape: {df.shape} -> {X_reduced.shape}")
    print(f"  Active dims: {ard_vae.n_components} / {config.max_latent_dim}")

    # Save/Load 测试
    with tempfile.TemporaryDirectory() as tmpdir:
        ard_vae.save(tmpdir, "test_model")
        loaded = ARDVAE.load(tmpdir, "test_model")
        X_reduced_loaded = loaded.transform(df)
        assert X_reduced.shape == X_reduced_loaded.shape
        assert np.allclose(X_reduced.values, X_reduced_loaded.values)
        print("  Save/Load test passed!")

    # 清理
    del df, X, X_reduced, ard_vae, loaded, X_reduced_loaded
    gc.collect()
    print_memory("After cleanup", baseline_mem)

    # ==================== 中等规模测试 ====================
    print("\n[Test 2] Medium scale: 5000 samples × 1000 features")
    np.random.seed(42)
    n_samples, n_features = 5000, 1000

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    data_size_mb = X.nbytes / 1024 / 1024
    print(f"  Data size: {data_size_mb:.1f} MB")
    print_memory("After data creation", baseline_mem)

    config = ARDVAEConfig(max_latent_dim=64, max_epochs=30, kl_threshold=0.01)
    ard_vae = ARDVAE(config)

    # 监控 fit_transform 过程
    mem_before_fit = get_memory_mb()
    X_reduced = ard_vae.fit_transform(df, verbose=False)
    mem_after_fit = get_memory_mb()
    peak_during_fit = mem_after_fit - mem_before_fit

    print_memory("After fit_transform", baseline_mem)
    print(f"  Peak memory during fit: ~{peak_during_fit:.1f} MB")
    print(f"  Shape: {df.shape} -> {X_reduced.shape}")
    print(f"  Active dims: {ard_vae.n_components} / {config.max_latent_dim}")

    # 验证 transform 一致性
    X_transformed_again = ard_vae.transform(df)
    assert np.allclose(X_reduced.values, X_transformed_again.values)
    print("  Transform consistency test passed!")

    # 清理
    del df, X, X_reduced, ard_vae, X_transformed_again
    gc.collect()
    print_memory("After cleanup", baseline_mem)

    # ==================== 内存追踪报告 ====================
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "=" * 60)
    print("Memory Tracking Summary (tracemalloc)")
    print("=" * 60)
    print(f"  Current: {current / 1024 / 1024:.1f} MB")
    print(f"  Peak:    {peak / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
