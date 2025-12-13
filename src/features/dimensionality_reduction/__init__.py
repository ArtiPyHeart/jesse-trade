"""
Dimensionality Reduction Module

Provides tools for reducing high-dimensional feature spaces while preserving
important information. Designed for financial feature engineering pipelines.

Example:
    >>> from src.features.dimensionality_reduction import (
    ...     ARDVAE, ARDVAEConfig, DimensionReducerProtocol
    ... )
    >>>
    >>> # 使用 ARD-VAE 降维
    >>> config = ARDVAEConfig(max_latent_dim=256)
    >>> reducer = ARDVAE(config)
    >>> X_reduced = reducer.fit_transform(X_train)
    >>> print(f"降维: {X_train.shape[1]} -> {reducer.n_components}")
"""

from .protocol import DimensionReducerProtocol
from .ard_vae import ARDVAE, ARDVAEConfig

__all__ = [
    "DimensionReducerProtocol",
    "ARDVAE",
    "ARDVAEConfig",
]
