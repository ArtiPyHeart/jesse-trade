"""
DeepSSM MPS 路径的 CPU 单元测试。
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.deep_ssm.deep_ssm import DeepSSMConfig, DeepSSMNet  # noqa: E402


def _make_spd(batch: int, dim: int, jitter: float) -> torch.Tensor:
    A = torch.randn(batch, dim, dim)
    return A @ A.transpose(-1, -2) + torch.eye(dim) * jitter


def test_kalman_gain_inv_matches_solve():
    torch.manual_seed(0)
    config = DeepSSMConfig(obs_dim=5, state_dim=3, lstm_hidden=8, seed=42)
    net = DeepSSMNet(config)

    batch = 2
    obs_dim = 5
    state_dim = 3

    S = _make_spd(batch, obs_dim, jitter=1e-3)
    H_batch = torch.randn(batch, obs_dim, state_dim)
    P_pred = _make_spd(batch, state_dim, jitter=1e-3)

    K_inv = net._compute_kalman_gain(S, H_batch, P_pred, use_inv=True)

    HP = torch.bmm(H_batch, P_pred)
    K_ref = torch.linalg.solve(S, HP).transpose(-1, -2)

    max_err = (K_inv - K_ref).abs().max().item()
    assert max_err < 1e-3


def test_logdet_cholesky_raises_without_eigvalsh():
    config = DeepSSMConfig(obs_dim=3, state_dim=2, lstm_hidden=4, seed=42)
    net = DeepSSMNet(config)

    P_sym = -torch.eye(3).unsqueeze(0)
    with pytest.raises(RuntimeError):
        net._logdet_from_cholesky(
            P_sym,
            jitter=1e-4,
            clamp_min=1e-6,
            allow_eigvalsh=False,
        )


def test_logdet_cholesky_fallback_eigvalsh_cpu():
    config = DeepSSMConfig(obs_dim=3, state_dim=2, lstm_hidden=4, seed=42)
    net = DeepSSMNet(config)

    P_sym = -torch.eye(3).unsqueeze(0)
    logdet = net._logdet_from_cholesky(
        P_sym,
        jitter=1e-4,
        clamp_min=1e-6,
        allow_eigvalsh=True,
    )
    assert torch.isfinite(logdet).all()
