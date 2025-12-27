"""
DeepSSM Jacobian batch=1 优化一致性测试
"""

import sys
from pathlib import Path

import torch
from torch.func import jacrev, vmap

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.deep_ssm.deep_ssm import DeepSSMConfig, DeepSSMNet  # noqa: E402


def _transition_jacobian_vmap(net: DeepSSMNet, z_prev: torch.Tensor) -> torch.Tensor:
    if z_prev.dim() == 1:
        z_prev = z_prev.unsqueeze(0)
    z_prev_detached = z_prev.detach()

    def prior_mean_fn(z: torch.Tensor) -> torch.Tensor:
        mean, _ = net.get_transition_prior(z.unsqueeze(0))
        return mean.squeeze(0)

    jacobian_fn = jacrev(prior_mean_fn, argnums=0, has_aux=False)
    F = vmap(jacobian_fn)(z_prev_detached)
    return F.squeeze(0)


def _observation_jacobian_vmap(net: DeepSSMNet, z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z_detached = z.detach()

    def obs_mean_fn(z_in: torch.Tensor) -> torch.Tensor:
        mean, _ = net.get_observation_dist(z_in.unsqueeze(0))
        return mean.squeeze(0)

    jacobian_fn = jacrev(obs_mean_fn, argnums=0, has_aux=False)
    H = vmap(jacobian_fn)(z_detached)
    return H.squeeze(0)


def test_transition_jacobian_batch1_matches_vmap():
    torch.manual_seed(42)
    config = DeepSSMConfig(obs_dim=10, state_dim=5, lstm_hidden=16, seed=42)
    net = DeepSSMNet(config)
    net.transition_prior.eval()

    z_prev = torch.randn(1, config.state_dim)
    F_new = net.compute_transition_jacobian(z_prev)
    F_ref = _transition_jacobian_vmap(net, z_prev)

    assert torch.allclose(F_new, F_ref, atol=1e-6, rtol=1e-5)


def test_observation_jacobian_batch1_matches_vmap():
    torch.manual_seed(123)
    config = DeepSSMConfig(obs_dim=12, state_dim=4, lstm_hidden=16, seed=123)
    net = DeepSSMNet(config)
    net.observation.eval()

    z = torch.randn(1, config.state_dim)
    H_new = net.compute_observation_jacobian(z)
    H_ref = _observation_jacobian_vmap(net, z)

    assert torch.allclose(H_new, H_ref, atol=1e-6, rtol=1e-5)
