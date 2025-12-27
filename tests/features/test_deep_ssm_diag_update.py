"""
DeepSSM EKF diag clamp 向量化一致性测试
"""

import torch


def test_diag_update_vectorized_matches_loop():
    torch.manual_seed(0)
    batch = 3
    state_dim = 5

    P_update = torch.randn(batch, state_dim, state_dim)
    P_update = 0.5 * (P_update + P_update.transpose(-1, -2))
    P_diag = torch.diagonal(P_update, dim1=-2, dim2=-1)
    P_diag_clamped = torch.clamp(P_diag, min=1e-6)

    P_loop = P_update.clone()
    for i in range(state_dim):
        P_loop[:, i, i] = P_diag_clamped[:, i]

    P_vec = P_update.clone()
    idx = torch.arange(state_dim)
    P_vec[:, idx, idx] = P_diag_clamped

    assert torch.allclose(P_loop, P_vec)
