import torch

from src.models.lgssm.kalman_filter import KalmanFilter, NUMERICAL_JITTER
from src.models.lgssm.lgssm import LGSSM, LGSSMConfig


def _manual_log_likelihood(
    kf: KalmanFilter,
    y: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    T = y.shape[0]
    device = y.device
    dtype = y.dtype

    z0 = torch.zeros(kf.state_dim, device=device, dtype=dtype)
    P0 = torch.eye(kf.state_dim, device=device, dtype=dtype)

    z = z0
    P = P0
    log_likelihood = torch.tensor(0.0, device=device, dtype=dtype)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype))

    for t in range(T):
        if t == 0:
            z_pred, P_pred = z0, P0
        else:
            z_pred, P_pred = kf.predict(z, P, A, Q)

        z, P, K = kf.update(z_pred, P_pred, y[t], C, R)

        if K is not None:
            y_pred = C @ z_pred
            innovation = y[t] - y_pred
            S = C @ P_pred @ C.T + R
            S_stable = S + NUMERICAL_JITTER * torch.eye(
                kf.obs_dim, device=device, dtype=dtype
            )

            _, log_det_S = torch.linalg.slogdet(S_stable)
            quad_form = torch.matmul(
                innovation.unsqueeze(0),
                torch.linalg.solve(S_stable, innovation.unsqueeze(-1)),
            ).squeeze()
            log_likelihood += -0.5 * (log_det_S + quad_form + kf.obs_dim * log_2pi)

    return log_likelihood


def test_kalman_filter_log_likelihood_matches_manual() -> None:
    torch.manual_seed(7)

    T = 120
    obs_dim = 6
    state_dim = 4

    kf = KalmanFilter(state_dim=state_dim, obs_dim=obs_dim, device=torch.device("cpu"))

    y = torch.randn(T, obs_dim)
    A = torch.eye(state_dim) * 0.9 + 0.05 * torch.randn(state_dim, state_dim)
    C = torch.randn(obs_dim, state_dim) * 0.1
    Q = torch.diag(torch.exp(torch.randn(state_dim) - 2.0))
    R = torch.diag(torch.exp(torch.randn(obs_dim) - 2.0))

    _, _, ll = kf(y, A, C, Q, R)
    ll_manual = _manual_log_likelihood(kf, y, A, C, Q, R)

    assert torch.allclose(ll, ll_manual, atol=1e-6, rtol=1e-6)


def test_kalman_filter_log_likelihood_fast_matches_forward() -> None:
    torch.manual_seed(11)

    T = 150
    obs_dim = 5
    state_dim = 3

    kf = KalmanFilter(state_dim=state_dim, obs_dim=obs_dim, device=torch.device("cpu"))

    y = torch.randn(T, obs_dim)
    A = torch.eye(state_dim) * 0.9 + 0.05 * torch.randn(state_dim, state_dim)
    C = torch.randn(obs_dim, state_dim) * 0.1
    Q = torch.diag(torch.exp(torch.randn(state_dim) - 2.0))
    R = torch.diag(torch.exp(torch.randn(obs_dim) - 2.0))

    _, _, ll = kf(y, A, C, Q, R)
    ll_fast = kf.log_likelihood(y, A, C, Q, R)

    assert torch.allclose(ll, ll_fast, atol=1e-6, rtol=1e-6)


def test_kalman_filter_assume_no_nan_matches_default() -> None:
    torch.manual_seed(17)

    T = 100
    obs_dim = 4
    state_dim = 2

    kf = KalmanFilter(state_dim=state_dim, obs_dim=obs_dim, device=torch.device("cpu"))

    y = torch.randn(T, obs_dim)
    A = torch.eye(state_dim) * 0.9 + 0.05 * torch.randn(state_dim, state_dim)
    C = torch.randn(obs_dim, state_dim) * 0.1
    Q = torch.diag(torch.exp(torch.randn(state_dim) - 2.0))
    R = torch.diag(torch.exp(torch.randn(obs_dim) - 2.0))

    _, _, ll_default = kf(y, A, C, Q, R)
    _, _, ll_no_nan = kf(y, A, C, Q, R, assume_no_nan=True)
    ll_fast = kf.log_likelihood(y, A, C, Q, R)
    ll_fast_no_nan = kf.log_likelihood(y, A, C, Q, R, assume_no_nan=True)

    assert torch.allclose(ll_default, ll_no_nan, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ll_fast, ll_fast_no_nan, atol=1e-6, rtol=1e-6)


def test_lgssm_forward_log_likelihood_matches_forward() -> None:
    torch.manual_seed(23)

    T = 80
    obs_dim = 4
    state_dim = 3

    config = LGSSMConfig(state_dim=state_dim, max_epochs=1, use_scaler=True, seed=23)
    model = LGSSM(config)
    model.build(obs_dim)

    y = torch.randn(T, obs_dim, dtype=model.dtype)
    model.scaler_mean = y.mean(dim=0)
    model.scaler_std = y.std(dim=0, unbiased=False).clamp(min=1e-8)

    model.eval()
    with torch.no_grad():
        _, _, ll = model(y)
        ll_fast = model._forward_log_likelihood(y)

    assert torch.allclose(ll, ll_fast, atol=1e-6, rtol=1e-6)
