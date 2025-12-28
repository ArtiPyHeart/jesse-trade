import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lgssm.kalman_filter import KalmanFilter, NUMERICAL_JITTER  # noqa: E402


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _generate_synthetic_inputs(
    T: int,
    obs_dim: int,
    state_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    y = torch.randn(T, obs_dim)
    A, C, Q, R = _generate_params(obs_dim, state_dim, seed)
    return y, A, C, Q, R


def _generate_params(
    obs_dim: int,
    state_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    A = torch.eye(state_dim) * 0.9 + 0.05 * torch.randn(state_dim, state_dim)
    C = torch.randn(obs_dim, state_dim) * 0.1
    Q = torch.diag(torch.exp(torch.randn(state_dim) - 2.0))
    R = torch.diag(torch.exp(torch.randn(obs_dim) - 2.0))
    return A, C, Q, R


def _load_real_features(
    start_date: str,
    end_date: str,
    max_features: int | None,
) -> np.ndarray:
    from research.model_pick.candle_fetch import FusionCandles, bar_container
    from src.features.pipeline.config import SSM_DEFAULT_INPUT_FEATURES
    from src.features.simple_feature_calculator import SimpleFeatureCalculator

    print("== 加载 candles ==")
    print(f"bar_container.THRESHOLD = {bar_container.THRESHOLD}")
    candle_container = FusionCandles(
        exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
    )
    candles = candle_container.get_candles(start_date, end_date)
    print(f"candles: {len(candles)}")
    print(f"time range: {candles[0, 0]} -> {candles[-1, 0]}")

    calc = SimpleFeatureCalculator(verbose=False)
    calc.load(candles, sequential=True)
    feature_names = list(SSM_DEFAULT_INPUT_FEATURES)
    if max_features is not None:
        feature_names = feature_names[:max_features]
    features_dict = calc.get(feature_names)
    features = np.column_stack([features_dict[name] for name in feature_names]).astype(
        np.float32
    )

    finite_mask = np.isfinite(features).all(axis=1)
    if not finite_mask.any():
        raise RuntimeError("No finite rows in features")
    first_valid = int(np.argmax(finite_mask))
    features = features[first_valid:]

    if not np.isfinite(features).all():
        raise RuntimeError("Features still contain NaN/Inf after trimming")

    print(f"features shape: {features.shape}")
    return features


def forward_old_like(
    kf: KalmanFilter,
    y: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = y.shape[0]
    device = y.device
    dtype = y.dtype

    z0 = torch.zeros(kf.state_dim, device=device, dtype=dtype)
    P0 = torch.eye(kf.state_dim, device=device, dtype=dtype)

    states = torch.zeros(T, kf.state_dim, device=device, dtype=dtype)
    covariances = torch.zeros(T, kf.state_dim, kf.state_dim, device=device, dtype=dtype)

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

        states[t] = z
        covariances[t] = P

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

    return states, covariances, log_likelihood


def time_it(func, repeats: int) -> list[float]:
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description="LGSSM CPU A/B benchmark.")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--start", default="2025-05-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--with-grad", action="store_true")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if args.real_data:
        _ensure_project_root()
        features = _load_real_features(args.start, args.end, args.max_features)
        y = torch.from_numpy(features)
        obs_dim = y.shape[1]
    else:
        obs_dim = args.obs_dim
        y, _, _, _, _ = _generate_synthetic_inputs(
            args.timesteps, obs_dim, args.state_dim, args.seed
        )

    kf = KalmanFilter(
        state_dim=args.state_dim,
        obs_dim=obs_dim,
        device=torch.device("cpu"),
    )

    A, C, Q, R = _generate_params(obs_dim, args.state_dim, args.seed)
    if args.with_grad:
        A.requires_grad_(True)
        C.requires_grad_(True)
        Q.requires_grad_(True)
        R.requires_grad_(True)

    with torch.no_grad():
        _, _, ll_new = kf(y, A, C, Q, R)
        _, _, ll_new_no_nan = kf(y, A, C, Q, R, assume_no_nan=True)
        ll_fast = kf.log_likelihood(y, A, C, Q, R)
        ll_fast_no_nan = kf.log_likelihood(y, A, C, Q, R, assume_no_nan=True)
        _, _, ll_old = forward_old_like(kf, y, A, C, Q, R)

    max_diff = (ll_new - ll_old).abs().item()
    fast_diff = (ll_new - ll_fast).abs().item()
    new_no_nan_diff = (ll_new - ll_new_no_nan).abs().item()
    fast_no_nan_diff = (ll_fast - ll_fast_no_nan).abs().item()
    print(f"Log-likelihood diff: {max_diff:.6e}")
    print(f"Log-likelihood fast diff: {fast_diff:.6e}")
    print(f"Forward assume_no_nan diff: {new_no_nan_diff:.6e}")
    print(f"Log-likelihood assume_no_nan diff: {fast_no_nan_diff:.6e}")
    print(f"Grad enabled: {args.with_grad}")

    def run_new() -> None:
        kf(y, A, C, Q, R)

    def run_new_no_nan() -> None:
        kf(y, A, C, Q, R, assume_no_nan=True)

    def run_old() -> None:
        forward_old_like(kf, y, A, C, Q, R)

    def run_ll_fast() -> None:
        kf.log_likelihood(y, A, C, Q, R)

    def run_ll_fast_no_nan() -> None:
        kf.log_likelihood(y, A, C, Q, R, assume_no_nan=True)

    for _ in range(args.warmup):
        run_new()
        run_new_no_nan()
        run_ll_fast()
        run_ll_fast_no_nan()
        run_old()

    new_times = time_it(run_new, args.repeats)
    new_no_nan_times = time_it(run_new_no_nan, args.repeats)
    fast_times = time_it(run_ll_fast, args.repeats)
    fast_no_nan_times = time_it(run_ll_fast_no_nan, args.repeats)
    old_times = time_it(run_old, args.repeats)

    new_avg = sum(new_times) / len(new_times)
    new_no_nan_avg = sum(new_no_nan_times) / len(new_no_nan_times)
    fast_avg = sum(fast_times) / len(fast_times)
    fast_no_nan_avg = sum(fast_no_nan_times) / len(fast_no_nan_times)
    old_avg = sum(old_times) / len(old_times)
    speedup = old_avg / new_avg if new_avg > 0 else float("inf")
    no_nan_speedup = new_avg / new_no_nan_avg if new_no_nan_avg > 0 else float("inf")
    fast_speedup = new_avg / fast_avg if fast_avg > 0 else float("inf")
    fast_no_nan_speedup = (
        fast_avg / fast_no_nan_avg if fast_no_nan_avg > 0 else float("inf")
    )

    print(
        f"New avg: {new_avg:.4f}s | Old avg: {old_avg:.4f}s | Speedup: {speedup:.2f}x"
    )
    print(
        f"Forward assume_no_nan avg: {new_no_nan_avg:.4f}s | "
        f"Speedup vs forward: {no_nan_speedup:.2f}x"
    )
    print(
        f"Log-likelihood avg: {fast_avg:.4f}s | Speedup vs forward: {fast_speedup:.2f}x"
    )
    print(
        f"Log-likelihood assume_no_nan avg: {fast_no_nan_avg:.4f}s | "
        f"Speedup vs log-likelihood: {fast_no_nan_speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
