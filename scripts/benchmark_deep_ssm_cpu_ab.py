#!/usr/bin/env python3
"""
DeepSSM CPU A/B 基准测试（逐项对比可能的优化方案）。

注意：脚本需在项目根目录运行，以确保 jesse 能读取 .env。
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.func import jacrev

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from src.models.deep_ssm.deep_ssm import DeepSSMConfig, DeepSSMNet


def _ensure_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSSM CPU A/B benchmark")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-02-05")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=5)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def _load_features(start_date: str, end_date: str) -> np.ndarray:
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
    features_dict = calc.get(feature_names)
    features = np.column_stack(
        [features_dict[name] for name in feature_names]
    ).astype(np.float32)

    finite_mask = np.isfinite(features).all(axis=1)
    if not finite_mask.any():
        raise RuntimeError("No finite rows in features")
    first_valid = int(np.argmax(finite_mask))
    features = features[first_valid:]

    if not np.isfinite(features).all():
        raise RuntimeError("Features still contain NaN/Inf after trimming")

    print(f"features shape: {features.shape}")
    return features


def _build_config(args: argparse.Namespace, obs_dim: int) -> DeepSSMConfig:
    from src.models.deep_ssm.deep_ssm import DeepSSMConfig

    config = DeepSSMConfig(
        obs_dim=obs_dim,
        state_dim=args.state_dim,
        lstm_hidden=args.lstm_hidden,
        max_epochs=args.epochs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_ekf_train=True,
        use_scaler=True,
        seed=args.seed,
    )
    return config


class _NoOpContext:
    def __enter__(self) -> "_NoOpContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _PatchSolveCholesky:
    def __init__(self) -> None:
        self._orig_solve = None

    def __enter__(self) -> "_PatchSolveCholesky":
        self._orig_solve = torch.linalg.solve
        torch.linalg.solve = self._solve_with_cholesky
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._orig_solve is not None:
            torch.linalg.solve = self._orig_solve
        return False

    def _solve_with_cholesky(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        try:
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(B, L)
        except RuntimeError:
            if self._orig_solve is None:
                raise
            return self._orig_solve(A, B)


class _PatchJacobiansBatch1:
    def __init__(self, net: torch.nn.Module) -> None:
        self._net = net
        self._orig_trans = net.compute_transition_jacobian
        self._orig_obs = net.compute_observation_jacobian
        self._trans_jacobian_fn = None
        self._obs_jacobian_fn = None

    def __enter__(self) -> "_PatchJacobiansBatch1":
        self._net.compute_transition_jacobian = self._patched_transition_jacobian
        self._net.compute_observation_jacobian = self._patched_observation_jacobian
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._net.compute_transition_jacobian = self._orig_trans
        self._net.compute_observation_jacobian = self._orig_obs
        return False

    def _patched_transition_jacobian(
        self, z_prev: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)
        if z_prev.shape[0] != 1:
            return self._orig_trans(z_prev, create_graph)

        if self._trans_jacobian_fn is None:
            def prior_mean_fn(z: torch.Tensor) -> torch.Tensor:
                mean, _ = self._net.get_transition_prior(z.unsqueeze(0))
                return mean.squeeze(0)

            self._trans_jacobian_fn = jacrev(
                prior_mean_fn, argnums=0, has_aux=False
            )

        z_prev_detached = z_prev.detach().squeeze(0)
        was_training = self._net.transition_prior.training
        self._net.transition_prior.eval()
        try:
            F = self._trans_jacobian_fn(z_prev_detached)
        finally:
            self._net.transition_prior.train(was_training)
        return F

    def _patched_observation_jacobian(
        self, z: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] != 1:
            return self._orig_obs(z, create_graph)

        if self._obs_jacobian_fn is None:
            def obs_mean_fn(z_in: torch.Tensor) -> torch.Tensor:
                mean, _ = self._net.get_observation_dist(z_in.unsqueeze(0))
                return mean.squeeze(0)

            self._obs_jacobian_fn = jacrev(
                obs_mean_fn, argnums=0, has_aux=False
            )

        z_detached = z.detach().squeeze(0)
        was_training = self._net.observation.training
        self._net.observation.eval()
        try:
            H = self._obs_jacobian_fn(z_detached)
        finally:
            self._net.observation.train(was_training)
        return H


class _NoOpLSTM(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        batch, seq_len = x.shape[0], x.shape[1]
        hidden_size = h.shape[-1]
        out = torch.zeros(
            batch, seq_len, hidden_size, device=x.device, dtype=x.dtype
        )
        return out, (h, c)


class _PatchNoOpLSTM:
    def __init__(self, net: torch.nn.Module) -> None:
        self._net = net
        self._orig_lstm = net.lstm

    def __enter__(self) -> "_PatchNoOpLSTM":
        self._net.lstm = _NoOpLSTM()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._net.lstm = self._orig_lstm
        return False


def _run_variant(
    name: str,
    features: np.ndarray,
    config: DeepSSMConfig,
    patcher_factory: Optional[Callable[[DeepSSMNet], object]],
    patch_transform: bool,
    repeat: int,
) -> dict:
    results = []
    states = None
    final_loss = None

    for _ in range(repeat):
        _set_seed(config.seed)
        gc.collect()

        from src.models.deep_ssm.deep_ssm import DeepSSM

        model = DeepSSM(config)
        net = model.model

        if patcher_factory is None:
            patch_ctx = _NoOpContext()
        else:
            patch_ctx = patcher_factory(net)

        start = time.perf_counter()
        with patch_ctx:
            model.fit(features, seed=config.seed)
            elapsed = time.perf_counter() - start
            final_loss = model.training_history[-1]["train_loss"]
            if patch_transform:
                states = model.transform(features)

        if not patch_transform:
            states = model.transform(features)

        results.append(elapsed)

    avg_elapsed = float(np.mean(results))
    return {
        "name": name,
        "elapsed": avg_elapsed,
        "final_loss": float(final_loss) if final_loss is not None else None,
        "states": states,
    }


def _compare_states(base: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(base - other)))
    corr = float(np.corrcoef(base.reshape(-1), other.reshape(-1))[0, 1])
    return mae, corr


def _microbench_diag_update(state_dim: int, batch: int, iters: int) -> None:
    P = torch.randn(batch, state_dim, state_dim)
    P = P @ P.transpose(-1, -2)
    P_diag = torch.diagonal(P, dim1=-2, dim2=-1)
    P_diag_clamped = torch.clamp(P_diag, min=1e-6)

    start = time.perf_counter()
    for _ in range(iters):
        P_update = P.clone()
        for i in range(state_dim):
            P_update[:, i, i] = P_diag_clamped[:, i]
    loop_ms = (time.perf_counter() - start) * 1e3

    idx = torch.arange(state_dim)
    start = time.perf_counter()
    for _ in range(iters):
        P_update = P.clone()
        P_update[:, idx, idx] = P_diag_clamped
    vector_ms = (time.perf_counter() - start) * 1e3

    print(f"diag update loop: {loop_ms:.2f} ms")
    print(f"diag update vectorized: {vector_ms:.2f} ms")


def _microbench_eye_cache(obs_dim: int, batch: int, iters: int) -> None:
    S = torch.randn(batch, obs_dim, obs_dim)
    S = S @ S.transpose(-1, -2)
    jitter = 1e-4

    start = time.perf_counter()
    for _ in range(iters):
        _ = S + jitter * torch.eye(obs_dim)
    loop_ms = (time.perf_counter() - start) * 1e3

    eye = torch.eye(obs_dim)
    start = time.perf_counter()
    for _ in range(iters):
        _ = S + jitter * eye
    cached_ms = (time.perf_counter() - start) * 1e3

    print(f"eye create each step: {loop_ms:.2f} ms")
    print(f"eye cached reuse: {cached_ms:.2f} ms")


def main() -> None:
    _ensure_project_root()
    args = _parse_args()

    print("== DeepSSM CPU A/B benchmark ==")
    print(
        f"start={args.start}, end={args.end}, epochs={args.epochs}, "
        f"state_dim={args.state_dim}, lstm_hidden={args.lstm_hidden}, "
        f"chunk_size={args.chunk_size}, overlap={args.overlap}, repeat={args.repeat}"
    )

    features = _load_features(args.start, args.end)
    config = _build_config(args, obs_dim=features.shape[1])

    print("\n== Baseline ==")
    baseline = _run_variant(
        name="baseline",
        features=features,
        config=config,
        patcher_factory=None,
        patch_transform=False,
        repeat=args.repeat,
    )
    base_states = baseline["states"]
    print(
        f"baseline elapsed={baseline['elapsed']:.2f}s, "
        f"final_loss={baseline['final_loss']:.6f}"
    )
    print(
        f"baseline states mean/std: "
        f"{base_states.mean():.6f}/{base_states.std():.6f}"
    )

    def _make_cholesky_patcher(net: DeepSSMNet) -> _PatchSolveCholesky:
        return _PatchSolveCholesky()

    def _make_jacobian_patcher(net: DeepSSMNet) -> _PatchJacobiansBatch1:
        return _PatchJacobiansBatch1(net)

    def _make_lstm_patcher(net: DeepSSMNet) -> _PatchNoOpLSTM:
        return _PatchNoOpLSTM(net)

    variants = [
        ("cholesky_solve", _make_cholesky_patcher, False),
        ("jacobian_batch1", _make_jacobian_patcher, True),
        ("skip_lstm_update", _make_lstm_patcher, False),
    ]

    for name, patcher, patch_transform in variants:
        print(f"\n== {name} ==")
        result = _run_variant(
            name=name,
            features=features,
            config=config,
            patcher_factory=patcher,
            patch_transform=patch_transform,
            repeat=args.repeat,
        )
        states = result["states"]
        mae, corr = _compare_states(base_states, states)
        print(
            f"{name} elapsed={result['elapsed']:.2f}s, "
            f"final_loss={result['final_loss']:.6f}"
        )
        print(f"{name} states mean/std: {states.mean():.6f}/{states.std():.6f}")
        print(f"{name} states MAE={mae:.6f}, corr={corr:.6f}")

    print("\n== Microbench: diag update & eye cache ==")
    _microbench_diag_update(state_dim=args.state_dim, batch=1, iters=2000)
    _microbench_eye_cache(obs_dim=features.shape[1], batch=1, iters=2000)


if __name__ == "__main__":
    main()
