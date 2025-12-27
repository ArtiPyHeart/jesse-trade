#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def _print_env() -> None:
    print("== Torch/MPS probe ==")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    mps_built = (
        torch.backends.mps.is_built()
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_built")
        else False
    )
    mps_available = (
        torch.backends.mps.is_available()
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available")
        else False
    )
    print(f"mps built: {mps_built}")
    print(f"mps available: {mps_available}")
    for key in ["PYTORCH_MPS_ENABLED", "PYTORCH_ENABLE_MPS_FALLBACK", "CUDA_VISIBLE_DEVICES"]:
        print(f"{key}={os.environ.get(key)}")
    print()


def _install_pytorch_config_stub() -> None:
    stub = types.ModuleType("pytorch_config")

    def _get_device() -> str:
        return "cpu"

    def _get_torch_dtype(dtype_str: str = "float32") -> torch.dtype:
        return torch.float32 if dtype_str == "float32" else torch.float64

    stub.get_device = _get_device
    stub.get_torch_dtype = _get_torch_dtype
    sys.modules["pytorch_config"] = stub


def _ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return False


def _run_deep_ssm(device: str, data: np.ndarray) -> None:
    from src.models.deep_ssm.deep_ssm import DeepSSM, DeepSSMConfig

    config = DeepSSMConfig(
        obs_dim=data.shape[1],
        state_dim=2,
        lstm_hidden=8,
        max_epochs=2,
        chunk_size=64,
        overlap=16,
        use_ekf_train=True,
        use_scaler=False,
    )
    object.__setattr__(config, "device", device)
    object.__setattr__(config, "torch_dtype", torch.float32)

    model = DeepSSM(config)
    start = time.perf_counter()
    model.fit(data, seed=42)
    elapsed = time.perf_counter() - start
    print(f"[DeepSSM][{device}] ok, elapsed={elapsed:.2f}s")


def _run_lgssm(device: str, data: np.ndarray) -> None:
    from src.models.lgssm.lgssm import LGSSM, LGSSMConfig

    config = LGSSMConfig(
        obs_dim=data.shape[1],
        state_dim=2,
        max_epochs=2,
        use_scaler=False,
    )
    object.__setattr__(config, "device", device)
    object.__setattr__(config, "torch_dtype", torch.float32)

    model = LGSSM(config)
    start = time.perf_counter()
    model.fit(data, verbose=False)
    elapsed = time.perf_counter() - start
    print(f"[LGSSM][{device}] ok, elapsed={elapsed:.2f}s")


def main() -> None:
    _print_env()
    _install_pytorch_config_stub()
    _ensure_project_root()

    rng = np.random.default_rng(0)
    data = rng.standard_normal((256, 6)).astype(np.float32)

    for device in ["cpu", "mps"]:
        if not _device_available(device):
            print(f"[skip] {device} not available")
            continue
        try:
            _run_deep_ssm(device, data)
        except Exception as exc:  # noqa: BLE001
            print(f"[DeepSSM][{device}] failed: {type(exc).__name__}: {exc}")
        try:
            _run_lgssm(device, data)
        except Exception as exc:  # noqa: BLE001
            print(f"[LGSSM][{device}] failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
