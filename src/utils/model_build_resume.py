from __future__ import annotations

from pathlib import Path


def model_artifacts_exist(model_dir: Path, model_name: str) -> bool:
    required = [
        "features.json",
        "tuning_result.json",
        f"model_{model_name}.txt",
        f"{model_name}.safetensors",
        f"{model_name}.json",
    ]
    return all((model_dir / name).exists() for name in required)
