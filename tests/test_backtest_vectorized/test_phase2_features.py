"""
Phase 2: Global feature config loading tests

Coverage:
- Collect model features and build a sorted global feature list
- Reject duplicate features inside a single features.json
"""

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backtest_no_jesse import (  # noqa: E402
    _collect_model_features,
    _load_model_config,
    _load_model_features,
)


def _write_features(model_path: Path, features: list[str]) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    with open(model_path / "features.json", "w") as f:
        json.dump(features, f)


def test_collect_model_features_dedup_sorted(tmp_path: Path):
    model_dir = tmp_path / "models"
    _write_features(model_dir / "model_a", ["b", "a", "c"])
    _write_features(model_dir / "model_b", ["c", "d"])

    model_features_map, global_features = _collect_model_features(
        model_dir, ["model_a", "model_b"]
    )

    assert model_features_map["model_a"] == ["b", "a", "c"]
    assert model_features_map["model_b"] == ["c", "d"]
    assert global_features == ["a", "b", "c", "d"]


def test_load_model_features_rejects_duplicates(tmp_path: Path):
    model_dir = tmp_path / "models"
    _write_features(model_dir / "model_dup", ["a", "a", "b"])

    with pytest.raises(ValueError, match="Duplicate features"):
        _load_model_features(model_dir, "model_dup")


def test_load_model_config_from_file(tmp_path: Path):
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "\n".join(
            [
                "def model_name_to_params(name):",
                "    return name, 1, 1, 0.5",
                "",
                "class LGBMContainer:",
                "    pass",
                "",
            ]
        )
    )

    model_name_to_params, lgbm_container_cls = _load_model_config(
        "unused.module", config_path
    )

    assert model_name_to_params("x") == ("x", 1, 1, 0.5)
    assert lgbm_container_cls.__name__ == "LGBMContainer"

