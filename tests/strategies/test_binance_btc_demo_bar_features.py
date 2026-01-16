"""
Strategy helper tests for BinanceBtcDemoBar

Coverage:
- features.json loading/validation
- global feature collection and sorting
- LGBM feature alignment
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from strategies.BinanceBtcDemoBar import (  # noqa: E402
    _align_lgbm_feature_columns,
    _collect_model_features,
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


def test_align_lgbm_feature_columns_reorders():
    df = pd.DataFrame({"b": [1.0], "a": [2.0]})
    aligned = _align_lgbm_feature_columns(df, ["a", "b"])

    assert list(aligned.columns) == ["a", "b"]
    assert aligned.iloc[0].tolist() == [2.0, 1.0]

