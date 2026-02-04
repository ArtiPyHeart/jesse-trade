from pathlib import Path

from src.utils.feature_store import (
    hash_feature_names,
    is_feature_store_compatible,
    load_feature_store_meta,
)


def test_hash_feature_names_is_order_sensitive() -> None:
    assert hash_feature_names(["a", "b"]) != hash_feature_names(["b", "a"])


def test_load_feature_store_meta_missing_returns_none(tmp_path: Path) -> None:
    assert load_feature_store_meta(tmp_path / "missing.json") is None


def test_load_feature_store_meta_invalid_json_returns_none(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{not-json", encoding="utf-8")

    assert load_feature_store_meta(meta_path) is None


def test_is_feature_store_compatible_matches_all_fields() -> None:
    meta = {
        "n_rows": 12,
        "n_cols": 34,
        "dtype": "float32",
        "feature_hash": "feat",
        "candles_hash": "candles",
    }

    assert is_feature_store_compatible(
        meta,
        n_rows=12,
        n_cols=34,
        dtype_name="float32",
        feature_hash="feat",
        candles_hash="candles",
    )

    assert not is_feature_store_compatible(
        meta,
        n_rows=12,
        n_cols=34,
        dtype_name="float64",
        feature_hash="feat",
        candles_hash="candles",
    )
