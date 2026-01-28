from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def hash_feature_names(feature_names: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for name in feature_names:
        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def load_feature_store_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_feature_store_compatible(
    meta: dict,
    *,
    n_rows: int,
    n_cols: int,
    dtype_name: str,
    feature_hash: str,
    candles_hash: str,
) -> bool:
    return (
        meta.get("n_rows") == n_rows
        and meta.get("n_cols") == n_cols
        and meta.get("dtype") == dtype_name
        and meta.get("feature_hash") == feature_hash
        and meta.get("candles_hash") == candles_hash
    )
