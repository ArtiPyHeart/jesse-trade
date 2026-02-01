"""
GrootCV 内存优化相关测试

覆盖:
- shadow 特征生成（DataFrame / ndarray）
- LGB pred_contrib SHAP 重要性
- 分块 NaN 对齐逻辑
- SimpleFeatureCalculator memmap 写入
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.utils import align_features_labels
from src.features.feature_selection.groot.shadow import create_shadow_features
from src.features.feature_selection.groot.shap_utils import compute_shap_importance
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.simple_feature_calculator.registry import SimpleFeatureRegistry
from src.utils.drop_na import drop_na_and_align_x_and_y


def test_create_shadow_features_dataframe():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    combined, shadow_names = create_shadow_features(df, random_state=42)

    assert combined.shape == (3, 4)
    assert shadow_names == ["ShadowVar1", "ShadowVar2"]
    assert combined.columns.tolist() == ["a", "b", "ShadowVar1", "ShadowVar2"]
    np.testing.assert_allclose(combined[["a", "b"]].to_numpy(), df.to_numpy())

    for i, name in enumerate(shadow_names):
        shadow = combined[name].to_numpy()
        original = df.iloc[:, i].to_numpy()
        assert np.array_equal(np.sort(shadow), np.sort(original))


def test_create_shadow_features_numpy():
    arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    combined, shadow_names = create_shadow_features(arr, random_state=7)

    assert combined.shape == (3, 4)
    assert shadow_names == ["ShadowVar1", "ShadowVar2"]
    np.testing.assert_allclose(combined[:, :2], arr)

    for i in range(2):
        shadow = combined[:, 2 + i]
        original = arr[:, i]
        assert np.array_equal(np.sort(shadow), np.sort(original))


def test_compute_shap_importance_lgb_backend():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=["f0", "f1", "f2", "f3"])
    y = (X["f0"] + X["f1"] * 0.5 > 0).astype(int)

    params = {
        "objective": "binary",
        "verbosity": -1,
        "num_threads": 1,
    }
    dtrain = lgb.Dataset(X, label=y, free_raw_data=True)
    model = lgb.train(params, dtrain, num_boost_round=20)

    importance = compute_shap_importance(
        X=X,
        model=model,
        objective="binary",
        fastshap=False,
        num_class=0,
        batch_size=32,
        backend="lgb",
    )

    assert len(importance) == 4
    assert all(np.isfinite(list(importance.values())))


def test_drop_na_and_align_x_and_y_chunked():
    x = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 1.0, 2.0],
            "b": [np.nan, np.nan, 3.0, 4.0],
        }
    )
    y = pd.Series([0, 1, 2, 3])

    x_aligned, y_aligned = drop_na_and_align_x_and_y(x, y)
    assert len(x_aligned) == 2
    assert y_aligned.tolist() == [2, 3]

    x_bad = pd.DataFrame(
        {
            "a": [np.nan, 1.0, np.nan, 2.0],
            "b": [np.nan, 3.0, 4.0, 5.0],
        }
    )
    y_bad = pd.Series([0, 1, 2, 3])
    with pytest.raises(ValueError):
        drop_na_and_align_x_and_y(x_bad, y_bad)


def test_align_features_labels_chunked():
    features = pd.DataFrame(
        {
            "a": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    labels = np.array([10, 11, 12, 13, 14])

    aligned_features, aligned_labels = align_features_labels(
        features, labels, log_return_lag=1, pred_next=1, nan_chunk_bytes=256
    )

    assert len(aligned_features) == 3
    assert aligned_labels.tolist() == [11, 13, 14]


def test_compute_to_memmap_basic(tmp_path: Path):
    registry = SimpleFeatureRegistry()

    def open_feature(candles: np.ndarray, sequential: bool) -> np.ndarray:
        if sequential:
            return candles[:, 1]
        return candles[-1, 1:]

    registry.register_function("open", open_feature)
    calc = SimpleFeatureCalculator(registry=registry, load_buildin=False, verbose=False)

    candles = np.arange(60, dtype=np.float32).reshape(10, 6)
    calc.load(candles, sequential=True)

    mmap_path = tmp_path / "features.mmap"
    mmap = calc.compute_to_memmap(
        ["open"], mmap_path, dtype=np.float32, clear_cache_every=1
    )

    assert mmap.shape == (10, 1)
    assert mmap.dtype == np.float32
    np.testing.assert_allclose(mmap[:, 0], candles[:, 1])
