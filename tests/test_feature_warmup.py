import numpy as np
import pandas as pd
import pytest

from src.utils.feature_warmup import determine_warmup_start_idx


def test_warmup_allows_prefix_nan():
    features = pd.DataFrame(
        {
            "a": [np.nan, 1.0, 2.0],
            "b": [np.nan, 2.0, 3.0],
        }
    )
    trade_start_idx, first_valid_idx = determine_warmup_start_idx(features, 0)
    assert first_valid_idx == 1
    assert trade_start_idx == 1


def test_warmup_rejects_nan_after_first_valid():
    features = pd.DataFrame(
        {
            "a": [0.0, 1.0, np.nan],
            "b": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="after warmup"):
        determine_warmup_start_idx(features, 0)


def test_warmup_rejects_all_nan():
    features = pd.DataFrame(
        {
            "a": [np.nan, np.nan],
            "b": [np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="no fully valid rows"):
        determine_warmup_start_idx(features, 0)


def test_trade_start_respects_min_start_idx():
    features = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    trade_start_idx, first_valid_idx = determine_warmup_start_idx(features, 5)
    assert first_valid_idx == 0
    assert trade_start_idx == 5
