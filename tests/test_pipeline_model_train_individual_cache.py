import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock


pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline_model_train_individual as pmi  # noqa: E402


@pytest.fixture(autouse=True)
def reset_ssm_cache():
    pmi._GLOBAL_SSM_PIPELINE = None
    yield
    pmi._GLOBAL_SSM_PIPELINE = None


def _make_pipeline(ssm_types):
    pipeline = MagicMock()
    config = MagicMock()
    config.ssm_types = ssm_types
    config.ssm_input_features = ["f1", "f2"]
    config.ssm_state_dim = 5
    pipeline.config = config
    pipeline.ssm_processors = {ssm_type: MagicMock() for ssm_type in ssm_types}
    pipeline.is_fitted = True
    pipeline.copy_ssm_from = MagicMock()
    return pipeline


def test_reuse_cached_ssm_when_compatible():
    cache = _make_pipeline(["deep_ssm", "lg_ssm"])
    pmi._GLOBAL_SSM_PIPELINE = cache

    target = _make_pipeline(["deep_ssm"])
    reused = pmi._maybe_reuse_cached_ssm(target, "demo")

    assert reused is True
    target.copy_ssm_from.assert_called_once_with(cache, ssm_types=["deep_ssm"])


def test_cache_updates_missing_types():
    cache = _make_pipeline(["deep_ssm"])
    pmi._GLOBAL_SSM_PIPELINE = cache

    source = _make_pipeline(["deep_ssm", "lg_ssm"])
    pmi._maybe_cache_ssm_pipeline(source, "demo")

    cache.copy_ssm_from.assert_called_once_with(source, ssm_types=["lg_ssm"])
