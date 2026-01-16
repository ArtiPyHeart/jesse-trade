"""
Verify LightGBM training preserves feature names for ARDVAE outputs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from flow_model_build import _train_final_lgbm_model  # noqa: E402


def test_train_final_lgbm_model_preserves_feature_names():
    x = pd.DataFrame(np.random.randn(20, 3), columns=["0", "1", "2"])
    y = np.random.randint(0, 2, size=20)

    params = {"objective": "binary", "verbose": -1}
    model = _train_final_lgbm_model(x, y, params, num_boost_round=5)

    assert model.feature_name() == ["0", "1", "2"]
