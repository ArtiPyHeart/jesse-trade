import numpy as np
import pandas as pd

from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.simple_feature_calculator.buildin.feature_names import (
    BUILDIN_FEATURES,
)

WINDOW = 20
BASIC = ["bar_open_dt", "bar_high_dt", "bar_low_dt", "bar_close_dt"]

basic_hurst_feats = [f"{i}_hurst{WINDOW}" for i in BASIC]
basic_curv_feats = [f"{i}_curv{WINDOW}" for i in BASIC]
basic_phent_feats = [f"{i}_phent{WINDOW}" for i in BASIC]

mean_feats = [f"{i}_mean{WINDOW}" for i in BUILDIN_FEATURES]
std_feats = [f"{i}_std{WINDOW}" for i in BUILDIN_FEATURES]
hurst_feats = [f"{i}_hurst{WINDOW}" for i in BUILDIN_FEATURES]
curv_feats = [f"{i}_curv{WINDOW}" for i in BUILDIN_FEATURES]
phent_feats = [f"{i}_phent{WINDOW}" for i in BUILDIN_FEATURES]

dt_feats = [f"{i}_dt" for i in BUILDIN_FEATURES]
ddt_feats = [f"{i}_ddt" for i in BUILDIN_FEATURES]

feats = (
    BUILDIN_FEATURES
    + basic_hurst_feats
    + basic_curv_feats
    + basic_phent_feats
    + mean_feats
    + std_feats
    + hurst_feats
    + curv_feats
    + dt_feats
    + ddt_feats
)

lag_feats = [f"{i}_lag{l}" for i in feats for l in range(1, 6)]

ALL_FEATS = feats + phent_feats + lag_feats


class FeatureLoader:
    def __init__(self, candles: np.ndarray):
        self.feature_calculator_seq = SimpleFeatureCalculator()
        self.feature_calculator_seq.load(candles, sequential=True)

        self._features = {}
        self._candles_index = candles[:, 0].astype(int)

    @property
    def features(self):
        if not self._features:
            self._features = self.feature_calculator_seq.get(ALL_FEATS)
        return self._features

    def get_feature_label_bundle(
        self, label: np.ndarray, pred_next: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        df_features = pd.DataFrame.from_dict(self.features)
        df_features.index = self._candles_index
        len_gap = len(df_features) - len(label) - pred_next
        df_features = df_features.iloc[len_gap:-pred_next]
        assert len(df_features) == len(label)

        max_na_len = df_features.isna().sum().max()
        df_features = df_features.iloc[max_na_len:]
        label = label[max_na_len:]
        assert len(df_features) == len(label)

        return df_features, label
