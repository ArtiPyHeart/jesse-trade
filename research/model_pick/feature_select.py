import pandas as pd

from src.features.feature_selection.rfcq_selector import RFCQSelector
from src.models.deep_ssm import DeepSSMConfig, DeepSSM
from src.models.lgssm import LGSSMConfig, LGSSM
from .features import ALL_FEATS

FRAC_FEATS = [i for i in ALL_FEATS if i.startswith("frac_") and i.endswith("_diff")]
deep_ssm_config = DeepSSMConfig(
    obs_dim=len(FRAC_FEATS),
)
lg_ssm_config = LGSSMConfig(
    obs_dim=len(FRAC_FEATS),
)


class FeatureSelector:
    def __init__(self):
        self.deep_ssm_model = DeepSSM(config=deep_ssm_config)
        self.lg_ssm_model = LGSSM(config=lg_ssm_config)

    @property
    def selector(self):
        return RFCQSelector(verbose=False)

    def fit(self, train_x):
        if not self.deep_ssm_model.is_fitted:
            self.deep_ssm_model.fit(train_x[FRAC_FEATS])
        if not self.lg_ssm_model.is_fitted:
            self.lg_ssm_model.fit(train_x[FRAC_FEATS])

    def get_deep_ssm_features(self, train_x):
        feat_deep_ssm = self.deep_ssm_model.transform(train_x[FRAC_FEATS])
        df_deep_ssm = pd.DataFrame(
            feat_deep_ssm,
            columns=[f"deep_ssm_{i}" for i in range(feat_deep_ssm.shape[1])],
            index=train_x.index,
        )
        return df_deep_ssm

    def get_lg_ssm_features(self, train_x):
        feat_lg_ssm = self.lg_ssm_model.transform(train_x[FRAC_FEATS])
        df_lg_ssm = pd.DataFrame(
            feat_lg_ssm,
            columns=[f"lg_ssm_{i}" for i in range(feat_lg_ssm.shape[1])],
            index=train_x.index,
        )
        return df_lg_ssm

    def get_all_features(self, train_x):
        self.fit(train_x)
        df_deep_ssm = self.get_deep_ssm_features(train_x)
        lg_ssm_features = self.get_lg_ssm_features(train_x)
        df = pd.concat([df_deep_ssm, lg_ssm_features, train_x], axis=1)
        return df

    def select_features(self, train_x, train_y) -> list[str]:
        _selector = self.selector
        df_feat = self.get_all_features(train_x)
        _selector.fit(df_feat, train_y)
        res = pd.Series(_selector.relevance_, index=_selector.variables_).sort_values(
            ascending=False
        )
        feature_names = res[res > 0].index.tolist()
        return feature_names
