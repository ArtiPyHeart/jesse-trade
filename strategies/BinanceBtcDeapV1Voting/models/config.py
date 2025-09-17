import json
from pathlib import Path

import pandas as pd

from src.models.deep_ssm.deep_ssm import DeepSSM
from src.models.lgssm import LGSSM

path_features = Path(__file__).parent / "feature_info.json"
with open(path_features) as f:
    feature_info = json.load(f)

FEAT_FRACDIFF = feature_info["fracdiff"]
FEAT_L4 = feature_info["L4"]
FEAT_L5 = feature_info["L5"]
FEAT_L6 = feature_info["L6"]


class DeepSSMContainer:
    def __init__(self):
        path_deep_ssm = Path(__file__).parent / "deep_ssm.pt"
        self.model: DeepSSM = DeepSSM.load(path_deep_ssm.resolve().as_posix())
        self.model_inference = self.model.create_realtime_processor()

    def transform(self, df):
        res = self.model.transform(df)
        df_res = pd.DataFrame(
            res, index=df.index, columns=[f"deep_ssm_{i}" for i in range(res.shape[1])]
        )
        return df_res

    def inference(self, df_one_row):
        arr = df_one_row.to_numpy()
        res = self.model_inference.process_single(arr)
        return pd.DataFrame(
            res.reshape(1, -1),
            index=df_one_row.index,
            columns=[f"deep_ssm_{i}" for i in range(len(res))],
        )


class LGSSMContainer:
    def __init__(self):
        path_lg_ssm = Path(__file__).parent / "lg_ssm.pt"
        self.model: LGSSM = LGSSM.load(path_lg_ssm.resolve().as_posix())

        self.state, self.covariance = self.model.get_initial_state()
        self.first_observation = True

    def transform(self, df):
        res = self.model.predict(df)
        df_res = pd.DataFrame(
            res, index=df.index, columns=[f"lg_ssm_{i}" for i in range(res.shape[1])]
        )
        return df_res

    def inference(self, df_one_row):
        arr = df_one_row.to_numpy()
        self.state, self.covariance = self.model.update_single(
            arr, is_first_observation=self.first_observation
        )
        if self.first_observation:
            self.first_observation = False
        df_res = pd.DataFrame(
            self.state,
            index=df_one_row.index,
            columns=[f"lg_ssm_{i}" for i in range(self.state.shape[1])],
        )
        return df_res
