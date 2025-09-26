import json
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.deep_ssm.deep_ssm import DeepSSM
from src.models.lgssm import LGSSM

path_features = Path(__file__).parent / "feature_info.json"
with open(path_features) as f:
    feature_info = json.load(f)

FEAT_FRACDIFF = feature_info["fracdiff"]
ALL_RAW_FEAT = set()
for k, v in feature_info.items():
    if not k.startswith("r_"):
        ALL_RAW_FEAT.update(v)
ALL_RAW_FEAT = [
    i
    for i in ALL_RAW_FEAT
    if not i.startswith("deep_ssm") and not i.startswith("lg_ssm")
]


class SSMContainer:
    """通用的状态空间模型容器，支持 DeepSSM 和 LGSSM"""

    def __init__(self, model_type: str = "deep_ssm"):
        """
        初始化 SSM 容器

        Args:
            model_type: 模型类型，"deep_ssm" 或 "lg_ssm"
        """
        self.model_type = model_type
        self.prefix = model_type  # 用于列名前缀

        # 加载模型
        model_path = Path(__file__).parent / model_type
        if model_type == "deep_ssm":
            self.model = DeepSSM.load(model_path.resolve().as_posix(), device="cpu")
            # DeepSSM 使用专门的实时处理器
            self.model_inference = self.model.create_realtime_processor()
        elif model_type == "lg_ssm":
            self.model = LGSSM.load(model_path.resolve().as_posix(), device="cpu")
            # LGSSM 需要初始化状态
            self.state, self.covariance = self.model.get_initial_state()
            self.first_observation = True
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def transform(self, df):
        """批量转换"""
        res = self.model.transform(df)
        df_res = pd.DataFrame(
            res,
            index=df.index,
            columns=[f"{self.prefix}_{i}" for i in range(res.shape[1])],
        )
        return df_res

    def inference(self, df_one_row):
        """单行实时推理"""
        arr = df_one_row.to_numpy()

        if self.model_type == "deep_ssm":
            # DeepSSM 使用 process_single
            res = self.model_inference.process_single(arr)
            return pd.DataFrame(
                res.reshape(1, -1),
                index=df_one_row.index,
                columns=[f"{self.prefix}_{i}" for i in range(len(res))],
            )
        else:  # lg_ssm
            # LGSSM 使用 update_single
            self.state, self.covariance = self.model.update_single(
                arr,
                self.state,
                self.covariance,
                is_first_observation=self.first_observation,
            )
            if self.first_observation:
                self.first_observation = False
            return pd.DataFrame(
                self.state.reshape(1, -1),
                index=df_one_row.index,
                columns=[f"{self.prefix}_{i}" for i in range(len(self.state))],
            )


class LGBMContainer:
    def __init__(
        self,
        model_type: Literal["c", "r"],
        lag: int,
        pred_next: int,
        is_livetrading=False,
    ):
        self.MODEL_NAME = f"{model_type}_L{lag}_N{pred_next}"

        if is_livetrading:
            path_model = Path(__file__).parent / f"model_{self.MODEL_NAME}_prod.txt"
        else:
            path_model = Path(__file__).parent / f"model_{self.MODEL_NAME}.txt"

        self.model = lgb.Booster(model_file=path_model)
        self.model_type = model_type
        self.lag = lag
        self.pred_next = pred_next

        self._preds = deque(np.full(pred_next, np.nan), maxlen=pred_next)

    @property
    def preds(self):
        return list(self._preds)

    @preds.setter
    def preds(self, value):
        self._preds.append(value)

    def predict_proba(self, all_feat_df_one_row: pd.DataFrame):
        _feature_names = feature_info[f"{self.MODEL_NAME}"]
        self.preds = self.model.predict(all_feat_df_one_row[_feature_names])[-1]
        return self.preds[0]
