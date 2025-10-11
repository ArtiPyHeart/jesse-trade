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
    # if not k.startswith("r_"):
    ALL_RAW_FEAT.update(v)
ALL_RAW_FEAT = sorted(
    [
        i
        for i in ALL_RAW_FEAT
        if not i.startswith("deep_ssm") and not i.startswith("lg_ssm")
    ]
)


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
        threshold: float,
    ):
        self.MODEL_NAME = f"{model_type}_L{lag}_N{pred_next}"

        self._is_livetrading = False
        self._model = None
        self.model_type = model_type
        self.lag = lag
        self.pred_next = pred_next
        self.threshold = threshold  # 多空分界阈值，分类模型通常为0.5

        self._preds = deque(np.full(pred_next, np.nan), maxlen=pred_next)

        # 置信度切片过滤器列表
        self._filters = []

        # 尝试自动加载filter配置
        self._auto_load_filters()

    @property
    def is_livetrading(self):
        return self._is_livetrading

    @is_livetrading.setter
    def is_livetrading(self, value: bool):
        self._is_livetrading = value
        if value:
            path_model = Path(__file__).parent / f"model_{self.MODEL_NAME}_prod.txt"
        else:
            path_model = Path(__file__).parent / f"model_{self.MODEL_NAME}.txt"

        self._model = lgb.Booster(model_file=path_model)

    @property
    def model(self):
        if self._model is None:
            raise ValueError("LightGBM model not initialized")
        return self._model

    @property
    def preds(self):
        return list(self._preds)

    @preds.setter
    def preds(self, value):
        self._preds.append(value)

    def add_giveup_filter(self, lower_bound: float, upper_bound: float):
        """
        添加放弃持仓的预测概率区间，落入此区间时最终pred输出0

        Args:
            lower_bound: 区间下界
            upper_bound: 区间上界
        """
        assert lower_bound < upper_bound, "lower_bound必须小于upper_bound"
        self._filters.append(
            {
                "type": "giveup",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    def add_reverse_filter(self, lower_bound: float, upper_bound: float):
        """
        添加反向持仓的预测概率区间，落入此区间时最终pred反向输出

        Args:
            lower_bound: 区间下界
            upper_bound: 区间上界

        说明：
            - 原始预测>=threshold（做多），反转后输出-1（做空）
            - 原始预测<threshold（做空），反转后输出1（做多）
        """
        assert lower_bound < upper_bound, "lower_bound必须小于upper_bound"
        self._filters.append(
            {
                "type": "reverse",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    def clear_filters(self):
        """清除所有过滤器"""
        self._filters = []

    def get_filters(self):
        """获取所有过滤器配置"""
        return self._filters.copy()

    def save_filters(self, filepath: str = None):
        """
        保存过滤器配置到JSON文件

        Args:
            filepath: 保存路径，默认为 model_<MODEL_NAME>_filters.json
        """
        if filepath is None:
            filepath = Path(__file__).parent / f"model_{self.MODEL_NAME}_filters.json"
        else:
            filepath = Path(filepath)

        with open(filepath, "w") as f:
            json.dump(self._filters, f, indent=2)

        print(f"Filters saved to {filepath}")

    def load_filters(self, filepath: str = None):
        """
        从JSON文件加载过滤器配置

        Args:
            filepath: 加载路径，默认为 model_<MODEL_NAME>_filters.json
        """
        if filepath is None:
            filepath = Path(__file__).parent / f"model_{self.MODEL_NAME}_filters.json"
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            print(f"Filter file not found: {filepath}")
            return

        with open(filepath, "r") as f:
            self._filters = json.load(f)

        print(f"Loaded {len(self._filters)} filters from {filepath}")

    def _auto_load_filters(self):
        """初始化时自动加载filter配置（如果存在）"""
        filter_path = Path(__file__).parent / f"model_{self.MODEL_NAME}_filters.json"
        if filter_path.exists():
            try:
                with open(filter_path, "r") as f:
                    self._filters = json.load(f)
                print(
                    f"Auto-loaded {len(self._filters)} filters for {self.MODEL_NAME}"
                )
            except Exception as e:
                print(f"Failed to auto-load filters: {e}")
                self._filters = []

    def predict_proba(self, all_feat_df_one_row: pd.DataFrame):
        """
        获取模型的原始预测概率

        Args:
            all_feat_df_one_row: 包含所有特征的单行DataFrame

        Returns:
            预测概率值
        """
        _feature_names = feature_info[f"{self.MODEL_NAME}"]
        self.preds = self.model.predict(all_feat_df_one_row[_feature_names])[-1]
        return self.preds[0]

    def _apply_filters(self, pred_proba: float) -> int:
        """
        应用所有filters并返回最终预测结果

        Args:
            pred_proba: 原始预测概率

        Returns:
            过滤后的预测结果: -1=做空, 0=不持仓, 1=做多
        """
        # 根据阈值得到原始预测方向
        raw_pred = 1 if pred_proba >= self.threshold else -1

        # 按顺序应用每个filter
        current_pred = raw_pred
        for filt in self._filters:
            lower = filt["lower_bound"]
            upper = filt["upper_bound"]

            # 检查pred_proba是否落在当前filter区间内
            if lower <= pred_proba < upper:
                if filt["type"] == "giveup":
                    # 放弃持仓，直接返回0
                    return 0
                elif filt["type"] == "reverse":
                    # 反转预测
                    current_pred = -current_pred

        return current_pred

    def final_predict(self, all_feat_df_one_row: pd.DataFrame) -> int:
        """
        执行完整预测流程：获取原始预测概率 -> 应用filters -> 输出最终结果

        Args:
            all_feat_df_one_row: 包含所有特征的单行DataFrame

        Returns:
            最终预测方向: -1=做空, 0=不持仓, 1=做多
        """
        # 1. 获取原始预测概率
        pred_proba = self.predict_proba(all_feat_df_one_row)

        # 2. 处理NaN值
        if np.isnan(pred_proba):
            return 0

        # 3. 应用filters得到最终预测
        final_pred = self._apply_filters(pred_proba)

        return final_pred
