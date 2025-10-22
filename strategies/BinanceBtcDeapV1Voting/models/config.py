import json
from collections import deque
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import jit

from src.models.deep_ssm.deep_ssm import DeepSSM
from src.models.lgssm import LGSSM


@jit(nopython=True, cache=True)
def _apply_filters_numba(
    pred_proba: float,
    threshold: float,
    filter_types: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> int:
    """
    Numba加速的filter应用函数

    Args:
        pred_proba: 原始预测概率
        threshold: 多空分界阈值
        filter_types: filter类型数组 (0=giveup, 1=reverse)
        lower_bounds: 下界数组
        upper_bounds: 上界数组

    Returns:
        过滤后的预测结果: -1=做空, 0=不持仓, 1=做多
    """
    # 根据阈值得到原始预测方向
    raw_pred = 1 if pred_proba >= threshold else -1

    # 按顺序应用每个filter
    current_pred = raw_pred
    for i in range(len(filter_types)):
        # 检查pred_proba是否落在当前filter区间内
        if lower_bounds[i] <= pred_proba < upper_bounds[i]:
            if filter_types[i] == 0:  # giveup
                return 0
            elif filter_types[i] == 1:  # reverse
                current_pred = -current_pred

    return current_pred


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

        # 修复：确保 arr 是 (obs_dim,) 而不是 (1, obs_dim)
        # DataFrame.iloc[[i]].to_numpy() 返回 (1, obs_dim)
        # 但 update_single 期望 (obs_dim,)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]

        if self.model_type == "deep_ssm":
            # DeepSSM 使用 process_single
            res = self.model_inference.process_single(arr)
            res = res.reshape(1, -1)
            return pd.DataFrame(
                res,
                index=df_one_row.index,
                columns=[f"{self.prefix}_{i}" for i in range(res.shape[1])],
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

            # self.state 是 (state_dim,) 形状，需要 reshape 为 (1, state_dim)
            state = self.state.reshape(1, -1)
            state_dim = state.shape[1]

            return pd.DataFrame(
                state,
                index=df_one_row.index,
                columns=[f"{self.prefix}_{i}" for i in range(state_dim)],
            )


def model_name_to_params(name: str) -> tuple[str, int, int, float]:
    """
    将模型名称转化为可以设定LGBMContainer的参数组
    比如，将r_L3_N2转化为r, 3, 2, 0.0
    也就是model_type, lag, pred_next, threshold的组合
    c开头的模型为分类模型，threshold=0.5
    r开头的模型为回归模型，threshold=0
    """
    # 分割模型名称，格式为 "{model_type}_L{lag}_N{pred_next}"
    parts = name.split("_")
    assert (
        len(parts) == 3
    ), f"Invalid model name format: {name}, expected format: {{model_type}}_L{{lag}}_N{{pred_next}}"

    # 提取 model_type
    model_type = parts[0]
    assert model_type.startswith(
        ("c", "r")
    ), f"Invalid model_type: {model_type}, expected to start with 'c' or 'r'"

    # 提取 lag (去掉 "L" 前缀)
    assert parts[1].startswith(
        "L"
    ), f"Invalid lag format: {parts[1]}, expected 'L' prefix"
    lag = int(parts[1][1:])

    # 提取 pred_next (去掉 "N" 前缀)
    assert parts[2].startswith(
        "N"
    ), f"Invalid pred_next format: {parts[2]}, expected 'N' prefix"
    pred_next = int(parts[2][1:])

    # 根据 model_type 设置 threshold
    # c开头的为分类模型，r开头的为回归模型
    threshold = 0.5 if model_type.startswith("c") else 0.0

    return model_type, lag, pred_next, threshold


class LGBMContainer:
    def __init__(
        self,
        model_type: Literal["c", "r", "r2"],
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

        # 编译后的filter数组（用于numba加速）
        self._filter_types = np.array([], dtype=np.int32)
        self._lower_bounds = np.array([], dtype=np.float64)
        self._upper_bounds = np.array([], dtype=np.float64)
        self._filters_compiled = False

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

    def _compile_filters(self):
        """将filter列表编译为numpy数组，用于numba加速"""
        if not self._filters:
            self._filter_types = np.array([], dtype=np.int32)
            self._lower_bounds = np.array([], dtype=np.float64)
            self._upper_bounds = np.array([], dtype=np.float64)
            self._filters_compiled = True
            return

        n_filters = len(self._filters)
        self._filter_types = np.empty(n_filters, dtype=np.int32)
        self._lower_bounds = np.empty(n_filters, dtype=np.float64)
        self._upper_bounds = np.empty(n_filters, dtype=np.float64)

        for i, filt in enumerate(self._filters):
            # 将type编码为整数: giveup=0, reverse=1
            self._filter_types[i] = 0 if filt["type"] == "giveup" else 1
            self._lower_bounds[i] = filt["lower_bound"]
            self._upper_bounds[i] = filt["upper_bound"]

        self._filters_compiled = True

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
        self._filters_compiled = False  # 标记需要重新编译

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
        self._filters_compiled = False  # 标记需要重新编译

    def clear_filters(self):
        """清除所有过滤器"""
        self._filters = []
        self._filters_compiled = False  # 标记需要重新编译

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

        self._filters_compiled = False  # 标记需要重新编译
        print(f"Loaded {len(self._filters)} filters from {filepath}")

    def _auto_load_filters(self):
        """初始化时自动加载filter配置（如果存在）"""
        filter_path = Path(__file__).parent / f"model_{self.MODEL_NAME}_filters.json"
        if filter_path.exists():
            try:
                with open(filter_path, "r") as f:
                    self._filters = json.load(f)
                self._filters_compiled = False  # 标记需要重新编译
                print(f"Auto-loaded {len(self._filters)} filters for {self.MODEL_NAME}")
            except Exception as e:
                print(f"Failed to auto-load filters: {e}")
                self._filters = []
                self._filters_compiled = False

    def predict_proba(self, feat_df_one_row: pd.DataFrame):
        """
        获取模型的原始预测概率

        Args:
            feat_df_one_row: 包含当前模型所有特征的单行DataFrame

        Returns:
            预测概率值
        """
        self.preds = self.model.predict(feat_df_one_row)[-1]
        return self.preds[0]

    def _apply_filters(self, pred_proba: float) -> int:
        """
        应用所有filters并返回最终预测结果（使用numba加速）

        Args:
            pred_proba: 原始预测概率

        Returns:
            过滤后的预测结果: -1=做空, 0=不持仓, 1=做多
        """
        # 如果filters未编译，先编译
        if not self._filters_compiled:
            self._compile_filters()

        # 如果没有filters，直接返回原始预测
        if len(self._filter_types) == 0:
            return 1 if pred_proba >= self.threshold else -1

        # 使用numba优化版本
        return _apply_filters_numba(
            pred_proba,
            self.threshold,
            self._filter_types,
            self._lower_bounds,
            self._upper_bounds,
        )

    def final_predict(self, feat_df_one_row: pd.DataFrame) -> int:
        """
        执行完整预测流程：获取原始预测概率 -> 应用filters -> 输出最终结果

        Args:
            feat_df_one_row: 包含当前模型特征的单行DataFrame

        Returns:
            最终预测方向: -1=做空, 0=不持仓, 1=做多
        """
        # 1. 获取原始预测概率
        pred_proba = self.predict_proba(feat_df_one_row)

        # 2. 处理NaN值
        if np.isnan(pred_proba):
            return 0

        # 3. 应用filters得到最终预测
        final_pred = self._apply_filters(pred_proba)

        return final_pred
