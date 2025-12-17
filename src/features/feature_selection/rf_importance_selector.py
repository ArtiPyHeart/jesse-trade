import gc
import os
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV

from src.utils.drop_na import drop_na_and_align_x_and_y


class RFImportanceSelector:
    """
    基于随机森林特征重要性的特征选择器

    使用 LightGBM 随机森林模式计算特征重要性（relevance），
    然后筛选 relevance > threshold 的特征。

    相比 RFCQSelector（MRMR 方法），本 selector 跳过了冗余度计算和迭代过程，
    适用于快速初筛有用特征的场景。

    Parameters
    ----------
    threshold : float, 默认=0.0
        特征重要性阈值。只有 relevance > threshold 的特征会被保留。
        默认为 0，即保留所有重要性大于 0 的特征。

    task_type : str, 默认='auto'
        任务类型：'classification', 'regression' 或 'auto'（自动检测）

    scoring : str, 默认=None
        评估随机森林性能的指标。如果为 None，根据任务类型自动设置：
        - 分类任务：'roc_auc'
        - 回归任务：'neg_root_mean_squared_error'

    cv : int, 默认=3
        交叉验证的折数。

    param_grid : dict, 默认=None
        随机森林的超参数网格。如果为 None，则使用默认网格 {"num_leaves": [31, 63]}。

    verbose : bool, 默认=True
        是否显示详细信息。

    random_state : int, 默认=42
        随机种子。

    n_jobs : int, 默认=None
        并行任务数。None 表示使用 CPU 核心数 - 1。

    Attributes
    ----------
    relevance_ : numpy.ndarray
        每个特征与目标的相关性（随机森林特征重要性）

    variables_ : list
        所有数值型特征名

    features_to_drop_ : list
        训练后要删除的特征列表

    selected_features_ : list
        选中的特征列表（relevance > threshold 的特征）
    """

    def __init__(
        self,
        threshold: float = 0.0,
        task_type: str = "auto",
        scoring: Optional[str] = None,
        cv: int = 3,
        param_grid: Optional[dict] = None,
        verbose: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = None,
    ):
        self.threshold = threshold
        self.task_type = task_type
        self.cv = cv
        self.param_grid = param_grid
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs is not None else max(os.cpu_count() - 1, 1)

        # 根据任务类型自动设置评分指标
        if scoring is None:
            if task_type == "classification":
                self.scoring = "roc_auc"
            elif task_type == "regression":
                self.scoring = "neg_root_mean_squared_error"
            else:  # auto
                self.scoring = None  # 将在 fit 时根据数据类型决定
        else:
            self.scoring = scoring

        # 属性初始化
        self.relevance_ = None
        self.variables_ = None
        self.features_to_drop_ = None
        self.selected_features_ = None

    def _find_numerical_variables(self, X: pd.DataFrame) -> List[str]:
        """找出数据框中的数值型变量"""
        return X.select_dtypes(include=["number"]).columns.tolist()

    def _calculate_relevance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        计算特征与目标的相关性（使用随机森林特征重要性）
        """
        # 使用 float32 减少内存占用
        X_values = np.asarray(X.values, dtype=np.float32, order="C")
        y_values = y.values

        # 自动检测任务类型
        if self.task_type == "auto":
            unique_values = len(np.unique(y_values))
            is_classification = unique_values < min(5, len(y_values) * 0.01)
        else:
            is_classification = self.task_type == "classification"

        # 确定 scoring
        if self.scoring is None:
            scoring = "roc_auc" if is_classification else "neg_root_mean_squared_error"
        else:
            scoring = self.scoring

        # 根据任务类型创建模型（使用 LightGBM 随机森林模式）
        # 参数平衡：在精度和速度之间取得折中
        if is_classification:
            model = LGBMClassifier(
                boosting_type="rf",
                n_estimators=500,
                num_leaves=31,  # 由 GridSearchCV 调优
                subsample=0.632,  # RF bootstrap 采样率（官方推荐）
                subsample_freq=1,
                colsample_bytree=0.75,  # 特征子采样（0.8→0.75）
                importance_type="gain",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                max_bin=127,  # 折中值（255→127），显著提速
                min_data_in_leaf=20,  # 防止过拟合
                histogram_pool_size=512,  # 限制 histogram 缓存大小
                free_raw_data=True,
            )
        else:
            model = LGBMRegressor(
                boosting_type="rf",
                n_estimators=500,
                num_leaves=31,  # 由 GridSearchCV 调优
                subsample=0.632,  # RF bootstrap 采样率（官方推荐）
                subsample_freq=1,
                colsample_bytree=0.75,  # 特征子采样（0.8→0.75）
                importance_type="gain",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                max_bin=127,  # 折中值（255→127），显著提速
                min_data_in_leaf=20,  # 防止过拟合
                histogram_pool_size=512,  # 限制 histogram 缓存大小
                free_raw_data=True,
            )

        # 设置参数网格（适度缩减搜索空间）
        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"num_leaves": [31, 63, 127]}

        # 网格搜索（refit=True 自动用最佳参数在全量数据上训练）
        cv_model = GridSearchCV(
            model,
            cv=self.cv,
            scoring=scoring,
            param_grid=param_grid,
            refit=True,
            return_train_score=False,
            n_jobs=1,
            pre_dispatch=2,
        )

        with joblib.parallel_backend("threading"):
            cv_model.fit(X_values, y_values)

        # 直接从 best_estimator_ 获取特征重要性
        relevance = cv_model.best_estimator_.feature_importances_.copy()

        # 清理资源
        del cv_model
        gc.collect()

        return relevance

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RFImportanceSelector":
        """
        训练特征选择器

        Parameters
        ----------
        X : pandas.DataFrame
            特征数据框

        y : pandas.Series
            目标变量

        Returns
        -------
        self
        """
        # 确保输入是 pandas 对象
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 pandas.DataFrame")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # 对齐 x 与 y 的长度并去除空值
        X, y = drop_na_and_align_x_and_y(X, y)

        # 找出数值型变量
        if self.verbose:
            print("➤ 识别数值型变量...")
        self.variables_ = self._find_numerical_variables(X)

        if len(self.variables_) < 1:
            raise ValueError("至少需要 1 个数值型特征")

        # 计算特征重要性
        if self.verbose:
            print("➤ 计算特征重要性（使用随机森林）...")
        X_numeric = X[self.variables_]
        self.relevance_ = self._calculate_relevance(X_numeric, y)

        # 筛选 relevance > threshold 的特征
        mask = self.relevance_ > self.threshold
        self.selected_features_ = [f for f, m in zip(self.variables_, mask) if m]
        self.features_to_drop_ = [f for f, m in zip(self.variables_, mask) if not m]

        if self.verbose:
            total = len(self.variables_)
            selected = len(self.selected_features_)
            dropped = len(self.features_to_drop_)
            print(
                f"\n✅ 特征选择完成：从 {total} 个特征中选择了 {selected} 个，"
                f"舍弃了 {dropped} 个（threshold={self.threshold}）"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，只保留选中的特征

        Parameters
        ----------
        X : pandas.DataFrame
            输入数据

        Returns
        -------
        pandas.DataFrame
            只包含选中特征的数据框
        """
        if self.features_to_drop_ is None:
            raise ValueError("在使用 transform 之前必须先调用 fit")

        return X.drop(columns=self.features_to_drop_, errors="ignore")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        训练选择器并转换数据

        Parameters
        ----------
        X : pandas.DataFrame
            特征数据框

        y : pandas.Series
            目标变量

        Returns
        -------
        pandas.DataFrame
            只包含选中特征的数据框
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        获取所选特征的布尔掩码或索引

        Parameters
        ----------
        indices : bool, 默认=False
            如果为 True，返回特征索引；否则返回布尔掩码

        Returns
        -------
        numpy.ndarray 或 List[int]
            特征选择掩码或索引
        """
        if self.features_to_drop_ is None:
            raise ValueError("在使用 get_support 之前必须先调用 fit")

        mask = self.relevance_ > self.threshold

        if indices:
            return np.where(mask)[0].tolist()
        else:
            return mask
