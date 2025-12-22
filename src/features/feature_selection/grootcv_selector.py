"""
GrootCV 特征选择器封装

使用 SHAP 值（比 gain/gini 更稳定可靠）进行特征重要性评估，
通过交叉验证 + 早停防止过拟合。

相比 RFImportanceSelector：
- 使用 SHAP 值而非原生 gain，更稳定可靠
- 内置交叉验证和早停机制
- 自动处理类别不平衡

自建实现，兼容 LightGBM 4.6.0 和 fasttreeshap (numpy 1.x)。
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.features.feature_selection.groot import GrootCV
from src.utils.drop_na import drop_na_and_align_x_and_y


class GrootCVConfig(BaseModel):
    """GrootCV 特征选择器配置

    Attributes:
        objective: 任务类型 - "binary" (二分类), "rmse" (回归) 或 "auto" (自动检测)
        cutoff: 特征选择阈值，默认为 10.0。
                cutoff 越大越宽松，越小越严格。
                具体公式：特征被选中需要 importance >= shadow_max / cutoff
        n_folds: 交叉验证折数，默认 5
        n_iter: 迭代次数，默认 3（仅 repeated_kfold 模式有效）
        cv_type: 交叉验证类型，默认 "time_series"（适合时序数据）
        silent: 是否静默模式，默认 True
        fastshap: 是否使用 fasttreeshap 加速（需 NumPy<2），默认 True
        n_jobs: 并行数，0 表示使用 OpenMP 默认线程数
        lgbm_params: 自定义 LightGBM 参数，如 {"min_data_in_leaf": 20}
        shap_max_samples: 计算 SHAP 的最大样本数（可选，下采样减少内存）
        shap_batch_size: 计算 SHAP 的批大小（可选，分批减少峰值内存）
    """

    objective: Literal["binary", "rmse", "auto"] = Field(
        default="auto",
        description="任务类型: binary (分类), rmse (回归), auto (自动检测)",
    )
    cutoff: float = Field(
        default=10.0,
        gt=0,
        description="特征选择阈值，越大越宽松（importance >= shadow_max / cutoff）",
    )
    n_folds: int = Field(default=5, ge=2, le=10, description="交叉验证折数")
    n_iter: int = Field(default=3, ge=1, le=20, description="迭代次数（仅 repeated_kfold 模式）")
    cv_type: Literal["blocked_kfold", "time_series", "repeated_kfold"] = Field(
        default="blocked_kfold",
        description="交叉验证类型: blocked_kfold（默认，分块无重复）, time_series（扩展窗口）, repeated_kfold（随机分割）",
    )
    silent: bool = Field(default=True, description="是否静默模式")
    fastshap: bool = Field(
        default=True, description="是否使用 fasttreeshap 加速（需 NumPy<2）"
    )
    n_jobs: int = Field(default=0, ge=0, description="并行数，0 表示使用 OpenMP 默认线程")
    lgbm_params: Optional[Dict[str, Any]] = Field(
        default=None, description="自定义 LightGBM 参数"
    )
    shap_max_samples: Optional[int] = Field(
        default=None, ge=1, description="SHAP 采样上限（None 表示不采样）"
    )
    shap_batch_size: Optional[int] = Field(
        default=None, ge=1, description="SHAP 分批大小（None 表示不分批）"
    )


class GrootCVSelector:
    """
    GrootCV 特征选择器

    使用 SHAP 值（比 gain/gini 更可靠）进行特征重要性评估，
    通过交叉验证 + 早停防止过拟合。

    相比 RFImportanceSelector：
    - 使用 SHAP 值而非原生 gain，更稳定可靠
    - 内置交叉验证和早停机制
    - 自动处理类别不平衡

    Parameters
    ----------
    config : GrootCVConfig, optional
        配置对象。如果为 None，使用默认配置。

    task_type : str, default='auto'
        任务类型：'classification', 'regression' 或 'auto'（自动检测）
        当设置为 'auto' 时，会根据目标变量的唯一值数量自动判断。
        注意：此参数会覆盖 config.objective

    verbose : bool, default=True
        是否显示详细信息

    random_state : int, optional
        随机种子

    Attributes
    ----------
    relevance_ : numpy.ndarray
        每个特征的 SHAP 重要性分数（ranking_）

    variables_ : list
        所有数值型特征名

    features_to_drop_ : list
        训练后要删除的特征列表

    selected_features_ : list
        选中的特征列表

    ranking_ : numpy.ndarray
        特征排名（2=选中，1=未选中）

    Examples
    --------
    >>> from src.features.feature_selection import GrootCVSelector, GrootCVConfig
    >>>
    >>> # 使用默认配置
    >>> selector = GrootCVSelector()
    >>> selector.fit(X, y)
    >>> X_selected = selector.transform(X)
    >>>
    >>> # 自定义配置
    >>> config = GrootCVConfig(cutoff=2, n_folds=3)
    >>> selector = GrootCVSelector(config=config, task_type='classification')
    >>> selector.fit_transform(X, y)
    """

    # 有效的 task_type 值
    _VALID_TASK_TYPES = ("auto", "classification", "regression")

    def __init__(
        self,
        config: Optional[GrootCVConfig] = None,
        task_type: str = "auto",
        verbose: bool = True,
        random_state: Optional[int] = 42,
    ):
        # 校验 task_type
        if task_type not in self._VALID_TASK_TYPES:
            raise ValueError(
                f"task_type 必须是 {self._VALID_TASK_TYPES} 之一，"
                f"收到: '{task_type}'"
            )

        self.config = config or GrootCVConfig()
        self.task_type = task_type
        self.verbose = verbose
        self.random_state = random_state

        # 根据 task_type 覆盖 config.objective（如果 task_type 不是 'auto'）
        if task_type == "classification":
            self.config = self.config.model_copy(update={"objective": "binary"})
        elif task_type == "regression":
            self.config = self.config.model_copy(update={"objective": "rmse"})
        # task_type == 'auto' 时保持 config.objective 不变

        # Attributes (set during fit)
        self.relevance_: Optional[np.ndarray] = None
        self.variables_: Optional[List[str]] = None
        self.features_to_drop_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None
        self.ranking_: Optional[np.ndarray] = None
        self._groot_cv: Optional[GrootCV] = None  # 存储内部 GrootCV 实例

    def _find_numerical_variables(self, X: pd.DataFrame) -> List[str]:
        """找出数据框中的数值型变量"""
        return X.select_dtypes(include=["number"]).columns.tolist()

    def _detect_task_type(self, y: pd.Series) -> str:
        """自动检测任务类型"""
        unique_values = len(np.unique(y.values))
        # 唯一值 < 5 或 < 1% 样本数，视为分类任务
        is_classification = unique_values < min(5, len(y) * 0.01)
        return "binary" if is_classification else "rmse"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GrootCVSelector":
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
        # 1. 输入验证
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 pandas.DataFrame")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # 2. 对齐 X 与 y 的长度并去除空值
        X, y = drop_na_and_align_x_and_y(X, y)

        # 3. 找出数值型变量
        if self.verbose:
            print("-> 识别数值型变量...")
        self.variables_ = self._find_numerical_variables(X)

        if len(self.variables_) < 1:
            raise ValueError("至少需要 1 个数值型特征")

        X_numeric = X[self.variables_]

        # 4. 确定 objective（使用 config.objective，若为 'auto' 则自动检测）
        if self.config.objective == "auto":
            objective = self._detect_task_type(y)
        else:
            objective = self.config.objective

        if self.verbose:
            print(f"-> 任务类型: {objective}")
            print(f"-> 使用 GrootCV 进行特征选择 (cutoff={self.config.cutoff})...")

        # 5. 创建 GrootCV 实例（使用自建实现）
        # 合并 lgbm_params，将 random_state 作为 seed 传入
        lgbm_params = dict(self.config.lgbm_params or {})
        if self.random_state is not None and "seed" not in lgbm_params:
            lgbm_params["seed"] = self.random_state

        self._groot_cv = GrootCV(
            objective=objective,
            cutoff=self.config.cutoff,
            n_folds=self.config.n_folds,
            n_iter=self.config.n_iter,
            cv_type=self.config.cv_type,
            silent=self.config.silent or not self.verbose,
            fastshap=self.config.fastshap,
            n_jobs=self.config.n_jobs,
            lgbm_params=lgbm_params if lgbm_params else None,
            random_state=self.random_state,
            shap_max_samples=self.config.shap_max_samples,
            shap_batch_size=self.config.shap_batch_size,
        )

        # 6. 拟合
        self._groot_cv.fit(X_numeric, y)

        # 7. 提取结果
        selected_names = self._groot_cv.get_feature_names_out()
        self.selected_features_ = list(selected_names)
        self.features_to_drop_ = [
            f for f in self.variables_ if f not in self.selected_features_
        ]

        # 8. 提取 ranking 作为 relevance_
        self.ranking_ = getattr(self._groot_cv, "ranking_", None)
        if self.ranking_ is not None:
            self.relevance_ = self.ranking_.copy()
        else:
            # 如果无法获取 ranking_，创建基于选择的二值重要性
            self.relevance_ = np.array(
                [1.0 if f in self.selected_features_ else 0.0 for f in self.variables_]
            )

        if self.verbose:
            total = len(self.variables_)
            selected = len(self.selected_features_)
            dropped = len(self.features_to_drop_)
            print(
                f"\n[OK] 特征选择完成：从 {total} 个特征中选择了 {selected} 个，"
                f"舍弃了 {dropped} 个 (cutoff={self.config.cutoff})"
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
        """训练选择器并转换数据"""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        获取所选特征的布尔掩码或索引

        Parameters
        ----------
        indices : bool, default=False
            如果为 True，返回特征索引；否则返回布尔掩码

        Returns
        -------
        numpy.ndarray 或 List[int]
            特征选择掩码或索引
        """
        if self.features_to_drop_ is None:
            raise ValueError("在使用 get_support 之前必须先调用 fit")

        mask = np.array([f in self.selected_features_ for f in self.variables_])

        if indices:
            return np.where(mask)[0].tolist()
        else:
            return mask
