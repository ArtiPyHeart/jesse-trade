"""
GrootCV 特征选择器核心实现

基于 SHAP 值的特征选择方法，使用 shadow features 和交叉验证。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.model_selection import RepeatedKFold
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from tqdm.auto import tqdm

from src.features.feature_selection.groot.lgbm_core import (
    set_lgb_parameters,
    split_train_val,
    train_lgb_model,
)
from src.features.feature_selection.groot.shadow import create_shadow_features
from src.features.feature_selection.groot.shap_utils import (
    compute_shap_importance,
    normalize_importance,
)


class GrootCV(SelectorMixin, BaseEstimator):
    """
    GrootCV 特征选择器

    基于 SHAP 值的特征选择方法：
    1. 创建 Shadow Features：复制所有特征并随机打乱
    2. 使用 RepeatedKFold 交叉验证训练 LightGBM
    3. 计算 SHAP 特征重要性
    4. 选择重要性超过 shadow 阈值的特征

    Parameters
    ----------
    objective : str, default=None
        LightGBM objective，如 'binary', 'rmse', 'softmax'
    cutoff : float, default=1.0
        特征选择阈值。cutoff 越小越严格，越大越宽松。
        公式：特征被选中需要 importance >= shadow_max / cutoff
    n_folds : int, default=5
        交叉验证折数
    n_iter : int, default=5
        交叉验证重复次数
    silent : bool, default=True
        是否静默模式
    fastshap : bool, default=False
        是否使用 fasttreeshap（需要 numpy<2.0）
    n_jobs : int, default=0
        并行数，0 表示使用 OpenMP 默认线程数
    lgbm_params : Dict[str, Any], optional
        自定义 LightGBM 参数
    random_state : int, optional
        随机种子

    Attributes
    ----------
    feature_names_in_ : np.ndarray
        输入特征名
    selected_features_ : np.ndarray
        选中的特征名
    support_ : np.ndarray
        特征选择掩码
    ranking_ : np.ndarray
        特征排名（2=选中，1=未选中）
    sha_cutoff_ : float
        shadow 阈值
    cv_importance_df_ : pd.DataFrame
        交叉验证中每个 fold 的特征重要性
    """

    def __init__(
        self,
        objective: Optional[str] = None,
        cutoff: float = 1.0,
        n_folds: int = 5,
        n_iter: int = 5,
        silent: bool = True,
        fastshap: bool = False,
        n_jobs: int = 0,
        lgbm_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        if cutoff <= 0:
            raise ValueError("cutoff 必须大于 0")
        if n_folds < 2:
            raise ValueError("n_folds 必须至少为 2")
        if n_iter < 1:
            raise ValueError("n_iter 必须至少为 1")

        self.objective = objective
        self.cutoff = cutoff
        self.n_folds = n_folds
        self.n_iter = n_iter
        self.silent = silent
        self.fastshap = fastshap
        self.n_jobs = n_jobs
        self.lgbm_params = lgbm_params
        self.random_state = random_state

        # Fitted attributes
        self.feature_names_in_: Optional[np.ndarray] = None
        self.selected_features_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.ranking_: Optional[np.ndarray] = None
        self.sha_cutoff_: Optional[float] = None
        self.cv_importance_df_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "GrootCV":
        """
        训练特征选择器

        Parameters
        ----------
        X : pd.DataFrame
            特征数据框
        y : pd.Series
            目标变量
        sample_weight : pd.Series, optional
            样本权重

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 pandas.DataFrame")

        self.feature_names_in_ = X.columns.to_numpy()

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if sample_weight is not None:
            sample_weight = pd.Series(_check_sample_weight(sample_weight, X))

        # 执行特征选择
        selected, cv_df, cutoff_threshold = self._reduce_vars_lgb_cv(
            X=X,
            y=y,
            sample_weight=sample_weight,
        )

        self.selected_features_ = selected.values
        self.cv_importance_df_ = cv_df
        self.sha_cutoff_ = cutoff_threshold

        # 计算 support 和 ranking
        self.support_ = np.isin(self.feature_names_in_, self.selected_features_)
        self.ranking_ = np.where(self.support_, 2, 1)

        if not self.silent:
            n_total = len(self.feature_names_in_)
            n_selected = len(self.selected_features_)
            print(
                f"[GrootCV] 从 {n_total} 个特征中选择了 {n_selected} 个 "
                f"(cutoff={self.cutoff}, 阈值={cutoff_threshold:.6f})"
            )

        return self

    def _reduce_vars_lgb_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series],
    ) -> Tuple[pd.Series, pd.DataFrame, float]:
        """
        使用交叉验证的 LightGBM 进行特征选择

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame, float]
            - 选中的特征名
            - 特征重要性 DataFrame
            - shadow 阈值
        """
        # 设置 LightGBM 参数
        params = set_lgb_parameters(
            y=y,
            objective=self.objective,
            n_jobs=self.n_jobs,
            lgbm_params=self.lgbm_params,
            silent=self.silent,
        )

        # 创建交叉验证分割器
        cv = RepeatedKFold(
            n_splits=self.n_folds,
            n_repeats=self.n_iter,
            random_state=self.random_state if self.random_state else 2652124,
        )

        # 初始化重要性 DataFrame
        importance_df = pd.DataFrame({"feature": X.columns})
        shadow_names: List[str] = []

        # 交叉验证循环
        fold_iter = enumerate(cv.split(X, y))
        if not self.silent:
            fold_iter = tqdm(
                fold_iter,
                total=cv.get_n_splits(),
                desc="GrootCV Cross Validation",
            )

        for fold_idx, (train_idx, val_idx) in fold_iter:
            # 分割数据
            X_train, X_val, y_train, y_val, w_train, w_val = split_train_val(
                X=X,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                weight=sample_weight,
            )

            # 创建 shadow 特征
            X_train_shadow, shadow_names = create_shadow_features(
                X_train, random_state=self.random_state
            )
            X_val_shadow, _ = create_shadow_features(
                X_val, random_state=self.random_state
            )

            # 训练模型
            model, _ = train_lgb_model(
                X_train=X_train_shadow,
                y_train=y_train,
                X_val=X_val_shadow,
                y_val=y_val,
                params=params,
                weight_train=w_train,
                weight_val=w_val,
                num_boost_round=10000,
                early_stopping_rounds=20,
                verbose_eval=0,
            )

            # 计算 SHAP 重要性
            importance = compute_shap_importance(
                X=X_train_shadow,
                model=model,
                objective=params["objective"],
                fastshap=self.fastshap,
            )

            # 归一化并合并
            importance = normalize_importance(importance)
            fold_df = pd.DataFrame(
                list(importance.items()),
                columns=["feature", f"fold_{fold_idx}"],
            )
            importance_df = importance_df.merge(fold_df, on="feature", how="outer")

        # 计算平均重要性
        numeric_cols = importance_df.select_dtypes(include=[np.number]).columns
        importance_df["mean_importance"] = importance_df[numeric_cols].mean(axis=1)

        # 分离真实特征和 shadow 特征
        real_df = importance_df[~importance_df["feature"].isin(shadow_names)]
        shadow_df = importance_df[importance_df["feature"].isin(shadow_names)]

        # 计算 shadow 阈值
        # 使用 shadow 特征的最大平均重要性 / cutoff
        shadow_max = shadow_df[numeric_cols].max().mean()
        cutoff_threshold = shadow_max / self.cutoff

        # 选择超过阈值的特征
        selected_df = real_df[real_df["mean_importance"] >= cutoff_threshold]
        selected_features = selected_df["feature"]

        return selected_features, importance_df, cutoff_threshold

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，只保留选中的特征

        Parameters
        ----------
        X : pd.DataFrame
            输入数据

        Returns
        -------
        pd.DataFrame
            只包含选中特征的数据框
        """
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 pandas.DataFrame")

        return X[self.selected_features_]

    def _get_support_mask(self) -> np.ndarray:
        """返回特征选择掩码"""
        check_is_fitted(self)
        return self.support_

    def get_feature_names_out(
        self, input_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        获取选中的特征名

        Parameters
        ----------
        input_features : np.ndarray, optional
            输入特征名（未使用，保持 sklearn 接口兼容）

        Returns
        -------
        np.ndarray
            选中的特征名数组
        """
        check_is_fitted(self)
        return self.selected_features_

    def _more_tags(self):
        """sklearn 元数据"""
        return {"allow_nan": True, "requires_y": True}
