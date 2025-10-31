import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class MIPreFilter:
    """
    互信息（Mutual Information）预筛选器

    用于大规模特征集的快速预筛选，在执行更复杂的特征选择（如 RF-MRMR）之前
    快速剔除明显无关的特征。

    Parameters
    ----------
    threshold: float or 'auto', 默认='auto'
        MI 阈值（单位：nats）。
        - 如果为 'auto'，自动计算数据驱动的阈值：max(0.014 nats, null_95th_percentile)
        - 如果为具体数值，直接使用该值作为阈值
        注：0.014 nats ≈ 0.01 bits，是极其宽松的阈值

    min_reduction_ratio: float, 默认=0.5
        最小削减比例。如果按阈值筛选后保留的特征比例 > (1 - min_reduction_ratio)，
        则进一步按 MI 分数排序削减到 (1 - min_reduction_ratio)。
        例如：min_reduction_ratio=0.5 表示最多保留 50% 的特征。

    task_type: str, 默认='auto'
        任务类型：'classification', 'regression' 或 'auto'（自动检测）

    n_neighbors: int, 默认=5
        MI 估计使用的邻居数量（k-NN 估计器参数）。
        推荐值：3-5，较小的值计算更快但可能不太稳定。

    verbose: bool, 默认=True
        是否显示进度信息和详细输出

    random_state: int, 默认=42
        随机种子，用于可重复性

    n_jobs: int, 默认=None
        并行任务数。None 表示 1，-1 表示使用所有处理器。

    Attributes
    ----------
    mi_scores_: numpy.ndarray
        每个特征的 MI 分数（单位：nats）

    threshold_: float
        实际使用的阈值（如果设置为 'auto'，则为自动计算的值）

    selected_features_: list
        选中的特征名称列表

    features_to_drop_: list
        要删除的特征名称列表

    null_threshold_: float or None
        如果使用 'auto' 模式，存储计算的 null 分布 95th percentile
    """

    def __init__(
        self,
        threshold: Union[float, str] = "auto",
        min_reduction_ratio: float = 0.5,
        task_type: str = "auto",
        n_neighbors: int = 5,
        verbose: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = max(os.cpu_count() - 1, 1),
    ):
        self.threshold = threshold
        self.min_reduction_ratio = min_reduction_ratio
        self.task_type = task_type
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Attributes set during fit
        self.mi_scores_ = None
        self.threshold_ = None
        self.selected_features_ = None
        self.features_to_drop_ = None
        self.null_threshold_ = None

    def _find_numerical_variables(self, X: pd.DataFrame) -> List[str]:
        """找出数据框中的数值型变量"""
        return X.select_dtypes(include=["number"]).columns.tolist()

    def _compute_null_threshold(
        self, X: np.ndarray, y: np.ndarray, is_classification: bool
    ) -> float:
        """
        计算 null 分布的 95th percentile 阈值

        通过打乱目标变量 y 来计算随机情况下的 MI 分布，
        从而确定一个数据驱动的阈值。
        """
        if self.verbose:
            print("  ↳ 计算 null 分布阈值（打乱 y 后的 MI 分布）...")

        # 打乱目标变量
        rng = np.random.RandomState(self.random_state)
        y_shuffled = rng.permutation(y)

        # 计算 null MI 分数
        if is_classification:
            mi_null = mutual_info_classif(
                X,
                y_shuffled,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            mi_null = mutual_info_regression(
                X,
                y_shuffled,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        null_threshold = np.percentile(mi_null, 95)

        if self.verbose:
            print(
                f"    Null 分布 95th percentile: {null_threshold:.6f} nats "
                f"({null_threshold / np.log(2):.6f} bits)"
            )

        return null_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIPreFilter":
        """
        计算 MI 分数并确定要保留的特征

        Parameters
        ----------
        X: pandas.DataFrame
            特征数据框

        y: pandas.Series
            目标变量

        Returns
        -------
        self
        """
        # 输入验证
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 pandas.DataFrame")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # 找出数值型变量
        numeric_cols = self._find_numerical_variables(X)
        if len(numeric_cols) == 0:
            raise ValueError("没有找到数值型特征")

        X_numeric = X[numeric_cols]
        # 互信息估计不支持 NaN，需要提前验证
        nan_counts = X_numeric.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            details = ", ".join(f"{col}: {int(count)}" for col, count in nan_cols.items())
            raise ValueError(f"检测到 NaN 值，请先处理后再调用 MI 预筛选。含 NaN 特征: {details}")
        X_values = X_numeric.values
        y_values = y.values

        # 自动检测任务类型
        if self.task_type == "auto":
            unique_values = len(np.unique(y_values))
            is_classification = unique_values < min(5, len(y_values) * 0.01)
            task_name = "classification" if is_classification else "regression"
        else:
            is_classification = self.task_type == "classification"
            task_name = self.task_type

        if self.verbose:
            print(f"➤ 互信息预筛选 (任务类型: {task_name})...")
            print(f"  初始特征数: {len(numeric_cols)}")

        # 确定阈值
        if self.threshold == "auto":
            # 计算 null 分布阈值
            self.null_threshold_ = self._compute_null_threshold(
                X_values, y_values, is_classification
            )
            # 使用 max(0.014 nats, null_95th)
            self.threshold_ = max(0.014, self.null_threshold_)

            if self.verbose:
                print(
                    f"  ↳ 自动阈值: {self.threshold_:.6f} nats "
                    f"({self.threshold_ / np.log(2):.6f} bits)"
                )
        else:
            self.threshold_ = self.threshold
            if self.verbose:
                print(
                    f"  ↳ 使用固定阈值: {self.threshold_:.6f} nats "
                    f"({self.threshold_ / np.log(2):.6f} bits)"
                )

        # 计算真实的 MI 分数
        if self.verbose:
            print("  ↳ 计算特征与目标的互信息...")

        if is_classification:
            self.mi_scores_ = mutual_info_classif(
                X_values,
                y_values,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            self.mi_scores_ = mutual_info_regression(
                X_values,
                y_values,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        # 第一阶段：按阈值筛选
        threshold_mask = self.mi_scores_ >= self.threshold_
        n_after_threshold = threshold_mask.sum()
        retention_ratio = n_after_threshold / len(numeric_cols)

        if self.verbose:
            print(
                f"  ↳ 阈值筛选后: {n_after_threshold}/{len(numeric_cols)} 特征 "
                f"({retention_ratio:.1%} 保留)"
            )

        # 第二阶段：如果保留比例过高，进一步按分数削减
        target_retention = 1.0 - self.min_reduction_ratio
        if retention_ratio > target_retention:
            if self.verbose:
                print(
                    f"  ↳ 保留比例 ({retention_ratio:.1%}) > 目标 ({target_retention:.1%})，"
                    f"按 MI 分数进一步削减..."
                )

            # 计算需要保留的特征数量
            n_to_keep = max(1, int(target_retention * len(numeric_cols)))

            # 按 MI 分数排序，保留 top n_to_keep
            top_indices = np.argsort(self.mi_scores_)[-n_to_keep:]
            final_mask = np.zeros(len(numeric_cols), dtype=bool)
            final_mask[top_indices] = True

            if self.verbose:
                print(f"    削减到 {n_to_keep} 个特征 ({target_retention:.1%} 保留)")
        else:
            # 保留比例已满足要求，直接使用阈值筛选结果
            final_mask = threshold_mask

        # 记录选中和删除的特征
        self.selected_features_ = [
            numeric_cols[i] for i in range(len(numeric_cols)) if final_mask[i]
        ]
        self.features_to_drop_ = [
            f for f in numeric_cols if f not in self.selected_features_
        ]

        # 输出统计信息
        if self.verbose:
            n_selected = len(self.selected_features_)
            n_dropped = len(self.features_to_drop_)
            final_retention = n_selected / len(numeric_cols)

            # MI 分数统计
            selected_mi = self.mi_scores_[final_mask]
            mi_min = selected_mi.min() if len(selected_mi) > 0 else 0
            mi_max = selected_mi.max() if len(selected_mi) > 0 else 0
            mi_mean = selected_mi.mean() if len(selected_mi) > 0 else 0

            print(f"\n✅ 互信息预筛选完成:")
            print(
                f"   {len(numeric_cols)} → {n_selected} 特征 "
                f"(保留 {final_retention:.1%}, 删除 {n_dropped})"
            )
            print(
                f"   选中特征 MI 范围: [{mi_min:.6f}, {mi_max:.6f}] nats "
                f"(均值: {mi_mean:.6f})"
            )
            print(
                f"   对应 bits: [{mi_min/np.log(2):.6f}, {mi_max/np.log(2):.6f}] "
                f"(均值: {mi_mean/np.log(2):.6f})"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，只保留选中的特征

        Parameters
        ----------
        X: pandas.DataFrame
            输入数据

        Returns
        -------
        pandas.DataFrame
            只包含选中特征的数据框
        """
        if self.selected_features_ is None:
            raise ValueError("在使用 transform 之前必须先调用 fit")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        训练过滤器并转换数据

        Parameters
        ----------
        X: pandas.DataFrame
            特征数据框

        y: pandas.Series
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
        indices: bool, 默认=False
            如果为 True，返回特征索引，否则返回布尔掩码

        Returns
        -------
        numpy.ndarray 或 List[int]
            特征选择掩码或索引
        """
        if self.selected_features_ is None:
            raise ValueError("在使用 get_support 之前必须先调用 fit")

        all_features = self.selected_features_ + self.features_to_drop_
        mask = np.array([f in self.selected_features_ for f in all_features])

        if indices:
            return np.where(mask)[0].tolist()
        else:
            return mask
