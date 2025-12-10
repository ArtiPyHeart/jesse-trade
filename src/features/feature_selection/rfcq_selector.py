import gc
import os
from typing import List, Optional, Union

import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils.drop_na import drop_na_and_align_x_and_y

nb.set_num_threads(max(1, os.cpu_count() - 1))


# numba加速的相关系数计算
@nb.njit(parallel=True)
def fast_corrwith_numba(X_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """
    使用numba加速计算X的每一列与y的相关系数

    Parameters
    ----------
    X_values: np.ndarray
        特征矩阵，形状为(n_samples, n_features)
    y_values: np.ndarray
        目标向量，形状为(n_samples,)

    Returns
    -------
    np.ndarray
        每个特征与y的相关系数的绝对值，形状为(n_features,)
    """
    n_features = X_values.shape[1]
    result = np.zeros(n_features)

    # 计算y的标准差
    y_mean = np.mean(y_values)
    y_std = np.std(y_values)

    if y_std == 0:
        return result  # 如果y是常数，返回全零数组

    # 标准化y (一次性计算)
    y_norm = (y_values - y_mean) / y_std

    # 对每列计算相关系数 (并行)
    for i in nb.prange(n_features):
        x = X_values[:, i]
        x_mean = np.mean(x)
        x_std = np.std(x)

        if x_std == 0:
            result[i] = 0  # 如果x是常数，相关系数为0
            continue

        # 标准化x并计算相关系数
        x_norm = (x - x_mean) / x_std
        corr = np.mean(x_norm * y_norm)
        result[i] = abs(corr)  # 取绝对值

    return result


# 基于随机森林的RFCQ特征选择器实现
class RFCQSelector:
    """
    基于随机森林的RFCQ特征选择器，关键部分使用numba加速

    RFCQ = Random Forest for relevance, Correlation for redundancy, Quotient for combining

    Parameters
    ----------
    max_features: int, 默认=None
        要选择的特征数量。如果为None，则默认为特征总数的20%。

    task_type: str, 默认='auto'
        任务类型：'classification', 'regression' 或 'auto'（自动检测）

    scoring: str, 默认=None
        评估随机森林性能的指标。如果为None，根据任务类型自动设置：
        - 分类任务：'roc_auc'
        - 回归任务：'neg_root_mean_squared_error'

    cv: int, 默认=3
        交叉验证的折数。

    param_grid: dict, 默认=None
        随机森林的超参数网格。如果为None，则使用默认的网格 {"max_depth": [1, 2, 3, 4]}。

    verbose: bool, 默认=True
        是否显示进度条和详细信息。

    random_state: int, 默认=None
        随机种子，用于随机森林的初始化。

    n_jobs: int, 默认=None
        并行任务数。None表示1，-1表示使用所有处理器。

    Attributes
    ----------
    features_to_drop_: list
        训练后要删除的特征列表

    variables_: list
        考虑的特征列表

    relevance_: numpy.ndarray
        每个特征与目标的相关性
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        task_type: str = "auto",
        scoring: Optional[str] = None,
        cv: int = 3,
        param_grid: Optional[dict] = None,
        verbose: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = max(os.cpu_count() - 1, 1),
    ):
        self.max_features = max_features
        self.task_type = task_type

        # 根据任务类型自动设置评分指标
        if scoring is None:
            if task_type == "classification":
                self.scoring = "roc_auc"
            elif task_type == "regression":
                self.scoring = "neg_root_mean_squared_error"
            else:  # auto
                self.scoring = None  # 将在fit时根据数据类型决定
        else:
            self.scoring = scoring

        self.cv = cv
        self.param_grid = param_grid
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.features_to_drop_ = None
        self.variables_ = None
        self.relevance_ = None

    def _find_numerical_variables(self, X: pd.DataFrame) -> List[str]:
        """
        找出数据框中的数值型变量
        """
        return X.select_dtypes(include=["number"]).columns.tolist()

    def _calculate_relevance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        计算特征与目标的相关性（使用随机森林特征重要性）

        使用 LightGBM 原生 CV 替代 GridSearchCV，优势：
        - 复用 binning 跨 fold（显著加速）
        - 支持 early_stopping 自动找最优迭代次数
        - 避免 sklearn 的额外开销
        """
        import lightgbm as lgb

        # 使用 float32 减少内存占用，对 LightGBM 精度影响可忽略
        X_values = np.asarray(X.values, dtype=np.float32, order="C")
        y_values = y.values

        # 自动检测任务类型
        if self.task_type == "auto":
            unique_values = len(np.unique(y_values))
            # 如果唯一值数量小于5或者比样本数的1%还少，视为分类任务
            is_classification = unique_values < min(5, len(y_values) * 0.01)
        else:
            is_classification = self.task_type == "classification"

        # 基础参数（随机森林模式）
        base_params = {
            "boosting_type": "rf",
            "num_leaves": 31,  # 默认值，后续会测试不同值
            "n_estimators": 100,
            "subsample": 0.632,  # RF bootstrap 采样率
            "subsample_freq": 1,  # 每棵树都采样
            "colsample_bytree": 0.7,  # 特征子采样
            "importance_type": "gain",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": -1,
            # M4 Pro 性能优化
            "max_bin": 63,
            # 加速 binning（codex 建议）
            "bin_construct_sample_cnt": min(50000, len(y_values)),
        }

        # 分类任务特定参数
        if is_classification:
            base_params["objective"] = "binary"
            base_params["is_unbalance"] = True
            base_params["metric"] = "auc"
            metric = "auc"
        else:
            base_params["objective"] = "regression"
            base_params["metric"] = "rmse"  # 显式设置，否则默认是 l2
            metric = "rmse"

        # 创建 LightGBM Dataset（free_raw_data=False 允许复用）
        lgb_train = lgb.Dataset(X_values, label=y_values, free_raw_data=False)

        # 获取要测试的 num_leaves 值
        if self.param_grid and "num_leaves" in self.param_grid:
            num_leaves_list = self.param_grid["num_leaves"]
        else:
            num_leaves_list = [31, 63]

        # 对每个 num_leaves 值运行 CV，找最佳配置
        # 使用 return_cvbooster=True 复用 CV 模型，省去额外的 lgb.train
        best_score = -np.inf if is_classification else np.inf
        best_cvbooster = None
        first_cvbooster = None  # 保存第一个有效的 cvbooster 作为兜底

        for num_leaves in num_leaves_list:
            params = base_params.copy()
            params["num_leaves"] = num_leaves

            # 使用 LightGBM 原生 CV（复用 binning，显著加速）
            # - early_stopping: 提前终止无效迭代
            # - return_cvbooster: 返回训练好的模型，避免重复训练
            cv_result = lgb.cv(
                params,
                lgb_train,
                num_boost_round=100,
                nfold=self.cv,
                stratified=is_classification,
                seed=self.random_state if self.random_state else 0,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
                return_cvbooster=True,
            )

            # 获取最终分数（early stopping 后的最佳迭代）
            score_key = f"valid {metric}-mean"
            if score_key in cv_result:
                # 保存第一个有效的 cvbooster 作为兜底
                if first_cvbooster is None:
                    first_cvbooster = cv_result["cvbooster"]

                final_score = cv_result[score_key][-1]
                # 对于 AUC 越大越好，对于 RMSE 越小越好
                if is_classification:
                    is_better = final_score > best_score
                else:
                    is_better = final_score < best_score

                if is_better:
                    best_score = final_score
                    best_cvbooster = cv_result["cvbooster"]

        # 确保有可用的 cvbooster
        if best_cvbooster is None:
            best_cvbooster = first_cvbooster

        # 从 CV 模型聚合 feature_importance（省去额外的 lgb.train）
        # 取各 fold 模型的平均重要性，比单模型更稳定
        importances = np.mean(
            [
                booster.feature_importance(importance_type="gain")
                for booster in best_cvbooster.boosters
            ],
            axis=0,
        )
        relevance = importances.astype(np.float64)

        # 清理资源
        del lgb_train, best_cvbooster

        return relevance

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RFCQSelector":
        """
        训练RFCQ特征选择器

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
        # 确保输入是pandas对象
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X必须是pandas.DataFrame")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # 对齐x与y的长度并去除x开头可能存在的空值
        X, y = drop_na_and_align_x_and_y(X, y)

        # 找出数值型变量
        if self.verbose:
            print("➤ 识别数值型变量...")
        self.variables_ = self._find_numerical_variables(X)

        if len(self.variables_) < 2:
            raise ValueError("至少需要2个数值型特征来执行特征选择")

        # 计算相关性
        if self.verbose:
            print("➤ 计算特征与目标变量的相关性(使用随机森林)...")
        X_numeric = X[self.variables_]
        self.relevance_ = self._calculate_relevance(X_numeric, y)

        # 预先获取所有特征数据（复用 X_numeric，避免重复切片）
        # 使用 float32 减少内存占用，相关系数计算精度足够
        X_data = np.asarray(X_numeric.values, dtype=np.float32, order="C")
        n_features = len(self.variables_)

        # 使用布尔掩码代替列表删除操作（O(1) vs O(n)）
        mask = np.ones(n_features, dtype=bool)
        relevance = self.relevance_.copy()

        # 找出最相关的特征
        first_idx = np.argmax(relevance)
        top_feature = self.variables_[first_idx]

        if self.verbose:
            print(
                f"✓ 选择第1个特征: {top_feature} (最大重要性: {relevance[first_idx]:.4f})"
            )

        # 更新状态
        selected_indices = [first_idx]
        mask[first_idx] = False

        # 计算其他特征与最佳特征的冗余度
        if self.verbose:
            print("➤ 计算特征冗余度...")
        X_remaining = X_data[:, mask]
        y_values = X_data[:, first_idx]
        initial_redundance = fast_corrwith_numba(X_remaining, y_values)

        # 初始化 running_mean（完整大小数组，仅 mask=True 位置有效）
        running_mean = np.zeros(n_features, dtype=np.float64)
        running_mean[mask] = initial_redundance

        # 确定要选择的特征数量
        if self.max_features is None:
            n_to_select = max(1, int(0.2 * n_features))
        else:
            n_to_select = min(self.max_features, n_features)

        # 第一轮已经选了一个特征，所以减1
        n_to_select = n_to_select - 1

        if self.verbose:
            print(
                f"➤ 总计选择{n_to_select + 1}个特征 (已选择1个，还需选择{n_to_select}个)..."
            )
            print("➤ 开始MRMR迭代选择过程...")

        redundance_count = 1
        eps = 1e-10

        # 主循环：迭代选择特征（使用掩码，避免 O(n) 的删除操作）
        for _ in tqdm(
            range(n_to_select),
            disable=not self.verbose,
            desc="选择特征",
            unit="特征",
            ncols=100,
        ):
            if not mask.any():
                break

            # 计算 MRMR 分数（仅对剩余特征）
            safe_redundance = np.maximum(running_mean, eps)
            mrmr_scores = np.where(mask, relevance / safe_redundance, -np.inf)
            best_idx = np.argmax(mrmr_scores)

            # 更新状态
            selected_indices.append(best_idx)
            mask[best_idx] = False

            # 如果已经选完了所有特征，退出循环
            if not mask.any():
                break

            # 计算新冗余度（当前选中特征与剩余特征的相关性）
            X_remaining = X_data[:, mask]
            y_values = X_data[:, best_idx]
            new_redundance = fast_corrwith_numba(X_remaining, y_values)

            # 增量更新均值（Welford 公式，仅更新 mask=True 位置）
            redundance_count += 1
            running_mean[mask] += (
                new_redundance - running_mean[mask]
            ) / redundance_count

        # 记录要丢弃的特征（使用索引列表转换为特征名）
        selected = [self.variables_[i] for i in selected_indices]
        self.features_to_drop_ = [f for f in self.variables_ if f not in selected]

        if self.verbose:
            total_features = len(self.variables_)
            selected_count = len(selected)
            dropped_count = len(self.features_to_drop_)
            print(
                f"\n✅ 特征选择完成：从{total_features}个特征中选择了{selected_count}个，舍弃了{dropped_count}个"
            )

        # 统一在 fit 结束时清理内存（避免频繁 gc 影响性能）
        gc.collect()

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
        if self.features_to_drop_ is None:
            raise ValueError("在使用transform之前必须先调用fit")

        return X.drop(columns=self.features_to_drop_)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        训练选择器并转换数据

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
            如果为True，返回特征索引，否则返回布尔掩码

        Returns
        -------
        numpy.ndarray 或 List[int]
            特征选择掩码或索引
        """
        if self.features_to_drop_ is None:
            raise ValueError("在使用get_support之前必须先调用fit")

        mask = np.ones(len(self.variables_), dtype=bool)
        for f in self.features_to_drop_:
            idx = self.variables_.index(f)
            mask[idx] = False

        if indices:
            return np.where(mask)[0].tolist()
        else:
            return mask
