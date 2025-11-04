import copy
import os
from typing import List, Optional, Union

import numba as nb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV
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
        """
        X_values = X.values
        y_values = y.values

        # 自动检测任务类型
        if self.task_type == "auto":
            unique_values = len(np.unique(y_values))
            # 如果唯一值数量小于5或者比样本数的1%还少，视为分类任务
            is_classification = unique_values < min(5, len(y_values) * 0.01)
        else:
            is_classification = self.task_type == "classification"

        # 确定scoring（如果在构造时未设置）
        if self.scoring is None:
            scoring = "roc_auc" if is_classification else "neg_root_mean_squared_error"
        else:
            scoring = self.scoring

        # 根据任务类型创建模型（使用LightGBM随机森林模式）
        if is_classification:
            model = LGBMClassifier(
                boosting_type="rf",  # 启用随机森林模式
                n_estimators=100,
                num_leaves=31,  # 默认值，可被GridSearchCV覆盖
                subsample=0.632,  # RF bootstrap采样率
                subsample_freq=1,  # 每棵树都采样
                colsample_bytree=1.0,  # 使用所有特征
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,  # 禁用LightGBM内部日志
                # M4 Pro性能优化
                max_bin=63,  # 减少bin数量，在Apple Silicon上显著提速
                histogram_pool_size=512,  # 限制histogram缓存大小(MB)
            )
        else:
            model = LGBMRegressor(
                boosting_type="rf",  # 启用随机森林模式
                n_estimators=100,
                num_leaves=31,  # 默认值，可被GridSearchCV覆盖
                subsample=0.632,  # RF bootstrap采样率
                subsample_freq=1,  # 每棵树都采样
                colsample_bytree=1.0,  # 使用所有特征
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,  # 禁用LightGBM内部日志
                # M4 Pro性能优化
                max_bin=63,  # 减少bin数量，在Apple Silicon上显著提速
                histogram_pool_size=512,  # 限制histogram缓存大小(MB)
            )

        # 设置参数网格
        if self.param_grid:
            param_grid = self.param_grid
        else:
            # 使用num_leaves替代max_depth，对应关系: 2^(depth+1)-1
            # 减少网格搜索空间以提速
            param_grid = {"num_leaves": [31, 63]}  # 对应max_depth 3,4

        # 网格搜索
        cv_model = GridSearchCV(
            model, cv=self.cv, scoring=scoring, param_grid=param_grid
        )

        cv_model.fit(X_values, y_values)

        # 获取特征重要性
        relevance = cv_model.best_estimator_.feature_importances_

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
            print("=" * 80)
            print("【阶段1/5】识别数值型变量")
            print("-" * 80)
        self.variables_ = self._find_numerical_variables(X)

        if len(self.variables_) < 2:
            raise ValueError("至少需要2个数值型特征来执行特征选择")

        if self.verbose:
            print(f"✓ 识别到 {len(self.variables_)} 个数值型变量")
            print(f"✓ 数据集维度: {X.shape[0]} 行 × {len(self.variables_)} 列\n")

        # 计算相关性
        if self.verbose:
            print("【阶段2/5】计算特征与目标的相关性（使用LightGBM随机森林）")
            print("-" * 80)
        X_numeric = X[self.variables_]
        self.relevance_ = self._calculate_relevance(X_numeric, y)

        if self.verbose:
            print(f"✓ 相关性计算完成")
            print(
                f"✓ 重要性范围: 最大={self.relevance_.max():.4f}, "
                f"最小={self.relevance_.min():.4f}, "
                f"平均={self.relevance_.mean():.4f}\n"
            )

        # 预先获取所有特征数据
        X_data = X[self.variables_].values

        # 初始化
        relevance = self.relevance_.copy()
        remaining = copy.deepcopy(self.variables_)

        # 找出最相关的特征
        n = np.argmax(relevance)
        top_feature = remaining[n]

        if self.verbose:
            print("【阶段3/5】选择种子特征")
            print("-" * 80)
            print(
                f"✓ 选择第1个特征（种子特征）: '{top_feature}' (重要性: {self.relevance_[n]:.4f})\n"
            )

        # 更新特征列表
        selected = [top_feature]
        remaining.remove(top_feature)
        relevance = np.delete(relevance, n)

        # 特征的索引映射
        feature_to_idx = {f: i for i, f in enumerate(self.variables_)}

        # 计算其他特征与最佳特征的冗余度
        if self.verbose:
            print("【阶段4/5】计算特征冗余度")
            print("-" * 80)
        top_feature_idx = feature_to_idx[top_feature]
        remaining_indices = [feature_to_idx[f] for f in remaining]
        X_remaining = X_data[:, remaining_indices]
        y_values = X_data[:, top_feature_idx]
        redundance = fast_corrwith_numba(X_remaining, y_values)

        if self.verbose:
            print(f"✓ 已计算 {len(remaining)} 个候选特征与种子特征的冗余度\n")

        # 确定要选择的特征数量
        if self.max_features is None:
            n_to_select = max(1, int(0.2 * len(self.variables_)))
        else:
            n_to_select = min(self.max_features, len(self.variables_))

        # 第一轮已经选了一个特征，所以减1
        n_to_select = n_to_select - 1

        if self.verbose:
            print("【阶段5/5】MRMR迭代选择")
            print("-" * 80)
            print(
                f"✓ 目标: 从 {len(self.variables_)} 个特征中选择 {n_to_select + 1} 个"
            )
            print(f"✓ 已选: 1 个，待选: {n_to_select} 个\n")

        # 主循环：迭代选择特征
        # 用于收集选择的特征信息（仅在verbose时记录详情）
        selection_details = []

        for i in tqdm(
            range(n_to_select),
            disable=not self.verbose,
            desc="迭代选择",
            unit="特征",
            ncols=100,
            leave=True,
        ):
            if len(remaining) == 0:
                break

            if i == 0:
                # 第一轮迭代，冗余度是一维的
                # 计算MRMR
                eps = 1e-10
                safe_redundance = np.maximum(redundance, eps)
                mrmr_scores = relevance / safe_redundance
                n = np.argmax(mrmr_scores)

                # 更新特征列表
                feature = remaining[n]
                feature_idx = feature_to_idx[feature]

                # 记录详情而非立即打印
                if self.verbose:
                    selection_details.append({
                        'rank': i+2,
                        'feature': feature,
                        'relevance': relevance[n],
                        'redundance': redundance[n],
                        'mrmr': mrmr_scores[n]
                    })

                selected.append(feature)
                remaining.remove(feature)

                # 更新索引
                remaining_indices.remove(feature_idx)

                relevance = np.delete(relevance, n)
                redundance = np.delete(redundance, n)
            else:
                # 后续迭代，冗余度是二维的
                # 计算平均冗余度
                mean_redundance = np.mean(redundance, axis=0)

                # 计算MRMR
                eps = 1e-10
                safe_redundance = np.maximum(mean_redundance, eps)
                mrmr_scores = relevance / safe_redundance
                n = np.argmax(mrmr_scores)

                # 更新特征列表
                feature = remaining[n]
                feature_idx = feature_to_idx[feature]

                # 记录详情而非立即打印
                if self.verbose:
                    selection_details.append({
                        'rank': i+2,
                        'feature': feature,
                        'relevance': relevance[n],
                        'redundance': mean_redundance[n],
                        'mrmr': mrmr_scores[n]
                    })

                selected.append(feature)
                remaining.remove(feature)

                # 更新索引
                remaining_indices.remove(feature_idx)

                relevance = np.delete(relevance, n)
                redundance = np.delete(redundance, n, axis=1)

            # 如果已经选完了所有特征，退出循环
            if len(remaining) == 0:
                break

            # 计算新的冗余度
            X_remaining = X_data[:, remaining_indices]
            y_values = X_data[:, feature_idx]
            new_redundance = fast_corrwith_numba(X_remaining, y_values)

            # 第一次添加时，创建2D数组
            if i == 0:
                redundance = np.vstack(
                    [redundance[np.newaxis, :], new_redundance[np.newaxis, :]]
                )
            else:
                # 添加新的冗余度
                redundance = np.vstack([redundance, new_redundance[np.newaxis, :]])

        # 记录要丢弃的特征
        self.features_to_drop_ = [f for f in self.variables_ if f not in selected]

        if self.verbose:
            total_features = len(self.variables_)
            selected_count = len(selected)
            dropped_count = len(self.features_to_drop_)

            print("\n" + "=" * 80)
            print("【特征选择完成】")
            print("=" * 80)
            print(
                f"✓ 从 {total_features} 个特征中选择了 {selected_count} 个 "
                f"(保留率: {selected_count/total_features*100:.1f}%)"
            )
            print(f"✓ 舍弃了 {dropped_count} 个冗余/低重要性特征")

            # 显示选中特征的统计信息
            selected_indices = [feature_to_idx[f] for f in selected]
            selected_importance = self.relevance_[selected_indices]
            print(
                f"✓ 重要性范围: 最大={selected_importance.max():.4f}, "
                f"最小={selected_importance.min():.4f}, "
                f"平均={selected_importance.mean():.4f}"
            )

            # 只显示前5个和后5个特征
            if selected_count <= 10:
                # 如果特征数不多，全部显示
                print(f"\n【选中特征】(共{selected_count}个):")
                for idx, feat in enumerate(selected, 1):
                    feat_idx = feature_to_idx[feat]
                    print(f"  {idx}. {feat} (重要性: {self.relevance_[feat_idx]:.4f})")
            else:
                # 只显示前5和后5
                print(f"\n【选中特征】(共{selected_count}个，显示前5和后5):")
                for idx in range(5):
                    feat = selected[idx]
                    feat_idx = feature_to_idx[feat]
                    print(f"  {idx+1}. {feat} (重要性: {self.relevance_[feat_idx]:.4f})")
                print(f"  ... 中间 {selected_count - 10} 个特征 ...")
                for idx in range(selected_count - 5, selected_count):
                    feat = selected[idx]
                    feat_idx = feature_to_idx[feat]
                    print(f"  {idx+1}. {feat} (重要性: {self.relevance_[feat_idx]:.4f})")

            # 显示迭代选择的部分详情（如果有记录）
            if selection_details:
                if len(selection_details) <= 5:
                    print(f"\n【迭代选择详情】(共{len(selection_details)}个):")
                    for detail in selection_details:
                        print(
                            f"  第{detail['rank']}个: {detail['feature']} "
                            f"(重要性={detail['relevance']:.4f}, "
                            f"冗余度={detail['redundance']:.4f}, "
                            f"MRMR={detail['mrmr']:.4f})"
                        )
                else:
                    print(f"\n【迭代选择详情】(共{len(selection_details)}个，显示前3和后2):")
                    for i in range(3):
                        detail = selection_details[i]
                        print(
                            f"  第{detail['rank']}个: {detail['feature']} "
                            f"(重要性={detail['relevance']:.4f}, "
                            f"冗余度={detail['redundance']:.4f}, "
                            f"MRMR={detail['mrmr']:.4f})"
                        )
                    print(f"  ... 中间 {len(selection_details) - 5} 个特征 ...")
                    for i in range(len(selection_details) - 2, len(selection_details)):
                        detail = selection_details[i]
                        print(
                            f"  第{detail['rank']}个: {detail['feature']} "
                            f"(重要性={detail['relevance']:.4f}, "
                            f"冗余度={detail['redundance']:.4f}, "
                            f"MRMR={detail['mrmr']:.4f})"
                        )

            print("=" * 80 + "\n")

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
