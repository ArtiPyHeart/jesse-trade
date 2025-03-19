"""
FCQSelector的Cython实现版本，用于提高特征选择的计算效率

此实现基于原始FCQSelector版本，但使用Cython进行性能优化
"""

import copy
import os
from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from tqdm.auto import tqdm

cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt

# 让Cython可以操作NumPy数组
np.import_array()

# 定义类型
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)  # 禁用边界检查
@cython.wraparound(False)   # 禁用负索引
@cython.cdivision(True)     # 禁用除零检查
def fast_corrwith_cython(double[:, :] X_values, double[:] y_values):
    """
    使用Cython加速计算X的每一列与y的相关系数

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
    cdef Py_ssize_t n_samples = X_values.shape[0]
    cdef Py_ssize_t n_features = X_values.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n_features, dtype=np.float64)
    
    # 计算y的均值和标准差
    cdef double y_mean = 0.0
    cdef double y_std = 0.0
    cdef double y_sum = 0.0
    cdef double y_sq_sum = 0.0
    cdef Py_ssize_t i, j
    
    # 计算y的均值
    for i in range(n_samples):
        y_sum += y_values[i]
    y_mean = y_sum / n_samples
    
    # 计算y的标准差
    for i in range(n_samples):
        y_sq_sum += (y_values[i] - y_mean) ** 2
    y_std = sqrt(y_sq_sum / n_samples)
    
    # 如果y是常数，返回全零数组
    if y_std == 0:
        return np.asarray(result)
    
    # 为每个特征计算相关系数
    cdef double x_mean, x_std, x_sum, x_sq_sum, corr_sum, x_val, y_norm
    
    # 预先计算规范化的y值
    cdef double[:] y_norm_array = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        y_norm_array[i] = (y_values[i] - y_mean) / y_std
    
    for j in range(n_features):
        x_sum = 0.0
        x_sq_sum = 0.0
        
        # 计算x的均值
        for i in range(n_samples):
            x_sum += X_values[i, j]
        x_mean = x_sum / n_samples
        
        # 计算x的标准差
        for i in range(n_samples):
            x_sq_sum += (X_values[i, j] - x_mean) ** 2
        x_std = sqrt(x_sq_sum / n_samples)
        
        # 如果x是常数，相关系数为0
        if x_std == 0:
            result[j] = 0.0
            continue
        
        # 计算相关系数
        corr_sum = 0.0
        for i in range(n_samples):
            x_val = (X_values[i, j] - x_mean) / x_std
            corr_sum += x_val * y_norm_array[i]
        
        # 相关系数是协方差除以标准差的乘积
        corr = corr_sum / n_samples
        result[j] = fabs(corr)  # 取绝对值
    
    return np.asarray(result)


cdef class CythonFCQSelector:
    """
    使用Cython优化的FCQ特征选择器

    FCQ = F-statistic for relevance, Correlation for redundancy, Quotient for combining
    """
    cdef public:
        object max_features  # Optional[int]
        bint regression
        bint verbose
        object features_to_drop_  # List[str]
        object variables_  # List[str]
        np.ndarray relevance_  # np.ndarray

    def __init__(self, max_features=None, regression=False, verbose=True):
        self.max_features = max_features
        self.regression = regression
        self.verbose = verbose
        self.features_to_drop_ = None
        self.variables_ = None
        self.relevance_ = None

    def _find_numerical_variables(self, X):
        """找出数据框中的数值型变量"""
        # 使用列表而不是字符串来指定数值类型
        include_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        return [col for col in X.columns if X[col].dtype.name in include_types]

    def _calculate_relevance(self, X, y):
        """计算特征与目标的相关性（使用F统计量）"""
        X_values = X.values
        y_values = y.values

        if self.regression:
            # 回归问题使用f_regression
            f_values, _ = f_regression(X_values, y_values)
        else:
            # 分类问题使用f_classif
            f_values, _ = f_classif(X_values, y_values)

        return f_values

    def fit(self, X, y):
        """
        训练FCQ特征选择器

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

        # 找出数值型变量
        if self.verbose:
            print(f"➤ 识别数值型变量...")
        self.variables_ = self._find_numerical_variables(X)

        if len(self.variables_) < 2:
            raise ValueError("至少需要2个数值型特征来执行特征选择")

        # 计算相关性
        if self.verbose:
            print(f"➤ 计算特征与目标变量的相关性...")
        X_numeric = X[self.variables_]
        self.relevance_ = self._calculate_relevance(X_numeric, y)

        # 预先获取所有特征数据并转为double类型
        X_data = X[self.variables_].values.astype(np.float64)

        # 初始化
        relevance = self.relevance_.copy()
        remaining = copy.deepcopy(self.variables_)

        # 找出最相关的特征
        n = np.argmax(relevance)
        top_feature = remaining[n]

        if self.verbose:
            print(f"✓ 选择第1个特征: {top_feature} (最大F值: {self.relevance_[n]:.4f})")

        # 更新特征列表
        selected = [top_feature]
        remaining.remove(top_feature)
        relevance = np.delete(relevance, n)

        # 特征的索引映射
        feature_to_idx = {f: i for i, f in enumerate(self.variables_)}

        # 计算其他特征与最佳特征的冗余度
        if self.verbose:
            print(f"➤ 计算特征冗余度...")
        top_feature_idx = feature_to_idx[top_feature]
        remaining_indices = [feature_to_idx[f] for f in remaining]
        X_remaining = X_data[:, remaining_indices]
        y_values = X_data[:, top_feature_idx]
        redundance = fast_corrwith_cython(X_remaining, y_values)

        # 确定要选择的特征数量
        if self.max_features is None:
            n_to_select = max(1, int(0.2 * len(self.variables_)))
        else:
            n_to_select = min(self.max_features, len(self.variables_))

        # 第一轮已经选了一个特征，所以减1
        n_to_select = n_to_select - 1

        if self.verbose:
            print(
                f"➤ 总计选择{n_to_select+1}个特征 (已选择1个，还需选择{n_to_select}个)..."
            )
            print(f"➤ 开始MRMR迭代选择过程...")

        # 主循环：迭代选择特征
        for i in tqdm(
            range(n_to_select),
            disable=not self.verbose,
            desc="选择特征",
            unit="特征",
            ncols=100,
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
            new_redundance = fast_corrwith_cython(X_remaining, y_values)

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
            print(
                f"\n✅ 特征选择完成：从{total_features}个特征中选择了{selected_count}个，舍弃了{dropped_count}个"
            )
            print(f"✅ 选择的特征: {selected}")

        return self

    def transform(self, X):
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

    def fit_transform(self, X, y):
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

    def get_support(self, indices=False):
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