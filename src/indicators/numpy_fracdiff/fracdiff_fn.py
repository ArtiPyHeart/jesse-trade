import numpy as np
from functools import lru_cache
from .find_truncation import find_truncation
from .frac_weights import frac_weights
from .apply_weights import apply_weights, apply_weights_2d


@lru_cache(maxsize=128)
def get_cached_weights(d: float, truncation: int = None, tau: float = 1e-5, mmax: int = 20000):
    """缓存权重计算结果，避免重复计算相同参数的权重"""
    if truncation is not None:
        weights = frac_weights(d, truncation)
    else:
        _, weights = find_truncation(d, tau=tau, mmax=mmax)
    return np.array(weights)


def fracdiff(
    X: np.ndarray,
    order: float = None,
    weights: np.ndarray = None,
    truncation: int = None,
    tau: float = 1e-5,
    mmax: int = 20000,
    dtype=None,
) -> np.ndarray:
    """
    计算时间序列的分数阶差分（Fractional Differentiation）
    
    分数阶差分是一种在保留记忆性的同时实现平稳性的方法，
    特别适用于金融时间序列数据的特征工程。
    
    参数:
    ----------
    X : np.ndarray
        输入时间序列数据，可以是1维或2维数组
        - 1维: 单个时间序列
        - 2维: 多个时间序列（每列为一个序列）
    
    order : float, optional
        分数阶差分的阶数（d值）
        - d=0: 无差分（原始序列）
        - 0<d<1: 分数阶差分（平衡记忆性和平稳性）
        - d=1: 一阶差分（完全平稳但失去记忆）
    
    weights : np.ndarray, optional
        预计算的权重系数，如果提供则忽略order参数
    
    truncation : int, optional
        权重截断长度，控制历史数据的使用范围
        - None: 自动通过tau参数找到最优截断点
        - int: 固定截断长度
    
    tau : float, default=1e-5
        权重截断阈值，当权重小于此值时停止计算
        仅在truncation=None时使用
    
    mmax : int, default=20000
        最大权重数量限制，防止计算过长
    
    dtype : type, optional
        输出数据类型，默认使用输入数据类型
    
    返回:
    ----------
    np.ndarray
        分数阶差分后的时间序列，形状与输入X相同
    """
    # 步骤1: 获取权重（使用缓存）
    if weights is None:
        d = order if order else 0
        weights = get_cached_weights(d, truncation, tau, mmax)
    elif not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    
    # 步骤2: 确定数据类型（优化类型检查）
    if dtype is None:
        dtype = X.dtype if X.dtype in [np.float32, np.float64] else np.float64
    
    # 步骤3: 应用权重
    if X.ndim == 1:
        # 1D处理
        if X.dtype != dtype:
            X = X.astype(dtype)
        Z = apply_weights(X, weights)
    elif X.ndim == 2:
        # 2D向量化处理
        if X.dtype != dtype:
            X = X.astype(dtype)
        Z = apply_weights_2d(X, weights)
    else:
        raise ValueError(f"Input must be 1D or 2D array, got {X.ndim}D")
    
    return Z
