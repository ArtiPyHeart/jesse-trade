from .find_truncation import find_truncation
from .frac_weights import frac_weights
from .apply_weights import apply_weights
import numpy as np


def fracdiff(
    X: np.ndarray,
    order: float = None,
    weights: list = None,
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
    
    weights : list, optional
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
    # 步骤1: 确定权重系数
    if weights is None:
        d = order if order else 0  # 默认值d=0，表示不进行差分
        if isinstance(truncation, int):
            # 使用固定截断长度计算权重
            weights = frac_weights(d, truncation)
        else:  # truncation为None或其他值
            # 自动寻找最优截断点并计算权重
            _, weights = find_truncation(d, tau=tau, mmax=mmax)

    # 步骤2: 确定数据类型
    if dtype is None:
        # 优先使用输入数据的类型，否则默认为float
        dtype = X[0].dtype if isinstance(X[0], float) else float

    # 步骤3: 应用权重进行分数阶差分计算
    weights = np.array(weights)
    if len(X.shape) == 1:
        # 处理1维数组（单个时间序列）
        Z = apply_weights(X.astype(dtype), weights)
    else:
        # 处理2维数组（多个时间序列）
        Z = np.empty(shape=X.shape)
        for j in range(X.shape[1]):
            # 对每一列（每个时间序列）分别应用权重
            Z[:, j] = apply_weights(X[:, j].astype(dtype), weights)

    # 返回分数阶差分后的结果
    return Z
