"""
Ripser - 持久同调计算

基于 Rust 的高性能 Vietoris-Rips 持久同调实现。

主要功能：
- 从点云或距离矩阵计算持久性同调
- 支持多种距离度量（欧几里得、曼哈顿、切比雪夫）
- 计算 0 维（连通分量）、1 维（环）、2 维（空洞）等拓扑特征

性能：相比 ripser-py，提供 2-5x 加速（取决于数据规模）
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Literal, Dict, List

from .._core import ripser_compute, ripser_compute_from_distance_matrix


def ripser(
    data: npt.NDArray[np.float64],
    max_dim: int = 1,
    threshold: Optional[float] = None,
    metric: Literal['euclidean', 'manhattan', 'chebyshev'] = 'euclidean',
    distance_matrix: bool = False,
    collapse_edges: bool = True,
) -> Dict[str, Union[List[npt.NDArray[np.float64]], int, float]]:
    """
    计算 Vietoris-Rips 持久同调

    Args:
        data: 点云数据 (n_points, n_features) 或压缩距离矩阵 (n*(n-1)/2,)
        max_dim: 计算的最大维度（0=连通分量, 1=环, 2=空洞）
            推荐值：1（大多数应用），2（需要检测空洞时）
        threshold: 距离阈值，超过此距离的边不考虑
            None 表示无限制（可能很慢），建议设置合理阈值
        metric: 距离度量
            - 'euclidean': 欧几里得距离（默认）
            - 'manhattan': 曼哈顿距离
            - 'chebyshev': 切比雪夫距离
        distance_matrix: data 是否为距离矩阵
            False: data 是点云 (n, d)
            True: data 是压缩距离矩阵 (n*(n-1)/2,)
        collapse_edges: 是否过滤零长度持久性对（birth == death）
            True: 过滤零长度对，符合标准实践（默认）
            False: 保留所有持久性对，包括零长度

    Returns:
        字典包含：
        - 'persistence': 各维度的持久性对列表
            每个元素是 (n_pairs, 2) 的 numpy 数组，列为 [birth, death]
            death = inf 表示无穷长的持久性特征
        - 'num_points': 点数量
        - 'max_dim': 最大维度
        - 'threshold': 使用的阈值

    Raises:
        ValueError: 非法输入
        RuntimeError: 计算失败

    Examples:
        >>> # 简单的 3 点示例
        >>> points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        >>> result = ripser(points, max_dim=1)
        >>> result['persistence'][0]  # 0 维持久性
        array([[0., 1.],
               [0., 1.],
               [0., inf]])

        >>> # 使用阈值限制计算
        >>> result = ripser(points, max_dim=1, threshold=2.0)

        >>> # 从距离矩阵计算
        >>> from scipy.spatial.distance import pdist
        >>> distances = pdist(points, metric='euclidean')
        >>> result = ripser(distances, max_dim=1, distance_matrix=True)
    """
    # ========================================================================
    # 参数验证（Fail Fast）
    # ========================================================================

    if not isinstance(data, np.ndarray):
        raise ValueError(f"data must be numpy array, got {type(data)}")

    if data.dtype != np.float64:
        raise ValueError(f"data must be float64, got {data.dtype}")

    if max_dim < 0:
        raise ValueError(f"max_dim must be non-negative, got {max_dim}")

    # 支持 0-2 维同调，更高维度会有性能警告
    if max_dim > 2:
        import warnings
        warnings.warn(
            f"max_dim > 2 may be very slow and memory-intensive. "
            f"Consider using max_dim=1 or 2 for most applications.",
            UserWarning
        )

    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    if metric not in ['euclidean', 'manhattan', 'chebyshev']:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            "Supported: 'euclidean', 'manhattan', 'chebyshev'"
        )

    # 验证数据形状
    if distance_matrix:
        if data.ndim != 1:
            raise ValueError(
                f"distance_matrix=True requires 1D array, got shape {data.shape}"
            )

        # 验证是否为有效的压缩距离矩阵长度
        n_distances = len(data)
        # n*(n-1)/2 = n_distances => n^2 - n - 2*n_distances = 0
        n = int((1 + np.sqrt(1 + 8 * n_distances)) / 2)
        if n * (n - 1) // 2 != n_distances:
            raise ValueError(
                f"Invalid distance matrix size: {n_distances}. "
                f"Expected n*(n-1)/2 for some integer n."
            )

        if n < 2:
            raise ValueError(f"Need at least 2 points, got {n}")

    else:
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2D array (n_points, n_features), got shape {data.shape}"
            )

        n_points, n_features = data.shape

        if n_points < 2:
            raise ValueError(f"Need at least 2 points, got {n_points}")

        if n_features < 1:
            raise ValueError(f"Need at least 1 feature, got {n_features}")

        # 检查 NaN/Inf
        if np.any(np.isnan(data)):
            raise ValueError("data contains NaN values")

        if np.any(np.isinf(data)):
            raise ValueError("data contains Inf values")

    # ========================================================================
    # 调用 Rust 核心
    # ========================================================================

    if distance_matrix:
        result = ripser_compute_from_distance_matrix(
            data,
            max_dim=max_dim,
            threshold=threshold,
        )
    else:
        result = ripser_compute(
            data,
            max_dim=max_dim,
            threshold=threshold,
            metric=metric,
        )

    # ========================================================================
    # 结果验证
    # ========================================================================

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected result type: {type(result)}")

    if 'persistence' not in result:
        raise RuntimeError("Result missing 'persistence' key")

    # 验证每个维度的持久性对
    for dim, pairs in enumerate(result['persistence']):
        if not isinstance(pairs, np.ndarray):
            raise RuntimeError(
                f"Dimension {dim} persistence is not ndarray: {type(pairs)}"
            )

        if pairs.ndim != 2 or (pairs.shape[1] != 2 and pairs.shape[0] > 0):
            raise RuntimeError(
                f"Dimension {dim} persistence has invalid shape: {pairs.shape}. "
                "Expected (n, 2)"
            )

        # 检查 birth <= death (除了 inf)
        finite_mask = np.isfinite(pairs[:, 1])
        if np.any(pairs[finite_mask, 0] > pairs[finite_mask, 1]):
            raise RuntimeError(
                f"Dimension {dim}: found birth > death (invalid persistence pair)"
            )

    # ========================================================================
    # 过滤零长度持久性对（可选）
    # ========================================================================

    if collapse_edges:
        # 过滤掉 birth == death 的持久性对
        filtered_persistence = []
        for dim, pairs in enumerate(result['persistence']):
            if pairs.shape[0] == 0:
                filtered_persistence.append(pairs)
                continue

            # 保留 birth < death 或 death == inf 的持久性对
            mask = (pairs[:, 0] < pairs[:, 1]) | np.isinf(pairs[:, 1])
            filtered_pairs = pairs[mask]
            filtered_persistence.append(filtered_pairs)

        result['persistence'] = filtered_persistence

    return result


# ============================================================================
# 辅助函数
# ============================================================================


def filter_persistence(
    pairs: npt.NDArray[np.float64],
    min_persistence: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    过滤持久性对，移除持久性太短的特征

    Args:
        pairs: (n, 2) 持久性对数组 [birth, death]
        min_persistence: 最小持久性长度（death - birth）

    Returns:
        过滤后的持久性对

    Example:
        >>> pairs = np.array([[0, 0.1], [0, 1.0], [0.5, 0.6]])
        >>> filter_persistence(pairs, min_persistence=0.5)
        array([[0. , 1. ]])
    """
    if not isinstance(pairs, np.ndarray):
        raise ValueError(f"pairs must be numpy array, got {type(pairs)}")

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (n, 2), got {pairs.shape}")

    if min_persistence < 0:
        raise ValueError(f"min_persistence must be non-negative, got {min_persistence}")

    # 保留无穷长的和持久性 >= min_persistence 的
    persistence = pairs[:, 1] - pairs[:, 0]
    mask = np.isinf(pairs[:, 1]) | (persistence >= min_persistence)

    return pairs[mask]


def get_betti_numbers(result: Dict, dim: int) -> int:
    """
    获取指定维度的 Betti 数（无穷长持久性特征的数量）

    Betti 数表示：
    - B0: 连通分量数量
    - B1: 环的数量
    - B2: 空洞的数量

    Args:
        result: ripser() 的返回结果
        dim: 维度 (0, 1, 2, ...)

    Returns:
        Betti 数

    Example:
        >>> result = ripser(points)
        >>> get_betti_numbers(result, 0)  # 连通分量数
        1
    """
    if dim >= len(result['persistence']):
        return 0

    pairs = result['persistence'][dim]
    if len(pairs) == 0:
        return 0

    # 统计 death = inf 的数量
    return np.sum(np.isinf(pairs[:, 1]))
