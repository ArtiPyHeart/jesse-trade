"""
Ripser 端到端集成测试

测试 Rust Ripser 实现的正确性和与标准实现的一致性。
"""

import numpy as np
import pytest

from pyrs_indicators.topology import ripser, filter_persistence, get_betti_numbers


# ============================================================================
# 基础功能测试
# ============================================================================


def test_ripser_simple_triangle():
    """测试简单三角形的持久性"""
    # 三个顶点形成三角形
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

    # 注意：当前版本只支持 0 维同调
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ripser(points, max_dim=0)

    # 验证结果结构
    assert 'persistence' in result
    assert 'num_points' in result
    assert 'max_dim' in result
    assert 'threshold' in result

    # 验证点数
    assert result['num_points'] == 3
    assert result['max_dim'] == 0

    # 验证 0 维持久性
    dim_0 = result['persistence'][0]
    assert dim_0.shape[1] == 2  # [birth, death]
    assert dim_0.shape[0] == 3  # 3 个持久性对

    # 应该有 1 个无穷长的分量（最终只有 1 个连通分量）
    infinite_count = np.sum(np.isinf(dim_0[:, 1]))
    assert infinite_count == 1

    print(f"✓ Triangle test passed")
    print(f"  0-dim pairs: {dim_0.shape[0]}, infinite: {infinite_count}")


def test_ripser_disconnected():
    """测试不连通图"""
    # 两个分离的点对
    points = np.array([
        [0, 0],
        [0.1, 0],
        [10, 10],
        [10.1, 10],
    ], dtype=np.float64)

    result = ripser(points, max_dim=0, threshold=1.0)

    # 应该有 2 个连通分量
    dim_0 = result['persistence'][0]
    infinite_count = np.sum(np.isinf(dim_0[:, 1]))
    assert infinite_count == 2

    print(f"✓ Disconnected test passed")
    print(f"  Connected components: {infinite_count}")


def test_ripser_with_threshold():
    """测试距离阈值"""
    points = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=np.float64)

    # 使用阈值 2.0，远离的点不会连接
    result = ripser(points, max_dim=0, threshold=2.0)

    dim_0 = result['persistence'][0]
    infinite_count = np.sum(np.isinf(dim_0[:, 1]))

    # 应该有 2 个连通分量（前 3 个点 + 最后 1 个点）
    assert infinite_count == 2

    print(f"✓ Threshold test passed")


def test_ripser_distance_matrix():
    """测试从距离矩阵计算"""
    from scipy.spatial.distance import pdist

    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

    # 计算压缩距离矩阵
    distances = pdist(points, metric='euclidean')

    result = ripser(distances, max_dim=0, distance_matrix=True)

    # 验证结果
    assert result['num_points'] == 3

    dim_0 = result['persistence'][0]
    assert dim_0.shape[0] == 3

    print(f"✓ Distance matrix test passed")


# ============================================================================
# 参数验证测试
# ============================================================================


def test_ripser_validation_errors():
    """测试参数验证"""
    points = np.array([[0, 0], [1, 0]], dtype=np.float64)

    # 错误的数据类型
    with pytest.raises(ValueError, match="must be numpy array"):
        ripser([[0, 0], [1, 0]])

    # 错误的 dtype
    with pytest.raises(ValueError, match="must be float64"):
        ripser(np.array([[0, 0], [1, 0]], dtype=np.int32))

    # 负的 max_dim
    with pytest.raises(ValueError, match="must be non-negative"):
        ripser(points, max_dim=-1)

    # 超过推荐维度（应该产生警告，但不抛出异常）
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ripser(points, max_dim=3)
        assert len(w) == 1
        assert "may be very slow" in str(w[0].message)

    # 负的阈值
    with pytest.raises(ValueError, match="must be non-negative"):
        ripser(points, threshold=-1.0)

    # 未知的 metric
    with pytest.raises(ValueError, match="Unknown metric"):
        ripser(points, metric='unknown')

    # 错误的形状
    with pytest.raises(ValueError, match="must be 2D array"):
        ripser(np.array([0, 1, 2], dtype=np.float64))

    # 点太少
    with pytest.raises(ValueError, match="Need at least 2 points"):
        ripser(np.array([[0, 0]], dtype=np.float64))

    # NaN 值
    with pytest.raises(ValueError, match="contains NaN"):
        ripser(np.array([[0, 0], [np.nan, 0]], dtype=np.float64))

    # Inf 值
    with pytest.raises(ValueError, match="contains Inf"):
        ripser(np.array([[0, 0], [np.inf, 0]], dtype=np.float64))

    print(f"✓ Validation tests passed")


# ============================================================================
# 辅助函数测试
# ============================================================================


def test_filter_persistence():
    """测试持久性过滤"""
    pairs = np.array([
        [0.0, 0.1],
        [0.0, 1.0],
        [0.5, 0.6],
        [0.0, np.inf],
    ])

    # 过滤掉持久性 < 0.5 的
    filtered = filter_persistence(pairs, min_persistence=0.5)

    assert len(filtered) == 2
    assert np.isinf(filtered[1, 1])  # 无穷长的保留
    assert filtered[0, 1] - filtered[0, 0] >= 0.5

    print(f"✓ Filter persistence test passed")


def test_get_betti_numbers():
    """测试 Betti 数计算"""
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    result = ripser(points, max_dim=0)

    # B0: 连通分量数
    b0 = get_betti_numbers(result, 0)
    assert b0 == 1  # 只有 1 个连通分量

    print(f"✓ Betti numbers test passed")
    print(f"  B0 (connected components): {b0}")


# ============================================================================
# 不同度量测试
# ============================================================================


def test_different_metrics():
    """测试不同的距离度量"""
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

    for metric in ['euclidean', 'manhattan', 'chebyshev']:
        result = ripser(points, max_dim=0, metric=metric)
        assert result['num_points'] == 3
        print(f"  {metric}: OK")

    print(f"✓ Different metrics test passed")


# ============================================================================
# 数值精度测试
# ============================================================================


def test_numerical_stability():
    """测试数值稳定性"""
    # 使用浮点数可能有精度问题的例子
    points = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],
    ], dtype=np.float64)

    result = ripser(points, max_dim=0)

    # 验证 birth <= death
    for dim in range(len(result['persistence'])):
        pairs = result['persistence'][dim]
        if len(pairs) > 0:
            finite_mask = np.isfinite(pairs[:, 1])
            assert np.all(pairs[finite_mask, 0] <= pairs[finite_mask, 1])

    print(f"✓ Numerical stability test passed")


# ============================================================================
# 边界情况测试
# ============================================================================


def test_edge_cases():
    """测试边界情况"""
    # 最小点数
    points = np.array([[0, 0], [1, 0]], dtype=np.float64)
    result = ripser(points, max_dim=0)
    assert result['num_points'] == 2

    # 单一特征
    points = np.array([[0], [1], [2]], dtype=np.float64).reshape(-1, 1)
    result = ripser(points, max_dim=0)
    assert result['num_points'] == 3

    # 阈值为 0（所有点独立）
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    result = ripser(points, max_dim=0, threshold=0.0)
    dim_0 = result['persistence'][0]
    infinite_count = np.sum(np.isinf(dim_0[:, 1]))
    assert infinite_count == 3  # 3 个独立分量

    print(f"✓ Edge cases test passed")


# ============================================================================
# 主测试入口
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("Ripser Integration Tests")
    print("=" * 70)

    test_ripser_simple_triangle()
    test_ripser_disconnected()
    test_ripser_with_threshold()
    test_ripser_distance_matrix()
    test_ripser_validation_errors()
    test_filter_persistence()
    test_get_betti_numbers()
    test_different_metrics()
    test_numerical_stability()
    test_edge_cases()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
