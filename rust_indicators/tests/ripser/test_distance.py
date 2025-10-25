"""
距离矩阵测试

对比 Rust 实现与 Python scipy 的计算结果，验证数值精度。

由于 ripser 模块还在开发中，暂时只测试数学正确性（使用纯 Python 参考实现）。
后续迭代会添加 Rust FFI 绑定。
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform


# ============================================================================
# Python 参考实现
# ============================================================================


def python_compressed_distance_matrix(points, metric='euclidean'):
    """
    Python 参考实现：压缩距离矩阵（下三角存储）

    Args:
        points: 点云数据，shape = (n, d)
        metric: 距离度量 ('euclidean', 'cityblock', 'chebyshev')

    Returns:
        (n, n) 距离矩阵（完整矩阵，用于验证）
    """
    # 使用 scipy.spatial.distance.pdist 计算压缩距离向量
    condensed = pdist(points, metric=metric)

    # 转换为完整方阵
    full_matrix = squareform(condensed)

    return full_matrix


def python_sparse_distance_matrix(points, metric='euclidean', threshold=None):
    """
    Python 参考实现：稀疏距离矩阵（邻接表）

    Args:
        points: 点云数据，shape = (n, d)
        metric: 距离度量
        threshold: 距离阈值（仅保留 ≤ threshold 的边）

    Returns:
        neighbors: 列表，neighbors[i] = [(j, distance), ...]
    """
    n = len(points)
    neighbors = [[] for _ in range(n)]

    # 计算全部距离矩阵
    dist_matrix = python_compressed_distance_matrix(points, metric)

    for i in range(n):
        for j in range(n):
            if i != j:
                d = dist_matrix[i, j]
                if threshold is None or d <= threshold:
                    neighbors[i].append((j, d))

    # 排序
    for i in range(n):
        neighbors[i].sort()

    return neighbors


# ============================================================================
# 测试用例
# ============================================================================


def test_compressed_matrix_basic():
    """基本功能测试：小规模点云"""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

    # Python 参考实现
    dist_py = python_compressed_distance_matrix(points, metric='euclidean')

    # 验证对角线为 0
    for i in range(4):
        assert abs(dist_py[i, i]) < 1e-6, f"Diagonal d({i},{i}) should be 0"

    # 验证对称性
    for i in range(4):
        for j in range(4):
            assert abs(dist_py[i, j] - dist_py[j, i]) < 1e-6, \
                f"Symmetry failed: d({i},{j}) != d({j},{i})"

    # 验证已知距离
    # 点0 (0,0) 到点1 (1,0) 的距离应该是 1.0
    assert abs(dist_py[0, 1] - 1.0) < 1e-6, "d(0,1) should be 1.0"

    # 点0 (0,0) 到点3 (1,1) 的距离应该是 sqrt(2)
    assert abs(dist_py[0, 3] - np.sqrt(2)) < 1e-6, "d(0,3) should be sqrt(2)"


def test_compressed_matrix_against_scipy():
    """
    全面对比 Python 实现与 scipy

    使用随机点云进行全面验证
    """
    np.random.seed(42)
    points = np.random.randn(20, 3)

    # Python 实现
    dist_py = python_compressed_distance_matrix(points, metric='euclidean')

    # 直接使用 scipy
    dist_scipy = squareform(pdist(points, metric='euclidean'))

    # 验证所有元素
    errors = []
    for i in range(20):
        for j in range(20):
            diff = abs(dist_py[i, j] - dist_scipy[i, j])
            if diff > 1e-6:
                errors.append(f"d({i},{j}): py={dist_py[i, j]:.6f}, scipy={dist_scipy[i, j]:.6f}")

    assert len(errors) == 0, f"Found {len(errors)} errors:\n" + "\n".join(errors[:10])


def test_metrics_comparison():
    """测试不同距离度量"""
    points = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0]
    ])

    # Euclidean
    dist_euclidean = python_compressed_distance_matrix(points, metric='euclidean')
    # 点0到点1的欧几里得距离: sqrt(3^2 + 4^2) = 5.0
    assert abs(dist_euclidean[0, 1] - 5.0) < 1e-6, "Euclidean distance failed"

    # Manhattan (cityblock)
    dist_manhattan = python_compressed_distance_matrix(points, metric='cityblock')
    # 点0到点1的曼哈顿距离: |3-0| + |4-0| = 7.0
    assert abs(dist_manhattan[0, 1] - 7.0) < 1e-6, "Manhattan distance failed"

    # Chebyshev
    dist_chebyshev = python_compressed_distance_matrix(points, metric='chebyshev')
    # 点0到点1的切比雪夫距离: max(|3-0|, |4-0|) = 4.0
    assert abs(dist_chebyshev[0, 1] - 4.0) < 1e-6, "Chebyshev distance failed"


def test_compressed_matrix_large():
    """大规模点云测试"""
    np.random.seed(123)
    points = np.random.randn(100, 5)

    dist_py = python_compressed_distance_matrix(points, metric='euclidean')
    dist_scipy = squareform(pdist(points, metric='euclidean'))

    # 验证最大误差
    max_error = np.max(np.abs(dist_py - dist_scipy))
    assert max_error < 1e-6, f"Max error {max_error} exceeds threshold 1e-6"


def test_sparse_matrix_basic():
    """稀疏矩阵基本测试"""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0]  # 远离其他点
    ])

    # 阈值 = 2.0，点3不应该有任何邻居
    neighbors = python_sparse_distance_matrix(points, metric='euclidean', threshold=2.0)

    # 点3距离其他点很远，应该没有邻居
    assert len(neighbors[3]) == 0, "Point 3 should have no neighbors"

    # 点0、1、2应该互相连接
    assert len(neighbors[0]) > 0, "Point 0 should have neighbors"
    assert len(neighbors[1]) > 0, "Point 1 should have neighbors"


def test_sparse_matrix_threshold():
    """稀疏矩阵阈值过滤测试"""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0]
    ])

    # 阈值 = 1.5，只保留距离 ≤ 1.5 的边
    neighbors = python_sparse_distance_matrix(points, metric='euclidean', threshold=1.5)

    # 点0应该只能看到点1
    point0_neighbors = [j for j, d in neighbors[0]]
    assert 1 in point0_neighbors, "Point 0 should see point 1"
    assert 2 not in point0_neighbors, "Point 0 should not see point 2"
    assert 3 not in point0_neighbors, "Point 0 should not see point 3"


def test_sparse_matrix_full():
    """稀疏矩阵（无阈值）应该与稠密矩阵一致"""
    np.random.seed(456)
    points = np.random.randn(10, 2)

    # 稠密矩阵
    dist_full = python_compressed_distance_matrix(points, metric='euclidean')

    # 稀疏矩阵（无阈值）
    neighbors_sparse = python_sparse_distance_matrix(points, metric='euclidean', threshold=None)

    # 验证每个点的邻居
    for i in range(10):
        for j, d in neighbors_sparse[i]:
            expected_d = dist_full[i, j]
            assert abs(d - expected_d) < 1e-6, \
                f"Sparse matrix d({i},{j})={d:.6f} != full matrix {expected_d:.6f}"


def test_edge_cases():
    """边界情况测试"""

    # 最小点云（2个点）
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    dist = python_compressed_distance_matrix(points, metric='euclidean')
    assert dist.shape == (2, 2), "Shape should be (2, 2)"
    assert abs(dist[0, 1] - 1.0) < 1e-6, "Distance should be 1.0"

    # 高维点云
    points = np.random.randn(5, 100)
    dist = python_compressed_distance_matrix(points, metric='euclidean')
    assert dist.shape == (5, 5), "Shape should be (5, 5)"

    # 所有点相同
    points = np.ones((5, 3))
    dist = python_compressed_distance_matrix(points, metric='euclidean')
    assert np.allclose(dist, 0.0), "All distances should be 0"


def test_numerical_stability():
    """数值稳定性测试"""

    # 非常小的距离
    points = np.array([
        [0.0, 0.0],
        [1e-7, 1e-7]
    ])
    dist = python_compressed_distance_matrix(points, metric='euclidean')
    expected = np.sqrt(2 * (1e-7) ** 2)
    assert abs(dist[0, 1] - expected) < 1e-14, "Small distance precision failed"

    # 非常大的距离
    points = np.array([
        [0.0, 0.0],
        [1e6, 1e6]
    ])
    dist = python_compressed_distance_matrix(points, metric='euclidean')
    expected = np.sqrt(2 * (1e6) ** 2)
    assert abs(dist[0, 1] - expected) / expected < 1e-6, "Large distance precision failed"


def test_memory_layout():
    """
    验证内存布局（下三角存储）

    虽然这个测试使用完整矩阵，但验证了 scipy.pdist 的压缩格式
    """
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # scipy.pdist 返回压缩格式（下三角，不包括对角线）
    condensed = pdist(points, metric='euclidean')

    # 压缩向量应该包含 n*(n-1)/2 个元素
    n = 3
    expected_len = n * (n - 1) // 2
    assert len(condensed) == expected_len, f"Expected {expected_len} elements, got {len(condensed)}"

    # 转换为完整矩阵
    full_matrix = squareform(condensed)

    # 验证对称性和对角线
    assert np.allclose(full_matrix, full_matrix.T), "Matrix should be symmetric"
    assert np.allclose(np.diag(full_matrix), 0.0), "Diagonal should be zero"


def test_consistency():
    """
    一致性测试：同一点云多次计算应得到相同结果
    """
    points = np.random.randn(15, 4)

    dist1 = python_compressed_distance_matrix(points, metric='euclidean')
    dist2 = python_compressed_distance_matrix(points, metric='euclidean')

    assert np.array_equal(dist1, dist2), "Results should be consistent"


def test_scipy_reference_values():
    """
    使用 scipy 的已知正确值进行验证
    """
    # 三角形点云
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2]  # 等边三角形
    ])

    dist = python_compressed_distance_matrix(points, metric='euclidean')

    # 等边三角形，所有边长应该都是 1.0
    assert abs(dist[0, 1] - 1.0) < 1e-6, "Edge 0-1 should be 1.0"
    assert abs(dist[0, 2] - 1.0) < 1e-6, "Edge 0-2 should be 1.0"
    assert abs(dist[1, 2] - 1.0) < 1e-6, "Edge 1-2 should be 1.0"


def test_sparse_matrix_symmetry():
    """
    验证稀疏矩阵的对称性
    """
    points = np.random.randn(10, 3)
    neighbors = python_sparse_distance_matrix(points, metric='euclidean', threshold=2.0)

    # 如果 i 的邻居中有 j，那么 j 的邻居中应该有 i
    for i in range(10):
        for j, d_ij in neighbors[i]:
            # 在 j 的邻居中查找 i
            found = False
            for k, d_jk in neighbors[j]:
                if k == i:
                    found = True
                    # 验证距离一致
                    assert abs(d_ij - d_jk) < 1e-6, \
                        f"Distance mismatch: d({i},{j})={d_ij}, d({j},{i})={d_jk}"
                    break
            assert found, f"Symmetry broken: {i} has neighbor {j}, but {j} doesn't have {i}"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
