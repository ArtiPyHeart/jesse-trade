"""
Simplex 编解码测试

对比 Rust 实现与 Python 参考实现的计算结果，验证算法正确性。

由于 ripser 模块还在开发中，暂时只测试数学正确性（使用纯 Python 参考实现）。
后续迭代会添加 Rust FFI 绑定。
"""

import numpy as np
import pytest
from scipy.special import comb


# ============================================================================
# Python 参考实现
# ============================================================================


def python_binomial(n, k):
    """
    二项式系数 C(n, k)

    数学上，当 k > n 时，C(n, k) = 0
    """
    if k > n:
        return 0
    return int(comb(n, k, exact=True))


def python_get_edge_index(i, j):
    """
    边的组合编码

    对于边 {i, j}（i > j），索引 = C(i, 2) + j
    """
    assert i > j, f"Invalid edge: i={i} must be > j={j}"
    return python_binomial(i, 2) + j


def python_get_edge_vertices(index):
    """
    边的解码

    返回 (i, j) 其中 i > j
    """
    # 使用精确公式求解 C(i, 2) <= index
    # C(i, 2) = i*(i-1)/2
    # 求解: i = floor((1 + sqrt(1 + 8*index)) / 2)
    i = int(np.floor((1 + np.sqrt(1 + 8 * index)) / 2))
    j = index - python_binomial(i, 2)
    return (i, j)


def python_encode_simplex(vertices):
    """
    通用 simplex 编码

    编码公式: index = C(v_k, k+1) + C(v_{k-1}, k) + ... + C(v_1, 2) + v_0
    """
    # 验证升序
    for i in range(1, len(vertices)):
        assert vertices[i] > vertices[i - 1], f"Vertices must be in ascending order: {vertices}"

    index = 0
    for k, v in enumerate(vertices[1:], start=1):
        index += python_binomial(v, k + 1)
    index += vertices[0]

    return index


def python_decode_simplex(index, dim, n):
    """
    通用 simplex 解码

    返回顶点列表（升序）
    """
    vertices = [0] * (dim + 1)
    n = n - 1

    # 从最高维度开始解码
    for k in range(dim, 0, -1):
        # 找最大的 v 使得 C(v, k+1) <= index
        v = python_get_max_vertex(index, k + 1, n)
        vertices[k] = v
        index -= python_binomial(v, k + 1)
        n = v

    vertices[0] = index
    return vertices


def python_get_max_vertex(index, k, n):
    """
    找最大的 v 使得 C(v, k) <= index

    使用二分查找（对于 k=2 可以用精确公式）
    """
    if k == 2:
        # 精确公式
        return int(np.floor((1 + np.sqrt(1 + 8 * index)) / 2))

    # 通用情况：二分查找
    low, high = 0, n

    while low < high:
        mid = (low + high + 1) // 2
        if python_binomial(mid, k) <= index:
            low = mid
        else:
            high = mid - 1

    return low


# ============================================================================
# 测试用例
# ============================================================================


def test_edge_encode_decode():
    """边的编码/解码一致性测试"""
    edges = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (10, 5), (20, 15)]

    for i, j in edges:
        index = python_get_edge_index(i, j)
        i2, j2 = python_get_edge_vertices(index)
        assert (i, j) == (i2, j2), f"Edge ({i}, {j}) encode-decode failed: decoded=({i2}, {j2})"


def test_edge_index_formula():
    """验证边索引公式 C(i, 2) + j"""
    for i in range(1, 20):
        for j in range(i):
            index = python_get_edge_index(i, j)
            expected = python_binomial(i, 2) + j
            assert index == expected, f"Edge ({i}, {j}) index formula failed"


def test_edge_sequential_indexing():
    """验证边的连续索引"""
    # 对于 n=5，边应该按顺序编号
    n = 5
    index = 0

    for i in range(1, n):
        for j in range(i):
            calculated_index = python_get_edge_index(i, j)
            assert calculated_index == index, f"Edge ({i}, {j}) should have index {index}, got {calculated_index}"

            i2, j2 = python_get_edge_vertices(index)
            assert (i, j) == (i2, j2), f"Decode edge {index} failed"

            index += 1


def test_simplex_encode_decode():
    """通用 simplex 的编码/解码一致性测试"""
    test_cases = [
        ([0, 1], 1, 10),          # 边（1-simplex, dim=1）
        ([0, 1, 2], 2, 10),       # 三角形（2-simplex, dim=2）
        ([0, 1, 2, 3], 3, 10),    # 四面体（3-simplex, dim=3）
        ([1, 3, 5, 7], 3, 10),    # 不同的四面体
        ([0, 1, 2, 3, 4], 4, 10), # 5-simplex（dim=4）
    ]

    for vertices, dim, n in test_cases:
        # dim应该等于 len(vertices) - 1
        assert dim == len(vertices) - 1, f"dim={dim} should equal len(vertices)-1={len(vertices)-1}"

        index = python_encode_simplex(vertices)
        decoded = python_decode_simplex(index, dim, n)
        assert vertices == decoded, f"Simplex {vertices} encode-decode failed: decoded={decoded}"


def test_simplex_consistency():
    """多次编码应得到相同结果"""
    simplex = [0, 1, 2, 3]
    index1 = python_encode_simplex(simplex)
    index2 = python_encode_simplex(simplex)
    assert index1 == index2, "Encoding should be deterministic"


def test_boundary_enumeration():
    """边界枚举测试（手动实现）"""
    # 三角形 {0, 1, 2} 的边界应该是 3 条边
    triangle = [0, 1, 2]

    # 手动计算边界
    boundaries = [
        [1, 2],  # 去掉顶点 0
        [0, 2],  # 去掉顶点 1
        [0, 1],  # 去掉顶点 2
    ]

    # 编码并验证
    boundary_indices = [python_encode_simplex(b) for b in boundaries]

    # 验证与边的编码一致
    edge_01 = python_get_edge_index(1, 0)
    edge_02 = python_get_edge_index(2, 0)
    edge_12 = python_get_edge_index(2, 1)

    assert edge_01 in boundary_indices
    assert edge_02 in boundary_indices
    assert edge_12 in boundary_indices


def test_edge_index_edge_cases():
    """边索引的边界情况"""
    # 最小的边 {1, 0}
    index = python_get_edge_index(1, 0)
    assert index == 0, "Edge {1, 0} should have index 0"

    i, j = python_get_edge_vertices(0)
    assert (i, j) == (1, 0), "Index 0 should decode to edge {1, 0}"


def test_simplex_index_properties():
    """Simplex 索引的数学性质"""
    # 对于固定的 k，索引应该是递增的
    indices_2d = []
    for v0 in range(5):
        for v1 in range(v0 + 1, 6):
            for v2 in range(v1 + 1, 7):
                simplex = [v0, v1, v2]
                index = python_encode_simplex(simplex)
                indices_2d.append((simplex, index))

    # 验证索引唯一性
    index_set = {idx for _, idx in indices_2d}
    assert len(index_set) == len(indices_2d), "All indices should be unique"


def test_binomial_coefficient_edge_case():
    """二项式系数的边界情况"""
    # C(n, k) = 0 when k > n
    assert python_binomial(1, 2) == 0, "C(1, 2) should be 0"
    assert python_binomial(0, 1) == 0, "C(0, 1) should be 0"

    # C(n, 0) = 1
    for n in range(10):
        assert python_binomial(n, 0) == 1, f"C({n}, 0) should be 1"

    # C(n, n) = 1
    for n in range(10):
        assert python_binomial(n, n) == 1, f"C({n}, {n}) should be 1"


def test_decode_all_edges():
    """解码所有 n 个点的边"""
    n = 6
    num_edges = n * (n - 1) // 2

    for index in range(num_edges):
        i, j = python_get_edge_vertices(index)

        # 验证 i > j
        assert i > j, f"Edge vertices should satisfy i > j, got ({i}, {j})"

        # 验证在范围内
        assert 0 <= j < i < n, f"Edge vertices out of range: ({i}, {j})"

        # 验证往返一致
        index2 = python_get_edge_index(i, j)
        assert index == index2, f"Round-trip failed for edge {index}"


def test_large_simplex():
    """大规模 simplex 测试"""
    # 测试较大的 simplex（但不超过溢出限制）
    vertices = list(range(10))
    dim = len(vertices) - 1

    index = python_encode_simplex(vertices)
    decoded = python_decode_simplex(index, dim, 15)

    assert vertices == decoded, "Large simplex encode-decode failed"


def test_encode_invalid_input():
    """测试非法输入"""
    # 未排序的顶点
    with pytest.raises(AssertionError, match="ascending order"):
        python_encode_simplex([2, 1, 0])

    # 边的非法索引
    with pytest.raises(AssertionError, match="must be >"):
        python_get_edge_index(1, 2)  # j >= i


def test_decode_consistency():
    """解码一致性：同一索引多次解码应得到相同结果"""
    index = 100
    dim = 2
    n = 20

    decoded1 = python_decode_simplex(index, dim, n)
    decoded2 = python_decode_simplex(index, dim, n)

    assert decoded1 == decoded2, "Decoding should be deterministic"


def test_edge_enumeration_order():
    """边枚举的顺序测试"""
    # 验证边的枚举顺序与索引一致
    n = 5
    edges = []

    for i in range(1, n):
        for j in range(i):
            edges.append((i, j))

    for idx, (i, j) in enumerate(edges):
        assert python_get_edge_index(i, j) == idx, f"Edge ({i}, {j}) should have index {idx}"


def test_simplex_reconstruction():
    """从索引重建 simplex"""
    # 创建几个 simplex
    simplices = [
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
    ]

    for simplex in simplices:
        dim = len(simplex) - 1
        index = python_encode_simplex(simplex)
        reconstructed = python_decode_simplex(index, dim, 10)
        assert simplex == reconstructed, f"Failed to reconstruct {simplex}"


def test_max_vertex_binary_search():
    """测试 get_max_vertex 的二分查找"""
    # 对于不同的 k 值测试
    for k in [2, 3, 4, 5]:
        for n in range(k, 20):
            # 计算 C(n, k)
            target = python_binomial(n, k)

            # get_max_vertex 应该返回 n
            v = python_get_max_vertex(target, k, n + 5)
            assert v == n, f"get_max_vertex({target}, {k}, {n+5}) should return {n}, got {v}"


def test_compare_encoding_methods():
    """对比不同维度的编码"""
    # 0-simplex（顶点）没有编码（直接使用索引）
    # 1-simplex（边）使用特殊公式
    # 2-simplex（三角形）使用通用公式

    # 边 {2, 1}
    edge = [1, 2]
    edge_index_direct = python_get_edge_index(2, 1)
    edge_index_general = python_encode_simplex(edge)
    assert edge_index_direct == edge_index_general, "Edge encoding methods should match"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
