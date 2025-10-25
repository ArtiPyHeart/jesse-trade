"""
二项式系数表测试

对比 Rust 实现与 Python scipy 的计算结果，验证数值精度。
"""

import numpy as np
import pytest
from scipy.special import comb


# 由于 ripser 模块还在开发中，暂时只测试数学正确性
# 后续迭代会添加 Rust FFI 绑定


def python_binomial_table(n, k):
    """
    Python 参考实现（使用 scipy）

    Args:
        n: 最大 n 值
        k: 最大 k 值

    Returns:
        二维数组，table[j][i] = C(i, j)
    """
    table = np.zeros((k + 1, n + 1), dtype=np.int64)

    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            table[j, i] = int(comb(i, j, exact=True))

    return table


def test_binomial_basic_values():
    """测试基本的二项式系数值"""
    table = python_binomial_table(10, 5)

    # C(0, 0) = 1
    assert table[0, 0] == 1

    # C(5, 0) = 1
    assert table[0, 5] == 1

    # C(5, 5) = 1
    assert table[5, 5] == 1

    # C(5, 1) = 5
    assert table[1, 5] == 5

    # C(5, 2) = 10
    assert table[2, 5] == 10

    # C(5, 3) = 10
    assert table[3, 5] == 10

    # C(5, 4) = 5
    assert table[4, 5] == 5


def test_binomial_pascal_triangle():
    """验证 Pascal's triangle 的正确性"""
    table = python_binomial_table(10, 5)

    # 第 0 行: 1
    assert table[0, 0] == 1

    # 第 1 行: 1 1
    assert table[0, 1] == 1
    assert table[1, 1] == 1

    # 第 2 行: 1 2 1
    assert table[0, 2] == 1
    assert table[1, 2] == 2
    assert table[2, 2] == 1

    # 第 3 行: 1 3 3 1
    assert table[0, 3] == 1
    assert table[1, 3] == 3
    assert table[2, 3] == 3
    assert table[3, 3] == 1

    # 第 4 行: 1 4 6 4 1
    assert table[0, 4] == 1
    assert table[1, 4] == 4
    assert table[2, 4] == 6
    assert table[3, 4] == 4
    assert table[4, 4] == 1


def test_binomial_well_known_values():
    """测试一些众所周知的二项式系数值"""
    table = python_binomial_table(20, 10)

    # C(10, 5) = 252
    assert table[5, 10] == 252

    # C(10, 3) = 120
    assert table[3, 10] == 120

    # C(20, 10) = 184756
    assert table[10, 20] == 184756

    # C(15, 7) = 6435
    assert table[7, 15] == 6435


def test_binomial_symmetry():
    """验证对称性: C(n, k) = C(n, n-k)"""
    table = python_binomial_table(10, 10)

    for n in range(11):
        for k in range(n + 1):
            assert table[k, n] == table[n - k, n], \
                f"Symmetry failed: C({n}, {k}) != C({n}, {n - k})"


def test_binomial_recursive_property():
    """验证递推性质: C(n, k) = C(n-1, k-1) + C(n-1, k)"""
    table = python_binomial_table(10, 5)

    for n in range(2, 11):
        for k in range(1, min(n, 5)):
            expected = table[k - 1, n - 1] + table[k, n - 1]
            assert table[k, n] == expected, \
                f"Recursive property failed at C({n}, {k})"


def test_binomial_large_values():
    """测试较大的值"""
    table = python_binomial_table(50, 25)

    # C(50, 25) 是一个非常大的数
    c_50_25 = table[25, 50]
    assert c_50_25 > 0
    assert c_50_25 == int(comb(50, 25, exact=True))

    # C(40, 20) = 137846528820
    c_40_20 = table[20, 40]
    assert c_40_20 == 137846528820


def test_binomial_against_scipy():
    """
    全面对比 Python 实现与 scipy

    验证所有值都与 scipy.special.comb 完全一致
    """
    n, k = 30, 15
    table = python_binomial_table(n, k)

    errors = []
    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            expected = int(comb(i, j, exact=True))
            actual = table[j, i]

            if actual != expected:
                errors.append(f"C({i}, {j}): expected {expected}, got {actual}")

    assert len(errors) == 0, f"Found {len(errors)} errors:\n" + "\n".join(errors[:10])


def test_binomial_consistency():
    """
    一致性测试：同一个表多次构建应该得到相同结果
    """
    table1 = python_binomial_table(20, 10)
    table2 = python_binomial_table(20, 10)

    assert np.array_equal(table1, table2), "Tables are not consistent"


def test_binomial_edge_cases():
    """边界情况测试"""
    table = python_binomial_table(5, 5)

    # C(0, 0) = 1
    assert table[0, 0] == 1

    # C(n, 0) = 1 对所有 n
    for n in range(6):
        assert table[0, n] == 1, f"C({n}, 0) should be 1"

    # C(n, n) = 1 对所有 n
    for n in range(6):
        assert table[n, n] == 1, f"C({n}, {n}) should be 1"

    # C(n, 1) = n 对所有 n
    for n in range(1, 6):
        assert table[1, n] == n, f"C({n}, 1) should be {n}"


def test_binomial_overflow_detection():
    """
    溢出检测测试

    虽然 scipy 可以处理任意大的整数，但 Rust 实现
    限制在 i64 范围内（为系数域预留 8 位）
    """
    # 这个测试只是验证我们的限制是合理的
    # C(100, 50) 约为 10^29，超出 i64 范围
    max_safe_n = 57  # 安全的最大 n（C(57, 28) < 2^55）

    table = python_binomial_table(max_safe_n, max_safe_n // 2)

    # 检查所有值都在 i64 范围内（减去系数位）
    MAX_SIMPLEX_INDEX = (1 << (64 - 1 - 8)) - 1  # 2^55 - 1

    for i in range(max_safe_n + 1):
        for j in range(min(i, max_safe_n // 2) + 1):
            value = table[j, i]
            assert value <= MAX_SIMPLEX_INDEX, \
                f"C({i}, {j}) = {value} exceeds max_index = {MAX_SIMPLEX_INDEX}"


def test_binomial_memory_layout():
    """
    验证内存布局

    Rust 使用转置存储：table[k][n] = C(n, k)
    Python 参考实现也应该使用相同的布局
    """
    n, k = 10, 5
    table = python_binomial_table(n, k)

    # 验证形状
    assert table.shape == (k + 1, n + 1), \
        f"Expected shape ({k + 1}, {n + 1}), got {table.shape}"

    # 验证访问模式：table[k, n] = C(n, k)
    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            assert table[j, i] == int(comb(i, j, exact=True)), \
                f"Memory layout error at C({i}, {j})"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
