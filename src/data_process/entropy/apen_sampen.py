import warnings

import numpy as np
from numba import jit


def _phi(x, m, r):
    N = len(x)
    X = np.array([x[i : i + m] for i in range(N - m + 1)])
    dist = np.abs(X[:, None, :] - X[None, :, :]).max(axis=2)
    C = (dist <= r).sum(axis=0) / (N - m + 1)  # 含自匹配
    return np.log(C).mean()


def _data_range(x, mode: str = "range"):
    if mode == "range":
        return np.max(x) - np.min(x)
    elif mode == "std":
        return np.std(x, ddof=0)
    else:
        raise ValueError("mode must be either 'range' or 'std'")


def approximate_entropy(x, m=2, r_ratio: float = 0.3, mode: str = "range"):
    x = np.asarray(x, float)
    r = r_ratio * _data_range(x, mode)
    return _phi(x, m, r) - _phi(x, m + 1, r)


def sample_entropy(x, m=2, r_ratio: float = 0.3, mode: str = "range"):
    x = np.asarray(x, float)
    r = r_ratio * _data_range(x, mode)
    N = len(x)
    Xm = np.array([x[i : i + m] for i in range(N - m)])
    Xm1 = np.array([x[i : i + m + 1] for i in range(N - m - 1)])

    def _count_pairs(Y):
        M = len(Y)
        # 如果Y是1维数组，将其重塑为2维数组
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        dist = np.abs(Y[:, None, :] - Y[None, :, :]).max(axis=2)
        return np.sum((dist <= r) & np.triu(np.ones((M, M), bool), 1))

    Bm = _count_pairs(Xm)
    Am = _count_pairs(Xm1)
    if Am == 0 or Bm == 0:
        warnings.warn("sample_entropy: Am or Bm is 0, return NaN")
        return np.nan
    else:
        return -np.log(Am / Bm)


def sample_entropy_fast(x, m=2, r_ratio: float = 0.3, mode: str = "range"):
    """
    优化版本的sample entropy计算函数
    通过避免创建大型中间矩阵来提高性能
    """
    x = np.asarray(x, float)
    r = r_ratio * _data_range(x, mode)
    N = len(x)

    def _count_pairs_fast(Y):
        M = len(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        count = 0
        # 使用向量化操作但避免创建大型中间矩阵
        for i in range(M):
            # 只计算上三角部分
            diff = np.abs(Y[i + 1 :] - Y[i])
            count += np.sum(np.all(diff <= r, axis=1))
        return count

    Xm = np.array([x[i : i + m] for i in range(N - m)])
    Xm1 = np.array([x[i : i + m + 1] for i in range(N - m - 1)])

    Bm = _count_pairs_fast(Xm)
    Am = _count_pairs_fast(Xm1)

    if Am == 0 or Bm == 0:
        warnings.warn("sample_entropy_fast: Am or Bm is 0, return NaN")
        return np.nan
    else:
        return -np.log(Am / Bm)


@jit(nopython=True)
def _count_pairs_numba(Y, r):
    """
    使用Numba优化的计数函数
    """
    M = len(Y)
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            max_diff = 0
            for k in range(Y.shape[1]):
                diff = abs(Y[i, k] - Y[j, k])
                if diff > max_diff:
                    max_diff = diff
            if max_diff <= r:
                count += 1
    return count


def sample_entropy_numba(x, m=2, r_ratio: float = 0.3, mode: str = "range"):
    """
    使用Numba优化的sample entropy计算函数
    """
    x = np.asarray(x, float)
    r = r_ratio * _data_range(x, mode)
    N = len(x)

    Xm = np.array([x[i : i + m] for i in range(N - m)])
    Xm1 = np.array([x[i : i + m + 1] for i in range(N - m - 1)])

    Bm = _count_pairs_numba(Xm, r)
    Am = _count_pairs_numba(Xm1, r)

    if Am == 0 or Bm == 0:
        warnings.warn("sample_entropy_numba: Am or Bm is 0, return NaN")
        return np.nan
    else:
        return -np.log(Am / Bm)


@jit(nopython=True)
def _phi_numba(x, m, r):
    """
    使用Numba优化的phi函数计算
    """
    N = len(x)
    total = 0

    for i in range(N - m + 1):
        template = x[i : i + m]
        matches = 0
        for j in range(N - m + 1):
            max_diff = 0
            for k in range(m):
                diff = abs(template[k] - x[j + k])
                if diff > max_diff:
                    max_diff = diff
            if max_diff <= r:
                matches += 1
        if matches > 0:
            total += np.log(matches / (N - m + 1))

    return total / (N - m + 1)


def approximate_entropy_numba(x, m=2, r_ratio: float = 0.3, mode: str = "range"):
    """
    使用Numba优化的approximate entropy计算函数
    """
    x = np.asarray(x, float)
    r = r_ratio * _data_range(x, mode)
    return _phi_numba(x, m, r) - _phi_numba(x, m + 1, r)


if __name__ == "__main__":
    import time

    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 1000)

    # 预热Numba
    _ = sample_entropy_numba(test_data)
    _ = approximate_entropy_numba(test_data)

    # 测试正确性
    print("=== Sample Entropy 测试 ===")
    result_original = sample_entropy(test_data)
    result_fast = sample_entropy_fast(test_data)
    result_numba = sample_entropy_numba(test_data)

    print(f"原始函数结果: {result_original}")
    print(f"优化函数结果: {result_fast}")
    print(f"Numba优化结果: {result_numba}")
    print(f"原始vs优化差异: {abs(result_original - result_fast)}")
    print(f"原始vsNumba差异: {abs(result_original - result_numba)}")

    print("\n=== Approximate Entropy 测试 ===")
    result_orig_apen = approximate_entropy(test_data)
    result_numba_apen = approximate_entropy_numba(test_data)

    print(f"原始函数结果: {result_orig_apen}")
    print(f"Numba优化结果: {result_numba_apen}")
    print(f"结果差异: {abs(result_orig_apen - result_numba_apen)}")

    # 性能测试
    def benchmark(func, data, iterations=100):
        start_time = time.time()
        for _ in range(iterations):
            func(data)
        end_time = time.time()
        return (end_time - start_time) / iterations

    # 运行性能测试
    print("\n=== Sample Entropy 性能测试 ===")
    original_time = benchmark(sample_entropy, test_data)
    fast_time = benchmark(sample_entropy_fast, test_data)
    numba_time = benchmark(sample_entropy_numba, test_data)

    print(f"原始函数平均执行时间: {original_time * 1000:.2f} ms")
    print(f"优化函数平均执行时间: {fast_time * 1000:.2f} ms")
    print(f"Numba优化平均执行时间: {numba_time * 1000:.2f} ms")
    print(f"优化函数性能提升: {original_time / fast_time:.2f}x")
    print(f"Numba优化性能提升: {original_time / numba_time:.2f}x")

    print("\n=== Approximate Entropy 性能测试 ===")
    orig_apen_time = benchmark(approximate_entropy, test_data)
    numba_apen_time = benchmark(approximate_entropy_numba, test_data)

    print(f"原始函数平均执行时间: {orig_apen_time * 1000:.2f} ms")
    print(f"Numba优化平均执行时间: {numba_apen_time * 1000:.2f} ms")
    print(f"性能提升: {orig_apen_time / numba_apen_time:.2f}x")
