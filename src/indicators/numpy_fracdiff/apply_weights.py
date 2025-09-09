import numpy as np
import numba


@numba.jit(nopython=True)
def apply_weights(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = w.shape[0]
    z = w[0] * x
    z[: (m - 1)] = np.nan
    for k in range(1, m):
        z[k:] += w[k] * x[:-k]
    return z


@numba.jit(nopython=True, parallel=True)
def apply_weights_2d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """向量化的2D权重应用，支持并行处理多列"""
    m = w.shape[0]
    n_rows, n_cols = x.shape
    z = np.empty_like(x)
    
    # 并行处理每一列
    for j in numba.prange(n_cols):
        z[:, j] = w[0] * x[:, j]
        z[:(m-1), j] = np.nan
        for k in range(1, m):
            z[k:, j] += w[k] * x[:-k, j]
    
    return z
