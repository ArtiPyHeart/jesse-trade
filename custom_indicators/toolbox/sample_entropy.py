import numpy as np
from numba import njit


@njit
def _maxdist(x_i: np.ndarray, x_j: np.ndarray):
    return max([abs(ua - va) for ua, va in zip(x_i, x_j)])


@njit
def _phi(data: np.ndarray, emb_dim: int, tol: float) -> float:
    n = len(data) - emb_dim + 1
    if n <= 1:
        return np.nan

    new_data = [data[i : i + emb_dim] for i in range(n)]
    C = [
        sum(
            [
                1
                for j in range(n)
                if i != j and _maxdist(new_data[i], new_data[j]) <= tol
            ]
        )
        for i in range(n)
    ]

    denominator = n * (n - 1)
    if denominator == 0:
        return np.nan

    result = sum(C) / denominator
    if result == 0:
        return np.nan
    return result


@njit
def sample_entropy(data: np.ndarray, emb_dim: int, tol: float) -> float:
    if len(data) < emb_dim + 2:
        return np.nan

    phi_emb_dim = _phi(data, emb_dim, tol)
    phi_emb_dim_plus_1 = _phi(data, emb_dim + 1, tol)

    if phi_emb_dim == 0 or phi_emb_dim_plus_1 == 0:
        return np.nan

    return -np.log(phi_emb_dim_plus_1 / phi_emb_dim)


def dynamic_sample_entropy(
    data: np.ndarray, emb_dim: int = 2, r_ratio: float = 0.3, mode: str = "range"
):
    if mode not in ["range", "std"]:
        raise ValueError("mode must be either 'range' or 'std'")

    if mode == "range":
        data_range = np.max(data) - np.min(data)
        r = r_ratio * data_range
    elif mode == "std":
        data_std = np.std(data)
        r = r_ratio * data_std

    return sample_entropy(data, emb_dim, r)
