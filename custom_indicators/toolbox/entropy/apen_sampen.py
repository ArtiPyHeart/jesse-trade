import numpy as np


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
    return np.nan if Am == 0 or Bm == 0 else -np.log(Am / Bm)
