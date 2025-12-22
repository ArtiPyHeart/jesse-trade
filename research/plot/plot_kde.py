import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm


def process_sign_sequence(values: np.ndarray, compare_shift: int = -1) -> np.ndarray:
    """
    根据相邻值符号关系对序列做正负翻转。

    compare_shift = -1: 与下一个值比较（shift(-1)）
    compare_shift = 1: 与上一个值比较（shift(1)）
    """
    values = np.asarray(values)
    assert values.ndim == 1, "values必须为一维数组"
    assert compare_shift in (-1, 1), "compare_shift必须为-1或1"

    processed = values.copy()
    if values.size < 2:
        return processed

    if compare_shift == -1:
        prod = values[:-1] * values[1:]
        same_sign = prod > 0
        diff_sign = prod < 0
        idx_same = np.where(same_sign)[0]
        idx_diff = np.where(diff_sign)[0]
    else:
        prod = values[1:] * values[:-1]
        same_sign = prod > 0
        diff_sign = prod < 0
        idx_same = np.where(same_sign)[0] + 1
        idx_diff = np.where(diff_sign)[0] + 1

    processed[idx_same] = np.abs(values[idx_same])
    processed[idx_diff] = -np.abs(values[idx_diff])
    return processed


def _standardize_returns(
    returns: np.ndarray, apply_sign_sequence: bool, sign_shift: int
) -> np.ndarray:
    if apply_sign_sequence:
        returns = process_sign_sequence(returns, compare_shift=sign_shift)
    std = returns.std()
    assert std > 0, "returns标准差必须大于0"
    return (returns - returns.mean()) / std


def plot_kde(
    array_1d: np.ndarray,
    lag=1,
    multi_lag: bool = True,
    apply_sign_sequence: bool = True,
    sign_shift: int = -1,
):
    """
    绘制收益率的核密度估计图，可选多重lag对比

    Parameters:
    -----------
    array_1d : np.ndarray
        一维价格数组
    lag : int, default=1
        单一lag模式：使用的lag值
        多重lag模式：最大lag值（将绘制lag 1到lag的所有峰度）
    multi_lag : bool, default=True
        是否绘制多重lag的峰度对比（从lag=1到lag=lag）
    apply_sign_sequence : bool, default=True
        是否应用相邻符号序列处理逻辑
    sign_shift : int, default=-1
        apply_sign_sequence=True时有效，-1对比下一个值，1对比上一个值
    """
    array_1d = np.asarray(array_1d)
    assert array_1d.ndim == 1, "array_1d必须为一维数组"
    assert lag >= 1, "lag必须 >= 1"
    assert array_1d.size > lag + 1, "array_1d长度必须大于lag+1"
    assert np.all(np.isfinite(array_1d)), "array_1d必须为有限数值"
    assert np.all(array_1d > 0), "array_1d必须为正数以计算对数收益率"
    if apply_sign_sequence:
        assert sign_shift in (-1, 1), "sign_shift必须为-1或1"

    plt.figure(figsize=(10, 6))

    if not multi_lag:
        # 原有的单一lag行为
        ret = np.log(array_1d[lag:]) - np.log(array_1d[:-lag])
        standard = _standardize_returns(ret, apply_sign_sequence, sign_shift)
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")

        sns.kdeplot(standard, label=f"lag={lag}", color="blue")
        sns.kdeplot(
            np.random.normal(size=1000000),
            label="Normal",
            color="black",
            linestyle="--",
        )

        title = f"bar_{array_1d.shape[0]}_kurtosis_{kurtosis:.4f}"
    else:
        # 多重lag模式：从lag=1到lag=lag
        assert lag >= 1, "lag必须 >= 1"

        # 生成颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, lag))
        kurtosis_list = []

        # 循环绘制不同lag的KDE
        for lag_i in range(1, lag + 1):
            ret = np.log(array_1d[lag_i:]) - np.log(array_1d[:-lag_i])
            standard = _standardize_returns(ret, apply_sign_sequence, sign_shift)
            kurtosis = stats.kurtosis(
                standard, axis=None, fisher=False, nan_policy="omit"
            )
            kurtosis_list.append(kurtosis)

            sns.kdeplot(
                standard,
                label=f"lag={lag_i} (K={kurtosis:.2f})",
                color=colors[lag_i - 1],
                linewidth=2,
            )

        # 绘制正态分布参考线
        sns.kdeplot(
            np.random.normal(size=1000000),
            label="Normal",
            color="black",
            linestyle="--",
            linewidth=2,
        )

        title = f"bar_{array_1d.shape[0]}_multi_lag_kurtosis_comparison"

    plt.xticks(range(-5, 6))
    plt.legend(loc="best", ncol=2 if multi_lag and lag > 5 else 1)
    plt.title(
        title,
        loc="center",
        fontsize=20,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.xlabel("Standardized Returns")
    plt.ylabel("Density")
    plt.xlim(-5, 5)
    plt.grid(True, alpha=0.3)
    plt.show()


def find_kde_cross(array_1d: np.ndarray) -> np.ndarray:
    # —— 2. 拟合 KDE & 拟合正态 φ(x|μ,σ) ——
    kde = gaussian_kde(array_1d)
    mu, sigma = array_1d.mean(), array_1d.std(ddof=0)

    def f(t):
        return kde(t) - norm.pdf(t, loc=mu, scale=sigma)

    # —— 3. 在细网格上预计算 f(x) ——
    xs = np.linspace(array_1d.min(), array_1d.max(), 10000)
    fs = f(xs)

    # —— 4. 扫描所有符号翻转区间，精确求根 ——
    roots = []
    for i in range(len(xs) - 1):
        # 精确落在网格点上的交点也一并收集
        if fs[i] == 0:
            roots.append(xs[i])
        # 在 (xs[i], xs[i+1]) 有符号翻转，说明这里有根
        elif fs[i] * fs[i + 1] < 0:
            c = brentq(f, xs[i], xs[i + 1])
            roots.append(c)

    # 去重、排序
    roots = sorted(set(roots))
    print("所有交点：", roots)

    # —— （可选）可视化检查 ——
    plt.figure(figsize=(8, 4))
    plt.plot(xs, kde(xs), label="Empirical KDE")
    plt.plot(xs, norm.pdf(xs, mu, sigma), "--", label="Fitted Normal")
    for c in roots:
        plt.axvline(c, color="red", ls=":")
    plt.hist(array_1d, bins=50, density=True, alpha=0.3, color="gray")
    plt.legend()
    plt.title("All KDE vs Normal intersections")
    plt.show()

    return roots
