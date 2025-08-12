import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm


def plot_kde(array_1d: np.ndarray, lag=1):
    ret = np.log(array_1d[lag:]) - np.log(array_1d[:-lag])
    standard = (ret - ret.mean()) / ret.std()
    kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(standard, label="bar", color="blue")
    sns.kdeplot(
        np.random.normal(size=1000000), label="Normal", color="black", linestyle="--"
    )
    plt.xticks(range(-5, 6))
    plt.legend(loc=8, ncol=5)
    plt.title(
        f"bar_{array_1d.shape[0]}_kurtosis_{kurtosis}",
        loc="center",
        fontsize=20,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.xlim(-5, 5)
    plt.grid(1)
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
