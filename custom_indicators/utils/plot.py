import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


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
