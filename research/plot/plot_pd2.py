import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


# 现代化调色板（高对比度，深浅背景均适用）
COLORS = [
    "#E63946",  # 珊瑚红
    "#2A9D8F",  # 青绿
    "#457B9D",  # 钢蓝
    "#F4A261",  # 琥珀橙
    "#9B5DE5",  # 紫罗兰
    "#00F5D4",  # 青色
    "#F72585",  # 玫红
]


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


def plot_pd2(
    array_1d: np.ndarray,
    max_lag: int = 5,
    sign_shift: int = -1,
    apply_sign_sequence: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    dark_mode: bool = True,
) -> None:
    """
    绘制多lag标准化收益率的核密度估计图。

    封装自 PD_2.ipynb 的绘图逻辑，支持两种符号序列处理模式。

    Parameters
    ----------
    array_1d : np.ndarray
        一维价格数组（必须为正数以计算对数收益率）
    max_lag : int, default=5
        最大lag值，将绘制 lag=1 到 lag=max_lag 的所有KDE曲线
    sign_shift : int, default=-1
        符号序列处理方向：
        - -1: 与下一个值比较（shift(-1)），标题后缀 "-1"
        - 1: 与上一个值比较（shift(1)），标题后缀 "+1"
    apply_sign_sequence : bool, default=True
        是否应用相邻符号序列处理逻辑
    title : str | None, default=None
        自定义标题，为 None 时自动生成
    save_path : str | None, default=None
        保存图片路径，为 None 时调用 plt.show()
    dark_mode : bool, default=True
        是否使用深色模式（适配 PyCharm 深色主题）
    """
    array_1d = np.asarray(array_1d)
    assert array_1d.ndim == 1, "array_1d必须为一维数组"
    assert max_lag >= 1, "max_lag必须 >= 1"
    assert array_1d.size > max_lag + 1, "array_1d长度必须大于max_lag+1"
    assert np.all(np.isfinite(array_1d)), "array_1d必须为有限数值"
    assert np.all(array_1d > 0), "array_1d必须为正数以计算对数收益率"
    if apply_sign_sequence:
        assert sign_shift in (-1, 1), "sign_shift必须为-1或1"

    # 设置样式
    if dark_mode:
        plt.style.use("dark_background")
        bg_color = "#1E1E1E"
        text_color = "#E0E0E0"
        grid_color = "#404040"
        normal_color = "#AAAAAA"
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        bg_color = "#FAFAFA"
        text_color = "#2D2D2D"
        grid_color = "#CCCCCC"
        normal_color = "#555555"

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # 绘制各 lag 的 KDE 曲线
    for lag_i in range(1, max_lag + 1):
        ret = np.log(array_1d[lag_i:]) - np.log(array_1d[:-lag_i])
        standard = _standardize_returns(ret, apply_sign_sequence, sign_shift)
        kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")
        color = COLORS[(lag_i - 1) % len(COLORS)]
        sns.kdeplot(
            standard,
            label=f"Lag {lag_i}  (K={kurtosis:.2f})",
            color=color,
            linewidth=2.5,
            ax=ax,
        )

    # 绘制正态分布参考线
    sns.kdeplot(
        np.random.normal(size=1000000),
        label="Normal (K=3.00)",
        color=normal_color,
        linestyle="--",
        linewidth=2,
        ax=ax,
    )

    # 坐标轴设置
    ax.set_xlim(-5, 5)
    ax.set_xticks(range(-5, 6))
    ax.set_xlabel(
        "Standardized Returns", fontsize=13, color=text_color, fontweight="medium"
    )
    ax.set_ylabel("Density", fontsize=13, color=text_color, fontweight="medium")
    ax.tick_params(colors=text_color, labelsize=11)

    # 网格线
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, color=grid_color)
    ax.set_axisbelow(True)

    # 图例
    legend = ax.legend(
        loc="upper right",
        fontsize=11,
        framealpha=0.85,
        edgecolor=grid_color,
        fancybox=True,
        borderpad=0.8,
        labelspacing=0.6,
    )
    legend.get_frame().set_facecolor(bg_color)
    for text in legend.get_texts():
        text.set_color(text_color)

    # 标题
    if title is None:
        suffix = "shift(-1)" if sign_shift == -1 else "shift(+1)"
        title = f"KDE of Standardized Returns  |  n={array_1d.shape[0]:,}  |  {suffix}"
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        color=text_color,
        pad=15,
    )

    # 边框美化
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, facecolor=bg_color, edgecolor="none")
        plt.close()
    else:
        plt.show()

    # 恢复默认样式
    plt.style.use("default")
