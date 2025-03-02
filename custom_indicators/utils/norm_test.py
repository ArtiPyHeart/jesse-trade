import numpy as np
import plotly.graph_objects as go
from scipy.stats import kstest, norm, shapiro


def is_norm_dist(data: np.ndarray):
    # 计算样本均值和标准差
    mean = np.mean(data)
    std_dev = np.std(data)

    # Shapiro-Wilk 检验
    shapiro_stat, shapiro_p = shapiro(data)

    # Kolmogorov-Smirnov 检验
    ks_stat, ks_p = kstest(data, "norm", args=(mean, std_dev))

    print(f"Shapiro-Wilk 检验: 统计量={shapiro_stat}, p值={shapiro_p}")
    print(f"Kolmogorov-Smirnov 检验: 统计量={ks_stat}, p值={ks_p}")


def norm_plot(data: np.ndarray | dict[str, np.ndarray]):
    """
    绘制数据分布与正态分布的对比图

    参数:
        data:
            1) 收益率一维数组（从-1到1的浮点数）; 或
            2) 一个字典，key为坐标轴图示名称，value为对应的np.ndarray。
    """
    # 如果传入的是单个数组，将其包装为字典，给出一个默认名称
    if isinstance(data, np.ndarray):
        data_dict = {"Data Distribution": data}
    else:
        data_dict = data

    fig = go.Figure()

    # 为字典中的每个numpy数组分别绘制实际数据分布曲线+对应正态分布曲线
    for label, arr in data_dict.items():
        # 计算直方图用于估计数据密度
        counts, bins = np.histogram(arr, bins=50, density=True)
        x_data = 0.5 * (bins[:-1] + bins[1:])

        # 计算数据分布的均值与标准差
        mean = np.mean(arr)
        std_dev = np.std(arr)

        # 计算对应正态分布的概率密度
        x_range = np.linspace(min(arr), max(arr), 500)
        pdf = norm.pdf(x_range, loc=mean, scale=std_dev)

        # 实际数据的密度曲线（实线）
        fig.add_trace(
            go.Scatter(x=x_data, y=counts, mode="lines", name=f"{label} (Data)")
        )

        # 正态分布的密度曲线（虚线）
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=pdf,
                mode="lines",
                line=dict(dash="dash"),
                name=f"{label} (Normal)",
            )
        )

    fig.update_layout(
        title="Density Plot of Data vs Normal Distribution",
        xaxis_title="Value",
        yaxis_title="Density",
    )

    fig.show()
