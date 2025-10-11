import os
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# 配置matplotlib中文显示
import platform

# 根据系统平台选择合适的中文字体
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial Unicode MS", "STHeiti"]
elif platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
else:  # Linux
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei"]

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class ConfidenceSliceAnalyzer:
    """
    模型输出切片分析工具

    用于分析机器学习模型（分类/回归）的输出分布，通过精细切片找出真正赚钱的输出区间。
    支持任意范围的值，根据阈值判断多空方向，通过价格变化和交易信号计算每个切片的实际盈亏。
    """

    def __init__(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        close_price_data: Union[pd.Series, np.ndarray, list],
        granularity: float = 0.01,
        capital: float = 10000,
        coefficient: float = 0.25,
        output_dir: str = "./temp/",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        threshold: float = 0.5,
    ):
        """
        初始化分析器

        Parameters
        ----------
        time_data : array-like
            时间数据
        score_data : array-like
            模型输出值（分类任务：0-1之间；回归任务：任意范围）
        close_price_data : array-like
            收盘价数据
        granularity : float
            切片粒度（建议0.01-0.1之间）
        capital : float
            初始资金
        coefficient : float
            收益系数
        output_dir : str
            输出目录
        lower_bound : float
            分析值下限（默认0）
        upper_bound : float
            分析值上限（默认1）
        threshold : float
            多空分界点（默认0.5）
        """
        self.granularity = granularity
        self.capital = capital
        self.coefficient = coefficient
        self.output_dir = output_dir
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.threshold = threshold

        # 验证参数
        assert lower_bound < upper_bound, "下限必须小于上限"
        assert lower_bound <= threshold <= upper_bound, "阈值必须在上下限范围内"

        # 验证并准备数据
        self._validate_data(time_data, score_data, close_price_data)
        self._prepare_dataframe(time_data, score_data, close_price_data)

    def _validate_data(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        close_price_data: Union[pd.Series, np.ndarray, list],
    ):
        """验证输入数据的有效性"""
        # 检查长度
        assert (
            len(time_data) == len(score_data) == len(close_price_data)
        ), "三列数据长度必须相等"

        # 转换为numpy array进行验证
        score_array = np.asarray(score_data)

        # 检查超出范围的值
        out_of_range_mask = (score_array < self.lower_bound) | (
            score_array > self.upper_bound
        )
        if np.any(out_of_range_mask):
            out_count = np.sum(out_of_range_mask)
            out_ratio = out_count / len(score_array)
            min_val = np.min(score_array)
            max_val = np.max(score_array)
            print(
                f"\n⚠️ 警告：发现 {out_count} 个超出范围 [{self.lower_bound}, {self.upper_bound}] 的值"
            )
            print(f"  - 占比: {out_ratio:.2%}")
            print(f"  - 实际范围: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  - 超出范围的值将被归类到最近的边界切片\n")

        # 检查粒度值的合理性
        range_size = self.upper_bound - self.lower_bound
        min_slices = range_size / self.granularity
        assert (
            min_slices >= 2
        ), f"粒度过大，至少需要2个切片。当前设置将产生 {min_slices:.1f} 个切片"
        if min_slices > 200:
            print(f"⚠️ 警告：粒度过小，将产生 {int(min_slices)} 个切片，可能影响性能")

    def _prepare_dataframe(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        close_price_data: Union[pd.Series, np.ndarray, list],
    ):
        """将输入数据整合为内部DataFrame"""
        # 创建基础DataFrame
        self.data = pd.DataFrame(
            {
                "timestamp": time_data,
                "score": score_data,
                "close_price": close_price_data,
            }
        )

        # 计算价格差分（第一个值设为0）
        close_prices = np.asarray(close_price_data)
        close_diff = np.diff(close_prices, prepend=close_prices[0])
        close_diff[0] = 0  # 确保第一个值为0

        # 根据阈值生成交易信号：>= threshold为做多(1)，< threshold为做空(-1)
        scores = np.asarray(score_data)
        signal = np.where(scores >= self.threshold, 1, -1)

        # 计算每个时间点的盈亏（价格变化 * 交易信号）
        pnl = close_diff * signal

        # 添加计算列到DataFrame
        self.data["close_diff"] = close_diff
        self.data["signal"] = signal
        self.data["pnl"] = pnl

        self.data_size = len(self.data)

    def _get_slice_params(self):
        """根据上下限和粒度动态生成切片参数"""
        slices = []

        # 计算切片数量
        range_size = self.upper_bound - self.lower_bound
        num_slices = int(range_size / self.granularity)

        # 从上限开始，向下生成切片
        upper = self.upper_bound
        for i in range(num_slices):
            lower = upper - self.granularity
            # 处理浮点数精度问题
            lower = round(lower, 10)

            # 确保最后一个切片的下限正好是lower_bound
            if i == num_slices - 1:
                lower = self.lower_bound

            slices.append((upper, lower))
            upper = lower

        return slices

    def analyze(self):
        """执行完整的置信度切片分析"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 获取切片参数
        slices = self._get_slice_params()

        # 计算x轴刻度数（基于数据量，最多25个）
        tick_count = min(25, self.data_size)

        # 对每个切片进行分析
        for upper, lower in slices:
            # 处理超出范围的值 - 归类到最近的边界切片
            score_clipped = self.data["score"].clip(self.lower_bound, self.upper_bound)

            # 筛选当前切片区间（包括被归类的超出范围值）
            mask = (score_clipped < upper) & (score_clipped >= lower)

            # 仅对当前切片内的数据点计算盈亏
            slice_pnl = np.where(mask, self.data["pnl"], 0)

            # 计算样本占比
            true_count = mask.sum() / self.data_size

            # 计算累积盈亏（这将显示该切片的真实盈亏曲线，有涨有跌）
            cumulative_pnl = slice_pnl.cumsum()

            # 按系数和资金缩放
            self.data["cumulative_return"] = (
                self.coefficient * cumulative_pnl / self.capital
            )

            # 绘图
            fig, ax = plt.subplots(figsize=(20, 10))

            # 检测并转换时间数据类型
            time_series = self.data["timestamp"]

            # 尝试转换为datetime（如果还不是）
            try:
                # 如果已经是datetime类型，这不会改变它
                # 如果是字符串，会尝试解析
                time_series = pd.to_datetime(time_series)
                is_datetime = True
            except:
                # 无法转换为datetime，保持原样
                is_datetime = False

            # 绘制主曲线
            ax.plot(
                time_series,
                self.data["cumulative_return"],
                label="累积盈亏",
                color="blue",
            )

            # 计算并绘制均值线
            mean_value = self.data["cumulative_return"].mean()
            ax.axhline(
                y=mean_value,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"均值: {mean_value:.4f}",
            )

            # 添加零线作为参考（使用黑色加粗线条以在灰色网格中突出显示）
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, zorder=5)

            # 添加图例
            ax.legend(loc="best", fontsize=10)

            # 设置x轴刻度
            if is_datetime:
                # 对于datetime数据，使用matplotlib.dates处理
                # 设置合理的刻度数量
                locator = mdates.AutoDateLocator(maxticks=tick_count)
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
            else:
                # 对于非datetime数据（字符串等），手动设置刻度
                tick_indices = list(
                    range(0, self.data_size, max(1, int(self.data_size / tick_count)))
                )
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([str(time_series.iloc[i]) for i in tick_indices])

            # 添加标题和标签
            slice_center = (upper + lower) / 2
            if slice_center >= self.threshold:
                direction = " (做多)"
            else:
                direction = " (做空)"

            # 获取最终收益值
            final_return = self.data["cumulative_return"].iloc[-1]
            mean_return = self.data["cumulative_return"].mean()

            # 判断盈亏状态
            if final_return > 0:
                profit_status = "盈利"
            elif final_return < 0:
                profit_status = "亏损"
            else:
                profit_status = "持平"

            plt.title(
                f"切片区间 [{lower:.4f}, {upper:.4f}]{direction} - 样本占比: {true_count:.2%}\n"
                + f"最终收益: {final_return:.4f} ({profit_status}) | 平均收益: {mean_return:.4f}",
                fontsize=12,
                fontweight="bold",
            )
            plt.xlabel("时间", fontsize=12)
            plt.ylabel(
                f"累积盈亏 (系数={self.coefficient}, 本金={self.capital})", fontsize=12
            )

            fig.autofmt_xdate()
            plt.grid(1)

            # 保存图片 - 更详细的文件名
            filename = f"slice_{lower:.4f}_{upper:.4f}_ratio_{true_count:.4f}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            print(
                f"已生成: {filename} (切片区间: [{lower:.4f}, {upper:.4f}]{direction}, 样本占比: {true_count:.2%})"
            )


def analyze_confidence_slices(
    time_data: Union[pd.Series, np.ndarray, list],
    score_data: Union[pd.Series, np.ndarray, list],
    close_price_data: Union[pd.Series, np.ndarray, list],
    granularity: float = 0.01,
    capital: float = 1000,
    coefficient: float = 0.25,
    output_dir: str = "./temp/",
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    threshold: float = 0.5,
):
    """
    便捷函数：执行置信度切片分析

    Parameters
    ----------
    time_data : array-like
        时间数据
    score_data : array-like
        模型输出值（分类任务：0-1之间；回归任务：任意范围）
    close_price_data : array-like
        收盘价数据
    granularity : float
        切片粒度（建议0.01-0.1之间）
    capital : float
        初始资金
    coefficient : float
        收益系数
    output_dir : str
        输出目录
    lower_bound : float
        分析值下限（默认0）
    upper_bound : float
        分析值上限（默认1）
    threshold : float
        多空分界点（默认0.5）
    """
    analyzer = ConfidenceSliceAnalyzer(
        time_data=time_data,
        score_data=score_data,
        close_price_data=close_price_data,
        granularity=granularity,
        capital=capital,
        coefficient=coefficient,
        output_dir=output_dir,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        threshold=threshold,
    )
    analyzer.analyze()
