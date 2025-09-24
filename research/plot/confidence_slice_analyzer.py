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
    支持任意范围的值，根据阈值判断多空方向，计算每个切片的累积收益并可视化。
    """

    def __init__(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        volume_data: Union[pd.Series, np.ndarray, list],
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
        volume_data : array-like
            交易量数据
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
        self._validate_data(time_data, score_data, volume_data)
        self._prepare_dataframe(time_data, score_data, volume_data)

    def _validate_data(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        volume_data: Union[pd.Series, np.ndarray, list],
    ):
        """验证输入数据的有效性"""
        # 检查长度
        assert (
            len(time_data) == len(score_data) == len(volume_data)
        ), "三列数据长度必须相等"

        # 转换为numpy array进行验证
        score_array = np.asarray(score_data)

        # 检查超出范围的值
        out_of_range_mask = (score_array < self.lower_bound) | (score_array > self.upper_bound)
        if np.any(out_of_range_mask):
            out_count = np.sum(out_of_range_mask)
            out_ratio = out_count / len(score_array)
            min_val = np.min(score_array)
            max_val = np.max(score_array)
            print(f"\n⚠️ 警告：发现 {out_count} 个超出范围 [{self.lower_bound}, {self.upper_bound}] 的值")
            print(f"  - 占比: {out_ratio:.2%}")
            print(f"  - 实际范围: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  - 超出范围的值将被归类到最近的边界切片\n")

        # 检查粒度值的合理性
        range_size = self.upper_bound - self.lower_bound
        min_slices = range_size / self.granularity
        assert min_slices >= 2, f"粒度过大，至少需要2个切片。当前设置将产生 {min_slices:.1f} 个切片"
        if min_slices > 200:
            print(f"⚠️ 警告：粒度过小，将产生 {int(min_slices)} 个切片，可能影响性能")

    def _prepare_dataframe(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        volume_data: Union[pd.Series, np.ndarray, list],
    ):
        """将输入数据整合为内部DataFrame"""
        self.data = pd.DataFrame(
            {"timestamp": time_data, "score": score_data, "volume": volume_data}
        )
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
            # 初始化final列
            self.data["final"] = 0

            # 处理超出范围的值 - 归类到最近的边界切片
            score_clipped = self.data["score"].clip(self.lower_bound, self.upper_bound)

            # 筛选当前切片区间（包括被归类的超出范围值）
            mask = (score_clipped < upper) & (score_clipped >= lower)

            # 根据阈值决定交易方向
            # 切片中心点用于判断该切片的主要交易方向
            slice_center = (upper + lower) / 2
            if slice_center >= self.threshold:
                # 高于阈值，做多，volume为正
                self.data["final"] = np.where(mask, self.data["volume"], 0)
            else:
                # 低于阈值，做空，volume为负（表示做空收益）
                self.data["final"] = np.where(mask, -self.data["volume"], 0)

            # 计算样本占比
            true_count = mask.sum() / self.data_size

            # 计算累积收益
            self.data["high"] = self.data["final"].cumsum()
            self.data["open"] = self.coefficient * self.data["high"] / self.capital

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

            # 绘制数据
            ax.plot(time_series, self.data["open"])

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

            plt.title(
                f"切片区间 [{lower:.4f}, {upper:.4f}]{direction} - 样本占比: {true_count:.2%} - 累积收益曲线",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("时间", fontsize=12)
            plt.ylabel(
                f"收益 (系数={self.coefficient}, 本金={self.capital})", fontsize=12
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
    volume_data: Union[pd.Series, np.ndarray, list],
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
    volume_data : array-like
        交易量数据
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
        volume_data=volume_data,
        granularity=granularity,
        capital=capital,
        coefficient=coefficient,
        output_dir=output_dir,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        threshold=threshold,
    )
    analyzer.analyze()
