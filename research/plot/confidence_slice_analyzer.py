import os
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    置信度切片分析工具

    用于分析机器学习模型的置信度分布，通过精细切片找出真正赚钱的score区间。
    将0.5以上的置信度进行切片，计算每个切片的累积收益并可视化。
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
    ):
        """
        初始化分析器

        Parameters
        ----------
        time_data : array-like
            时间数据
        score_data : array-like
            置信度数据 (0-1之间)
        volume_data : array-like
            交易量数据
        granularity : float
            切片粒度 (0.1, 0.05, 或 0.01)
        capital : float
            初始资金
        coefficient : float
            收益系数
        output_dir : str
            输出目录
        """
        self.granularity = granularity
        self.capital = capital
        self.coefficient = coefficient
        self.output_dir = output_dir

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

        # 检查score范围
        assert np.all(
            (score_array >= 0) & (score_array <= 1)
        ), "置信度数据必须在[0,1]范围内"

        # 检查粒度值
        assert self.granularity in [0.1, 0.05, 0.01], "粒度必须是0.1, 0.05或0.01之一"

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
        """根据粒度获取切片参数"""
        if self.granularity == 0.1:
            return 1.0, 0.9, 5
        elif self.granularity == 0.05:
            return 1.0, 0.95, 10
        elif self.granularity == 0.01:
            return 1.0, 0.99, 50
        else:
            raise ValueError(
                f"不支持的粒度值: {self.granularity}。"
                f"请使用以下值之一: 0.1, 0.05, 0.01"
            )

    def analyze(self):
        """执行完整的置信度切片分析"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 获取切片参数
        da, xiao, z = self._get_slice_params()

        # 计算x轴刻度数（基于数据量，最多25个）
        tick_count = min(25, self.data_size)

        # 对每个切片进行分析
        for i in range(z):
            # 初始化final列
            self.data["final"] = 0

            # 筛选当前置信度区间
            mask = (self.data["score"] < da) & (self.data["score"] >= xiao)
            self.data["final"] = np.where(mask, self.data["volume"], 0)

            # 计算样本占比
            true_count = mask.sum() / self.data_size

            # 更新置信度边界
            da = round((da - self.granularity), 2)
            xiao = round((xiao - self.granularity), 2)

            # 计算累积收益
            self.data["high"] = self.data["final"].cumsum()
            self.data["open"] = self.coefficient * self.data["high"] / self.capital

            # 绘图
            fig = plt.figure(figsize=(20, 10))
            plt.plot(self.data["timestamp"], self.data["open"])
            plt.xticks(range(0, self.data_size, int(self.data_size / tick_count)))

            # 添加标题和标签
            plt.title(
                f"置信度区间 [{xiao:.2f}, {da:.2f}] - 样本占比: {true_count:.2%} - 累积收益曲线",
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
            filename = f"conf_{xiao:.2f}_{da:.2f}_ratio_{true_count:.4f}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            print(
                f"已生成: {filename} (置信度区间: [{xiao:.2f}, {da:.2f}], 样本占比: {true_count:.2%})"
            )


def analyze_confidence_slices(
    time_data: Union[pd.Series, np.ndarray, list],
    score_data: Union[pd.Series, np.ndarray, list],
    volume_data: Union[pd.Series, np.ndarray, list],
    granularity: float = 0.01,
    capital: float = 1000,
    coefficient: float = 0.25,
    output_dir: str = "./temp/",
):
    """
    便捷函数：执行置信度切片分析

    Parameters
    ----------
    time_data : array-like
        时间数据
    score_data : array-like
        置信度数据 (0-1之间)
    volume_data : array-like
        交易量数据
    granularity : float
        切片粒度 (0.1, 0.05, 或 0.01)
    capital : float
        初始资金
    coefficient : float
        收益系数
    output_dir : str
        输出目录
    """
    analyzer = ConfidenceSliceAnalyzer(
        time_data=time_data,
        score_data=score_data,
        volume_data=volume_data,
        granularity=granularity,
        capital=capital,
        coefficient=coefficient,
        output_dir=output_dir,
    )
    analyzer.analyze()
