import os

# 配置matplotlib中文显示
import platform
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    时序对齐说明：
    - 模型在时刻 i 的预测值 score[i] 是针对未来第 pred_next 根K线的预测
    - 算法将 score[i] 与 close_diff[i+pred_next] 进行对齐
    - 图表显示的时间轴是验证预测的时刻，而非做出预测的时刻
    - 原始数据长度为 N，对齐后有效数据长度为 N - pred_next
    """

    def __init__(
        self,
        time_data: Union[pd.Series, np.ndarray, list],
        score_data: Union[pd.Series, np.ndarray, list],
        close_price_data: Union[pd.Series, np.ndarray, list],
        pred_next: int,
        granularity: float = 0.01,
        capital: float = 10000,
        coefficient: float = 0.25,
        output_dir: str = "./temp/",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        threshold: float = 0.5,
        show_color_blocks: bool = True,
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
        pred_next : int
            模型预测的是未来第几根K线（必须>=1）
            例如：pred_next=1表示预测下一根K线，pred_next=5表示预测未来第5根K线
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
        show_color_blocks : bool
            是否在图表中显示盈利/亏损的红绿色块（默认True）
        """
        self.granularity = granularity
        self.capital = capital
        self.coefficient = coefficient
        self.output_dir = output_dir
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.threshold = threshold
        self.pred_next = pred_next
        self.show_color_blocks = show_color_blocks

        # 验证参数
        assert isinstance(pred_next, int) and pred_next >= 1, "pred_next必须是>=1的整数"
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

        # 检查数据长度是否足够进行时序对齐
        data_len = len(time_data)
        assert (
            data_len > self.pred_next
        ), f"数据长度({data_len})必须大于pred_next({self.pred_next})才能进行时序对齐"

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
            print("  - 超出范围的值将被归类到最近的边界切片\n")

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
        """将输入数据整合为内部DataFrame，实现正确的时序对齐"""
        # 转换为numpy array便于处理
        time_array = np.asarray(time_data)
        score_array = np.asarray(score_data)
        close_array = np.asarray(close_price_data)

        # 计算价格差分（第一个值设为0）
        close_diff = np.diff(close_array, prepend=close_array[0])
        close_diff[0] = 0  # 确保第一个值为0

        # 时序对齐：
        # score[i] 预测 -> close_diff[i+pred_next]
        # 截断数据以对齐（去掉最后pred_next个没有对应未来价格的预测）
        aligned_score = score_array[: -self.pred_next]  # 前 N-pred_next 个预测
        aligned_close_diff = close_diff[self.pred_next :]  # 后 N-pred_next 个价格变化
        aligned_timestamp = time_array[
            self.pred_next :
        ]  # 后 N-pred_next 个时间戳（验证时刻）
        aligned_close_price = close_array[self.pred_next :]  # 后 N-pred_next 个收盘价

        # 根据阈值生成交易信号：>= threshold为做多(1)，< threshold为做空(-1)
        signal = np.where(aligned_score >= self.threshold, 1, -1)

        # 计算每个时间点的盈亏（价格变化 * 交易信号）
        pnl = aligned_close_diff * signal

        # 创建对齐后的DataFrame
        self.data = pd.DataFrame(
            {
                "timestamp": aligned_timestamp,
                "score": aligned_score,
                "close_price": aligned_close_price,
                "close_diff": aligned_close_diff,
                "signal": signal,
                "pnl": pnl,
            }
        )

        self.data_size = len(self.data)

        # 打印时序对齐信息
        print("\n📊 时序对齐信息:")
        print(f"  - pred_next: {self.pred_next} (预测未来第{self.pred_next}根K线)")
        print(f"  - 原始数据长度: {len(time_data)}")
        print(f"  - 对齐后数据长度: {self.data_size}")
        print(f"  - 丢弃的尾部数据: {self.pred_next} 条\n")

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

            # 准备绘图数据
            # 使用简单索引作为x轴（每个点间隔1单位）
            x_axis = np.arange(len(self.data))
            y_values = self.data["cumulative_return"].values

            # 计算每一步的变化量（基于变化方向）
            dy = np.diff(y_values, prepend=y_values[0])  # 第一个点变化为0

            # 盈利面积 = 上升时的增量之和
            profit_area = np.sum(dy[dy > 0])
            # 亏损面积 = 下降时的减量之和（取绝对值）
            loss_area = np.sum(np.abs(dy[dy < 0]))
            total_area = profit_area + loss_area

            # 计算盈利面积占比（防止除零）
            if total_area > 1e-10:
                profit_ratio = profit_area / total_area
            else:
                profit_ratio = 0.5  # 曲线始终为0时设为中性

            # 绘图
            fig, ax = plt.subplots(figsize=(20, 10))

            # 根据变化方向填充颜色（状态延续逻辑）
            # 上升后持平保持绿色，下降后持平保持红色，直到方向改变
            if self.show_color_blocks and len(y_values) > 1:
                # 计算变化方向：1=上升，-1=下降，0=持平
                direction = np.sign(np.diff(y_values))

                # 构建状态序列（忽略持平，保持上一个非零状态）
                state = np.zeros(len(direction), dtype=int)
                current_state = 0  # 初始状态未知

                for i in range(len(direction)):
                    if direction[i] != 0:  # 非持平，更新状态
                        current_state = direction[i]
                    state[i] = current_state

                # 识别连续相同状态的段
                segments = []
                if len(state) > 0:
                    # 跳过开头的0状态（如果有）
                    start_idx = 0
                    while start_idx < len(state) and state[start_idx] == 0:
                        start_idx += 1

                    if start_idx < len(state):
                        current_state = state[start_idx]

                        for i in range(start_idx + 1, len(state)):
                            if state[i] != current_state:
                                # 状态改变，保存上一段
                                if current_state != 0:
                                    segments.append((start_idx, i, current_state))
                                start_idx = i
                                current_state = state[i]

                        # 添加最后一段
                        if current_state != 0:
                            segments.append((start_idx, len(state), current_state))

                # 填充每个状态段
                for start, end, state_sign in segments:
                    if state_sign > 0:  # 上升状态（绿色）
                        ax.fill_between(
                            x_axis[start : end + 1],
                            y_values[start : end + 1],
                            0,
                            color="lightgreen",
                            alpha=0.3,
                            edgecolor="none",
                        )
                    elif state_sign < 0:  # 下降状态（红色）
                        ax.fill_between(
                            x_axis[start : end + 1],
                            y_values[start : end + 1],
                            0,
                            color="lightcoral",
                            alpha=0.3,
                            edgecolor="none",
                        )

                # 添加图例（手动创建）
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor="lightgreen", alpha=0.3, label="上升状态（盈利）"),
                    Patch(facecolor="lightcoral", alpha=0.3, label="下降状态（亏损）"),
                ]

            # 绘制主曲线
            ax.plot(x_axis, y_values, label="累积盈亏", color="blue", linewidth=1.5)

            # 计算并绘制均值线
            mean_value = y_values.mean()
            ax.axhline(
                y=mean_value,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"均值: {mean_value:.4f}",
            )

            # 添加零线作为参考（使用黑色加粗线条以在灰色网格中突出显示）
            ax.axhline(
                y=0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, zorder=5
            )

            # 添加图例（包含颜色填充说明）
            if self.show_color_blocks and len(y_values) > 1:
                handles, labels = ax.get_legend_handles_labels()
                handles = legend_elements + handles
                ax.legend(handles=handles, loc="best", fontsize=10)
            else:
                ax.legend(loc="best", fontsize=10)

            # 设置x轴刻度 - 显示实际时间但计算时用索引
            time_series = self.data["timestamp"]
            try:
                time_series = pd.to_datetime(time_series)
                is_datetime = True
            except Exception:
                is_datetime = False

            tick_indices = np.linspace(
                0, len(x_axis) - 1, min(tick_count, len(x_axis)), dtype=int
            )
            ax.set_xticks(tick_indices)

            if is_datetime:
                ax.set_xticklabels(
                    [time_series.iloc[i].strftime("%Y-%m-%d") for i in tick_indices]
                )
            else:
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
                + f"最终收益: {final_return:.4f} ({profit_status}) | 平均收益: {mean_return:.4f}\n"
                + f"盈利面积占比: {profit_ratio:.2%} (盈利:{profit_area:.2f} vs 亏损:{loss_area:.2f})",
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
    pred_next: int,
    granularity: float = 0.01,
    capital: float = 1000,
    coefficient: float = 0.25,
    output_dir: str = "./temp/",
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    threshold: float = 0.5,
    show_color_blocks: bool = False,
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
    pred_next : int
        模型预测的是未来第几根K线（必须>=1）
        例如：pred_next=1表示预测下一根K线，pred_next=5表示预测未来第5根K线
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
    show_color_blocks : bool
        是否在图表中显示盈利/亏损的红绿色块（默认False）
    """
    analyzer = ConfidenceSliceAnalyzer(
        time_data=time_data,
        score_data=score_data,
        close_price_data=close_price_data,
        pred_next=pred_next,
        granularity=granularity,
        capital=capital,
        coefficient=coefficient,
        output_dir=output_dir,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        threshold=threshold,
        show_color_blocks=show_color_blocks,
    )
    analyzer.analyze()
