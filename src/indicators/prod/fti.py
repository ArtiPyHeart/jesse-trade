"""
FTI (Frequency Tunable Indicator) - 频率可调谐指标

架构说明:
- 生产环境: 强制使用 Rust 实现 (_rust_indicators.fti_process_py)
- 测试环境: 保留 Python numba 函数用于数值对齐验证

已废弃的 Python 实现部分已注释，仅保留用于测试的 numba 函数。
"""

from typing import NamedTuple

# 导入 Rust 实现
import _rust_indicators
import numpy as np
from jesse import helpers
from numba import njit


class FTIResult(NamedTuple):
    """FTI指标的返回结果"""

    fti: float  # 最佳周期的FTI值
    filtered_value: float  # 最佳周期的滤波值
    width: float  # 最佳周期的宽度
    best_period: float  # 具有最大FTI的周期


# ============================================================================
# Python Numba 实现 - 仅用于测试数值对齐验证
# 生产环境使用 Rust 实现，不调用以下函数
# ============================================================================


@njit
def _find_coefs_numba(min_period, period, half_length):
    """计算特定周期的滤波器系数 - numba加速版"""
    # 系数设置
    d = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])
    coefs = np.zeros(half_length + 1)

    # 计算中心系数
    fact = 2.0 / period
    coefs[0] = fact

    # 计算其他系数
    fact *= np.pi
    for i in range(1, half_length + 1):
        coefs[i] = np.sin(i * fact) / (i * np.pi)

    # 调整末端点
    coefs[half_length] *= 0.5

    # 应用加权窗口并归一化
    sumg = coefs[0]
    for i in range(1, half_length + 1):
        sum_val = d[0]
        fact_i = i * np.pi / half_length
        for j in range(1, 4):
            sum_val += 2.0 * d[j] * np.cos(j * fact_i)
        coefs[i] *= sum_val
        sumg += 2.0 * coefs[i]

    # 归一化系数
    if sumg != 0:
        coefs /= sumg

    return coefs


@njit
def _extrapolate_data_numba(y, lookback, half_length):
    """使用最小二乘线外推数据 - numba加速版"""
    # 计算最近half_length+1个数据点的均值
    xmean = -0.5 * half_length
    ymean = 0.0
    for i in range(lookback - half_length - 1, lookback):
        ymean += y[i]
    ymean /= half_length + 1

    # 计算最小二乘线的斜率
    xsq = 0.0
    xy = 0.0
    for i in range(half_length + 1):
        xdiff = -i - xmean
        ydiff = y[lookback - 1 - i] - ymean
        xsq += xdiff * xdiff
        xy += xdiff * ydiff

    slope = xy / xsq if xsq != 0 else 0.0

    # 扩展数据
    for i in range(half_length):
        y[lookback + i] = (i + 1.0 - xmean) * slope + ymean

    return y


@njit
def _apply_filter_numba(y, coefs, half_length, lookback, diff_work, leg_work):
    """应用滤波器并收集移动腿 - numba加速版"""
    # 初始化变量
    extreme_type = 0  # 未定义。1=高点; -1=低点
    extreme_value = 0.0
    n_legs = 0
    longest_leg = 0.0
    prior = 0.0
    filtered_value = 0.0

    # 对数据块中的每个点应用滤波器
    for iy in range(half_length, lookback):
        # 应用卷积滤波器
        sum_val = coefs[0] * y[iy]  # 中心点
        for i in range(1, half_length + 1):
            sum_val += coefs[i] * (y[iy + i] + y[iy - i])  # 对称滤波

        # 如果这是当前数据点的滤波值，保存它
        if iy == lookback - 1:
            filtered_value = sum_val

        # 保存实际值与滤波值之间的差异，用于宽度计算
        diff_work[iy - half_length] = abs(y[iy] - sum_val)

        # 收集移动腿
        if iy == half_length:  # 第一个点
            extreme_type = 0
            extreme_value = sum_val
            n_legs = 0
            longest_leg = 0.0

        elif extreme_type == 0:  # 等待第一个滤波价格变化
            if sum_val > extreme_value:
                extreme_type = -1  # 第一个点是低点
            elif sum_val < extreme_value:
                extreme_type = 1  # 第一个点是高点

        elif iy == lookback - 1:  # 最后一点，视为转折点
            if extreme_type != 0:
                leg_length = abs(extreme_value - sum_val)
                leg_work[n_legs] = leg_length
                n_legs += 1
                if leg_length > longest_leg:
                    longest_leg = leg_length

        else:  # 内部前进
            if extreme_type == 1 and sum_val > prior:  # 下降后转为上升
                leg_length = extreme_value - prior
                leg_work[n_legs] = leg_length
                n_legs += 1
                if leg_length > longest_leg:
                    longest_leg = leg_length
                extreme_type = -1
                extreme_value = prior

            elif extreme_type == -1 and sum_val < prior:  # 上升后转为下降
                leg_length = prior - extreme_value
                leg_work[n_legs] = leg_length
                n_legs += 1
                if leg_length > longest_leg:
                    longest_leg = leg_length
                extreme_type = 1
                extreme_value = prior

        prior = sum_val

    return filtered_value, longest_leg, n_legs, diff_work, leg_work


@njit
def _calculate_width_numba(diff_work, lookback, half_length, beta):
    """计算通道宽度 - numba加速版"""
    # 创建副本进行排序以避免修改原始数组
    sorted_diffs = np.sort(diff_work[: lookback - half_length])
    i = int(beta * (lookback - half_length)) - 1
    if i < 0:
        i = 0

    # 返回通道宽度
    return sorted_diffs[i]


@njit
def _calculate_fti_numba(leg_work, width, n_legs, longest_leg, noise_cut):
    """计算FTI值 - numba加速版"""
    # 计算噪声水平
    noise_level = noise_cut * longest_leg

    # 计算所有大于噪声水平的腿的平均值
    sum_val = 0.0
    n = 0
    for i in range(n_legs):
        if leg_work[i] > noise_level:
            sum_val += leg_work[i]
            n += 1

    # 计算非噪声腿的平均移动
    if n > 0:
        mean_move = sum_val / n
        return mean_move / (width + 1.0e-5)
    else:
        return 0.0


@njit
def _sort_local_maxima_numba(fti_values, min_period, max_period):
    """排序FTI局部最大值并保存排序后的索引 - numba加速版"""
    num_periods = max_period - min_period + 1
    sorted_indices = np.zeros(num_periods, dtype=np.int32)
    sort_work = np.zeros(num_periods)

    # 找到局部最大值（包括两个端点）
    n = 0
    for i in range(num_periods):
        if (
            i == 0
            or i == num_periods - 1
            or (
                fti_values[i] >= fti_values[i - 1]
                and fti_values[i] >= fti_values[i + 1]
            )
        ):
            sort_work[n] = -fti_values[i]  # 要降序排列FTI，但排序是升序
            sorted_indices[n] = i
            n += 1

    # 对局部最大值进行排序
    if n > 0:
        # 手动实现简单的排序，因为numba不支持argsort的完整功能
        for i in range(n):
            for j in range(i + 1, n):
                if sort_work[i] > sort_work[j]:
                    # 交换值和索引
                    temp_val = sort_work[i]
                    temp_idx = sorted_indices[i]
                    sort_work[i] = sort_work[j]
                    sorted_indices[i] = sorted_indices[j]
                    sort_work[j] = temp_val
                    sorted_indices[j] = temp_idx

    # 填充剩余的排序索引为0
    if n < num_periods:
        for i in range(n, num_periods):
            sorted_indices[i] = 0

    return sorted_indices


class FTI:
    """
    频率可调谐指标 (Frequency Tunable Indicator)

    原始指标由Govinda Khalsa开发，用于在价格数据中识别最佳周期结构。

    注意: 生产环境强制使用 Rust 实现，Python 实现部分已废弃。
    """

    def __init__(
        self,
        use_log: bool = True,  # 是否对价格取对数
        min_period: int = 5,  # 最短周期，至少为2
        max_period: int = 65,  # 最长周期
        half_length: int = 35,  # 中心系数两侧的系数数量
        lookback: int = 150,  # 处理数据的块长度
        beta: float = 0.95,  # 宽度计算的分位数（通常0.8-0.99）
        noise_cut: float = 0.20,  # 定义FTI噪声的最长内部移动的分数
    ):
        """初始化FTI指标计算器（仅存储参数，实际计算由 Rust 完成）"""
        # 检查参数有效性
        if max_period < min_period or min_period < 2:
            raise ValueError("max_period必须大于min_period且min_period至少为2")
        if 2 * half_length < max_period:
            raise ValueError("2*half_length必须大于max_period")
        if lookback - half_length < 2:
            raise ValueError("lookback必须比half_length至少大2")

        # 存储参数（传递给 Rust）
        self.use_log = use_log
        self.min_period = min_period
        self.max_period = max_period
        self.half_length = half_length
        self.lookback = lookback
        self.beta = beta
        self.noise_cut = noise_cut

        # 以下数组初始化已废弃 - Rust 实现不需要
        # self.y = np.zeros(lookback + half_length)
        # self.coefs = np.zeros((max_period - min_period + 1, half_length + 1))
        # self.filtered = np.zeros(max_period - min_period + 1)
        # self.width = np.zeros(max_period - min_period + 1)
        # self.fti_values = np.zeros(max_period - min_period + 1)
        # self.sorted = np.zeros(max_period - min_period + 1, dtype=np.int32)
        # self.diff_work = np.zeros(lookback)
        # self.leg_work = np.zeros(lookback)
        # self.sort_work = np.zeros(max_period - min_period + 1)

        # 以下系数预计算已废弃 - Rust 实现不需要
        # for i in range(min_period, max_period + 1):
        #     self._find_coefs(i)

    # 已废弃方法 - Rust 实现不需要
    # def _find_coefs(self, period: int):
    #     """计算特定周期的滤波器系数"""
    #     idx = period - self.min_period
    #     self.coefs[idx] = _find_coefs_numba(self.min_period, period, self.half_length)

    def process(self, data):
        """
        处理价格数据块并计算FTI指标

        :param data: 价格数据，最近的数据点在索引0
        """

        fti, filtered_value, width, best_period = _rust_indicators.fti_process_py(
            data,
            use_log=self.use_log,
            min_period=self.min_period,
            max_period=self.max_period,
            half_length=self.half_length,
            lookback=self.lookback,
            beta=self.beta,
            noise_cut=self.noise_cut,
        )
        return FTIResult(
            fti=fti,
            filtered_value=filtered_value,
            width=width,
            best_period=best_period,
        )

    # 已废弃方法 - Rust 实现不需要
    # def _process_period(self, period_idx, period):
    #     """处理单个周期的数据"""
    #     # 获取该周期的滤波器系数
    #     coefs = self.coefs[period_idx]
    #
    #     # 应用滤波器到数据块中的每个值
    #     filtered_value, longest_leg, n_legs, self.diff_work, self.leg_work = (
    #         _apply_filter_numba(
    #             self.y,
    #             coefs,
    #             self.half_length,
    #             self.lookback,
    #             self.diff_work,
    #             self.leg_work,
    #         )
    #     )
    #
    #     # 保存滤波值
    #     self.filtered[period_idx] = filtered_value
    #
    #     # 计算通道宽度
    #     self.width[period_idx] = _calculate_width_numba(
    #         self.diff_work, self.lookback, self.half_length, self.beta
    #     )
    #
    #     # 计算FTI值
    #     self.fti_values[period_idx] = _calculate_fti_numba(
    #         self.leg_work, self.width[period_idx], n_legs, longest_leg, self.noise_cut
    #     )


def fti(
    candles: np.ndarray,
    lookback: int = 150,
    half_length: int = 35,
    min_period: int = 5,
    max_period: int = 65,
    use_log: bool = True,
    sequential: bool = False,
):
    """
    频率可调谐指标 (Frequency Tunable Indicator)

    由Govinda Khalsa开发的指标，用于识别价格数据中的优势周期结构。
    返回包含多个指标值的命名元组。

    :param candles: K线数据
    :param lookback: 回看周期长度
    :param half_length: 滤波器半长度
    :param min_period: 最小周期
    :param max_period: 最大周期
    :param use_log: 是否使用对数价格
    :param sequential: 是否返回整个序列
    :return: FTIResult命名元组或其数组
    """
    candles = helpers.slice_candles(candles, sequential)

    # 获取收盘价
    source = helpers.get_candle_source(candles, "close")

    # 初始化结果数组
    if sequential:
        fti_values = np.zeros(len(candles))
        filtered_values = np.zeros(len(candles))
        width_values = np.zeros(len(candles))
        period_values = np.zeros(len(candles))

        # 初始化未定义的值
        front_bad = lookback - 1
        fti_values[:front_bad] = 0
        filtered_values[:front_bad] = 0
        width_values[:front_bad] = 0
        period_values[:front_bad] = 0

        # 创建FTI计算器
        calculator = FTI(
            use_log=use_log,
            min_period=min_period,
            max_period=max_period,
            half_length=half_length,
            lookback=lookback,
            beta=0.95,
            noise_cut=0.20,
        )

        # 对每个有效点计算FTI
        for i in range(front_bad, len(candles)):
            data_window = source[i - lookback + 1 : i + 1][
                ::-1
            ]  # 反转使最近的数据点在索引0
            result = calculator.process(data_window)

            fti_values[i] = result.fti
            filtered_values[i] = result.filtered_value
            width_values[i] = result.width
            period_values[i] = result.best_period

        # 返回序列结果
        return FTIResult(
            fti=fti_values,
            filtered_value=filtered_values,
            width=width_values,
            best_period=period_values,
        )
    else:
        # 仅计算最后一个点的FTI
        data_window = source[-lookback:][::-1]  # 反转使最近的数据点在索引0

        # 创建FTI计算器并处理数据
        calculator = FTI(
            use_log=use_log,
            min_period=min_period,
            max_period=max_period,
            half_length=half_length,
            lookback=lookback,
            beta=0.95,
            noise_cut=0.20,
        )

        # 返回结果
        return calculator.process(data_window)
