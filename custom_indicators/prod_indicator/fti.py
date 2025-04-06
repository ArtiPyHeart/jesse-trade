from typing import NamedTuple

import numpy as np
from jesse import helpers
from scipy import special


class FTIResult(NamedTuple):
    """FTI指标的返回结果"""

    fti: float  # 最佳周期的FTI值
    filtered_value: float  # 最佳周期的滤波值
    width: float  # 最佳周期的宽度
    best_period: float  # 具有最大FTI的周期


class FTI:
    """
    频率可调谐指标 (Frequency Tunable Indicator)

    原始指标由Govinda Khalsa开发，用于在价格数据中识别最佳周期结构。
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
        """初始化FTI指标计算器"""
        # 检查参数有效性
        if max_period < min_period or min_period < 2:
            raise ValueError("max_period必须大于min_period且min_period至少为2")
        if 2 * half_length < max_period:
            raise ValueError("2*half_length必须大于max_period")
        if lookback - half_length < 2:
            raise ValueError("lookback必须比half_length至少大2")

        self.use_log = use_log
        self.min_period = min_period
        self.max_period = max_period
        self.half_length = half_length
        self.lookback = lookback
        self.beta = beta
        self.noise_cut = noise_cut

        # 初始化数组
        self.y = np.zeros(lookback + half_length)
        self.coefs = np.zeros((max_period - min_period + 1, half_length + 1))
        self.filtered = np.zeros(max_period - min_period + 1)
        self.width = np.zeros(max_period - min_period + 1)
        self.fti_values = np.zeros(max_period - min_period + 1)
        self.sorted = np.zeros(max_period - min_period + 1, dtype=int)
        self.diff_work = np.zeros(lookback)
        self.leg_work = np.zeros(lookback)
        self.sort_work = np.zeros(max_period - min_period + 1)

        # 计算每个周期的滤波器系数
        for i in range(min_period, max_period + 1):
            self._find_coefs(i)

    def _find_coefs(self, period: int):
        """
        计算特定周期的滤波器系数

        该FIR低通滤波器来自Otnes: Applied Time Series Analysis
        """
        # 系数设置
        d = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])
        idx = period - self.min_period

        # 计算中心系数
        fact = 2.0 / period
        self.coefs[idx, 0] = fact

        # 计算其他系数
        fact *= np.pi
        for i in range(1, self.half_length + 1):
            self.coefs[idx, i] = np.sin(i * fact) / (i * np.pi)

        # 调整末端点
        self.coefs[idx, self.half_length] *= 0.5

        # 应用加权窗口并归一化
        sumg = self.coefs[idx, 0]
        for i in range(1, self.half_length + 1):
            sum_val = d[0]
            fact = i * np.pi / self.half_length
            for j in range(1, 4):
                sum_val += 2.0 * d[j] * np.cos(j * fact)
            self.coefs[idx, i] *= sum_val
            sumg += 2.0 * self.coefs[idx, i]

        # 归一化系数
        self.coefs[idx, :] /= sumg

    def process(self, data):
        """
        处理价格数据块并计算FTI指标

        :param data: 价格数据，最近的数据点在索引0
        """
        # 检查数据长度
        if len(data) < self.lookback:
            raise ValueError(f"数据长度必须至少为{self.lookback}")

        # 收集数据到本地数组，使其按时间顺序排列
        # 最近的案例将在索引lookback-1
        for i in range(self.lookback):
            if self.use_log:
                self.y[self.lookback - 1 - i] = np.log(data[i])
            else:
                self.y[self.lookback - 1 - i] = data[i]

        # 拟合最小二乘线并扩展
        self._extrapolate_data()

        # 处理每个周期
        for period_idx, period in enumerate(
            range(self.min_period, self.max_period + 1)
        ):
            self._process_period(period_idx, period)

        # 排序FTI局部最大值并保存排序后的索引
        self._sort_local_maxima()

        # 返回结果
        best_idx = self.sorted[0]
        best_period = self.min_period + best_idx

        # 获取相应的指标值
        if self.use_log:
            filtered_value = np.exp(self.filtered[best_idx])
            width_value = 0.5 * (
                np.exp(self.filtered[best_idx] + self.width[best_idx])
                - np.exp(self.filtered[best_idx] - self.width[best_idx])
            )
        else:
            filtered_value = self.filtered[best_idx]
            width_value = self.width[best_idx]

        # 计算最终的FTI值，包括Gamma累积分布函数变换
        fti_value = self.fti_values[best_idx]
        fti_transformed = 100.0 * special.gammainc(2.0, fti_value / 3.0) - 50.0

        return FTIResult(
            fti=fti_transformed,
            filtered_value=filtered_value,
            width=width_value,
            best_period=float(best_period),
        )

    def _extrapolate_data(self):
        """使用最小二乘线外推数据"""
        # 计算最近half_length+1个数据点的均值
        xmean = -0.5 * self.half_length
        ymean = np.mean(self.y[self.lookback - 1 - self.half_length : self.lookback])

        # 计算最小二乘线的斜率
        xsq = xy = 0.0
        for i in range(self.half_length + 1):
            xdiff = -i - xmean
            ydiff = self.y[self.lookback - 1 - i] - ymean
            xsq += xdiff * xdiff
            xy += xdiff * ydiff

        slope = xy / xsq

        # 扩展数据
        for i in range(self.half_length):
            self.y[self.lookback + i] = (i + 1.0 - xmean) * slope + ymean

    def _process_period(self, period_idx, period):
        """处理单个周期的数据"""
        # 获取该周期的滤波器系数
        coefs = self.coefs[period_idx]

        # 应用滤波器到数据块中的每个值
        self._apply_filter(period_idx, period, coefs)

        # 计算通道宽度
        self._calculate_width(period_idx)

        # 计算FTI值
        self._calculate_fti(period_idx)

    def _apply_filter(self, period_idx, period, coefs):
        """应用滤波器并收集移动腿"""
        # 初始化变量
        extreme_type = 0  # 未定义。1=高点; -1=低点
        extreme_value = 0.0
        n_legs = 0
        longest_leg = 0.0
        prior = 0.0

        # 对数据块中的每个点应用滤波器
        for iy in range(self.half_length, self.lookback):
            # 应用卷积滤波器
            sum_val = coefs[0] * self.y[iy]  # 中心点
            for i in range(1, self.half_length + 1):
                sum_val += coefs[i] * (self.y[iy + i] + self.y[iy - i])  # 对称滤波

            # 如果这是当前数据点的滤波值，保存它
            if iy == self.lookback - 1:
                self.filtered[period_idx] = sum_val

            # 保存实际值与滤波值之间的差异，用于宽度计算
            self.diff_work[iy - self.half_length] = abs(self.y[iy] - sum_val)

            # 收集移动腿
            if iy == self.half_length:  # 第一个点
                extreme_type = 0
                extreme_value = sum_val
                n_legs = 0
                longest_leg = 0.0

            elif extreme_type == 0:  # 等待第一个滤波价格变化
                if sum_val > extreme_value:
                    extreme_type = -1  # 第一个点是低点
                elif sum_val < extreme_value:
                    extreme_type = 1  # 第一个点是高点

            elif iy == self.lookback - 1:  # 最后一点，视为转折点
                if extreme_type:
                    leg_length = abs(extreme_value - sum_val)
                    self.leg_work[n_legs] = leg_length
                    n_legs += 1
                    if leg_length > longest_leg:
                        longest_leg = leg_length

            else:  # 内部前进
                if extreme_type == 1 and sum_val > prior:  # 下降后转为上升
                    leg_length = extreme_value - prior
                    self.leg_work[n_legs] = leg_length
                    n_legs += 1
                    if leg_length > longest_leg:
                        longest_leg = leg_length
                    extreme_type = -1
                    extreme_value = prior

                elif extreme_type == -1 and sum_val < prior:  # 上升后转为下降
                    leg_length = prior - extreme_value
                    self.leg_work[n_legs] = leg_length
                    n_legs += 1
                    if leg_length > longest_leg:
                        longest_leg = leg_length
                    extreme_type = 1
                    extreme_value = prior

            prior = sum_val

        # 保存最长腿长度和总腿数，供FTI计算使用
        self._longest_leg = longest_leg
        self._n_legs = n_legs

    def _calculate_width(self, period_idx):
        """计算通道宽度"""
        # 排序差异并找到分位数
        sorted_diffs = np.sort(self.diff_work[: self.lookback - self.half_length])
        i = int(self.beta * (self.lookback - self.half_length)) - 1
        if i < 0:
            i = 0

        # 保存通道宽度
        self.width[period_idx] = sorted_diffs[i]

    def _calculate_fti(self, period_idx):
        """计算FTI值"""
        # 计算噪声水平
        noise_level = self.noise_cut * self._longest_leg

        # 计算所有大于噪声水平的腿的平均值
        sum_val = 0.0
        n = 0
        for i in range(self._n_legs):
            if self.leg_work[i] > noise_level:
                sum_val += self.leg_work[i]
                n += 1

        # 计算非噪声腿的平均移动
        if n > 0:
            mean_move = sum_val / n
            self.fti_values[period_idx] = mean_move / (self.width[period_idx] + 1.0e-5)
        else:
            self.fti_values[period_idx] = 0.0

    def _sort_local_maxima(self):
        """排序FTI局部最大值并保存排序后的索引"""
        n = 0
        num_periods = self.max_period - self.min_period + 1

        # 找到局部最大值（包括两个端点）
        for i in range(num_periods):
            if (
                i == 0
                or i == num_periods - 1
                or (
                    self.fti_values[i] >= self.fti_values[i - 1]
                    and self.fti_values[i] >= self.fti_values[i + 1]
                )
            ):
                self.sort_work[n] = -self.fti_values[i]  # 要降序排列FTI，但qsort是升序
                self.sorted[n] = i
                n += 1

        # 对局部最大值进行排序
        if n > 0:
            # 使用argsort代替原始C++代码中的qsortdsi
            idx = np.argsort(self.sort_work[:n])
            temp_sorted = self.sorted[:n].copy()
            for i in range(n):
                self.sorted[i] = temp_sorted[idx[i]]

        # 填充剩余的排序索引为0，以防万一
        if n < num_periods:
            self.sorted[n:] = 0


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
