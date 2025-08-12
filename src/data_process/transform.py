import warnings

import numpy as np


def fisher_transform(src: np.ndarray) -> np.ndarray:
    """
    Fisher变换的目的在于将任何原本均值接近0且取值在-1到+1之间的指标转换成一个近似正态分布的指标。
    计算公式为：fisher(x) = 0.5 * ln((1 + x) / (1 - x))
    注意: 当输入x接近1或-1时，(1 - x)或(1 + x)接近0，可能会导致数值不稳定，因此需要先对输入值进行clip处理。

    参数:
        src (np.ndarray): 输入的指标数组，假定其数值在[-1, 1]之间。

    返回:
        np.ndarray: 经Fisher变换后的指标数组。
    """
    # 新增警告，若任何值的绝对值大于1则告警
    if np.any(np.abs(src) > 1):
        warnings.warn(
            "fisher_transform: 输入数组中存在超过[-1, 1]的值，这可能导致失真或数值错误。"
        )

    # 对输入数组进行限制，避免出现1或-1导致分母为0的问题
    x = np.clip(src, -0.999, 0.999)
    # 根据公式计算Fisher变换值
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    return fisher


def inverse_fisher_transform(src: np.ndarray) -> np.ndarray:
    """
    逆Fisher变换的目的是对Fisher变换后极端区域进行压缩，
    通过压缩在极值附近的数值，可以减少无关的细微波动，使得指标的真实含义更易于理解。
    逆Fisher变换实际上是Fisher变换的反函数，计算公式为:
        inverse_fisher(y) = (exp(2*y) - 1) / (exp(2*y) + 1)

    参数:
        src (np.ndarray): 输入的Fisher变换后的数组。

    返回:
        np.ndarray: 经逆Fisher变换后的指标数组，数值被限制在-1到+1之间。
    """
    # 直接根据逆变换公式进行计算
    inv_fisher = (np.exp(2 * src) - 1) / (np.exp(2 * src) + 1)
    return inv_fisher


def cube_transform(src: np.ndarray) -> np.ndarray:
    """
    立方变换的目的是对位于零附近的信号进行压缩。
    当交易系统中的指标在-1到+1之间波动时，
    通过对这些指标进行立方运算可以使得接近0的细微波动被大大减少，而远离零的信号保持较大幅度。
    计算公式为: cube(x) = x^3

    参数:
        src (np.ndarray): 输入的指标数组，其数值通常被限制在-1到+1之间。

    返回:
        np.ndarray: 经立方变换后的指标数组。
    """
    # 新增警告，若任何值的绝对值大于1则告警
    if np.any(np.abs(src) > 1):
        warnings.warn(
            "cube_transform: 输入数组中存在超过[-1, 1]的值，这可能导致失真或数值错误。"
        )

    # 使用numpy的power函数进行立方运算
    return np.power(src, 3)
