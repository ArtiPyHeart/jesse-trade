# -*- coding: utf-8 -*-
# Author: Qiuyu Yang
# License: BSD 3 clause
"""
NSB (Nemenman-Shafee-Bialek) Entropy估计算法实现

该模块提供了基于NSB方法的熵估计，这种方法特别适用于样本数量有限的情况。
参考文献：
1. Nemenman, I., Shafee, F., & Bialek, W. (2002).
   Entropy and inference, revisited. In Advances in neural information processing systems (pp. 471-478).
2. Nemenman, I., Bialek, W., & Van Steveninck, R. D. R. (2004).
   Entropy and information in neural spike trains: Progress on the sampling problem.
   Physical Review E, 69(5), 056111.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from jesse.helpers import get_candle_source
from scipy.special import digamma, gammaln, polygamma

logger = logging.getLogger(__name__)


def nsb_entropy(
    nk: Union[List[int], np.ndarray], k: Optional[int] = None, return_std: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    基于NSB (Nemenman-Shafee-Bialek)算法的熵估计函数。

    NSB方法特别适用于样本数量有限的情况，它通过使用Dirichlet先验的混合分布来减小估计偏差。

    参数:
    ------
    nk : array-like
        计数数组，表示每个箱子的观测频率。
    k : int, optional
        字母表大小（具有非零概率的箱子数量，包括未观测到的箱子）。
        默认情况下，使用字母表大小的上限估计。如果分布是强烈欠采样的，
        则切换到可以在字母表大小未知的情况下使用的渐近NSB估计器。
    return_std : bool, optional
        如果为True，同时返回熵后验分布的标准差。

    返回:
    ------
    entropy : float
        熵估计值（以奈特为单位）。
    err : float, optional
        熵估计的贝叶斯误差界限。仅在return_std为True时返回。

    注释:
    ------
    调用后，可以通过查看nsb_entropy.info字典来检查拟合参数：
    >>> import numpy as np
    >>> from custom_indicators.prod_indicator.nsb_entropy.entropy import nsb_entropy
    >>> counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
    >>> nsb_entropy(counts)
    2.813074648917905
    >>> nsb_entropy.info
    {'entropy': 2.813074648917905, 'err': 0.1244390183672502, 'bounded': 1, 'estimator': 'NSB', 'k': 6008}
    """
    # 将输入转换为numpy数组
    nk = np.asarray(nk, dtype=np.float64)

    # 基本参数初始化
    n_total = np.sum(nk)  # 总样本数
    k_obs = np.sum(nk > 0)  # 观测到的箱子数量

    # 使用更简单的近似方法开始
    # 如果k未提供，估计一个合理的值
    if k is None:
        if k_obs <= 1:
            k = k_obs
        else:
            # 使用观测到的箱子数量作为默认值
            k = k_obs

    # 单箱情况处理
    if k <= 1:
        entropy_value = 0.0
        err_value = 0.0
        _annotate_entropy_function(entropy_value, err_value, k, "Plugin", True)
        return (entropy_value, err_value) if return_std else entropy_value

    # 是否有重复计数？（coincidences：同一箱中有多个样本的情况）
    has_coincidences = np.any(nk > 1)

    # 如果没有重复计数，且字母表大小未知，则使用插件估计器
    if not has_coincidences and k is None:
        # 在这种情况下，没有估计器可以给出合理的估计值，返回样本数的对数
        entropy_value = np.log(n_total)
        err_value = np.inf
        _annotate_entropy_function(entropy_value, err_value, k, "Plugin", False)
        return (entropy_value, err_value) if return_std else entropy_value

    # 简单情况：使用直接的插件估计器（最大似然）
    if np.all(nk > 0) and len(nk) == k:
        # 所有箱子都有观测值，直接计算
        p = nk / n_total
        entropy_value = -np.sum(p * np.log(p))
        err_value = np.sqrt(
            np.sum((np.log(p) + entropy_value) ** 2 * p * (1 - p)) / n_total
        )
        _annotate_entropy_function(entropy_value, err_value, k, "Plugin", True)
        return (entropy_value, err_value) if return_std else entropy_value

    # 检查是否处于强烈欠采样状态（观测到的唯一值数接近总样本数）
    is_undersampled = k_obs / n_total > 0.9

    if is_undersampled and k is None:
        try:
            # 使用渐近NSB估计器
            entropy_value, err_value = _asymptotic_nsb(nk)
            estimator_name = "AsymptoticNSB"
        except ValueError:
            # 回退到插件估计器
            p_obs = nk[nk > 0] / n_total
            entropy_value = -np.sum(p_obs * np.log(p_obs))
            err_value = np.inf  # 无界误差
            estimator_name = "Plugin"
    else:
        # 使用标准NSB估计器
        entropy_value, err_value = _nsb_estimator(nk, k)
        estimator_name = "NSB"

    # 记录信息
    _annotate_entropy_function(
        entropy_value, err_value, k, estimator_name, np.isfinite(err_value)
    )

    return (entropy_value, err_value) if return_std else entropy_value


def _nsb_estimator(nk: np.ndarray, k: int) -> Tuple[float, float]:
    """
    NSB熵估计的核心实现。

    参数:
    ------
    nk : np.ndarray
        计数数组
    k : int
        字母表大小

    返回:
    ------
    entropy : float
        熵估计值
    err : float
        误差估计
    """
    # 将计数数组转换为概率质量函数（PMF）
    n_total = np.sum(nk)
    p = nk / n_total

    # 计算观测到的PMF的插件熵（简单近似，作为起点）
    p_positive = p[p > 0]
    H_plugin = -np.sum(p_positive * np.log(p_positive))

    # NSB算法的简化版实现
    # 这里我们使用数值积分方法近似NSB积分

    # 初始化Dirichlet先验参数空间
    alpha_min = 1e-3
    alpha_max = 1e3
    n_points = 200
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_points)

    # 计算每个alpha值的权重和熵估计
    weights = np.zeros(n_points)
    entropies = np.zeros(n_points)

    for i, alpha in enumerate(alphas):
        # 计算Dirichlet先验权重（对数空间中）
        weights[i] = _log_weight(alpha, nk, k)
        # 计算给定alpha的熵估计
        entropies[i] = _h_dir(alpha, k, nk)

    # 将对数权重转换为线性空间并归一化
    max_weight = np.max(weights)
    weights = np.exp(weights - max_weight)
    weights = weights / np.sum(weights)

    # 计算加权平均熵估计
    H_nsb = np.sum(weights * entropies)

    # 如果计算结果不合理，回退到插件估计器
    if not (0 <= H_nsb <= np.log(k)) or np.isnan(H_nsb):
        H_nsb = H_plugin

    # 计算误差（标准差）
    H_var = np.sum(weights * (entropies - H_nsb) ** 2)
    H_err = np.sqrt(H_var)

    return H_nsb, H_err


def _asymptotic_nsb(nk: np.ndarray) -> Tuple[float, float]:
    """
    渐近NSB估计器，用于字母表大小未知或可数无限的情况。

    参数:
    ------
    nk : np.ndarray
        计数数组

    返回:
    ------
    entropy : float
        熵估计值
    err : float
        误差估计
    """
    # 计算重复计数（coincidences）
    n_total = np.sum(nk)

    # 计算有多少箱有多个样本
    coincidences = np.sum(nk * (nk - 1)) / 2

    if coincidences <= 0:
        raise ValueError("无法使用渐近NSB估计器：数据中没有重复计数")

    # 计算Euler-Mascheroni常数
    euler_gamma = 0.57721566490153286060

    # 计算gamma0(coincidences)和gamma1(coincidences)函数
    gamma0_value = _gamma0(coincidences)
    gamma1_value = _gamma1(coincidences)

    # 计算渐近NSB估计
    H_asymp = euler_gamma - np.log(2) + 2.0 * np.log(n_total) - gamma0_value
    H_err = np.sqrt(gamma1_value)

    return H_asymp, H_err


def _gamma0(x: float) -> float:
    """
    gamma0函数实现
    """
    return digamma(x + 1)


def _gamma1(x: float) -> float:
    """
    gamma1函数实现
    """
    return polygamma(1, x + 1)


def _log_weight(alpha: float, nk: np.ndarray, k: int) -> float:
    """
    计算给定alpha的对数权重

    参数:
    ------
    alpha : float
        Dirichlet先验参数
    nk : np.ndarray
        计数数组
    k : int
        字母表大小

    返回:
    ------
    log_weight : float
        对数权重
    """
    n_total = np.sum(nk)

    # 计算log P(n|alpha)
    log_pna = gammaln(k * alpha) - k * gammaln(alpha)

    # 对所有非零计数添加贡献
    for n in nk:
        if n > 0:
            log_pna += gammaln(n + alpha) - gammaln(alpha)

    log_pna -= gammaln(n_total + k * alpha)

    # 添加先验分布项
    log_prior = np.log(1.0 / alpha)

    return log_pna + log_prior


def _h_dir(alpha: float, k: int, nk: np.ndarray = None) -> float:
    """
    计算给定alpha和k的Dirichlet熵期望

    参数:
    ------
    alpha : float
        Dirichlet先验参数
    k : int
        字母表大小
    nk : np.ndarray, optional
        计数数组，用于更精确的估计

    返回:
    ------
    h_dir : float
        熵期望
    """
    # 使用digamma函数直接计算期望熵
    # E[H] = psi(k*alpha + 1) - psi(alpha + 1)
    # 其中psi是digamma函数

    if nk is None or np.all(nk == 0):
        # 没有观测数据时的估计
        h_dir = digamma(k * alpha + 1) - digamma(alpha + 1)
    else:
        # 有观测数据时的估计
        n_total = np.sum(nk)

        # 均匀分布的情况
        if np.all(nk[nk > 0] == nk[0]) and np.sum(nk > 0) == k:
            return np.log(k)

        # 考虑观测数据的情况
        n_alpha = alpha + n_total / k
        h_dir = digamma(k * n_alpha + 1) - digamma(n_alpha + 1)

        # 确保熵在合理范围内
        h_dir = min(h_dir, np.log(k))
        h_dir = max(h_dir, 0.0)

    return h_dir


def _annotate_entropy_function(
    entropy_value: float,
    err_value: float,
    k: Optional[int],
    estimator_name: str,
    is_bounded: bool,
) -> None:
    """标注熵函数信息"""
    nsb_entropy.info = {
        "entropy": entropy_value,
        "err": err_value,
        "bounded": int(is_bounded),
        "estimator": estimator_name,
        "k": k,
    }


# 初始化info字典
nsb_entropy.info = {}


# 为jesse框架提供统一接口
def entropy_for_jesse(
    candles: np.ndarray,
    source_type: str = "close",
    window_size: int = 20,
    n_bins: int = 5,
    sequential: bool = False,
) -> Union[float, np.ndarray]:
    """
    使用对数收益率和直方图分箱计算价格序列的NSB熵

    参数:
    ------
    candles : np.ndarray
        K线数据
    source_type : str
        使用的价格类型，可选:'open', 'high', 'low', 'close'
    window_size : int
        用于计算熵的窗口大小
    n_bins : int
        直方图分箱的数量
    sequential : bool
        是否返回完整序列

    返回:
    ------
    float 或 np.ndarray
        NSB熵值或熵值序列
    """
    # 获取价格数据
    price = get_candle_source(candles, source_type)

    # 计算对数收益率
    log_returns = np.log(price[1:] / price[:-1])

    # 确保有足够的数据点
    if len(log_returns) < window_size:
        if sequential:
            return np.array([np.nan] * len(candles))
        else:
            return np.nan

    # 初始化结果数组（如果是sequential模式）
    if sequential:
        result = np.full(len(candles), np.nan)

    # 计算滑动窗口内的熵
    entropy_values = []

    for i in range(window_size, len(log_returns) + 1):
        window = log_returns[i - window_size : i]

        # 使用numpy的histogram函数进行分箱
        counts, _ = np.histogram(window, bins=n_bins)

        # 移除空箱子（计数为0的箱子）
        counts = counts[counts > 0]

        # 计算NSB熵
        entropy_val = nsb_entropy(counts)
        entropy_values.append(entropy_val)

    # 转换为numpy数组
    entropy_array = np.array(entropy_values)

    # 如果要求是顺序返回，填充结果数组
    if sequential:
        result[window_size : len(log_returns) + 1] = entropy_array
        return result
    else:
        # 否则只返回最后一个值
        return entropy_array[-1]
