import math

import numpy as np
from jesse.helpers import get_candle_source
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _apply_filters_window(src: np.ndarray, alpha1: float, a1: float, b1: float, window_size: int) -> np.ndarray:
    """使用滑动窗口计算高通滤波和Super Smoother"""
    length = len(src)
    result = np.full(length, np.nan)
    
    # Super Smoother系数
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # 对每个位置使用固定窗口计算
    for end_idx in range(length):
        # 确定窗口起始位置
        start_idx = max(0, end_idx - window_size + 1)
        window_data = src[start_idx:end_idx + 1]
        window_len = len(window_data)
        
        if window_len < 3:  # 至少需要3个数据点
            continue
            
        HP = np.zeros(window_len)
        Filt = np.zeros(window_len)
        
        for i in range(window_len):
            # 高通滤波
            if i >= 2:
                HP[i] = (
                    ((1 - alpha1 / 2.0) ** 2) * (window_data[i] - 2 * window_data[i - 1] + window_data[i - 2])
                    + 2 * (1 - alpha1) * HP[i - 1]
                    - (1 - alpha1) ** 2 * HP[i - 2]
                )
            elif i == 1:
                HP[i] = window_data[i] - window_data[i - 1]
            else:
                HP[i] = window_data[i]

            # Super Smoother
            if i >= 2:
                Filt[i] = (
                    c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
                )
            elif i == 1:
                Filt[i] = (HP[i] + HP[i - 1]) / 2.0
            else:
                Filt[i] = HP[i]
        
        result[end_idx] = Filt[-1]
    
    return result


@njit
def _calculate_correlation_window(Filt_window: np.ndarray, lag: int, M: int) -> float:
    """在窗口内计算单个lag的皮尔逊相关系数"""
    window_len = len(Filt_window)
    if M <= 0 or window_len < (lag + M):
        return 0.5

    Sx = Sy = Sxx = Syy = Sxy = 0.0
    
    # 从窗口末尾开始计算
    for count in range(M):
        idx = window_len - 1 - count
        if idx < 0 or idx - lag < 0:
            break
            
        X = Filt_window[idx]
        Y = Filt_window[idx - lag]

        Sx += X
        Sy += Y
        Sxx += X * X
        Syy += Y * Y
        Sxy += X * Y

    actual_M = min(M, window_len - lag)
    if actual_M <= 0:
        return 0.5
        
    denom = (actual_M * Sxx - Sx * Sx) * (actual_M * Syy - Sy * Sy)
    if denom > 0:
        corr = (actual_M * Sxy - Sx * Sy) / np.sqrt(denom)
        # 归一化到[0,1]
        return 0.5 * (corr + 1)
    return 0.5


@njit
def _calculate_all_correlations_window(
    src: np.ndarray, alpha1: float, a1: float, b1: float, 
    window_size: int, max_lag: int, AvgLength: int
) -> np.ndarray:
    """使用滑动窗口计算所有时间点的所有lag的相关系数"""
    length = len(src)
    Corr_all = np.full((length, max_lag + 1), np.nan)
    
    # Super Smoother系数
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    for end_idx in range(length):
        # 确定窗口起始位置
        start_idx = max(0, end_idx - window_size + 1)
        window_data = src[start_idx:end_idx + 1]
        window_len = len(window_data)
        
        if window_len < 3:  # 至少需要3个数据点
            continue
            
        # 计算窗口内的滤波值
        HP = np.zeros(window_len)
        Filt = np.zeros(window_len)
        
        for i in range(window_len):
            # 高通滤波
            if i >= 2:
                HP[i] = (
                    ((1 - alpha1 / 2.0) ** 2) * (window_data[i] - 2 * window_data[i - 1] + window_data[i - 2])
                    + 2 * (1 - alpha1) * HP[i - 1]
                    - (1 - alpha1) ** 2 * HP[i - 2]
                )
            elif i == 1:
                HP[i] = window_data[i] - window_data[i - 1]
            else:
                HP[i] = window_data[i]

            # Super Smoother
            if i >= 2:
                Filt[i] = (
                    c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
                )
            elif i == 1:
                Filt[i] = (HP[i] + HP[i - 1]) / 2.0
            else:
                Filt[i] = HP[i]
        
        # 计算所有lag的相关系数
        for lag in range(max_lag + 1):
            M = AvgLength if AvgLength > 0 else lag
            Corr_all[end_idx, lag] = _calculate_correlation_window(Filt, lag, M)

    return Corr_all


def autocorrelation(
    candles: np.ndarray,
    source_type: str = "close",
    avg_length: int = 128,
    sequential: bool = False,
    window_size: int = 240,
) -> np.ndarray:
    """主函数：计算自相关指标
    
    Args:
        candles: K线数据
        source_type: 价格类型
        avg_length: 平均长度
        sequential: 是否返回序列
        window_size: 滑动窗口大小，保证线上线下计算一致性
    """
    # 获取源数据
    src = get_candle_source(candles, source_type)

    # 使用 EasyLanguage 角度制计算方式
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )
    a1 = np.exp(-1.414 * math.pi / 10)
    b1 = 2.0 * a1 * deg_cos(1.414 * 180 / 10)

    # 使用滑动窗口计算相关系数
    max_lag = 48
    Corr_all = _calculate_all_correlations_window(
        src, alpha1, a1, b1, window_size, max_lag, avg_length
    )

    if sequential:
        return Corr_all[:, 2:]
    else:
        return Corr_all[-1:, 2:]
