import numpy as np
import numpy.typing as npt
from jesse.helpers import get_candle_source
from numba import njit


@njit
def _calculate_asi_window(
    o: np.ndarray,
    c: np.ndarray, 
    h: np.ndarray,
    l: np.ndarray,
    period: int,
    window_size: int
) -> np.ndarray:
    """使用numba优化的滚动窗口ASI计算"""
    n = len(c)
    acc_swing_index = np.full(n, np.nan)
    
    for i in range(n):
        # 确定窗口范围
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window_len = end_idx - start_idx
        
        if window_len < 2:  # 至少需要2个数据点
            continue
        
        # 创建窗口数据
        window_o = o[start_idx:end_idx]
        window_c = c[start_idx:end_idx]
        window_h = h[start_idx:end_idx]
        window_l = l[start_idx:end_idx]
        
        # 计算R值和swing index
        r = np.zeros(window_len)
        swing_index = np.zeros(window_len)
        
        for j in range(1, window_len):  # 从1开始，因为需要prev值
            # 获取当前和前一个值
            curr_c = window_c[j]
            curr_o = window_o[j]
            curr_h = window_h[j]
            curr_l = window_l[j]
            prev_c = window_c[j-1]
            prev_o = window_o[j-1]
            
            # 计算条件值
            conda = abs(curr_h - prev_c)
            condb = abs(curr_l - prev_c)
            condc = curr_h - curr_l
            pdir = abs(prev_c - prev_o)
            
            # 计算maxk
            maxk = max(conda, condb)
            
            # 计算R值
            if conda >= condb and conda >= condc:
                r[j] = conda - 0.5 * condb + 0.25 * pdir
            elif condb >= conda and condb >= condc:
                r[j] = condb - 0.5 * conda + 0.25 * pdir
            elif condc >= conda and condc >= condb:
                r[j] = condc + 0.25 * pdir
            
            # 计算limit move
            period_start = max(0, j - period + 1)
            if j >= period_start and j > 0:
                # 计算标准差
                r_subset = r[period_start:j+1]
                if len(r_subset) > 0:
                    mean_r = np.mean(r_subset)
                    std_r = 0.0
                    for k in range(len(r_subset)):
                        std_r += (r_subset[k] - mean_r) ** 2
                    if len(r_subset) > 1:
                        std_r = np.sqrt(std_r / (len(r_subset) - 1))
                    else:
                        std_r = 0.0
                    lim = 3.5 * std_r
                else:
                    lim = 0.0
            else:
                lim = 0.0
            
            # 计算swing index
            if r[j] != 0 and lim != 0:
                swing_index[j] = (
                    50 * (maxk / lim) * 
                    ((curr_c - prev_c) + 0.5 * (curr_c - curr_o) + 0.25 * pdir) / r[j]
                )
        
        # 累积求和
        cumsum = 0.0
        for j in range(window_len):
            cumsum += swing_index[j]
        acc_swing_index[i] = cumsum
    
    # 处理首行
    if not np.isnan(acc_swing_index[0]):
        acc_swing_index[0] = 0
    
    return acc_swing_index


def accumulated_swing_index(
    candles: npt.NDArray,
    period: int = 50,
    sequential=False,
    window_size: int = 240,
):
    """计算Wilder的累积摆动指数(Accumulated Swing Index)

    参数:
        candles: n*6的numpy数组, 包含timestamp, open, close, high, low, volume
        period: 标准差计算周期
        sequential: 是否返回序列
        window_size: 滚动窗口大小，保证线上线下计算一致性

    返回:
        累积摆动指数数组
    """
    # 提取价格数据
    o = get_candle_source(candles, "open")
    c = get_candle_source(candles, "close")
    h = get_candle_source(candles, "high")
    l = get_candle_source(candles, "low")
    
    # 调用numba优化的计算函数
    acc_swing_index = _calculate_asi_window(o, c, h, l, period, window_size)

    if sequential:
        return acc_swing_index
    else:
        return acc_swing_index[-1:]
