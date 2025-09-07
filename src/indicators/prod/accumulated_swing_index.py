import numpy as np
import numpy.typing as npt
from jesse.helpers import get_candle_source


def accumulated_swing_index(
    candles: npt.NDArray,
    period: int = 50,
    sequential=False,
):
    """计算Wilder的累积摆动指数(Accumulated Swing Index)

    参数:
        candles: n*6的numpy数组, 包含timestamp, open, close, high, low, volume

    返回:
        累积摆动指数数组
    """
    # 提取价格数据
    o = get_candle_source(candles, "open")
    c = get_candle_source(candles, "close")
    h = get_candle_source(candles, "high")
    l = get_candle_source(candles, "low")

    # 计算中间变量
    prev_close = np.roll(c, 1)
    prev_open = np.roll(o, 1)

    # 计算条件值
    conda = np.abs(h - prev_close)
    condb = np.abs(l - prev_close)
    condc = h - l
    pdir = np.abs(prev_close - prev_open)

    # 计算maxk
    maxk = np.maximum(conda, condb)

    # 计算R值
    r = np.zeros_like(c)

    # 条件1: conda >= condb and conda >= condc
    mask1 = (conda >= condb) & (conda >= condc)
    r[mask1] = conda[mask1] - 0.5 * condb[mask1] + 0.25 * pdir[mask1]

    # 条件2: condb >= conda and condb >= condc
    mask2 = (condb >= conda) & (condb >= condc)
    r[mask2] = condb[mask2] - 0.5 * conda[mask2] + 0.25 * pdir[mask2]

    # 条件3: condc >= conda and condc >= condb
    mask3 = (condc >= conda) & (condc >= condb)
    r[mask3] = condc[mask3] + 0.25 * pdir[mask3]

    # 计算limit move
    lim = 3.5 * np.array([np.std(r[max(0, i - period) : i + 1]) for i in range(len(r))])

    # 计算swing index
    swing_index = np.zeros_like(c)
    valid = (r != 0) & (lim != 0)
    swing_index[valid] = (
        50
        * (maxk[valid] / lim[valid])
        * (
            (c[valid] - prev_close[valid])
            + 0.5 * (c[valid] - o[valid])
            + 0.25 * pdir[valid]
        )
        / r[valid]
    )

    # 计算累积摆动指数
    acc_swing_index = np.cumsum(swing_index)

    # 处理首行
    acc_swing_index[0] = 0

    if sequential:
        return acc_swing_index
    else:
        return acc_swing_index[-1]
