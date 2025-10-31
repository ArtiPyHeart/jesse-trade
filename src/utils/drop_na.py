from typing import Union

import numpy as np
import pandas as pd


def drop_na_and_align_x_and_y(
    x: pd.DataFrame, y: Union[pd.Series, np.ndarray]
) -> tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
    """
    去除x开头的空值，保证x中不存在空值，并对齐输入的x与y的行数
    Args:
        x: 特征集合，pandas DataFrame形式
        y: 分类或回归的标签，np.ndarray形式或者pandas Series形式

    Returns:
        对齐后的x与y
    """
    # 1. 检查x与y是否长度有差异，有差异先对齐
    len_gap = len(y) - len(x)
    if len_gap > 0:
        y = y[len_gap:]
    elif len_gap < 0:
        x = x.iloc[abs(len_gap) :]
    assert len(x) == len(
        y
    ), "drop_na_and_align_x_and_y: x and y length mismatch after len_gap check"

    # 2. 检查x开头是否有NaN
    max_na_len = x.isna().sum().max()
    if max_na_len > 0:
        x = x.iloc[max_na_len:]
        y = y[max_na_len:]
    assert len(x) == len(
        y
    ), "drop_na_and_align_x_and_y: x and y length mismatch after max_na_len check"
    return x, y
