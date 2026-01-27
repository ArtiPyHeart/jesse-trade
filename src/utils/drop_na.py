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

    def _nan_counts(
        values: np.ndarray, max_bytes: int = 64 * 1024 * 1024
    ) -> np.ndarray:
        if values.size == 0:
            return np.zeros(values.shape[1] if values.ndim == 2 else 0, dtype=np.int64)
        n_rows, n_cols = values.shape
        bytes_per_row = n_cols  # bool array uses 1 byte per entry
        chunk_size = max(1, min(n_rows, max_bytes // bytes_per_row))
        counts = np.zeros(n_cols, dtype=np.int64)
        for start in range(0, n_rows, chunk_size):
            block = values[start : start + chunk_size]
            try:
                counts += np.isnan(block).sum(axis=0)
            except TypeError:
                counts += pd.isna(block).sum(axis=0)
        return counts

    # 1. 检查x与y是否长度有差异，有差异先对齐
    len_gap = len(y) - len(x)
    if len_gap > 0:
        y = y[len_gap:]
    elif len_gap < 0:
        x = x.iloc[abs(len_gap) :]
    assert len(x) == len(y), (
        "drop_na_and_align_x_and_y: x and y length mismatch after len_gap check"
    )

    # 2. 检查x开头是否有NaN
    x_values = x.to_numpy(copy=False)
    counts = _nan_counts(x_values)
    max_na_len = int(counts.max()) if counts.size > 0 else 0
    if max_na_len > 0:
        x = x.iloc[max_na_len:]
        y = y[max_na_len:]
    assert len(x) == len(y), (
        "drop_na_and_align_x_and_y: x and y length mismatch after max_na_len check"
    )

    # 3. 最终检查：确保x中不再存在任何NaN值
    remaining_counts = _nan_counts(x.to_numpy(copy=False))
    columns_with_na = [
        (col_name, int(count))
        for col_name, count in zip(x.columns, remaining_counts, strict=False)
        if count > 0
    ]
    if len(columns_with_na) > 0:
        error_msg = (
            "drop_na_and_align_x_and_y: x still contains NaN values after alignment:\n"
        )
        for col_name, na_count in columns_with_na:
            error_msg += f"  - Column '{col_name}': {na_count} NaN values\n"
        raise ValueError(error_msg.strip())

    return x, y
