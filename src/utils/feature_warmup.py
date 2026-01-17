from __future__ import annotations

import numpy as np
import pandas as pd


def _build_nan_diagnostics(
    features_df: pd.DataFrame, nan_mask: np.ndarray, bad_rows: np.ndarray
) -> tuple[list[int], list[str], dict[int, list[str]], int]:
    rows_with_nan = [int(i) for i in bad_rows[:10]]
    cols_with_nan = features_df.columns[np.any(nan_mask[bad_rows], axis=0)].tolist()
    sample = {}
    for idx in rows_with_nan[:5]:
        row_mask = nan_mask[idx]
        sample[idx] = list(features_df.columns[row_mask])[:10]
    total_nan = int(nan_mask[bad_rows].sum())
    return rows_with_nan, cols_with_nan, sample, total_nan


def determine_warmup_start_idx(
    features_df: pd.DataFrame,
    min_start_idx: int,
    context: str = "Feature dataframe",
) -> tuple[int, int]:
    """
    允许开头 NaN，直到出现首个全量特征有效行。
    若 warmup 之后仍有 NaN，直接报错。

    Returns:
        trade_start_idx: 实际开始交易的索引（>= min_start_idx）
        first_valid_idx: 首个全量特征有效的索引
    """
    if features_df.empty:
        raise ValueError(f"{context} is empty; cannot determine warmup start")

    nan_mask = features_df.isna().to_numpy()
    row_has_nan = nan_mask.any(axis=1)
    valid_rows = np.flatnonzero(~row_has_nan)
    if len(valid_rows) == 0:
        raise ValueError(f"{context} has no fully valid rows (all NaN)")

    first_valid_idx = int(valid_rows[0])
    if row_has_nan[first_valid_idx:].any():
        bad_rows = np.flatnonzero(row_has_nan[first_valid_idx:]) + first_valid_idx
        rows_with_nan, cols_with_nan, sample, total_nan = _build_nan_diagnostics(
            features_df, nan_mask, bad_rows
        )
        raise ValueError(
            f"{context} contains NaN after warmup. total_nan={total_nan}, "
            f"first_valid_idx={first_valid_idx}, rows_with_nan={rows_with_nan}, "
            f"cols_with_nan={cols_with_nan[:10]}, sample={sample}"
        )

    trade_start_idx = max(min_start_idx, first_valid_idx)
    return trade_start_idx, first_valid_idx
