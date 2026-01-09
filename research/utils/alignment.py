"""
Feature-Label alignment utilities

Provides functions to align features and labels for model training.
"""

import numpy as np
import pandas as pd


def align_features_labels(
    features: pd.DataFrame,
    labels: np.ndarray,
    log_return_lag: int,
    pred_next: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    对齐特征和标签

    在训练预测模型时，需要处理以下对齐问题：
    1. 标签可能基于未来数据计算（需要 shift）
    2. 标签可能有初始化延迟（log_return_lag）
    3. 特征可能有 NaN 值需要处理

    Args:
        features: 特征 DataFrame，索引与原始 candles 对齐
        labels: 标签数组，长度 = len(candles) - log_return_lag
        log_return_lag: 标签计算所需的历史数据量
            例如：GMMLabeler 的 lag_n 参数
        pred_next: 预测时间范围
            例如：用当前特征预测 pred_next 步后的标签

    Returns:
        tuple[pd.DataFrame, np.ndarray]: 对齐后的 (features, labels)

    Example:
        >>> labeler = GMMLabeler(candles, lag_n=5)
        >>> labels = labeler.label_hard_state  # len = len(candles) - 5
        >>> features = calc.get(feature_names)  # len = len(candles)
        >>> features, labels = align_features_labels(
        ...     features, labels, log_return_lag=5, pred_next=3
        ... )
    """
    # 1. 截断 feature 开头，对齐 label 的 lag
    aligned_features = features.iloc[log_return_lag:]

    # 2. 按 pred_next 进行 shift 对齐
    #    features[i] 对应 labels[i + pred_next]
    aligned_features = aligned_features.iloc[:-pred_next]
    aligned_labels = labels[pred_next:]

    # 3. 去掉 feature 中的 NaN 行
    na_mask = aligned_features.isna().any(axis=1).values
    aligned_features = aligned_features.iloc[~na_mask]
    aligned_labels = aligned_labels[~na_mask]

    assert len(aligned_features) == len(aligned_labels), (
        f"Length mismatch after alignment: "
        f"features={len(aligned_features)}, labels={len(aligned_labels)}"
    )

    return aligned_features, aligned_labels
