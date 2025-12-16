"""
特征工具函数，支持 FeaturePipeline 集成

提供特征配置构建、特征-标签对齐、特征筛选等功能。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.features.dimensionality_reduction import ARDVAEConfig
from src.features.feature_selection.rf_importance_selector import RFImportanceSelector
from src.features.pipeline import PipelineConfig


@dataclass
class FeatureSelectionResult:
    """特征筛选结果"""

    selected_features: List[str]  # 含 SSM 特征名
    n_total: int
    n_selected: int


def build_full_feature_config(
    raw_feature_names: List[str],
    ssm_state_dim: int = 5,
) -> PipelineConfig:
    """
    构建全量特征配置（不降维）

    Args:
        raw_feature_names: 原始特征名称列表
        ssm_state_dim: SSM 输出维度

    Returns:
        PipelineConfig 配置对象
    """
    ssm_features = [
        f"{ssm_type}_{i}"
        for ssm_type in ["deep_ssm", "lg_ssm"]
        for i in range(ssm_state_dim)
    ]
    return PipelineConfig(
        feature_names=ssm_features + list(raw_feature_names),
        ssm_state_dim=ssm_state_dim,
        use_dimension_reducer=False,
        verbose=True,
    )


def build_model_config(
    selected_features: List[str],
    ssm_state_dim: int = 5,
    reducer_config: Optional[ARDVAEConfig] = None,
) -> PipelineConfig:
    """
    构建模型特定配置（启用降维）

    Args:
        selected_features: 筛选后的特征名称列表
        ssm_state_dim: SSM 输出维度
        reducer_config: 降维器配置（ARDVAEConfig 实例），None 时使用默认配置

    Returns:
        PipelineConfig 配置对象
    """
    if reducer_config is None:
        # ARDVAE 默认配置（参考 codex 建议）：
        # - max_latent_dim=512：over-complete 设计，ARD prior 自动确定 active dims
        # - kl_threshold=0.01：判断维度是否 active 的阈值
        # - patience=15：早停耐心，避免过拟合
        # - 如果 active dims 经常逼近 512，考虑增加到 1024
        reducer_config = ARDVAEConfig(
            max_latent_dim=512,  # 对 1000-10000 特征通常足够
            kl_threshold=0.01,  # 合理起点，可微调
            max_epochs=200,  # 默认值
            patience=15,  # 早停
            seed=42,
        )
    return PipelineConfig(
        feature_names=selected_features,
        ssm_state_dim=ssm_state_dim,
        use_dimension_reducer=True,
        dimension_reducer_type="ard_vae",
        dimension_reducer_config=reducer_config,
        verbose=True,
    )


def align_features_and_labels(
    features_df: pd.DataFrame,
    label: np.ndarray,
    pred_next: int,
    timestamps: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    对齐特征和标签

    四步清晰逻辑：
    0. 如果 label 长度 < features 长度，在 label 开头 pad NaN 使长度一致
    1. 检查 features 和 label 长度相同（N 行）
    2. 按 pred_next 进行 shift：label[pred_next:] 和 feature[:-pred_next]
    3. 去掉开头的 NaN 行（feature 和 label 同时去掉）

    Args:
        features_df: N 行特征（FeaturePipeline 输出，开头可能有 NaN）
        label: 可能是 N 行或 N - lag_n 行（GMMLabeler 输出）
        pred_next: 预测未来第几个标签（如 pred_next=2 表示预测未来第2个）
        timestamps: N 行时间戳

    Returns:
        (aligned_features, aligned_labels): 完全对齐且无 NaN 的数据
    """
    N = len(features_df)

    # Step 0: 如果 label 长度 < N，在开头 pad NaN
    if len(label) < N:
        pad_len = N - len(label)
        label = np.concatenate([np.full(pad_len, np.nan), label])

    # Step 1: 检查长度一致
    assert len(label) == N, f"Pad 后长度不一致: features={N}, label={len(label)}"

    # Step 2: 按 pred_next 进行 shift
    # feature[:-pred_next] 对应 label[pred_next:]
    # 即用当前特征预测未来第 pred_next 个标签
    shifted_features = features_df.iloc[:-pred_next].copy()
    shifted_labels = label[pred_next:]
    shifted_timestamps = timestamps[:-pred_next]

    assert len(shifted_features) == len(shifted_labels)

    # Step 3: 去掉开头的 NaN 行（feature 或 label 中任一有 NaN）
    feature_nan_mask = shifted_features.isna().any(axis=1).values
    label_nan_mask = np.isnan(shifted_labels)
    combined_nan_mask = feature_nan_mask | label_nan_mask

    # 找到第一个非 NaN 行
    first_valid_idx = (
        (~combined_nan_mask).argmax()
        if (~combined_nan_mask).any()
        else len(combined_nan_mask)
    )
    M = first_valid_idx

    aligned_features = shifted_features.iloc[M:].reset_index(drop=True)
    aligned_labels = shifted_labels[M:]

    # 设置时间戳索引
    aligned_features.index = shifted_timestamps[M:].astype(int)

    # 最终验证：无 NaN
    assert not aligned_features.isna().any().any(), "对齐后 features 仍有 NaN"
    assert not np.isnan(aligned_labels).any(), "对齐后 labels 仍有 NaN"
    assert len(aligned_features) == len(aligned_labels)

    return aligned_features, aligned_labels


def select_features(
    features_df: pd.DataFrame,
    labels: np.ndarray,
) -> FeatureSelectionResult:
    """
    特征筛选（返回含 SSM 特征名）

    Args:
        features_df: 对齐后的特征 DataFrame
        labels: 对齐后的标签数组

    Returns:
        FeatureSelectionResult 包含筛选后的特征名称
    """
    selector = RFImportanceSelector(verbose=True)
    selector.fit(features_df, labels)

    importance = pd.Series(selector.relevance_, index=selector.variables_).sort_values(
        ascending=False
    )

    selected = importance[importance > 0].index.tolist()

    return FeatureSelectionResult(
        selected_features=selected,
        n_total=len(selector.variables_),
        n_selected=len(selected),
    )
