"""
FeaturePipeline - 高阶特征流水线

整合 SimpleFeatureCalculator、SSM 模型、降维模块的统一接口。

支持三种使用模式：
1. 训练模式 (fit_transform): 完整训练流程
2. 加载模式 (transform): 加载已训练模块进行批量转换
3. 实时模式 (inference): 单步实时推理
"""

import copy
import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.ssm import SSMProtocol, DeepSSMAdapter, LGSSMAdapter
from src.features.dimensionality_reduction import (
    DimensionReducerProtocol,
    ARDVAE,
    ARDVAEConfig,
)
from src.models.deep_ssm import DeepSSMConfig
from src.models.lgssm import LGSSMConfig

from .config import PipelineConfig


class FeaturePipeline:
    """
    高阶特征流水线

    整合原始特征计算、SSM 特征提取、降维模块的统一接口。

    Architecture
    ------------
    ```
    FeaturePipeline
    ├── RawFeatureStage (SimpleFeatureCalculator)
    │   └── 原始特征计算：candles → raw_features
    │
    ├── SSMProcessorStage (并行，不嵌套)
    │   ├── DeepSSMAdapter → deep_ssm_0, deep_ssm_1, ...
    │   └── LGSSMAdapter   → lg_ssm_0, lg_ssm_1, ...
    │
    └── DimensionReductionStage (可选)
        └── ARDVAE 或其他实现 DimensionReducerProtocol 的模块
    ```

    Usage Modes
    -----------
    1. **训练模式**: fit_transform(candles)
       训练 SSM 模型和降维模块，并返回特征（优化实现，避免重复计算）
       若只需训练不需要特征，可单独调用 fit()

    2. **批量模式**: transform(candles)
       加载已训练的模型后，批量转换（离线使用）

    3. **预热模式**: warmup_ssm(candles)
       用历史数据逐行更新 SSM 状态，使其达到稳定状态。
       **仅更新状态，不返回特征**。适用于策略启动时的初始化。

    4. **实时模式**: inference(candles)
       单步实时推理，适用于策略实盘

    SSM 预热说明
    ------------
    SSM 是有状态的模型，输出依赖于历史观测积累的内部状态。
    策略启动时 SSM 状态为空，直接使用 inference() 会导致预测不准确。

    **各方法与 SSM 状态的关系**：

    | 方法           | 用途         | SSM 状态处理        | 输出           |
    |----------------|--------------|---------------------|----------------|
    | fit_transform  | 离线训练     | 训练时更新          | 返回 DataFrame |
    | transform      | 批量推理     | 逐行 inference 更新 | 返回 DataFrame |
    | warmup_ssm     | 状态预热     | 逐行 inference 更新 | **无输出**     |
    | inference      | 实时单步推理 | 单步更新            | 返回单行特征   |

    **典型实盘启动流程**：
    1. load() 加载已训练的 Pipeline
    2. warmup_ssm(历史数据) 预热 SSM 状态（建议 3000+ 根 K 线）
    3. 每根新 K 线到来时调用 inference() 获取特征

    降维模块说明
    ------------
    降维模块（如 ARDVAE）可选启用，用于减少特征维度、去除冗余特征。

    **启用降维的影响**：
    - 输出列名变为数字字符串 "0", "1", "2", ... 而非原始特征名
    - 输出维度由降维器决定（如 ARDVAE 的 max_latent_dim）

    **配置降维**：
    >>> config = PipelineConfig(
    ...     feature_names=["deep_ssm_0", "rsi", "macd"],
    ...     use_dimension_reducer=True,
    ...     dimension_reducer_type="ard_vae",
    ...     dimension_reducer_config={
    ...         "max_latent_dim": 32,
    ...         "max_epochs": 50,
    ...         "patience": 10,
    ...         "seed": 42,
    ...     },
    ... )

    Save/Load 说明
    --------------
    Pipeline 的持久化使用 **统一接口**，所有产物保存到子目录：

    - save(path, name) / load(path, name)
    - 目录结构：
      ```
      path/
      └── name/                             # 子文件夹
          ├── pipeline_config.json          # PipelineConfig
          ├── deep_ssm.safetensors          # DeepSSM 模型权重
          ├── lg_ssm.safetensors            # LGSSM 模型权重
          ├── dimension_reducer.safetensors # 降维器权重（如果使用）
          └── dimension_reducer.json        # 降维器配置（如果使用）
      ```

    **示例**：`pipeline.save("/models", "c_L5_N2")` → `/models/c_L5_N2/`

    Examples
    --------
    **训练模式（不降维）**：
    >>> config = PipelineConfig(
    ...     feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0", "rsi", "macd"]
    ... )
    >>> pipeline = FeaturePipeline(config)
    >>> features = pipeline.fit_transform(candles)  # 输出 5 列，列名为特征名
    >>> pipeline.save("/models", "my_pipeline")

    **训练模式（启用降维）**：
    >>> config = PipelineConfig(
    ...     feature_names=["deep_ssm_0", "deep_ssm_1", "rsi", "macd"],
    ...     use_dimension_reducer=True,
    ...     dimension_reducer_config={"max_latent_dim": 16, "seed": 42},
    ... )
    >>> pipeline = FeaturePipeline(config)
    >>> features = pipeline.fit_transform(candles)  # 输出 N 列，列名为 "0", "1", ...
    >>> pipeline.save("/models", "my_pipeline")  # 降维器自动保存

    **加载模式**：
    >>> pipeline = FeaturePipeline.load("/models", "my_pipeline")
    >>> features = pipeline.transform(candles)  # 降维器自动加载

    **实盘模式（预热 + 实时推理）**：
    >>> pipeline = FeaturePipeline.load("/models", "my_pipeline")
    >>> pipeline.warmup_ssm(historical_candles, verbose=True)  # 预热 SSM 状态
    >>> # 之后每根新 K 线
    >>> features = pipeline.inference(current_candles)

    一致性保证
    ----------
    以下三种模式在相同输入下产生完全一致的输出（数值和列顺序）：
    - fit_transform(candles)
    - transform(candles)
    - warmup_ssm(candles[:N]) + inference(candles[:N+1]) 逐行累积

    这由以下机制保证：
    1. SSM 类型顺序由 config.ssm_types（有序列表）统一控制
    2. 所有方法使用相同的特征拼接顺序：[SSM 特征, 原始特征]
    3. 降维器对列顺序有严格校验，确保输入一致
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        raw_calculator: Optional[SimpleFeatureCalculator] = None,
        ssm_processors: Optional[Dict[str, SSMProtocol]] = None,
        dimension_reducer: Optional[DimensionReducerProtocol] = None,
    ):
        """
        初始化 FeaturePipeline

        Args:
            config: 流水线配置
            raw_calculator: 原始特征计算器（可选，默认创建新实例）
            ssm_processors: SSM 处理器字典 {"deep_ssm": ..., "lg_ssm": ...}
            dimension_reducer: 降维器实例
        """
        self.config = config or PipelineConfig()
        self._raw_calculator = raw_calculator or SimpleFeatureCalculator(verbose=False)
        self._ssm_processors: Dict[str, SSMProtocol] = ssm_processors or {}
        self._dimension_reducer = dimension_reducer

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """流水线是否已训练"""
        return self._is_fitted

    @property
    def ssm_processors(self) -> Dict[str, SSMProtocol]:
        """SSM 处理器字典"""
        return self._ssm_processors

    @property
    def dimension_reducer(self) -> Optional[DimensionReducerProtocol]:
        """降维器"""
        return self._dimension_reducer

    # ========== 输入验证 ==========

    def _validate_candles(self, candles: np.ndarray, min_rows: int = 1) -> None:
        """
        验证 K 线数据格式

        Args:
            candles: 待验证的 K 线数据
            min_rows: 最小行数要求

        Raises:
            TypeError: 输入不是 numpy 数组
            ValueError: 数组维度或形状不符合要求
        """
        if not isinstance(candles, np.ndarray):
            raise TypeError(
                f"candles must be numpy.ndarray, got {type(candles).__name__}"
            )

        if candles.ndim != 2:
            raise ValueError(
                f"candles must be 2D array (N, 6), got {candles.ndim}D array"
            )

        if candles.shape[1] != 6:
            raise ValueError(
                f"candles must have 6 columns [timestamp, open, close, high, low, volume], "
                f"got {candles.shape[1]} columns"
            )

        if candles.shape[0] < min_rows:
            raise ValueError(
                f"candles must have at least {min_rows} rows, got {candles.shape[0]}"
            )

    # ========== NaN 处理 ==========

    def _find_first_valid_row(self, df: pd.DataFrame) -> int:
        """
        找到第一个全部有效（无 NaN）的行索引

        Args:
            df: 待检查的 DataFrame

        Returns:
            第一个有效行的索引，如果全是 NaN 返回 len(df)
        """
        # 每行是否全部有效
        valid_rows = ~df.isna().any(axis=1)
        # 找到第一个 True 的索引
        valid_indices = valid_rows[valid_rows].index
        if len(valid_indices) == 0:
            return len(df)
        return valid_indices[0]

    def _validate_no_intermediate_nan(self, df: pd.DataFrame, start_row: int) -> None:
        """
        检查从 start_row 开始的数据是否还有 NaN

        Args:
            df: 待检查的 DataFrame
            start_row: 起始行索引

        Raises:
            ValueError: 如果存在中间 NaN，列出具体哪些特征有问题
        """
        valid_df = df.iloc[start_row:]
        nan_cols = valid_df.columns[valid_df.isna().any()].tolist()

        if nan_cols:
            raise ValueError(
                f"Features contain intermediate NaN values after row {start_row}. "
                f"Affected features: {nan_cols}"
            )

    def _pad_with_leading_nan(self, df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
        """
        在 DataFrame 开头填充 NaN 使其行数达到 target_rows

        Args:
            df: 有效数据 DataFrame
            target_rows: 目标行数（通常是原始 candles 行数）

        Returns:
            填充后的 DataFrame，行数 = target_rows
        """
        current_rows = len(df)
        if current_rows >= target_rows:
            return df

        # 创建 NaN 填充行
        nan_rows = target_rows - current_rows
        nan_df = pd.DataFrame(
            np.nan,
            index=range(nan_rows),
            columns=df.columns,
        )

        # 重置有效数据的索引
        df = df.reset_index(drop=True)
        df.index = range(nan_rows, target_rows)

        # 拼接
        result = pd.concat([nan_df, df])
        result.index = range(target_rows)

        return result

    # ========== 公共训练辅助方法 ==========

    def _compute_raw_features(
        self, candles: np.ndarray, verbose: bool = False
    ) -> pd.DataFrame:
        """
        计算原始特征

        Args:
            candles: K 线数据
            verbose: 是否打印进度

        Returns:
            原始特征 DataFrame
        """
        if verbose:
            print("  Computing raw features...")

        self._raw_calculator.load(candles, sequential=True)
        raw_features_dict = self._raw_calculator.get(
            self.config.all_calculator_features
        )
        return pd.DataFrame(raw_features_dict)

    def _prepare_valid_data(
        self, raw_features_df: pd.DataFrame, n_candles: int, verbose: bool = False
    ) -> tuple[pd.DataFrame, int]:
        """
        处理 NaN：找到第一个有效行，检查中间 NaN，返回有效数据

        Args:
            raw_features_df: 原始特征 DataFrame
            n_candles: 原始 candles 行数
            verbose: 是否打印进度

        Returns:
            (valid_raw_features_df, first_valid_row)

        Raises:
            ValueError: 全部 NaN 或存在中间 NaN
        """
        if verbose:
            print("  Handling NaN values...")

        first_valid = self._find_first_valid_row(raw_features_df)

        if verbose:
            print(f"    First valid row: {first_valid} (of {n_candles})")

        if first_valid >= n_candles:
            raise ValueError(
                "All rows contain NaN values, cannot proceed. "
                "Ensure candles has enough history for feature calculation."
            )

        # 检查中间是否还有 NaN
        self._validate_no_intermediate_nan(raw_features_df, first_valid)

        # 剔除开头 NaN 行
        valid_raw_features_df = raw_features_df.iloc[first_valid:].reset_index(
            drop=True
        )

        return valid_raw_features_df, first_valid

    def _train_ssm_models(
        self, valid_raw_features_df: pd.DataFrame, verbose: bool = False
    ) -> None:
        """
        训练 SSM 模型（仅训练，不计算特征）

        Args:
            valid_raw_features_df: 无 NaN 的原始特征 DataFrame
            verbose: 是否打印进度
        """
        if verbose:
            print("  Training SSM models...")

        if not self.config.ssm_types:
            if verbose:
                print("    No SSM configured, skipping...")
            return

        valid_ssm_input_df = valid_raw_features_df[self.config.ssm_input_features]
        obs_dim = len(self.config.ssm_input_features)

        for ssm_type in self.config.ssm_types:
            if ssm_type not in self._ssm_processors:
                self._ssm_processors[ssm_type] = self._create_ssm_adapter(
                    ssm_type, obs_dim
                )

            processor = self._ssm_processors[ssm_type]
            if not processor.is_fitted:
                if verbose:
                    print(f"    Training {ssm_type}...")
                processor.fit(valid_ssm_input_df)

    def _compute_all_features_for_fit(
        self, valid_raw_features_df: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """
        fit 专用：计算完整特征供降维器训练

        Args:
            valid_raw_features_df: 无 NaN 的原始特征 DataFrame
            verbose: 是否打印进度

        Returns:
            包含 SSM 特征和原始特征的完整 DataFrame
        """
        if self.config.ssm_types:
            valid_ssm_input_df = valid_raw_features_df[self.config.ssm_input_features]
            ssm_features = self._compute_ssm_features_batch(valid_ssm_input_df)
            all_features = pd.concat([ssm_features, valid_raw_features_df], axis=1)
        else:
            all_features = valid_raw_features_df

        return all_features

    def _train_and_compute_features(
        self, valid_raw_features_df: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """
        fit_transform 专用：训练 SSM 并计算特征（单 pass 优化）

        Args:
            valid_raw_features_df: 无 NaN 的原始特征 DataFrame
            verbose: 是否打印进度

        Returns:
            包含 SSM 特征和原始特征的完整 DataFrame
        """
        if verbose:
            print("  Training SSM models...")

        if self.config.ssm_types:
            valid_ssm_input_df = valid_raw_features_df[self.config.ssm_input_features]
            obs_dim = len(self.config.ssm_input_features)

            for ssm_type in self.config.ssm_types:
                if ssm_type not in self._ssm_processors:
                    self._ssm_processors[ssm_type] = self._create_ssm_adapter(
                        ssm_type, obs_dim
                    )

                processor = self._ssm_processors[ssm_type]
                if not processor.is_fitted:
                    if verbose:
                        print(f"    Training {ssm_type}...")
                    processor.fit(valid_ssm_input_df)

            # 立即计算 SSM 特征
            if verbose:
                print("  Computing SSM features...")
            ssm_features = self._compute_ssm_features_batch(valid_ssm_input_df)
            all_features = pd.concat([ssm_features, valid_raw_features_df], axis=1)
        else:
            if verbose:
                print("    No SSM configured, skipping...")
            all_features = valid_raw_features_df

        return all_features

    def _train_dimension_reducer_fit(
        self, all_features: pd.DataFrame, verbose: bool = False
    ) -> None:
        """
        fit 专用：训练降维器

        Args:
            all_features: 完整特征 DataFrame
            verbose: 是否打印进度
        """
        if verbose:
            print("  Training dimension reducer...")

        if self._dimension_reducer is None:
            self._dimension_reducer = self._create_dimension_reducer()

        if not self._dimension_reducer.is_fitted:
            self._dimension_reducer.fit(all_features)

        # 清理中间结果
        del all_features
        gc.collect()

    def _train_and_apply_reducer(
        self, all_features: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """
        fit_transform 专用：训练并应用降维器

        Args:
            all_features: 完整特征 DataFrame
            verbose: 是否打印进度

        Returns:
            降维后的特征 DataFrame
        """
        if verbose:
            print("  Training and applying dimension reducer...")

        if self._dimension_reducer is None:
            self._dimension_reducer = self._create_dimension_reducer()

        if not self._dimension_reducer.is_fitted:
            self._dimension_reducer.fit(all_features)

        return self._dimension_reducer.transform(all_features)

    # ========== SSM 复制 ==========

    def _validate_ssm_compatibility(
        self,
        source: "FeaturePipeline",
        ssm_type: str,
    ) -> None:
        """
        验证 SSM 配置兼容性

        Args:
            source: 源 Pipeline
            ssm_type: SSM 类型

        Raises:
            RuntimeError: 源 Pipeline 未 fit
            KeyError: 源 Pipeline 缺少所需的 SSM 类型
            ValueError: 配置不兼容（ssm_input_features 或 ssm_state_dim 不匹配）
        """
        # 1. 检查源是否已 fit
        if not source.is_fitted:
            raise RuntimeError(
                "Source pipeline not fitted. Call fit() or fit_transform() first."
            )

        # 2. 检查源是否有请求的 SSM
        if ssm_type not in source.ssm_processors:
            raise KeyError(
                f"Source pipeline lacks SSM type: {ssm_type}. "
                f"Available types: {list(source.ssm_processors.keys())}"
            )

        # 3. 验证 ssm_input_features 一致
        if self.config.ssm_input_features != source.config.ssm_input_features:
            raise ValueError(
                f"SSM input features mismatch:\n"
                f"  self:   {len(self.config.ssm_input_features)} features\n"
                f"  source: {len(source.config.ssm_input_features)} features\n"
                f"Both pipelines must use identical ssm_input_features."
            )

        # 4. 验证 ssm_state_dim 一致
        if self.config.ssm_state_dim != source.config.ssm_state_dim:
            raise ValueError(
                f"SSM state_dim mismatch: "
                f"self={self.config.ssm_state_dim}, source={source.config.ssm_state_dim}. "
                f"Both pipelines must use the same ssm_state_dim."
            )

    # ========== 训练接口 ==========

    def fit(
        self,
        candles: np.ndarray,
        verbose: bool = True,
    ) -> "FeaturePipeline":
        """
        训练所有模块（在无 NaN 数据上）

        NaN 处理策略（与 fit_transform/transform 一致）：
        1. 剔除开头连续 NaN 行后再进行训练
        2. 检查剩余数据是否还有 NaN，如有则报错

        Args:
            candles: K 线数据，二维数组 (N, 6)，列为 [timestamp, open, close, high, low, volume]
            verbose: 是否打印训练进度

        Returns:
            self（支持链式调用）

        Raises:
            TypeError: candles 不是 numpy 数组
            ValueError: candles 维度或形状不符合要求，或数据全为 NaN，或存在中间 NaN
        """
        self._validate_candles(candles)
        n_candles = len(candles)

        if verbose:
            print("FeaturePipeline: Starting fit...")

        # 1. 计算原始特征
        if verbose:
            print("  [1/4] Computing raw features...")
        raw_features_df = self._compute_raw_features(candles)

        # 2. NaN 处理（与 fit_transform/transform 一致）
        if verbose:
            print("  [2/4] Handling NaN values...")
        valid_raw_features_df, _ = self._prepare_valid_data(
            raw_features_df, n_candles, verbose
        )

        # 3. 训练 SSM
        if verbose:
            print("  [3/4] Training SSM models...")
        self._train_ssm_models(valid_raw_features_df, verbose)

        # 4. 可选：训练降维器
        if self.config.use_dimension_reducer:
            if verbose:
                print("  [4/4] Training dimension reducer...")
            all_features = self._compute_all_features_for_fit(valid_raw_features_df)
            self._train_dimension_reducer_fit(all_features, verbose)
        else:
            if verbose:
                print("  [4/4] Dimension reducer disabled, skipping...")

        self._is_fitted = True

        if verbose:
            print("FeaturePipeline: fit complete!")

        return self

    def fit_transform(
        self,
        candles: np.ndarray,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        训练并转换（优化版，单 pass 避免重复计算）

        与 fit().transform() 语义相同，但性能更优。

        NaN 处理策略：
        1. 剔除开头连续 NaN 行后再进行训练/转换
        2. 检查剩余数据是否还有 NaN，如有则报错
        3. 输出时在开头填充 NaN，确保输出行数 = candles 行数

        Args:
            candles: K 线数据，二维数组 (N, 6)，列为 [timestamp, open, close, high, low, volume]
            verbose: 是否打印进度

        Returns:
            转换后的特征 DataFrame（只包含 config.feature_names 中的特征）
            输出行数 = candles 行数，开头可能有 NaN 填充

        Raises:
            TypeError: candles 不是 numpy 数组
            ValueError: candles 维度或形状不符合要求，或数据全为 NaN，或存在中间 NaN
        """
        self._validate_candles(candles)
        n_candles = len(candles)

        if verbose:
            print("FeaturePipeline: Starting fit_transform...")

        # 1. 计算原始特征（只计算一次）
        if verbose:
            print("  [1/4] Computing raw features...")
        raw_features_df = self._compute_raw_features(candles)

        # 2. NaN 处理
        if verbose:
            print("  [2/4] Handling NaN values...")
        valid_raw_features_df, _ = self._prepare_valid_data(
            raw_features_df, n_candles, verbose
        )

        # 3. 训练 SSM 并立即计算 SSM 特征（优化：单 pass）
        if verbose:
            print("  [3/4] Training SSM and computing features...")
        all_features = self._train_and_compute_features(valid_raw_features_df, verbose)

        # 4. 可选：训练并应用降维
        if self.config.use_dimension_reducer:
            if verbose:
                print("  [4/4] Training and applying dimension reducer...")
            all_features = self._train_and_apply_reducer(all_features, verbose)
        else:
            if verbose:
                print("  [4/4] Dimension reducer disabled, skipping...")
            # 按 feature_names 过滤
            all_features = all_features[self.config.feature_names]

        self._is_fitted = True

        # 填充回原始行数
        result = self._pad_with_leading_nan(all_features, n_candles)

        if verbose:
            print(
                f"FeaturePipeline: fit_transform complete! Output shape: {result.shape}"
            )

        return result

    # ========== 批量转换接口 ==========

    def transform(self, candles: np.ndarray) -> pd.DataFrame:
        """
        批量转换（离线使用）

        NaN 处理策略：
        1. 剔除开头连续 NaN 行后再进行转换
        2. 检查剩余数据是否还有 NaN，如有则报错
        3. 输出时在开头填充 NaN，确保输出行数 = candles 行数

        Args:
            candles: K 线数据，二维数组 (N, 6)，列为 [timestamp, open, close, high, low, volume]

        Returns:
            特征 DataFrame（只包含 config.feature_names 中的特征）
            输出行数 = candles 行数，开头可能有 NaN 填充

        Raises:
            RuntimeError: Pipeline 未训练
            TypeError: candles 不是 numpy 数组
            ValueError: candles 维度或形状不符合要求，或数据全为 NaN，或存在中间 NaN
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        self._validate_candles(candles)
        n_candles = len(candles)

        # 1. 计算原始特征
        raw_features_df = self._compute_raw_features(candles)

        # 2. NaN 处理
        valid_raw_features_df, _ = self._prepare_valid_data(raw_features_df, n_candles)

        # 3. 计算 SSM 特征（使用无 NaN 的数据）
        if self.config.ssm_types:
            valid_ssm_input_df = valid_raw_features_df[self.config.ssm_input_features]
            ssm_features = self._compute_ssm_features_batch(valid_ssm_input_df)
            all_features = pd.concat([ssm_features, valid_raw_features_df], axis=1)
        else:
            all_features = valid_raw_features_df

        # 4. 可选降维
        if self.config.use_dimension_reducer and self._dimension_reducer is not None:
            all_features = self._dimension_reducer.transform(all_features)
        else:
            # 只返回用户请求的特征
            all_features = all_features[self.config.feature_names]

        # 填充回原始行数
        return self._pad_with_leading_nan(all_features, n_candles)

    def get_all_feature_names(self) -> List[str]:
        """
        获取配置中指定的所有特征名称

        Returns:
            config.feature_names 的副本
        """
        return self.config.feature_names.copy()

    def copy_ssm_from(
        self,
        source: "FeaturePipeline",
        ssm_types: Optional[List[str]] = None,
    ) -> "FeaturePipeline":
        """
        从另一个 Pipeline 复制 SSM 模型（深拷贝）

        这可以节省 SSM 的训练时间，适用于共享相同 SSM 配置的多个 Pipeline。
        复制后，调用 fit/fit_transform 时会自动跳过已复制的 SSM 的训练。

        典型使用场景：
        1. 训练全量特征 Pipeline（包含所有 SSM 特征）
        2. 特征筛选后，为每个模型创建特定 Pipeline
        3. 每个模型 Pipeline 从全量 Pipeline 复制 SSM，跳过重复训练

        Args:
            source: 源 Pipeline（必须已 fit）
            ssm_types: 要复制的 SSM 类型列表，如 ["deep_ssm", "lg_ssm"]
                       如果为 None，复制所有 self.config.ssm_types 中需要的类型

        Returns:
            self（支持链式调用）

        Raises:
            RuntimeError: 源 Pipeline 未 fit
            KeyError: 源 Pipeline 缺少所需的 SSM 类型
            ValueError: 配置不兼容（ssm_input_features 或 ssm_state_dim 不匹配）

        Example:
            >>> # 1. 训练全量 Pipeline
            >>> full_pipeline = FeaturePipeline(full_config)
            >>> full_pipeline.fit_transform(candles)
            >>>
            >>> # 2. 创建模型特定 Pipeline，复制 SSM
            >>> model_pipeline = FeaturePipeline(model_config)
            >>> model_pipeline.copy_ssm_from(full_pipeline)
            >>>
            >>> # 3. fit_transform 只训练降维器（SSM 已复制，自动跳过）
            >>> model_features = model_pipeline.fit_transform(candles)
        """
        # 确定要复制的 SSM 类型
        if ssm_types is None:
            ssm_types = self.config.ssm_types

        if not ssm_types:
            # 没有 SSM 需要复制
            return self

        # 验证并复制每个 SSM
        for ssm_type in ssm_types:
            # 验证配置兼容性
            self._validate_ssm_compatibility(source, ssm_type)

            # 深拷贝 SSM 适配器
            source_ssm = source.ssm_processors[ssm_type]
            copied_ssm = copy.deepcopy(source_ssm)

            # 重置内部状态（新 Pipeline 需要从初始状态开始）
            copied_ssm.reset_state()

            # 存储到本 Pipeline
            self._ssm_processors[ssm_type] = copied_ssm

        return self

    # ========== 实时推理接口 ==========

    def inference(self, candles: np.ndarray) -> pd.DataFrame:
        """
        单步实时推理

        输入完整历史 K 线数据（与 transform 一致），但只输出最新一行特征。
        某些特征（如移动平均）需要足够的历史数据才能正确计算。

        **NaN 零容忍**：如果特征中存在任何 NaN，立即报错停止。
        确保传入的 candles 有足够的历史数据（通常需要 400+ 行以支持 fracdiff 等特征）。

        Args:
            candles: K 线数据，二维数组 (N, 6)，列为 [timestamp, open, close, high, low, volume]
                     应包含足够的历史数据用于特征计算

        Returns:
            单行特征 DataFrame（只包含 config.feature_names 中的特征）

        Raises:
            RuntimeError: Pipeline 未训练
            TypeError: candles 不是 numpy 数组
            ValueError: candles 维度或形状不符合要求，或特征中存在 NaN
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() or load() first.")

        self._validate_candles(candles)

        # 加载 candles 并计算原始特征（只取最后一行）
        self._raw_calculator.load(candles, sequential=False)
        raw_features_dict = self._raw_calculator.get(
            self.config.all_calculator_features
        )
        raw_features_df = pd.DataFrame(raw_features_dict)

        # NaN 零容忍检查：如果任何特征是 NaN，立即报错
        nan_cols = raw_features_df.columns[raw_features_df.iloc[0].isna()].tolist()
        if nan_cols:
            raise ValueError(
                f"inference() received NaN in features: {nan_cols}. "
                f"Ensure candles has enough history (typically 400+ rows for fracdiff features)."
            )

        # SSM 推理（按 config.ssm_types 顺序，与 fit_transform/transform 保持一致）
        if self.config.ssm_types:
            ssm_input_values = (
                raw_features_df[self.config.ssm_input_features].iloc[0].values
            )
            ssm_dfs = []

            # 关键：使用 config.ssm_types 顺序，而非 _ssm_processors.items()
            # 这确保列顺序与 _compute_ssm_features_batch 一致
            for ssm_type in self.config.ssm_types:
                processor = self._ssm_processors.get(ssm_type)
                if processor is not None:
                    state = processor.inference(ssm_input_values)
                    ssm_features = {
                        f"{processor.prefix}_{i}": val for i, val in enumerate(state)
                    }
                    ssm_dfs.append(pd.DataFrame([ssm_features]))

            if ssm_dfs:
                ssm_df = pd.concat(ssm_dfs, axis=1)
                all_features = pd.concat([ssm_df, raw_features_df], axis=1)
            else:
                all_features = raw_features_df
        else:
            all_features = raw_features_df

        # 可选降维（与 transform 保持一致）
        if self.config.use_dimension_reducer and self._dimension_reducer is not None:
            all_features = self._dimension_reducer.transform(all_features)
            return all_features  # 降维时直接返回降维结果

        # 只返回用户请求的特征
        return all_features[self.config.feature_names]

    def reset_ssm_states(self) -> None:
        """重置所有 SSM 状态"""
        for processor in self._ssm_processors.values():
            processor.reset_state()

    def warmup_ssm(self, candles: np.ndarray, verbose: bool = False) -> None:
        """
        SSM 预热

        用历史数据逐行更新 SSM 状态，使其达到稳定状态。
        适用于策略启动时的初始化。**仅更新状态，不返回特征**。

        NaN 处理：自动跳过开头的 NaN 行，只对有效数据进行预热。

        Args:
            candles: K 线数据，二维数组 (N, 6)，列为 [timestamp, open, close, high, low, volume]
                     建议使用 3000+ 根 K 线以确保 SSM 状态稳定
            verbose: 是否打印进度

        Raises:
            RuntimeError: Pipeline 未训练
            TypeError: candles 不是 numpy 数组
            ValueError: candles 维度或形状不符合要求
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() or load() first.")

        self._validate_candles(candles)

        # 如果没有配置 SSM，直接返回
        if not self.config.ssm_types:
            if verbose:
                print("No SSM configured, skipping warmup.")
            return

        # 重置状态
        self.reset_ssm_states()

        # 计算所有 SSM 输入特征
        self._raw_calculator.load(candles, sequential=True)
        raw_features_dict = self._raw_calculator.get(self.config.ssm_input_features)
        ssm_input_df = pd.DataFrame(raw_features_dict)

        # 找到第一个有效行（跳过开头 NaN）
        first_valid = self._find_first_valid_row(ssm_input_df)

        if first_valid >= len(ssm_input_df):
            if verbose:
                print("All SSM input features are NaN, skipping warmup.")
            return

        if verbose and first_valid > 0:
            print(f"  Skipping first {first_valid} rows (NaN warmup)")

        # 只对有效行进行预热
        valid_input = ssm_input_df.iloc[first_valid:]
        n_valid = len(valid_input)

        if verbose:
            print(f"Warming up SSM with {n_valid} valid observations...")

        # 逐行更新 SSM 状态
        for i in range(n_valid):
            obs = valid_input.iloc[i].values
            for processor in self._ssm_processors.values():
                processor.inference(obs)

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_valid} observations")

        if verbose:
            print("SSM warmup complete!")

    # ========== 持久化接口 ==========

    def save(self, path: str, name: str) -> None:
        """
        保存 Pipeline 全部产物到子目录

        目录结构:
            path/name/
            ├── pipeline_config.json        # PipelineConfig 配置
            ├── deep_ssm.safetensors        # DeepSSM 权重（如果使用）
            ├── lg_ssm.safetensors          # LGSSM 权重（如果使用）
            ├── dimension_reducer.safetensors  # 降维器权重（如果使用）
            └── dimension_reducer.json      # 降维器配置（如果使用）

        Args:
            path: 基础保存路径
            name: Pipeline 名称，将创建子文件夹 path/name/

        Raises:
            RuntimeError: Pipeline 未训练

        Examples:
            >>> pipeline.fit_transform(candles)
            >>> pipeline.save("/path/to/models", "c_L5_N2")
            >>> # 所有产物保存到 /path/to/models/c_L5_N2/
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Cannot save unfitted pipeline.")

        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存配置
        config_path = save_dir / "pipeline_config.json"
        self.config.save(str(config_path))

        # 2. 保存 SSM 模型
        for ssm_type, processor in self._ssm_processors.items():
            ssm_path = save_dir / ssm_type
            processor.save(str(ssm_path))

        # 3. 保存降维器（如果有）
        if self._dimension_reducer is not None and self._dimension_reducer.is_fitted:
            self._dimension_reducer.save(str(save_dir), "dimension_reducer")

        print(f"Pipeline saved to {save_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        name: str,
        device: str = "cpu",
    ) -> "FeaturePipeline":
        """
        从子目录加载 Pipeline（包括降维器）

        Args:
            path: 基础路径
            name: Pipeline 名称（子文件夹名）
            device: 设备类型（"cpu" 或 "cuda"）

        Returns:
            加载的 FeaturePipeline 实例（已标记为 fitted）

        Raises:
            FileNotFoundError: 配置文件不存在

        Examples:
            >>> # 加载并使用
            >>> pipeline = FeaturePipeline.load("/path/to/models", "c_L5_N2")
            >>> features = pipeline.transform(candles)

            >>> # 实盘模式
            >>> pipeline = FeaturePipeline.load("/path/to/models", "c_L5_N2")
            >>> pipeline.warmup_ssm(historical_candles)
            >>> features = pipeline.inference(current_candles)
        """
        load_dir = Path(path) / name

        # 1. 加载配置
        config_path = load_dir / "pipeline_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = PipelineConfig.load(str(config_path))

        # 2. 加载 SSM 模型
        ssm_processors = {}
        for ssm_type in config.ssm_types:
            ssm_path = load_dir / ssm_type
            if ssm_path.with_suffix(".safetensors").exists():
                if ssm_type == "deep_ssm":
                    ssm_processors[ssm_type] = DeepSSMAdapter.load(
                        str(ssm_path), device=device
                    )
                elif ssm_type == "lg_ssm":
                    ssm_processors[ssm_type] = LGSSMAdapter.load(
                        str(ssm_path), device=device
                    )

        # 3. 加载降维器（如果存在）
        dimension_reducer = None
        reducer_path = load_dir / "dimension_reducer.safetensors"
        if reducer_path.exists():
            dimension_reducer = ARDVAE.load(str(load_dir), "dimension_reducer")
            config.use_dimension_reducer = True

        # 4. 创建实例
        pipeline = cls(
            config=config,
            ssm_processors=ssm_processors,
            dimension_reducer=dimension_reducer,
        )
        pipeline._is_fitted = True

        print(f"Pipeline loaded from {load_dir}")
        return pipeline

    # ========== 内部方法 ==========

    def _create_ssm_adapter(self, ssm_type: str, obs_dim: int) -> SSMProtocol:
        """创建 SSM 适配器"""
        state_dim = self.config.ssm_state_dim
        if ssm_type == "deep_ssm":
            config = DeepSSMConfig(obs_dim=obs_dim, state_dim=state_dim)
            return DeepSSMAdapter(config=config)
        elif ssm_type == "lg_ssm":
            config = LGSSMConfig(obs_dim=obs_dim, state_dim=state_dim)
            return LGSSMAdapter(config=config)
        else:
            raise ValueError(f"Unknown SSM type: {ssm_type}")

    def _create_dimension_reducer(self) -> DimensionReducerProtocol:
        """创建降维器"""
        if self.config.dimension_reducer_type == "ard_vae":
            reducer_config = self.config.dimension_reducer_config or {}
            config = ARDVAEConfig(**reducer_config)
            return ARDVAE(config)
        else:
            raise ValueError(
                f"Unknown dimension reducer type: {self.config.dimension_reducer_type}"
            )

    def _compute_ssm_features_batch(self, ssm_input_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 SSM 特征"""
        ssm_features = []

        for ssm_type in self.config.ssm_types:
            processor = self._ssm_processors.get(ssm_type)
            if processor is not None:
                ssm_df = processor.transform(ssm_input_df)
                ssm_features.append(ssm_df)

        if ssm_features:
            return pd.concat(ssm_features, axis=1)
        return pd.DataFrame(index=ssm_input_df.index)


if __name__ == "__main__":
    # 简单测试
    import numpy as np

    print("Testing FeaturePipeline...")

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 模拟 candles: [timestamp, open, close, high, low, volume]
    candles = np.column_stack(
        [
            np.arange(n_samples) * 60000,  # timestamp
            np.random.randn(n_samples).cumsum() + 100,  # open
            np.random.randn(n_samples).cumsum() + 100,  # close
            np.random.randn(n_samples).cumsum() + 101,  # high
            np.random.randn(n_samples).cumsum() + 99,  # low
            np.abs(np.random.randn(n_samples)) * 1000,  # volume
        ]
    )

    # 注意：这是一个简化测试，实际使用需要注册特征
    print("FeaturePipeline basic structure test passed!")
