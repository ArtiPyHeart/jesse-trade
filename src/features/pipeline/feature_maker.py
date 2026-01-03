"""
FeatureMaker - 特征加工器

整合 SimpleFeatureCalculator 和 SSM 模型，输出完整原始特征 DataFrame。
不包含降维功能，降维由 Reducer 独立处理。
"""

import copy
import gc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.ssm import SSMProtocol, DeepSSMAdapter, LGSSMAdapter
from src.models.deep_ssm import DeepSSMConfig
from src.models.lgssm import LGSSMConfig

from .feature_maker_config import FeatureMakerConfig


class FeatureMaker:
    """
    特征加工器

    整合 SimpleFeatureCalculator 和 SSM 模型，输出完整原始特征 DataFrame。

    Architecture
    ------------
    ```
    FeatureMaker
    ├── SimpleFeatureCalculator
    │   └── 一阶特征计算
    └── SSMProcessors
        ├── DeepSSMAdapter
        └── LGSSMAdapter
    ```

    Usage Modes
    -----------
    1. **训练模式**: fit_transform(candles)
       训练 SSM 模型并返回特征

    2. **批量模式**: transform(candles)
       加载已训练的模型后，批量转换（离线使用）

    3. **预热模式**: warmup_ssm(candles)
       用历史数据逐行更新 SSM 状态

    4. **实时模式**: inference(candles)
       单步实时推理，只返回最后一行

    Output Guarantee
    ----------------
    - 输出 DataFrame 列顺序固定：[SSM 特征, 原始特征]
    - fit_transform/transform/inference 在相同输入下产生一致的输出
    - 输出行数 = candles 行数，开头可能有 NaN 填充

    Examples
    --------
    >>> config = FeatureMakerConfig(
    ...     feature_names=["deep_ssm_0", "lg_ssm_0", "rsi", "macd"]
    ... )
    >>> maker = FeatureMaker(config)
    >>> features = maker.fit_transform(candles)
    >>> maker.save("/models", "feature_maker")
    >>>
    >>> # 加载并推理
    >>> maker = FeatureMaker.load("/models", "feature_maker")
    >>> maker.warmup_ssm(historical_candles)
    >>> latest_features = maker.inference(current_candles)
    """

    def __init__(
        self,
        config: Optional[FeatureMakerConfig] = None,
        raw_calculator: Optional[SimpleFeatureCalculator] = None,
        ssm_processors: Optional[Dict[str, SSMProtocol]] = None,
    ):
        """
        初始化 FeatureMaker

        Args:
            config: 配置对象
            raw_calculator: 原始特征计算器（可选，默认创建新实例）
            ssm_processors: SSM 处理器字典 {"deep_ssm": ..., "lg_ssm": ...}
        """
        self.config = config or FeatureMakerConfig()
        self._raw_calculator = raw_calculator or SimpleFeatureCalculator(
            verbose=self.config.verbose
        )
        self._ssm_processors: Dict[str, SSMProtocol] = ssm_processors or {}
        self._is_fitted = False

        # 幂等保护：记录最后处理的 timestamp，避免重复推进 SSM 状态
        self._last_inference_timestamp: Optional[float] = None
        self._last_inference_result: Optional[pd.DataFrame] = None

    # ==================== 核心属性 ====================

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted

    @property
    def ssm_processors(self) -> Dict[str, SSMProtocol]:
        """SSM 处理器字典"""
        return self._ssm_processors

    @property
    def output_columns(self) -> list[str]:
        """输出 DataFrame 的列名（保证顺序一致）"""
        return self.config.feature_names.copy()

    @property
    def verbose(self) -> bool:
        """当前 verbose 配置"""
        return self.config.verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """统一更新 verbose 设置"""
        self.config.verbose = bool(value)
        if self._raw_calculator is not None:
            self._raw_calculator.verbose = self.config.verbose

    # ==================== 输入验证 ====================

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

    # ==================== NaN 处理 ====================

    def _find_first_valid_row(self, df: pd.DataFrame) -> int:
        """
        找到第一个全部有效（无 NaN）的行索引

        Args:
            df: 待检查的 DataFrame

        Returns:
            第一个有效行的索引，如果全是 NaN 返回 len(df)
        """
        valid_rows = ~df.isna().any(axis=1)
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
            ValueError: 如果存在中间 NaN
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
            target_rows: 目标行数

        Returns:
            填充后的 DataFrame
        """
        current_rows = len(df)
        if current_rows >= target_rows:
            return df

        nan_rows = target_rows - current_rows
        nan_df = pd.DataFrame(
            np.float32(np.nan),
            index=range(nan_rows),
            columns=df.columns,
        )

        df = df.reset_index(drop=True)
        df.index = range(nan_rows, target_rows)

        result = pd.concat([nan_df, df])
        result.index = range(target_rows)

        return result

    # ==================== 特征计算 ====================

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

        self._validate_no_intermediate_nan(raw_features_df, first_valid)

        valid_raw_features_df = raw_features_df.iloc[first_valid:].reset_index(
            drop=True
        )

        return valid_raw_features_df, first_valid

    def _train_and_compute_features(
        self, valid_raw_features_df: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """
        训练 SSM 并计算特征（单 pass 优化）

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

            if verbose:
                print("  Computing SSM features...")
            ssm_features = self._compute_ssm_features_batch(valid_ssm_input_df)

            del valid_ssm_input_df
            gc.collect()

            all_features = pd.concat([ssm_features, valid_raw_features_df], axis=1)
        else:
            if verbose:
                print("    No SSM configured, skipping...")
            all_features = valid_raw_features_df

        return all_features

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

    # ==================== 训练接口 ====================

    def fit(self, candles: np.ndarray) -> "FeatureMaker":
        """
        训练 SSM 模型（不返回特征）

        Args:
            candles: K 线数据 (N, 6)

        Returns:
            self（支持链式调用）
        """
        verbose = self.config.verbose

        self._validate_candles(candles)
        n_candles = len(candles)

        if verbose:
            print("FeatureMaker: Starting fit...")

        raw_features_df = self._compute_raw_features(candles, verbose)
        valid_raw_features_df, _ = self._prepare_valid_data(
            raw_features_df, n_candles, verbose
        )

        # 训练 SSM
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

        self._is_fitted = True

        if verbose:
            print("FeatureMaker: fit complete!")

        return self

    def fit_transform(self, candles: np.ndarray) -> pd.DataFrame:
        """
        训练并转换（优化版，单 pass 避免重复计算）

        NaN 处理策略：
        1. 剔除开头连续 NaN 行后再进行训练/转换
        2. 检查剩余数据是否还有 NaN，如有则报错
        3. 输出时在开头填充 NaN，确保输出行数 = candles 行数

        Args:
            candles: K 线数据 (N, 6)

        Returns:
            特征 DataFrame，行数 = candles 行数（开头可能有 NaN）
        """
        verbose = self.config.verbose

        self._validate_candles(candles)
        n_candles = len(candles)

        if verbose:
            print("FeatureMaker: Starting fit_transform...")

        # 1. 计算原始特征
        if verbose:
            print("  [1/3] Computing raw features...")
        raw_features_df = self._compute_raw_features(candles)

        # 2. NaN 处理
        if verbose:
            print("  [2/3] Handling NaN values...")
        valid_raw_features_df, first_valid = self._prepare_valid_data(
            raw_features_df, n_candles, verbose
        )

        del raw_features_df
        gc.collect()

        # 3. 训练 SSM 并计算特征
        if verbose:
            print("  [3/3] Training SSM and computing features...")
        all_features = self._train_and_compute_features(valid_raw_features_df, verbose)

        del valid_raw_features_df
        gc.collect()

        # 按 feature_names 过滤
        all_features = all_features[self.config.feature_names]

        self._is_fitted = True

        # 填充回原始行数
        result_df = self._pad_with_leading_nan(all_features, n_candles)

        if verbose:
            print(
                f"FeatureMaker: fit_transform complete! Output shape: {result_df.shape}"
            )

        return result_df

    # ==================== 批量转换接口 ====================

    def transform(self, candles: np.ndarray) -> pd.DataFrame:
        """
        批量转换（已训练后使用）

        Args:
            candles: K 线数据 (N, 6)

        Returns:
            特征 DataFrame，行数 = candles 行数（开头可能有 NaN）
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureMaker not fitted. Call fit() first.")

        self._validate_candles(candles)
        n_candles = len(candles)

        raw_features_df = self._compute_raw_features(candles)
        valid_raw_features_df, first_valid = self._prepare_valid_data(
            raw_features_df, n_candles
        )

        if self.config.ssm_types:
            valid_ssm_input_df = valid_raw_features_df[self.config.ssm_input_features]
            ssm_features = self._compute_ssm_features_batch(valid_ssm_input_df)
            all_features = pd.concat([ssm_features, valid_raw_features_df], axis=1)
        else:
            all_features = valid_raw_features_df

        all_features = all_features[self.config.feature_names]
        result_df = self._pad_with_leading_nan(all_features, n_candles)

        return result_df

    # ==================== 实时推理接口 ====================

    def inference(self, candles: np.ndarray) -> pd.DataFrame:
        """
        单步实时推理

        输入完整历史 K 线，但只输出最新一行特征。
        特征中若有 NaN，立即报错（零容忍策略）。

        **幂等保护**：同一 timestamp 的重复调用不会推进 SSM 状态，
        直接返回缓存结果。这保证了"每根 bar 只推理一次"的设计。

        Args:
            candles: K 线数据 (N, 6)，需包含足够历史（400+ 行）

        Returns:
            单行特征 DataFrame

        Raises:
            RuntimeError: 未训练
            ValueError: 特征中存在 NaN
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureMaker not fitted. Call fit() or load() first.")

        self._validate_candles(candles)

        # 幂等保护：检查 timestamp 是否已处理过
        current_timestamp = float(candles[-1, 0])
        if (
            self._last_inference_timestamp is not None
            and self._last_inference_timestamp == current_timestamp
            and self._last_inference_result is not None
        ):
            return self._last_inference_result.copy()

        # 计算原始特征（只取最后一行）
        # 注意：如果 warmup_ssm 用 sequential=True 缓存过同一 candles 引用，
        # get() 会返回全序列。因此必须显式取最后一行。
        self._raw_calculator.load(candles, sequential=False)
        raw_features_dict = self._raw_calculator.get(
            self.config.all_calculator_features
        )
        raw_features_df = pd.DataFrame(raw_features_dict)

        # 显式取最后一行（处理缓存返回全序列的情况）
        if len(raw_features_df) > 1:
            raw_features_df = raw_features_df.iloc[[-1]].reset_index(drop=True)

        # NaN 零容忍检查
        nan_cols = raw_features_df.columns[raw_features_df.iloc[0].isna()].tolist()
        if nan_cols:
            raise ValueError(
                f"inference() received NaN in features: {nan_cols}. "
                f"Ensure candles has enough history (typically 400+ rows)."
            )

        # SSM 推理
        if self.config.ssm_types:
            ssm_input_values = (
                raw_features_df[self.config.ssm_input_features].iloc[0].values
            )
            ssm_dfs = []

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

        result = all_features[self.config.feature_names]

        # 缓存结果用于幂等保护
        self._last_inference_timestamp = current_timestamp
        self._last_inference_result = result.copy()

        return result

    # ==================== SSM 状态管理 ====================

    def warmup_ssm(self, candles: np.ndarray) -> None:
        """
        SSM 预热

        用历史数据逐行更新 SSM 状态，使其达到稳定状态。
        仅更新状态，不返回特征。

        Args:
            candles: K 线数据，建议 3000+ 根
        """
        verbose = self.config.verbose

        if not self._is_fitted:
            raise RuntimeError("FeatureMaker not fitted. Call fit() or load() first.")

        self._validate_candles(candles)

        if not self.config.ssm_types:
            if verbose:
                print("No SSM configured, skipping warmup.")
            return

        self.reset_ssm_states()

        self._raw_calculator.load(candles, sequential=True)
        raw_features_dict = self._raw_calculator.get(self.config.ssm_input_features)
        ssm_input_df = pd.DataFrame(raw_features_dict)

        first_valid = self._find_first_valid_row(ssm_input_df)

        if first_valid >= len(ssm_input_df):
            if verbose:
                print("All SSM input features are NaN, skipping warmup.")
            return

        if verbose and first_valid > 0:
            print(f"  Skipping first {first_valid} rows (NaN warmup)")

        valid_input = ssm_input_df.iloc[first_valid:]
        n_valid = len(valid_input)

        if verbose:
            print(f"Warming up SSM with {n_valid} valid observations...")

        for i in range(n_valid):
            obs = valid_input.iloc[i].values
            for processor in self._ssm_processors.values():
                processor.inference(obs)

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_valid} observations")

        if verbose:
            print("SSM warmup complete!")

    def reset_ssm_states(self) -> None:
        """重置所有 SSM 状态和幂等缓存"""
        for processor in self._ssm_processors.values():
            processor.reset_state()

        # 清除幂等缓存
        self._last_inference_timestamp = None
        self._last_inference_result = None

    # ==================== SSM 复制 ====================

    def copy_ssm_from(
        self,
        source: "FeatureMaker",
        ssm_types: Optional[list[str]] = None,
    ) -> "FeatureMaker":
        """
        从另一个 FeatureMaker 复制 SSM 模型（深拷贝）

        Args:
            source: 源 FeatureMaker（必须已 fit）
            ssm_types: 要复制的 SSM 类型列表

        Returns:
            self（支持链式调用）
        """
        if ssm_types is None:
            ssm_types = self.config.ssm_types

        if not ssm_types:
            return self

        if not source.is_fitted:
            raise RuntimeError(
                "Source FeatureMaker not fitted. Call fit() or fit_transform() first."
            )

        for ssm_type in ssm_types:
            if ssm_type not in source.ssm_processors:
                raise KeyError(
                    f"Source FeatureMaker lacks SSM type: {ssm_type}. "
                    f"Available types: {list(source.ssm_processors.keys())}"
                )

            if self.config.ssm_input_features != source.config.ssm_input_features:
                raise ValueError(
                    "SSM input features mismatch. "
                    "Both FeatureMakers must use identical ssm_input_features."
                )

            if self.config.ssm_state_dim != source.config.ssm_state_dim:
                raise ValueError(
                    f"SSM state_dim mismatch: "
                    f"self={self.config.ssm_state_dim}, source={source.config.ssm_state_dim}."
                )

            source_ssm = source.ssm_processors[ssm_type]
            copied_ssm = copy.deepcopy(source_ssm)
            copied_ssm.reset_state()
            self._ssm_processors[ssm_type] = copied_ssm

        # 清除幂等缓存（SSM 状态已变更）
        self._last_inference_timestamp = None
        self._last_inference_result = None

        return self

    def share_raw_calculator_from(self, source: "FeatureMaker") -> "FeatureMaker":
        """
        共享另一个 FeatureMaker 的 SimpleFeatureCalculator

        Args:
            source: 源 FeatureMaker

        Returns:
            self（支持链式调用）
        """
        self._raw_calculator = source._raw_calculator
        if self._raw_calculator is not None:
            self._raw_calculator.verbose = self.config.verbose

        # 清除幂等缓存（calculator 状态已变更）
        self._last_inference_timestamp = None
        self._last_inference_result = None

        return self

    # ==================== 持久化接口 ====================

    def save(self, path: str, name: str) -> None:
        """
        保存 FeatureMaker 到子目录

        目录结构:
            path/name/
            ├── feature_maker_config.json
            ├── deep_ssm.safetensors (如果使用)
            └── lg_ssm.safetensors (如果使用)

        Args:
            path: 基础路径
            name: FeatureMaker 名称
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureMaker.")

        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self.config.save(str(save_dir / "feature_maker_config.json"))

        # 保存 SSM 模型
        for ssm_type, processor in self._ssm_processors.items():
            processor.save(str(save_dir / ssm_type))

        print(f"FeatureMaker saved to {save_dir}")

    @classmethod
    def load(cls, path: str, name: str, device: str = "cpu") -> "FeatureMaker":
        """
        加载 FeatureMaker

        Args:
            path: 基础路径
            name: FeatureMaker 名称
            device: 设备类型

        Returns:
            加载的 FeatureMaker 实例
        """
        load_dir = Path(path) / name

        config_path = load_dir / "feature_maker_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = FeatureMakerConfig.load(str(config_path))

        ssm_processors = {}
        for ssm_type in config.ssm_types:
            ssm_path = load_dir / ssm_type
            ssm_weight_path = ssm_path.with_suffix(".safetensors")

            # 严格要求 SSM 权重必须存在
            if not ssm_weight_path.exists():
                raise FileNotFoundError(
                    f"SSM weight file not found: {ssm_weight_path}. "
                    f"Config requires SSM type '{ssm_type}' but weights are missing."
                )

            if ssm_type == "deep_ssm":
                ssm_processors[ssm_type] = DeepSSMAdapter.load(
                    str(ssm_path), device=device
                )
            elif ssm_type == "lg_ssm":
                ssm_processors[ssm_type] = LGSSMAdapter.load(
                    str(ssm_path), device=device
                )
            else:
                raise ValueError(f"Unknown SSM type in config: {ssm_type}")

        maker = cls(config=config, ssm_processors=ssm_processors)
        maker._is_fitted = True

        print(f"FeatureMaker loaded from {load_dir}")
        return maker
