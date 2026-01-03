"""
Reducer - 特征降维器

包装降维模型（如 ARDVAE），提供 sklearn 风格的 fit/transform 接口。
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.dimensionality_reduction import (
    ARDVAE,
    ARDVAEConfig,
    DimensionReducerProtocol,
)

from .reducer_config import ReducerConfig


class Reducer:
    """
    特征降维器

    包装降维模型（如 ARDVAE），提供 sklearn 风格的 fit/transform 接口。

    Architecture
    ------------
    ```
    Reducer
    └── DimensionReducer (ARDVAE / 其他)
        └── 降维器实现
    ```

    Output Guarantee
    ----------------
    - 输出 DataFrame 列名为整数字符串 "0", "1", "2", ...
    - 列数由降维器自动确定（如 ARDVAE 的 active_dims）

    Examples
    --------
    >>> config = ReducerConfig(
    ...     reducer_type="ard_vae",
    ...     ard_vae_config=ARDVAEConfig(max_latent_dim=512)
    ... )
    >>> reducer = Reducer(config)
    >>> reduced = reducer.fit_transform(raw_features)
    >>> reducer.save("/models", "model_c_L5_N2")
    >>>
    >>> # 加载并推理
    >>> reducer = Reducer.load("/models", "model_c_L5_N2")
    >>> single_reduced = reducer.inference(single_row_features)
    """

    def __init__(
        self,
        config: Optional[ReducerConfig] = None,
        dimension_reducer: Optional[DimensionReducerProtocol] = None,
    ):
        """
        初始化 Reducer

        Args:
            config: 配置对象
            dimension_reducer: 降维器实例（可选，默认延迟创建）
        """
        self.config = config or ReducerConfig()
        self._dimension_reducer = dimension_reducer
        self._is_fitted = False

        # 记录训练时的输入列名，用于 transform/inference 时验证
        self._input_feature_names: Optional[list[str]] = None

    # ==================== 核心属性 ====================

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted

    @property
    def n_components(self) -> int:
        """有效降维维度"""
        if self._dimension_reducer is None:
            return 0
        return self._dimension_reducer.n_components

    @property
    def input_feature_names(self) -> list[str]:
        """输入特征列名（训练后可用）"""
        if self._input_feature_names is None:
            raise ValueError("Reducer not fitted. Call fit() first.")
        return self._input_feature_names.copy()

    @property
    def input_schema_hash(self) -> str:
        """输入特征的 schema hash"""
        if self._input_feature_names is None:
            return ""
        schema_str = json.dumps(self._input_feature_names, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    # ==================== 输入验证 ====================

    def _validate_input(self, X: pd.DataFrame) -> None:
        """验证输入数据（类型和 NaN 检查）"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(
                f"Input contains NaN values in columns: {nan_cols}. "
                "Please handle missing data before calling Reducer."
            )

    # ==================== 训练接口 ====================

    def fit(self, X: pd.DataFrame, verbose: Optional[bool] = None) -> "Reducer":
        """
        训练降维器

        Args:
            X: 输入特征 DataFrame（无 NaN）
            verbose: 是否打印进度（None 表示使用 config.verbose）

        Returns:
            self
        """
        self._validate_input(X)

        if verbose is None:
            verbose = self.config.verbose

        # 记录输入列名
        self._input_feature_names = list(X.columns)

        # 可选：选择 subset 列
        if self.config.input_feature_names is not None:
            X = X[self.config.input_feature_names]

        # 创建并训练降维器
        if self._dimension_reducer is None:
            self._dimension_reducer = self._create_dimension_reducer()

        self._dimension_reducer.fit(X, verbose=verbose)
        self._is_fitted = True

        return self

    def fit_transform(
        self, X: pd.DataFrame, verbose: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        训练并转换

        Args:
            X: 输入特征 DataFrame（无 NaN）
            verbose: 是否打印进度

        Returns:
            降维后的 DataFrame，列名为 "0", "1", ...
        """
        self.fit(X, verbose=verbose)
        return self.transform(X)

    # ==================== 转换接口 ====================

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        批量降维

        Args:
            X: 输入特征 DataFrame，需包含训练时的所有列（可以有额外列）

        Returns:
            降维后的 DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")

        # 先选择训练时的列（自动 subset，保持顺序）
        if self._input_feature_names is not None:
            missing = set(self._input_feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            X = X[self._input_feature_names]

        # 后验证（类型和 NaN 检查）
        self._validate_input(X)

        # config 中的额外 subset（如果有）
        if self.config.input_feature_names is not None:
            X = X[self.config.input_feature_names]

        return self._dimension_reducer.transform(X)

    def inference(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        单行降维（实时推理）

        Args:
            X: 单行特征 DataFrame

        Returns:
            单行降维后的 DataFrame

        Notes:
            本质上调用 transform，但语义上表示实时推理场景。
        """
        if not self._is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() or load() first.")

        return self.transform(X)

    # ==================== 持久化接口 ====================

    def save(self, path: str, name: str) -> None:
        """
        保存 Reducer

        目录结构:
            path/
            ├── {name}.safetensors   # 降维器权重
            ├── {name}.json          # 降维器配置
            └── {name}_reducer_config.json  # Reducer 元数据

        Args:
            path: 保存目录
            name: 模型名称（如 "model_c_L5_N2"）
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted Reducer.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存降维器
        self._dimension_reducer.save(str(save_dir), name)

        # 保存 Reducer 配置和元数据
        reducer_meta = {
            "reducer_type": self.config.reducer_type,
            # 训练时实际使用的输入特征列名
            "input_feature_names": self._input_feature_names,
            # 配置中指定的 subset 选择（可能为 None）
            "config_input_feature_names": self.config.input_feature_names,
            "input_schema_hash": self.input_schema_hash,
            "n_components": self.n_components,
            "version": self.config.version,
        }

        # 保存 ard_vae_config（如果有）
        if self.config.ard_vae_config is not None:
            reducer_meta["ard_vae_config"] = self.config.ard_vae_config.model_dump()

        meta_path = save_dir / f"{name}_reducer_config.json"
        with open(meta_path, "w") as f:
            json.dump(reducer_meta, f, indent=2)

        print(f"Reducer saved to {save_dir}/{name}")

    @classmethod
    def load(cls, path: str, name: str) -> "Reducer":
        """
        加载 Reducer

        Args:
            path: 模型目录
            name: 模型名称

        Returns:
            加载的 Reducer 实例
        """
        load_dir = Path(path)

        # 加载 Reducer 元数据
        meta_path = load_dir / f"{name}_reducer_config.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Reducer config not found: {meta_path}")

        with open(meta_path, "r") as f:
            reducer_meta = json.load(f)

        # 重建配置
        ard_vae_config = None
        if "ard_vae_config" in reducer_meta and reducer_meta["ard_vae_config"]:
            ard_vae_config = ARDVAEConfig(**reducer_meta["ard_vae_config"])

        # 检查旧版元数据兼容性
        if "config_input_feature_names" not in reducer_meta:
            import warnings

            warnings.warn(
                "Loading reducer from old metadata format (missing 'config_input_feature_names'). "
                "If this model was trained with input_feature_names subset, "
                "please retrain to ensure correct behavior.",
                UserWarning,
                stacklevel=2,
            )

        config = ReducerConfig(
            reducer_type=reducer_meta.get("reducer_type", "ard_vae"),
            ard_vae_config=ard_vae_config,
            # 恢复配置中的 subset 选择
            input_feature_names=reducer_meta.get("config_input_feature_names"),
            version=reducer_meta.get("version", "1.0.0"),
        )

        # 加载降维器
        dimension_reducer = ARDVAE.load(str(load_dir), name)

        reducer = cls(config=config, dimension_reducer=dimension_reducer)
        reducer._is_fitted = True
        reducer._input_feature_names = reducer_meta.get("input_feature_names")

        # 校验 schema hash
        saved_hash = reducer_meta.get("input_schema_hash")
        if saved_hash and reducer.input_schema_hash != saved_hash:
            raise ValueError(
                f"Input schema hash mismatch: saved={saved_hash}, "
                f"computed={reducer.input_schema_hash}. "
                "The input feature configuration has changed."
            )

        print(f"Reducer loaded from {load_dir}/{name}")
        return reducer

    # ==================== 内部方法 ====================

    def _create_dimension_reducer(self) -> DimensionReducerProtocol:
        """创建降维器"""
        if self.config.reducer_type == "ard_vae":
            ard_config = self.config.ard_vae_config or ARDVAEConfig()
            return ARDVAE(ard_config)
        else:
            raise ValueError(f"Unknown reducer type: {self.config.reducer_type}")

    def __repr__(self) -> str:
        return (
            f"Reducer(\n"
            f"  is_fitted={self._is_fitted},\n"
            f"  n_components={self.n_components if self._is_fitted else 'N/A'},\n"
            f"  reducer_type={self.config.reducer_type},\n"
            f"  input_feature_count={len(self._input_feature_names) if self._input_feature_names else 'N/A'}\n"
            f")"
        )
