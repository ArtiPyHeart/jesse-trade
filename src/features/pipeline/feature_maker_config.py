"""
FeatureMakerConfig - 特征加工器配置类

定义 FeatureMaker 的配置参数，支持自动识别特征层级。
FeatureMaker 负责特征加工（SimpleFeatureCalculator + SSM），不包含降维。
"""

import hashlib
import json
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


# SSM 默认输入特征（fracdiff 特征）
SSM_DEFAULT_INPUT_FEATURES = [
    f"frac_{p1}_{p2}{lag}_diff"
    for p1 in ["o", "h", "l", "c"]
    for p2 in ["o", "h", "l", "c"]
    for lag in range(1, 6)
]


class FeatureMakerConfig(BaseModel):
    """
    FeatureMaker 配置

    用户只需指定最终想要的特征名称，配置会自动识别特征层级：
    - 一阶特征：由 SimpleFeatureCalculator 直接计算（如 rsi, macd）
    - 二阶特征：由 SSM 处理器生成（如 deep_ssm_0, lg_ssm_1）

    Attributes
    ----------
    feature_names : List[str]
        最终想要的特征名称列表（一阶 + 二阶特征）
    ssm_state_dim : int
        SSM 输出维度（默认 5，即 deep_ssm_0 到 deep_ssm_4）
    ssm_input_features : List[str]
        SSM 输入特征列表（默认为 fracdiff 特征，用户通常无需修改）
    verbose : bool
        是否打印进度信息（默认 False）
    calculator_feature_order : Optional[List[str]]
        固定的特征计算顺序（保存时记录，加载时校验）
    version : str
        配置版本号

    Examples
    --------
    >>> config = FeatureMakerConfig(
    ...     feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0", "rsi", "macd"]
    ... )
    >>> config.ssm_types
    ['deep_ssm', 'lg_ssm']
    >>> config.raw_feature_names
    ['rsi', 'macd']
    """

    model_config = ConfigDict(validate_assignment=True)

    # 用户接口 - 最终想要的特征
    feature_names: List[str] = Field(default_factory=list)

    # SSM 配置
    ssm_state_dim: int = 5
    ssm_input_features: List[str] = Field(
        default_factory=lambda: SSM_DEFAULT_INPUT_FEATURES.copy()
    )

    # 运行时配置
    verbose: bool = False

    # 持久化：用于固定 SimpleFeatureCalculator 的特征顺序
    calculator_feature_order: Optional[List[str]] = None

    # 元信息
    version: str = "1.0.0"

    # 内部解析结果（使用 PrivateAttr，不参与序列化）
    _raw_features: List[str] = PrivateAttr(default_factory=list)
    _ssm_features: List[str] = PrivateAttr(default_factory=list)
    # 使用 List 而非 Set，保证 SSM 类型的顺序一致性
    # 顺序由 feature_names 中首次出现的 SSM 类型决定
    _ssm_types: List[str] = PrivateAttr(default_factory=list)
    _config_signature: Optional[tuple[tuple[str, ...], int, tuple[str, ...]]] = (
        PrivateAttr(default=None)
    )

    @model_validator(mode="after")
    def parse_and_validate(self) -> "FeatureMakerConfig":
        """解析特征并验证配置"""
        self._parse_features()
        self._validate_ssm_feature_count()
        self._ensure_calculator_feature_order()
        self._update_signature()
        return self

    def _parse_features(self):
        """解析特征名称，识别层级"""
        self._raw_features = []
        self._ssm_features = []
        self._ssm_types = []

        for name in self.feature_names:
            if name.startswith("deep_ssm_"):
                self._ssm_features.append(name)
                if "deep_ssm" not in self._ssm_types:
                    self._ssm_types.append("deep_ssm")
                self._validate_ssm_index(name)
            elif name.startswith("lg_ssm_"):
                self._ssm_features.append(name)
                if "lg_ssm" not in self._ssm_types:
                    self._ssm_types.append("lg_ssm")
                self._validate_ssm_index(name)
            else:
                self._raw_features.append(name)

    def _current_signature(self) -> tuple[tuple[str, ...], int, tuple[str, ...]]:
        return (
            tuple(self.feature_names),
            self.ssm_state_dim,
            tuple(self.ssm_input_features),
        )

    def _update_signature(self) -> None:
        self._config_signature = self._current_signature()

    def _ensure_cache_uptodate(self) -> None:
        current = self._current_signature()
        if self._config_signature is None:
            self._config_signature = current
            return

        if self._config_signature != current:
            raise ValueError(
                "FeatureMakerConfig was mutated in-place after initialization. "
                "Recreate FeatureMakerConfig or call refresh(force_reset=True) "
                "to rebuild derived fields."
            )

    def refresh(self, *, force_reset: bool = False) -> None:
        """
        重新解析配置并同步内部缓存

        Args:
            force_reset: 是否强制重建 calculator_feature_order
        """
        if force_reset:
            self.calculator_feature_order = None
        self._parse_features()
        self._validate_ssm_feature_count()
        self._ensure_calculator_feature_order()
        self._update_signature()

    def _validate_ssm_index(self, name: str):
        """验证 SSM 特征索引范围"""
        try:
            idx = int(name.split("_")[-1])
            if idx >= self.ssm_state_dim:
                raise ValueError(
                    f"Invalid SSM feature '{name}': index {idx} >= state_dim {self.ssm_state_dim}"
                )
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid SSM feature name: {name}")
            raise

    def _validate_ssm_feature_count(self) -> None:
        """验证 SSM 特征数量不超过 state_dim"""
        if not self._ssm_types:
            return

        counts = {ssm_type: 0 for ssm_type in self._ssm_types}
        for name in self.feature_names:
            if name.startswith("deep_ssm_") and "deep_ssm" in counts:
                counts["deep_ssm"] += 1
            elif name.startswith("lg_ssm_") and "lg_ssm" in counts:
                counts["lg_ssm"] += 1

        for ssm_type, count in counts.items():
            if count > self.ssm_state_dim:
                raise ValueError(
                    f"Requested {ssm_type} features ({count}) exceeds "
                    f"ssm_state_dim {self.ssm_state_dim}."
                )

    def _ensure_calculator_feature_order(self) -> None:
        """确保 calculator_feature_order 已设置且一致"""
        computed_order: List[str] = []
        seen = set()

        # 先添加一阶特征
        for name in self._raw_features:
            if name not in seen:
                computed_order.append(name)
                seen.add(name)

        # 如果需要 SSM，添加 SSM 输入特征
        if self._ssm_types:
            for name in self.ssm_input_features:
                if name not in seen:
                    computed_order.append(name)
                    seen.add(name)

        if self.calculator_feature_order is None:
            self.calculator_feature_order = computed_order
            return

        if self.calculator_feature_order != computed_order:
            raise ValueError(
                "calculator_feature_order mismatch with current configuration. "
                "Rebuild the FeatureMaker with consistent settings."
            )

    @property
    def raw_feature_names(self) -> List[str]:
        """一阶特征名称（直接输出到结果）"""
        self._ensure_cache_uptodate()
        return self._raw_features.copy()

    @property
    def ssm_feature_names(self) -> List[str]:
        """二阶特征名称（SSM 输出）"""
        self._ensure_cache_uptodate()
        return self._ssm_features.copy()

    @property
    def ssm_types(self) -> List[str]:
        """需要启用的 SSM 类型列表（顺序由 feature_names 中首次出现决定）"""
        self._ensure_cache_uptodate()
        return self._ssm_types.copy()

    @property
    def all_calculator_features(self) -> List[str]:
        """
        需要从 SimpleFeatureCalculator 计算的所有特征

        包括：
        - 一阶特征（直接输出）
        - SSM 输入特征（如果需要 SSM）
        """
        self._ensure_cache_uptodate()
        if self.calculator_feature_order is None:
            raise ValueError("calculator_feature_order not initialized.")
        return self.calculator_feature_order.copy()

    @property
    def schema_hash(self) -> str:
        """
        计算配置的 schema hash，用于校验兼容性

        基于以下字段计算：
        - feature_names
        - ssm_state_dim
        - ssm_input_features
        - calculator_feature_order
        """
        self._ensure_cache_uptodate()
        schema_dict = {
            "feature_names": self.feature_names,
            "ssm_state_dim": self.ssm_state_dim,
            "ssm_input_features": self.ssm_input_features,
            "calculator_feature_order": self.calculator_feature_order,
        }
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def save(self, path: str) -> None:
        """
        保存配置到 JSON 文件

        Args:
            path: 保存路径
        """
        config_dict = {
            "feature_names": self.feature_names,
            "ssm_state_dim": self.ssm_state_dim,
            "ssm_input_features": self.ssm_input_features,
            "verbose": self.verbose,
            "version": self.version,
            "calculator_feature_order": self.calculator_feature_order,
            "schema_hash": self.schema_hash,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureMakerConfig":
        """
        从 JSON 文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            加载的配置实例
        """
        with open(path, "r") as f:
            config_dict = json.load(f)

        # 移除不需要传入构造器的字段
        saved_hash = config_dict.pop("schema_hash", None)
        config_dict.pop("_raw_features", None)
        config_dict.pop("_ssm_features", None)
        config_dict.pop("_ssm_types", None)

        if "calculator_feature_order" not in config_dict:
            raise ValueError(
                "Missing calculator_feature_order in config. "
                "Delete old models and retrain."
            )

        instance = cls(**config_dict)

        # 校验 schema hash
        if saved_hash is not None and instance.schema_hash != saved_hash:
            raise ValueError(
                f"Schema hash mismatch: saved={saved_hash}, computed={instance.schema_hash}. "
                "Configuration has changed. Delete old models and retrain."
            )

        return instance

    def validate_features_exist(self, available_features: List[str]) -> None:
        """
        验证配置中的一阶特征是否都可用

        Args:
            available_features: 可用特征列表

        Raises:
            ValueError: 如果有特征不可用
        """
        available_set = set(available_features)
        features_to_check = set(self.all_calculator_features)
        missing = features_to_check - available_set

        if missing:
            raise ValueError(f"Missing features: {missing}")

    def __repr__(self) -> str:
        return (
            f"FeatureMakerConfig(\n"
            f"  feature_names={len(self.feature_names)} features,\n"
            f"  raw_features={self._raw_features},\n"
            f"  ssm_features={self._ssm_features},\n"
            f"  ssm_types={self.ssm_types},\n"
            f"  ssm_state_dim={self.ssm_state_dim},\n"
            f"  schema_hash={self.schema_hash}\n"
            f")"
        )
