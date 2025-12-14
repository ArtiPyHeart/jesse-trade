"""
Pipeline Configuration - FeaturePipeline 配置类

定义 FeaturePipeline 的配置参数，支持自动识别特征层级。
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# SSM 默认输入特征（fracdiff 特征）
SSM_DEFAULT_INPUT_FEATURES = [
    f"frac_{p1}_{p2}{l}_diff"
    for p1 in ["o", "h", "l", "c"]
    for p2 in ["o", "h", "l", "c"]
    for l in range(1, 6)
]


@dataclass
class PipelineConfig:
    """
    FeaturePipeline 配置

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
    use_dimension_reducer : bool
        是否使用降维模块
    dimension_reducer_type : str
        降维器类型，目前支持 "ard_vae"
    dimension_reducer_config : Optional[Dict]
        降维器配置参数
    verbose : bool
        是否打印进度信息（默认 False）。
        此设置会被 FeaturePipeline 及其子模块继承：
        - SimpleFeatureCalculator：构造时继承
        - ARDVAE：fit 时通过参数传递
        - fit/fit_transform/warmup_ssm：默认继承，可通过参数覆盖
    version : str
        配置版本号

    Examples
    --------
    >>> # 用户只需指定最终想要的特征
    >>> config = PipelineConfig(
    ...     feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0", "rsi", "macd"]
    ... )
    >>> # 自动识别需要 deep_ssm 和 lg_ssm
    >>> config.ssm_types
    ['deep_ssm', 'lg_ssm']
    >>> # 自动识别一阶特征
    >>> config.raw_feature_names
    ['rsi', 'macd']
    """

    # 用户接口 - 最终想要的特征
    feature_names: List[str] = field(default_factory=list)

    # SSM 配置
    ssm_state_dim: int = 5
    ssm_input_features: List[str] = field(
        default_factory=lambda: SSM_DEFAULT_INPUT_FEATURES.copy()
    )

    # 降维配置
    use_dimension_reducer: bool = False
    dimension_reducer_type: str = "ard_vae"
    dimension_reducer_config: Optional[Dict] = None

    # 运行时配置
    verbose: bool = False

    # 元信息
    version: str = "2.0.0"

    # 内部解析结果（不序列化）
    _raw_features: List[str] = field(default_factory=list, repr=False)
    _ssm_features: List[str] = field(default_factory=list, repr=False)
    # 使用 List 而非 Set，保证 SSM 类型的顺序一致性
    # 顺序由 feature_names 中首次出现的 SSM 类型决定
    _ssm_types: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """解析特征并验证配置"""
        self._parse_features()
        self._validate()

    def _parse_features(self):
        """解析特征名称，识别层级"""
        self._raw_features = []
        self._ssm_features = []
        self._ssm_types = []  # 有序列表，保持首次出现顺序

        for name in self.feature_names:
            if name.startswith("deep_ssm_"):
                self._ssm_features.append(name)
                # 使用 list 去重，保持首次出现顺序
                if "deep_ssm" not in self._ssm_types:
                    self._ssm_types.append("deep_ssm")
                # 验证索引范围
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
            elif name.startswith("lg_ssm_"):
                self._ssm_features.append(name)
                if "lg_ssm" not in self._ssm_types:
                    self._ssm_types.append("lg_ssm")
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
            else:
                self._raw_features.append(name)

    def _validate(self):
        """验证配置"""
        # 验证降维器类型
        valid_reducer_types = {"ard_vae"}
        if self.dimension_reducer_type not in valid_reducer_types:
            raise ValueError(
                f"Invalid dimension reducer type: {self.dimension_reducer_type}. "
                f"Must be one of {valid_reducer_types}"
            )

    @property
    def raw_feature_names(self) -> List[str]:
        """一阶特征名称（直接输出到结果）"""
        return self._raw_features

    @property
    def ssm_feature_names(self) -> List[str]:
        """二阶特征名称（SSM 输出）"""
        return self._ssm_features

    @property
    def ssm_types(self) -> List[str]:
        """需要启用的 SSM 类型列表（顺序由 feature_names 中首次出现决定）"""
        return self._ssm_types.copy()

    @property
    def all_calculator_features(self) -> List[str]:
        """
        需要从 SimpleFeatureCalculator 计算的所有特征

        包括：
        - 一阶特征（直接输出）
        - SSM 输入特征（如果需要 SSM）
        """
        features = set(self._raw_features)
        if self._ssm_types:
            features.update(self.ssm_input_features)
        return list(features)

    def save(self, path: str) -> None:
        """
        保存配置到 JSON 文件

        Args:
            path: 保存路径
        """
        # 只保存用户可配置的字段，不保存内部解析结果
        config_dict = {
            "feature_names": self.feature_names,
            "ssm_state_dim": self.ssm_state_dim,
            "ssm_input_features": self.ssm_input_features,
            "use_dimension_reducer": self.use_dimension_reducer,
            "dimension_reducer_type": self.dimension_reducer_type,
            "dimension_reducer_config": self.dimension_reducer_config,
            "verbose": self.verbose,
            "version": self.version,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """
        从 JSON 文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            加载的配置实例
        """
        with open(path, "r") as f:
            config_dict = json.load(f)

        # 移除内部字段（如果存在）
        config_dict.pop("_raw_features", None)
        config_dict.pop("_ssm_features", None)
        config_dict.pop("_ssm_types", None)

        return cls(**config_dict)

    def copy(self, **kwargs) -> "PipelineConfig":
        """
        创建配置副本，可选覆盖部分参数

        Args:
            **kwargs: 要覆盖的参数

        Returns:
            新的配置实例
        """
        config_dict = {
            "feature_names": self.feature_names.copy(),
            "ssm_state_dim": self.ssm_state_dim,
            "ssm_input_features": self.ssm_input_features.copy(),
            "use_dimension_reducer": self.use_dimension_reducer,
            "dimension_reducer_type": self.dimension_reducer_type,
            "dimension_reducer_config": self.dimension_reducer_config,
            "verbose": self.verbose,
            "version": self.version,
        }
        config_dict.update(kwargs)
        return PipelineConfig(**config_dict)

    def validate_features_exist(self, available_features: List[str]) -> None:
        """
        验证配置中的一阶特征是否都可用

        Args:
            available_features: 可用特征列表

        Raises:
            ValueError: 如果有特征不可用
        """
        available_set = set(available_features)
        # 只验证一阶特征（SSM 输入特征和用户请求的原始特征）
        features_to_check = set(self.all_calculator_features)
        missing = features_to_check - available_set

        if missing:
            raise ValueError(f"Missing features: {missing}")

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  feature_names={len(self.feature_names)} features,\n"
            f"  raw_features={self._raw_features},\n"
            f"  ssm_features={self._ssm_features},\n"
            f"  ssm_types={self.ssm_types},\n"
            f"  ssm_state_dim={self.ssm_state_dim},\n"
            f"  use_dimension_reducer={self.use_dimension_reducer}\n"
            f")"
        )
