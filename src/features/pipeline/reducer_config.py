"""
ReducerConfig - 特征降维器配置类

定义 Reducer 的配置参数。
"""

import hashlib
import json
from typing import List, Optional

from pydantic import BaseModel

from src.features.dimensionality_reduction import ARDVAEConfig


class ReducerConfig(BaseModel):
    """
    Reducer 配置

    配置降维器的参数和行为。

    Attributes
    ----------
    reducer_type : str
        降维器类型，目前支持 "ard_vae"
    ard_vae_config : Optional[ARDVAEConfig]
        ARD-VAE 降维器配置
    input_feature_names : Optional[List[str]]
        输入特征列名（可选，用于 subset 选择）。
        如果为 None，使用输入 DataFrame 的全部列。
    verbose : bool
        是否打印进度信息
    version : str
        配置版本号

    Examples
    --------
    >>> config = ReducerConfig(
    ...     reducer_type="ard_vae",
    ...     ard_vae_config=ARDVAEConfig(max_latent_dim=32, seed=42)
    ... )
    """

    # 降维器类型
    reducer_type: str = "ard_vae"

    # ARD-VAE 配置
    ard_vae_config: Optional[ARDVAEConfig] = None

    # 输入特征列名（可选，用于 subset 选择）
    input_feature_names: Optional[List[str]] = None

    # 运行时配置
    verbose: bool = False

    # 元信息
    version: str = "1.0.0"

    @property
    def schema_hash(self) -> str:
        """
        计算配置的 schema hash，用于校验兼容性

        包含 reducer_type, input_feature_names, ard_vae_config.max_latent_dim
        """
        schema_dict = {
            "reducer_type": self.reducer_type,
            # 显式包含 input_feature_names（用于 subset 选择）
            "input_feature_names": self.input_feature_names,
        }
        if self.ard_vae_config is not None:
            schema_dict["ard_vae_config"] = {
                "max_latent_dim": self.ard_vae_config.max_latent_dim,
            }
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def save(self, path: str) -> None:
        """
        保存配置到 JSON 文件

        Args:
            path: 保存路径
        """
        ard_vae_dict = None
        if self.ard_vae_config is not None:
            ard_vae_dict = self.ard_vae_config.model_dump()

        config_dict = {
            "reducer_type": self.reducer_type,
            "ard_vae_config": ard_vae_dict,
            "input_feature_names": self.input_feature_names,
            "verbose": self.verbose,
            "version": self.version,
            "schema_hash": self.schema_hash,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReducerConfig":
        """
        从 JSON 文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            加载的配置实例
        """
        with open(path, "r") as f:
            config_dict = json.load(f)

        saved_hash = config_dict.pop("schema_hash", None)

        # 反序列化 ard_vae_config
        ard_vae_dict = config_dict.get("ard_vae_config")
        if ard_vae_dict is not None:
            config_dict["ard_vae_config"] = ARDVAEConfig(**ard_vae_dict)

        instance = cls(**config_dict)

        # 校验 schema hash
        if saved_hash is not None and instance.schema_hash != saved_hash:
            raise ValueError(
                f"Schema hash mismatch: saved={saved_hash}, computed={instance.schema_hash}. "
                "Configuration has changed. Delete old models and retrain."
            )

        return instance

    def __repr__(self) -> str:
        return (
            f"ReducerConfig(\n"
            f"  reducer_type={self.reducer_type},\n"
            f"  input_feature_names={self.input_feature_names},\n"
            f"  ard_vae_config={self.ard_vae_config},\n"
            f"  schema_hash={self.schema_hash}\n"
            f")"
        )
