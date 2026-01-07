"""
有状态特征缓存管理器

负责：
1. 缓存状态检查（有效/不存在/参数不匹配/版本过期）
2. 状态保存（safetensors + JSON）
3. 状态加载
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

CacheStatus = Literal["valid", "not_exists", "params_mismatch", "version_outdated"]


class StatefulCacheManager:
    """
    有状态特征缓存管理器

    缓存目录结构：
    {cache_dir}/{feature_name}/
    ├── meta.json          # 元数据（params, version, is_trained）
    └── state.safetensors  # 模型状态
    """

    def __init__(self, cache_dir: Path, feature_name: str):
        self.cache_dir = cache_dir
        self.feature_name = feature_name
        self.feature_cache_dir = cache_dir / feature_name
        self.meta_path = self.feature_cache_dir / "meta.json"
        self.state_path = self.feature_cache_dir / "state.safetensors"

        self._cached_meta: Optional[Dict] = None

    def check_cache(self, params: Dict, version: str) -> CacheStatus:
        """
        检查缓存状态

        Args:
            params: 当前特征参数
            version: 当前版本号

        Returns:
            - "valid": 缓存有效
            - "not_exists": 缓存不存在
            - "params_mismatch": 参数不匹配
            - "version_outdated": 版本过期
        """
        if not self.meta_path.exists() or not self.state_path.exists():
            return "not_exists"

        # 加载元数据
        self._load_meta()

        # 检查参数
        if self._cached_meta["params"] != params:
            return "params_mismatch"

        # 检查版本
        if self._cached_meta["version"] != version:
            return "version_outdated"

        return "valid"

    def _load_meta(self) -> None:
        """加载元数据"""
        if self._cached_meta is None:
            with open(self.meta_path, "r") as f:
                self._cached_meta = json.load(f)

    def get_cached_params(self) -> Optional[Dict]:
        """获取缓存的参数"""
        if self._cached_meta is None:
            if self.meta_path.exists():
                self._load_meta()
            else:
                return None
        return self._cached_meta.get("params") if self._cached_meta else None

    def get_cached_version(self) -> Optional[str]:
        """获取缓存的版本"""
        if self._cached_meta is None:
            if self.meta_path.exists():
                self._load_meta()
            else:
                return None
        return self._cached_meta.get("version") if self._cached_meta else None

    def load_state(self) -> Dict[str, Any]:
        """
        加载模型状态

        Returns:
            状态字典
        """
        # 优先尝试 numpy 格式
        try:
            from safetensors.numpy import load_file

            return dict(load_file(str(self.state_path)))
        except Exception:
            pass

        # 尝试 torch 格式
        try:
            from safetensors.torch import load_file as torch_load_file

            return dict(torch_load_file(str(self.state_path)))
        except Exception:
            raise RuntimeError(
                f"Failed to load state from {self.state_path}. "
                f"File may be corrupted or in unsupported format."
            )

    def save_state(
        self,
        state_dict: Dict[str, Any],
        params: Dict,
        version: str,
    ) -> None:
        """
        保存模型状态和元数据

        Args:
            state_dict: 模型状态字典
            params: 特征参数
            version: 版本号
        """
        # 确保目录存在
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        # 判断是 numpy 还是 torch
        has_torch = self._has_torch_tensors(state_dict)

        if has_torch:
            from safetensors.torch import save_file
        else:
            # 确保所有值都是 numpy array
            state_dict = self._ensure_numpy(state_dict)
            from safetensors.numpy import save_file

        # 保存状态
        save_file(state_dict, str(self.state_path))

        # 保存元数据
        meta = {
            "params": params,
            "version": version,
            "is_trained": True,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 更新内部缓存
        self._cached_meta = meta

    def _has_torch_tensors(self, state_dict: Dict[str, Any]) -> bool:
        """检查状态字典是否包含 torch tensor"""
        try:
            import torch

            return any(isinstance(v, torch.Tensor) for v in state_dict.values())
        except ImportError:
            return False

    def _ensure_numpy(self, state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """确保所有值都是 numpy array"""
        result = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                result[k] = v
            elif hasattr(v, "numpy"):
                # torch tensor
                result[k] = v.numpy()
            elif isinstance(v, (list, tuple)):
                result[k] = np.array(v)
            elif isinstance(v, (int, float)):
                result[k] = np.array([v])
            else:
                raise ValueError(
                    f"Cannot convert state_dict['{k}'] of type {type(v)} to numpy array"
                )
        return result

    def clear(self) -> None:
        """清除缓存"""
        import shutil

        if self.feature_cache_dir.exists():
            shutil.rmtree(self.feature_cache_dir)
        self._cached_meta = None
