from typing import Any, Dict, List, Optional, Union

import numpy as np
from jesse import helpers

from .registry import FeatureRegistry, get_global_registry
from .transformations import TransformationPipeline


class FlexibleFeatureCalculator:
    """灵活的特征计算器"""

    def __init__(
        self, registry: Optional[FeatureRegistry] = None, load_buildin: bool = True
    ):
        """
        初始化特征计算器

        Args:
            registry: 特征注册中心，如果不提供则使用全局注册中心
        """
        self.registry = registry or get_global_registry()
        self.pipeline = TransformationPipeline()
        self.candles: Optional[np.ndarray] = None
        self.sequential: bool = False
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_class_instances: Dict[str, Any] = {}

        if load_buildin:
            from .features import builtin  # noqa

    def load(self, candles: np.ndarray, sequential: bool = False) -> None:
        """
        加载K线数据

        Args:
            candles: K线数据
            sequential: 是否返回序列数据
        """
        self.candles = helpers.slice_candles(candles, sequential)
        self.sequential = sequential
        # 清空缓存
        self.cache.clear()
        self.cache_class_instances.clear()

    def get(self, features: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        获取特征

        Args:
            features: 特征名称或特征名称列表

        Returns:
            特征字典
        """
        if isinstance(features, str):
            features = [features]

        result = {}
        for feature_name in features:
            if feature_name in self.cache:
                result[feature_name] = self.cache[feature_name]
            else:
                # 先检查是否包含转换
                base_name, transformations = self._parse_feature_with_transformations(
                    feature_name
                )

                if transformations:
                    # 如果有转换链，需要强制使用sequential=True来获取完整序列
                    force_sequential = not self.sequential

                    # 获取基础特征值（强制sequential以进行转换）
                    base_value = self._compute_base_feature(
                        base_name, force_sequential=force_sequential
                    )

                    # 应用转换链
                    transformed_value = self.pipeline.apply_transformations(
                        base_value, transformations
                    )

                    # 如果用户要求非sequential，只取最后一个值
                    if not self.sequential:
                        transformed_value = (
                            transformed_value[-1:]
                            if len(transformed_value.shape) == 1
                            else transformed_value[-1:, :]
                        )

                    self.cache[feature_name] = transformed_value
                    result[feature_name] = transformed_value
                else:
                    # 没有转换链，直接计算基础特征
                    try:
                        base_value = self._compute_base_feature(feature_name)
                        self.cache[feature_name] = base_value
                        result[feature_name] = base_value
                    except ValueError:
                        # 特征不存在
                        raise ValueError(
                            f"Feature '{feature_name}' not found in registry"
                        )

        # 处理sequential参数（对于没有转换的特征）
        if not self.sequential:
            for key in result:
                # 只处理不是已经处理过的转换特征
                if "_" not in key or not any(
                    t in key for t in self.pipeline._transformers.keys()
                ):
                    v = result[key]
                    result[key] = (
                        v[-1:]
                        if isinstance(v, (np.ndarray, list)) and len(v) > 1
                        else (np.array([v]) if not isinstance(v, np.ndarray) else v)
                    )

        return result

    def _parse_feature_with_transformations(self, feature_name: str):
        """解析带转换的特征名"""
        parts = feature_name.split("_")
        transformations = []
        base_parts = []

        # 从左到右解析，先找基础特征，再找转换
        i = 0
        while i < len(parts):
            # 构建潜在的基础特征名
            potential_base = "_".join(parts[: i + 1])

            # 检查后续是否包含转换器
            has_transformer = False
            if i + 1 < len(parts):
                # 检查下一个部分是否是转换器
                next_part = parts[i + 1]
                if next_part in self.pipeline._transformers:
                    has_transformer = True
                # 或者是带参数的转换器（如mean20）
                for transformer in self.pipeline._transformers:
                    if next_part.startswith(transformer):
                        has_transformer = True
                        break

            # 如果找到了转换器，当前位置是基础特征的结束
            if has_transformer:
                base_parts = parts[: i + 1]
                # 解析剩余部分作为转换链
                remaining = "_".join(parts[i + 1 :])
                if remaining:
                    _, transformations = self.pipeline.parse_transformations(
                        f"dummy_{remaining}"
                    )
                break

            # 如果是有效的基础特征且后面没有更多部分，这就是基础特征
            if self._is_valid_base_feature(potential_base) and i == len(parts) - 1:
                base_parts = parts
                transformations = []
                break

            i += 1

        # 如果没找到基础特征，尝试使用原始的解析方法
        if not base_parts:
            return self.pipeline.parse_transformations(feature_name)

        base_name = "_".join(base_parts)
        return base_name, transformations

    def _is_valid_base_feature(self, feature_name: str) -> bool:
        """检查是否是有效的基础特征"""
        # 检查缓存
        if feature_name in self.cache:
            return True

        # 直接检查注册中心
        spec = self.registry.get(feature_name)
        if spec is not None:
            return True

        # 检查是否是多列特征的索引形式
        parts = feature_name.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            potential_base = "_".join(parts[:-1])
            potential_spec = self.registry.get(potential_base)
            if potential_spec and potential_spec.returns_multiple:
                return True

        return False

    def _compute_base_feature(
        self, feature_name: str, force_sequential: bool = False
    ) -> np.ndarray:
        """计算基础特征值

        Args:
            feature_name: 特征名称
            force_sequential: 是否强制使用sequential=True（用于需要后续转换的特征）
        """
        # 对于强制sequential的情况，使用不同的缓存键
        cache_key = (
            f"{feature_name}_seq"
            if force_sequential and not self.sequential
            else feature_name
        )

        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 直接尝试从注册中心获取特征
        spec = self.registry.get(feature_name)
        index = None
        params = {}

        if spec is None:
            # 如果直接获取失败，尝试解析多列索引（如 vmd_32_0 中的最后一个数字）
            # 但只有当基础特征标记为 returns_multiple=True 时才这样做
            parts = feature_name.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                potential_base = "_".join(parts[:-1])
                potential_spec = self.registry.get(potential_base)

                # 只有当特征存在且返回多列时，才将最后的数字作为索引
                if potential_spec and potential_spec.returns_multiple:
                    spec = potential_spec
                    index = int(parts[-1])

        if spec is None:
            raise ValueError(f"Feature '{feature_name}' not found in registry")

        # 计算特征
        if spec.feature_type == "function":
            result = self._compute_function_feature(
                spec, params, index, force_sequential
            )
        else:  # class
            result = self._compute_class_feature(spec, params, index, force_sequential)

        # 缓存结果
        self.cache[cache_key] = result
        return result

    def _compute_function_feature(
        self,
        spec,
        params: Dict[str, Any],
        index: Optional[int] = None,
        force_sequential: bool = False,
    ) -> np.ndarray:
        """计算函数型特征

        Args:
            spec: 特征规格
            params: 参数
            index: 多列索引
            force_sequential: 强制使用sequential=True（用于需要后续转换的特征）
        """
        # 合并默认参数和传入参数
        final_params = {**spec.params, **params}

        # 决定是否使用sequential
        # 如果需要强制sequential或者用户本身就要求sequential，则使用True
        use_sequential = force_sequential or self.sequential

        # 对于多列特征，检查是否已经计算过完整结果
        if spec.returns_multiple:
            # 生成基础缓存键（不带索引）
            base_cache_key = f"{spec.name}_full_seq{use_sequential}"

            if base_cache_key not in self.cache:
                # 计算完整的多列结果
                full_result = spec.func(
                    self.candles, sequential=use_sequential, **final_params
                )

                # 缓存所有列
                if isinstance(full_result, np.ndarray) and full_result.ndim > 1:
                    # 缓存完整结果
                    self.cache[base_cache_key] = full_result
                    # 缓存每一列
                    for col_idx in range(full_result.shape[1]):
                        col_cache_key = f"{spec.name}_{col_idx}"
                        if force_sequential and not self.sequential:
                            col_cache_key = f"{col_cache_key}_seq"
                        self.cache[col_cache_key] = full_result[:, col_idx]
                elif isinstance(full_result, tuple):
                    # 处理元组返回
                    self.cache[base_cache_key] = full_result
                    for col_idx, col_data in enumerate(full_result):
                        col_cache_key = f"{spec.name}_{col_idx}"
                        if force_sequential and not self.sequential:
                            col_cache_key = f"{col_cache_key}_seq"
                        self.cache[col_cache_key] = col_data

            # 从缓存中获取指定列
            if index is not None:
                col_cache_key = f"{spec.name}_{index}"
                if force_sequential and not self.sequential:
                    col_cache_key = f"{col_cache_key}_seq"
                if col_cache_key in self.cache:
                    return self.cache[col_cache_key]
                else:
                    # 从完整结果中提取
                    full_result = self.cache[base_cache_key]
                    if isinstance(full_result, tuple):
                        return full_result[index]
                    elif isinstance(full_result, np.ndarray) and full_result.ndim > 1:
                        return full_result[:, index]
            else:
                # 返回完整结果
                return self.cache[base_cache_key]
        else:
            # 单列特征，直接计算
            result = spec.func(self.candles, sequential=use_sequential, **final_params)
            return result

    def _compute_class_feature(
        self,
        spec,
        params: Dict[str, Any],
        index: Optional[int] = None,
        force_sequential: bool = False,
    ) -> np.ndarray:
        """计算类型特征

        Args:
            spec: 特征规格
            params: 参数
            index: 多列索引
            force_sequential: 强制使用sequential=True（用于需要后续转换的特征）
        """
        # 决定是否使用sequential
        use_sequential = force_sequential or self.sequential

        # 创建类实例的唯一键（包含sequential状态）
        instance_key = f"{spec.name}_{str(params)}_seq{use_sequential}"

        # 对于多列特征，使用特殊的缓存策略
        if spec.returns_multiple:
            # 生成基础缓存键（不带索引）
            base_cache_key = f"{spec.name}_full_seq{use_sequential}"

            if base_cache_key not in self.cache:
                # 检查是否已有实例
                if instance_key not in self.cache_class_instances:
                    # 合并默认参数和传入参数
                    final_params = {**spec.params, **params}
                    # 创建类实例
                    instance = spec.cls(
                        self.candles, sequential=use_sequential, **final_params
                    )
                    self.cache_class_instances[instance_key] = instance
                else:
                    instance = self.cache_class_instances[instance_key]

                # 获取完整结果
                if hasattr(instance, "res"):
                    full_result = instance.res()
                elif hasattr(instance, "result"):
                    full_result = instance.result()
                elif hasattr(instance, "get"):
                    full_result = instance.get()
                else:
                    raise ValueError(
                        f"Class feature {spec.name} doesn't have a result method"
                    )

                # 缓存所有列
                if isinstance(full_result, np.ndarray) and full_result.ndim > 1:
                    # 缓存完整结果
                    self.cache[base_cache_key] = full_result
                    # 缓存每一列
                    for col_idx in range(full_result.shape[1]):
                        col_cache_key = f"{spec.name}_{col_idx}"
                        if force_sequential and not self.sequential:
                            col_cache_key = f"{col_cache_key}_seq"
                        self.cache[col_cache_key] = full_result[:, col_idx]
                elif isinstance(full_result, tuple):
                    # 处理元组返回
                    self.cache[base_cache_key] = full_result
                    for col_idx, col_data in enumerate(full_result):
                        col_cache_key = f"{spec.name}_{col_idx}"
                        if force_sequential and not self.sequential:
                            col_cache_key = f"{col_cache_key}_seq"
                        self.cache[col_cache_key] = col_data

            # 从缓存中获取指定列
            if index is not None:
                col_cache_key = f"{spec.name}_{index}"
                if force_sequential and not self.sequential:
                    col_cache_key = f"{col_cache_key}_seq"
                if col_cache_key in self.cache:
                    return self.cache[col_cache_key]
                else:
                    # 从完整结果中提取
                    full_result = self.cache[base_cache_key]
                    if isinstance(full_result, tuple):
                        return full_result[index]
                    elif isinstance(full_result, np.ndarray) and full_result.ndim > 1:
                        return full_result[:, index]
            else:
                # 返回完整结果
                return self.cache[base_cache_key]
        else:
            # 单列特征的处理
            # 检查是否已有实例
            if instance_key not in self.cache_class_instances:
                # 合并默认参数和传入参数
                final_params = {**spec.params, **params}
                # 创建类实例
                instance = spec.cls(
                    self.candles, sequential=use_sequential, **final_params
                )
                self.cache_class_instances[instance_key] = instance
            else:
                instance = self.cache_class_instances[instance_key]

            # 获取结果
            if hasattr(instance, "res"):
                result = instance.res()
            elif hasattr(instance, "result"):
                result = instance.result()
            elif hasattr(instance, "get"):
                result = instance.get()
            else:
                raise ValueError(
                    f"Class feature {spec.name} doesn't have a result method"
                )

            return result

    def register_feature(
        self,
        name: str,
        func: Optional[callable] = None,
        cls: Optional[type] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        动态注册特征

        Args:
            name: 特征名称
            func: 函数型特征的计算函数
            cls: 类型特征的类
            params: 默认参数
            description: 特征描述
            returns_multiple: 是否返回多列
            aliases: 别名列表
        """
        if func is not None:
            self.registry.register_function(
                name, func, params, description, returns_multiple, aliases
            )
        elif cls is not None:
            self.registry.register_class(
                name, cls, params, description, returns_multiple, aliases
            )
        else:
            raise ValueError("Either func or cls must be provided")

    def list_features(
        self, feature_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        列出所有已注册的特征

        Args:
            feature_type: 特征类型过滤 ("function" 或 "class")

        Returns:
            特征信息字典
        """
        return self.registry.list_features(feature_type)

    def register_transformer(self, name: str, func: callable) -> None:
        """
        注册自定义转换器

        Args:
            name: 转换器名称
            func: 转换函数
        """
        self.pipeline.register_transformer(name, func)

    def remove_features(self, features: Union[str, List[str]]) -> None:
        """
        从缓存中移除指定的特征

        Args:
            features: 要移除的特征名称或特征名称列表

        Raises:
            ValueError: 如果指定的特征不在缓存中
        """
        if isinstance(features, str):
            features = [features]

        for feature_name in features:
            # 检查特征是否存在于缓存中
            if feature_name not in self.cache:
                raise ValueError(f"Feature '{feature_name}' not found in cache")

            # 移除缓存中的特征
            del self.cache[feature_name]

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.cache_class_instances.clear()
