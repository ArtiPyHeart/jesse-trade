"""
简化的特征计算器

核心设计：
1. 清晰分离基础特征计算和转换
2. 简单的缓存策略
3. 严格的输出验证
"""

import sys
import time
from typing import Dict, List, Union, Optional, Tuple

import numpy as np

from src.utils.math_tools import np_array_fill_nan
from .registry import SimpleFeatureRegistry, get_global_registry
from .transforms import TransformChain
from .validator import FeatureOutputValidator


class SimpleFeatureCalculator:
    """简化的特征计算器"""

    def __init__(
        self,
        registry: Optional[SimpleFeatureRegistry] = None,
        load_buildin: bool = True,
        verbose: bool = False,
    ):
        """
        初始化计算器

        Args:
            registry: 特征注册中心，如果不提供则使用全局注册中心
            verbose: 是否显示计算进度和内存占用
        """
        self.registry = registry or get_global_registry()
        self.validator = FeatureOutputValidator()
        self.transform_chain = TransformChain()
        self.verbose = verbose

        # 状态变量
        self.candles: Optional[np.ndarray] = None
        self.sequential: bool = False
        self.cache: Dict[Tuple[str, bool], np.ndarray] = {}

        if load_buildin:
            from .buildin import features  # noqa

    def load(self, candles: np.ndarray, sequential: bool = False) -> None:
        """
        加载K线数据

        Args:
            candles: K线数据（Jesse格式）
            sequential: 是否返回序列数据
        """
        # 不截断数据，让特征自己处理
        self.candles = candles
        self.sequential = sequential
        # 清空缓存
        self.cache.clear()

    def get(self, features: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        获取特征

        Args:
            features: 特征名称或特征名称列表

        Returns:
            特征字典 {feature_name: feature_array}
        """
        if isinstance(features, str):
            features = [features]

        # 检查重复的特征名称
        if len(features) != len(set(features)):
            duplicates = [f for f in set(features) if features.count(f) > 1]
            raise ValueError(f"Duplicate feature names found: {duplicates}")

        if self.candles is None:
            raise RuntimeError("Please call load() first to load candles data")

        result = {}
        total = len(features)

        for i, feature_name in enumerate(features, 1):
            start_time = time.perf_counter()
            result[feature_name] = self._compute_feature(feature_name)
            elapsed = time.perf_counter() - start_time

            if self.verbose:
                self._print_progress(feature_name, i, total, elapsed)

        # 计算完成后换行（避免后续输出覆盖进度条）
        if self.verbose:
            print()

        return result

    def _compute_feature(self, feature_name: str) -> np.ndarray:
        """
        计算单个特征

        Args:
            feature_name: 特征名称

        Returns:
            特征数组
        """
        # 检查缓存
        cache_key = (feature_name, self.sequential)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 解析特征名，分离基础特征和转换
        base_name, transforms, column_index = self._parse_feature_name(feature_name)

        # 计算基础特征
        if transforms:
            # 如果有转换，需要先获取完整序列
            # 检查是否是类特征
            metadata = self.registry.get_metadata(base_name)
            is_class_feature = metadata and metadata.get("type") == "class"

            if is_class_feature:
                # 对于类特征，获取raw_result用于转换
                raw_result = self._compute_base_feature(
                    base_name, force_sequential=True, return_raw=True
                )
                # 处理raw_result为适合转换链的格式
                # 注意：这里暂不处理column_index，留到后面统一处理
                base_value = self._process_raw_result_for_transform(raw_result)
            else:
                # 函数型特征，正常处理
                base_value = self._compute_base_feature(
                    base_name, force_sequential=True
                )
        else:
            # 没有转换，直接按用户需求计算
            base_value = self._compute_base_feature(base_name, force_sequential=False)

        # 处理多列索引
        if column_index is not None:
            # 获取特征元信息
            metadata = self.registry.get_metadata(base_name)
            returns_multiple = metadata and metadata.get("returns_multiple", False)

            if not returns_multiple:
                # 单列特征不应该有列索引（除了_0）
                if column_index != 0:
                    raise ValueError(
                        f"Feature '{feature_name}' is not a multi-column feature, "
                        f"column index {column_index} is invalid. "
                        f"Single-column features only accept index 0."
                    )
                # 索引0对于单列特征是允许的，但不做任何操作
            else:
                # 多列特征必须是2D数组
                if base_value.ndim != 2:
                    raise ValueError(
                        f"Feature '{feature_name}' is registered as multi-column but returned {base_value.ndim}D array. "
                        f"Multi-column features must return 2D arrays. "
                        f"Got shape: {base_value.shape}"
                    )

                # 检查列索引是否有效
                if column_index >= base_value.shape[1]:
                    raise ValueError(
                        f"Feature '{feature_name}' column index {column_index} out of range. "
                        f"Feature has {base_value.shape[1]} columns."
                    )

                # 提取指定列
                base_value = base_value[:, column_index]

        # 应用转换链
        if transforms:
            # 验证转换输入
            self.validator.validate_transform_input(
                base_value, transforms[0], base_name
            )
            # 应用转换
            transformed_value = self.transform_chain.apply_chain(base_value, transforms)

            # 如果用户要求非sequential，但我们使用了sequential数据进行转换
            # 需要截取最后一行
            if not self.sequential:
                if transformed_value.ndim == 1:
                    transformed_value = transformed_value[-1:]
                elif transformed_value.ndim == 2:
                    # 对于多列特征，返回一维数组而不是 (1, n_columns) 的二维数组
                    transformed_value = transformed_value[-1, :]

            result = transformed_value
        else:
            # 没有转换的情况
            # 如果是非sequential且是多列特征，需要确保返回正确的维度
            if not self.sequential and base_value.ndim == 2:
                # 返回最后一行作为一维数组
                result = base_value[-1, :]
            else:
                result = base_value

        # 缓存结果
        self.cache[cache_key] = result
        return result

    def _process_raw_result_for_transform(self, raw_result: list) -> np.ndarray:
        """
        处理类特征的raw_result，转换为适合转换链处理的格式

        Args:
            raw_result: 类特征的raw_result列表
            column_index: 列索引（如果有）

        Returns:
            处理后的数组，适合转换链处理
        """
        if not raw_result:
            raise ValueError("raw_result is empty")

        # raw_result是一个列表，每个元素是(窗口大小, 列数)的数组
        # 我们需要从每个窗口提取最后一行，组成时间序列

        # 提取每个窗口的最后一行
        time_series = []
        for window_data in raw_result:
            if window_data.ndim == 2:
                # 多列数据，取最后一行
                last_row = window_data[-1, :]
            else:
                # 单列数据
                last_row = window_data[-1]
            time_series.append(last_row)

        # 转换为数组
        result = np.array(time_series)

        # 检查结果长度是否与candles一致，如果不一致则填充
        result = np_array_fill_nan(result, self.candles)
        # if self.candles is not None:
        #     candles_len = len(self.candles)
        #     result_len = len(result)
        #
        #     if result_len < candles_len:
        #         # 需要在前面填充 np.nan
        #         pad_len = candles_len - result_len
        #
        #         if result.ndim == 1:
        #             # 一维数组
        #             result = np.concatenate([np.full(pad_len, np.nan), result])
        #         elif result.ndim == 2:
        #             # 二维数组
        #             pad_shape = (pad_len, result.shape[1])
        #             result = np.concatenate(
        #                 [np.full(pad_shape, np.nan), result], axis=0
        #             )

        return result

    def _compute_base_feature(
        self, feature_name: str, force_sequential: bool, return_raw: bool = False
    ) -> np.ndarray:
        """
        计算基础特征（不含转换）

        Args:
            feature_name: 基础特征名
            force_sequential: 是否强制使用sequential=True
            return_raw: 是否返回raw_result（用于类特征的转换链处理）

        Returns:
            特征数组
        """
        # 检查特征是否注册
        feature_func = self.registry.get(feature_name)
        if feature_func is None:
            raise ValueError(
                f"Feature '{feature_name}' not found in registry. "
                f"Available features: {list(self.registry.list_features().keys())}"
            )

        # 获取特征元信息
        metadata = self.registry.get_metadata(feature_name)
        returns_multiple = metadata.get("returns_multiple", False)
        is_class_feature = metadata.get("type") == "class"

        # 决定使用的sequential参数
        use_sequential = force_sequential or self.sequential

        # 如果是强制sequential但用户要求非sequential，使用不同的缓存键
        # 如果是return_raw，需要特殊的缓存键
        if return_raw:
            cache_key = (f"{feature_name}_raw", use_sequential)
        elif force_sequential and not self.sequential:
            cache_key = (f"{feature_name}_seq", True)
        else:
            cache_key = (feature_name, use_sequential)

        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 计算特征
        # 对于类特征，如果需要raw数据（用于转换），传递return_raw参数
        if is_class_feature and return_raw:
            # 类特征支持return_raw参数
            output = feature_func(self.candles, use_sequential, return_raw=True)
            # raw_result是列表，不需要验证，直接返回
            self.cache[cache_key] = output
            return output
        else:
            # 普通调用
            output = feature_func(self.candles, use_sequential)

        # 验证输出格式（只对非raw_result进行验证）
        self.validator.validate(
            output=output,
            feature_name=feature_name,
            candles_length=len(self.candles),
            sequential=use_sequential,
            returns_multiple=returns_multiple,
        )

        # 缓存结果
        self.cache[cache_key] = output

        return output

    def _parse_feature_name(
        self, feature_name: str
    ) -> Tuple[str, List[str], Optional[int]]:
        """
        解析特征名称

        例如:
        - "rsi" -> ("rsi", [], None)
        - "rsi_dt" -> ("rsi", ["dt"], None)
        - "rsi_mean20_dt" -> ("rsi", ["mean20", "dt"], None)
        - "vmd_32_0" -> ("vmd_32", [], 0)
        - "vmd_32_0_dt" -> ("vmd_32", ["dt"], 0)

        Args:
            feature_name: 完整特征名

        Returns:
            (基础特征名, 转换列表, 列索引)
        """
        parts = feature_name.split("_")

        # 从右向左查找转换
        transforms = []
        i = len(parts) - 1

        while i >= 0:
            part = parts[i]
            # 检查是否是转换
            transform_name, param = self.transform_chain.parse_transform_name(part)
            if transform_name:
                transforms.insert(0, part)
                i -= 1
            else:
                break

        # 剩余部分是基础特征名（可能包含列索引）
        remaining_parts = parts[: i + 1]

        # 检查最后一部分是否是数字（列索引）
        column_index = None
        if len(remaining_parts) > 1 and remaining_parts[-1].isdigit():
            # 可能是列索引，需要验证基础特征是否存在
            potential_base = "_".join(remaining_parts[:-1])
            if self.registry.has_feature(potential_base):
                # 检查是否声明为多列
                metadata = self.registry.get_metadata(potential_base)
                if metadata and metadata.get("returns_multiple", False):
                    column_index = int(remaining_parts[-1])
                    base_name = potential_base
                else:
                    # 不是多列特征，数字是名称的一部分
                    base_name = "_".join(remaining_parts)
            else:
                # 基础特征不存在，尝试完整名称
                base_name = "_".join(remaining_parts)
        else:
            base_name = "_".join(remaining_parts)

        return base_name, transforms, column_index

    def register_feature(
        self,
        name: str,
        func: Optional[callable] = None,
        cls: Optional[type] = None,
        params: Optional[Dict] = None,
        description: str = "",
        returns_multiple: bool = False,
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
        """
        if func is not None:
            self.registry.register_function(
                name, func, params, description, returns_multiple
            )
        elif cls is not None:
            self.registry.register_class(
                name, cls, params, description, returns_multiple
            )
        else:
            raise ValueError("Either func or cls must be provided")

        # 清空缓存
        self.cache.clear()

    def list_features(self) -> Dict[str, Dict]:
        """列出所有已注册的特征"""
        return self.registry.list_features()

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()

    def _get_cache_memory_bytes(self) -> int:
        """计算缓存中所有特征数组的总内存占用（字节）"""
        total = 0
        for value in self.cache.values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
            elif isinstance(value, list):
                # raw_result 是列表
                for arr in value:
                    if isinstance(arr, np.ndarray):
                        total += arr.nbytes
        return total

    def _format_memory_size(self, bytes_size: int) -> str:
        """将字节数格式化为人类可读的形式"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} TB"

    def _print_progress(
        self,
        feature_name: str,
        current: int,
        total: int,
        elapsed_time: float,
    ) -> None:
        """打印进度信息（同一行刷新）"""
        memory = self._get_cache_memory_bytes()
        memory_str = self._format_memory_size(memory)
        progress = (
            f"[{current}/{total}] {feature_name}: {elapsed_time:.3f}s | "
            f"Memory: {memory_str}"
        )

        # 清除当前行并打印
        sys.stdout.write(f"\r{progress:<80}")
        sys.stdout.flush()
