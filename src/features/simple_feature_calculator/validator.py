"""
严格的特征输出验证器

核心功能：
1. 验证输出是否为numpy array
2. 验证输出形状是否符合要求
3. 不自动修正，只报错并提供详细信息
"""

from typing import Any

import numpy as np


class FeatureOutputValidator:
    """特征输出验证器"""

    @staticmethod
    def validate(
        output: Any,
        feature_name: str,
        candles_length: int,
        sequential: bool,
        returns_multiple: bool = False,
    ) -> None:
        """
        验证特征输出格式

        Args:
            output: 特征计算的输出
            feature_name: 特征名称（用于错误信息）
            candles_length: 输入K线的长度
            sequential: 是否要求序列输出
            returns_multiple: 是否声明返回多列

        Raises:
            TypeError: 输出类型不正确
            ValueError: 输出形状不符合要求
        """
        # 检查是否为numpy array
        if not isinstance(output, np.ndarray):
            raise TypeError(
                f"Feature '{feature_name}' output validation failed:\n"
                f"  Expected: numpy.ndarray\n"
                f"  Got: {type(output).__name__}\n"
                f"  Value: {output}\n"
                f"  Fix: Ensure the feature function returns a numpy array, "
                f"not {type(output).__name__}. "
                f"Use np.array() to convert if necessary."
            )

        # 获取输出形状信息
        shape = output.shape
        ndim = output.ndim
        length = shape[0] if len(shape) > 0 else 0

        if sequential:
            # sequential=True 的验证规则
            if ndim == 1:
                # 一维数组：长度必须等于candles长度
                if length != candles_length:
                    raise ValueError(
                        f"Feature '{feature_name}' output validation failed:\n"
                        f"  Mode: sequential=True (single column)\n"
                        f"  Expected shape: ({candles_length},)\n"
                        f"  Got shape: {shape}\n"
                        f"  Fix: The output length must equal candles length when sequential=True.\n"
                        f"  Consider using np.full({candles_length}, np.nan) and filling values."
                    )
            elif ndim == 2:
                # 二维数组：行数必须等于candles长度
                if length != candles_length:
                    raise ValueError(
                        f"Feature '{feature_name}' output validation failed:\n"
                        f"  Mode: sequential=True (multiple columns)\n"
                        f"  Expected shape: ({candles_length}, N)\n"
                        f"  Got shape: {shape}\n"
                        f"  Fix: The output rows must equal candles length when sequential=True.\n"
                        f"  Each column should have {candles_length} values."
                    )
                # 如果声明为单列但返回多列，警告
                if not returns_multiple and shape[1] > 1:
                    raise ValueError(
                        f"Feature '{feature_name}' output validation failed:\n"
                        f"  Mode: sequential=True\n"
                        f"  Issue: Feature returns {shape[1]} columns but wasn't registered with returns_multiple=True\n"
                        f"  Got shape: {shape}\n"
                        f"  Fix: Either register the feature with returns_multiple=True, "
                        f"or return only one column."
                    )
            else:
                # 不支持3维及以上
                raise ValueError(
                    f"Feature '{feature_name}' output validation failed:\n"
                    f"  Mode: sequential=True\n"
                    f"  Issue: Output has {ndim} dimensions (only 1D or 2D allowed)\n"
                    f"  Got shape: {shape}\n"
                    f"  Fix: Feature output must be 1D or 2D numpy array."
                )

        else:
            # sequential=False 的验证规则
            if ndim == 1:
                # 一维数组：长度必须为1（单值）或N（多列单行）
                if returns_multiple:
                    # 声明返回多列，长度可以>1
                    pass  # 任何长度都接受
                else:
                    # 单列，长度必须为1
                    if length != 1:
                        raise ValueError(
                            f"Feature '{feature_name}' output validation failed:\n"
                            f"  Mode: sequential=False (single value)\n"
                            f"  Expected shape: (1,)\n"
                            f"  Got shape: {shape}\n"
                            f"  Fix: When sequential=False and returning single column, "
                            f"output must be 1D array with length 1.\n"
                            f"  Example: np.array([value]) or array[-1:]"
                        )
            elif ndim == 2:
                # 二维数组在sequential=False时不合理
                raise ValueError(
                    f"Feature '{feature_name}' output validation failed:\n"
                    f"  Mode: sequential=False\n"
                    f"  Issue: Got 2D array with shape {shape}\n"
                    f"  Expected: 1D array with length 1 (single column) or N (multiple columns)\n"
                    f"  Fix: When sequential=False, output must be 1D array.\n"
                    f"  For multiple columns, return np.array([col1_val, col2_val, ...])"
                )
            else:
                # 不支持3维及以上
                raise ValueError(
                    f"Feature '{feature_name}' output validation failed:\n"
                    f"  Mode: sequential=False\n"
                    f"  Issue: Output has {ndim} dimensions (only 1D allowed)\n"
                    f"  Got shape: {shape}\n"
                    f"  Fix: When sequential=False, output must be 1D numpy array."
                )

    @staticmethod
    def validate_transform_input(
        data: Any, transform_name: str, feature_name: str
    ) -> None:
        """
        验证转换函数的输入

        Args:
            data: 输入数据
            transform_name: 转换函数名称
            feature_name: 特征名称

        Raises:
            TypeError: 输入类型不正确
            ValueError: 输入不满足转换要求
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Transform '{transform_name}' input validation failed:\n"
                f"  Feature: {feature_name}\n"
                f"  Expected: numpy.ndarray\n"
                f"  Got: {type(data).__name__}\n"
                f"  This is likely an internal error. "
                f"Please check the feature '{feature_name}' returns proper numpy array."
            )

        if data.ndim > 2:
            raise ValueError(
                f"Transform '{transform_name}' input validation failed:\n"
                f"  Feature: {feature_name}\n"
                f"  Issue: Input has {data.ndim} dimensions (max 2D allowed)\n"
                f"  Shape: {data.shape}\n"
                f"  Transforms only work with 1D or 2D arrays."
            )

        # 对于需要历史数据的转换，检查是否有足够的数据
        if len(data) < 2 and transform_name in ["dt", "ddt"]:
            raise ValueError(
                f"Transform '{transform_name}' validation failed:\n"
                f"  Feature: {feature_name}\n"
                f"  Issue: Not enough data points for {transform_name}\n"
                f"  Data length: {len(data)}\n"
                f"  {transform_name} requires at least 2 data points.\n"
                f"  Make sure to use sequential=True when applying transforms."
            )
