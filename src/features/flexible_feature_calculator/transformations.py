from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numba import njit


class TransformationPipeline:
    """转换管道，处理复杂的特征转换链"""
    
    def __init__(self):
        self._transformers: Dict[str, Callable] = {}
        self._register_builtin_transformers()
    
    def _register_builtin_transformers(self):
        """注册内置转换器"""
        # 基础转换
        self._transformers["dt"] = self._dt
        self._transformers["ddt"] = self._ddt
        self._transformers["lag"] = self._lag
        
        # 统计转换
        self._transformers["mean"] = self._rolling_mean
        self._transformers["std"] = self._rolling_std
        self._transformers["sum"] = self._rolling_sum
        self._transformers["max"] = self._rolling_max
        self._transformers["min"] = self._rolling_min
        self._transformers["median"] = self._rolling_median
        self._transformers["skew"] = self._rolling_skew
        self._transformers["kurt"] = self._rolling_kurt
        
        # 比例转换
        self._transformers["pct"] = self._pct_change
        self._transformers["rank"] = self._rolling_rank
        self._transformers["zscore"] = self._rolling_zscore
    
    def register_transformer(self, name: str, func: Callable):
        """注册自定义转换器"""
        self._transformers[name] = func
    
    def parse_transformations(self, feature_name: str) -> Tuple[str, List[Tuple[str, Any]]]:
        """
        解析特征名称中的转换链
        
        例如:
        - "rsi_dt" -> ("rsi", [("dt", None)])
        - "rsi_mean20_lag5" -> ("rsi", [("mean", 20), ("lag", 5)])
        - "vmd_32_0_std10_dt_lag3" -> ("vmd_32_0", [("std", 10), ("dt", None), ("lag", 3)])
        """
        parts = feature_name.split("_")
        transformations = []
        base_parts = []
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # 检查是否是转换器名称（包括带参数的形式）
            found_transformer = False
            for transformer_name in self._transformers:
                if part == transformer_name:
                    # 精确匹配转换器名称
                    # 检查下一个部分是否是数字参数
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        transformations.append((transformer_name, int(parts[i + 1])))
                        i += 2
                    else:
                        transformations.append((transformer_name, None))
                        i += 1
                    found_transformer = True
                    break
                elif part.startswith(transformer_name) and part[len(transformer_name):].isdigit():
                    # 转换器名称和参数合并在一起（如mean20）
                    param = int(part[len(transformer_name):])
                    transformations.append((transformer_name, param))
                    i += 1
                    found_transformer = True
                    break
            
            if not found_transformer:
                # 如果还没有找到转换器，这是基础特征名的一部分
                if not transformations:
                    base_parts.append(part)
                    i += 1
                else:
                    # 已经找到转换器，但这不是转换器名，可能是错误
                    break
        
        base_name = "_".join(base_parts) if base_parts else feature_name
        return base_name, transformations
    
    def apply_transformations(
        self, 
        data: np.ndarray, 
        transformations: List[Tuple[str, Any]]
    ) -> np.ndarray:
        """应用转换链到数据"""
        result = data.copy() if isinstance(data, np.ndarray) else np.array(data)
        
        for transform_name, param in transformations:
            if transform_name in self._transformers:
                if param is not None:
                    result = self._transformers[transform_name](result, param)
                else:
                    result = self._transformers[transform_name](result)
            else:
                raise ValueError(f"Unknown transformation: {transform_name}")
        
        return result
    
    # 转换器实现
    @staticmethod
    @njit(cache=True)
    def _dt(array: np.ndarray) -> np.ndarray:
        """一阶差分"""
        res = np.empty_like(array)
        res[0] = np.nan
        res[1:] = array[1:] - array[:-1]
        return res
    
    @staticmethod
    @njit(cache=True)
    def _ddt(array: np.ndarray) -> np.ndarray:
        """二阶差分"""
        res = np.empty_like(array)
        res[0] = np.nan
        dt_array = TransformationPipeline._dt(array)
        res[1:] = dt_array[1:] - dt_array[:-1]
        return res
    
    @staticmethod
    @njit(cache=True)
    def _lag(array: np.ndarray, n: int) -> np.ndarray:
        """滞后n期"""
        result = np.full_like(array, np.nan)
        if n > 0:
            result[n:] = array[:-n]
        elif n < 0:
            result[:n] = array[-n:]
        else:
            result = array.copy()
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_mean(array: np.ndarray, window: int) -> np.ndarray:
        """滚动均值"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.mean(array[i - window + 1:i + 1])
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_std(array: np.ndarray, window: int) -> np.ndarray:
        """滚动标准差"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.std(array[i - window + 1:i + 1])
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_sum(array: np.ndarray, window: int) -> np.ndarray:
        """滚动求和"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.sum(array[i - window + 1:i + 1])
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_max(array: np.ndarray, window: int) -> np.ndarray:
        """滚动最大值"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.max(array[i - window + 1:i + 1])
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_min(array: np.ndarray, window: int) -> np.ndarray:
        """滚动最小值"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.min(array[i - window + 1:i + 1])
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_median(array: np.ndarray, window: int) -> np.ndarray:
        """滚动中位数"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.median(array[i - window + 1:i + 1])
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_skew(array: np.ndarray, window: int) -> np.ndarray:
        """滚动偏度"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1:i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std != 0:
                m3 = np.mean((window_data - mean) ** 3)
                result[i] = m3 / (std ** 3)
            else:
                result[i] = 0
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_kurt(array: np.ndarray, window: int) -> np.ndarray:
        """滚动峰度"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1:i + 1]
            mean = np.mean(window_data)
            var = np.var(window_data)
            if var != 0:
                m4 = np.mean((window_data - mean) ** 4)
                result[i] = (m4 / var ** 2) - 3
            else:
                result[i] = 0
        return result
    
    @staticmethod
    @njit(cache=True)
    def _pct_change(array: np.ndarray, periods: int = 1) -> np.ndarray:
        """百分比变化"""
        result = np.full_like(array, np.nan)
        if periods > 0:
            result[periods:] = (array[periods:] - array[:-periods]) / array[:-periods]
        return result
    
    @staticmethod
    def _rolling_rank(array: np.ndarray, window: int) -> np.ndarray:
        """滚动排名"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1:i + 1]
            # 计算当前值在窗口中的排名百分比
            rank = np.sum(window_data <= array[i]) / len(window_data)
            result[i] = rank
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_zscore(array: np.ndarray, window: int) -> np.ndarray:
        """滚动Z分数"""
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1:i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std != 0:
                result[i] = (array[i] - mean) / std
            else:
                result[i] = 0
        return result