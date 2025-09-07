"""
示例：如何注册和使用特征

这个文件展示了如何使用新的简化特征计算器
"""

import jesse.indicators as ta
import numpy as np
from jesse.indicators import aroon
from jesse.indicators.aroon import AROON

from src.features.simple_feature_calculator import (
    SimpleFeatureCalculator,
    feature,
    class_feature,
)
from src.indicators.prod import VMD_NRBO, adaptive_rsi


# ============ 示例1：注册简单函数特征 ============


@feature(name="rsi_14", params={"period": 14}, description="RSI with period 14")
def calculate_rsi(
    candles: np.ndarray, sequential: bool = True, period: int = 14
) -> np.ndarray:
    """计算RSI指标"""
    result = ta.rsi(candles, period=period, sequential=sequential)
    # 确保返回numpy array
    if not isinstance(result, np.ndarray):
        result = np.array([result])
    return result


@feature(name="sma_20", params={"period": 20}, description="Simple Moving Average 20")
def calculate_sma(
    candles: np.ndarray, sequential: bool = True, period: int = 20
) -> np.ndarray:
    """计算简单移动平均"""
    result = ta.sma(candles, period=period, sequential=sequential)
    if not isinstance(result, np.ndarray):
        result = np.array([result])
    return result


# ============ 示例2：注册带多参数的特征 ============


@feature(
    name="bb_20_2",
    params={"period": 20, "devup": 2, "devdown": 2},
    description="Bollinger Bands (20, 2)",
)
def calculate_bb(
    candles: np.ndarray,
    sequential: bool = True,
    period: int = 20,
    devup: float = 2,
    devdown: float = 2,
) -> np.ndarray:
    """计算布林带中轨"""
    result = ta.bollinger_bands(
        candles, period=period, devup=devup, devdown=devdown, sequential=sequential
    )
    # 返回中轨
    return result.middleband


# ============ 示例3：注册返回多列的特征 ============


@feature(name="aroon_25", params={"period": 25}, returns_multiple=True)
def calculate_aroon(
    candles: np.ndarray, sequential: bool = True, period: int = 25
) -> np.ndarray:
    """计算Aroon指标，返回up和down两列"""
    aroon_result: AROON = aroon(candles, period=period, sequential=sequential)

    if sequential:
        # 返回2D数组，每行包含[up, down]
        result = np.column_stack([aroon_result.up, aroon_result.down])
    else:
        # sequential=False时，返回1D数组[up_value, down_value]
        result = np.array([aroon_result.up, aroon_result.down])

    return result


# ============ 示例4：注册自适应指标 ============


@feature(name="adaptive_rsi", description="Adaptive RSI")
def calculate_adaptive_rsi(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    """自适应RSI"""
    result = adaptive_rsi(candles, sequential=sequential)
    if not isinstance(result, np.ndarray):
        result = np.array([result])
    return result


# ============ 示例5：注册类型特征 ============


@class_feature(
    name="vmd_32",
    params={"alpha": 2000, "tau": 0, "K": 32, "DC": 0, "init": 1, "tol": 1e-7},
    returns_multiple=True,
    description="Variational Mode Decomposition with 32 modes",
)
class VMDFeature:
    """VMD特征包装类"""

    def __init__(
        self,
        candles: np.ndarray,
        sequential: bool = True,
        alpha: float = 2000,
        tau: float = 0,
        K: int = 32,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
    ):
        self.instance = VMD_NRBO(
            candles,
            sequential=sequential,
            alpha=alpha,
            tau=tau,
            K=K,
            DC=DC,
            init=init,
            tol=tol,
        )

    def res(self) -> np.ndarray:
        """返回VMD结果"""
        result = self.instance.res()
        # 确保返回正确格式的numpy array
        if isinstance(result, tuple):
            # 如果是元组，转换为2D数组
            result = np.column_stack(result)
        return result


# ============ 使用示例 ============


def demo_usage():
    """演示如何使用简化的特征计算器"""

    # 创建计算器实例
    calc = SimpleFeatureCalculator()

    # 假设有K线数据（这里用随机数据演示）
    candles = np.random.randn(100, 6)  # 100根K线

    # 加载数据
    calc.load(candles, sequential=True)

    # 获取单个特征
    features = calc.get("rsi_14")
    print(f"RSI shape: {features['rsi_14'].shape}")

    # 获取多个特征
    features = calc.get(["rsi_14", "sma_20", "adaptive_rsi"])
    for name, values in features.items():
        print(f"{name} shape: {values.shape}")

    # 获取带转换的特征
    features = calc.get(
        [
            "rsi_14_dt",  # RSI的一阶差分
            "sma_20_lag5",  # SMA滞后5期
            "rsi_14_mean10_dt",  # RSI的10期均值的一阶差分
        ]
    )
    for name, values in features.items():
        print(f"{name} shape: {values.shape}")

    # 获取多列特征的特定列
    features = calc.get(
        [
            "aroon_25_0",  # Aroon up
            "aroon_25_1",  # Aroon down
            "vmd_32_0",  # VMD第一个mode
            "vmd_32_5_dt",  # VMD第6个mode的一阶差分
        ]
    )
    for name, values in features.items():
        print(f"{name} shape: {values.shape}")

    # 切换到非sequential模式
    calc.load(candles, sequential=False)

    # 获取最新值
    features = calc.get(["rsi_14", "sma_20"])
    for name, values in features.items():
        print(f"{name} (latest) shape: {values.shape}, value: {values}")

    # 列出所有已注册的特征
    print("\nRegistered features:")
    for name, info in calc.list_features().items():
        print(f"  {name}: {info}")


if __name__ == "__main__":
    # 运行演示
    demo_usage()
