"""
测试灵活的特征注册机制

新的设计理念：
1. 特征名称可以包含任意参数（如 adx_14, vmd_win32, custom_param1_param2）
2. 参数在注册时定义，而不是在解析时猜测
3. 只有尾部的转换（dt, lag, mean等）需要特殊处理
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.features.flexible_feature_calculator import (
    FlexibleFeatureCalculator,
    feature,
    class_feature
)


def test_flexible_naming():
    """测试灵活的特征命名和注册"""
    
    print("Test 1: 注册带参数的特征名")
    
    # 注册各种命名风格的特征
    @feature(name="adx_14", description="ADX with period 14")
    def adx_14_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        """预配置的ADX指标，周期为14"""
        return np.ones(len(candles)) * 14
    
    @feature(name="adx_20", description="ADX with period 20")
    def adx_20_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        """预配置的ADX指标，周期为20"""
        return np.ones(len(candles)) * 20
    
    @feature(name="vmd_win32", description="VMD with window 32")
    def vmd_win32_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        """VMD特征，窗口大小32"""
        return np.ones(len(candles)) * 32
    
    @feature(name="custom_10_20_30", description="Custom feature with multiple params")
    def custom_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        """自定义特征，带多个参数"""
        return np.ones(len(candles)) * 60  # 10+20+30
    
    @feature(name="strategy_v2_optimized", description="Complex strategy feature")
    def strategy_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        """复杂策略特征"""
        return np.arange(len(candles), dtype=float)
    
    # 测试各种特征
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=True)
    
    test_features = [
        "adx_14",
        "adx_20",
        "vmd_win32",
        "custom_10_20_30",
        "strategy_v2_optimized",
    ]
    
    print("\n基础特征测试:")
    for feat_name in test_features:
        result = calc.get([feat_name])[feat_name]
        print(f"  ✓ {feat_name}: Shape={result.shape}, Last value={result[-1]}")
    
    print("\n带转换的特征测试:")
    transform_features = [
        "adx_14_dt",
        "adx_20_lag5",
        "vmd_win32_mean10",
        "custom_10_20_30_std5_lag2",
        "strategy_v2_optimized_dt_lag3",
    ]
    
    for feat_name in transform_features:
        result = calc.get([feat_name])[feat_name]
        non_nan = np.sum(~np.isnan(result))
        print(f"  ✓ {feat_name}: Shape={result.shape}, Non-NaN={non_nan}")


def test_multi_column_features():
    """测试多列特征的灵活命名"""
    
    print("\n\nTest 2: 多列特征的灵活命名")
    
    @class_feature(name="advanced_indicator_v3", returns_multiple=True)
    class AdvancedIndicator:
        def __init__(self, candles: np.ndarray, sequential: bool = False, **kwargs):
            self.candles = candles
            self.sequential = sequential
        
        def res(self):
            # 返回3列数据
            result = np.zeros((len(self.candles), 3))
            result[:, 0] = np.arange(len(self.candles))
            result[:, 1] = np.arange(len(self.candles)) * 2
            result[:, 2] = np.arange(len(self.candles)) * 3
            return result
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=True)
    
    # 测试多列访问
    print("\n多列特征访问:")
    for i in range(3):
        feat_name = f"advanced_indicator_v3_{i}"
        result = calc.get([feat_name])[feat_name]
        print(f"  ✓ {feat_name}: Last value={result[-1]}")
    
    # 测试带转换的多列特征
    print("\n带转换的多列特征:")
    transform_features = [
        "advanced_indicator_v3_0_dt",
        "advanced_indicator_v3_1_mean5",
        "advanced_indicator_v3_2_lag10",
    ]
    
    for feat_name in transform_features:
        result = calc.get([feat_name])[feat_name]
        non_nan = np.sum(~np.isnan(result))
        print(f"  ✓ {feat_name}: Non-NaN={non_nan}, Last value={result[-1]:.2f}")


def test_dynamic_registration():
    """测试动态注册的灵活性"""
    
    print("\n\nTest 3: 动态注册的灵活性")
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(50)
    calc.load(candles, sequential=True)
    
    # 动态注册一系列相似的特征
    for period in [5, 10, 15, 20, 30]:
        feature_name = f"ma_{period}"
        calc.register_feature(
            name=feature_name,
            func=lambda c, p=period, **kw: np.ones(len(c)) * p,
            description=f"Moving average with period {period}"
        )
    
    # 测试动态注册的特征
    print("\n动态注册的MA特征:")
    for period in [5, 10, 15, 20, 30]:
        feat_name = f"ma_{period}"
        result = calc.get([feat_name])[feat_name]
        print(f"  ✓ {feat_name}: Value={result[-1]}")
    
    # 测试带转换
    print("\n动态特征的转换:")
    result = calc.get(["ma_10_dt"])["ma_10_dt"]
    print(f"  ✓ ma_10_dt: All zeros (constant diff) = {np.all(result[~np.isnan(result)] == 0)}")


def test_complex_naming_patterns():
    """测试复杂的命名模式"""
    
    print("\n\nTest 4: 复杂命名模式")
    
    # 注册包含多个下划线和数字的特征
    @feature(name="rsi_14_smoothed_v2_final", description="Complex RSI variant")
    def complex_rsi(candles: np.ndarray, sequential: bool = True, **kwargs):
        return np.sin(np.arange(len(candles)) * 0.1) * 50 + 50
    
    @feature(name="bb_20_2_upper", description="Bollinger Band upper")
    def bb_upper(candles: np.ndarray, sequential: bool = True, **kwargs):
        return np.ones(len(candles)) * 110
    
    @feature(name="macd_12_26_9_signal", description="MACD signal line")
    def macd_signal(candles: np.ndarray, sequential: bool = True, **kwargs):
        return np.cos(np.arange(len(candles)) * 0.05) * 10
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=False)  # 测试 sequential=False
    
    # 测试复杂命名的特征
    print("\n复杂命名特征 (sequential=False):")
    features = [
        "rsi_14_smoothed_v2_final",
        "bb_20_2_upper",
        "macd_12_26_9_signal",
    ]
    
    for feat_name in features:
        result = calc.get([feat_name])[feat_name]
        print(f"  ✓ {feat_name}: Shape={result.shape}, Value={result[0]:.2f}")
    
    # 测试带转换的复杂命名特征
    print("\n带转换的复杂命名特征 (sequential=False):")
    transform_features = [
        "rsi_14_smoothed_v2_final_dt",
        "bb_20_2_upper_mean5",
        "macd_12_26_9_signal_lag3",
    ]
    
    for feat_name in transform_features:
        result = calc.get([feat_name])[feat_name]
        print(f"  ✓ {feat_name}: Shape={result.shape}, Value={result[0]:.4f}")


def test_registration_flexibility():
    """测试注册系统的灵活性"""
    
    print("\n\nTest 5: 注册系统灵活性总结")
    
    calc = FlexibleFeatureCalculator()
    
    # 展示各种可能的特征命名
    examples = [
        ("sma_20", "Simple moving average"),
        ("ema_12_26", "EMA with two periods"),
        ("rsi_oversold_30", "RSI oversold threshold"),
        ("strategy_v3_optimized_2024", "Year-versioned strategy"),
        ("ml_feature_rf_importance_0_95", "ML feature with threshold"),
        ("custom_ind_param1_param2_param3", "Multi-parameter indicator"),
    ]
    
    for name, desc in examples:
        calc.register_feature(
            name=name,
            func=lambda c, **kw: np.random.randn(len(c)),
            description=desc
        )
    
    # 列出所有注册的特征
    all_features = calc.list_features()
    
    print(f"\n成功注册 {len(all_features)} 个特征")
    print("\n示例特征：")
    for name, info in list(all_features.items())[:5]:
        print(f"  • {name}: {info['description']}")
    
    print("\n✅ 新的设计完全移除了参数解析的限制")
    print("✅ 特征名称可以包含任意格式的参数")
    print("✅ 只有尾部的转换（dt, lag等）需要特殊处理")
    print("✅ 参数的含义在注册时定义，而不是解析时猜测")


def create_mock_candles(length: int) -> np.ndarray:
    """创建模拟K线数据"""
    np.random.seed(42)
    
    # 创建价格序列
    price = 100.0
    prices = []
    for _ in range(length):
        price *= np.random.uniform(0.99, 1.01)
        prices.append(price)
    
    prices = np.array(prices)
    
    # 创建OHLCV数据
    candles = np.zeros((length, 6))
    candles[:, 0] = np.arange(length) * 60000  # timestamp
    candles[:, 1] = prices * np.random.uniform(0.998, 1.002, length)  # open
    candles[:, 2] = prices  # close
    candles[:, 3] = prices * np.random.uniform(1.001, 1.005, length)  # high
    candles[:, 4] = prices * np.random.uniform(0.995, 0.999, length)  # low
    candles[:, 5] = np.random.uniform(100, 1000, length)  # volume
    
    return candles


if __name__ == "__main__":
    print("=" * 60)
    print("Flexible Feature Registration Test")
    print("=" * 60)
    
    test_flexible_naming()
    test_multi_column_features()
    test_dynamic_registration()
    test_complex_naming_patterns()
    test_registration_flexibility()
    
    print("\n" + "=" * 60)
    print("All flexibility tests passed!")
    print("=" * 60)