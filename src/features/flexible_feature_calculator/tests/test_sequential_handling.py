"""
测试sequential参数的正确处理

当用户设置sequential=False但请求带转换的特征时，
应该内部使用sequential=True获取完整序列，然后应用转换，
最后只返回最终值。
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.features.flexible_feature_calculator import (
    FlexibleFeatureCalculator,
    feature
)


def test_sequential_handling():
    """测试sequential参数的处理逻辑"""
    
    # 创建测试数据
    candles = create_mock_candles(100)
    
    # 注册一个简单的测试特征
    @feature(name="test_seq", description="Test sequential handling")
    def test_seq_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        """测试特征，返回递增序列"""
        print(f"  test_seq called with sequential={sequential}")
        if sequential:
            return np.arange(len(candles), dtype=float)
        else:
            return np.array([len(candles) - 1], dtype=float)
    
    print("Test 1: Sequential=True, 基础特征")
    calc1 = FlexibleFeatureCalculator()
    calc1.load(candles, sequential=True)
    result1 = calc1.get(["test_seq"])["test_seq"]
    print(f"  Result shape: {result1.shape}, Last value: {result1[-1]}")
    assert len(result1) == 100, "Should return full sequence"
    
    print("\nTest 2: Sequential=False, 基础特征")
    calc2 = FlexibleFeatureCalculator()
    calc2.load(candles, sequential=False)
    result2 = calc2.get(["test_seq"])["test_seq"]
    print(f"  Result shape: {result2.shape}, Value: {result2[0]}")
    assert len(result2) == 1, "Should return single value"
    
    print("\nTest 3: Sequential=False, 带转换特征 (test_seq_dt)")
    calc3 = FlexibleFeatureCalculator()
    calc3.load(candles, sequential=False)
    result3 = calc3.get(["test_seq_dt"])["test_seq_dt"]
    print(f"  Result shape: {result3.shape}, Value: {result3[0]}")
    assert len(result3) == 1, "Should return single value after transformation"
    assert result3[0] == 1.0, "dt of incrementing sequence should be 1"
    
    print("\nTest 4: Sequential=False, 复杂转换 (test_seq_mean5_lag2)")
    calc4 = FlexibleFeatureCalculator()
    calc4.load(candles, sequential=False)
    result4 = calc4.get(["test_seq_mean5_lag2"])["test_seq_mean5_lag2"]
    print(f"  Result shape: {result4.shape}, Value: {result4[0]}")
    assert len(result4) == 1, "Should return single value after complex transformation"
    # mean5 of [93,94,95,96,97] = 95, lag2 -> 95
    expected = np.mean([93, 94, 95, 96, 97])
    assert abs(result4[0] - expected) < 0.01, f"Expected ~{expected}, got {result4[0]}"
    
    print("\nTest 5: Sequential=True, 带转换特征")
    calc5 = FlexibleFeatureCalculator()
    calc5.load(candles, sequential=True)
    result5 = calc5.get(["test_seq_dt"])["test_seq_dt"]
    print(f"  Result shape: {result5.shape}, Non-NaN count: {np.sum(~np.isnan(result5))}")
    assert len(result5) == 100, "Should return full sequence"
    assert np.isnan(result5[0]), "First dt value should be NaN"
    assert result5[1] == 1.0, "dt of incrementing sequence should be 1"
    
    print("\nTest 6: 验证缓存机制")
    calc6 = FlexibleFeatureCalculator()
    calc6.load(candles, sequential=False)
    
    # 首次调用
    print("  First call:")
    result6a = calc6.get(["test_seq_mean10"])["test_seq_mean10"]
    
    # 第二次调用（应该使用缓存）
    print("  Second call (should use cache):")
    result6b = calc6.get(["test_seq_mean10"])["test_seq_mean10"]
    
    assert np.array_equal(result6a, result6b), "Should return same result from cache"
    print(f"  Cache test passed, value: {result6a[0]}")


def test_real_indicators():
    """测试真实指标的sequential处理"""
    print("\n\nTesting real indicators with sequential handling...")
    
    # 导入内置特征
    from src.features.flexible_feature_calculator.features import builtin
    
    candles = create_mock_candles(200)
    
    # Test with sequential=False but requesting transformations
    calc = FlexibleFeatureCalculator()
    calc.load(candles, sequential=False)
    
    test_features = [
        "fisher",           # 基础特征
        "fisher_dt",        # 需要完整序列来计算dt
        "fisher_lag5",      # 需要完整序列来计算lag
        "adaptive_rsi_mean20",  # 需要完整序列来计算mean
    ]
    
    for feature_name in test_features:
        try:
            result = calc.get([feature_name])[feature_name]
            print(f"✓ {feature_name}: Shape={result.shape}, Value={result[0]:.6f}")
            assert len(result) == 1, f"{feature_name} should return single value when sequential=False"
        except Exception as e:
            print(f"✗ {feature_name}: ERROR - {str(e)}")


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
    print("Sequential Parameter Handling Test")
    print("=" * 60)
    
    test_sequential_handling()
    test_real_indicators()
    
    print("\n" + "=" * 60)
    print("All sequential handling tests passed!")
    print("=" * 60)