"""
测试多列特征的缓存机制
确保返回多列的特征只计算一次，而不是每列都重新计算
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.features.flexible_feature_calculator import (
    FlexibleFeatureCalculator,
    class_feature
)


# 全局计数器，用于跟踪计算次数
computation_count = 0


@class_feature(name="test_multi", returns_multiple=True, description="Test multi-column caching")
class TestMultiFeature:
    """测试多列特征，返回5列数据"""
    
    def __init__(self, candles: np.ndarray, sequential: bool = False, **kwargs):
        self.candles = candles
        self.sequential = sequential
        
    def res(self):
        global computation_count
        computation_count += 1
        print(f"  Computing TestMultiFeature... (computation #{computation_count})")
        
        # 返回5列数据
        result = np.zeros((len(self.candles), 5))
        for i in range(5):
            result[:, i] = np.arange(len(self.candles)) * (i + 1)
        return result


def test_multicolumn_caching():
    """测试多列特征的缓存"""
    global computation_count
    
    print("Test 1: 多列特征缓存测试")
    print("-" * 40)
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=True)
    
    # 重置计数器
    computation_count = 0
    
    print("\n请求 test_multi_0:")
    result0 = calc.get(["test_multi_0"])["test_multi_0"]
    print(f"  Result shape: {result0.shape}, Last value: {result0[-1]}")
    
    print("\n请求 test_multi_1:")
    result1 = calc.get(["test_multi_1"])["test_multi_1"]
    print(f"  Result shape: {result1.shape}, Last value: {result1[-1]}")
    
    print("\n请求 test_multi_2:")
    result2 = calc.get(["test_multi_2"])["test_multi_2"]
    print(f"  Result shape: {result2.shape}, Last value: {result2[-1]}")
    
    print("\n请求 test_multi_3:")
    result3 = calc.get(["test_multi_3"])["test_multi_3"]
    print(f"  Result shape: {result3.shape}, Last value: {result3[-1]}")
    
    print("\n请求 test_multi_4:")
    result4 = calc.get(["test_multi_4"])["test_multi_4"]
    print(f"  Result shape: {result4.shape}, Last value: {result4[-1]}")
    
    print(f"\n总计算次数: {computation_count}")
    
    if computation_count > 1:
        print("❌ 问题：多列特征被重复计算了！")
        print("   期望：应该只计算一次，然后缓存所有列")
    else:
        print("✅ 优秀：多列特征只计算了一次，所有列都被缓存")
    
    # 验证结果正确性
    assert result0[-1] == 99 * 1, f"Column 0 incorrect: {result0[-1]}"
    assert result1[-1] == 99 * 2, f"Column 1 incorrect: {result1[-1]}"
    assert result2[-1] == 99 * 3, f"Column 2 incorrect: {result2[-1]}"
    assert result3[-1] == 99 * 4, f"Column 3 incorrect: {result3[-1]}"
    assert result4[-1] == 99 * 5, f"Column 4 incorrect: {result4[-1]}"
    print("✅ 所有列的值都正确")
    
    return computation_count == 1


def test_batch_request():
    """测试批量请求多列"""
    global computation_count
    
    print("\n\nTest 2: 批量请求多列特征")
    print("-" * 40)
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=True)
    
    # 重置计数器
    computation_count = 0
    
    print("\n批量请求所有列:")
    features = calc.get([
        "test_multi_0",
        "test_multi_1", 
        "test_multi_2",
        "test_multi_3",
        "test_multi_4"
    ])
    
    for i in range(5):
        key = f"test_multi_{i}"
        print(f"  {key}: Last value = {features[key][-1]}")
    
    print(f"\n总计算次数: {computation_count}")
    
    if computation_count > 1:
        print("❌ 问题：批量请求时多列特征被重复计算")
    else:
        print("✅ 优秀：批量请求时也只计算一次")
    
    return computation_count == 1


def test_with_transformations():
    """测试带转换的多列特征"""
    global computation_count
    
    print("\n\nTest 3: 带转换的多列特征缓存")
    print("-" * 40)
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=False)  # 使用 sequential=False
    
    # 重置计数器
    computation_count = 0
    
    print("\n请求带转换的多列特征:")
    result0_dt = calc.get(["test_multi_0_dt"])["test_multi_0_dt"]
    print(f"  test_multi_0_dt: Shape={result0_dt.shape}, Value={result0_dt[0]}")
    
    result1_lag5 = calc.get(["test_multi_1_lag5"])["test_multi_1_lag5"]
    print(f"  test_multi_1_lag5: Shape={result1_lag5.shape}, Value={result1_lag5[0]}")
    
    print(f"\n总计算次数: {computation_count}")
    print("注：带转换的特征需要分别计算基础特征，这是正常的")
    
    # 现在请求不带转换的列
    print("\n请求不带转换的列（应该使用缓存）:")
    computation_count_before = computation_count
    
    result2 = calc.get(["test_multi_2"])["test_multi_2"]
    print(f"  test_multi_2: Value={result2[0]}")
    
    if computation_count > computation_count_before:
        print("❌ 问题：即使有缓存也重新计算了")
    else:
        print("✅ 可能需要改进：理想情况下，如果之前计算过完整结果，应该缓存所有列")


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
    print("Multi-Column Feature Caching Test")
    print("=" * 60)
    
    test1_passed = test_multicolumn_caching()
    test2_passed = test_batch_request()
    test_with_transformations()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✅ 多列特征缓存机制工作正常！")
    else:
        print("⚠️ 多列特征缓存需要优化")
    print("=" * 60)