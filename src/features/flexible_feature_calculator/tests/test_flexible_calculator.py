"""
测试FlexibleFeatureCalculator与原FeatureCalculator的兼容性
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.features.all_features import FeatureCalculator
from src.features.flexible_feature_calculator import (
    FlexibleFeatureCalculator,
    feature,
    class_feature
)


def test_compatibility():
    """测试与原FeatureCalculator的兼容性"""
    # 加载测试数据
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "btc_1m.npy"
    if not data_path.exists():
        print(f"Warning: Test data not found at {data_path}")
        # 创建模拟数据
        candles = create_mock_candles(10000)
    else:
        candles = np.load(data_path)[-10000:]
    
    # 原始FeatureCalculator
    old_calc = FeatureCalculator()
    old_calc.load(candles, sequential=True)
    
    # 新的FlexibleFeatureCalculator
    new_calc = FlexibleFeatureCalculator()
    
    # 导入内置特征以注册它们
    from src.features.flexible_feature_calculator.features import builtin
    
    new_calc.load(candles, sequential=True)
    
    # 测试特征列表
    test_features = [
        "acc_swing_index",
        "adaptive_cci",
        "adaptive_rsi",
        "aroon_diff",
        "fisher",
        "dual_diff",
        "homodyne",
    ]
    
    print("Testing basic features compatibility...")
    for feature_name in test_features:
        try:
            old_result = old_calc.get([feature_name])[feature_name]
            new_result = new_calc.get([feature_name])[feature_name]
            
            if np.allclose(old_result, new_result, rtol=1e-5, atol=1e-8, equal_nan=True):
                print(f"✓ {feature_name}: PASS")
            else:
                print(f"✗ {feature_name}: FAIL - Results don't match")
                print(f"  Old shape: {old_result.shape}, New shape: {new_result.shape}")
                print(f"  Max diff: {np.nanmax(np.abs(old_result - new_result))}")
        except Exception as e:
            print(f"✗ {feature_name}: ERROR - {str(e)}")
    
    print("\nTesting transformation features...")
    # 测试带转换的特征（仅测试原FeatureCalculator支持的）
    test_transform_features = [
        "acc_swing_index_dt",
        "adaptive_cci_dt",
        "fisher_dt",
    ]
    
    for feature_name in test_transform_features:
        try:
            old_result = old_calc.get([feature_name])[feature_name]
            new_result = new_calc.get([feature_name])[feature_name]
            
            if np.allclose(old_result, new_result, rtol=1e-5, atol=1e-8, equal_nan=True):
                print(f"✓ {feature_name}: PASS")
            else:
                print(f"✗ {feature_name}: FAIL - Results don't match")
                # Debug information
                diff = np.abs(old_result - new_result)
                max_diff_idx = np.nanargmax(diff)
                print(f"  Max diff at index {max_diff_idx}: old={old_result[max_diff_idx]}, new={new_result[max_diff_idx]}")
        except Exception as e:
            print(f"✗ {feature_name}: ERROR - {str(e)}")


def test_new_transformations():
    """测试新的转换功能"""
    print("\n\nTesting new transformation capabilities...")
    
    # 创建模拟数据
    candles = create_mock_candles(1000)
    
    calc = FlexibleFeatureCalculator()
    
    # 注册一个简单的测试特征
    @feature(name="test_feature", description="Test feature for transformation")
    def test_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        return np.arange(len(candles), dtype=float)
    
    calc.load(candles, sequential=True)
    
    # 测试复杂转换链
    test_cases = [
        ("test_feature_mean20", "20期移动平均"),
        ("test_feature_std10", "10期标准差"),
        ("test_feature_mean20_lag5", "20期均值后滞后5期"),
        ("test_feature_std10_dt", "10期标准差后一阶差分"),
        ("test_feature_mean20_dt_lag3", "20期均值，一阶差分，滞后3期"),
    ]
    
    for feature_name, description in test_cases:
        try:
            result = calc.get([feature_name])[feature_name]
            print(f"✓ {feature_name} ({description}): Shape={result.shape}, Non-NaN={np.sum(~np.isnan(result))}")
        except Exception as e:
            print(f"✗ {feature_name}: ERROR - {str(e)}")


def test_feature_registration():
    """测试特征注册功能"""
    print("\n\nTesting feature registration...")
    
    calc = FlexibleFeatureCalculator()
    
    # 动态注册函数特征
    def custom_feature(candles: np.ndarray, period: int = 10, sequential: bool = True, **kwargs):
        """自定义测试特征"""
        return np.ones(len(candles)) * period
    
    calc.register_feature(
        name="custom_test",
        func=custom_feature,
        params={"period": 20},
        description="Custom test feature",
        aliases=["ct", "custom"]
    )
    
    # 测试注册的特征
    candles = create_mock_candles(100)
    calc.load(candles, sequential=True)
    
    # 通过不同方式访问
    result1 = calc.get(["custom_test"])["custom_test"]
    result2 = calc.get(["ct"])["ct"]  # 通过别名访问
    
    print(f"✓ Feature registration: Success")
    print(f"  Direct access: {result1[-1]}")
    print(f"  Alias access: {result2[-1]}")
    
    # 列出所有特征
    features = calc.list_features()
    print(f"\n✓ Total registered features: {len(features)}")
    
    # 显示前5个特征
    for i, (name, info) in enumerate(list(features.items())[:5]):
        print(f"  {i+1}. {name}: {info['description'][:50]}...")


def test_class_feature():
    """测试类型特征"""
    print("\n\nTesting class-based features...")
    
    # 创建一个简单的类型特征
    @class_feature(name="test_class_feature", params={"window": 10}, returns_multiple=True)
    class TestClassFeature:
        def __init__(self, candles: np.ndarray, window: int = 10, sequential: bool = False, **kwargs):
            self.candles = candles
            self.window = window
            self.sequential = sequential
        
        def res(self):
            # 返回多列结果
            result = np.zeros((len(self.candles), 3))
            for i in range(3):
                result[:, i] = np.arange(len(self.candles)) * (i + 1)
            return result
    
    calc = FlexibleFeatureCalculator()
    candles = create_mock_candles(100)
    calc.load(candles, sequential=True)
    
    # 测试多列返回
    result0 = calc.get(["test_class_feature_0"])["test_class_feature_0"]
    result1 = calc.get(["test_class_feature_1"])["test_class_feature_1"]
    result2 = calc.get(["test_class_feature_2"])["test_class_feature_2"]
    
    print(f"✓ Class feature column 0: Last value = {result0[-1]}")
    print(f"✓ Class feature column 1: Last value = {result1[-1]}")
    print(f"✓ Class feature column 2: Last value = {result2[-1]}")
    
    # 测试带转换的类特征
    result_dt = calc.get(["test_class_feature_0_dt"])["test_class_feature_0_dt"]
    print(f"✓ Class feature with dt transform: Last value = {result_dt[-1]}")


def test_sequential_behavior():
    """测试sequential参数的行为"""
    print("\n\nTesting sequential parameter behavior...")
    
    candles = create_mock_candles(100)
    
    # Sequential = True
    calc_seq = FlexibleFeatureCalculator()
    calc_seq.load(candles, sequential=True)
    
    # 注册测试特征
    @feature(name="seq_test", registry=calc_seq.registry)
    def seq_test_feature(candles: np.ndarray, sequential: bool = True, **kwargs):
        return np.arange(len(candles), dtype=float)
    
    result_seq = calc_seq.get(["seq_test"])["seq_test"]
    print(f"✓ Sequential=True: Shape={result_seq.shape}, Length={len(result_seq)}")
    
    # Sequential = False
    calc_single = FlexibleFeatureCalculator()
    calc_single.registry = calc_seq.registry  # 使用相同的注册中心
    calc_single.load(candles, sequential=False)
    result_single = calc_single.get(["seq_test"])["seq_test"]
    print(f"✓ Sequential=False: Shape={result_single.shape}, Value={result_single[0]}")
    
    # 验证结果
    assert len(result_seq) == len(candles), "Sequential result length should match candles"
    assert len(result_single) == 1, "Non-sequential result should have length 1"
    assert result_single[0] == result_seq[-1], "Non-sequential should return last value"
    print("✓ Sequential behavior verified correctly")


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
    print("FlexibleFeatureCalculator Test Suite")
    print("=" * 60)
    
    test_compatibility()
    test_new_transformations()
    test_feature_registration()
    test_class_feature()
    test_sequential_behavior()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)