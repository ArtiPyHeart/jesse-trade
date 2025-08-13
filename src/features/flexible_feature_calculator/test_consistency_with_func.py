"""
测试 FlexibleFeatureCalculator 与 all_features_func.py 的计算一致性

使用 btc_1m.npy 的最后10000行数据进行测试
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.features.all_features_func import feature_bundle
from src.features.flexible_feature_calculator import FlexibleFeatureCalculator
from src.features.flexible_feature_calculator.features import builtin  # 导入内置特征


def load_test_data():
    """加载测试数据"""
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "btc_1m.npy"
    
    if not data_path.exists():
        print(f"Error: Test data not found at {data_path}")
        return None
    
    # 加载最后5000行数据
    candles = np.load(data_path)[-5000:]
    print(f"Loaded test data: shape={candles.shape}")
    print(f"Date range: {candles[0, 0]} to {candles[-1, 0]}")
    print(f"Price range: ${candles[:, 2].min():.2f} - ${candles[:, 2].max():.2f}")
    
    return candles


def test_basic_features_consistency(candles):
    """测试基础特征的一致性"""
    print("\n" + "=" * 60)
    print("Testing Basic Features Consistency")
    print("=" * 60)
    
    # 计算原始特征
    print("\nComputing features with all_features_func...")
    original_features = feature_bundle(candles, sequential=True, lightweighted=True)
    
    # 计算新特征
    print("Computing features with FlexibleFeatureCalculator...")
    calc = FlexibleFeatureCalculator()
    calc.load(candles, sequential=True)
    
    # 测试的特征列表
    test_features = [
        # ADX相关
        "adx_7", "adx_7_dt", "adx_7_ddt", 
        "adx_7_lag1", "adx_7_lag5", "adx_7_lag10",
        "adx_14", "adx_14_dt", "adx_14_ddt",
        
        # Aroon
        "aroon_diff", "aroon_diff_dt", "aroon_diff_ddt",
        "aroon_diff_lag1", "aroon_diff_lag5",
        
        # Fisher
        "fisher", "fisher_dt", "fisher_ddt",
        "fisher_lag1", "fisher_lag10",
        
        # 其他基础指标
        "forecast_oscillator", "forecast_oscillator_dt",
        "hurst_coef_30", "hurst_coef_200",
        "iqr_ratio", "iqr_ratio_dt",
    ]
    
    passed = 0
    failed = 0
    errors = 0
    
    for feature_name in test_features:
        try:
            # 获取原始特征值
            if feature_name not in original_features:
                print(f"⚠️  {feature_name}: Not in original features, skipping")
                continue
            
            original_value = original_features[feature_name]
            
            # 获取新特征值
            new_value = calc.get([feature_name])[feature_name]
            
            # 比较结果
            if np.allclose(original_value, new_value, rtol=1e-5, atol=1e-8, equal_nan=True):
                print(f"✓ {feature_name}: PASS")
                passed += 1
            else:
                print(f"✗ {feature_name}: FAIL")
                print(f"  Shape: orig={original_value.shape}, new={new_value.shape}")
                diff = np.abs(original_value - new_value)
                max_diff = np.nanmax(diff)
                print(f"  Max diff: {max_diff}")
                failed += 1
                
        except Exception as e:
            print(f"✗ {feature_name}: ERROR - {str(e)}")
            errors += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed, {errors} errors")
    return passed, failed, errors


def test_multicolumn_features(candles):
    """测试多列特征的一致性"""
    print("\n" + "=" * 60)
    print("Testing Multi-Column Features Consistency")
    print("=" * 60)
    
    # 计算原始特征 (使用lightweighted=True避免计算较重的特征)
    print("\nComputing multi-column features with all_features_func (lightweight)...")
    original_features = feature_bundle(candles, sequential=True, lightweighted=True)
    
    # 计算新特征
    print("Computing multi-column features with FlexibleFeatureCalculator...")
    calc = FlexibleFeatureCalculator()
    calc.load(candles, sequential=True)
    
    # 测试多列特征 (减少测试列数以加快速度)
    multicolumn_tests = [
        # 自相关特征 (只测试前10列)
        ("ac", 10),  # autocorrelation
        ("acp", 10),  # autocorrelation_periodogram
        ("conv", 10),  # ehlers_convolution
        ("dft_spectrum", 10),  # DFT spectrum
        ("comb_spectrum", 10),  # comb spectrum
    ]
    
    passed = 0
    failed = 0
    errors = 0
    
    for base_name, num_cols in multicolumn_tests:
        print(f"\nTesting {base_name} ({num_cols} columns)...")
        
        for col_idx in range(min(5, num_cols)):  # 测试前5列
            feature_name = f"{base_name}_{col_idx}"
            
            try:
                # 获取原始特征值
                if feature_name not in original_features:
                    print(f"  ⚠️  Column {col_idx}: Not in original features")
                    continue
                
                original_value = original_features[feature_name]
                
                # 获取新特征值
                new_value = calc.get([feature_name])[feature_name]
                
                # 比较结果
                if np.allclose(original_value, new_value, rtol=1e-5, atol=1e-8, equal_nan=True):
                    print(f"  ✓ Column {col_idx}: PASS")
                    passed += 1
                else:
                    print(f"  ✗ Column {col_idx}: FAIL")
                    diff = np.abs(original_value - new_value)
                    max_diff = np.nanmax(diff)
                    print(f"    Max diff: {max_diff}")
                    failed += 1
                    
            except Exception as e:
                print(f"  ✗ Column {col_idx}: ERROR - {str(e)}")
                errors += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed, {errors} errors")
    return passed, failed, errors


def test_transformation_consistency(candles):
    """测试转换操作的一致性"""
    print("\n" + "=" * 60)
    print("Testing Transformation Consistency")
    print("=" * 60)
    
    # 使用较小的数据集进行转换测试
    test_candles = candles[-1000:]
    
    # 计算原始特征
    original_features = feature_bundle(test_candles, sequential=True, lightweighted=True)
    
    # 计算新特征
    calc = FlexibleFeatureCalculator()
    calc.load(test_candles, sequential=True)
    
    # 测试各种转换组合
    transformation_tests = [
        # 基础转换
        ("fisher_dt", "Fisher一阶差分"),
        ("adx_14_ddt", "ADX二阶差分"),
        ("aroon_diff_lag5", "Aroon差值滞后5"),
        
        # 复杂转换链
        ("fisher_dt_lag3", "Fisher差分后滞后3"),
        ("adx_7_ddt_lag1", "ADX二阶差分后滞后1"),
    ]
    
    passed = 0
    failed = 0
    
    for feature_name, description in transformation_tests:
        try:
            if feature_name not in original_features:
                print(f"⚠️  {feature_name} ({description}): Not in original")
                continue
            
            original_value = original_features[feature_name]
            new_value = calc.get([feature_name])[feature_name]
            
            if np.allclose(original_value, new_value, rtol=1e-5, atol=1e-8, equal_nan=True):
                print(f"✓ {feature_name} ({description}): PASS")
                passed += 1
            else:
                print(f"✗ {feature_name} ({description}): FAIL")
                failed += 1
                
        except Exception as e:
            print(f"✗ {feature_name} ({description}): ERROR - {str(e)}")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return passed, failed


def test_performance_comparison(candles):
    """性能对比测试"""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    import time
    
    # 使用较小的数据集进行性能测试
    test_candles = candles[-1000:]
    
    # 测试原始实现的性能
    print("\nTiming all_features_func (lightweight)...")
    start = time.time()
    original_features = feature_bundle(test_candles, sequential=True, lightweighted=True)
    original_time = time.time() - start
    print(f"  Time: {original_time:.2f}s")
    print(f"  Features computed: {len(original_features)}")
    
    # 测试新实现的性能
    print("\nTiming FlexibleFeatureCalculator...")
    calc = FlexibleFeatureCalculator()
    calc.load(test_candles, sequential=True)
    
    # 获取相同的特征集
    feature_names = list(original_features.keys())[:30]  # 测试前30个特征
    
    start = time.time()
    new_features = calc.get(feature_names)
    new_time = time.time() - start
    print(f"  Time: {new_time:.2f}s")
    print(f"  Features computed: {len(new_features)}")
    
    # 测试缓存性能
    print("\nTiming cached access...")
    start = time.time()
    cached_features = calc.get(feature_names)
    cached_time = time.time() - start
    print(f"  Time: {cached_time:.3f}s (should be near instant)")
    
    if new_time > 0:
        speedup = original_time / new_time
        print(f"\nPerformance comparison: {speedup:.2f}x")
    
    if cached_time < 0.1:
        print("✓ Cache working correctly")
    else:
        print("⚠️  Cache might not be working properly")


def main():
    """主测试函数"""
    print("=" * 60)
    print("FlexibleFeatureCalculator Consistency Test")
    print("with all_features_func.py")
    print("=" * 60)
    
    # 加载数据
    candles = load_test_data()
    if candles is None:
        return
    
    # 运行各项测试
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    # 基础特征测试
    passed, failed, errors = test_basic_features_consistency(candles)
    total_passed += passed
    total_failed += failed
    total_errors += errors
    
    # 多列特征测试
    passed, failed, errors = test_multicolumn_features(candles)
    total_passed += passed
    total_failed += failed
    total_errors += errors
    
    # 转换一致性测试
    passed, failed = test_transformation_consistency(candles)
    total_passed += passed
    total_failed += failed
    
    # 性能对比
    test_performance_comparison(candles)
    
    # 总结
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_passed + total_failed + total_errors}")
    print(f"✓ Passed: {total_passed}")
    print(f"✗ Failed: {total_failed}")
    print(f"⚠️  Errors: {total_errors}")
    
    success_rate = total_passed / (total_passed + total_failed + total_errors) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate > 95:
        print("\n✅ FlexibleFeatureCalculator is highly consistent with all_features_func!")
    elif success_rate > 80:
        print("\n⚠️  Most features are consistent, but some need attention")
    else:
        print("\n❌ Significant inconsistencies found, needs investigation")


if __name__ == "__main__":
    main()