"""
快速测试 FlexibleFeatureCalculator 与 all_features_func.py 的一致性
使用较小的数据集和特征子集
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.features.all_features_func import feature_bundle
from src.features.flexible_feature_calculator import FlexibleFeatureCalculator
from src.features.flexible_feature_calculator.features import builtin


def quick_consistency_test():
    """快速一致性测试"""
    print("=" * 60)
    print("Quick Consistency Test")
    print("=" * 60)
    
    # 加载数据
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "btc_1m.npy"
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return
    
    # 只使用最后1000行数据
    candles = np.load(data_path)[-1000:]
    print(f"Loaded {len(candles)} candles for testing")
    
    # 计算原始特征 (lightweight mode)
    print("\nComputing original features...")
    original = feature_bundle(candles, sequential=True, lightweighted=True)
    print(f"Original features computed: {len(original)}")
    
    # 初始化新计算器
    print("\nInitializing FlexibleFeatureCalculator...")
    calc = FlexibleFeatureCalculator()
    calc.load(candles, sequential=True)
    
    # 选择要测试的特征子集
    test_features = [
        # 基础特征
        "adx_7", "adx_14",
        "aroon_diff", 
        "fisher",
        "forecast_oscillator",
        "hurst_coef_30",
        "iqr_ratio",
        
        # 一阶差分
        "adx_7_dt", "adx_14_dt",
        "fisher_dt",
        
        # 滞后
        "adx_7_lag1", "adx_14_lag5",
        "fisher_lag1",
        
        # 多列特征 (只测试前3列)
        "ac_0", "ac_1", "ac_2",
    ]
    
    print(f"\nTesting {len(test_features)} features...")
    passed = 0
    failed = 0
    
    for feat in test_features:
        try:
            if feat not in original:
                print(f"  ⚠️  {feat}: Not in original, skipping")
                continue
            
            orig_val = original[feat]
            new_val = calc.get([feat])[feat]
            
            if np.allclose(orig_val, new_val, rtol=1e-5, atol=1e-8, equal_nan=True):
                print(f"  ✓ {feat}")
                passed += 1
            else:
                print(f"  ✗ {feat}: Values differ")
                failed += 1
        except Exception as e:
            print(f"  ✗ {feat}: Error - {str(e)}")
            failed += 1
    
    print(f"\nResults: {passed}/{len(test_features)} passed")
    
    if passed == len(test_features):
        print("✅ Perfect consistency!")
    elif passed > len(test_features) * 0.8:
        print("⚠️  Most features consistent")
    else:
        print("❌ Significant inconsistencies")
    
    # 测试缓存
    print("\nTesting cache...")
    import time
    
    # 第一次访问
    start = time.time()
    _ = calc.get(["adx_7"])
    first_time = time.time() - start
    
    # 第二次访问（应该使用缓存）
    start = time.time()
    _ = calc.get(["adx_7"])
    cached_time = time.time() - start
    
    if cached_time < first_time * 0.1:
        print(f"✓ Cache working: {cached_time:.4f}s vs {first_time:.4f}s")
    else:
        print(f"⚠️  Cache issue: {cached_time:.4f}s vs {first_time:.4f}s")
    
    # 测试多列缓存
    print("\nTesting multi-column cache...")
    calc.clear_cache()
    
    # 请求第一列
    start = time.time()
    _ = calc.get(["ac_0"])
    col0_time = time.time() - start
    
    # 请求第二列（应该已缓存）
    start = time.time()
    _ = calc.get(["ac_1"])
    col1_time = time.time() - start
    
    if col1_time < col0_time * 0.1:
        print(f"✓ Multi-column cache working: {col1_time:.4f}s vs {col0_time:.4f}s")
    else:
        print(f"⚠️  Multi-column cache issue: {col1_time:.4f}s vs {col0_time:.4f}s")


if __name__ == "__main__":
    quick_consistency_test()