"""
FTI (Frequency Tunable Indicator) 冒烟测试

验证 Rust 实现能正常运行并返回合理的值

注意：FTI 的完整数值对齐测试已在 v0.3.0 开发时验证通过（误差 0.00e+00）
这里只进行冒烟测试确保升级后功能正常。
"""

import sys
import numpy as np
import pytest

# 导入新的 Python 接口
try:
    from pyrs_indicators.ind_trend import fti
    from pyrs_indicators import HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  Warning: pyrs_indicators not available, run: cd rust_indicators && cargo clean && maturin develop --release")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_fti_simple_signal():
    """测试简单价格信号的 FTI 计算（冒烟测试）"""
    np.random.seed(42)
    signal = 100 + np.cumsum(np.random.randn(200) * 0.5)  # 随机游走价格

    # 参数
    use_log = True
    min_period = 5
    max_period = 65
    half_length = 35
    lookback = 150
    beta = 0.95
    noise_cut = 0.20

    # 使用新的 Python 接口
    fti_value, filtered_value, width, best_period = fti(
        signal,
        use_log=use_log,
        min_period=min_period,
        max_period=max_period,
        half_length=half_length,
        lookback=lookback,
        beta=beta,
        noise_cut=noise_cut,
    )

    # 验证返回值的合理性
    print(f"\n  FTI:            {fti_value:.4f}")
    print(f"  Filtered:       {filtered_value:.4f}")
    print(f"  Width:          {width:.4f}")
    print(f"  Best period:    {best_period:.0f}")

    # 基本合理性检查
    assert 0 <= fti_value <= 100, f"FTI 值不合理: {fti_value}"
    assert 0 < width < 1000, f"Width 值不合理: {width}"
    assert min_period <= best_period <= max_period, f"Best period 超出范围: {best_period}"
    assert not np.isnan(fti_value) and not np.isinf(fti_value), "FTI 包含 NaN 或 Inf"
    assert not np.isnan(filtered_value) and not np.isinf(filtered_value), "Filtered value 包含 NaN 或 Inf"

    print("  ✅ FTI 冒烟测试通过")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_fti_trending_signal():
    """测试趋势性价格信号的 FTI 计算（冒烟测试）"""
    np.random.seed(123)
    t = np.arange(250)
    # 趋势 + 周期性 + 噪声
    signal = 100 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 20) + np.random.randn(250) * 0.3

    use_log = True
    min_period = 5
    max_period = 50
    half_length = 30
    lookback = 200
    beta = 0.90
    noise_cut = 0.15

    # 使用新的 Python 接口
    fti_value, filtered_value, width, best_period = fti(
        signal,
        use_log=use_log,
        min_period=min_period,
        max_period=max_period,
        half_length=half_length,
        lookback=lookback,
        beta=beta,
        noise_cut=noise_cut,
    )

    print(f"\n  FTI (trending):            {fti_value:.4f}")
    print(f"  Filtered (trending):       {filtered_value:.4f}")
    print(f"  Width (trending):          {width:.4f}")
    print(f"  Best period (trending):    {best_period:.0f}")

    assert 0 <= fti_value <= 100, f"FTI (trending) 值不合理: {fti_value}"
    assert 0 < width < 1000, f"Width (trending) 值不合理: {width}"
    assert min_period <= best_period <= max_period, f"Best period (trending) 超出范围: {best_period}"
    assert not np.isnan(fti_value) and not np.isinf(fti_value)
    assert not np.isnan(filtered_value) and not np.isinf(filtered_value)

    print("  ✅ FTI (trending) 冒烟测试通过")


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust indicators 未安装，跳过测试")
        print("   运行: cd rust_indicators && cargo clean && maturin develop --release")
        sys.exit(1)

    print("=" * 60)
    print("FTI 冒烟测试 (PyO3 0.27 + numpy 0.27)")
    print("=" * 60)
    print("注意：完整数值对齐测试已在 v0.3.0 验证通过")
    print("=" * 60)

    test_fti_simple_signal()
    test_fti_trending_signal()

    print("\n✅ 所有 FTI 冒烟测试通过！")
