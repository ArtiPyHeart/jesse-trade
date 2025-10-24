"""
生产指标集成测试

测试从生产代码导入的指标类（VMD、CWT、FTI）在实际 candles 数据上的运行情况。
使用模拟的 Jesse candles 数据来验证完整的集成流程。
"""

import sys
import numpy as np
import pytest

# 添加项目路径以导入生产代码
sys.path.insert(0, "/Users/yangqiuyu/Github/jesse-trade")

# 导入生产指标
try:
    from src.indicators.prod.emd.cls_vmd_indicator import VMD_NRBO
    from src.indicators.prod.wavelets.cls_cwt_swt import CWT_SWT
    from src.indicators.prod.fti import rust_fti, FTIResult
    from pyrs_indicators import HAS_RUST
except ImportError as e:
    HAS_RUST = False
    print(f"⚠️  Warning: Could not import production indicators: {e}")


def generate_fake_candles(num_candles: int = 500, seed: int = 42) -> np.ndarray:
    """
    生成假的 Jesse candles 数据

    格式：[timestamp, open, close, high, low, volume]

    Args:
        num_candles: 生成的 K 线数量
        seed: 随机种子

    Returns:
        candles: (num_candles, 6) 的 NumPy 数组
    """
    np.random.seed(seed)

    # 生成时间戳（以毫秒为单位，假设30分钟K线）
    start_time = 1609459200000  # 2021-01-01 00:00:00
    timestamps = start_time + np.arange(num_candles) * 1800000  # 30分钟 = 1800秒

    # 生成价格数据：趋势 + 周期 + 噪声
    base_price = 100.0
    trend = 0.05 * np.arange(num_candles)
    cycle = 5.0 * np.sin(2 * np.pi * np.arange(num_candles) / 20)
    noise = np.random.randn(num_candles) * 0.5

    close_prices = base_price + trend + cycle + noise

    # 生成 OHLC
    opens = close_prices - np.random.uniform(-0.5, 0.5, num_candles)
    highs = np.maximum(opens, close_prices) + np.random.uniform(0, 1, num_candles)
    lows = np.minimum(opens, close_prices) - np.random.uniform(0, 1, num_candles)

    # 生成成交量
    volumes = np.random.uniform(1000, 10000, num_candles)

    # 组装 candles: [timestamp, open, close, high, low, volume]
    candles = np.column_stack([
        timestamps,
        opens,
        close_prices,
        highs,
        lows,
        volumes
    ])

    return candles


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_nrbo_with_candles():
    """测试 VMD_NRBO 指标在 candles 数据上的运行"""
    candles = generate_fake_candles(num_candles=500, seed=42)

    # 创建 VMD_NRBO 指标实例
    window = 200
    vmd_indicator = VMD_NRBO(
        candles=candles,
        window=window,
        source_type="close",
        sequential=False  # 只返回最新值
    )

    result = vmd_indicator.res()

    print(f"\n  VMD_NRBO 测试:")
    print(f"    输入 candles: {candles.shape}")
    print(f"    窗口大小: {window}")
    print(f"    输出形状: {result.shape}")
    print(f"    输出范围: [{np.min(result):.4f}, {np.max(result):.4f}]")

    # 验证输出（sequential=False 返回 (1, 3) 形状）
    assert result.shape == (1, 3), f"VMD_NRBO 应输出 (1, 3)，实际: {result.shape}"
    result = result[0]  # 提取单个值 (3,)
    assert not np.any(np.isnan(result)), "VMD_NRBO 输出包含 NaN"
    assert not np.any(np.isinf(result)), "VMD_NRBO 输出包含 Inf"

    print("    ✅ VMD_NRBO 集成测试通过")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_nrbo_sequential():
    """测试 VMD_NRBO 指标的 sequential 模式"""
    candles = generate_fake_candles(num_candles=300, seed=123)

    window = 200
    vmd_indicator = VMD_NRBO(
        candles=candles,
        window=window,
        source_type="close",
        sequential=True  # 返回全序列
    )

    result = vmd_indicator.res()

    print(f"\n  VMD_NRBO Sequential 测试:")
    print(f"    输入 candles: {candles.shape}")
    print(f"    窗口大小: {window}")
    print(f"    输出形状: {result.shape}")
    print(f"    期望形状: ({candles.shape[0]}, 3)")

    # 验证输出
    assert result.shape == (candles.shape[0], 3), f"Sequential 输出形状不正确: {result.shape}"

    # 前 window-1 个值应该是 NaN
    warmup_nans = np.sum(np.isnan(result[:window-1, 0]))
    print(f"    Warmup NaN 数量: {warmup_nans}/{window-1}")

    # 后续值应该都有效
    valid_values = result[window-1:, :]
    assert not np.any(np.isnan(valid_values)), "Sequential 输出包含意外的 NaN"
    assert not np.any(np.isinf(valid_values)), "Sequential 输出包含 Inf"

    print("    ✅ VMD_NRBO Sequential 测试通过")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_cwt_swt_with_candles():
    """测试 CWT_SWT 指标在 candles 数据上的运行"""
    candles = generate_fake_candles(num_candles=500, seed=42)

    # 创建 CWT_SWT 指标实例
    window = 300
    cwt_indicator = CWT_SWT(
        candles=candles,
        window=window,
        source_type="close",
        sequential=False  # 只返回最新值
    )

    result = cwt_indicator.res()

    print(f"\n  CWT_SWT 测试:")
    print(f"    输入 candles: {candles.shape}")
    print(f"    窗口大小: {window}")
    print(f"    输出形状: {result.shape}")
    print(f"    输出范围 (dB): [{np.min(result):.4f}, {np.max(result):.4f}]")

    # 验证输出（sequential=False 返回 (1, N) 形状）
    assert result.ndim == 2, f"CWT_SWT 输出应该是2D，实际: {result.ndim}D"
    result = result[0]  # 提取单个值
    assert not np.any(np.isnan(result)), "CWT_SWT 输出包含 NaN"
    assert not np.any(np.isinf(result)), "CWT_SWT 输出包含 Inf"

    print("    ✅ CWT_SWT 集成测试通过")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_fti_with_price_data():
    """测试 FTI 函数在价格数据上的运行"""
    candles = generate_fake_candles(num_candles=500, seed=42)

    # 提取 close 价格
    close_prices = candles[:, 2]  # candles 的第3列是 close

    # 使用最近的 200 根 K 线
    lookback = 200
    price_data = close_prices[-lookback:]

    # 调用 FTI
    fti_value, filtered_value, width, best_period = rust_fti(
        price_data,
        use_log=True,
        min_period=5,
        max_period=65,
        half_length=35,
        lookback=150,
        beta=0.95,
        noise_cut=0.20
    )

    print(f"\n  FTI 测试:")
    print(f"    输入数据长度: {len(price_data)}")
    print(f"    价格范围: [{np.min(price_data):.2f}, {np.max(price_data):.2f}]")
    print(f"    FTI 值: {fti_value:.4f}")
    print(f"    Filtered 值: {filtered_value:.4f}")
    print(f"    Width: {width:.4f}")
    print(f"    Best Period: {best_period}")

    # 验证输出
    assert 0 <= fti_value <= 100, f"FTI 值应在 [0, 100]，实际: {fti_value}"
    assert not np.isnan(fti_value), "FTI 值是 NaN"
    assert not np.isnan(filtered_value), "Filtered 值是 NaN"
    assert not np.isnan(width), "Width 是 NaN"
    assert 5 <= best_period <= 65, f"Best period 应在 [5, 65]，实际: {best_period}"

    print("    ✅ FTI 集成测试通过")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_all_indicators_together():
    """综合测试：同时运行所有指标"""
    print(f"\n  综合集成测试:")

    # 生成统一的 candles 数据
    candles = generate_fake_candles(num_candles=500, seed=999)
    print(f"    生成 candles: {candles.shape}")

    # 1. VMD_NRBO
    vmd_result = VMD_NRBO(candles, window=200, sequential=False).res()[0]
    print(f"    ✓ VMD_NRBO 输出: {vmd_result.shape}")

    # 2. CWT_SWT
    cwt_result = CWT_SWT(candles, window=300, sequential=False).res()[0]
    print(f"    ✓ CWT_SWT 输出: {cwt_result.shape}")

    # 3. FTI
    close_prices = candles[:, 2]
    fti_result = rust_fti(close_prices[-200:])
    print(f"    ✓ FTI 输出: fti={fti_result[0]:.2f}, period={fti_result[3]}")

    # 验证所有输出都有效
    assert not np.any(np.isnan(vmd_result))
    assert not np.any(np.isnan(cwt_result))
    assert not np.isnan(fti_result[0])

    print("    ✅ 所有指标同时运行成功")


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust indicators 未安装或生产代码不可用，跳过测试")
        print("   1. 安装 Rust 指标: cd rust_indicators && pip install -e .")
        print("   2. 确保项目路径正确")
        sys.exit(1)

    print("=" * 60)
    print("生产指标集成测试 (pyrs_indicators v0.4.0)")
    print("=" * 60)

    test_vmd_nrbo_with_candles()
    test_vmd_nrbo_sequential()
    test_cwt_swt_with_candles()
    test_fti_with_price_data()
    test_all_indicators_together()

    print("\n" + "=" * 60)
    print("✅ 所有生产指标集成测试通过！")
    print("=" * 60)
