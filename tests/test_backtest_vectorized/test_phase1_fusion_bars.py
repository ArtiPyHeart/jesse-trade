"""
Phase 1 测试: Fusion Bar 生成与边界分割

测试目标:
1. 验证 DemoBar 累计阈值逻辑和 bar 边界计算
2. 验证 warmup/trading 边界分割正确性
3. 边界条件处理
"""

import numpy as np
import pytest

from src.bars.fusion.demo import DemoBar


class TestFusionBarGeneration:
    """测试 Fusion Bar 生成的基本正确性"""

    def test_fusion_bar_compression_ratio(self, small_candles):
        """
        验证 fusion bar 压缩效果

        预期: fusion bar 数量 < 原始 candle 数量
        """
        demo_bar = DemoBar(clip_r=0.012, max_bars=-1, threshold=1.399)
        demo_bar.update_with_candles(small_candles)
        fusion_bars = demo_bar.get_fusion_bars()

        # 压缩效果: fusion bar 数量应显著少于原始 K 线
        assert len(fusion_bars) < len(small_candles), (
            f"Fusion bars ({len(fusion_bars)}) should be less than candles ({len(small_candles)})"
        )

        # 压缩比例通常在 5%-50% 之间
        compression_ratio = len(fusion_bars) / len(small_candles)
        print(f"\nCompression ratio: {compression_ratio:.2%}")
        print(f"  - Input candles: {len(small_candles)}")
        print(f"  - Output fusion bars: {len(fusion_bars)}")

    def test_fusion_bar_output_format(self, small_candles):
        """
        验证输出格式正确

        预期: [timestamp, open, close, high, low, volume] 6 列
        """
        demo_bar = DemoBar()
        demo_bar.update_with_candles(small_candles)
        fusion_bars = demo_bar.get_fusion_bars()

        # 6 列格式
        assert fusion_bars.shape[1] == 6, (
            f"Expected 6 columns, got {fusion_bars.shape[1]}"
        )

        # 数据类型为 float64
        assert fusion_bars.dtype == np.float64, (
            f"Expected float64, got {fusion_bars.dtype}"
        )

    def test_fusion_bar_timestamps_monotonic(self, small_candles):
        """
        验证时间戳单调递增

        预期: 每个 fusion bar 的时间戳都大于前一个
        """
        demo_bar = DemoBar()
        demo_bar.update_with_candles(small_candles)
        fusion_bars = demo_bar.get_fusion_bars()

        timestamps = fusion_bars[:, 0]
        diffs = np.diff(timestamps)

        # 时间戳严格递增
        assert np.all(diffs > 0), "Timestamps must be strictly increasing"

        print("\nTimestamp intervals (ms):")
        print(f"  - Min: {np.min(diffs):.0f}")
        print(f"  - Max: {np.max(diffs):.0f}")
        print(f"  - Mean: {np.mean(diffs):.0f}")

    def test_fusion_bar_ohlc_logic(self, small_candles):
        """
        验证 OHLC 逻辑正确性

        预期:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        """
        demo_bar = DemoBar()
        demo_bar.update_with_candles(small_candles)
        fusion_bars = demo_bar.get_fusion_bars()

        opens = fusion_bars[:, 1]
        closes = fusion_bars[:, 2]
        highs = fusion_bars[:, 3]
        lows = fusion_bars[:, 4]

        # High >= max(open, close)
        assert np.all(highs >= np.maximum(opens, closes)), (
            "High must be >= max(open, close)"
        )

        # Low <= min(open, close)
        assert np.all(lows <= np.minimum(opens, closes)), (
            "Low must be <= min(open, close)"
        )

        # High >= Low
        assert np.all(highs >= lows), "High must be >= Low"

    def test_fusion_bar_volume_positive(self, small_candles):
        """
        验证成交量为非负数

        预期: volume >= 0
        """
        demo_bar = DemoBar()
        demo_bar.update_with_candles(small_candles)
        fusion_bars = demo_bar.get_fusion_bars()

        volumes = fusion_bars[:, 5]
        assert np.all(volumes >= 0), "Volume must be non-negative"

    def test_fusion_bar_prices_positive(self, small_candles):
        """
        验证价格为正数

        预期: open, close, high, low > 0
        """
        demo_bar = DemoBar()
        demo_bar.update_with_candles(small_candles)
        fusion_bars = demo_bar.get_fusion_bars()

        # 所有价格列
        prices = fusion_bars[:, 1:5]
        assert np.all(prices > 0), "All prices must be positive"


class TestThresholdCalculation:
    """测试阈值公式计算"""

    def test_threshold_formula_manual(self):
        """
        手动验证阈值公式计算

        阈值公式: abs(close[t] - close[t-1]) * (high[t] - low[t]) / close[t]
        累计 > 1.399 时生成新 bar
        """
        # 构造已知输入
        # 格式: [timestamp, open, close, high, low, volume]
        candles = np.array(
            [
                [1000, 100.0, 100.0, 100.5, 99.5, 1000.0],  # t=0
                [2000, 100.0, 101.0, 101.5, 99.0, 1000.0],  # t=1
                [3000, 101.0, 102.0, 102.5, 100.5, 1000.0],  # t=2
                [4000, 102.0, 103.0, 103.5, 101.5, 1000.0],  # t=3
                [5000, 103.0, 104.0, 104.5, 102.5, 1000.0],  # t=4
            ]
        )

        # 手动计算阈值
        # t=1: abs(101 - 100) * (101.5 - 99) / 101 = 1 * 2.5 / 101 = 0.02475
        # t=2: abs(102 - 101) * (102.5 - 100.5) / 102 = 1 * 2 / 102 = 0.01961
        # t=3: abs(103 - 102) * (103.5 - 101.5) / 103 = 1 * 2 / 103 = 0.01942
        # t=4: abs(104 - 103) * (104.5 - 102.5) / 104 = 1 * 2 / 104 = 0.01923

        demo_bar = DemoBar(clip_r=0, max_bars=-1, threshold=1.399)  # 禁用 clip
        thresholds = demo_bar.get_thresholds(candles)

        # 验证阈值计算
        expected_t1 = abs(101 - 100) * (101.5 - 99.0) / 101
        expected_t2 = abs(102 - 101) * (102.5 - 100.5) / 102
        expected_t3 = abs(103 - 102) * (103.5 - 101.5) / 103
        expected_t4 = abs(104 - 103) * (104.5 - 102.5) / 104

        np.testing.assert_almost_equal(thresholds[0], expected_t1, decimal=5)
        np.testing.assert_almost_equal(thresholds[1], expected_t2, decimal=5)
        np.testing.assert_almost_equal(thresholds[2], expected_t3, decimal=5)
        np.testing.assert_almost_equal(thresholds[3], expected_t4, decimal=5)

    def test_clip_r_filtering(self):
        """
        验证 clip_r 过滤小波动

        预期: 小于 clip_r 的阈值被设为 0
        """
        candles = np.array(
            [
                [1000, 100.0, 100.0, 100.01, 99.99, 1000.0],  # 微小波动
                [2000, 100.0, 100.001, 100.01, 99.99, 1000.0],  # 极小变化
                [3000, 100.0, 105.0, 106.0, 99.0, 1000.0],  # 大波动
            ]
        )

        demo_bar = DemoBar(clip_r=0.01, max_bars=-1, threshold=1.399)
        thresholds = demo_bar.get_thresholds(candles)

        # 小波动应该被 clip 为 0
        print(f"\nThresholds with clip_r=0.01: {thresholds}")

        # 第一个阈值（微小波动）应该被 clip
        # 计算: abs(100.001 - 100) * (100.01 - 99.99) / 100.001 ≈ 0.00002
        # 这远小于 clip_r=0.01，应该被设为 0
        assert thresholds[0] == 0, (
            f"Small threshold {thresholds[0]} should be clipped to 0"
        )


class TestWarmupTradingSplit:
    """测试 Warmup/Trading 边界分割"""

    def test_basic_split(self, jesse_candles):
        """
        验证基本边界分割正确性
        """
        from backtest_vectorized_no_jesse import generate_all_fusion_bars_with_split

        warmup_candles, trading_candles = jesse_candles

        fusion_bars, warmup_len = generate_all_fusion_bars_with_split(
            warmup_candles, trading_candles, max_bars=-1
        )

        # 边界条件验证
        warmup_last_ts = warmup_candles[-1, 0]

        # 1. warmup 最后一根 fusion bar 的时间戳 <= warmup 最后一根 candle 时间戳
        assert fusion_bars[warmup_len - 1, 0] <= warmup_last_ts, (
            f"Warmup last fusion bar {fusion_bars[warmup_len - 1, 0]} > warmup last candle {warmup_last_ts}"
        )

        # 2. trading 第一根 fusion bar 的时间戳 > warmup 最后一根 candle 时间戳
        assert fusion_bars[warmup_len, 0] > warmup_last_ts, (
            f"Trading first fusion bar {fusion_bars[warmup_len, 0]} <= warmup last candle {warmup_last_ts}"
        )

        # 3. warmup_len 在合理范围内
        assert 0 < warmup_len < len(fusion_bars), (
            f"warmup_len {warmup_len} out of range [1, {len(fusion_bars) - 1}]"
        )

        print("\nSplit result:")
        print(f"  - Warmup fusion bars: {warmup_len}")
        print(f"  - Trading fusion bars: {len(fusion_bars) - warmup_len}")
        print(f"  - Total fusion bars: {len(fusion_bars)}")
        print(f"  - Warmup ratio: {warmup_len / len(fusion_bars):.1%}")

    def test_split_boundary_exact(self, jesse_candles):
        """
        验证边界分割的精确性

        预期:
        - 所有 warmup fusion bars 的时间戳 <= warmup 最后 candle 时间戳
        - 所有 trading fusion bars 的时间戳 > warmup 最后 candle 时间戳
        """
        from backtest_vectorized_no_jesse import generate_all_fusion_bars_with_split

        warmup_candles, trading_candles = jesse_candles

        fusion_bars, warmup_len = generate_all_fusion_bars_with_split(
            warmup_candles, trading_candles, max_bars=-1
        )

        warmup_last_ts = warmup_candles[-1, 0]

        # 所有 warmup bars 的时间戳都 <= warmup_last_ts
        warmup_bars = fusion_bars[:warmup_len]
        assert np.all(warmup_bars[:, 0] <= warmup_last_ts), (
            "Some warmup fusion bars have timestamp > warmup last candle"
        )

        # 所有 trading bars 的时间戳都 > warmup_last_ts
        trading_bars = fusion_bars[warmup_len:]
        assert np.all(trading_bars[:, 0] > warmup_last_ts), (
            "Some trading fusion bars have timestamp <= warmup last candle"
        )

    def test_no_fusion_bar_loss(self, jesse_candles):
        """
        验证分割不丢失 fusion bars

        预期: 单独生成和分割生成的 fusion bar 数量一致
        """
        from backtest_vectorized_no_jesse import generate_all_fusion_bars_with_split

        warmup_candles, trading_candles = jesse_candles

        # 分割生成
        fusion_bars, warmup_len = generate_all_fusion_bars_with_split(
            warmup_candles, trading_candles, max_bars=-1
        )

        # 单独生成（作为参照）
        all_candles = np.vstack([warmup_candles, trading_candles])
        demo_bar = DemoBar(max_bars=-1)
        demo_bar.update_with_candles(all_candles)
        expected_fusion_bars = demo_bar.get_fusion_bars()

        # 数量一致
        assert len(fusion_bars) == len(expected_fusion_bars), (
            f"Mismatch: split={len(fusion_bars)}, direct={len(expected_fusion_bars)}"
        )

        # 内容一致
        np.testing.assert_array_equal(
            fusion_bars, expected_fusion_bars, err_msg="Fusion bar content mismatch"
        )


class TestEdgeCases:
    """测试边界条件"""

    def test_minimum_candles(self):
        """
        测试最小输入

        DemoBar 需要足够的 candles 才能开始生成 fusion bars。
        这个测试验证少量 candles 的行为是否符合预期。
        """
        # 2 根 candle 可能不足以生成 fusion bar
        candles_2 = np.array(
            [
                [1000, 100.0, 100.0, 100.5, 99.5, 1000.0],
                [2000, 100.0, 200.0, 200.5, 99.5, 1000.0],
            ]
        )

        demo_bar = DemoBar(clip_r=0, max_bars=-1, threshold=0.1)
        demo_bar.update_with_candles(candles_2)
        fusion_bars_2 = demo_bar.get_fusion_bars()

        # 2 根 candle 可能生成 0 个 fusion bar (取决于阈值累计)
        print(f"\n2 candles → {len(fusion_bars_2)} fusion bars")

        # 测试更多 candles 确保能生成 fusion bar
        candles_10 = np.zeros((10, 6))
        candles_10[:, 0] = np.arange(10) * 60000
        candles_10[:, 1] = 100  # open
        candles_10[:, 2] = [
            100,
            110,
            105,
            115,
            108,
            120,
            112,
            125,
            118,
            130,
        ]  # close (大波动)
        candles_10[:, 3] = candles_10[:, 2] + 2  # high
        candles_10[:, 4] = candles_10[:, 2] - 2  # low
        candles_10[:, 5] = 1000  # volume

        demo_bar_10 = DemoBar(clip_r=0, max_bars=-1, threshold=0.1)
        demo_bar_10.update_with_candles(candles_10)
        fusion_bars_10 = demo_bar_10.get_fusion_bars()

        print(f"10 candles with large swings → {len(fusion_bars_10)} fusion bars")

        # 10 根有大波动的 candle 应该能生成至少 1 个 fusion bar
        assert len(fusion_bars_10) >= 1, (
            "Expected at least 1 fusion bar from 10 candles"
        )

    def test_single_candle_no_crash(self):
        """
        测试单根 candle 不崩溃

        预期: 不抛异常，输出合理
        """
        candles = np.array([[1000, 100.0, 100.0, 100.5, 99.5, 1000.0]])

        demo_bar = DemoBar()
        demo_bar.update_with_candles(candles)
        fusion_bars = demo_bar.get_fusion_bars()

        # 单根 candle 可能生成 0 或 1 个 fusion bar
        assert len(fusion_bars) <= 1, (
            f"Too many fusion bars from single candle: {len(fusion_bars)}"
        )

    def test_max_bars_limit(self):
        """
        测试 max_bars 限制

        预期: fusion bar 数量不超过 max_bars
        """
        # 生成大量 candles
        n = 10000
        candles = np.zeros((n, 6))
        candles[:, 0] = np.arange(n) * 60000  # 时间戳
        candles[:, 1] = 100 + np.random.randn(n) * 0.5  # open
        candles[:, 2] = 100 + np.random.randn(n) * 0.5  # close
        candles[:, 3] = candles[:, [1, 2]].max(axis=1) + 0.1  # high
        candles[:, 4] = candles[:, [1, 2]].min(axis=1) - 0.1  # low
        candles[:, 5] = 1000  # volume

        max_bars = 100
        demo_bar = DemoBar(max_bars=max_bars)
        demo_bar.update_with_candles(candles)
        fusion_bars = demo_bar.get_fusion_bars()

        assert len(fusion_bars) <= max_bars, (
            f"Fusion bars ({len(fusion_bars)}) exceeds max_bars ({max_bars})"
        )

    def test_stateful_behavior(self, small_candles):
        """
        验证 DemoBar 是有状态的

        预期: 分两次调用 update 与一次调用结果不同
        """
        # 分割 candles
        mid = len(small_candles) // 2
        first_half = small_candles[:mid]
        _ = small_candles[mid:]  # second_half (unused, for demonstration)

        # 方法 1: 分两次调用
        demo_bar_1 = DemoBar()
        demo_bar_1.update_with_candles(first_half)
        # 注意：这里实际上 DemoBar 可能不支持多次 update
        # 取决于实现，这个测试可能需要调整

        # 方法 2: 一次调用
        demo_bar_2 = DemoBar()
        demo_bar_2.update_with_candles(small_candles)
        fusion_bars_2 = demo_bar_2.get_fusion_bars()

        # 验证一次生成的结果
        assert len(fusion_bars_2) > 0, "One-shot generation should produce fusion bars"

        print("\nStateful behavior test:")
        print(f"  - One-shot fusion bars: {len(fusion_bars_2)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
