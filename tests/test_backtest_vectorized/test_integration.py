"""
集成测试: 端到端验证回测流程

测试目标:
- 使用真实 Jesse K线数据
- 完整运行回测流程 (fusion bars → features → predictions → voting → trading)
- 验证结果完整性和一致性

运行方式:
    pytest tests/test_backtest_vectorized/test_integration.py -v -s

注意: 这些测试需要连接数据库和加载模型，可能需要较长时间。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 测试用模型列表 (来自 strategies/BinanceBtcDemoBarV2/models/config.py)
TEST_MODELS = ["c_L6_N1", "r_L5_N2"]


class TestEndToEndSmallData:
    """端到端小数据测试"""

    def test_complete_pipeline_no_error(self, jesse_candles):
        """测试完整流程无报错"""
        warmup_candles, trading_candles = jesse_candles

        # 使用足够数据确保生成足够的 fusion bars (某些特征需要 512 窗口)
        warmup = warmup_candles  # 40000 根 warmup
        trading = trading_candles  # 全部 trading

        # 导入回测模块
        from backtest_vectorized_no_jesse import (
            generate_all_fusion_bars_with_split,
            compute_features_vectorized,
            predict_all_models,
            aggregate_votes,
            FastBacktester,
            BacktestAnalyzer,
        )

        # 加载特征信息
        feature_info_path = (
            ROOT_DIR
            / "strategies"
            / "BinanceBtcDemoBarV2"
            / "models"
            / "feature_info.json"
        )
        with open(feature_info_path) as f:
            feature_info = json.load(f)

        # Phase 1: Fusion Bars
        fusion_bars, warmup_fusion_bars_len = generate_all_fusion_bars_with_split(
            warmup, trading, max_bars=-1
        )
        assert len(fusion_bars) > 0
        assert warmup_fusion_bars_len > 0

        # Phase 2: Features
        df_features = compute_features_vectorized(
            fusion_bars,
            warmup_fusion_bars_len,
            feature_info,
            TEST_MODELS,
        )
        trading_bars_count = len(fusion_bars) - warmup_fusion_bars_len
        assert len(df_features) == trading_bars_count
        assert not df_features.empty

        # Phase 3: Predictions
        predictions = predict_all_models(df_features, TEST_MODELS, feature_info)
        for model in TEST_MODELS:
            assert model in predictions
            assert len(predictions[model]) == len(df_features)

        # Phase 4: Voting
        signals = aggregate_votes(predictions, TEST_MODELS)
        assert len(signals) == len(df_features)
        assert all(s in ["long", "short", "flat"] for s in signals)

        # Phase 5: Trading simulation
        bt = FastBacktester(10000, 0.0004, leverage=3)
        trading_fusion_bars = fusion_bars[warmup_fusion_bars_len:]

        for i in range(len(trading_fusion_bars)):
            bar = trading_fusion_bars[i]
            timestamp = int(bar[0])
            open_price = float(bar[1])
            close_price = float(bar[2])
            high_price = float(bar[3])
            low_price = float(bar[4])
            signal = signals[i]

            # 止损检查
            if not bt.position.is_flat:
                bt.check_stop_loss(timestamp, high_price, low_price, close_price)

            # 交易逻辑
            if bt.position.is_flat and signal == "long":
                stop_loss = open_price * 0.98
                bt.open_long(timestamp, open_price, stop_loss)
            elif bt.position.is_flat and signal == "short":
                stop_loss = open_price * 1.02
                bt.open_short(timestamp, open_price, stop_loss)

            # 记录权益
            bt.record_equity(timestamp, close_price)

        # 强制平仓
        if not bt.position.is_flat:
            final_ts = int(trading_fusion_bars[-1, 0])
            final_price = float(trading_fusion_bars[-1, 2])
            bt.close_position(final_ts, final_price, reason="force_close")

        # Phase 6: Metrics
        metrics = BacktestAnalyzer.calculate_metrics(
            bt.trades, bt.equity_curve, 10000
        )

        # 验证指标存在
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert "total_trades" in metrics

        print(f"\n[Integration] Total Return: {metrics['total_return_pct']}")
        print(f"[Integration] Max Drawdown: {metrics['max_drawdown_pct']}")
        print(f"[Integration] Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"[Integration] Total Trades: {metrics['total_trades']}")

    def test_determinism(self, jesse_candles):
        """测试多次运行结果一致 (确定性)"""
        warmup_candles, trading_candles = jesse_candles

        # 使用足够数据确保生成足够的 fusion bars
        warmup = warmup_candles  # 40000 根 warmup
        trading = trading_candles  # 全部 trading

        from backtest_vectorized_no_jesse import (
            generate_all_fusion_bars_with_split,
        )

        # 运行两次
        fusion1, warmup_len1 = generate_all_fusion_bars_with_split(
            warmup, trading, max_bars=-1
        )
        fusion2, warmup_len2 = generate_all_fusion_bars_with_split(
            warmup, trading, max_bars=-1
        )

        # 验证结果一致
        assert warmup_len1 == warmup_len2
        assert len(fusion1) == len(fusion2)
        np.testing.assert_array_equal(fusion1, fusion2)


class TestDataFlow:
    """测试数据流完整性"""

    def test_fusion_bars_to_features_length(self, jesse_candles):
        """测试 fusion bars 到 features 的长度对应"""
        warmup_candles, trading_candles = jesse_candles

        # 使用足够数据确保生成足够的 fusion bars (某些特征需要 512 窗口)
        warmup = warmup_candles  # 40000 根 warmup
        trading = trading_candles  # 全部 trading

        from backtest_vectorized_no_jesse import (
            generate_all_fusion_bars_with_split,
            compute_features_vectorized,
        )

        feature_info_path = (
            ROOT_DIR
            / "strategies"
            / "BinanceBtcDemoBarV2"
            / "models"
            / "feature_info.json"
        )
        with open(feature_info_path) as f:
            feature_info = json.load(f)

        fusion_bars, warmup_len = generate_all_fusion_bars_with_split(
            warmup, trading, max_bars=-1
        )
        trading_bars_count = len(fusion_bars) - warmup_len

        df_features = compute_features_vectorized(
            fusion_bars,
            warmup_len,
            feature_info,
            TEST_MODELS,
        )

        # 特征数量 == trading fusion bars 数量
        assert len(df_features) == trading_bars_count
        print(f"\n[DataFlow] Trading bars: {trading_bars_count}")
        print(f"[DataFlow] Features rows: {len(df_features)}")

    def test_predictions_to_signals_length(self, jesse_candles):
        """测试预测到信号的长度对应"""
        warmup_candles, trading_candles = jesse_candles

        # 使用足够数据确保生成足够的 fusion bars (某些特征需要 512 窗口)
        warmup = warmup_candles  # 40000 根 warmup
        trading = trading_candles  # 全部 trading

        from backtest_vectorized_no_jesse import (
            generate_all_fusion_bars_with_split,
            compute_features_vectorized,
            predict_all_models,
            aggregate_votes,
        )

        feature_info_path = (
            ROOT_DIR
            / "strategies"
            / "BinanceBtcDemoBarV2"
            / "models"
            / "feature_info.json"
        )
        with open(feature_info_path) as f:
            feature_info = json.load(f)

        fusion_bars, warmup_len = generate_all_fusion_bars_with_split(
            warmup, trading, max_bars=-1
        )

        df_features = compute_features_vectorized(
            fusion_bars,
            warmup_len,
            feature_info,
            TEST_MODELS,
        )

        predictions = predict_all_models(df_features, TEST_MODELS, feature_info)
        signals = aggregate_votes(predictions, TEST_MODELS)

        # signals 长度 == features 长度
        assert len(signals) == len(df_features)

        for model in TEST_MODELS:
            assert len(predictions[model]) == len(df_features)

        print(f"\n[DataFlow] Features: {len(df_features)}")
        print(f"[DataFlow] Signals: {len(signals)}")


class TestResultValidity:
    """测试结果有效性"""

    def test_equity_curve_length(self, jesse_candles):
        """测试权益曲线长度合理"""
        warmup_candles, trading_candles = jesse_candles

        from backtest_vectorized_no_jesse import (
            FastBacktester,
        )

        # 模拟交易
        bt = FastBacktester(10000, 0.0004, leverage=1)

        # 记录 100 个权益点
        for i in range(100):
            bt.record_equity(timestamp=i * 60000, current_price=50000 + i * 10)

        assert len(bt.equity_curve) == 100

    def test_trades_record_complete(self, jesse_candles):
        """测试交易记录完整"""
        from backtest_vectorized_no_jesse import FastBacktester

        bt = FastBacktester(10000, 0.0004, leverage=1)

        # 开多仓
        bt.open_long(timestamp=1000, price=50000, stop_loss_price=49000)
        assert len(bt.trades) == 1
        assert bt.trades[0].action == "open_long"

        # 平仓
        bt.close_position(timestamp=2000, price=51000, reason="close")
        assert len(bt.trades) == 2
        assert bt.trades[1].action == "close_long"
        assert bt.trades[1].pnl != 0

    def test_metrics_keys_complete(self, jesse_candles):
        """测试指标键完整"""
        from backtest_vectorized_no_jesse import (
            BacktestAnalyzer,
            EquityPoint,
            Trade,
        )

        trades = [
            Trade(timestamp=1000, action="close_long", price=51000, qty=0.1, fee=2, pnl=100, balance=10100),
        ]
        equity_curve = [
            EquityPoint(timestamp=1000, equity=10000, benchmark_equity=10000),
            EquityPoint(timestamp=2000, equity=10100, benchmark_equity=10050),
        ]

        metrics = BacktestAnalyzer.calculate_metrics(trades, equity_curve, 10000)

        required_keys = [
            "starting_balance",
            "final_equity",
            "total_return",
            "total_return_pct",
            "benchmark_return",
            "max_drawdown",
            "sharpe_ratio",
            "calmar_ratio",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
        ]

        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"


class TestSignalDistribution:
    """测试信号分布"""

    def test_signals_have_variety(self, jesse_candles):
        """测试信号有多样性 (不全是同一个方向)"""
        warmup_candles, trading_candles = jesse_candles

        # 使用足够数据确保生成足够的 fusion bars (某些特征需要 512 窗口)
        warmup = warmup_candles  # 40000 根 warmup
        trading = trading_candles  # 全部 trading

        from backtest_vectorized_no_jesse import (
            generate_all_fusion_bars_with_split,
            compute_features_vectorized,
            predict_all_models,
            aggregate_votes,
        )

        feature_info_path = (
            ROOT_DIR
            / "strategies"
            / "BinanceBtcDemoBarV2"
            / "models"
            / "feature_info.json"
        )
        with open(feature_info_path) as f:
            feature_info = json.load(f)

        fusion_bars, warmup_len = generate_all_fusion_bars_with_split(
            warmup, trading, max_bars=-1
        )

        df_features = compute_features_vectorized(
            fusion_bars,
            warmup_len,
            feature_info,
            TEST_MODELS,
        )

        predictions = predict_all_models(df_features, TEST_MODELS, feature_info)
        signals = aggregate_votes(predictions, TEST_MODELS)

        long_count = signals.count("long")
        short_count = signals.count("short")
        flat_count = signals.count("flat")

        print(f"\n[Signal] Long: {long_count}, Short: {short_count}, Flat: {flat_count}")

        # 至少应该有一些信号 (不全是 flat)
        assert long_count > 0 or short_count > 0, "No trading signals generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
