"""
Phase 5: 交易模拟测试

测试目标:
- Position 类: unrealized_pnl 计算, side 属性
- FastBacktester: open_long/short, close_position, check_stop_loss
- 手续费计算正确性
- 止损逻辑正确性 (使用 high/low 检查, 以止损价平仓)
- 同 bar 入场后立即止损检查
- 权益曲线记录逻辑

运行方式:
    pytest tests/test_backtest_vectorized/test_phase5_trading.py -v
"""

import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 导入被测类
from backtest_vectorized_no_jesse import (
    FastBacktester,
    Position,
    Trade,
    STOP_LOSS_RATIO_NO_LEVERAGE,
    POSITION_SIZE_RATIO,
)


class TestPositionClass:
    """测试 Position 数据类"""

    def test_initial_position_is_flat(self):
        """测试初始仓位为 flat"""
        pos = Position()
        assert pos.is_flat
        assert not pos.is_long
        assert not pos.is_short
        assert pos.side == "flat"
        assert pos.entry_price == 0.0
        assert pos.qty == 0.0

    def test_long_position_properties(self):
        """测试多头仓位属性"""
        pos = Position(side="long", entry_price=50000, qty=0.1, stop_loss_price=49000)
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat

    def test_short_position_properties(self):
        """测试空头仓位属性"""
        pos = Position(side="short", entry_price=50000, qty=0.1, stop_loss_price=51000)
        assert pos.is_short
        assert not pos.is_long
        assert not pos.is_flat

    def test_unrealized_pnl_long_profit(self):
        """测试多头仓位盈利计算"""
        pos = Position(side="long", entry_price=50000, qty=0.1)
        # 价格上涨到 51000
        pnl = pos.unrealized_pnl(51000)
        # (51000 - 50000) * 0.1 = 100
        assert pnl == pytest.approx(100.0)

    def test_unrealized_pnl_long_loss(self):
        """测试多头仓位亏损计算"""
        pos = Position(side="long", entry_price=50000, qty=0.1)
        # 价格下跌到 49000
        pnl = pos.unrealized_pnl(49000)
        # (49000 - 50000) * 0.1 = -100
        assert pnl == pytest.approx(-100.0)

    def test_unrealized_pnl_short_profit(self):
        """测试空头仓位盈利计算"""
        pos = Position(side="short", entry_price=50000, qty=0.1)
        # 价格下跌到 49000
        pnl = pos.unrealized_pnl(49000)
        # (50000 - 49000) * 0.1 = 100
        assert pnl == pytest.approx(100.0)

    def test_unrealized_pnl_short_loss(self):
        """测试空头仓位亏损计算"""
        pos = Position(side="short", entry_price=50000, qty=0.1)
        # 价格上涨到 51000
        pnl = pos.unrealized_pnl(51000)
        # (50000 - 51000) * 0.1 = -100
        assert pnl == pytest.approx(-100.0)

    def test_unrealized_pnl_flat_is_zero(self):
        """测试 flat 仓位无盈亏"""
        pos = Position()
        assert pos.unrealized_pnl(50000) == 0.0
        assert pos.unrealized_pnl(100000) == 0.0


class TestFastBacktesterOpenPosition:
    """测试开仓逻辑"""

    def test_open_long_fee_calculation(self):
        """测试开多仓手续费计算"""
        starting_balance = 10000
        fee_rate = 0.0004
        leverage = 1

        bt = FastBacktester(starting_balance, fee_rate, leverage)
        price = 50000

        bt.open_long(timestamp=1000, price=price, stop_loss_price=49000)

        # 验证仓位
        assert bt.position.is_long
        assert bt.position.entry_price == price

        # 验证数量计算:
        # available = 10000 * 1 * 0.95 = 9500
        # qty = 9500 / 50000 / (1 + 0.0004) = 0.189924...
        available = starting_balance * leverage * POSITION_SIZE_RATIO
        expected_qty = available / price / (1 + fee_rate)
        assert bt.position.qty == pytest.approx(expected_qty, rel=1e-6)

        # 验证手续费:
        # fee = qty * price * fee_rate
        expected_fee = expected_qty * price * fee_rate
        assert bt.trades[0].fee == pytest.approx(expected_fee, rel=1e-6)

        # 验证余额减少手续费
        assert bt.balance == pytest.approx(starting_balance - expected_fee, rel=1e-6)

    def test_open_short_fee_calculation(self):
        """测试开空仓手续费计算"""
        starting_balance = 10000
        fee_rate = 0.0004
        leverage = 3

        bt = FastBacktester(starting_balance, fee_rate, leverage)
        price = 50000

        bt.open_short(timestamp=1000, price=price, stop_loss_price=51000)

        # 验证仓位
        assert bt.position.is_short
        assert bt.position.entry_price == price

        # 验证数量计算 (带杠杆):
        # available = 10000 * 3 * 0.95 = 28500
        available = starting_balance * leverage * POSITION_SIZE_RATIO
        expected_qty = available / price / (1 + fee_rate)
        assert bt.position.qty == pytest.approx(expected_qty, rel=1e-6)

        # 验证手续费
        expected_fee = expected_qty * price * fee_rate
        assert bt.trades[0].fee == pytest.approx(expected_fee, rel=1e-6)

    def test_cannot_open_when_in_position(self):
        """测试已有仓位时不能再开仓"""
        bt = FastBacktester(10000, 0.0004, 1)

        bt.open_long(timestamp=1000, price=50000, stop_loss_price=49000)

        with pytest.raises(AssertionError):
            bt.open_long(timestamp=2000, price=51000, stop_loss_price=50000)

        with pytest.raises(AssertionError):
            bt.open_short(timestamp=2000, price=51000, stop_loss_price=52000)


class TestFastBacktesterClosePosition:
    """测试平仓逻辑"""

    def test_close_long_profit(self):
        """测试平多仓盈利"""
        starting_balance = 10000
        fee_rate = 0.0004
        bt = FastBacktester(starting_balance, fee_rate, leverage=1)

        entry_price = 50000
        bt.open_long(timestamp=1000, price=entry_price, stop_loss_price=49000)

        balance_after_open = bt.balance
        qty = bt.position.qty

        # 价格上涨 2%
        exit_price = 51000
        bt.close_position(timestamp=2000, price=exit_price, reason="close")

        # 验证仓位已清空
        assert bt.position.is_flat

        # 验证平仓盈亏计算:
        # unrealized_pnl = (51000 - 50000) * qty
        # close_fee = qty * 51000 * fee_rate
        # pnl = unrealized_pnl - close_fee
        unrealized_pnl = (exit_price - entry_price) * qty
        close_fee = qty * exit_price * fee_rate
        expected_pnl = unrealized_pnl - close_fee

        assert bt.trades[1].pnl == pytest.approx(expected_pnl, rel=1e-6)
        assert bt.trades[1].action == "close_long"

        # 验证余额更新
        expected_balance = balance_after_open + expected_pnl
        assert bt.balance == pytest.approx(expected_balance, rel=1e-6)

    def test_close_short_profit(self):
        """测试平空仓盈利"""
        starting_balance = 10000
        fee_rate = 0.0004
        bt = FastBacktester(starting_balance, fee_rate, leverage=1)

        entry_price = 50000
        bt.open_short(timestamp=1000, price=entry_price, stop_loss_price=51000)

        balance_after_open = bt.balance
        qty = bt.position.qty

        # 价格下跌 2%
        exit_price = 49000
        bt.close_position(timestamp=2000, price=exit_price, reason="close")

        # 验证平仓盈亏
        unrealized_pnl = (entry_price - exit_price) * qty
        close_fee = qty * exit_price * fee_rate
        expected_pnl = unrealized_pnl - close_fee

        assert bt.trades[1].pnl == pytest.approx(expected_pnl, rel=1e-6)
        assert bt.trades[1].action == "close_short"

    def test_close_long_loss(self):
        """测试平多仓亏损"""
        starting_balance = 10000
        fee_rate = 0.0004
        bt = FastBacktester(starting_balance, fee_rate, leverage=1)

        entry_price = 50000
        bt.open_long(timestamp=1000, price=entry_price, stop_loss_price=49000)

        qty = bt.position.qty

        # 价格下跌
        exit_price = 49000
        bt.close_position(timestamp=2000, price=exit_price, reason="close")

        unrealized_pnl = (exit_price - entry_price) * qty  # 负值
        close_fee = qty * exit_price * fee_rate
        expected_pnl = unrealized_pnl - close_fee

        assert expected_pnl < 0  # 确认是亏损
        assert bt.trades[1].pnl == pytest.approx(expected_pnl, rel=1e-6)

    def test_cannot_close_when_flat(self):
        """测试无仓位时不能平仓"""
        bt = FastBacktester(10000, 0.0004, 1)

        with pytest.raises(AssertionError):
            bt.close_position(timestamp=1000, price=50000)


class TestStopLoss:
    """测试止损逻辑"""

    def test_long_stop_loss_triggered_by_low(self):
        """测试多头止损: low 触及止损价"""
        bt = FastBacktester(10000, 0.0004, leverage=3)

        entry_price = 50000
        stop_loss_price = entry_price * (1 - STOP_LOSS_RATIO_NO_LEVERAGE / 3)
        # stop_loss_price ≈ 50000 * (1 - 0.05/3) ≈ 49166.67

        bt.open_long(timestamp=1000, price=entry_price, stop_loss_price=stop_loss_price)

        # Bar: low 触及止损价
        triggered = bt.check_stop_loss(
            timestamp=2000, high=50500, low=49000, close=49500  # low < stop_loss_price
        )

        assert triggered
        assert bt.position.is_flat
        # 以止损价平仓，不是以 close 价
        assert bt.trades[1].price == pytest.approx(stop_loss_price)
        assert "stop_loss" in bt.trades[1].action

    def test_long_stop_loss_not_triggered(self):
        """测试多头止损: low 未触及止损价"""
        bt = FastBacktester(10000, 0.0004, leverage=3)

        entry_price = 50000
        stop_loss_price = 49000

        bt.open_long(timestamp=1000, price=entry_price, stop_loss_price=stop_loss_price)

        # Bar: low 未触及止损价
        triggered = bt.check_stop_loss(
            timestamp=2000, high=50500, low=49500, close=50200  # low > stop_loss_price
        )

        assert not triggered
        assert bt.position.is_long

    def test_short_stop_loss_triggered_by_high(self):
        """测试空头止损: high 触及止损价"""
        bt = FastBacktester(10000, 0.0004, leverage=3)

        entry_price = 50000
        stop_loss_price = entry_price * (1 + STOP_LOSS_RATIO_NO_LEVERAGE / 3)
        # stop_loss_price ≈ 50000 * (1 + 0.05/3) ≈ 50833.33

        bt.open_short(timestamp=1000, price=entry_price, stop_loss_price=stop_loss_price)

        # Bar: high 触及止损价
        triggered = bt.check_stop_loss(
            timestamp=2000, high=51000, low=49500, close=50500  # high > stop_loss_price
        )

        assert triggered
        assert bt.position.is_flat
        assert bt.trades[1].price == pytest.approx(stop_loss_price)
        assert "stop_loss" in bt.trades[1].action

    def test_short_stop_loss_not_triggered(self):
        """测试空头止损: high 未触及止损价"""
        bt = FastBacktester(10000, 0.0004, leverage=3)

        entry_price = 50000
        stop_loss_price = 51000

        bt.open_short(timestamp=1000, price=entry_price, stop_loss_price=stop_loss_price)

        # Bar: high 未触及止损价
        triggered = bt.check_stop_loss(
            timestamp=2000, high=50500, low=49500, close=50200  # high < stop_loss_price
        )

        assert not triggered
        assert bt.position.is_short

    def test_stop_loss_on_flat_returns_false(self):
        """测试无仓位时检查止损返回 False"""
        bt = FastBacktester(10000, 0.0004, 1)

        triggered = bt.check_stop_loss(timestamp=1000, high=51000, low=49000, close=50000)

        assert not triggered


class TestEntryBarStopLoss:
    """测试入场 bar 立即止损检查 (跳空场景)"""

    def test_long_entry_immediate_stop_loss(self):
        """测试多头入场后同 bar 止损 (跳空低开)"""
        bt = FastBacktester(10000, 0.0004, leverage=3)

        entry_price = 50000
        stop_loss_price = 49000

        bt.open_long(timestamp=1000, price=entry_price, stop_loss_price=stop_loss_price)

        # 同 bar 的 low 跳空低于止损价
        triggered = bt.check_stop_loss(
            timestamp=1000,  # 同一时间戳
            high=50200,
            low=48500,  # 跳空低于止损价
            close=49200,
        )

        assert triggered
        assert bt.position.is_flat
        # 止损价平仓
        assert bt.trades[1].price == stop_loss_price

    def test_short_entry_immediate_stop_loss(self):
        """测试空头入场后同 bar 止损 (跳空高开)"""
        bt = FastBacktester(10000, 0.0004, leverage=3)

        entry_price = 50000
        stop_loss_price = 51000

        bt.open_short(timestamp=1000, price=entry_price, stop_loss_price=stop_loss_price)

        # 同 bar 的 high 跳空高于止损价
        triggered = bt.check_stop_loss(
            timestamp=1000,
            high=51500,  # 跳空高于止损价
            low=49800,
            close=50800,
        )

        assert triggered
        assert bt.position.is_flat
        assert bt.trades[1].price == stop_loss_price


class TestEquityCalculation:
    """测试权益计算"""

    def test_equity_with_long_position(self):
        """测试持多仓时的权益计算"""
        starting_balance = 10000
        bt = FastBacktester(starting_balance, 0.0004, leverage=1)

        bt.open_long(timestamp=1000, price=50000, stop_loss_price=49000)
        balance_after_open = bt.balance

        # 当前价格 51000
        equity = bt.equity(51000)

        # equity = balance + unrealized_pnl
        unrealized_pnl = bt.position.unrealized_pnl(51000)
        expected_equity = balance_after_open + unrealized_pnl

        assert equity == pytest.approx(expected_equity)

    def test_equity_flat_equals_balance(self):
        """测试无仓位时权益等于余额"""
        bt = FastBacktester(10000, 0.0004, 1)

        assert bt.equity(50000) == bt.balance
        assert bt.equity(100000) == bt.balance

    def test_record_equity_creates_point(self):
        """测试记录权益曲线"""
        bt = FastBacktester(10000, 0.0004, 1)

        bt.record_equity(timestamp=1000, current_price=50000)
        bt.record_equity(timestamp=2000, current_price=51000)

        assert len(bt.equity_curve) == 2
        assert bt.equity_curve[0].timestamp == 1000
        assert bt.equity_curve[1].timestamp == 2000


class TestBenchmark:
    """测试 Buy & Hold 基准"""

    def test_benchmark_initialization(self):
        """测试基准初始化"""
        starting_balance = 10000
        fee_rate = 0.0004
        bt = FastBacktester(starting_balance, fee_rate, 1)

        price = 50000
        bt.init_benchmark(price)

        # 基准买入全部资金的币
        # fee = 10000 * 0.0004 = 4
        # qty = (10000 - 4) / 50000 = 0.19992
        expected_fee = starting_balance * fee_rate
        expected_qty = (starting_balance - expected_fee) / price

        assert bt.benchmark_entry_price == price
        assert bt.benchmark_qty == pytest.approx(expected_qty)

    def test_benchmark_equity_calculation(self):
        """测试基准权益计算"""
        starting_balance = 10000
        fee_rate = 0.0004
        bt = FastBacktester(starting_balance, fee_rate, 1)

        entry_price = 50000
        bt.init_benchmark(entry_price)

        # 价格上涨 10%
        current_price = 55000
        equity = bt.benchmark_equity(current_price)

        # benchmark_equity = starting_balance + (current - entry) * qty
        expected = starting_balance + (current_price - entry_price) * bt.benchmark_qty

        assert equity == pytest.approx(expected)

    def test_benchmark_not_reinitialized(self):
        """测试基准不会被重复初始化"""
        bt = FastBacktester(10000, 0.0004, 1)

        bt.init_benchmark(50000)
        first_qty = bt.benchmark_qty

        # 再次调用不会改变
        bt.init_benchmark(60000)

        assert bt.benchmark_entry_price == 50000
        assert bt.benchmark_qty == first_qty


class TestTradeRecording:
    """测试交易记录"""

    def test_trade_record_on_open(self):
        """测试开仓时记录交易"""
        bt = FastBacktester(10000, 0.0004, 1)

        bt.open_long(timestamp=1000, price=50000, stop_loss_price=49000)

        assert len(bt.trades) == 1
        trade = bt.trades[0]
        assert trade.action == "open_long"
        assert trade.timestamp == 1000
        assert trade.price == 50000
        assert trade.pnl == 0.0  # 开仓无盈亏

    def test_trade_record_on_close(self):
        """测试平仓时记录交易"""
        bt = FastBacktester(10000, 0.0004, 1)

        bt.open_short(timestamp=1000, price=50000, stop_loss_price=51000)
        bt.close_position(timestamp=2000, price=49000, reason="close")

        assert len(bt.trades) == 2
        close_trade = bt.trades[1]
        assert close_trade.action == "close_short"
        assert close_trade.pnl != 0.0
        assert close_trade.balance == bt.balance

    def test_trade_to_dict(self):
        """测试交易记录转字典"""
        trade = Trade(
            timestamp=1000,
            action="open_long",
            price=50000,
            qty=0.1,
            fee=2.0,
            pnl=0.0,
            balance=9998.0,
        )

        d = trade.to_dict()

        assert d["timestamp"] == 1000
        assert d["action"] == "open_long"
        assert d["price"] == 50000
        assert d["qty"] == 0.1
        assert d["fee"] == 2.0


class TestLeverageEffect:
    """测试杠杆效果"""

    def test_leverage_increases_position_size(self):
        """测试杠杆增加仓位大小"""
        price = 50000
        fee_rate = 0.0004

        bt1 = FastBacktester(10000, fee_rate, leverage=1)
        bt1.open_long(timestamp=1000, price=price, stop_loss_price=49000)
        qty_1x = bt1.position.qty

        bt3 = FastBacktester(10000, fee_rate, leverage=3)
        bt3.open_long(timestamp=1000, price=price, stop_loss_price=49000)
        qty_3x = bt3.position.qty

        # 3x 杠杆应该是 1x 的 3 倍仓位
        assert qty_3x == pytest.approx(qty_1x * 3, rel=1e-6)

    def test_leverage_affects_pnl(self):
        """测试杠杆影响盈亏"""
        entry_price = 50000
        exit_price = 51000  # +2%
        fee_rate = 0.0004

        # 1x 杠杆
        bt1 = FastBacktester(10000, fee_rate, leverage=1)
        bt1.open_long(timestamp=1000, price=entry_price, stop_loss_price=49000)
        bt1.close_position(timestamp=2000, price=exit_price)
        pnl_1x = bt1.trades[1].pnl

        # 3x 杠杆
        bt3 = FastBacktester(10000, fee_rate, leverage=3)
        bt3.open_long(timestamp=1000, price=entry_price, stop_loss_price=49000)
        bt3.close_position(timestamp=2000, price=exit_price)
        pnl_3x = bt3.trades[1].pnl

        # 杠杆放大盈亏（大约 3 倍，但手续费也放大）
        assert pnl_3x > pnl_1x * 2.9  # 至少 2.9 倍


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
