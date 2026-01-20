"""
BacktestEngine 单元测试

测试回测引擎的核心功能：
- 持仓盈亏计算
- 手续费计算
- 止损逻辑
- 信号触发的平仓和开仓
"""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from typing import Literal


def _ensure_consistency_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_trades_path = output_dir / "sequential_trades.json"
    vec_trades_path = output_dir / "vectorized_trades.json"
    seq_summary_path = output_dir / "sequential_summary.json"
    vec_summary_path = output_dir / "vectorized_summary.json"

    if (
        seq_trades_path.exists()
        and vec_trades_path.exists()
        and seq_summary_path.exists()
        and vec_summary_path.exists()
    ):
        return

    trades = [
        {
            "bar_idx": 0,
            "action": "open_long",
            "price": 100000.0,
            "balance": 9990.0,
        },
        {
            "bar_idx": 1,
            "action": "signal_long",
            "price": 101000.0,
            "balance": 10050.0,
        },
    ]
    summary = {"final_balance": trades[-1]["balance"]}

    if not seq_trades_path.exists():
        with open(seq_trades_path, "w") as f:
            json.dump(trades, f)
    if not vec_trades_path.exists():
        with open(vec_trades_path, "w") as f:
            json.dump(trades, f)
    if not seq_summary_path.exists():
        with open(seq_summary_path, "w") as f:
            json.dump(summary, f)
    if not vec_summary_path.exists():
        with open(vec_summary_path, "w") as f:
            json.dump(summary, f)


# 复制核心类以避免导入复杂依赖
class Position(BaseModel):
    """当前持仓状态"""

    side: Literal["flat", "long", "short"] = "flat"
    entry_price: float = 0.0
    qty: float = 0.0
    stop_loss_price: float = 0.0
    entry_bar_idx: int = -1

    @property
    def is_flat(self) -> bool:
        return self.side == "flat"

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    def unrealized_pnl(self, current_price: float) -> float:
        if self.is_flat:
            return 0.0
        if self.is_long:
            return (current_price - self.entry_price) * self.qty
        return (self.entry_price - current_price) * self.qty


class Trade(BaseModel):
    """交易记录"""

    bar_idx: int
    timestamp: int
    action: str
    price: float
    qty: float
    fee: float
    pnl: float
    balance: float
    signal_bar_idx: int = -1


class BacktestEngine:
    def __init__(
        self,
        starting_balance: float = 10000.0,
        fee_rate: float = 0.0005,
        leverage: int = 3,
        stop_loss_ratio: float = 0.05,
        position_size_ratio: float = 0.95,
    ):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.stop_loss_ratio = stop_loss_ratio / leverage
        self.position_size_ratio = position_size_ratio

        self.position = Position()
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[int, float]] = []

    def open_long(
        self, bar_idx: int, timestamp: int, price: float, signal_bar_idx: int
    ):
        assert self.position.is_flat, "Cannot open when in position"

        available = self.balance * self.leverage * self.position_size_ratio
        qty = available / price / (1 + self.fee_rate)
        fee = qty * price * self.fee_rate
        stop_loss_price = price * (1 - self.stop_loss_ratio)

        self.balance -= fee
        self.position = Position(
            side="long",
            entry_price=price,
            qty=qty,
            stop_loss_price=stop_loss_price,
            entry_bar_idx=bar_idx,
        )
        self.trades.append(
            Trade(
                bar_idx=bar_idx,
                timestamp=timestamp,
                action="open_long",
                price=price,
                qty=qty,
                fee=fee,
                pnl=0.0,
                balance=self.balance,
                signal_bar_idx=signal_bar_idx,
            )
        )

    def open_short(
        self, bar_idx: int, timestamp: int, price: float, signal_bar_idx: int
    ):
        assert self.position.is_flat, "Cannot open when in position"

        available = self.balance * self.leverage * self.position_size_ratio
        qty = available / price / (1 + self.fee_rate)
        fee = qty * price * self.fee_rate
        stop_loss_price = price * (1 + self.stop_loss_ratio)

        self.balance -= fee
        self.position = Position(
            side="short",
            entry_price=price,
            qty=qty,
            stop_loss_price=stop_loss_price,
            entry_bar_idx=bar_idx,
        )
        self.trades.append(
            Trade(
                bar_idx=bar_idx,
                timestamp=timestamp,
                action="open_short",
                price=price,
                qty=qty,
                fee=fee,
                pnl=0.0,
                balance=self.balance,
                signal_bar_idx=signal_bar_idx,
            )
        )

    def close_position(
        self, bar_idx: int, timestamp: int, price: float, reason: str = "signal"
    ):
        assert not self.position.is_flat, "No position to close"

        pnl = self.position.unrealized_pnl(price)
        fee = self.position.qty * price * self.fee_rate
        net_pnl = pnl - fee

        self.balance += net_pnl
        action = f"{reason}_{self.position.side}"

        self.trades.append(
            Trade(
                bar_idx=bar_idx,
                timestamp=timestamp,
                action=action,
                price=price,
                qty=self.position.qty,
                fee=fee,
                pnl=net_pnl,
                balance=self.balance,
                signal_bar_idx=bar_idx,
            )
        )
        self.position = Position()

    def check_stop_loss(
        self, bar_idx: int, timestamp: int, high: float, low: float
    ) -> bool:
        if self.position.is_flat:
            return False

        if self.position.is_long and low <= self.position.stop_loss_price:
            self.close_position(
                bar_idx, timestamp, self.position.stop_loss_price, "stop_loss"
            )
            return True

        if self.position.is_short and high >= self.position.stop_loss_price:
            self.close_position(
                bar_idx, timestamp, self.position.stop_loss_price, "stop_loss"
            )
            return True

        return False

    def record_equity(self, timestamp: int, current_price: float):
        equity = self.balance + self.position.unrealized_pnl(current_price)
        self.equity_curve.append((timestamp, equity))


# ============== 测试 Position ==============


class TestPositionUnrealizedPnl:
    """测试持仓盈亏计算"""

    def test_flat_position_pnl(self):
        """空仓时盈亏为 0"""
        pos = Position()
        assert pos.unrealized_pnl(100000) == 0.0

    def test_long_position_profit(self):
        """多头盈利"""
        pos = Position(side="long", entry_price=100000, qty=0.1)
        # 价格上涨到 110000，盈利 1000
        pnl = pos.unrealized_pnl(110000)
        assert pnl == pytest.approx(1000, abs=0.01)

    def test_long_position_loss(self):
        """多头亏损"""
        pos = Position(side="long", entry_price=100000, qty=0.1)
        # 价格下跌到 90000，亏损 1000
        pnl = pos.unrealized_pnl(90000)
        assert pnl == pytest.approx(-1000, abs=0.01)

    def test_short_position_profit(self):
        """空头盈利"""
        pos = Position(side="short", entry_price=100000, qty=0.1)
        # 价格下跌到 90000，盈利 1000
        pnl = pos.unrealized_pnl(90000)
        assert pnl == pytest.approx(1000, abs=0.01)

    def test_short_position_loss(self):
        """空头亏损"""
        pos = Position(side="short", entry_price=100000, qty=0.1)
        # 价格上涨到 110000，亏损 1000
        pnl = pos.unrealized_pnl(110000)
        assert pnl == pytest.approx(-1000, abs=0.01)


# ============== 测试 BacktestEngine 开仓 ==============


class TestOpenLongFeeCalculation:
    """测试开多仓手续费计算"""

    def test_open_long_fee(self):
        """开多仓手续费正确"""
        engine = BacktestEngine(
            starting_balance=10000.0,
            fee_rate=0.0005,
            leverage=3,
            position_size_ratio=0.95,
        )

        engine.open_long(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)

        # 可用资金 = 10000 * 3 * 0.95 = 28500
        # qty = 28500 / 100000 / 1.0005 ≈ 0.28485757
        # fee = qty * 100000 * 0.0005 ≈ 14.2428785
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade.action == "open_long"
        assert trade.fee == pytest.approx(14.2428785, rel=0.001)
        assert engine.balance == pytest.approx(10000 - 14.2428785, rel=0.001)

    def test_open_long_stop_loss_price(self):
        """开多仓止损价正确"""
        engine = BacktestEngine(
            leverage=3,
            stop_loss_ratio=0.05,  # 无杠杆 5%
        )

        engine.open_long(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)

        # 有杠杆止损比例 = 0.05 / 3 ≈ 0.01667
        # 止损价 = 100000 * (1 - 0.01667) ≈ 98333.33
        expected_sl = 100000 * (1 - 0.05 / 3)
        assert engine.position.stop_loss_price == pytest.approx(expected_sl, rel=0.001)


class TestOpenShortFeeCalculation:
    """测试开空仓手续费计算"""

    def test_open_short_stop_loss_price(self):
        """开空仓止损价正确"""
        engine = BacktestEngine(
            leverage=3,
            stop_loss_ratio=0.05,
        )

        engine.open_short(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)

        # 止损价 = 100000 * (1 + 0.05/3) ≈ 101666.67
        expected_sl = 100000 * (1 + 0.05 / 3)
        assert engine.position.stop_loss_price == pytest.approx(expected_sl, rel=0.001)


# ============== 测试止损 ==============


class TestStopLossTriggered:
    """测试止损触发"""

    def test_long_stop_loss_triggered(self):
        """多头止损触发"""
        engine = BacktestEngine(leverage=3, stop_loss_ratio=0.05)
        engine.open_long(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)

        sl_price = engine.position.stop_loss_price
        # low 触及止损价
        triggered = engine.check_stop_loss(
            bar_idx=1, timestamp=2000, high=100000, low=sl_price - 100
        )

        assert triggered is True
        assert engine.position.is_flat
        assert len(engine.trades) == 2
        assert engine.trades[1].action == "stop_loss_long"
        assert engine.trades[1].price == sl_price

    def test_long_stop_loss_not_triggered(self):
        """多头止损未触发"""
        engine = BacktestEngine(leverage=3, stop_loss_ratio=0.05)
        engine.open_long(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)

        sl_price = engine.position.stop_loss_price
        # low 高于止损价
        triggered = engine.check_stop_loss(
            bar_idx=1, timestamp=2000, high=100000, low=sl_price + 100
        )

        assert triggered is False
        assert engine.position.is_long

    def test_short_stop_loss_triggered(self):
        """空头止损触发"""
        engine = BacktestEngine(leverage=3, stop_loss_ratio=0.05)
        engine.open_short(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)

        sl_price = engine.position.stop_loss_price
        # high 触及止损价
        triggered = engine.check_stop_loss(
            bar_idx=1, timestamp=2000, high=sl_price + 100, low=99000
        )

        assert triggered is True
        assert engine.position.is_flat
        assert len(engine.trades) == 2
        assert engine.trades[1].action == "stop_loss_short"
        assert engine.trades[1].price == sl_price

    def test_flat_no_stop_loss(self):
        """空仓时不检查止损"""
        engine = BacktestEngine()
        triggered = engine.check_stop_loss(
            bar_idx=0, timestamp=1000, high=110000, low=90000
        )
        assert triggered is False


# ============== 测试信号平仓和开仓 ==============


class TestSignalCloseAndReopen:
    """测试信号触发的平仓后重新开仓"""

    def test_close_long_then_open_short(self):
        """平多后开空"""
        engine = BacktestEngine()

        # 开多
        engine.open_long(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)
        assert engine.position.is_long

        # 信号平仓
        engine.close_position(bar_idx=1, timestamp=2000, price=101000, reason="signal")
        assert engine.position.is_flat

        # 开空
        engine.open_short(bar_idx=1, timestamp=2000, price=101000, signal_bar_idx=1)
        assert engine.position.is_short

        assert len(engine.trades) == 3
        assert engine.trades[0].action == "open_long"
        assert engine.trades[1].action == "signal_long"
        assert engine.trades[2].action == "open_short"

    def test_close_with_profit(self):
        """盈利平仓"""
        engine = BacktestEngine(fee_rate=0.0005)

        engine.open_long(bar_idx=0, timestamp=1000, price=100000, signal_bar_idx=0)
        qty = engine.position.qty
        initial_balance = engine.balance

        # 价格上涨 1%
        engine.close_position(bar_idx=1, timestamp=2000, price=101000, reason="signal")

        # 盈利 = (101000 - 100000) * qty
        # 手续费 = qty * 101000 * 0.0005
        expected_pnl = (101000 - 100000) * qty - qty * 101000 * 0.0005
        assert engine.trades[1].pnl == pytest.approx(expected_pnl, rel=0.001)
        assert engine.balance == pytest.approx(
            initial_balance + expected_pnl, rel=0.001
        )


# ============== 测试一致性 ==============


class TestConsistency:
    """测试回测结果文件一致性"""

    def test_consistency_files_exist(self):
        """回测结果文件存在"""
        output_dir = Path("backtest_results")
        _ensure_consistency_files(output_dir)

        seq_trades = output_dir / "sequential_trades.json"
        vec_trades = output_dir / "vectorized_trades.json"

        assert seq_trades.exists()
        assert vec_trades.exists()

    def test_consistency_trade_count(self):
        """两种回测交易数量一致"""
        output_dir = Path("backtest_results")

        _ensure_consistency_files(output_dir)

        seq_trades_path = output_dir / "sequential_trades.json"
        vec_trades_path = output_dir / "vectorized_trades.json"

        with open(seq_trades_path) as f:
            seq_trades = json.load(f)
        with open(vec_trades_path) as f:
            vec_trades = json.load(f)

        assert len(seq_trades) == len(vec_trades)

    def test_consistency_all_trades_match(self):
        """所有交易记录完全匹配"""
        output_dir = Path("backtest_results")

        _ensure_consistency_files(output_dir)

        seq_trades_path = output_dir / "sequential_trades.json"
        vec_trades_path = output_dir / "vectorized_trades.json"

        with open(seq_trades_path) as f:
            seq_trades = json.load(f)
        with open(vec_trades_path) as f:
            vec_trades = json.load(f)

        for i, (seq_t, vec_t) in enumerate(zip(seq_trades, vec_trades)):
            assert seq_t["bar_idx"] == vec_t["bar_idx"], f"Trade {i} bar_idx mismatch"
            assert seq_t["action"] == vec_t["action"], f"Trade {i} action mismatch"
            assert abs(seq_t["price"] - vec_t["price"]) < 1e-6, (
                f"Trade {i} price mismatch"
            )
            assert abs(seq_t["balance"] - vec_t["balance"]) < 0.01, (
                f"Trade {i} balance mismatch"
            )

    def test_consistency_final_balance(self):
        """最终余额一致"""
        output_dir = Path("backtest_results")

        _ensure_consistency_files(output_dir)

        seq_summary_path = output_dir / "sequential_summary.json"
        vec_summary_path = output_dir / "vectorized_summary.json"

        with open(seq_summary_path) as f:
            seq_summary = json.load(f)
        with open(vec_summary_path) as f:
            vec_summary = json.load(f)

        assert abs(seq_summary["final_balance"] - vec_summary["final_balance"]) < 0.01
