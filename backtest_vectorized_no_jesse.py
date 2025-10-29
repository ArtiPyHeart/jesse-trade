"""
向量化快速回测引擎 - 不依赖 Jesse 库

核心改进：
1. 一次性生成所有 fusion bars（避免逐根 candle 更新）
2. 批量计算所有特征（sequential=True）
3. SSM 使用 transform（warmup 后）替代逐行 inference
4. 预先计算所有预测结果，交易模拟时直接查表

性能提升：
- 特征计算阶段：~10-50x 加速
- 预期整体：5-20x 加速
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.bars.fusion.demo import DemoBar
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from strategies.BinanceBtcDemoBarV2.models.config import (
    model_name_to_params,
    LGBMContainer,
    SSMContainer,
)

# ==================== 配置常量 ====================
STOP_LOSS_RATIO_NO_LEVERAGE = 0.05
POSITION_SIZE_RATIO = 0.95


# ==================== 数据类 ====================
@dataclass
class Trade:
    """交易记录"""

    timestamp: int  # 毫秒时间戳
    action: str  # open_long, open_short, close_long, close_short, stop_loss
    price: float
    qty: float
    fee: float
    pnl: float = 0.0  # 本次交易的盈亏（仅平仓时有值）
    balance: float = 0.0  # 交易后的余额

    def to_dict(self):
        return asdict(self)


@dataclass
class Position:
    """仓位信息"""

    side: Literal["long", "short", "flat"] = "flat"
    entry_price: float = 0.0
    qty: float = 0.0
    stop_loss_price: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    @property
    def is_flat(self) -> bool:
        return self.side == "flat"

    def unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.is_flat:
            return 0.0
        if self.is_long:
            return (current_price - self.entry_price) * self.qty
        else:  # short
            return (self.entry_price - current_price) * self.qty


@dataclass
class EquityPoint:
    """权益曲线点"""

    timestamp: int
    equity: float
    benchmark_equity: float  # Buy & Hold 基准


# ==================== 回测引擎 ====================
class FastBacktester:
    """快速回测引擎"""

    def __init__(
        self,
        starting_balance: float,
        fee_rate: float,
        leverage: int = 1,
    ):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.fee_rate = fee_rate
        self.leverage = leverage

        # 仓位
        self.position = Position()

        # 记录
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []

        # Buy & Hold 基准
        self.benchmark_entry_price: Optional[float] = None
        self.benchmark_qty: float = 0.0

    @property
    def available_margin(self) -> float:
        """可用保证金"""
        return self.balance * self.leverage

    def equity(self, current_price: float) -> float:
        """当前权益"""
        return self.balance + self.position.unrealized_pnl(current_price)

    def benchmark_equity(self, current_price: float) -> float:
        """Buy & Hold 基准权益"""
        if self.benchmark_entry_price is None:
            return self.starting_balance
        return (
            self.starting_balance
            + (current_price - self.benchmark_entry_price) * self.benchmark_qty
        )

    def init_benchmark(self, price: float):
        """初始化 Buy & Hold 基准"""
        if self.benchmark_entry_price is None:
            self.benchmark_entry_price = price
            # 买入全部资金的币
            fee = self.starting_balance * self.fee_rate
            self.benchmark_qty = (self.starting_balance - fee) / price

    def open_long(self, timestamp: int, price: float, stop_loss_price: float):
        """开多仓"""
        assert self.position.is_flat, "Cannot open long when already in position"

        # 计算仓位大小
        available = self.available_margin * POSITION_SIZE_RATIO
        qty = available / price / (1 + self.fee_rate)

        # 计算手续费
        fee = qty * price * self.fee_rate

        # 更新余额
        self.balance -= fee

        # 更新仓位
        self.position.side = "long"
        self.position.entry_price = price
        self.position.qty = qty
        self.position.stop_loss_price = stop_loss_price

        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action="open_long",
            price=price,
            qty=qty,
            fee=fee,
            balance=self.balance,
        )
        self.trades.append(trade)

    def open_short(self, timestamp: int, price: float, stop_loss_price: float):
        """开空仓"""
        assert self.position.is_flat, "Cannot open short when already in position"

        # 计算仓位大小
        available = self.available_margin * POSITION_SIZE_RATIO
        qty = available / price / (1 + self.fee_rate)

        # 计算手续费
        fee = qty * price * self.fee_rate

        # 更新余额
        self.balance -= fee

        # 更新仓位
        self.position.side = "short"
        self.position.entry_price = price
        self.position.qty = qty
        self.position.stop_loss_price = stop_loss_price

        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action="open_short",
            price=price,
            qty=qty,
            fee=fee,
            balance=self.balance,
        )
        self.trades.append(trade)

    def close_position(self, timestamp: int, price: float, reason: str = "close"):
        """平仓"""
        assert not self.position.is_flat, "Cannot close position when flat"

        # 计算手续费和盈亏
        fee = self.position.qty * price * self.fee_rate
        pnl = self.position.unrealized_pnl(price) - fee

        # 更新余额
        self.balance += pnl + fee  # pnl已经减去了fee，所以这里加回fee再减
        self.balance -= fee

        # 记录交易
        action = f"{reason}_{self.position.side}"
        trade = Trade(
            timestamp=timestamp,
            action=action,
            price=price,
            qty=self.position.qty,
            fee=fee,
            pnl=pnl,
            balance=self.balance,
        )
        self.trades.append(trade)

        # 清空仓位
        self.position = Position()

    def check_stop_loss(self, timestamp: int, current_price: float) -> bool:
        """检查止损，如果触发则平仓"""
        if self.position.is_flat:
            return False

        triggered = False
        if self.position.is_long and current_price <= self.position.stop_loss_price:
            triggered = True
        elif self.position.is_short and current_price >= self.position.stop_loss_price:
            triggered = True

        if triggered:
            self.close_position(timestamp, current_price, reason="stop_loss")
            return True

        return False

    def record_equity(self, timestamp: int, current_price: float):
        """记录权益曲线"""
        point = EquityPoint(
            timestamp=timestamp,
            equity=self.equity(current_price),
            benchmark_equity=self.benchmark_equity(current_price),
        )
        self.equity_curve.append(point)


# ==================== 结果分析 ====================
class BacktestAnalyzer:
    """回测结果分析器"""

    @staticmethod
    def calculate_metrics(
        trades: list[Trade], equity_curve: list[EquityPoint], starting_balance: float
    ) -> dict:
        """计算统计指标"""
        if len(equity_curve) == 0:
            return {}

        # 提取权益数据
        equity_values = np.array([e.equity for e in equity_curve])
        benchmark_values = np.array([e.benchmark_equity for e in equity_curve])

        # 基础指标
        final_equity = equity_values[-1]
        total_return = (final_equity - starting_balance) / starting_balance
        benchmark_return = (benchmark_values[-1] - starting_balance) / starting_balance

        # 最大回撤
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # 收益率序列（日收益率近似）
        returns = np.diff(equity_values) / equity_values[:-1]

        # 夏普比率（年化）
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365)
        else:
            sharpe = 0.0

        # Calmar 比率
        if max_drawdown < 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = 0.0

        # 交易统计
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        total_trades = len([t for t in trades if t.pnl != 0])  # 只统计平仓交易
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

        profit_factor = (
            abs(
                sum([t.pnl for t in winning_trades])
                / sum([t.pnl for t in losing_trades])
            )
            if losing_trades
            else 0.0
        )

        return {
            "starting_balance": starting_balance,
            "final_equity": float(final_equity),
            "total_return": float(total_return),
            "total_return_pct": f"{total_return*100:.2f}%",
            "benchmark_return": float(benchmark_return),
            "benchmark_return_pct": f"{benchmark_return*100:.2f}%",
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": f"{max_drawdown*100:.2f}%",
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": float(win_rate),
            "win_rate_pct": f"{win_rate*100:.2f}%",
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
        }

    @staticmethod
    def plot_equity_curve(
        equity_curve: list[EquityPoint],
        starting_balance: float,
        output_path: Path,
        trades: list[Trade] = None,
    ):
        """
        绘制权益曲线对比图（含交易标记）

        Args:
            equity_curve: 权益曲线数据点
            starting_balance: 初始资金
            output_path: 输出路径
            trades: 交易记录列表（可选，用于在图上标注交易点）
        """
        if len(equity_curve) == 0:
            print("⚠️  无权益数据，跳过绘图")
            return

        # 配置中文字体
        plt.rcParams["font.sans-serif"] = [
            "Arial Unicode MS",
            "PingFang SC",
            "SimHei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 提取数据
        timestamps = [e.timestamp for e in equity_curve]
        equity_values = [e.equity for e in equity_curve]
        benchmark_values = [e.benchmark_equity for e in equity_curve]

        # 转换为 datetime
        dates = pd.to_datetime(timestamps, unit="ms")

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # ========== 上图：权益曲线对比 ==========
        ax1.plot(dates, equity_values, label="Strategy", linewidth=2, color="#2E86DE", zorder=1)
        ax1.plot(
            dates,
            benchmark_values,
            label="Buy & Hold",
            linewidth=2,
            linestyle="--",
            color="#EE5A6F",
            zorder=1,
        )
        ax1.axhline(
            y=starting_balance, color="gray", linestyle=":", alpha=0.5, label="Initial"
        )

        # ========== 标注交易点 ==========
        if trades:
            # 创建时间戳到权益值的映射（用于在交易时间点查找对应的权益）
            timestamp_to_equity = {ts: eq for ts, eq in zip(timestamps, equity_values)}

            # 分类交易（只保留开仓和止损）
            open_long_trades = []
            open_short_trades = []
            stop_loss_trades = []

            for trade in trades:
                trade_time = pd.to_datetime(trade.timestamp, unit="ms")
                # 找到最接近的权益值
                equity_at_trade = timestamp_to_equity.get(
                    trade.timestamp,
                    None  # 如果找不到精确匹配，使用 None
                )

                # 如果找不到精确匹配，插值估算
                if equity_at_trade is None:
                    # 找到最近的两个时间点进行线性插值
                    idx = np.searchsorted(timestamps, trade.timestamp)
                    if idx == 0:
                        equity_at_trade = equity_values[0]
                    elif idx >= len(timestamps):
                        equity_at_trade = equity_values[-1]
                    else:
                        # 线性插值
                        t0, t1 = timestamps[idx - 1], timestamps[idx]
                        e0, e1 = equity_values[idx - 1], equity_values[idx]
                        ratio = (trade.timestamp - t0) / (t1 - t0) if t1 != t0 else 0
                        equity_at_trade = e0 + ratio * (e1 - e0)

                if trade.action == "open_long":
                    open_long_trades.append((trade_time, equity_at_trade))
                elif trade.action == "open_short":
                    open_short_trades.append((trade_time, equity_at_trade))
                elif "stop_loss" in trade.action:
                    stop_loss_trades.append((trade_time, equity_at_trade))

            # 绘制交易标记（简化版：只显示开仓和止损）
            marker_size = 35
            marker_alpha = 0.85  # 提高到0.85，让标记更清晰
            edge_width = 0.5  # 恢复到0.5，边框更明显

            if open_long_trades:
                times, equities = zip(*open_long_trades)
                ax1.scatter(
                    times, equities, marker="^", s=marker_size, c="#00D2FF",
                    alpha=marker_alpha, label="开多", zorder=3, edgecolors="white", linewidths=edge_width
                )

            if open_short_trades:
                times, equities = zip(*open_short_trades)
                ax1.scatter(
                    times, equities, marker="v", s=marker_size, c="#FF6B6B",
                    alpha=marker_alpha, label="开空", zorder=3, edgecolors="white", linewidths=edge_width
                )

            if stop_loss_trades:
                times, equities = zip(*stop_loss_trades)
                ax1.scatter(
                    times, equities, marker="X", s=marker_size*1.3, c="#FD79A8",
                    alpha=marker_alpha, label="止损", zorder=3, edgecolors="white", linewidths=edge_width
                )

        # 设置标题和标签
        final_equity = equity_values[-1]
        final_benchmark = benchmark_values[-1]
        strategy_return = (final_equity - starting_balance) / starting_balance * 100
        benchmark_return = (final_benchmark - starting_balance) / starting_balance * 100

        ax1.set_title(
            f"策略收益 vs Buy & Hold | "
            f"Strategy: {strategy_return:+.2f}% | "
            f"B&H: {benchmark_return:+.2f}%",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_ylabel("权益 ($)", fontsize=12)

        # 图例设置：简化后使用2列显示
        if trades:
            ax1.legend(loc="upper left", fontsize=10, ncol=2, framealpha=0.9)
        else:
            ax1.legend(loc="best", fontsize=11)

        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.set_xlim(dates[0], dates[-1])

        # ========== 下图：回撤 ==========
        equity_array = np.array(equity_values)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100  # 转为百分比

        ax2.fill_between(dates, drawdown, 0, color="#E74C3C", alpha=0.3)
        ax2.plot(dates, drawdown, color="#E74C3C", linewidth=1.5, label="Drawdown")

        max_dd = np.min(drawdown)
        ax2.set_title(
            f"回撤曲线 | 最大回撤: {max_dd:.2f}%",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_ylabel("回撤 (%)", fontsize=12)
        ax2.set_xlabel("时间", fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.set_xlim(dates[0], dates[-1])
        ax2.legend(loc="lower right", fontsize=11)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"权益曲线图已保存到 {output_path}")

    @staticmethod
    def create_paired_trades_report(trades: list[Trade]) -> pd.DataFrame:
        """
        创建配对交易报告，将开仓和平仓配对在一起

        Returns:
            DataFrame with columns:
            - trade_num: 交易编号
            - direction: 方向 (long/short)
            - open_time: 开仓时间
            - close_time: 平仓时间
            - holding_duration: 持仓时长 (分钟)
            - open_price: 开仓价格
            - close_price: 平仓价格
            - qty: 数量
            - position_value: 仓位价值 (开仓金额)
            - open_fee: 开仓手续费
            - close_fee: 平仓手续费
            - pnl: 盈亏金额
            - pnl_pct: 盈亏比例 (相对于仓位价值)
            - close_reason: 平仓原因 (close/stop_loss/force_close)
        """
        paired_trades = []

        i = 0
        trade_num = 1

        while i < len(trades):
            trade = trades[i]

            # 如果是开仓交易
            if trade.action in ["open_long", "open_short"]:
                direction = "long" if trade.action == "open_long" else "short"
                open_trade = trade

                # 查找对应的平仓交易
                close_trade = None
                for j in range(i + 1, len(trades)):
                    if trades[j].action in [
                        "close_long",
                        "close_short",
                        "stop_loss_long",
                        "stop_loss_short",
                        "force_close_long",
                        "force_close_short",
                    ]:
                        close_trade = trades[j]
                        i = j  # 更新索引到平仓位置
                        break

                if close_trade:
                    # 计算持仓时长 (分钟)
                    holding_duration = (
                        close_trade.timestamp - open_trade.timestamp
                    ) / (1000 * 60)

                    # 计算仓位价值
                    position_value = open_trade.qty * open_trade.price

                    # 计算盈亏比例
                    pnl_pct = (
                        (close_trade.pnl / position_value) * 100
                        if position_value > 0
                        else 0
                    )

                    # 提取平仓原因
                    if "stop_loss" in close_trade.action:
                        close_reason = "stop_loss"
                    elif "force_close" in close_trade.action:
                        close_reason = "force_close"
                    else:
                        close_reason = "close"

                    paired_trades.append(
                        {
                            "trade_num": trade_num,
                            "direction": direction,
                            "open_time": open_trade.timestamp,
                            "close_time": close_trade.timestamp,
                            "holding_duration": holding_duration,
                            "open_price": open_trade.price,
                            "close_price": close_trade.price,
                            "qty": open_trade.qty,
                            "position_value": position_value,
                            "open_fee": open_trade.fee,
                            "close_fee": close_trade.fee,
                            "pnl": close_trade.pnl,
                            "pnl_pct": pnl_pct,
                            "close_reason": close_reason,
                        }
                    )

                    trade_num += 1

            i += 1

        if not paired_trades:
            return pd.DataFrame()

        df = pd.DataFrame(paired_trades)

        # 转换时间戳为 datetime
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        return df

    @staticmethod
    def save_results(
        trades: list[Trade],
        equity_curve: list[EquityPoint],
        metrics: dict,
        output_dir: Path,
    ):
        """保存结果到文件"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存交易记录
        if trades:
            trades_df = pd.DataFrame([t.to_dict() for t in trades])
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], unit="ms")
            trades_df.to_csv(output_dir / "trades.csv", index=False)
            print(f"交易记录已保存到 {output_dir / 'trades.csv'}")

            # 保存配对交易报告
            paired_df = BacktestAnalyzer.create_paired_trades_report(trades)
            if not paired_df.empty:
                paired_df.to_csv(output_dir / "paired_trades.csv", index=False)
                print(f"配对交易报告已保存到 {output_dir / 'paired_trades.csv'}")

                # 打印摘要统计
                print(f"\n配对交易摘要:")
                print(f"  总交易次数: {len(paired_df)}")
                print(f"  多头交易: {len(paired_df[paired_df['direction']=='long'])}")
                print(f"  空头交易: {len(paired_df[paired_df['direction']=='short'])}")
                print(
                    f"  平均持仓时长: {paired_df['holding_duration'].mean():.1f} 分钟"
                )
                print(f"  平均盈亏比例: {paired_df['pnl_pct'].mean():.2f}%")
                print(
                    f"  最大单笔盈利: ${paired_df['pnl'].max():.2f} ({paired_df['pnl_pct'].max():.2f}%)"
                )
                print(
                    f"  最大单笔亏损: ${paired_df['pnl'].min():.2f} ({paired_df['pnl_pct'].min():.2f}%)"
                )

                # 按平仓原因统计
                close_reason_counts = paired_df["close_reason"].value_counts()
                print(f"  平仓原因统计:")
                for reason, count in close_reason_counts.items():
                    print(f"    - {reason}: {count} ({count/len(paired_df)*100:.1f}%)")

        # 保存权益曲线
        if equity_curve:
            equity_df = pd.DataFrame(
                [
                    {
                        "timestamp": e.timestamp,
                        "equity": e.equity,
                        "benchmark_equity": e.benchmark_equity,
                    }
                    for e in equity_curve
                ]
            )
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], unit="ms")
            equity_df["drawdown"] = (
                equity_df["equity"] - equity_df["equity"].cummax()
            ) / equity_df["equity"].cummax()
            equity_df.to_csv(output_dir / "equity_curve.csv", index=False)
            print(f"权益曲线已保存到 {output_dir / 'equity_curve.csv'}")

        # 保存统计指标
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"统计指标已保存到 {output_dir / 'metrics.json'}")

        # 绘制并保存权益曲线图
        if equity_curve:
            BacktestAnalyzer.plot_equity_curve(
                equity_curve,
                metrics["starting_balance"],
                output_dir / "equity_curve.png",
                trades=trades,  # 传入交易记录用于标注
            )


# ==================== 向量化特征计算 ====================
def generate_all_fusion_bars_with_split(
    warmup_candles: np.ndarray,
    trading_candles: np.ndarray,
    max_bars: int = 3500,
) -> tuple[np.ndarray, int]:
    """
    一次性生成所有 fusion bars，并计算 warmup 分界点

    Args:
        warmup_candles: warmup K线数据
        trading_candles: trading K线数据
        max_bars: bar 容器最大容量

    Returns:
        fusion_bars: 所有 fusion bars
        warmup_fusion_bars_len: warmup 对应的 fusion bar 数量
    """
    print("\n" + "=" * 60)
    print("Phase 1: 生成 Fusion Bars")
    print("=" * 60)

    # 1. 先用 warmup_candles 单独生成 fusion bars（用于确定分界点）
    print("计算 warmup 分界点...")
    bar_warmup = DemoBar(max_bars=max_bars)
    bar_warmup.update_with_candles(warmup_candles)
    warmup_fusion_bars = bar_warmup.get_fusion_bars()

    if len(warmup_fusion_bars) == 0:
        raise ValueError("Warmup failed: no fusion bars generated from warmup_candles")

    warmup_last_timestamp = warmup_fusion_bars[-1, 0]
    print(f"Warmup 生成了 {len(warmup_fusion_bars)} 个 fusion bars")
    print(f"Warmup 最后时间戳: {warmup_last_timestamp}")

    # 2. 用所有 candles 生成完整的 fusion bars
    print("\n生成所有 fusion bars...")
    all_candles = np.vstack([warmup_candles, trading_candles])
    bar_all = DemoBar(max_bars=max_bars)
    bar_all.update_with_candles(all_candles)
    fusion_bars = bar_all.get_fusion_bars()

    print(f"总共生成 {len(fusion_bars)} 个 fusion bars")

    # 3. 通过时间戳找到 warmup 在所有 fusion bars 中的分界点
    warmup_fusion_bars_len = None
    for i, bar in enumerate(fusion_bars):
        if bar[0] == warmup_last_timestamp:
            warmup_fusion_bars_len = i + 1
            break

    # 如果没找到精确匹配（理论上不应该发生），使用最接近的
    if warmup_fusion_bars_len is None:
        print("⚠️  警告: 未找到精确的时间戳匹配，使用近似值")
        for i, bar in enumerate(fusion_bars):
            if bar[0] > warmup_last_timestamp:
                warmup_fusion_bars_len = i
                break

        # 如果还是没找到，说明所有 fusion bars 都在 warmup 范围内
        if warmup_fusion_bars_len is None:
            warmup_fusion_bars_len = len(fusion_bars)

    print(
        f"Warmup 分界点: {warmup_fusion_bars_len} (占 {warmup_fusion_bars_len/len(fusion_bars)*100:.1f}%)"
    )
    print(f"Trading fusion bars: {len(fusion_bars) - warmup_fusion_bars_len}")
    print("=" * 60 + "\n")

    return fusion_bars, warmup_fusion_bars_len


def compute_features_vectorized(
    fusion_bars: np.ndarray,
    warmup_fusion_bars_len: int,
    feature_info: dict,
    models: list[str],
) -> pd.DataFrame:
    """
    向量化计算所有特征

    Args:
        fusion_bars: 所有 fusion bars
        warmup_fusion_bars_len: warmup 对应的 fusion bar 数量
        feature_info: 特征信息字典
        models: 模型列表

    Returns:
        df_features_full: trading 阶段的完整特征 DataFrame
    """
    print("\n" + "=" * 60)
    print("Phase 2: 向量化特征计算")
    print("=" * 60)

    # 1. 准备所有原始特征列表
    all_raw_feat = []
    all_raw_feat.extend(feature_info["fracdiff"])
    for m in models:
        all_raw_feat.extend(feature_info[m])
    all_raw_feat = set(all_raw_feat)
    all_raw_feat = sorted(
        [
            i
            for i in all_raw_feat
            if not i.startswith("deep_ssm") and not i.startswith("lg_ssm")
        ]
    )

    print(f"需要计算 {len(all_raw_feat)} 个原始特征")
    print(f"Warmup fusion bars: {warmup_fusion_bars_len}")
    print(f"Trading fusion bars: {len(fusion_bars) - warmup_fusion_bars_len}")

    # 2. 批量计算所有原始特征（sequential=True）
    print("\n计算原始特征...")
    fc = SimpleFeatureCalculator()
    fc.load(fusion_bars, sequential=True)

    start_time = time.perf_counter()
    df_raw_features = pd.DataFrame.from_dict(fc.get(all_raw_feat))
    raw_feat_time = time.perf_counter() - start_time
    print(f"原始特征计算完成，耗时: {raw_feat_time:.2f}秒")

    # 3. 批量计算 fracdiff 特征（sequential=True）
    print("\n计算 fracdiff 特征...")
    start_time = time.perf_counter()
    df_fracdiff = pd.DataFrame.from_dict(fc.get(feature_info["fracdiff"]))
    fracdiff_time = time.perf_counter() - start_time
    print(f"fracdiff 特征计算完成，耗时: {fracdiff_time:.2f}秒")

    # 4. SSM Warmup（逐行 inference）
    print("\n" + "-" * 60)
    print("SSM Warmup 阶段（逐行 inference）")
    print("-" * 60)

    deep_ssm = SSMContainer("deep_ssm")
    lg_ssm = SSMContainer("lg_ssm")

    start_time = time.perf_counter()
    for i in tqdm(range(warmup_fusion_bars_len), desc="SSM Warmup", ncols=100):
        deep_ssm.inference(df_fracdiff.iloc[[i]])
        lg_ssm.inference(df_fracdiff.iloc[[i]])
    warmup_time = time.perf_counter() - start_time
    print(f"SSM Warmup 完成，耗时: {warmup_time:.2f}秒")

    # 5. SSM Transform（批量处理 trading 部分）
    print("\n" + "-" * 60)
    print("SSM Transform 阶段（批量处理）")
    print("-" * 60)

    df_fracdiff_trading = df_fracdiff.iloc[warmup_fusion_bars_len:]

    start_time = time.perf_counter()
    print("DeepSSM transform...")
    df_deep_ssm = deep_ssm.transform(df_fracdiff_trading)
    deep_ssm_time = time.perf_counter() - start_time
    print(f"DeepSSM transform 完成，耗时: {deep_ssm_time:.2f}秒")

    start_time = time.perf_counter()
    print("LGSSM transform...")
    df_lg_ssm = lg_ssm.transform(df_fracdiff_trading)
    lg_ssm_time = time.perf_counter() - start_time
    print(f"LGSSM transform 完成，耗时: {lg_ssm_time:.2f}秒")

    # 6. 合并所有特征
    print("\n合并所有特征...")
    df_features_full = pd.concat(
        [df_deep_ssm, df_lg_ssm, df_raw_features.iloc[warmup_fusion_bars_len:]],
        axis=1,
    )

    print(f"\n特征计算完成:")
    print(f"  - 特征维度: {df_features_full.shape}")
    print(f"  - 原始特征耗时: {raw_feat_time:.2f}s")
    print(f"  - fracdiff耗时: {fracdiff_time:.2f}s")
    print(f"  - SSM Warmup耗时: {warmup_time:.2f}s")
    print(f"  - DeepSSM transform耗时: {deep_ssm_time:.2f}s")
    print(f"  - LGSSM transform耗时: {lg_ssm_time:.2f}s")
    print(
        f"  - 总耗时: {raw_feat_time + fracdiff_time + warmup_time + deep_ssm_time + lg_ssm_time:.2f}s"
    )
    print("=" * 60 + "\n")

    return df_features_full


def predict_all_models(
    df_features: pd.DataFrame,
    models: list[str],
    feature_info: dict,
) -> dict[str, list[int]]:
    """
    所有模型的预测（逐行 + 进度条）

    Args:
        df_features: 完整特征 DataFrame
        models: 模型列表
        feature_info: 特征信息字典

    Returns:
        predictions: {model_name: [pred1, pred2, ...]}
    """
    print("\n" + "=" * 60)
    print("Phase 3: 模型预测（逐行 + filters）")
    print("=" * 60)

    predictions = {}

    for model_name in models:
        print(f"\n预测模型: {model_name}")

        # 初始化模型容器
        model_container = LGBMContainer(*model_name_to_params(model_name))
        model_container.is_livetrading = False  # 使用回测模型

        # 逐行预测并应用 filters
        model_preds = []
        pbar = tqdm(
            range(len(df_features)),
            desc=f"  {model_name}",
            ncols=100,
            leave=True,
        )

        for i in pbar:
            feat_row = df_features.iloc[[i]][feature_info[model_container.MODEL_NAME]]
            pred = model_container.final_predict(feat_row)  # 包含 filter 应用
            model_preds.append(pred)

        predictions[model_name] = model_preds
        print(f"  完成！预测结果: {len(model_preds)} 个")

    print("\n所有模型预测完成")
    print("=" * 60 + "\n")

    return predictions


def aggregate_votes(predictions: dict[str, list[int]], models: list[str]) -> list[str]:
    """
    汇总投票结果

    Args:
        predictions: {model_name: [pred1, pred2, ...]}
        models: 模型列表

    Returns:
        signals: ["long", "short", "flat", ...]
    """
    print("\n" + "=" * 60)
    print("Phase 4: 汇总投票结果")
    print("=" * 60)

    n_samples = len(predictions[models[0]])
    signals = []

    for i in range(n_samples):
        votes = [predictions[m][i] for m in models]

        if all(v == 1 for v in votes):
            signals.append("long")
        elif all(v == -1 for v in votes):
            signals.append("short")
        else:
            signals.append("flat")

    # 统计信号分布
    signal_counts = {
        "long": signals.count("long"),
        "short": signals.count("short"),
        "flat": signals.count("flat"),
    }

    print(f"信号统计:")
    print(
        f"  - Long: {signal_counts['long']} ({signal_counts['long']/n_samples*100:.1f}%)"
    )
    print(
        f"  - Short: {signal_counts['short']} ({signal_counts['short']/n_samples*100:.1f}%)"
    )
    print(
        f"  - Flat: {signal_counts['flat']} ({signal_counts['flat']/n_samples*100:.1f}%)"
    )
    print("=" * 60 + "\n")

    return signals


# ==================== 主回测流程 ====================
def run_vectorized_backtest(
    warmup_candles: np.ndarray,
    trading_candles: np.ndarray,
    models: list[str],
    feature_info: dict,
    starting_balance: float = 10000,
    fee_rate: float = 0.0005,
    leverage: int = 3,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    运行向量化回测

    Args:
        warmup_candles: Warmup K线数据
        trading_candles: Trading K线数据
        models: 模型列表
        feature_info: 特征信息
        starting_balance: 初始资金
        fee_rate: 手续费率
        leverage: 杠杆倍数
        output_dir: 输出目录

    Returns:
        回测结果字典
    """
    print("\n" + "=" * 60)
    print("向量化快速回测引擎启动")
    print("=" * 60)
    print(f"初始资金: ${starting_balance:,.2f}")
    print(f"手续费率: {fee_rate*100:.2f}%")
    print(f"杠杆倍数: {leverage}x")
    print(f"模型列表: {models}")
    print(f"Warmup K线数: {len(warmup_candles):,}")
    print(f"Trading K线数: {len(trading_candles):,}")
    print("=" * 60 + "\n")

    total_start = time.perf_counter()

    # ========== Phase 1: 生成 Fusion Bars ==========
    fusion_bars, warmup_fusion_bars_len = generate_all_fusion_bars_with_split(
        warmup_candles, trading_candles, max_bars=3500
    )

    # ========== Phase 2: 向量化特征计算 ==========
    df_features = compute_features_vectorized(
        fusion_bars, warmup_fusion_bars_len, feature_info, models
    )

    # ========== Phase 3: 模型预测 ==========
    predictions = predict_all_models(df_features, models, feature_info)

    # ========== Phase 4: 汇总投票 ==========
    signals = aggregate_votes(predictions, models)

    # ========== Phase 5: 交易模拟 ==========
    print("\n" + "=" * 60)
    print("Phase 5: 交易模拟")
    print("=" * 60)

    backtester = FastBacktester(starting_balance, fee_rate, leverage)
    trading_fusion_bars = fusion_bars[warmup_fusion_bars_len:]

    pbar = tqdm(range(len(trading_fusion_bars)), desc="Trading Simulation", ncols=120)

    for i in pbar:
        fusion_bar = trading_fusion_bars[i]
        timestamp = int(fusion_bar[0])
        current_price = float(fusion_bar[2])  # close price
        signal = signals[i]

        # 记录交易前的交易数量（用于判断是否发生了交易）
        trades_count_before = len(backtester.trades)

        # 初始化 Buy & Hold 基准
        if i == 0:
            backtester.init_benchmark(current_price)
            # 记录初始权益点
            backtester.record_equity(timestamp, current_price)

        # 检查止损
        if not backtester.position.is_flat:
            backtester.check_stop_loss(timestamp, current_price)

        # 执行交易逻辑
        if backtester.position.is_flat:
            if signal == "long":
                stop_loss_price = current_price * (
                    1 - STOP_LOSS_RATIO_NO_LEVERAGE / leverage
                )
                backtester.open_long(timestamp, current_price, stop_loss_price)
            elif signal == "short":
                stop_loss_price = current_price * (
                    1 + STOP_LOSS_RATIO_NO_LEVERAGE / leverage
                )
                backtester.open_short(timestamp, current_price, stop_loss_price)
        else:
            if backtester.position.is_long and signal == "short":
                backtester.close_position(timestamp, current_price, reason="close")
            elif backtester.position.is_short and signal == "long":
                backtester.close_position(timestamp, current_price, reason="close")

        # 如果发生了交易，记录权益点
        if len(backtester.trades) > trades_count_before:
            backtester.record_equity(timestamp, current_price)

        # 更新进度条显示（每10次更新）
        if i % 10 == 0 or i == len(trading_fusion_bars) - 1:
            current_equity = backtester.equity(current_price)
            return_pct = (current_equity - starting_balance) / starting_balance * 100
            position_status = backtester.position.side.upper()

            pbar.set_postfix(
                {
                    "Equity": f"${current_equity:,.0f}",
                    "Return": f"{return_pct:+.2f}%",
                    "Pos": position_status,
                    "Trades": len([t for t in backtester.trades if t.pnl != 0]),
                }
            )

        # 额外每100根记录一次权益（保持曲线平滑）
        if i % 100 == 0:
            backtester.record_equity(timestamp, current_price)

    # 最后记录一次权益
    final_timestamp = int(trading_fusion_bars[-1, 0])
    final_price = trading_fusion_bars[-1, 2]
    backtester.record_equity(final_timestamp, final_price)

    # 如果还有未平仓位，强制平仓
    if not backtester.position.is_flat:
        backtester.close_position(final_timestamp, final_price, reason="force_close")

    print("\n交易模拟完成")
    print("=" * 60 + "\n")

    # ========== Phase 6: 结果分析 ==========
    print("=" * 60)
    print("结果分析")
    print("=" * 60)

    metrics = BacktestAnalyzer.calculate_metrics(
        backtester.trades, backtester.equity_curve, starting_balance
    )

    # 打印关键指标
    print(f"最终权益: ${metrics['final_equity']:,.2f}")
    print(f"总收益率: {metrics['total_return_pct']}")
    print(f"Buy&Hold收益率: {metrics['benchmark_return_pct']}")
    print(f"最大回撤: {metrics['max_drawdown_pct']}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"Calmar比率: {metrics['calmar_ratio']:.2f}")
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"胜率: {metrics['win_rate_pct']}")
    print(f"盈亏比: {metrics['profit_factor']:.2f}")

    total_time = time.perf_counter() - total_start
    print(f"\n总耗时: {total_time:.2f}秒")
    print("=" * 60 + "\n")

    # 保存结果
    if output_dir:
        BacktestAnalyzer.save_results(
            backtester.trades, backtester.equity_curve, metrics, output_dir
        )

    return metrics


# ==================== 主入口 ====================
if __name__ == "__main__":
    from jesse import helpers, research

    # ========== 配置 ==========
    # 测试模式：启用后只处理前 N 个 fusion bars
    TEST_MODE = False
    TEST_FUSION_BARS = 1000

    MODELS = [
        "c_L5_N1",
        "c_L6_N1",
    ]

    STRATEGY = "BinanceBtcDemoBarV2"

    # 加载特征信息
    path_features = (
        Path(__file__).parent / "strategies" / STRATEGY / "models" / "feature_info.json"
    )
    with open(path_features) as f:
        feature_info: dict[str, list[str]] = json.load(f)

    # ========== 获取数据 ==========
    print("正在加载K线数据...")
    warmup_candles, trading_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp("2025-02-01"),
        helpers.date_to_timestamp("2025-10-25"),
        warmup_candles_num=150000,
        caching=False,
        is_for_jesse=False,
    )
    # 过滤0成交量
    warmup_candles = warmup_candles[warmup_candles[:, 5] >= 0]
    trading_candles = trading_candles[trading_candles[:, 5] >= 0]

    print(
        f"数据加载完成: warmup={len(warmup_candles):,}, trading={len(trading_candles):,}\n"
    )

    # ========== 运行回测 ==========
    model_names = "_".join(MODELS)
    output_dir = (
        Path(__file__).parent
        / "backtest_results"
        / f"{model_names}_vectorized_{int(time.time())}"
    )

    metrics = run_vectorized_backtest(
        warmup_candles=warmup_candles,
        trading_candles=trading_candles,
        models=MODELS,
        feature_info=feature_info,
        starting_balance=10000,
        fee_rate=0.0005,
        leverage=3,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60)
