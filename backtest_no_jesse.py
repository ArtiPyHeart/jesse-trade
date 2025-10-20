"""
快速回测引擎 - 不依赖 Jesse 库

核心设计：
1. Warmup 阶段：批量处理 warmup_candles，初始化状态
2. Trading 阶段：逐根处理 trading_candles，只在生成新 fusion bar 时交易
3. 轻量级订单执行：简化的仓位管理和订单系统
4. 详细记录：每笔交易和权益变化
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
from strategies.BinanceBtcDeapV1Voting.models.config import (
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


# ==================== 策略核心 ====================
class StrategyCore:
    """策略核心 - 封装 DemoBar、特征计算、模型推理"""

    def __init__(self, models: list[str], feature_info: dict):
        # Bar 容器
        self.bar_container = DemoBar(max_bars=3500)

        # 特征计算器
        self.fc = SimpleFeatureCalculator()

        # 模型配置
        self.models = models
        self.feature_info = feature_info

        # SSM 模型
        self.deep_ssm_model = SSMContainer("deep_ssm")
        self.lg_ssm_model = SSMContainer("lg_ssm")

        # LGBM 模型
        self._init_lgbm_models()

        # 状态
        self._all_raw_feat = self._prepare_all_raw_feat()
        self._warmup_done = False

    def _init_lgbm_models(self):
        """初始化 LGBM 模型"""
        for m in self.models:
            model_container = LGBMContainer(*model_name_to_params(m))
            model_container.is_livetrading = False  # 使用回测模型
            setattr(self, f"model_{m}", model_container)

    def _prepare_all_raw_feat(self) -> list[str]:
        """准备所有原始特征列表"""
        all_raw_feat = []
        all_raw_feat.extend(self.feature_info["fracdiff"])
        for m in self.models:
            all_raw_feat.extend(self.feature_info[m])
        all_raw_feat = set(all_raw_feat)
        all_raw_feat = sorted(
            [
                i
                for i in all_raw_feat
                if not i.startswith("deep_ssm") and not i.startswith("lg_ssm")
            ]
        )
        return all_raw_feat

    def update_with_candles(self, candles: np.ndarray):
        """更新 bar 容器"""
        self.bar_container.update_with_candles(candles)

    @property
    def is_new_bar(self) -> bool:
        """是否生成了新的 fusion bar"""
        return self.bar_container.is_latest_bar_complete

    @property
    def fusion_bars(self) -> np.ndarray:
        """获取所有 fusion bars"""
        return self.bar_container.get_fusion_bars()

    def warmup(self, candles: np.ndarray):
        """Warmup 阶段：用历史数据逐行初始化 SSM 状态"""
        print("\n" + "=" * 60)
        print("Warmup 阶段开始")
        print("=" * 60)

        # 更新 bar 容器
        self.update_with_candles(candles)

        # 获取所有 fusion bars
        fusion_bars = self.fusion_bars
        print(f"生成了 {len(fusion_bars)} 个 fusion bars")

        if len(fusion_bars) == 0:
            raise ValueError("Warmup failed: no fusion bars generated")

        # 批量计算所有 fracdiff 特征（效率优化）
        print(f"初始化 SSM 状态（逐行处理 {len(fusion_bars)} 个 fusion bars）...")
        self.fc.load(fusion_bars, sequential=True)
        df_feat_fracdiff_all = pd.DataFrame.from_dict(
            self.fc.get(self.feature_info["fracdiff"])
        )

        # 逐行喂给 SSM inference，更新内部状态
        for i in range(len(df_feat_fracdiff_all)):
            df_one_row = df_feat_fracdiff_all.iloc[[i]]
            # 不保存输出，只更新 SSM 内部状态
            self.deep_ssm_model.inference(df_one_row)
            self.lg_ssm_model.inference(df_one_row)

        # 标记 warmup 完成
        self._warmup_done = True

        print("SSM 状态已初始化")
        print("Warmup 阶段完成")
        print("=" * 60 + "\n")

    def compute_features_incremental(self) -> pd.DataFrame:
        """增量计算特征（仅最新值）"""
        # 获取最新的 fusion bar
        fusion_bars = self.fusion_bars

        # 计算特征（sequential=False 只返回最新值）
        self.fc.load(fusion_bars, sequential=False)
        df_feats = pd.DataFrame.from_dict(self.fc.get(self._all_raw_feat))
        df_feat_fracdiff = df_feats[self.feature_info["fracdiff"]]

        # SSM 增量推理
        df_feat_deep_ssm = self.deep_ssm_model.inference(df_feat_fracdiff)
        df_feat_lg_ssm = self.lg_ssm_model.inference(df_feat_fracdiff)

        # 合并特征
        df_final = pd.concat([df_feat_deep_ssm, df_feat_lg_ssm, df_feats], axis=1)

        return df_final

    def get_votes(self, features: pd.DataFrame) -> list[int]:
        """获取所有模型的投票"""
        votes = []
        for m in self.models:
            mc: LGBMContainer = getattr(self, f"model_{m}")
            pred = mc.final_predict(features[self.feature_info[mc.MODEL_NAME]])
            votes.append(pred)
        return votes

    def get_trading_signal(self) -> Literal["long", "short", "flat"]:
        """
        获取交易信号（只在新 fusion bar 生成时调用）

        Returns:
            "long": 做多信号
            "short": 做空信号
            "flat": 不交易
        """
        if not self.is_new_bar:
            return "flat"

        # 计算特征（只计算一次）
        features = self.compute_features_incremental()

        # 获取投票
        votes = self.get_votes(features)

        # 判断信号
        if all([v == 1 for v in votes]):
            return "long"
        elif all([v == -1 for v in votes]):
            return "short"
        else:
            return "flat"


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
    ):
        """绘制权益曲线对比图"""
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
        ax1.plot(dates, equity_values, label="Strategy", linewidth=2, color="#2E86DE")
        ax1.plot(
            dates,
            benchmark_values,
            label="Buy & Hold",
            linewidth=2,
            linestyle="--",
            color="#EE5A6F",
        )
        ax1.axhline(
            y=starting_balance, color="gray", linestyle=":", alpha=0.5, label="Initial"
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
            )


# ==================== 主回测流程 ====================
def run_backtest(
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
    运行快速回测

    Args:
        warmup_candles: Warmup K线数据
        trading_candles: 交易K线数据
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
    print("快速回测引擎启动")
    print("=" * 60)
    print(f"初始资金: ${starting_balance:,.2f}")
    print(f"手续费率: {fee_rate*100:.2f}%")
    print(f"杠杆倍数: {leverage}x")
    print(f"模型列表: {models}")
    print(f"Warmup K线数: {len(warmup_candles):,}")
    print(f"Trading K线数: {len(trading_candles):,}")
    print("=" * 60 + "\n")

    # 初始化组件
    backtester = FastBacktester(starting_balance, fee_rate, leverage)
    strategy = StrategyCore(models, feature_info)

    # ========== Phase 1: Warmup ==========
    start_time = time.time()
    strategy.warmup(warmup_candles)
    warmup_time = time.time() - start_time
    print(f"Warmup 耗时: {warmup_time:.2f}秒\n")

    # ========== Phase 2: Trading ==========
    print("=" * 60)
    print("Trading 阶段开始")
    print("=" * 60)

    start_time = time.time()

    for i in tqdm(
        range(len(trading_candles)), desc="Trading", unit="candle", ncols=100
    ):
        candle = trading_candles[i, :]

        timestamp = int(candle[0])
        current_price = float(candle[2])  # close price

        # 初始化 Buy & Hold 基准
        if i == 0:
            backtester.init_benchmark(current_price)

        # 每根 candle 都检查止损（不管是否生成新 bar）
        if not backtester.position.is_flat:
            backtester.check_stop_loss(timestamp, current_price)

        # 更新 bar 容器（传入完整历史数据：warmup + trading[:i+1]）
        # FusionBarContainerBase 内部会通过时间戳对齐避免重复处理
        all_candles = np.vstack([warmup_candles, trading_candles[: i + 1]])
        strategy.update_with_candles(all_candles)

        # 检查是否生成新的 fusion bar
        if strategy.is_new_bar:
            # 获取交易信号（只计算一次特征）
            signal = strategy.get_trading_signal()

            # 如果当前无仓位，检查开仓信号
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

            # 如果有仓位，检查平仓或反向开仓信号
            else:
                if backtester.position.is_long and signal == "short":
                    # 平多仓（也可以考虑直接反手开空，但为简单起见，只平仓）
                    backtester.close_position(timestamp, current_price, reason="close")
                elif backtester.position.is_short and signal == "long":
                    # 平空仓
                    backtester.close_position(timestamp, current_price, reason="close")

        # 记录权益（每100根记录一次）
        if i % 100 == 0:
            backtester.record_equity(timestamp, current_price)

    # 最后记录一次权益
    final_timestamp = int(trading_candles[-1, 0])
    final_price = trading_candles[-1, 2]
    backtester.record_equity(final_timestamp, final_price)

    # 如果还有未平仓位，强制平仓
    if not backtester.position.is_flat:
        backtester.close_position(final_timestamp, final_price, reason="force_close")

    trading_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Trading 阶段完成")
    print(f"Trading 耗时: {trading_time:.2f}秒")
    print(f"平均速度: {len(trading_candles)/trading_time:.0f} 根/秒")
    print("=" * 60 + "\n")

    # ========== Phase 3: 结果分析 ==========
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
    MODELS = [
        "c_L4_N1",
        "c_L5_N1",
    ]

    # 加载特征信息
    path_features = (
        Path(__file__).parent
        / "strategies"
        / "BinanceBtcDeapV1Voting"
        / "models"
        / "feature_info.json"
    )
    with open(path_features) as f:
        feature_info: dict[str, list[str]] = json.load(f)

    # ========== 获取数据 ==========
    print("正在加载K线数据...")
    warmup_candles, trading_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp("2025-01-01"),
        helpers.date_to_timestamp("2025-10-15"),
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
        Path(__file__).parent / "backtest_results" / f"{model_names}_{int(time.time())}"
    )

    metrics = run_backtest(
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
