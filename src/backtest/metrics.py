"""
回测金融指标计算模块

提供全面的金融指标计算，用于评估交易策略表现。
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class TradeMetrics(BaseModel):
    """交易层面指标"""

    total_trades: int = Field(description="总交易数")
    closing_trades: int = Field(description="平仓交易数")
    winning_trades: int = Field(description="盈利交易数")
    losing_trades: int = Field(description="亏损交易数")
    win_rate: float = Field(description="胜率")
    avg_win: float = Field(description="平均盈利")
    avg_loss: float = Field(description="平均亏损")
    max_win: float = Field(description="最大单笔盈利")
    max_loss: float = Field(description="最大单笔亏损")
    payoff_ratio: float = Field(description="盈亏比 (Avg Win / Avg Loss)")
    profit_factor: float = Field(description="盈亏因子 (Total Profit / Total Loss)")
    expected_value: float = Field(description="期望值 (每笔交易)")
    total_profit: float = Field(description="盈利交易总额")
    total_loss: float = Field(description="亏损交易总额")
    total_fees: float = Field(description="总手续费")
    fee_ratio: float = Field(description="手续费占比")
    max_consecutive_wins: int = Field(description="最大连续盈利次数")
    max_consecutive_losses: int = Field(description="最大连续亏损次数")
    avg_holding_bars: float = Field(description="平均持仓周期(bars)")


class DrawdownMetrics(BaseModel):
    """回撤指标"""

    max_drawdown: float = Field(description="最大回撤")
    max_drawdown_duration: int = Field(description="最大回撤持续周期(bars)")
    max_drawdown_start_idx: int = Field(description="最大回撤开始索引")
    max_drawdown_end_idx: int = Field(description="最大回撤结束索引")
    max_drawdown_recovery_idx: int = Field(
        description="最大回撤恢复索引 (-1 表示未恢复)"
    )
    avg_drawdown: float = Field(description="平均回撤 (Pain Index)")
    ulcer_index: float = Field(description="Ulcer Index")
    drawdown_area: float = Field(description="回撤面积")
    recovery_factor: float = Field(description="恢复因子 (Net PnL / Max DD)")


class RiskMetrics(BaseModel):
    """风险调整指标"""

    sharpe_ratio: float = Field(description="夏普比率 (年化)")
    sortino_ratio: float = Field(description="索提诺比率 (年化)")
    calmar_ratio: float = Field(description="卡玛比率 (CAGR / Max DD)")
    volatility: float = Field(description="年化波动率")
    downside_deviation: float = Field(description="下行标准差")
    var_95: float = Field(description="95% VaR")
    cvar_95: float = Field(description="95% CVaR (Expected Shortfall)")


class ReturnMetrics(BaseModel):
    """收益指标"""

    starting_balance: float = Field(description="初始资金")
    final_balance: float = Field(description="最终资金")
    net_pnl: float = Field(description="净盈亏")
    total_return: float = Field(description="总收益率")
    cagr: float = Field(description="年化收益率 (CAGR)")
    avg_return_per_bar: float = Field(description="平均每bar收益率")


class BenchmarkMetrics(BaseModel):
    """基准对比指标"""

    benchmark_return: float = Field(description="基准(Buy&Hold)收益率")
    benchmark_max_drawdown: float = Field(description="基准最大回撤")
    excess_return: float = Field(description="超额收益")
    alpha: float = Field(description="阿尔法")
    beta: float = Field(description="贝塔")
    correlation: float = Field(description="相关性")


class BacktestMetrics(BaseModel):
    """完整回测指标"""

    # 元数据
    backtest_name: str = Field(description="回测名称")
    backtest_type: str = Field(description="回测类型 (sequential/vectorized)")
    models: list[str] = Field(description="使用的模型")
    start_date: str = Field(description="开始日期")
    end_date: str = Field(description="结束日期")
    created_at: str = Field(description="创建时间")
    leverage: int = Field(description="杠杆倍数")
    fee_rate: float = Field(description="手续费率")
    stop_loss_ratio: float = Field(description="止损比例")
    total_bars: int = Field(description="总K线数")
    trading_bars: int = Field(description="交易K线数")

    # 各类指标
    returns: ReturnMetrics = Field(description="收益指标")
    trades: TradeMetrics = Field(description="交易指标")
    drawdown: DrawdownMetrics = Field(description="回撤指标")
    risk: RiskMetrics = Field(description="风险指标")
    benchmark: BenchmarkMetrics = Field(description="基准对比指标")


def calculate_returns(
    equity_curve: list[tuple[int, float]],
    starting_balance: float,
    total_bars: int,
) -> tuple[ReturnMetrics, np.ndarray]:
    """计算收益指标"""
    equity_values = np.array([e[1] for e in equity_curve])
    final_balance = equity_values[-1] if len(equity_values) > 0 else starting_balance

    net_pnl = final_balance - starting_balance
    total_return = net_pnl / starting_balance

    # 计算收益率序列
    returns = (
        np.diff(equity_values) / equity_values[:-1]
        if len(equity_values) > 1
        else np.array([])
    )

    # 年化收益率 (假设每个 bar 约 90 分钟，一年约 5840 个 bars)
    # 更保守的估计：使用实际交易时间
    bars_per_year = 365 * 24 * 60 / 90  # 约 5840 bars/year
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1

    if years > 0 and final_balance > 0 and starting_balance > 0:
        cagr = (final_balance / starting_balance) ** (1 / years) - 1
    else:
        cagr = 0.0

    avg_return_per_bar = np.mean(returns) if len(returns) > 0 else 0.0

    return ReturnMetrics(
        starting_balance=starting_balance,
        final_balance=final_balance,
        net_pnl=net_pnl,
        total_return=total_return,
        cagr=cagr,
        avg_return_per_bar=avg_return_per_bar,
    ), returns


def calculate_trade_metrics(trades: list[dict[str, Any]]) -> TradeMetrics:
    """计算交易层面指标"""
    closing_trades = [t for t in trades if t["pnl"] != 0]
    winning_trades = [t for t in closing_trades if t["pnl"] > 0]
    losing_trades = [t for t in closing_trades if t["pnl"] < 0]

    total_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0.0
    total_loss = sum(t["pnl"] for t in losing_trades) if losing_trades else 0.0
    total_fees = sum(t["fee"] for t in trades)

    n_closing = len(closing_trades)
    n_win = len(winning_trades)
    n_loss = len(losing_trades)

    win_rate = n_win / n_closing if n_closing > 0 else 0.0
    avg_win = total_profit / n_win if n_win > 0 else 0.0
    avg_loss = abs(total_loss) / n_loss if n_loss > 0 else 0.0
    max_win = max((t["pnl"] for t in winning_trades), default=0.0)
    max_loss = min((t["pnl"] for t in losing_trades), default=0.0)

    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    profit_factor = total_profit / abs(total_loss) if total_loss != 0 else 0.0
    expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss

    # 手续费占比
    gross_pnl = total_profit + abs(total_loss)
    fee_ratio = total_fees / gross_pnl if gross_pnl > 0 else 0.0

    # 连续盈亏统计
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for t in closing_trades:
        if t["pnl"] > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

    # 平均持仓周期
    holding_bars = []
    for i in range(0, len(trades) - 1, 2):  # 假设开仓和平仓交替
        if i + 1 < len(trades):
            holding_bars.append(trades[i + 1]["bar_idx"] - trades[i]["bar_idx"])
    avg_holding_bars = np.mean(holding_bars) if holding_bars else 0.0

    return TradeMetrics(
        total_trades=len(trades),
        closing_trades=n_closing,
        winning_trades=n_win,
        losing_trades=n_loss,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_win=max_win,
        max_loss=max_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        expected_value=expected_value,
        total_profit=total_profit,
        total_loss=total_loss,
        total_fees=total_fees,
        fee_ratio=fee_ratio,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        avg_holding_bars=avg_holding_bars,
    )


def calculate_drawdown_metrics(
    equity_curve: list[tuple[int, float]],
    net_pnl: float,
) -> tuple[DrawdownMetrics, np.ndarray]:
    """计算回撤指标"""
    equity_values = np.array([e[1] for e in equity_curve])

    if len(equity_values) == 0:
        return DrawdownMetrics(
            max_drawdown=0.0,
            max_drawdown_duration=0,
            max_drawdown_start_idx=0,
            max_drawdown_end_idx=0,
            max_drawdown_recovery_idx=-1,
            avg_drawdown=0.0,
            ulcer_index=0.0,
            drawdown_area=0.0,
            recovery_factor=0.0,
        ), np.array([])

    # 计算峰值和回撤
    peaks = np.maximum.accumulate(equity_values)
    drawdowns = (peaks - equity_values) / peaks  # 正值表示回撤

    max_drawdown = np.max(drawdowns)

    # 找最大回撤的起始和结束位置
    max_dd_end_idx = int(np.argmax(drawdowns))
    max_dd_start_idx = (
        int(np.argmax(equity_values[: max_dd_end_idx + 1])) if max_dd_end_idx > 0 else 0
    )

    # 找恢复位置
    recovery_idx = -1
    peak_value = equity_values[max_dd_start_idx]
    for i in range(max_dd_end_idx + 1, len(equity_values)):
        if equity_values[i] >= peak_value:
            recovery_idx = i
            break

    # 最大回撤持续时间
    max_dd_duration = max_dd_end_idx - max_dd_start_idx

    # 平均回撤 (Pain Index)
    avg_drawdown = np.mean(drawdowns)

    # Ulcer Index
    ulcer_index = np.sqrt(np.mean(drawdowns**2))

    # 回撤面积
    drawdown_area = np.sum(drawdowns)

    # 恢复因子
    recovery_factor = (
        net_pnl / (max_drawdown * equity_values[0]) if max_drawdown > 0 else 0.0
    )

    return DrawdownMetrics(
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        max_drawdown_start_idx=max_dd_start_idx,
        max_drawdown_end_idx=max_dd_end_idx,
        max_drawdown_recovery_idx=recovery_idx,
        avg_drawdown=avg_drawdown,
        ulcer_index=ulcer_index,
        drawdown_area=drawdown_area,
        recovery_factor=recovery_factor,
    ), drawdowns


def calculate_risk_metrics(
    returns: np.ndarray,
    cagr: float,
    max_drawdown: float,
) -> RiskMetrics:
    """计算风险调整指标"""
    if len(returns) == 0:
        return RiskMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            var_95=0.0,
            cvar_95=0.0,
        )

    # 年化因子 (假设 90 分钟 bar)
    bars_per_year = 365 * 24 * 60 / 90
    sqrt_annual = np.sqrt(bars_per_year)

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # 年化波动率
    volatility = std_return * sqrt_annual

    # 夏普比率 (假设无风险利率为 0)
    sharpe_ratio = (mean_return / std_return * sqrt_annual) if std_return > 0 else 0.0

    # 下行标准差
    negative_returns = returns[returns < 0]
    downside_deviation = (
        np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0.0
    )

    # 索提诺比率
    sortino_ratio = (
        (mean_return / downside_deviation * sqrt_annual)
        if downside_deviation > 0
        else 0.0
    )

    # 卡玛比率
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0.0

    # VaR 和 CVaR (95%)
    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
    cvar_95 = (
        np.mean(returns[returns <= var_95])
        if len(returns[returns <= var_95]) > 0
        else 0.0
    )

    return RiskMetrics(
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        volatility=volatility,
        downside_deviation=downside_deviation,
        var_95=var_95,
        cvar_95=cvar_95,
    )


def calculate_benchmark_metrics(
    equity_curve: list[tuple[int, float]],
    fusion_bars: np.ndarray,
    starting_balance: float,
    trade_start_idx: int,
) -> tuple[BenchmarkMetrics, np.ndarray]:
    """计算基准对比指标 (Buy & Hold)"""
    if len(equity_curve) == 0 or len(fusion_bars) == 0:
        return BenchmarkMetrics(
            benchmark_return=0.0,
            benchmark_max_drawdown=0.0,
            excess_return=0.0,
            alpha=0.0,
            beta=0.0,
            correlation=0.0,
        ), np.array([])

    # 策略收益
    equity_values = np.array([e[1] for e in equity_curve])
    strategy_returns = (
        np.diff(equity_values) / equity_values[:-1]
        if len(equity_values) > 1
        else np.array([])
    )
    strategy_total_return = (
        (equity_values[-1] / starting_balance - 1) if len(equity_values) > 0 else 0.0
    )

    # 基准收益 (Buy & Hold from trade_start_idx)
    n_equity = len(equity_curve)
    bar_start = trade_start_idx
    bar_end = bar_start + n_equity

    if bar_end > len(fusion_bars):
        bar_end = len(fusion_bars)
        n_equity = bar_end - bar_start

    benchmark_prices = fusion_bars[bar_start:bar_end, 2]  # close prices
    benchmark_equity = starting_balance * benchmark_prices / benchmark_prices[0]
    benchmark_returns = (
        np.diff(benchmark_equity) / benchmark_equity[:-1]
        if len(benchmark_equity) > 1
        else np.array([])
    )
    benchmark_total_return = (
        (benchmark_equity[-1] / starting_balance - 1)
        if len(benchmark_equity) > 0
        else 0.0
    )

    # 基准最大回撤
    benchmark_peaks = np.maximum.accumulate(benchmark_equity)
    benchmark_drawdowns = (benchmark_peaks - benchmark_equity) / benchmark_peaks
    benchmark_max_dd = (
        np.max(benchmark_drawdowns) if len(benchmark_drawdowns) > 0 else 0.0
    )

    # 超额收益
    excess_return = strategy_total_return - benchmark_total_return

    # Alpha 和 Beta
    if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        cov_matrix = np.cov(strategy_returns, benchmark_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
        correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
    else:
        alpha = 0.0
        beta = 0.0
        correlation = 0.0

    return BenchmarkMetrics(
        benchmark_return=benchmark_total_return,
        benchmark_max_drawdown=benchmark_max_dd,
        excess_return=excess_return,
        alpha=alpha,
        beta=beta,
        correlation=correlation,
    ), benchmark_equity


def calculate_all_metrics(
    equity_curve: list[tuple[int, float]],
    trades: list[dict[str, Any]],
    fusion_bars: np.ndarray,
    starting_balance: float,
    leverage: int,
    fee_rate: float,
    stop_loss_ratio: float,
    models: list[str],
    start_date: str,
    end_date: str,
    backtest_type: str,
    trade_start_idx: int,
) -> tuple[BacktestMetrics, dict[str, np.ndarray]]:
    """计算所有回测指标"""

    # 收益指标
    returns_metrics, returns_array = calculate_returns(
        equity_curve, starting_balance, len(fusion_bars)
    )

    # 交易指标
    trade_metrics = calculate_trade_metrics(trades)

    # 回撤指标
    drawdown_metrics, drawdowns_array = calculate_drawdown_metrics(
        equity_curve, returns_metrics.net_pnl
    )

    # 风险指标
    risk_metrics = calculate_risk_metrics(
        returns_array, returns_metrics.cagr, drawdown_metrics.max_drawdown
    )

    # 基准对比
    benchmark_metrics, benchmark_equity = calculate_benchmark_metrics(
        equity_curve, fusion_bars, starting_balance, trade_start_idx
    )

    # 生成回测名称
    model_names = "_".join(models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_name = f"{model_names}_{backtest_type}_{timestamp}"

    metrics = BacktestMetrics(
        backtest_name=backtest_name,
        backtest_type=backtest_type,
        models=models,
        start_date=start_date,
        end_date=end_date,
        created_at=datetime.now().isoformat(),
        leverage=leverage,
        fee_rate=fee_rate,
        stop_loss_ratio=stop_loss_ratio,
        total_bars=len(fusion_bars),
        trading_bars=len(equity_curve),
        returns=returns_metrics,
        trades=trade_metrics,
        drawdown=drawdown_metrics,
        risk=risk_metrics,
        benchmark=benchmark_metrics,
    )

    # 返回用于绘图的数组
    arrays = {
        "equity": np.array([e[1] for e in equity_curve]),
        "timestamps": np.array([e[0] for e in equity_curve]),
        "drawdowns": drawdowns_array,
        "benchmark_equity": benchmark_equity,
    }

    return metrics, arrays
