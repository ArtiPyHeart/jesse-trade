"""
回测结果可视化模块

生成策略走势图、回撤图等可视化报告。
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import BacktestMetrics


def plot_backtest_results(
    metrics: BacktestMetrics,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """
    绘制回测结果图表

    包含两个子图：
    1. 上图：策略收益曲线 vs Buy & Hold 基准
    2. 下图：回撤曲线

    Args:
        metrics: 回测指标
        arrays: 包含 equity, timestamps, drawdowns, benchmark_equity 的数组字典
        output_dir: 输出目录

    Returns:
        图片文件路径
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # 转换时间戳为日期
    timestamps = arrays["timestamps"]
    dates = pd.to_datetime(timestamps, unit="ms")

    equity = arrays["equity"]
    benchmark = arrays["benchmark_equity"]
    drawdowns = arrays["drawdowns"]

    # === 上图：收益曲线 ===
    # 策略收益率曲线
    strategy_returns = (equity / equity[0] - 1) * 100
    ax1.plot(dates, strategy_returns, label="Strategy", color="#1f77b4", linewidth=1.5)

    # Buy & Hold 基准曲线
    if len(benchmark) > 0:
        benchmark_returns = (benchmark / benchmark[0] - 1) * 100
        ax1.plot(
            dates,
            benchmark_returns,
            label="Buy & Hold",
            color="#7f7f7f",
            linewidth=1.2,
            alpha=0.7,
        )

    # 标记最大回撤区间
    dd_start = metrics.drawdown.max_drawdown_start_idx
    dd_end = metrics.drawdown.max_drawdown_end_idx
    if dd_start < len(dates) and dd_end < len(dates):
        ax1.axvspan(
            dates[dd_start],
            dates[dd_end],
            alpha=0.2,
            color="red",
            label="Max DD Period",
        )
        ax1.plot(
            [dates[dd_start], dates[dd_end]],
            [strategy_returns[dd_start], strategy_returns[dd_end]],
            linestyle="--",
            color="red",
            linewidth=1.5,
        )

    # 添加零线
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 构建图例文本
    legend_text = (
        f"Total Return: {metrics.returns.total_return * 100:.2f}% | "
        f"CAGR: {metrics.returns.cagr * 100:.2f}% | "
        f"Sharpe: {metrics.risk.sharpe_ratio:.2f} | "
        f"Sortino: {metrics.risk.sortino_ratio:.2f} | "
        f"Calmar: {metrics.risk.calmar_ratio:.2f}\n"
        f"Max DD: {metrics.drawdown.max_drawdown * 100:.2f}% | "
        f"Win Rate: {metrics.trades.win_rate * 100:.1f}% | "
        f"Profit Factor: {metrics.trades.profit_factor:.2f} | "
        f"Trades: {metrics.trades.closing_trades}"
    )
    ax1.legend(loc="upper left", fontsize=10)

    # 添加统计信息文本框
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9)
    ax1.text(
        0.02,
        0.98,
        legend_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # === 下图：回撤曲线 ===
    drawdown_pct = -drawdowns * 100  # 转为负值百分比
    ax2.fill_between(dates, drawdown_pct, 0, alpha=0.3, color="red")
    ax2.plot(dates, drawdown_pct, color="red", linewidth=0.8)

    # 标记最大回撤点
    if dd_end < len(dates):
        max_dd_value = -metrics.drawdown.max_drawdown * 100
        ax2.scatter([dates[dd_end]], [max_dd_value], color="darkred", s=50, zorder=5)
        ax2.annotate(
            f"Max DD: {max_dd_value:.1f}%",
            xy=(dates[dd_end], max_dd_value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="darkred",
        )

    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 设置 x 轴日期格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # 标题
    title = (
        f"Backtest: {metrics.backtest_name}\n"
        f"Models: {', '.join(metrics.models)} | "
        f"Period: {metrics.start_date} to {metrics.end_date} | "
        f"Leverage: {metrics.leverage}x"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    # 保存图片
    output_path = output_dir / "backtest_chart.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path


def print_metrics_report(metrics: BacktestMetrics) -> str:
    """
    生成文本格式的指标报告

    Returns:
        格式化的报告字符串
    """
    sep = "=" * 70
    lines = []

    lines.append(sep)
    lines.append(f"{'BACKTEST REPORT':^70}")
    lines.append(sep)
    lines.append("")

    # 元数据
    lines.append(f"Name:           {metrics.backtest_name}")
    lines.append(f"Type:           {metrics.backtest_type}")
    lines.append(f"Models:         {', '.join(metrics.models)}")
    lines.append(f"Period:         {metrics.start_date} to {metrics.end_date}")
    lines.append(f"Created:        {metrics.created_at}")
    lines.append(f"Leverage:       {metrics.leverage}x")
    lines.append(f"Fee Rate:       {metrics.fee_rate * 100:.4f}%")
    lines.append(f"Stop Loss:      {metrics.stop_loss_ratio * 100:.2f}%")
    lines.append(f"Total Bars:     {metrics.total_bars}")
    lines.append(f"Trading Bars:   {metrics.trading_bars}")
    lines.append("")

    # 收益指标
    lines.append("-" * 70)
    lines.append("RETURN METRICS")
    lines.append("-" * 70)
    r = metrics.returns
    lines.append(f"Starting Balance:     ${r.starting_balance:,.2f}")
    lines.append(f"Final Balance:        ${r.final_balance:,.2f}")
    lines.append(f"Net P&L:              ${r.net_pnl:,.2f}")
    lines.append(f"Total Return:         {r.total_return * 100:+.2f}%")
    lines.append(f"CAGR:                 {r.cagr * 100:+.2f}%")
    lines.append("")

    # 交易指标
    lines.append("-" * 70)
    lines.append("TRADE METRICS")
    lines.append("-" * 70)
    t = metrics.trades
    lines.append(f"Total Trades:         {t.total_trades}")
    lines.append(f"Closing Trades:       {t.closing_trades}")
    lines.append(f"Winning Trades:       {t.winning_trades}")
    lines.append(f"Losing Trades:        {t.losing_trades}")
    lines.append(f"Win Rate:             {t.win_rate * 100:.2f}%")
    lines.append(f"Avg Win:              ${t.avg_win:,.2f}")
    lines.append(f"Avg Loss:             ${t.avg_loss:,.2f}")
    lines.append(f"Max Win:              ${t.max_win:,.2f}")
    lines.append(f"Max Loss:             ${t.max_loss:,.2f}")
    lines.append(f"Payoff Ratio:         {t.payoff_ratio:.2f}")
    lines.append(f"Profit Factor:        {t.profit_factor:.2f}")
    lines.append(f"Expected Value:       ${t.expected_value:,.2f}")
    lines.append(f"Total Fees:           ${t.total_fees:,.2f}")
    lines.append(f"Fee Ratio:            {t.fee_ratio * 100:.2f}%")
    lines.append(f"Max Consecutive Wins: {t.max_consecutive_wins}")
    lines.append(f"Max Consecutive Losses: {t.max_consecutive_losses}")
    lines.append(f"Avg Holding Bars:     {t.avg_holding_bars:.1f}")
    lines.append("")

    # 回撤指标
    lines.append("-" * 70)
    lines.append("DRAWDOWN METRICS")
    lines.append("-" * 70)
    d = metrics.drawdown
    lines.append(f"Max Drawdown:         {d.max_drawdown * 100:.2f}%")
    lines.append(f"Max DD Duration:      {d.max_drawdown_duration} bars")
    lines.append(f"Max DD Start:         Bar {d.max_drawdown_start_idx}")
    lines.append(f"Max DD End:           Bar {d.max_drawdown_end_idx}")
    recovery = (
        d.max_drawdown_recovery_idx
        if d.max_drawdown_recovery_idx >= 0
        else "Not Recovered"
    )
    lines.append(f"Max DD Recovery:      {recovery}")
    lines.append(f"Avg Drawdown:         {d.avg_drawdown * 100:.2f}%")
    lines.append(f"Ulcer Index:          {d.ulcer_index:.4f}")
    lines.append(f"Drawdown Area:        {d.drawdown_area:.4f}")
    lines.append(f"Recovery Factor:      {d.recovery_factor:.2f}")
    lines.append("")

    # 风险指标
    lines.append("-" * 70)
    lines.append("RISK METRICS")
    lines.append("-" * 70)
    k = metrics.risk
    lines.append(f"Sharpe Ratio:         {k.sharpe_ratio:.2f}")
    lines.append(f"Sortino Ratio:        {k.sortino_ratio:.2f}")
    lines.append(f"Calmar Ratio:         {k.calmar_ratio:.2f}")
    lines.append(f"Volatility (Annual):  {k.volatility * 100:.2f}%")
    lines.append(f"Downside Deviation:   {k.downside_deviation:.4f}")
    lines.append(f"VaR (95%):            {k.var_95 * 100:.2f}%")
    lines.append(f"CVaR (95%):           {k.cvar_95 * 100:.2f}%")
    lines.append("")

    # 基准对比
    lines.append("-" * 70)
    lines.append("BENCHMARK COMPARISON (vs Buy & Hold)")
    lines.append("-" * 70)
    b = metrics.benchmark
    lines.append(f"Benchmark Return:     {b.benchmark_return * 100:+.2f}%")
    lines.append(f"Benchmark Max DD:     {b.benchmark_max_drawdown * 100:.2f}%")
    lines.append(f"Excess Return:        {b.excess_return * 100:+.2f}%")
    lines.append(f"Alpha:                {b.alpha:.4f}")
    lines.append(f"Beta:                 {b.beta:.2f}")
    lines.append(f"Correlation:          {b.correlation:.2f}")
    lines.append("")

    lines.append(sep)

    return "\n".join(lines)


def save_metrics_json(metrics: BacktestMetrics, output_dir: Path) -> Path:
    """保存指标为 JSON 文件"""
    output_path = output_dir / "metrics.json"
    with open(output_path, "w") as f:
        f.write(metrics.model_dump_json(indent=2))
    return output_path


def save_metrics_report(metrics: BacktestMetrics, output_dir: Path) -> Path:
    """保存指标报告为文本文件"""
    report = print_metrics_report(metrics)
    output_path = output_dir / "report.txt"
    with open(output_path, "w") as f:
        f.write(report)
    return output_path
