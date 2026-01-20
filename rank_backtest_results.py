#!/usr/bin/env python3
"""
回测结果排名统计脚本

按 Calmar Ratio 从高到低排序，输出所有回测结果的关键指标到 CSV。

使用方式:
    python rank_backtest_results.py                    # 输出所有结果
    python rank_backtest_results.py --models 2        # 只输出2模型组合
    python rank_backtest_results.py --models 3        # 只输出3模型组合
    python rank_backtest_results.py --top 20          # 只输出前20名
    python rank_backtest_results.py --positive        # 只输出 Calmar > 0 的结果
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="回测结果排名统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=int,
        choices=[2, 3],
        help="筛选模型数量（2 或 3）",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="只输出前 N 名",
    )
    parser.add_argument(
        "--positive",
        action="store_true",
        help="只输出 Calmar Ratio > 0 的结果",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_ranking.csv",
        help="输出文件名（默认: backtest_ranking.csv）",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="backtest_results",
        help="回测结果目录（默认: backtest_results）",
    )
    return parser.parse_args()


def load_metrics(results_dir: Path) -> list[dict]:
    """加载所有回测结果的 metrics.json"""
    records = []

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue
        if "_vectorized_" not in subdir.name:
            continue

        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                data = json.load(f)

            # 提取模型信息
            models = data.get("models", [])
            model_count = len(models)

            # 提取关键指标
            returns = data.get("returns", {})
            trades = data.get("trades", {})
            drawdown = data.get("drawdown", {})
            risk = data.get("risk", {})

            record = {
                # 基本信息
                "name": data.get("backtest_name", subdir.name),
                "models": " + ".join(models),
                "model_count": model_count,
                "start_date": data.get("start_date", ""),
                "end_date": data.get("end_date", ""),
                # 风险指标（排序依据）
                "calmar_ratio": risk.get("calmar_ratio", 0),
                "sharpe_ratio": risk.get("sharpe_ratio", 0),
                "sortino_ratio": risk.get("sortino_ratio", 0),
                "volatility": risk.get("volatility", 0),
                # 收益指标
                "total_return": returns.get("total_return", 0),
                "cagr": returns.get("cagr", 0),
                "net_pnl": returns.get("net_pnl", 0),
                "final_balance": returns.get("final_balance", 0),
                # 回撤指标
                "max_drawdown": drawdown.get("max_drawdown", 0),
                "max_dd_duration": drawdown.get("max_drawdown_duration", 0),
                "avg_drawdown": drawdown.get("avg_drawdown", 0),
                "ulcer_index": drawdown.get("ulcer_index", 0),
                "recovery_factor": drawdown.get("recovery_factor", 0),
                # 交易指标
                "total_trades": trades.get("total_trades", 0),
                "win_rate": trades.get("win_rate", 0),
                "profit_factor": trades.get("profit_factor", 0),
                "payoff_ratio": trades.get("payoff_ratio", 0),
                "expected_value": trades.get("expected_value", 0),
                "avg_holding_bars": trades.get("avg_holding_bars", 0),
                "fee_ratio": trades.get("fee_ratio", 0),
            }
            records.append(record)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {metrics_file}: {e}")
            continue

    return records


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading metrics from: {results_dir}")
    records = load_metrics(results_dir)
    print(f"Loaded {len(records)} backtest results")

    if not records:
        print("No results found.")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(records)

    # 筛选模型数量
    if args.models:
        df = df[df["model_count"] == args.models]
        print(f"Filtered to {args.models}-model combos: {len(df)} results")

    # 筛选 Calmar > 0
    if args.positive:
        df = df[df["calmar_ratio"] > 0]
        print(f"Filtered to positive Calmar: {len(df)} results")

    # 按 Calmar Ratio 排序（降序）
    df = df.sort_values("calmar_ratio", ascending=False)

    # 添加排名列
    df.insert(0, "rank", range(1, len(df) + 1))

    # 只输出前 N 名
    if args.top:
        df = df.head(args.top)
        print(f"Top {args.top} results")

    # 保存到 CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\nSaved to: {output_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("Top 10 by Calmar Ratio:")
    print("=" * 80)

    summary_cols = [
        "rank",
        "models",
        "calmar_ratio",
        "sharpe_ratio",
        "total_return",
        "max_drawdown",
        "win_rate",
    ]
    print(df[summary_cols].head(10).to_string(index=False))

    # 统计信息
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    print(f"Total results:        {len(df)}")
    print(f"  2-model combos:     {len(df[df['model_count'] == 2])}")
    print(f"  3-model combos:     {len(df[df['model_count'] == 3])}")
    print(f"Positive Calmar:      {len(df[df['calmar_ratio'] > 0])}")
    print(f"Positive Sharpe:      {len(df[df['sharpe_ratio'] > 0])}")
    print(f"Profitable (PnL > 0): {len(df[df['net_pnl'] > 0])}")


if __name__ == "__main__":
    main()
