"""
向量化回测流程

批量计算预测信号，性能优化版本。
与 flow_backtest_sequential.py 结果一致，但计算效率更高。

交易规则（与逐步回测相同）：
1. 批量计算所有 bar 的预测（应用 pred_next - 1 延迟）
2. 全票一致时产生交易信号
3. 当前 bar 的 close 价格执行交易

使用方式：在项目根目录运行
    python flow_backtest_vectorized.py                    # 使用默认模型
    python flow_backtest_vectorized.py c_L9_N1 c_L9_N2   # 使用指定模型
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

# === 数据加载模块 ===
from jesse import helpers, research

from src.backtest import (
    calculate_all_metrics,
    plot_backtest_results,
    print_metrics_report,
    save_metrics_json,
    save_metrics_report,
)
from src.bars.fusion.demo import DemoBar
from src.features.dimensionality_reduction import ARDVAE
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.utils.env_dates import get_env_date, get_env_value, load_env_values
from src.utils.feature_warmup import determine_warmup_start_idx


def _load_strategy_config(strategy: str):
    del strategy
    from src.models.lgbm_container import LGBMContainer, model_name_to_params

    return LGBMContainer, model_name_to_params


# === 配置参数（与逐步回测完全相同）===
STARTING_BALANCE = 10000.0
FEE_RATE = 0.0005
LEVERAGE = 3
STOP_LOSS_RATIO_NO_LEVERAGE = 0.05
POSITION_SIZE_RATIO = 0.95
MIN_FUSION_BARS = 512

# 默认模型（可通过命令行覆盖）
DEFAULT_MODELS = ["c_L9_N1", "c_L9_N2"]

ENV_VALUES = load_env_values(Path(".env"))
STRATEGY = get_env_value("STRATEGY_NAME", ENV_VALUES)
TEST_START = get_env_date("TEST_START_DATE", ENV_VALUES)
TEST_END = get_env_date("TEST_END_DATE", ENV_VALUES)

MODEL_DIR = Path(f"strategies/{STRATEGY}/models")
LGBMContainer, model_name_to_params = _load_strategy_config(STRATEGY)


# === 数据结构（与逐步回测完全相同）===
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


# === 交易引擎（与逐步回测完全相同）===
class BacktestEngine:
    def __init__(
        self,
        starting_balance: float = STARTING_BALANCE,
        fee_rate: float = FEE_RATE,
        leverage: int = LEVERAGE,
        stop_loss_ratio: float = STOP_LOSS_RATIO_NO_LEVERAGE,
        position_size_ratio: float = POSITION_SIZE_RATIO,
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


# === 辅助函数 ===
def load_model_features(model_dir: Path, model_name: str) -> list[str]:
    features_path = model_dir / model_name / "features.json"
    with open(features_path, "r") as f:
        return json.load(f)


def collect_model_features(
    model_dir: Path, models: list[str]
) -> tuple[dict[str, list[str]], list[str]]:
    model_features = {}
    global_features_set: set[str] = set()

    for model_name in models:
        features = load_model_features(model_dir, model_name)
        model_features[model_name] = features
        global_features_set.update(features)

    return model_features, sorted(global_features_set)


def align_lgbm_feature_columns(
    df_features: pd.DataFrame, expected_columns: list[str]
) -> pd.DataFrame:
    if list(df_features.columns) == expected_columns:
        return df_features
    if set(df_features.columns) == set(expected_columns):
        return df_features[expected_columns]
    raise ValueError("Feature mismatch")


# === 向量化信号生成 ===
def generate_all_signals_with_delay(
    features_df: pd.DataFrame,
    models: list[str],
    model_containers: dict[str, LGBMContainer],
    model_features: dict[str, list[str]],
    reducers: dict[str, ARDVAE],
    trade_start_idx: int,
) -> np.ndarray:
    """
    批量生成所有 bar 的信号（已应用 pred_next 延迟）

    关键逻辑：
    - bar[i] 的预测在 bar[i + (pred_next - 1)] 时才生效
    - 前 pred_next - 1 个 bar 无有效预测，设为 0
    - 只处理 trade_start_idx 之后的有效数据

    Returns:
        signals: shape (n_bars,), 值为 -1/0/1
    """
    print("Generating vectorized signals...")
    n_bars = len(features_df)

    print(f"  Trade start idx: {trade_start_idx}")

    # 逐步回测从 trade_start_idx 开始，内部 pred_next 队列也从这里起步
    start_idx = trade_start_idx
    if start_idx >= n_bars:
        print("  No bars available after trade_start_idx; returning all-zero signals")
        return np.zeros(n_bars, dtype=np.int32)
    valid_features = features_df.iloc[start_idx:].copy()
    print(f"  Processing rows {start_idx} to {n_bars - 1} ({len(valid_features)} rows)")

    # 每个模型生成延迟后的预测（单次延迟 pred_next - 1）
    delayed_preds = {}
    for m in models:
        mc = model_containers[m]
        pred_next = mc.pred_next
        delay = pred_next - 1
        print(f"  Processing model {m} (pred_next={pred_next})...")

        # 获取该模型需要的特征（只使用有效行）
        model_feat = valid_features[model_features[m]]

        # ARDVAE 降维
        reduced = reducers[m].transform(model_feat)

        # 对齐列名
        expected_cols = mc.model.feature_name()
        reduced = align_lgbm_feature_columns(reduced, expected_cols)

        # 批量预测（原始概率）
        probs = mc.model.predict(reduced)

        # 将原始概率映射回全长度数组（start_idx 之前为空）
        raw_probs = np.full(n_bars, np.nan, dtype=np.float64)
        raw_probs[start_idx:] = probs

        # 应用 pred_next 延迟（单次延迟 pred_next - 1）
        delayed_probs = np.full(n_bars, np.nan, dtype=np.float64)
        if delay == 0:
            delayed_probs = raw_probs
        else:
            first_valid = start_idx + delay
            if first_valid < n_bars:
                delayed_probs[first_valid:] = raw_probs[start_idx : n_bars - delay]

        # 应用 threshold 和 filters 得到最终预测
        preds = np.zeros(n_bars, dtype=np.int32)
        valid_mask = ~np.isnan(delayed_probs)
        if valid_mask.any():
            preds[valid_mask] = np.array(
                [mc._apply_filters(p) for p in delayed_probs[valid_mask]]
            )

        delayed_preds[m] = preds
        unique, counts = np.unique(preds, return_counts=True)
        print(f"    Unique predictions: {dict(zip(unique, counts))}")
        non_zero_idx = np.argmax(preds != 0) if (preds != 0).any() else -1
        print(
            f"  Model {m}: pred_next={pred_next}, delay={delay}, first non-zero at idx {non_zero_idx}"
        )

    # 聚合投票：全票做多=1，全票做空=-1，其他=0
    print("  Aggregating votes...")
    signals = np.zeros(n_bars, dtype=np.int32)

    for i in range(n_bars):
        votes = [delayed_preds[m][i] for m in models]
        if all(v == 1 for v in votes):
            signals[i] = 1
        elif all(v == -1 for v in votes):
            signals[i] = -1

    # 统计信号分布
    unique, counts = np.unique(signals, return_counts=True)
    print(f"  Signal distribution: {dict(zip(unique, counts))}")

    return signals


# === 向量化回测主循环 ===
def run_vectorized_backtest(
    fusion_bars: np.ndarray, signals: np.ndarray
) -> BacktestEngine:
    """
    向量化回测：信号批量预计算，交易顺序执行

    交易逻辑与逐步回测完全相同，确保结果一致
    """
    print("Running vectorized backtest...")
    engine = BacktestEngine()

    n_bars = len(fusion_bars)

    for bar_idx in range(MIN_FUSION_BARS, n_bars):
        bar = fusion_bars[bar_idx]
        timestamp = int(bar[0])
        close_price = bar[2]
        high_price = bar[3]
        low_price = bar[4]

        # 止损检查
        stop_loss_triggered = engine.check_stop_loss(
            bar_idx, timestamp, high_price, low_price
        )

        if not stop_loss_triggered:
            signal = signals[bar_idx]

            # 持仓管理
            if engine.position.is_long and signal != 1:
                engine.close_position(bar_idx, timestamp, close_price, "signal")
            elif engine.position.is_short and signal != -1:
                engine.close_position(bar_idx, timestamp, close_price, "signal")

            # 入场
            if engine.position.is_flat:
                if signal == 1:
                    engine.open_long(bar_idx, timestamp, close_price, bar_idx)
                elif signal == -1:
                    engine.open_short(bar_idx, timestamp, close_price, bar_idx)

        engine.record_equity(timestamp, close_price)

        # 进度显示
        if (bar_idx - MIN_FUSION_BARS) % 500 == 0:
            progress = (bar_idx - MIN_FUSION_BARS) / (n_bars - MIN_FUSION_BARS) * 100
            print(f"  Progress: {progress:.1f}% (bar {bar_idx}/{n_bars})")

    # 强制平仓
    if not engine.position.is_flat:
        last_bar = fusion_bars[-1]
        engine.close_position(n_bars - 1, int(last_bar[0]), last_bar[2], "end")
        print("  Force closed position at end")

    return engine


# === 结果保存和展示 ===
def create_output_dir(models: list[str], backtest_type: str) -> Path:
    """创建带时间戳的输出目录"""
    model_str = "_".join(sorted(models))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_str}_{backtest_type}_{timestamp}"
    output_dir = Path("backtest_results") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_backtest_results(
    engine: BacktestEngine,
    fusion_bars: np.ndarray,
    output_dir: Path,
    models: list[str],
    trade_start_idx: int,
    backtest_type: str = "vectorized",
):
    """保存回测结果（使用增强的指标模块）"""
    print(f"\nSaving results to {output_dir}/")

    # 保存交易记录
    trades_data = [t.model_dump() for t in engine.trades]
    with open(output_dir / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    print(f"  - trades.json: {len(engine.trades)} trades")

    # 保存权益曲线
    equity_df = pd.DataFrame(engine.equity_curve, columns=["timestamp", "equity"])
    equity_df.to_csv(output_dir / "equity.csv", index=False)
    print(f"  - equity.csv: {len(engine.equity_curve)} data points")

    # 计算全面的金融指标
    metrics, arrays = calculate_all_metrics(
        equity_curve=engine.equity_curve,
        trades=trades_data,
        fusion_bars=fusion_bars,
        starting_balance=engine.starting_balance,
        leverage=engine.leverage,
        fee_rate=engine.fee_rate,
        stop_loss_ratio=engine.stop_loss_ratio,
        models=models,
        start_date=TEST_START,
        end_date=TEST_END,
        backtest_type=backtest_type,
        trade_start_idx=trade_start_idx,
    )

    # 保存指标 JSON
    json_path = save_metrics_json(metrics, output_dir)
    print(f"  - {json_path.name}")

    # 保存文本报告
    report_path = save_metrics_report(metrics, output_dir)
    print(f"  - {report_path.name}")

    # 绘制并保存图表
    chart_path = plot_backtest_results(metrics, arrays, output_dir)
    print(f"  - {chart_path.name}")

    # 打印报告到控制台
    print("\n" + print_metrics_report(metrics))


# === 一致性验证 ===
def verify_consistency(seq_trades_path: Path, vec_trades_path: Path) -> bool:
    """验证两种回测方式结果一致"""
    print("\n" + "=" * 60)
    print("CONSISTENCY VERIFICATION")
    print("=" * 60)

    with open(seq_trades_path, "r") as f:
        seq_trades = json.load(f)
    with open(vec_trades_path, "r") as f:
        vec_trades = json.load(f)

    if len(seq_trades) != len(vec_trades):
        print(f"FAILED: Trade count mismatch: {len(seq_trades)} vs {len(vec_trades)}")
        return False

    mismatches = []
    for i, (seq_t, vec_t) in enumerate(zip(seq_trades, vec_trades)):
        if seq_t["bar_idx"] != vec_t["bar_idx"]:
            mismatches.append(
                f"Trade {i}: bar_idx {seq_t['bar_idx']} vs {vec_t['bar_idx']}"
            )
        if seq_t["action"] != vec_t["action"]:
            mismatches.append(
                f"Trade {i}: action {seq_t['action']} vs {vec_t['action']}"
            )
        if abs(seq_t["price"] - vec_t["price"]) > 1e-6:
            mismatches.append(f"Trade {i}: price {seq_t['price']} vs {vec_t['price']}")

    if mismatches:
        print(f"FAILED: Found {len(mismatches)} mismatches:")
        for m in mismatches[:10]:
            print(f"  - {m}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        return False

    # 验证最终余额
    seq_final = seq_trades[-1]["balance"] if seq_trades else STARTING_BALANCE
    vec_final = vec_trades[-1]["balance"] if vec_trades else STARTING_BALANCE
    if abs(seq_final - vec_final) > 0.01:
        print(f"FAILED: Final balance mismatch: ${seq_final:.2f} vs ${vec_final:.2f}")
        return False

    print(f"PASSED: {len(seq_trades)} trades match exactly")
    print(f"Final balance: ${seq_final:.2f}")
    print("=" * 60)
    return True


# === 主函数 ===
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Vectorized Backtest Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python flow_backtest_vectorized.py                    # 使用默认模型
  python flow_backtest_vectorized.py c_L9_N1 c_L9_N2   # 使用指定模型
  python flow_backtest_vectorized.py c_L6_N1 r_L5_N2 c_L9_N1  # 使用多个模型
        """,
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help=f"模型名称列表（默认: {DEFAULT_MODELS}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = args.models

    print("=" * 60)
    print("Vectorized Backtest Flow")
    print("=" * 60)
    print(f"Period: {TEST_START} to {TEST_END}")
    print(f"Models: {models}")
    print()

    # 1. 加载模型
    print("Loading models...")
    model_features, global_features = collect_model_features(MODEL_DIR, models)
    print(f"  Global features: {len(global_features)}")

    reducers = {}
    model_containers = {}
    for m in models:
        reducers[m] = ARDVAE.load(str(MODEL_DIR / m), m)
        mc = LGBMContainer(*model_name_to_params(m), model_dir=MODEL_DIR)
        mc.is_livetrading = True
        model_containers[m] = mc

    # 2. 加载数据
    print("Loading raw candles from Jesse...")
    _, raw_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(TEST_START),
        helpers.date_to_timestamp(TEST_END),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )
    raw_candles = raw_candles[raw_candles[:, 5] > 0]
    print(f"  Loaded {len(raw_candles)} 1m candles")

    # 3. 生成融合K线
    print("Generating fusion bars...")
    bar_container = DemoBar(max_bars=-1)
    bar_container.update_with_candles(raw_candles)
    fusion_bars = bar_container.get_fusion_bars()
    print(f"  Generated {len(fusion_bars)} fusion bars")

    if len(fusion_bars) < MIN_FUSION_BARS:
        raise ValueError(
            f"Not enough fusion bars: {len(fusion_bars)} < {MIN_FUSION_BARS}"
        )

    # 4. 计算特征
    print("Computing features...")
    calc = SimpleFeatureCalculator(verbose=False)
    calc.load(fusion_bars, sequential=True)
    features_dict = calc.get(global_features)
    features_df = pd.DataFrame(features_dict)
    if len(features_df) != len(fusion_bars):
        raise ValueError(
            f"Feature length mismatch: features={len(features_df)}, bars={len(fusion_bars)}"
        )
    print(f"  Computed {len(global_features)} features, shape: {features_df.shape}")

    trade_start_idx, first_valid_idx = determine_warmup_start_idx(
        features_df, MIN_FUSION_BARS
    )
    print(
        f"Warmup complete at bar {first_valid_idx}, trade starts at {trade_start_idx}"
    )

    # 5. 批量生成信号（含延迟）
    signals = generate_all_signals_with_delay(
        features_df=features_df,
        models=models,
        model_containers=model_containers,
        model_features=model_features,
        reducers=reducers,
        trade_start_idx=trade_start_idx,
    )

    # 6. 运行回测
    engine = run_vectorized_backtest(fusion_bars, signals)

    # 7. 创建输出目录并保存结果
    output_dir = create_output_dir(models, "vectorized")
    save_backtest_results(
        engine=engine,
        fusion_bars=fusion_bars,
        output_dir=output_dir,
        models=models,
        trade_start_idx=trade_start_idx,
        backtest_type="vectorized",
    )

    # 8. 验证一致性（如果存在对应的逐步回测结果）
    # 注意：由于新增了时间戳目录，一致性验证需要手动指定路径
    print("\nNote: For consistency verification, compare trades.json files manually")


if __name__ == "__main__":
    main()
