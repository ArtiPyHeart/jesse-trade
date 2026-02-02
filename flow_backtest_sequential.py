"""
逐步回测流程

模拟线上策略逻辑，便于调试和验证。
基于 strategies/BinanceBtcDemoBar/__init__.py 的交易逻辑。

交易规则：
1. 当前 bar 计算特征和预测（预测会延迟 pred_next - 1 步生效）
2. 获取延迟后的有效投票
3. 全票一致时产生交易信号
4. 当前 bar 的 close 价格执行交易

使用方式：在项目根目录运行
    python flow_backtest_sequential.py                    # 使用默认模型
    python flow_backtest_sequential.py c_L9_N1 c_L9_N2   # 使用指定模型
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


# === 配置参数 ===
STARTING_BALANCE = 10000.0
FEE_RATE = 0.0005  # 买卖各万分之五
LEVERAGE = 3
STOP_LOSS_RATIO_NO_LEVERAGE = 0.05
POSITION_SIZE_RATIO = 0.95
MIN_FUSION_BARS = 512

# 默认模型（可通过命令行覆盖）
DEFAULT_MODELS = ["c_L6_N1", "r_L5_N2"]

# 回测时间范围
ENV_VALUES = load_env_values(Path(".env"))
STRATEGY = get_env_value("STRATEGY_NAME", ENV_VALUES)
TEST_START = get_env_date("TEST_START_DATE", ENV_VALUES)
TEST_END = get_env_date("TEST_END_DATE", ENV_VALUES)

MODEL_DIR = Path(f"strategies/{STRATEGY}/models")
LGBMContainer, model_name_to_params = _load_strategy_config(STRATEGY)


# === 数据结构 ===
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

    bar_idx: int  # fusion bar 索引
    timestamp: int  # 毫秒时间戳
    action: str  # open_long, open_short, close_long, close_short, stop_loss_long, stop_loss_short
    price: float
    qty: float
    fee: float
    pnl: float  # 平仓盈亏（开仓时为0）
    balance: float  # 交易后余额
    signal_bar_idx: int = -1  # 产生信号的 bar 索引


# === 交易引擎 ===
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
        self.stop_loss_ratio = stop_loss_ratio / leverage  # 调整为有杠杆的止损比例
        self.position_size_ratio = position_size_ratio

        self.position = Position()
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[int, float]] = []  # (timestamp, equity)

    def open_long(
        self, bar_idx: int, timestamp: int, price: float, signal_bar_idx: int
    ):
        assert self.position.is_flat, "Cannot open when in position"

        # 计算可用资金和数量
        available = self.balance * self.leverage * self.position_size_ratio
        # 扣除手续费后的数量
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
        """检查止损，返回是否触发"""
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


# === 模型加载辅助函数 ===
def load_model_features(model_dir: Path, model_name: str) -> list[str]:
    """加载模型特征配置"""
    features_path = model_dir / model_name / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.json: {features_path}")

    with open(features_path, "r") as f:
        features = json.load(f)

    return features


def collect_model_features(
    model_dir: Path, models: list[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """收集所有模型的特征"""
    model_features = {}
    global_features_set: set[str] = set()

    for model_name in models:
        features = load_model_features(model_dir, model_name)
        model_features[model_name] = features
        global_features_set.update(features)

    global_features = sorted(global_features_set)
    return model_features, global_features


def align_lgbm_feature_columns(
    df_features: pd.DataFrame, expected_columns: list[str]
) -> pd.DataFrame:
    """对齐 LightGBM 特征列"""
    if df_features.shape[1] != len(expected_columns):
        raise ValueError(
            f"Feature count mismatch: reducer={df_features.shape[1]}, model={len(expected_columns)}"
        )

    if list(df_features.columns) == expected_columns:
        return df_features

    if set(df_features.columns) == set(expected_columns):
        return df_features[expected_columns]

    raise ValueError("Feature name mismatch between reducer output and model")


# === 预测管理器 ===
class PredictionManager:
    """管理多模型预测（pred_next 延迟由 LGBMContainer 内部处理）"""

    def __init__(
        self,
        models: list[str],
        model_containers: dict[str, LGBMContainer],
        model_features: dict[str, list[str]],
        reducers: dict[str, ARDVAE],
    ):
        self.models = models
        self.model_containers = model_containers
        self.model_features = model_features
        self.reducers = reducers

        for m in models:
            mc = model_containers[m]
            print(f"  Model {m}: pred_next={mc.pred_next}, threshold={mc.threshold}")

    def update_and_get_votes(
        self, features_df: pd.DataFrame, bar_idx: int
    ) -> list[int]:
        """
        1. 计算当前 bar 的预测（pred_next 延迟由 LGBMContainer 内部处理）
        2. 返回投票结果

        Returns:
            votes: [-1, 0, 1] 的列表
        """
        latest_features = features_df.iloc[[bar_idx]]
        votes = []

        for m in self.models:
            mc = self.model_containers[m]
            model_feat = latest_features[self.model_features[m]]

            # ARDVAE 降维
            reduced = self.reducers[m].transform(model_feat)

            # 对齐列名
            expected_cols = mc.model.feature_name()
            reduced = align_lgbm_feature_columns(reduced, expected_cols)

            # 计算当前预测（已包含 pred_next 延迟）
            current_pred = mc.final_predict(reduced)

            votes.append(int(current_pred))

        return votes


# === 主函数 ===
def load_data() -> np.ndarray:
    """加载原始K线数据"""
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
    # 过滤零成交量
    raw_candles = raw_candles[raw_candles[:, 5] > 0]
    print(f"  Loaded {len(raw_candles)} 1m candles")
    return raw_candles


def generate_fusion_bars(raw_candles: np.ndarray) -> np.ndarray:
    """生成融合K线"""
    print("Generating fusion bars...")
    bar_container = DemoBar(max_bars=3500)
    bar_container.update_with_candles(raw_candles)
    fusion_bars = bar_container.get_fusion_bars()
    print(f"  Generated {len(fusion_bars)} fusion bars")
    return fusion_bars


def compute_features(
    fusion_bars: np.ndarray, global_features: list[str]
) -> pd.DataFrame:
    """计算全局特征"""
    print("Computing features...")
    calc = SimpleFeatureCalculator(verbose=False)
    calc.load(fusion_bars, sequential=True)
    features_dict = calc.get(global_features)
    features_df = pd.DataFrame(features_dict)
    print(f"  Computed {len(global_features)} features, shape: {features_df.shape}")
    return features_df


def load_models(
    models: list[str],
) -> tuple[
    dict[str, list[str]],
    list[str],
    dict[str, ARDVAE],
    dict[str, LGBMContainer],
]:
    """加载所有模型组件"""
    print("Loading models...")

    # 特征配置
    model_features, global_features = collect_model_features(MODEL_DIR, models)
    print(f"  Global features: {len(global_features)}")

    # 降维器
    reducers = {}
    for m in models:
        reducers[m] = ARDVAE.load(str(MODEL_DIR / m), m)

    # LightGBM 模型
    model_containers = {}
    for m in models:
        mc = LGBMContainer(*model_name_to_params(m))
        mc.is_livetrading = True  # 触发模型加载
        model_containers[m] = mc

    return model_features, global_features, reducers, model_containers


def run_sequential_backtest(
    fusion_bars: np.ndarray,
    features_df: pd.DataFrame,
    prediction_manager: PredictionManager,
    trade_start_idx: int,
) -> BacktestEngine:
    """
    逐步回测主循环

    交易规则：
    1. 当前 bar 计算特征和预测（预测会延迟 pred_next - 1 步生效）
    2. 获取延迟后的有效投票
    3. 全票一致时产生交易信号
    4. 当前 bar 的 close 价格执行交易
    """
    print("Running sequential backtest...")
    engine = BacktestEngine()

    n_bars = len(fusion_bars)
    print(f"  Total fusion bars: {n_bars}")
    print(f"  Trading starts at bar {trade_start_idx}")

    # 遍历每个 fusion bar（从 MIN_FUSION_BARS 开始）
    for bar_idx in range(MIN_FUSION_BARS, n_bars):
        bar = fusion_bars[bar_idx]
        timestamp = int(bar[0])
        close_price = bar[2]
        high_price = bar[3]
        low_price = bar[4]

        if bar_idx < trade_start_idx:
            engine.record_equity(timestamp, close_price)
            continue

        # === 阶段1：计算预测（pred_next 延迟由 LGBMContainer 内部处理）===
        # 重要：每个 bar 都必须计算预测，以推进内部延迟队列
        votes = prediction_manager.update_and_get_votes(features_df, bar_idx)
        model_shows_long = all(v == 1 for v in votes)
        model_shows_short = all(v == -1 for v in votes)

        # === 阶段2：止损检查 ===
        stop_loss_triggered = engine.check_stop_loss(
            bar_idx, timestamp, high_price, low_price
        )

        # === 阶段3：信号检查与交易执行 ===
        if not stop_loss_triggered:
            # 持仓管理：信号不再一致时平仓
            if engine.position.is_long and not model_shows_long:
                engine.close_position(bar_idx, timestamp, close_price, "signal")
            elif engine.position.is_short and not model_shows_short:
                engine.close_position(bar_idx, timestamp, close_price, "signal")

            # 入场：空仓时检查信号
            if engine.position.is_flat:
                if model_shows_long:
                    engine.open_long(bar_idx, timestamp, close_price, bar_idx)
                elif model_shows_short:
                    engine.open_short(bar_idx, timestamp, close_price, bar_idx)

        # === 阶段3：记录权益 ===
        engine.record_equity(timestamp, close_price)

        # 进度显示
        if (bar_idx - MIN_FUSION_BARS) % 500 == 0:
            progress = (bar_idx - MIN_FUSION_BARS) / (n_bars - MIN_FUSION_BARS) * 100
            print(f"  Progress: {progress:.1f}% (bar {bar_idx}/{n_bars})")

    # 回测结束，强制平仓
    if not engine.position.is_flat:
        last_bar = fusion_bars[-1]
        engine.close_position(n_bars - 1, int(last_bar[0]), last_bar[2], "end")
        print("  Force closed position at end")

    return engine


def create_output_dir(models: list[str], backtest_type: str) -> Path:
    """创建带时间戳的输出目录"""
    # 模型名称组合
    model_str = "_".join(sorted(models))
    # 时间戳（自然时间到秒）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 完整目录名
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
    backtest_type: str = "sequential",
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Sequential Backtest Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python flow_backtest_sequential.py                    # 使用默认模型
  python flow_backtest_sequential.py c_L9_N1 c_L9_N2   # 使用指定模型
  python flow_backtest_sequential.py c_L6_N1 r_L5_N2 c_L9_N1  # 使用多个模型
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
    """主入口"""
    args = parse_args()
    models = args.models

    print("=" * 60)
    print("Sequential Backtest Flow")
    print("=" * 60)
    print(f"Period: {TEST_START} to {TEST_END}")
    print(f"Models: {models}")
    print("Parameters:")
    print(f"  - Leverage: {LEVERAGE}")
    print(f"  - Fee Rate: {FEE_RATE}")
    print(f"  - Stop Loss Ratio: {STOP_LOSS_RATIO_NO_LEVERAGE}")
    print(f"  - Position Size Ratio: {POSITION_SIZE_RATIO}")
    print(f"  - Min Fusion Bars: {MIN_FUSION_BARS}")
    print()

    # 1. 加载模型
    model_features, global_features, reducers, model_containers = load_models(models)

    # 2. 加载数据
    raw_candles = load_data()

    # 3. 生成融合K线
    fusion_bars = generate_fusion_bars(raw_candles)

    # 验证融合K线数量
    if len(fusion_bars) < MIN_FUSION_BARS:
        raise ValueError(
            f"Not enough fusion bars: {len(fusion_bars)} < {MIN_FUSION_BARS}"
        )

    # 4. 计算特征
    features_df = compute_features(fusion_bars, global_features)

    # 验证特征长度
    assert len(features_df) == len(fusion_bars), "Feature length mismatch"

    # 计算 warmup 结束位置（允许开头 NaN，warmup 后不允许 NaN）
    trade_start_idx, first_valid_idx = determine_warmup_start_idx(
        features_df, MIN_FUSION_BARS
    )
    print(
        f"Warmup complete at bar {first_valid_idx}, trade starts at {trade_start_idx}"
    )

    # 5. 创建预测管理器
    print("\nInitializing prediction manager...")
    prediction_manager = PredictionManager(
        models=models,
        model_containers=model_containers,
        model_features=model_features,
        reducers=reducers,
    )

    # 6. 运行回测
    engine = run_sequential_backtest(
        fusion_bars, features_df, prediction_manager, trade_start_idx
    )

    # 7. 创建输出目录并保存结果
    output_dir = create_output_dir(models, "sequential")
    save_backtest_results(
        engine=engine,
        fusion_bars=fusion_bars,
        output_dir=output_dir,
        models=models,
        trade_start_idx=trade_start_idx,
        backtest_type="sequential",
    )


if __name__ == "__main__":
    main()
