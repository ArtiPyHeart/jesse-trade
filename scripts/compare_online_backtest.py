"""
Compare online (deque-aligned) vs backtest predictions on real candles.

Usage:
    python scripts/compare_online_backtest.py
    python scripts/compare_online_backtest.py c_L4_N1 c_L4_N2 c_L4_N3
    python scripts/compare_online_backtest.py --candles-path data/test_candles/jesse_*.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
from jesse import helpers, research

from backtest_no_jesse import (
    _collect_model_features,
    _compute_global_features,
    _predict_single_model,
    _reduce_features_with_vae,
    _slice_trading_features,
    aggregate_votes,
    generate_all_fusion_bars_with_split,
)
from src.bars.fusion.demo import DemoBar
from strategies.BinanceBtcDemoBar.models.config import (
    LGBMContainer,
    model_name_to_params,
)

# ==================== 配置 ====================
STRATEGY = "BinanceBtcDemoBar"
MODEL_DIR = Path(f"strategies/{STRATEGY}/models")

MODELS = ["c_L4_N1", "c_L4_N2", "c_L4_N3"]

TEST_START = "2025-06-01"
TEST_END = "2025-06-03"
WARMUP_CANDLES_NUM = 40000
DEFAULT_CACHE = Path(
    "data/test_candles/jesse_Binance_Perpetual_Futures_BTC-USDT_1m_2025-05-01_2025-07-01.npz"
)


def _parse_pred_next(model_name: str) -> int:
    parts = model_name.split("_")
    assert len(parts) == 3, f"Invalid model name format: {model_name}"
    assert parts[2].startswith("N"), f"Invalid pred_next format: {parts[2]}"
    return int(parts[2][1:])


def _load_candles_from_cache(
    cache_path: Path, warmup_candles_num: int
) -> tuple[np.ndarray, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if cache_path.suffix == ".npz":
        data = np.load(cache_path, allow_pickle=False)
        if "warmup_candles" in data and "trading_candles" in data:
            return data["warmup_candles"], data["trading_candles"]
        if "all_candles" in data:
            all_candles = data["all_candles"]
        elif "trading_candles" in data:
            all_candles = data["trading_candles"]
        else:
            raise ValueError("Invalid cache file: missing all_candles/trading_candles")
    elif cache_path.suffix == ".npy":
        all_candles = np.load(cache_path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported cache file type: {cache_path.suffix}")

    warmup_candles = all_candles[:warmup_candles_num]
    trading_candles = all_candles[warmup_candles_num:]
    return warmup_candles, trading_candles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare online vs backtest predictions on real candles."
    )
    parser.add_argument("models", nargs="*")
    parser.add_argument("--candles-path", default=None)
    parser.add_argument("--start", default=TEST_START)
    parser.add_argument("--end", default=TEST_END)
    parser.add_argument("--warmup-candles-num", type=int, default=WARMUP_CANDLES_NUM)
    parser.add_argument("--exchange", default="Binance Perpetual Futures")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="1m")
    args = parser.parse_args()

    models = args.models if args.models else MODELS

    print("加载真实K线数据...")
    if args.candles_path:
        warmup_candles, trading_candles = _load_candles_from_cache(
            Path(args.candles_path), args.warmup_candles_num
        )
        print(f"使用缓存: {args.candles_path}")
    elif DEFAULT_CACHE.exists():
        warmup_candles, trading_candles = _load_candles_from_cache(
            DEFAULT_CACHE, args.warmup_candles_num
        )
        print(f"使用默认缓存: {DEFAULT_CACHE}")
    else:
        warmup_candles, trading_candles = research.get_candles(
            args.exchange,
            args.symbol,
            args.timeframe,
            helpers.date_to_timestamp(args.start),
            helpers.date_to_timestamp(args.end),
            warmup_candles_num=args.warmup_candles_num,
            caching=False,
            is_for_jesse=False,
        )

    # 过滤0成交量
    warmup_candles = warmup_candles[warmup_candles[:, 5] >= 0]
    trading_candles = trading_candles[trading_candles[:, 5] >= 0]

    print(f"warmup={len(warmup_candles)}, trading={len(trading_candles)}")

    fusion_bars, warmup_fusion_bars_len = generate_all_fusion_bars_with_split(
        warmup_candles, trading_candles, max_bars=-1
    )
    print(
        f"fusion_bars={len(fusion_bars)}, warmup_fusion_bars={warmup_fusion_bars_len}"
    )

    # 对齐线上 Fusion Bars
    bar_container = DemoBar(max_bars=-1)
    bar_container.update_with_candles(warmup_candles)
    warmup_bars = bar_container.get_fusion_bars()
    bar_container.update_with_candles(np.vstack([warmup_candles, trading_candles]))
    online_bars = bar_container.get_fusion_bars()

    if len(warmup_bars) != warmup_fusion_bars_len:
        raise ValueError(
            "Warmup fusion bars length mismatch. "
            f"online={len(warmup_bars)}, backtest={warmup_fusion_bars_len}"
        )
    if not np.array_equal(fusion_bars, online_bars):
        raise ValueError("Fusion bars mismatch between online and backtest")

    model_features_map, global_features = _collect_model_features(MODEL_DIR, models)

    global_features_df = _compute_global_features(
        fusion_bars,
        warmup_fusion_bars_len,
        global_features,
        verbose=False,
    )

    reduced_by_model = {}
    for m in models:
        model_df = _slice_trading_features(
            global_features_df,
            model_features_map[m],
            warmup_fusion_bars_len,
            m,
        )
        reduced_by_model[m] = _reduce_features_with_vae(model_df, MODEL_DIR, m)

    expected_feature_names = {}
    for m in models:
        container = LGBMContainer(*model_name_to_params(m))
        container.is_livetrading = False
        expected_feature_names[m] = container.model.feature_name()
        actual_feature_names = list(reduced_by_model[m].columns)
        if expected_feature_names[m] != actual_feature_names:
            raise ValueError(
                "LGBM feature names mismatch with ARDVAE outputs. "
                f"model={m}, expected={expected_feature_names[m]}, "
                f"actual={actual_feature_names}"
            )

    print("开始回测式预测...")
    start = time.perf_counter()
    preds_vector = {}
    for m in models:
        preds_vector[m] = _predict_single_model(
            m,
            reduced_by_model[m],
            model_name_to_params,
            LGBMContainer,
        )
    print(f"回测式预测完成: {time.perf_counter() - start:.2f}s")

    signals_vector = aggregate_votes(preds_vector, models)

    print("开始模拟线上时序预测...")
    start = time.perf_counter()
    containers = {m: LGBMContainer(*model_name_to_params(m)) for m in models}
    for c in containers.values():
        c.is_livetrading = False

    preds_seq = {m: [] for m in models}
    rows = len(next(iter(reduced_by_model.values())))
    for i in range(rows):
        for m in models:
            row = reduced_by_model[m].iloc[[i]]
            preds_seq[m].append(containers[m].final_predict(row))
    print(f"线上时序预测完成: {time.perf_counter() - start:.2f}s")

    signals_seq = aggregate_votes(preds_seq, models)

    mismatch_models = {}
    pred_next_gt1 = []
    for m in models:
        pred_next = _parse_pred_next(m)
        mismatch = sum(1 for a, b in zip(preds_vector[m], preds_seq[m]) if a != b)
        mismatch_models[m] = mismatch
        if pred_next > 1:
            pred_next_gt1.append(m)

    signal_mismatch = sum(1 for a, b in zip(signals_vector, signals_seq) if a != b)

    print("\n对比结果:")
    print("模型预测不一致数:", mismatch_models)
    print(f"投票信号不一致数: {signal_mismatch} / {len(signals_vector)}")

    if not pred_next_gt1:
        print("警告: 未包含 pred_next > 1 的模型，无法验证对齐。")

    mismatched_models = {m: c for m, c in mismatch_models.items() if c > 0}
    if mismatched_models:
        raise ValueError(f"Model prediction mismatch detected: {mismatched_models}")
    if signal_mismatch > 0:
        raise ValueError(f"Signal mismatch detected: {signal_mismatch}")


if __name__ == "__main__":
    main()
