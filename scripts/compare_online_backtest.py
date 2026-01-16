"""
Compare online (deque-aligned) vs backtest predictions on real candles.

Usage:
    python scripts/compare_online_backtest.py
    python scripts/compare_online_backtest.py c_L4_N1 c_L4_N2
"""

import sys
import time
from pathlib import Path

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
from strategies.BinanceBtcDemoBar.models.config import (
    LGBMContainer,
    model_name_to_params,
)

# ==================== 配置 ====================
STRATEGY = "BinanceBtcDemoBar"
MODEL_DIR = Path(f"strategies/{STRATEGY}/models")

MODELS = [
    "c_L4_N1",
    "c_L4_N2",
]

TEST_START = "2025-06-01"
TEST_END = "2025-06-03"
WARMUP_CANDLES_NUM = 40000


def _parse_models(args: list[str]) -> list[str]:
    cli_models = [arg for arg in args if arg]
    return cli_models if cli_models else MODELS


def main() -> None:
    models = _parse_models(sys.argv[1:])

    print("加载真实K线数据...")
    warmup_candles, trading_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(TEST_START),
        helpers.date_to_timestamp(TEST_END),
        warmup_candles_num=WARMUP_CANDLES_NUM,
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
    print(f"fusion_bars={len(fusion_bars)}, warmup_fusion_bars={warmup_fusion_bars_len}")

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
    for m in models:
        mismatch = sum(
            1 for a, b in zip(preds_vector[m], preds_seq[m]) if a != b
        )
        mismatch_models[m] = mismatch

    signal_mismatch = sum(
        1 for a, b in zip(signals_vector, signals_seq) if a != b
    )

    print("\n对比结果:")
    print("模型预测不一致数:", mismatch_models)
    print(f"投票信号不一致数: {signal_mismatch} / {len(signals_vector)}")


if __name__ == "__main__":
    main()
