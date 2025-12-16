"""
保存测试用的 candles 数据到文件

运行方式（从项目根目录）:
    python scripts/save_test_candles.py
"""

import os
from pathlib import Path

import numpy as np

# 配置
TEST_START = "2025-04-01"
TEST_END = "2025-06-01"
OUTPUT_DIR = Path("data/test_candles")


def main():
    from research.model_pick.candle_fetch import FusionCandles, bar_container

    print(f"加载 FusionCandles: {TEST_START} ~ {TEST_END}")
    print(f"bar_container.THRESHOLD = {bar_container.THRESHOLD}")

    candle_container = FusionCandles(
        exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
    )
    candles = candle_container.get_candles(TEST_START, TEST_END)

    print(f"加载完成: {len(candles)} 条 fusion bars")
    print(f"Shape: {candles.shape}")

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存
    output_path = OUTPUT_DIR / f"fusion_candles_{TEST_START}_{TEST_END}.npy"
    np.save(output_path, candles)
    print(f"保存到: {output_path}")

    # 验证
    loaded = np.load(output_path)
    assert np.array_equal(loaded, candles), "验证失败：保存的数据与原始数据不一致"
    print("验证通过")


if __name__ == "__main__":
    main()
