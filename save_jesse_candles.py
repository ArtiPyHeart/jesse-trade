"""
保存 Jesse candles 到 data/ 目录（单次全量）

说明:
- 读取 .env 中的 TRAIN_START_DATE 和 TEST_END_DATE 作为时间范围
- 使用 jesse.research.get_candles 获取原始 1m candles
- 保存为 npy 文件，文件名包含日期区间

Usage:
    python save_jesse_candles.py
"""

from pathlib import Path

import numpy as np

from jesse import helpers, research
from src.utils.env_dates import get_env_date, load_env_values


def main() -> None:
    env_values = load_env_values(Path(".env"))
    train_start = get_env_date("TRAIN_START_DATE", env_values)
    test_end = get_env_date("TEST_END_DATE", env_values)

    print(f"保存范围: {train_start} ~ {test_end}")

    _, raw_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(train_start),
        helpers.date_to_timestamp(test_end),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"btc_1m_{train_start}_{test_end}.npy"
    file_path = data_dir / file_name

    np.save(file_path, raw_candles)
    print(f"已保存: {file_path}  shape={raw_candles.shape}")


if __name__ == "__main__":
    main()
