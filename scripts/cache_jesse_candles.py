"""
Cache Jesse candles to a local file for repeatable tests.

Usage:
    python scripts/cache_jesse_candles.py
    python scripts/cache_jesse_candles.py --start 2025-05-01 --end 2025-07-01
    python scripts/cache_jesse_candles.py --output data/test_candles/custom.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_EXCHANGE = "Binance Perpetual Futures"
DEFAULT_SYMBOL = "BTC-USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_START = "2025-05-01"
DEFAULT_END = "2025-07-01"
DEFAULT_WARMUP = 40000
DEFAULT_OUTPUT_DIR = Path("data/test_candles")


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _default_output_path(
    exchange: str, symbol: str, timeframe: str, start: str, end: str
) -> Path:
    exchange_slug = _slugify(exchange)
    symbol_slug = symbol.replace("/", "-")
    timeframe_slug = timeframe.replace("/", "-")
    filename = f"jesse_{exchange_slug}_{symbol_slug}_{timeframe_slug}_{start}_{end}.npz"
    return DEFAULT_OUTPUT_DIR / filename


def _save_npz(
    path: Path, warmup_candles: np.ndarray, trading_candles: np.ndarray, meta: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        warmup_candles=warmup_candles,
        trading_candles=trading_candles,
        meta=json.dumps(meta),
    )

    loaded = np.load(path, allow_pickle=False)
    assert np.array_equal(loaded["warmup_candles"], warmup_candles), (
        "Warmup candles mismatch after save"
    )
    assert np.array_equal(loaded["trading_candles"], trading_candles), (
        "Trading candles mismatch after save"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Jesse candles to data/.")
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(
            args.exchange, args.symbol, args.timeframe, args.start, args.end
        )
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file exists: {output_path}. Use --overwrite to replace."
        )

    from jesse import helpers, research

    print("Loading candles via research.get_candles...")
    warmup_candles, trading_candles = research.get_candles(
        args.exchange,
        args.symbol,
        args.timeframe,
        helpers.date_to_timestamp(args.start),
        helpers.date_to_timestamp(args.end),
        warmup_candles_num=args.warmup,
        caching=False,
        is_for_jesse=False,
    )

    if warmup_candles is None:
        if args.warmup > 0:
            raise ValueError(
                "warmup_candles is None despite warmup > 0. Check DB data coverage."
            )
        warmup_candles = np.empty((0, 6))
    if trading_candles is None:
        raise ValueError("trading_candles is None. Check DB connection/data.")

    warmup_candles = warmup_candles[warmup_candles[:, 5] >= 0]
    trading_candles = trading_candles[trading_candles[:, 5] >= 0]

    print(f"Warmup candles: {len(warmup_candles)}")
    print(f"Trading candles: {len(trading_candles)}")
    print(f"Output: {output_path}")

    meta = {
        "exchange": args.exchange,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "warmup_candles_num": args.warmup,
    }

    _save_npz(output_path, warmup_candles, trading_candles, meta)
    print("Saved and verified.")


if __name__ == "__main__":
    main()
