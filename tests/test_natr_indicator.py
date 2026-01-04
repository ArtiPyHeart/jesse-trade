import numpy as np

from src.indicators.prod.natr import natr


def _make_candles(n: int) -> np.ndarray:
    ts = np.arange(n, dtype=np.float64)
    open_ = np.linspace(100, 100 + n - 1, n, dtype=np.float64)
    close = open_ + 1.0
    high = close + 1.0
    low = open_ - 1.0
    volume = np.ones(n, dtype=np.float64)
    return np.column_stack([ts, open_, close, high, low, volume])


def test_natr_sequential_matches_last():
    candles = _make_candles(30)
    seq = natr(candles, period=14, sequential=True)
    last = natr(candles, period=14, sequential=False)

    assert seq.shape == (30,)
    assert np.isnan(seq[:13]).all()
    assert np.isclose(seq[-1], last, equal_nan=True)


def test_natr_short_series_returns_nan():
    candles = _make_candles(10)
    seq = natr(candles, period=14, sequential=True)
    last = natr(candles, period=14, sequential=False)

    assert np.isnan(seq).all()
    assert np.isnan(last)
