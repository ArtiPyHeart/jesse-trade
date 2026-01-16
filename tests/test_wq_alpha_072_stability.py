import numpy as np

from src.indicators.prod.wq_alpha.alpha_072 import alpha_072


def _make_candles(n: int = 120) -> np.ndarray:
    timestamps = np.arange(n, dtype=np.int64)
    base = np.linspace(100.0, 110.0, n, dtype=np.float64)
    close = base + np.sin(np.linspace(0, 4 * np.pi, n))
    open_ = close + 0.1
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = np.full(n, 1000.0, dtype=np.float64)
    return np.column_stack([timestamps, open_, close, high, low, volume])


def test_alpha_072_no_inf_when_denominator_near_zero():
    candles = _make_candles()
    result = alpha_072(candles, sequential=True)

    finite_mask = np.isfinite(result)
    assert finite_mask.any(), "Expected finite values in alpha_072 output"
    assert np.isfinite(result[finite_mask]).all(), "alpha_072 contains inf values"
