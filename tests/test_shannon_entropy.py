"""Shannon entropy / surprisal tests."""

import numpy as np


def test_shannon_entropy_gaussian_matches_formula():
    from pyrs_indicators.util_entropy import shannon_entropy_gaussian

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    entropy = shannon_entropy_gaussian(x)

    std = np.std(x, ddof=0)
    expected = 0.5 * np.log(2.0 * np.pi) + np.log(std) + 0.5
    np.testing.assert_allclose(entropy, expected, rtol=1e-10)


def test_shannon_entropy_hist_uniform():
    from pyrs_indicators.util_entropy import shannon_entropy_hist

    x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    entropy = shannon_entropy_hist(x, bins=2)
    np.testing.assert_allclose(entropy, np.log(2.0), rtol=1e-10)


def test_shannon_entropy_gaussian_rolling_shape():
    from pyrs_indicators.util_entropy import shannon_entropy_gaussian_rolling

    data = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    res = shannon_entropy_gaussian_rolling(data, period=3)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isfinite(res[2])
    assert np.isfinite(res[3])


def test_shannon_entropy_hist_rolling_values():
    from pyrs_indicators.util_entropy import shannon_entropy_hist_rolling

    data = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    res = shannon_entropy_hist_rolling(data, period=2, bins=2, min_prob=1e-12)

    assert np.isnan(res[0])
    np.testing.assert_allclose(res[1], 0.0, rtol=1e-10)
    np.testing.assert_allclose(res[2], np.log(2.0), rtol=1e-10)
    np.testing.assert_allclose(res[3], 0.0, rtol=1e-10)


def test_entropy_bar_thresholds_gaussian_and_hist():
    from src.bars.fusion.entropy_bar import EntropyBar

    n = 10
    close = 100 + np.arange(n, dtype=np.float64)
    candles = np.column_stack(
        [
            np.arange(n, dtype=np.float64),
            close,
            close,
            close,
            close,
            np.ones(n, dtype=np.float64),
        ]
    )

    gaussian_bar = EntropyBar(period=3, threshold=1.0, method="gaussian_nll")
    gaussian_thresholds = gaussian_bar.get_thresholds(candles)
    assert len(gaussian_thresholds) == len(candles) - gaussian_bar.period
    assert np.all(np.isfinite(gaussian_thresholds))
    assert np.all(gaussian_thresholds >= 0)

    hist_bar = EntropyBar(
        period=3,
        threshold=1.0,
        method="hist_surprisal",
        hist_bins=5,
        min_prob=1e-6,
    )
    hist_thresholds = hist_bar.get_thresholds(candles)
    assert len(hist_thresholds) == len(candles) - hist_bar.period
    assert np.all(np.isfinite(hist_thresholds))
    assert np.all(hist_thresholds >= 0)
