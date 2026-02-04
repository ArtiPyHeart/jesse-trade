"""Tests for stationarity helpers (arch backend)."""

import numpy as np

from research.hurst_adf_kpss.stationarity import run_adf_test, run_kpss_test


def test_adf_kpss_basic_returns_finite() -> None:
    rng = np.random.default_rng(0)
    series = rng.standard_normal(200)

    adf_stat, adf_p = run_adf_test(series)
    kpss_stat, kpss_p = run_kpss_test(series)

    assert np.isfinite(adf_stat)
    assert 0.0 <= adf_p <= 1.0
    assert np.isfinite(kpss_stat)
    assert 0.0 <= kpss_p <= 1.0


def test_adf_kpss_nan_handling() -> None:
    rng = np.random.default_rng(1)
    series = rng.standard_normal(200)
    series[::10] = np.nan

    adf_stat, adf_p = run_adf_test(series)
    kpss_stat, kpss_p = run_kpss_test(series)

    assert np.isfinite(adf_stat)
    assert 0.0 <= adf_p <= 1.0
    assert np.isfinite(kpss_stat)
    assert 0.0 <= kpss_p <= 1.0


def test_adf_kpss_short_series_returns_nan() -> None:
    series = np.arange(5, dtype=float)

    adf_stat, adf_p = run_adf_test(series)
    kpss_stat, kpss_p = run_kpss_test(series)

    assert np.isnan(adf_stat)
    assert np.isnan(adf_p)
    assert np.isnan(kpss_stat)
    assert np.isnan(kpss_p)
