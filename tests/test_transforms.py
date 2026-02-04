import numpy as np

from src.features.simple_feature_calculator.transforms import rolling_median


def test_rolling_median_nan_in_window_returns_nan():
    data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = rolling_median(data, 3)
    # index 2 window includes nan -> nan
    assert np.isnan(result[2])
    # index 3 window has no nan -> valid median
    assert result[3] == 2.0


def test_rolling_median_2d_nan_in_window_returns_nan():
    data = np.array(
        [
            [np.nan, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )
    result = rolling_median(data, 2)
    assert np.isnan(result[1, 0])
    assert result[2, 0] == 2.5
