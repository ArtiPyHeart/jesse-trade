"""
WorldQuant 101 Alphas - Core Operators

All time series operators for implementing WQ101 alphas.
Functions are optimized with numba @njit for performance.
"""

import numpy as np
from numba import njit


# =============================================================================
# Time Series Functions
# =============================================================================


@njit(cache=True)
def ts_delay(x: np.ndarray, n: int) -> np.ndarray:
    """
    Delay (lag) time series by n periods.

    Args:
        x: Input array
        n: Number of periods to delay

    Returns:
        Delayed array with NaN for first n values
    """
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n > 0 and n < len(x):
        result[n:] = x[:-n]
    elif n == 0:
        result[:] = x
    return result


@njit(cache=True)
def ts_delta(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute n-period difference: x[t] - x[t-n]

    Args:
        x: Input array
        n: Number of periods

    Returns:
        Delta array with NaN for first n values
    """
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n > 0 and n < len(x):
        result[n:] = x[n:] - x[:-n]
    elif n == 0:
        result[:] = 0.0
    return result


@njit(cache=True)
def ts_sum(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling sum over n periods.
    Handles NaN values by propagating them.

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling sum array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        window = x[i - n + 1 : i + 1]
        # Check if any NaN in window
        has_nan = False
        for j in range(n):
            if np.isnan(window[j]):
                has_nan = True
                break
        if not has_nan:
            result[i] = np.sum(window)

    return result


@njit(cache=True)
def ts_mean(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling mean over n periods.

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling mean array with NaN for first n-1 values
    """
    return ts_sum(x, n) / n


@njit(cache=True)
def ts_stddev(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling standard deviation over n periods.
    Uses sample std (ddof=1).

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling std array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 1 or n > length:
        return result

    for i in range(n - 1, length):
        window = x[i - n + 1 : i + 1]
        mean = np.mean(window)
        var_sum = 0.0
        for j in range(n):
            var_sum += (window[j] - mean) ** 2
        result[i] = np.sqrt(var_sum / (n - 1))

    return result


@njit(cache=True)
def ts_rank(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling rank (percentile) within window.
    Returns the percentile rank of current value in the window (0 to 1).

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling rank array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        window = x[i - n + 1 : i + 1]
        current_val = x[i]
        # Count how many values are less than current
        count_less = 0
        for j in range(n):
            if window[j] < current_val:
                count_less += 1
        # Percentile rank (add 1 to make it 1-based, then normalize)
        result[i] = (count_less + 1) / n

    return result


@njit(cache=True)
def ts_min(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling minimum over n periods.

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling min array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        result[i] = np.min(x[i - n + 1 : i + 1])

    return result


@njit(cache=True)
def ts_max(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling maximum over n periods.

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling max array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        result[i] = np.max(x[i - n + 1 : i + 1])

    return result


@njit(cache=True)
def ts_argmax(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling argmax: position of maximum value within window.
    Returns days since the max (0 = current day is max, n-1 = first day in window is max).

    Args:
        x: Input array
        n: Window size

    Returns:
        Position of max in window (0 to n-1)
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        window = x[i - n + 1 : i + 1]
        max_idx = 0
        max_val = window[0]
        for j in range(1, n):
            if window[j] > max_val:
                max_val = window[j]
                max_idx = j
        # Return days since max (n-1-max_idx means: if max is at end, result is 0)
        result[i] = float(n - 1 - max_idx)

    return result


@njit(cache=True)
def ts_argmin(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling argmin: position of minimum value within window.
    Returns days since the min (0 = current day is min).

    Args:
        x: Input array
        n: Window size

    Returns:
        Position of min in window (0 to n-1)
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        window = x[i - n + 1 : i + 1]
        min_idx = 0
        min_val = window[0]
        for j in range(1, n):
            if window[j] < min_val:
                min_val = window[j]
                min_idx = j
        result[i] = float(n - 1 - min_idx)

    return result


@njit(cache=True)
def ts_product(x: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling product over n periods.

    Args:
        x: Input array
        n: Window size

    Returns:
        Rolling product array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    for i in range(n - 1, length):
        prod = 1.0
        for j in range(i - n + 1, i + 1):
            prod *= x[j]
        result[i] = prod

    return result


@njit(cache=True)
def ts_corr(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling Pearson correlation between x and y over n periods.

    Args:
        x: First input array
        y: Second input array
        n: Window size

    Returns:
        Rolling correlation array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 1 or n > length:
        return result

    for i in range(n - 1, length):
        wx = x[i - n + 1 : i + 1]
        wy = y[i - n + 1 : i + 1]

        mean_x = np.mean(wx)
        mean_y = np.mean(wy)

        cov = 0.0
        var_x = 0.0
        var_y = 0.0

        for j in range(n):
            dx = wx[j] - mean_x
            dy = wy[j] - mean_y
            cov += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        if var_x > 0 and var_y > 0:
            result[i] = cov / np.sqrt(var_x * var_y)
        else:
            result[i] = 0.0

    return result


@njit(cache=True)
def ts_cov(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling covariance between x and y over n periods.
    Uses sample covariance (ddof=1).

    Args:
        x: First input array
        y: Second input array
        n: Window size

    Returns:
        Rolling covariance array with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 1 or n > length:
        return result

    for i in range(n - 1, length):
        wx = x[i - n + 1 : i + 1]
        wy = y[i - n + 1 : i + 1]

        mean_x = np.mean(wx)
        mean_y = np.mean(wy)

        cov = 0.0
        for j in range(n):
            cov += (wx[j] - mean_x) * (wy[j] - mean_y)

        result[i] = cov / (n - 1)

    return result


# =============================================================================
# Math Functions
# =============================================================================


@njit(cache=True)
def signed_power(x: np.ndarray, a: float) -> np.ndarray:
    """
    Signed power: sign(x) * abs(x)^a
    Preserves sign when raising to power.

    Args:
        x: Input array
        a: Exponent

    Returns:
        Signed power result
    """
    result = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = np.power(np.abs(x[i]), a)
        else:
            result[i] = -np.power(np.abs(x[i]), a)
    return result


@njit(cache=True)
def decay_linear(x: np.ndarray, n: int) -> np.ndarray:
    """
    Linear decay weighted average.
    Weights: [1, 2, 3, ..., n] normalized to sum to 1.
    Most recent value gets highest weight.

    Args:
        x: Input array
        n: Window size

    Returns:
        Decay weighted average with NaN for first n-1 values
    """
    length = len(x)
    result = np.full(length, np.nan, dtype=np.float64)

    if n <= 0 or n > length:
        return result

    # Compute weights: [1, 2, 3, ..., n]
    weights = np.arange(1, n + 1, dtype=np.float64)
    weight_sum = np.sum(weights)

    for i in range(n - 1, length):
        window = x[i - n + 1 : i + 1]
        weighted_sum = 0.0
        for j in range(n):
            weighted_sum += window[j] * weights[j]
        result[i] = weighted_sum / weight_sum

    return result


@njit(cache=True)
def sign_array(x: np.ndarray) -> np.ndarray:
    """
    Element-wise sign function.
    Returns 1 for positive, -1 for negative, 0 for zero.

    Args:
        x: Input array

    Returns:
        Sign array
    """
    result = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        if x[i] > 0:
            result[i] = 1.0
        elif x[i] < 0:
            result[i] = -1.0
        else:
            result[i] = 0.0
    return result


# =============================================================================
# Data Source Functions
# =============================================================================


@njit(cache=True)
def get_returns(close: np.ndarray) -> np.ndarray:
    """
    Compute simple returns: close[t] / close[t-1] - 1

    Args:
        close: Close price array

    Returns:
        Returns array with NaN for first value
    """
    result = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(1, len(close)):
        if close[i - 1] != 0:
            result[i] = close[i] / close[i - 1] - 1.0
    return result


@njit(cache=True)
def get_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Approximate VWAP as typical price: (high + low + close) / 3

    Args:
        high: High price array
        low: Low price array
        close: Close price array

    Returns:
        Approximate VWAP array
    """
    return (high + low + close) / 3.0


@njit(cache=True)
def get_adv(volume: np.ndarray, n: int) -> np.ndarray:
    """
    Average daily volume over n periods.

    Args:
        volume: Volume array
        n: Window size

    Returns:
        ADV array with NaN for first n-1 values
    """
    return ts_mean(volume, n)


# =============================================================================
# Test
# =============================================================================


if __name__ == "__main__":
    print("Testing WQ Alpha Operators...")

    # Test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    # Test ts_delay
    delayed = ts_delay(x, 2)
    assert np.isnan(delayed[0]) and np.isnan(delayed[1])
    assert delayed[2] == 1.0 and delayed[3] == 2.0
    print("  ts_delay: OK")

    # Test ts_delta
    delta = ts_delta(x, 2)
    assert np.isnan(delta[0]) and np.isnan(delta[1])
    assert delta[2] == 2.0 and delta[3] == 2.0  # 3-1=2, 4-2=2
    print("  ts_delta: OK")

    # Test ts_sum
    rolled_sum = ts_sum(x, 3)
    assert np.isnan(rolled_sum[0]) and np.isnan(rolled_sum[1])
    assert rolled_sum[2] == 6.0  # 1+2+3
    assert rolled_sum[3] == 9.0  # 2+3+4
    print("  ts_sum: OK")

    # Test ts_mean
    rolled_mean = ts_mean(x, 3)
    assert rolled_mean[2] == 2.0  # (1+2+3)/3
    print("  ts_mean: OK")

    # Test ts_corr (perfect negative correlation)
    corr = ts_corr(x, y, 5)
    assert abs(corr[4] - (-1.0)) < 1e-10
    print("  ts_corr: OK")

    # Test ts_rank
    rank = ts_rank(x, 3)
    assert rank[2] == 1.0  # 3 is max in [1,2,3], rank = 3/3 = 1.0
    print("  ts_rank: OK")

    # Test ts_min/max
    assert ts_min(x, 3)[2] == 1.0
    assert ts_max(x, 3)[2] == 3.0
    print("  ts_min/ts_max: OK")

    # Test ts_argmax/argmin
    test_arr = np.array([1.0, 5.0, 2.0, 3.0, 4.0])
    argmax = ts_argmax(test_arr, 3)
    # Window [1,5,2]: max is 5 at index 1, days since = 3-1-1 = 1
    assert argmax[2] == 1.0
    print("  ts_argmax/ts_argmin: OK")

    # Test decay_linear
    decay = decay_linear(x, 3)
    # weights [1,2,3], sum=6, window [1,2,3]: (1*1 + 2*2 + 3*3)/6 = 14/6
    expected = (1*1 + 2*2 + 3*3) / 6
    assert abs(decay[2] - expected) < 1e-10
    print("  decay_linear: OK")

    # Test signed_power
    neg_arr = np.array([-2.0, 2.0])
    sp = signed_power(neg_arr, 2)
    assert sp[0] == -4.0 and sp[1] == 4.0
    print("  signed_power: OK")

    # Test sign_array
    sign_test = np.array([-5.0, 0.0, 5.0])
    signs = sign_array(sign_test)
    assert signs[0] == -1.0 and signs[1] == 0.0 and signs[2] == 1.0
    print("  sign_array: OK")

    # Test get_returns
    close = np.array([100.0, 110.0, 105.0])
    returns = get_returns(close)
    assert np.isnan(returns[0])
    assert abs(returns[1] - 0.1) < 1e-10  # (110-100)/100
    print("  get_returns: OK")

    # Test get_vwap
    high = np.array([105.0, 115.0])
    low = np.array([95.0, 105.0])
    close = np.array([100.0, 110.0])
    vwap = get_vwap(high, low, close)
    assert vwap[0] == 100.0  # (105+95+100)/3
    print("  get_vwap: OK")

    print("\nAll operator tests passed!")
