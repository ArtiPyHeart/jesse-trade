from collections import namedtuple

import numpy as np
from jesse.helpers import get_candle_source
from numba import njit


@njit(cache=True)
def _high_pass_fast(
    source, period
):  # Function is compiled to machine code when called the first time
    k = 1
    alpha = 1 + (np.sin(2 * np.pi * k / period) - 1) / np.cos(2 * np.pi * k / period)
    newseries = np.copy(source)
    for i in range(1, source.shape[0]):
        newseries[i] = (
            (1 - alpha / 2) * source[i]
            - (1 - alpha / 2) * source[i - 1]
            + (1 - alpha) * newseries[i - 1]
        )
    return newseries


@njit(cache=True)
def _bp_fast(
    source, hp, alpha, beta
):  # Function is compiled to machine code when called the first time

    bp = np.copy(hp)
    for i in range(2, source.shape[0]):
        bp[i] = (
            0.5 * (1 - alpha) * hp[i]
            - (1 - alpha) * 0.5 * hp[i - 2]
            + beta * (1 + alpha) * bp[i - 1]
            - alpha * bp[i - 2]
        )

    # fast attack-slow decay AGC
    K = 0.991
    peak = np.copy(bp)
    for i in range(source.shape[0]):
        if i > 0:
            peak[i] = peak[i - 1] * K
        if np.abs(bp[i]) > peak[i]:
            peak[i] = np.abs(bp[i])

    return bp, peak


BandPass = namedtuple("BandPass", ["bp", "bp_normalized", "signal", "trigger"])


def bandpass(
    candles: np.ndarray,
    period: int = 20,
    bandwidth: float = 0.3,
    source_type: str = "close",
    sequential: bool = False,
    min_candles: int = 500,
) -> BandPass:
    """
    BandPass Filter

    :param candles: np.ndarray
    :param period: int - default: 20
    :param bandwidth: float - default: 0.3
    :param source_type: str - default: "close"
    :param sequential: bool - default: False
    :param min_candles: int - minimum required candles for stable calculation
                       default: 5 * period (empirically determined)
                       If len(candles) < min_candles, a warning will be issued

    :return: BandPass(bp, bp_normalized, signal, trigger)
    """
    # 计算默认最小K线数量要求
    if min_candles is None:
        min_candles = 5 * period  # 经验值：至少需要5倍周期的数据才能稳定

    # 检查数据量是否充足
    if len(candles) < min_candles:
        import warnings

        warnings.warn(
            f"⚠️Insufficient candles for stable bandpass calculation. "
            f"Minimum recommended: {min_candles}, got: {len(candles)}. "
            f"Results may be inconsistent with different data slices.",
            RuntimeWarning,
        )

    source = get_candle_source(candles, source_type=source_type)

    # 计算滤波器
    hp = _high_pass_fast(source, 4 * period / bandwidth)

    beta = np.cos(2 * np.pi / period)
    gamma = np.cos(2 * np.pi * bandwidth / period)
    alpha = 1 / gamma - np.sqrt(1 / gamma**2 - 1)

    bp, peak = _bp_fast(source, hp, alpha, beta)

    bp_normalized = bp / peak

    trigger = _high_pass_fast(bp_normalized, period / bandwidth / 1.5)
    signal = (bp_normalized < trigger) * 1 - (trigger < bp_normalized) * 1

    if sequential:
        return BandPass(bp, bp_normalized, signal, trigger)
    else:
        return BandPass(bp[-1:], bp_normalized[-1:], signal[-1:], trigger[-1:])


if __name__ == "__main__":
    import warnings

    candles = np.load("/Users/yangqiuyu/Github/jesse-trade/data/bar_deap_v1.npy")

    print("=" * 60)
    print("Testing bandpass with min_candles parameter")
    print("=" * 60)

    # 完整数据的结果作为基准
    print("\n1. Full data baseline:")
    bp_full = bandpass(candles, sequential=False)
    bp_full_seq = bandpass(candles, sequential=True)
    assert abs(bp_full.bp_normalized[0] - bp_full_seq.bp_normalized[-1]) < 1e-8
    print(f"   Full dataset ({len(candles)} candles): {bp_full.bp_normalized[0]:.8f}")

    # 测试不同切片大小的一致性
    print("\n2. Testing different slice sizes (default min_candles=100):")
    slice_sizes = [30, 50, 80, 100, 200, 500, 1000]

    for size in slice_sizes:
        if size > len(candles):
            continue

        candles_slice = candles[-size:]

        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bp_slice = bandpass(candles_slice)  # 默认min_candles=5*20=100

            # 计算与完整数据的误差
            diff = abs(bp_slice.bp_normalized[0] - bp_full.bp_normalized[0])

            # 检查是否有警告
            has_warning = len(w) > 0 and "Insufficient candles" in str(w[0].message)
            warning_msg = "⚠️ WARNING" if has_warning else "✓ OK"

            print(f"   Size {size:4d}: diff={diff:.6e}  {warning_msg}")

    # 测试自定义min_candles
    print("\n3. Testing custom min_candles settings:")
    test_cases = [
        (50, 30),  # 50个K线，要求最少30个 - 应该OK
        (50, 100),  # 50个K线，要求最少100个 - 应该警告
        (200, 150),  # 200个K线，要求最少150个 - 应该OK
        (200, 300),  # 200个K线，要求最少300个 - 应该警告
    ]

    for candle_count, min_req in test_cases:
        candles_test = candles[-candle_count:]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bp_test = bandpass(candles_test, min_candles=min_req)

            has_warning = len(w) > 0 and "Insufficient candles" in str(w[0].message)
            status = "⚠️ WARNING" if has_warning else "✓ PASS"

            print(f"   {candle_count} candles, min_candles={min_req}: {status}")

    # 测试sequential模式
    print("\n4. Testing sequential mode:")
    candles_300 = candles[-300:]
    candles_50 = candles[-50:]

    bp_300_seq = bandpass(candles_300, sequential=True)
    bp_50_seq = bandpass(candles_50, sequential=True)  # 应该触发警告

    print(f"   300 candles last value: {bp_300_seq.bp_normalized[-1]:.8f}")
    print(f"   50 candles last value:  {bp_50_seq.bp_normalized[-1]:.8f}")
    print(
        f"   Difference: {abs(bp_300_seq.bp_normalized[-1] - bp_50_seq.bp_normalized[-1]):.6e}"
    )

    print("\n" + "=" * 60)
    print("All tests completed!")
