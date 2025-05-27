import numpy as np
from numba import njit


@njit
def build_bar_by_threshold_greater_than(
    candles: np.ndarray,
    condition: np.ndarray,
    threshold: float,
    max_bars: int = -1,
    reverse: bool = False,
) -> np.ndarray:
    if reverse:
        candles = candles[::-1]
        condition = condition[::-1]

    bars = np.zeros((len(candles), 6))
    bar_index = 0

    # bar init
    bar_timestamp = candles[0, 0]
    bar_open = candles[0, 1]
    bar_close = candles[0, 2]
    bar_high = candles[0, 3]
    bar_low = candles[0, 4]
    bar_volume = candles[0, 5]
    is_empty_bar = False

    for i in range(1, len(candles)):
        if condition[i] > threshold:
            if not is_empty_bar:
                # 先要添加老bar
                bars[bar_index, 0] = bar_timestamp
                bars[bar_index, 1] = bar_open
                bars[bar_index, 2] = bar_close
                bars[bar_index, 3] = bar_high
                bars[bar_index, 4] = bar_low
                bars[bar_index, 5] = bar_volume
                bar_index += 1

            # 然后添加新bar
            bars[bar_index, 0] = candles[i, 0]
            bars[bar_index, 1] = candles[i, 1]
            bars[bar_index, 2] = candles[i, 2]
            bars[bar_index, 3] = candles[i, 3]
            bars[bar_index, 4] = candles[i, 4]
            bars[bar_index, 5] = candles[i, 5]
            bar_index += 1

            if 0 < max_bars < bar_index:
                break

            # 重置bar
            bar_timestamp = 0
            bar_open = 0.0
            bar_close = 0.0
            bar_high = 0.0
            bar_low = 9999999999999.0
            bar_volume = 0.0
            is_empty_bar = True
        else:
            if is_empty_bar:
                bar_timestamp = candles[i, 0]
                bar_open = candles[i, 1]
                bar_close = candles[i, 2]
                bar_high = candles[i, 3]
                bar_low = candles[i, 4]
                bar_volume = candles[i, 5]
                is_empty_bar = False
            else:
                bar_timestamp = max(bar_timestamp, candles[i, 0])
                bar_volume += candles[i, 5]
                bar_high = max(bar_high, candles[i, 3])
                bar_low = min(bar_low, candles[i, 4])
                if reverse:
                    bar_open = candles[i, 1]
                else:
                    bar_close = candles[i, 2]

    if reverse:
        return bars[:bar_index][::-1]
    else:
        return bars[:bar_index]


@njit
def build_bar_by_threshold_less_than(
    candles: np.ndarray,
    condition: np.ndarray,
    threshold: float,
    max_bars: int = -1,
    reverse: bool = False,
) -> np.ndarray:
    if reverse:
        candles = candles[::-1]
        condition = condition[::-1]

    bars = np.zeros((len(candles), 6))
    bar_index = 0

    # bar init
    bar_timestamp = candles[0, 0]
    bar_open = candles[0, 1]
    bar_close = candles[0, 2]
    bar_high = candles[0, 3]
    bar_low = candles[0, 4]
    bar_volume = candles[0, 5]
    is_empty_bar = False

    for i in range(1, len(candles)):
        if condition[i] > threshold:
            if not is_empty_bar:
                # 先要添加老bar
                bars[bar_index, 0] = bar_timestamp
                bars[bar_index, 1] = bar_open
                bars[bar_index, 2] = bar_close
                bars[bar_index, 3] = bar_high
                bars[bar_index, 4] = bar_low
                bars[bar_index, 5] = bar_volume
                bar_index += 1

            # 然后添加新bar
            bars[bar_index, 0] = candles[i, 0]
            bars[bar_index, 1] = candles[i, 1]
            bars[bar_index, 2] = candles[i, 2]
            bars[bar_index, 3] = candles[i, 3]
            bars[bar_index, 4] = candles[i, 4]
            bars[bar_index, 5] = candles[i, 5]
            bar_index += 1

            if 0 < max_bars < bar_index:
                break

            # 重置bar
            bar_timestamp = 0
            bar_open = 0.0
            bar_close = 0.0
            bar_high = 0.0
            bar_low = 9999999999999.0
            bar_volume = 0.0
            is_empty_bar = True
        else:
            if is_empty_bar:
                bar_timestamp = candles[i, 0]
                bar_open = candles[i, 1]
                bar_close = candles[i, 2]
                bar_high = candles[i, 3]
                bar_low = candles[i, 4]
                bar_volume = candles[i, 5]
                is_empty_bar = False
            else:
                bar_timestamp = max(bar_timestamp, candles[i, 0])
                bar_volume += candles[i, 5]
                bar_high = max(bar_high, candles[i, 3])
                bar_low = min(bar_low, candles[i, 4])
                if reverse:
                    bar_open = candles[i, 1]
                else:
                    bar_close = candles[i, 2]

    if reverse:
        return bars[:bar_index][::-1]
    else:
        return bars[:bar_index]


@njit
def build_bar_by_cumsum(
    candles: np.ndarray,
    condition: np.ndarray,
    threshold: float,
    max_bars: int = -1,
    reverse: bool = False,
) -> np.ndarray:
    if reverse:
        candles = candles[::-1]
        condition = condition[::-1]

    bars = np.zeros((len(candles), 6))
    bar_index = 0

    # bar init
    bar_timestamp = candles[0, 0]
    bar_open = candles[0, 1]
    bar_close = candles[0, 2]
    bar_high = candles[0, 3]
    bar_low = candles[0, 4]
    bar_volume = candles[0, 5]
    bar_cumsum = condition[0]

    for i in range(1, len(candles)):
        if bar_cumsum <= threshold:
            bar_cumsum += condition[i]
            bar_timestamp = max(bar_timestamp, candles[i, 0])
            bar_volume += candles[i, 5]
            bar_high = max(bar_high, candles[i, 3])
            bar_low = min(bar_low, candles[i, 4])
            if reverse:
                bar_open = candles[i, 1]
            else:
                bar_close = candles[i, 2]
        else:
            bars[bar_index, 0] = bar_timestamp
            bars[bar_index, 1] = bar_open
            bars[bar_index, 2] = bar_close
            bars[bar_index, 3] = bar_high
            bars[bar_index, 4] = bar_low
            bars[bar_index, 5] = bar_volume
            bar_index += 1

            if 0 < max_bars < bar_index:
                break

            # 重置bar
            bar_timestamp = candles[i, 0]
            bar_open = candles[i, 1]
            bar_close = candles[i, 2]
            bar_high = candles[i, 3]
            bar_low = candles[i, 4]
            bar_volume = candles[i, 5]
            bar_cumsum = condition[i]

    if reverse:
        return bars[:bar_index][::-1]
    else:
        return bars[:bar_index]
