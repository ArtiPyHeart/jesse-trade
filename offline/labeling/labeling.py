"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
"""

from os import cpu_count

import numpy as np
import pandas as pd
from jesse import helpers
from numba import njit

from custom_indicators.utils.multiprocess import mp_pandas_obj
from custom_indicators.utils.volatility import _get_daily_vol


# 新增的加速函数（可选）
@njit
def _cusum_filter_numba(log_rets, thresholds):
    t_events = []
    s_pos = 0.0
    s_neg = 0.0
    for i in range(len(log_rets)):
        ret = log_rets[i]
        thresh = thresholds[i]
        pos = s_pos + ret
        neg = s_neg + ret
        s_pos = pos if pos > 0 else 0.0
        s_neg = neg if neg < 0 else 0.0
        if s_neg < -thresh or s_pos > thresh:
            t_events.append(i)  # 此处返回的是数组下标，需要对应到时间索引上
            s_pos = 0.0
            s_neg = 0.0
    return t_events


# Snippet 2.4, page 39, The Symmetric CUSUM Filter.
def _cusum_filter(raw_time_series, threshold, time_stamps=True):
    """
    Advances in Financial Machine Learning, Snippet 2.4, page 39.

    The Symmetric Dynamic/Fixed CUSUM Filter.

    The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of a measured quantity
    away from a target value. The filter is set up to identify a sequence of upside or downside divergences from any
    reset level zero. We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple events are not triggered by raw_time_series
    hovering around a threshold level, which is a flaw suffered by popular market signals such as Bollinger Bands.
    It will require a full run of length threshold for raw_time_series to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine whether the occurrence
    of such events constitutes actionable intelligence. Below is an implementation of the Symmetric CUSUM filter.

    Note: As per the book this filter is applied to closing prices but we extended it to also work on other
    time series such as volatility.

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, the function captures
                      it as an event, can be dynamic if threshold is pd.Series
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """
    # 转换数据
    raw_time_series = pd.DataFrame(raw_time_series)
    raw_time_series.columns = ["price"]
    raw_time_series["log_ret"] = raw_time_series.price.apply(np.log).diff()

    if isinstance(threshold, (float, int)):
        raw_time_series["threshold"] = threshold
    elif isinstance(threshold, pd.Series):
        raw_time_series.loc[threshold.index, "threshold"] = threshold
    else:
        raise ValueError("threshold is neither float nor pd.Series!")

    raw_time_series = raw_time_series.iloc[1:]  # 去除首个NaN

    # 提取计算所需的数组
    log_rets = raw_time_series["log_ret"].values
    thresholds = raw_time_series["threshold"].values

    # 调用Numba加速版本
    event_indices = _cusum_filter_numba(log_rets, thresholds)

    # 将下标转回索引
    t_events = raw_time_series.index[event_indices]

    if time_stamps:
        return pd.DatetimeIndex(t_events)
    return list(t_events)


def z_score_filter(
    raw_time_series, mean_window, std_window, z_score=3, time_stamps=True
):
    """
    Filter which implements z_score filter
    (https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data)

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param mean_window: (int): Rolling mean window
    :param std_window: (int): Rolling std window
    :param z_score: (float): Number of standard deviations to trigger the event
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """
    t_events = raw_time_series[
        raw_time_series
        >= raw_time_series.rolling(window=mean_window).mean()
        + z_score * raw_time_series.rolling(window=std_window).std()
    ].index
    if time_stamps:
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps
    return t_events


@njit
def _apply_pt_sl_for_single_event(
    close_values,
    start_idx,
    end_idx,
    initial_price,
    side_value,
    stop_loss_val,
    profit_taking_val,
):
    """
    使用Numba加速的子函数，对单个event的SL/PT最早触发点做计算。
    返回 (sl_idx, pt_idx)，分别表示触发SL和PT的数组下标；
    如果没有触发则返回 -1。
    """
    sl_idx = -1
    pt_idx = -1

    for i in range(start_idx, end_idx + 1):
        # 计算从 start_idx 到当前 i 的收益
        cum_return = (close_values[i] / initial_price - 1.0) * side_value

        # 检查是否触发SL
        if cum_return < stop_loss_val and sl_idx < 0:
            sl_idx = i

        # 检查是否触发PT
        if cum_return > profit_taking_val and pt_idx < 0:
            pt_idx = i

        # 如果都触发了，就可以提前退出
        if sl_idx >= 0 and pt_idx >= 0:
            break

    return sl_idx, pt_idx


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def _apply_pt_sl_on_t1(close, events, pt_sl, molecule):  # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # 如果没有启用 pt/sl，则保留NaN
    if profit_taking_multiple > 0:
        profit_taking_series = profit_taking_multiple * events_["trgt"]
    else:
        profit_taking_series = pd.Series(index=events.index, dtype=float)

    if stop_loss_multiple > 0:
        stop_loss_series = -stop_loss_multiple * events_["trgt"]
    else:
        stop_loss_series = pd.Series(index=events.index, dtype=float)

    out["pt"] = pd.Series(dtype=events.index.dtype)
    out["sl"] = pd.Series(dtype=events.index.dtype)

    # 将 close.index 映射为数组下标，便于Numba加速子函数调用
    close_index_to_pos = {v: i for i, v in enumerate(close.index)}
    close_values = close.values  # numpy array

    for loc, vertical_barrier in events_["t1"].fillna(close.index[-1]).items():
        # 获取事件所在位置的数组下标
        start_idx = close_index_to_pos[loc]
        # 如果 vertical_barrier 超出范围，会自动取 close.index[-1]
        end_idx = (
            close_index_to_pos[vertical_barrier]
            if vertical_barrier in close_index_to_pos
            else len(close) - 1
        )

        side_val = events_.at[loc, "side"] if "side" in events_.columns else 1.0
        stop_loss_val = (
            stop_loss_series[loc] if not pd.isna(stop_loss_series[loc]) else -999999.0
        )
        profit_taking_val = (
            profit_taking_series[loc]
            if not pd.isna(profit_taking_series[loc])
            else 999999.0
        )

        # 调用 numba 子函数，获取最早触发SL / PT的数组下标
        sl_idx, pt_idx = _apply_pt_sl_for_single_event(
            close_values,
            start_idx,
            end_idx,
            close_values[start_idx],
            side_val,
            stop_loss_val,
            profit_taking_val,
        )

        # 将数组下标映射回时间索引
        if sl_idx >= 0:
            out.at[loc, "sl"] = close.index[sl_idx]
        if pt_idx >= 0:
            out.at[loc, "pt"] = close.index[pt_idx]

    return out


# Snippet 3.4 page 49, Adding a Vertical Barrier
def _add_vertical_barrier(
    t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0
):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.

    Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(
        "{} days, {} hours, {} minutes, {} seconds".format(
            num_days, num_hours, num_minutes, num_seconds
        )
    )
    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[: nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def _get_events(
    close,
    t_events,
    pt_sl,
    target,
    min_ret,
    num_threads=cpu_count() - 1,
    vertical_barrier_times=False,
    side_prediction=None,
    verbose=True,
):
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (pd.Series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] is profit taking multiple
            -events['sl']  is stop loss multiple
    """

    # 1) Get target
    target = target.reindex(t_events)
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.reindex(
            target.index
        )  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat(
        {"t1": vertical_barrier_times, "trgt": target, "side": side_}, axis=1
    )
    events = events.dropna(subset=["trgt"])

    # Apply Triple Barrier
    first_touch_dates = mp_pandas_obj(
        func=_apply_pt_sl_on_t1,
        pd_obj=("molecule", events.index),
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl_,
        verbose=verbose,
    )

    for ind in events.index:
        events.at[ind, "t1"] = first_touch_dates.loc[ind, :].dropna().min()

    if side_prediction is None:
        events = events.drop("side", axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events["pt"] = pt_sl[0]
    events["sl"] = pt_sl[1]

    return events


# Snippet 3.9, pg 55, Question 3.3
def _barrier_touched(out_df, events):
    """
    Advances in Financial Machine Learning, Snippet 3.9, page 55, Question 3.3.

    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    :param out_df: (pd.DataFrame) Returns and target
    :param events: (pd.DataFrame) The original events data frame. Contains the pt sl multiples needed here.
    :return: (pd.DataFrame) Returns, target, and labels
    """
    # 1. 先对齐 'events' 中的 'pt' 和 'sl'，令其与 out_df 的索引相同
    pt_series = events["pt"].reindex(out_df.index)
    sl_series = events["sl"].reindex(out_df.index)

    # 2. 计算 log_scale 以及上下轨阈值
    log_scale = np.log(1 + out_df["trgt"])
    pt_threshold = log_scale * pt_series
    sl_threshold = -log_scale * sl_series

    # 3. 使用布尔过滤批量生成 bin 的值
    bin_series = pd.Series(data=0, index=out_df.index)
    bin_series[(out_df["ret"] > 0) & (out_df["ret"] > pt_threshold)] = 1
    bin_series[(out_df["ret"] < 0) & (out_df["ret"] < sl_threshold)] = -1

    out_df["bin"] = bin_series.values  # 将结果赋值回 out_df
    return out_df


# Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
def _get_bins(triple_barrier_events, close):
    """
    Advances in Financial Machine Learning, Snippet 3.7, page 51.

    Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (pd.DataFrame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (pd.Series) Close prices
    :return: (pd.DataFrame) Meta-labeled events
    """

    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=["t1"])
    all_dates = events_.index.union(other=events_["t1"].array).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df["ret"] = np.log(prices.loc[events_["t1"].array].array) - np.log(
        prices.loc[events_.index]
    )
    out_df["trgt"] = events_["trgt"]

    # Meta labeling: Events that were correct will have pos returns
    if "side" in events_:
        out_df["ret"] = out_df["ret"] * events_["side"]  # meta-labeling

    # Added code: label 0 when vertical barrier reached
    out_df = _barrier_touched(out_df, triple_barrier_events)

    # Meta labeling: label incorrect events with a 0
    if "side" in events_:
        out_df.loc[out_df["ret"] <= 0, "bin"] = 0

    # Transform the log returns back to normal returns.
    out_df["ret"] = np.exp(out_df["ret"]) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if "side" in tb_cols:
        out_df["side"] = triple_barrier_events["side"]

    return out_df


# Snippet 3.8 page 54
def drop_labels(events, min_pct=0.05):
    """
    Advances in Financial Machine Learning, Snippet 3.8 page 54.

    This function recursively eliminates rare observations.

    :param events: (dp.DataFrame) Events.
    :param min_pct: (float) A fraction used to decide if the observation occurs less than that fraction.
    :return: (pd.DataFrame) Events.
    """
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)

        if df0.min() > min_pct or df0.shape[0] < 3:
            break

        print("dropped label: ", df0.idxmin(), df0.min())
        events = events[events["bin"] != df0.idxmin()]

    return events


class TripleBarrierLabeler:
    def __init__(
        self,
        candles: np.ndarray,
        min_ret: float = 0.00025,
        num_days: int = 0,
        num_hours: int = 0,
        num_minutes: int = 0,
        num_seconds: int = 0,
        verbose: bool = True,
    ):
        self._verbose = verbose
        self._candles = candles
        self._close = helpers.get_candle_source(self._candles, "close")
        self._timestamp = candles[:, 0]

        self._index = pd.DatetimeIndex(
            [helpers.timestamp_to_time(i) for i in self._timestamp]
        )
        self._close_series = pd.Series(self._close, index=self._index)

        self._cusum_res, self._daily_vol, self._vertical_barriers = self._prepare(
            min_ret, num_days, num_hours, num_minutes, num_seconds
        )

    def _prepare(self, min_ret, num_days, num_hours, num_minutes, num_seconds):
        cusum_res = _cusum_filter(self._close_series, min_ret)
        daily_vol = _get_daily_vol(self._close_series)

        if num_days == 0 and num_hours == 0 and num_minutes == 0 and num_seconds == 0:
            vertical_barriers = False
        else:
            vertical_barriers = _add_vertical_barrier(
                cusum_res,
                self._close_series,
                num_days=num_days,
                num_hours=num_hours,
                num_minutes=num_minutes,
                num_seconds=num_seconds,
            )
        return cusum_res, daily_vol, vertical_barriers

    def side_labels(
        self,
        pt=1,
        sl=1,
        target_ret=0.0005,
    ):
        events = _get_events(
            self._close_series,
            self._cusum_res,
            [pt, sl],
            self._daily_vol,
            target_ret,
            max(1, cpu_count() - 1),
            vertical_barrier_times=self._vertical_barriers,
            verbose=self._verbose,
        )
        side_labels = _get_bins(events, self._close_series)
        return side_labels

    def meta_labels(
        self,
        side_labels: pd.DataFrame,
        pt=1,
        sl=1,
        target_ret=0.0005,
    ):
        side_events = _get_events(
            self._close_series,
            self._cusum_res,
            [pt, sl],
            self._daily_vol,
            target_ret,
            max(1, cpu_count() - 1),
            vertical_barrier_times=self._vertical_barriers,
            side_prediction=side_labels[
                "pred"
            ],  # 使用预测的标签，而非side_label打出的标签
            verbose=self._verbose,
        )
        meta_labels = _get_bins(side_events, self._close_series)
        return meta_labels


def get_candle_series(candles, source="close"):
    timestamp_index = pd.DatetimeIndex(
        [helpers.timestamp_to_time(i) for i in candles[:, 0]]
    )
    src = helpers.get_candle_source(candles, source)
    return pd.Series(src, index=timestamp_index, name=source)


def expand_labels(labels, candles, source="close", fill=0):
    candle_df = get_candle_series(candles, source).to_frame()
    candle_df = candle_df.join(labels)
    if "bin" in candle_df.columns:
        candle_df["bin"] = candle_df["bin"].fillna(fill)
    if "side" in candle_df.columns:
        candle_df["side"] = candle_df["side"].fillna(fill)
    return candle_df


def return_of_label(expanded_labels, source="close"):
    close = expanded_labels[source]
    labels = expanded_labels["bin"]
    HOLD_RETURN = close[-1] - close[0]
    PROFIT = 0
    START_PRICE = 0
    for idx, (c, l) in enumerate(zip(close, labels)):
        if idx == 0:
            continue
        else:
            l = int(l)
            last_l = int(labels[idx - 1])
            # 开多
            if l == 1 and last_l != 1:
                START_PRICE = c
            # 开空
            if l == -1 and last_l != -1:
                START_PRICE = c

            # 平仓
            if l != last_l:
                if last_l == 1:
                    PROFIT += c - START_PRICE
                elif last_l == -1:
                    PROFIT += START_PRICE - c

                if last_l == 0:
                    START_PRICE = c
    return PROFIT / HOLD_RETURN


if __name__ == "__main__":
    from jesse import research

    warmup_candles, trading_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "3m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-12-31"),
    )

    labeler = TripleBarrierLabeler(trading_candles)
    side_labels = labeler.side_labels()
    meta_labels = labeler.meta_labels(side_labels)

    print(meta_labels["bin"].astype(int).value_counts())
    print(meta_labels["side"].astype(int).value_counts())
