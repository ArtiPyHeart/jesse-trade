from typing import Callable, Tuple

import numpy as np


def _simulate_returns_per_bar(
    close_prices: np.ndarray,
    pred_prob: np.ndarray,
    hold_long: Callable,
    hold_short: Callable,
    no_position: Callable,
    trading_fee: float = 0.0005,
) -> Tuple[float, np.ndarray]:
    """
    按照 simple_backtest 的成交/结算逻辑，在每个bar上记录实现的对数收益增量（含手续费）。

    - 与 simple_backtest 保持同一套开平仓规则与费用计提时点；
    - 在开仓/平仓发生的bar记录收益或费用，其他bar收益为0；
    - 返回：总收益（对数）与逐bar收益数组（对数）。
    """
    assert len(close_prices) == len(pred_prob)
    n = len(close_prices)
    ret_per_bar = np.zeros(n, dtype=float)

    is_long = False
    is_short = False
    start_price = close_prices[0]
    long_return = 0.0
    short_return = 0.0

    for i, (c, p) in enumerate(zip(close_prices, pred_prob)):
        # 多头信号
        if bool(hold_long(p)):
            if not is_long:
                if is_short:
                    # 空头平仓（实现收益 + 手续费）
                    is_short = False
                    inc = -np.log(c / start_price) - trading_fee
                    short_return += inc
                    ret_per_bar[i] += inc
                # 多头开仓（仅计提手续费）
                is_long = True
                start_price = c
                long_return -= trading_fee
                ret_per_bar[i] -= trading_fee
            # 持有多头则不在bar内重复计收益（与原实现一致）

        # 空头信号
        if bool(hold_short(p)):
            if not is_short:
                if is_long:
                    # 多头平仓
                    is_long = False
                    inc = np.log(c / start_price) - trading_fee
                    long_return += inc
                    ret_per_bar[i] += inc
                # 空头开仓（仅计提手续费）
                is_short = True
                start_price = c
                short_return -= trading_fee
                ret_per_bar[i] -= trading_fee
            # 持有空头则不在bar内重复计收益

        # 空仓信号
        if bool(no_position(p)):
            if is_long:
                inc = np.log(c / start_price) - trading_fee
                long_return += inc
                ret_per_bar[i] += inc
                is_long = False
            if is_short:
                inc = -np.log(c / start_price) - trading_fee
                short_return += inc
                ret_per_bar[i] += inc
                is_short = False

    total_return = float(long_return + short_return)
    return total_return, ret_per_bar


def simple_backtest_with_calmar(
    close_prices: np.ndarray,
    pred_prob: np.ndarray,
    hold_long: Callable,
    hold_short: Callable,
    no_position: Callable,
    trading_fee: float = 0.0005,
    periods_per_year: int | None = None,
) -> tuple[float, float]:
    """
    在 simple_backtest 的基础上，额外返回 Calmar Ratio。

    - 收益基于对数收益（含手续费），与 simple_backtest 一致；
    - 逐bar构造权益曲线：equity = exp(cumsum(ret_per_bar))；
    - 最大回撤：max_drawdown = max(1 - equity / cummax(equity))；
    - CAGR：若给定 periods_per_year，则按年化；否则以整个样本期为1个“年”。

    :return: (total_log_return, calmar_ratio)
    """
    total_ret, ret_per_bar = _simulate_returns_per_bar(
        close_prices,
        pred_prob,
        hold_long,
        hold_short,
        no_position,
        trading_fee,
    )

    # 构造权益曲线（起始1.0）
    equity = np.exp(np.cumsum(ret_per_bar))
    if equity.size == 0:
        return total_ret, np.nan

    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / peak
    max_dd = float(np.max(drawdown)) if drawdown.size > 0 else 0.0

    # 年化收益率（CAGR）。若未指定periods_per_year，则按样本期=1年处理。
    n = len(close_prices)
    years = (n / periods_per_year) if periods_per_year else 1.0
    final_equity = float(equity[-1])
    cagr = final_equity ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Calmar Ratio：CAGR / Max Drawdown，处理0回撤的边界情况
    if max_dd == 0:
        if cagr > 0:
            calmar = float(np.inf)
        elif cagr < 0:
            calmar = float(-np.inf)
        else:
            calmar = 0.0
    else:
        calmar = cagr / max_dd

    return total_ret, calmar
