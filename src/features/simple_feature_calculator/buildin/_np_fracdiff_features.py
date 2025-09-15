"""
基于numpy fracdiff的特征组合
"""

import numpy as np

from src.indicators.numpy_fracdiff.indicator import np_fracdiff
from ..registry import feature

FRAC = 0.7

for l in range(1, 6):

    @feature(name=f"frac_o_o{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="open",
            minus_type="open",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_o_h{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="open",
            minus_type="high",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_o_l{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="open",
            minus_type="low",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_o_c{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="open",
            minus_type="close",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_h_o{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="high",
            minus_type="open",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_h_h{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="high",
            minus_type="high",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_h_l{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="high",
            minus_type="low",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_h_c{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="high",
            minus_type="close",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_l_o{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="low",
            minus_type="open",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_l_h{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="low",
            minus_type="high",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_l_l{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="low",
            minus_type="low",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_l_c{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="low",
            minus_type="close",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_c_o{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="close",
            minus_type="open",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_c_h{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="close",
            minus_type="high",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_c_l{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="close",
            minus_type="low",
            frac=FRAC,
            lag=_lag,
        )

    @feature(name=f"frac_c_c{l}_diff")
    def np_fracdiff_open_open(
        candles: np.ndarray,
        sequential: bool = True,
        _lag=l,
    ):
        return np_fracdiff(
            candles,
            sequential=sequential,
            source_type="close",
            minus_type="close",
            frac=FRAC,
            lag=_lag,
        )
