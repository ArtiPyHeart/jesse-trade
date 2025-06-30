from abc import ABC, abstractmethod

import numpy as np


def _fill_gap(res: np.ndarray, candles: np.ndarray):
    len_gap = len(candles) - len(res)
    if len_gap > 0:
        res = np.vstack([np.full((len_gap, res.shape[1]), np.nan), res])
    return res


class IndicatorBase(ABC):
    """
    用于应对复杂的滑动窗口指标计算
    """

    def __init__(self, candles: np.ndarray, sequential: bool = False):
        self.candles = candles
        self.sequential = sequential
        self.raw_result: list[np.ndarray] = []

    def process(self):
        if self.sequential:
            self._sequential_process()
        else:
            self._single_process()

    @abstractmethod
    def _single_process(self):
        pass

    def _sequential_process(self):
        pass

    def lag(self, value, n: int = 0):
        if n == 0:
            return value
        lag_result = [i[:-n] for i in value]
        return lag_result

    def dt(self, value):
        dt_result = [i[1:] - i[:-1] for i in value]
        return dt_result

    def ddt(self, value):
        dt_result = self.dt(value)
        ddt_result = self.dt(dt_result)
        return ddt_result

    def res(self, lag: int = 0, dt: bool = False, ddt: bool = False):
        if dt:
            dt_result = self.dt(self.raw_result)
            lag_result = self.lag(dt_result, lag)
        elif ddt:
            ddt_result = self.ddt(self.raw_result)
            lag_result = self.lag(ddt_result, lag)
        else:
            lag_result = self.lag(self.raw_result, lag)

        final_res = np.array([i[-1] for i in lag_result])
        if self.sequential:
            final_res = _fill_gap(final_res, self.candles)
        return final_res
