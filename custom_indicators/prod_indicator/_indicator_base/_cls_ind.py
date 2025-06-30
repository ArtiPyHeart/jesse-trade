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

    def dt(self):
        dt_result = [i[1:] - i[:-1] for i in self.raw_result]
        dt_result = np.array([i[-1] for i in dt_result])
        if self.sequential:
            dt_result = _fill_gap(dt_result, self.candles)
        return dt_result

    def ddt(self):
        dt_result = [i[1:] - i[:-1] for i in self.raw_result]
        ddt_result = [i[1:] - i[:-1] for i in dt_result]
        ddt_result = np.array([i[-1] for i in ddt_result])
        if self.sequential:
            ddt_result = _fill_gap(ddt_result, self.candles)
        return ddt_result

    def res(self, n: int = 0, dt: bool = False, ddt: bool = False):
        if dt:
            lag_result = [i[-n] for i in self.dt()]
        elif ddt:
            lag_result = [i[-n] for i in self.ddt()]
        else:
            lag_result = [i[-n] for i in self.raw_result]
        lag_result = np.array(lag_result)
        if self.sequential:
            lag_result = _fill_gap(lag_result, self.candles)
        return lag_result
