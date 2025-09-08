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
        # raw_result形式为，保存了所有有效window的切片列表，内部元素为(窗口大小*列数)，获取原始结果调用res()
        # ⚠️需进一步加工就直接取出raw_result进一步使用转换链加工后再采用类似于res()函数的方式返回
        # sequential=True时raw_result会保存所有窗口，sequential=False时只会计算最后一个窗口
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

    def _lag(self, value, n: int = 0):
        if n == 0:
            return value
        lag_result = [i[:-n] for i in value]
        return lag_result

    def _dt(self, value):
        dt_result = [i[1:] - i[:-1] for i in value]
        return dt_result

    def _ddt(self, value):
        dt_result = self._dt(value)
        ddt_result = self._dt(dt_result)
        return ddt_result

    def res(self, lag: int = 0, dt: bool = False, ddt: bool = False):
        if dt:
            dt_result = self._dt(self.raw_result)
            lag_result = self._lag(dt_result, lag)
        elif ddt:
            ddt_result = self._ddt(self.raw_result)
            lag_result = self._lag(ddt_result, lag)
        else:
            lag_result = self._lag(self.raw_result, lag)

        final_res = np.array([i[-1] for i in lag_result])
        if self.sequential:
            final_res = _fill_gap(final_res, self.candles)
            return final_res
        else:
            return final_res[-1:]
