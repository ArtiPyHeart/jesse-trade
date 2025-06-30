from abc import ABC, abstractmethod

import numpy as np


class IndicatorBase(ABC):
    """
    用于应对复杂的滑动窗口指标计算
    """

    def __init__(self, candles: np.ndarray, sequential: bool = False):
        self.candles = candles
        self.sequential = sequential
        self.raw_result: list[np.ndarray] = []

    @property
    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def _single_process(self):
        pass

    def _sequential_process(self):
        pass

    def dt(self):
        self.raw_result = [i[1:] - i[:-1] for i in self.raw_result]
        return self

    def ddt(self):
        self.dt()
        self.dt()
        return self

    def lag(self, n: int):
        self.raw_result = [i[:-n] for i in self.raw_result]
        return self
