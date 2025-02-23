import jesse.indicators as ta
import numpy as np
from jesse import utils
from jesse.strategies import Strategy, cached

from custom_indicators import roofing_filter


class TaStrategy(Strategy):

    def __init__(self):
        super().__init__()

    @property
    @cached
    def roofing(self):
        return roofing_filter(self.candles, sequential=True)

    @property
    @cached
    def roofing_1(self):
        return np.roll(self.roofing, 1)

    @property
    @cached
    def long_trigger(self):
        crossed_above = utils.crossed(
            self.roofing[1:], self.roofing_1[1:], direction="above"
        )
        return crossed_above

    @property
    @cached
    def short_trigger(self):
        crossed_below = utils.crossed(
            self.roofing[1:], self.roofing_1[1:], direction="below"
        )
        return crossed_below

    @property
    @cached
    def should_hold_long(self):
        return self.roofing[-1] > self.roofing_1[-1]

    @property
    @cached
    def should_hold_short(self):
        return self.roofing[-1] < self.roofing_1[-1]

    # def example_filter(self):
    #     return abs(self.price - self.long_EMA) < abs(self.price - self.longer_EMA)

    def filters(self) -> list:
        return [
            # self.example_filter,
        ]

    def should_long(self) -> bool:
        # 是否打开多仓
        qty = utils.size_to_qty(
            self.available_margin, self.price, fee_rate=self.fee_rate
        )
        return self.long_trigger and qty > 0

    def should_short(self) -> bool:
        # 是否打开空仓
        qty = utils.size_to_qty(
            self.available_margin, self.price, fee_rate=self.fee_rate
        )
        return self.short_trigger and qty > 0

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        return True

    def go_long(self):
        # 打开多仓
        entry_price = self.price
        qty = utils.size_to_qty(
            self.available_margin, entry_price, fee_rate=self.fee_rate
        )
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * 0.95

    def go_short(self):
        # 打开空仓
        entry_price = self.price
        qty = utils.size_to_qty(
            self.available_margin, entry_price, fee_rate=self.fee_rate
        )
        self.sell = qty, entry_price
        self.stop_loss = qty, entry_price * 1.05

    def update_position(self) -> None:
        if self.is_long:
            if self.short_trigger or not self.should_hold_long:
                self.liquidate()
        if self.is_short:
            if self.long_trigger or not self.should_hold_short:
                self.liquidate()

    def before_terminate(self):
        # 在策略结束时执行，可以用于关闭仓位
        self.liquidate()

    def terminate(self):
        # 在策略结束时执行，用于数据记录等，不可用于关闭仓位
        super().terminate()
