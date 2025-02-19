import pandas as pd
from jesse import utils
from jesse.strategies import Strategy, cached

from custom_indicators.features_1m import get_features_1m
from custom_indicators.features_3m import get_features_3m
from custom_indicators.features_15m import get_features_15m
from custom_indicators.selection import (
    META_1M,
    META_3M,
    META_15M,
    SIDE_1M,
    SIDE_3M,
    SIDE_15M,
)
from custom_indicators.model import get_meta_model, get_side_model


class MLV1Strategy(Strategy):

    def __init__(self):
        super().__init__()
        self.meta_model = get_meta_model(self.is_livetrading)
        self.side_model = get_side_model(self.is_livetrading)

    @property
    @cached
    def ml_predict(self):
        candle_1m = self.get_candles(self.exchange, "BTC-USDT", "1m")
        candle_3m = self.get_candles(self.exchange, "BTC-USDT", "3m")
        candle_15m = self.get_candles(self.exchange, "BTC-USDT", "15m")
        f_1m = get_features_1m(candle_1m)
        f_3m = get_features_3m(candle_3m)
        f_15m = get_features_15m(candle_15m)
        all_features = {**f_1m, **f_3m, **f_15m}
        meta_features = pd.DataFrame(
            {k: all_features[k] for k in META_1M + META_3M + META_15M}
        )
        meta_pred = self.meta_model.predict(meta_features)[0]
        if meta_pred <= 0.5:
            return 0
        else:
            side_features = pd.DataFrame(
                {k: all_features[k] for k in SIDE_1M + SIDE_3M + SIDE_15M}
            )
            side_pred = self.side_model.predict(side_features)[0]
            if side_pred > 0.5:
                return 1
            else:
                return -1

    def should_long(self) -> bool:
        # 是否打开多仓
        return self.ml_predict == 1

    def should_short(self) -> bool:
        # 是否打开空仓
        return self.ml_predict == -1

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        return True

    def go_long(self):
        # 打开多仓
        entry_price = self.price
        qty = utils.size_to_qty(self.balance, entry_price)
        self.buy = qty, entry_price
        # self.stop_loss = qty, entry_price * 0.95

    def go_short(self):
        # 打开空仓
        entry_price = self.price
        qty = utils.size_to_qty(self.balance, entry_price)
        self.sell = qty, entry_price
        # self.stop_loss = qty, entry_price * 1.05

    def update_position(self) -> None:
        # 更新仓位
        if self.is_long:
            if not self.should_long():
                self.liquidate()
        if self.is_short:
            if not self.should_short():
                self.liquidate()

    def before_terminate(self):
        # 在策略结束时执行，可以用于关闭仓位
        self.liquidate()
