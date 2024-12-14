import jesse.indicators as ta
from jesse import utils
from jesse.strategies import Strategy


class ExampleStrategy(Strategy):

    @property
    def short_trend(self):
        # 15m
        short_ema = ta.ema(self.candles, 21)
        long_ema = ta.ema(self.candles, 50)
        if short_ema > long_ema:
            return 1
        else:
            return -1

    @property
    def btc_long_trend(self):
        # 1h
        # short_ema = ta.ema(self.get_candles(self.exchange, self.symbol, self.timeframe), 21)
        # long_ema = ta.ema(self.get_candles(self.exchange, self.symbol, '1h'), 50)
        short_ema = ta.ema(self.get_candles(self.exchange, "BTC-USDT", "1h"), 21)
        long_ema = ta.ema(self.get_candles(self.exchange, "BTC-USDT", "1h"), 50)
        if short_ema > long_ema:
            return 1
        else:
            return -1

    def should_long(self) -> bool:
        return self.short_trend == 1

    def should_short(self) -> bool:
        return False

    def should_cancel_entry(self) -> bool:
        # Only for limit orders
        return True

    def go_long(self):
        entry_price = self.price
        qty = utils.size_to_qty(self.balance, entry_price)
        self.buy = qty, entry_price

    def go_short(self):
        pass

    def update_position(self) -> None:
        if self.short_trend == -1:
            self.liquidate()
