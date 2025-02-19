import jesse.indicators as ta
from jesse import utils
from jesse.models import Order
from jesse.strategies import Strategy, cached


class DemoStrategy(Strategy):

    def __init__(self):
        super().__init__()

    @property
    @cached
    def short_trend(self):
        # 15m
        short_ema = ta.ema(self.candles, 21)
        long_ema = ta.ema(self.candles, 50)
        if short_ema > long_ema:
            return 1
        else:
            return -1

    @property
    @cached
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

    def example_filter(self):
        return abs(self.price - self.long_EMA) < abs(self.price - self.longer_EMA)

    def filters(self) -> list:
        return [
            self.example_filter,
        ]

    def should_long(self) -> bool:
        # 是否打开多仓
        return self.short_trend == 1

    def should_short(self) -> bool:
        # 是否打开空仓
        return False

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        return True

    def go_long(self):
        # 打开多仓
        entry_price = self.price
        qty = utils.size_to_qty(self.balance, entry_price)
        self.buy = qty, entry_price

    def go_short(self):
        # 打开空仓
        pass

    def update_position(self) -> None:
        # 更新仓位
        if self.short_trend == -1:
            self.liquidate()

    def before(self) -> None:
        # 在每个candle开始时执行
        super().before()

    def after(self) -> None:
        # 在每个candle结束时执行
        super().after()

    def before_terminate(self):
        # 在策略结束时执行，可以用于关闭仓位
        super().before_terminate()

    def terminate(self):
        # 在策略结束时执行，用于数据记录等，不可用于关闭仓位
        super().terminate()

    ### more events ###

    def on_open_position(self, order: Order) -> None:
        # 当仓位打开时立即执行
        super().on_open_position(order)

    def on_close_position(self, order: Order) -> None:
        # 当仓位关闭时立即执行(已平仓)
        super().on_close_position(order)

    def on_increased_position(self, order: Order) -> None:
        # 当仓位增加时执行
        super().on_increased_position(order)

    def on_reduced_position(self, order: Order) -> None:
        # 当仓位减少时执行(未平仓)
        super().on_reduced_position(order)

    def on_cancel(self) -> None:
        # 该函数在所有有效订单取消后调用。举例来说，如果您使用的自定义值需要在每次完成交易后清空，就可以使用该函数。
        super().on_cancel()

    def on_route_open_position(self, strategy: Strategy) -> None:
        # 当另一个策略打开仓位时执行
        super().on_route_open_position(strategy)

    def on_route_close_position(self, strategy: Strategy) -> None:
        # 当另一个策略关闭仓位时执行
        super().on_route_close_position(strategy)

    def on_route_increased_position(self, strategy: Strategy) -> None:
        # 当另一个策略增加仓位时执行
        super().on_route_increased_position(strategy)

    def on_route_reduced_position(self, strategy: Strategy) -> None:
        # 当另一个策略减少仓位时执行
        super().on_route_reduced_position(strategy)

    def on_route_canceled(self, strategy: Strategy) -> None:
        # 当另一个策略取消订单时执行
        super().on_route_canceled(strategy)
