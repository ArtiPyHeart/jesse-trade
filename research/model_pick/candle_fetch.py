from jesse import helpers, research

from src.bars.fusion.demo_v2 import DemoV2Bar

bar_container = DemoV2Bar(max_bars=-1)


class FusionCandles:
    def __init__(
        self, exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

        self._fusion_candles = None

    def _fetch_jesse_candles(self, start_date, end_date):
        _, candles = research.get_candles(
            self.exchange,
            self.symbol,
            self.timeframe,
            helpers.date_to_timestamp(start_date),
            helpers.date_to_timestamp(end_date),
            warmup_candles_num=0,
            caching=False,
            is_for_jesse=False,
        )
        return candles

    def get_candles(self, start_date, end_date):
        if self._fusion_candles is None:
            raw_candles = self._fetch_jesse_candles(start_date, end_date)
            bar_container.update_with_candles(raw_candles)
            self._fusion_candles = bar_container.get_fusion_bars()
        return self._fusion_candles
