import pandas as pd
from jesse import utils
from jesse.strategies import Strategy, cached

from custom_indicators.all_features import feature_bundle
from custom_indicators.config import META_ALL, SIDE_ALL
from custom_indicators.model import get_meta_model, get_side_model
from custom_indicators.toolbox.bet_sizing import discretize_position, power_mapping

SHORT_TERM = "10m"
MID_TERM = "25m"
LONG_TERM = "2h"


class MLV1Strategy(Strategy):

    def __init__(self):
        super().__init__()
        self.meta_model = get_meta_model(self.is_livetrading)
        self.side_model = get_side_model(self.is_livetrading)

    @property
    def candles_short(self):
        return self.get_candles(self.exchange, "BTC-USDT", SHORT_TERM)

    @property
    def candles_mid(self):
        return self.get_candles(self.exchange, "BTC-USDT", MID_TERM)

    @property
    def candles_long(self):
        return self.get_candles(self.exchange, "BTC-USDT", LONG_TERM)

    @property
    @cached
    def ml_predict(self):
        f_short = {
            f"{SHORT_TERM}_{k}": v
            for k, v in feature_bundle(self.candles_short).items()
        }
        f_mid = {
            f"{MID_TERM}_{k}": v for k, v in feature_bundle(self.candles_mid).items()
        }
        f_long = {
            f"{LONG_TERM}_{k}": v for k, v in feature_bundle(self.candles_long).items()
        }
        all_features = pd.DataFrame({**f_short, **f_mid, **f_long})
        side_pred = self.side_model.predict(all_features[SIDE_ALL])[0]
        all_features["model_side_res"] = side_pred
        meta_pred = self.meta_model.predict(all_features[META_ALL])[0]
        if meta_pred <= 0.5:
            return 0, meta_pred
        else:
            if side_pred > 0.5:
                return 1, meta_pred
            elif side_pred <= 0.5:
                return -1, meta_pred

    @property
    def discrete_position_threshold(self):
        return 0.05

    @property
    def used_margin(self):
        used = (self.portfolio_value - self.available_margin) / self.leverage
        return used if used > 0 else 0

    @property
    def total_margin(self):
        return self.available_margin + self.used_margin

    @property
    def max_qty(self):
        return utils.size_to_qty(
            self.available_margin, self.price, fee_rate=self.fee_rate
        )

    def get_margin_ratio_from_model(self, meta):
        return discretize_position(
            power_mapping(meta),
            self.used_margin / self.total_margin,
        )

    def percentage_to_qty(self, percentage):
        target_margin = self.total_margin * abs(percentage)
        margin = min(target_margin, self.available_margin)
        qty = utils.size_to_qty(margin, self.price, fee_rate=self.fee_rate)
        return qty

    def open_position_qty(self, long=True):
        entry_price = self.price
        if long:
            stop_loss = entry_price * 0.97
        else:
            stop_loss = entry_price * 1.03
        side, meta = self.ml_predict
        entry_percentage = (
            discretize_position(
                power_mapping(meta),
                self.used_margin / self.total_margin,
            )
            * 100
        )
        qty = utils.risk_to_qty(
            self.available_margin,
            entry_percentage,
            entry_price,
            stop_loss,
            fee_rate=self.fee_rate,
        )
        qty = min(qty, self.max_qty)
        return qty

    def update_position_qty(self):
        _, meta = self.ml_predict
        # 当前持仓比例
        current_margin_ratio = self.used_margin / self.total_margin
        # 目标持仓比例
        target_margin_ratio = (
            self.get_margin_ratio_from_model(meta) - current_margin_ratio
        )
        if abs(target_margin_ratio) < self.discrete_position_threshold:
            return 0, 0
        qty = self.percentage_to_qty(target_margin_ratio)
        return target_margin_ratio, qty

    def should_long(self) -> bool:
        # 是否打开多仓
        side, _ = self.ml_predict
        qty = self.open_position_qty(long=True)
        if side == 1 and qty > 0:
            return True
        else:
            return False

    def should_short(self) -> bool:
        # 是否打开空仓
        side, _ = self.ml_predict
        qty = self.open_position_qty(long=False)
        if side == -1 and qty > 0:
            return True
        else:
            return False

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        return True

    def dollar_bar_filter(self):
        pass

    def filters(self) -> list:
        return [
            self.dollar_bar_filter,
        ]

    def go_long(self):
        # 打开多仓
        entry_price = self.price
        qty = self.open_position_qty(long=True)
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * 0.97

    def go_short(self):
        # 打开空仓
        entry_price = self.price
        qty = self.open_position_qty(long=False)
        qty = min(qty, self.max_qty)
        self.sell = qty, entry_price
        self.stop_loss = qty, entry_price * 1.03

    def update_position(self) -> None:
        # 更新仓位
        _, meta = self.ml_predict
        if self.is_long:
            if not self.should_long():
                self.liquidate()
            else:
                target_margin_ratio, qty = self.update_position_qty()
                if target_margin_ratio > 0:
                    self.buy = qty, self.price
                elif target_margin_ratio < 0:
                    self.sell = qty, self.price
        if self.is_short:
            if not self.should_short():
                self.liquidate()
            else:
                target_margin_ratio, qty = self.update_position_qty()
                if target_margin_ratio > 0:
                    self.sell = qty, self.price
                elif target_margin_ratio < 0:
                    self.buy = qty, self.price

    # def before_terminate(self):
    #     # 在策略结束时执行，可以用于关闭仓位
    #     self.liquidate()
