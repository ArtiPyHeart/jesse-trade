import numpy as np
import pandas as pd
from jesse import helpers, utils
from jesse.strategies import Strategy, cached

from custom_indicators.all_features import FeatureCalculator
from custom_indicators.toolbox.bar.dollar_bar import (
    DollarBarContainer,
    build_dollar_bar,
)

from .config import (
    DOLLAR_BAR_LONG_TERM,
    DOLLAR_BAR_MID_TERM,
    DOLLAR_BAR_SHORT_TERM,
    DOLLAR_BAR_THRESHOLD_LONG,
    DOLLAR_BAR_THRESHOLD_MID,
    DOLLAR_BAR_THRESHOLD_SHORT,
    META_ALL,
    META_DOLLAR_BAR_LONG_FEATURES,
    META_DOLLAR_BAR_MID_FEATURES,
    META_DOLLAR_BAR_SHORT_FEATURES,
    SIDE_ALL,
    SIDE_DOLLAR_BAR_LONG_FEATURES,
    SIDE_DOLLAR_BAR_MID_FEATURES,
    SIDE_DOLLAR_BAR_SHORT_FEATURES,
    get_meta_model,
    get_side_model,
)

META_MODEL_THRESHOLD = 0.5
SIDE_MODEL_THRESHOLD = 0.5
STOP_LOSS_RATIO = 0.05
ORDER_TIMEOUT = 300 * 1000


class BinanceBtcDBar5hAllOrNothing(Strategy):
    def __init__(self):
        super().__init__()
        self.meta_model = get_meta_model(self.is_livetrading)
        self.side_model = get_side_model(self.is_livetrading)

        self.main_bar_container = DollarBarContainer(
            DOLLAR_BAR_THRESHOLD_MID,
            max_bars=5000,
        )

        self.dollar_bar_short_term_fc = FeatureCalculator()
        self.dollar_bar_mid_term_fc = FeatureCalculator()
        self.dollar_bar_long_term_fc = FeatureCalculator()

    def cancel_active_orders(self):
        # 检查超时的活跃订单，如果订单超时依然没有成交，则取消订单
        alive_orders = [o for o in self.orders if o.is_active or o.is_partially_filled]
        for order in alive_orders:
            if helpers.now_to_timestamp() - order.created_at > ORDER_TIMEOUT:
                order.cancel()

    ############################### dollar bar 预处理 ##############################
    def before(self):
        self.main_bar_container.update_with_candle(self.candles)
        # 检查超时的活跃订单，如果订单超时依然没有成交，则取消订单
        self.cancel_active_orders()

    @property
    def should_trade_main_bar(self) -> bool:
        return self.main_bar_container.is_latest_bar_complete()

    @property
    @cached
    def dollar_bar_short_term(self) -> np.ndarray:
        return build_dollar_bar(
            self.candles,
            DOLLAR_BAR_THRESHOLD_SHORT,
            max_bars=5000,
        )

    @property
    def dollar_bar_mid_term(self) -> np.ndarray:
        return self.main_bar_container.get_dollar_bars()

    @property
    @cached
    def dollar_bar_long_term(self) -> np.ndarray:
        return build_dollar_bar(
            self.candles,
            DOLLAR_BAR_THRESHOLD_LONG,
            max_bars=5000,
        )

    ############################ 机器学习模型 ############################
    @property
    @cached
    def dollar_bar_short_term_features(self) -> dict:
        self.dollar_bar_short_term_fc.load(self.dollar_bar_short_term)
        feature_names = sorted(
            [
                i.replace(f"{DOLLAR_BAR_SHORT_TERM}_", "")
                for i in set(
                    SIDE_DOLLAR_BAR_SHORT_FEATURES + META_DOLLAR_BAR_SHORT_FEATURES
                )
            ]
        )
        features = self.dollar_bar_short_term_fc.get(feature_names)
        return {f"{DOLLAR_BAR_SHORT_TERM}_{k}": v for k, v in features.items()}

    @property
    @cached
    def dollar_bar_mid_term_features(self) -> dict:
        self.dollar_bar_mid_term_fc.load(self.dollar_bar_mid_term)
        feature_names = sorted(
            [
                i.replace(f"{DOLLAR_BAR_MID_TERM}_", "")
                for i in set(
                    SIDE_DOLLAR_BAR_MID_FEATURES + META_DOLLAR_BAR_MID_FEATURES
                )
            ]
        )
        features = self.dollar_bar_mid_term_fc.get(feature_names)
        return {f"{DOLLAR_BAR_MID_TERM}_{k}": v for k, v in features.items()}

    @property
    @cached
    def dollar_bar_long_term_features(self) -> dict:
        self.dollar_bar_long_term_fc.load(self.dollar_bar_long_term)
        feature_names = sorted(
            [
                i.replace(f"{DOLLAR_BAR_LONG_TERM}_", "")
                for i in set(
                    SIDE_DOLLAR_BAR_LONG_FEATURES + META_DOLLAR_BAR_LONG_FEATURES
                )
            ]
        )
        features = self.dollar_bar_long_term_fc.get(feature_names)
        return {f"{DOLLAR_BAR_LONG_TERM}_{k}": v for k, v in features.items()}

    @property
    def side_model_features(self) -> pd.DataFrame:
        all_features = {
            **self.dollar_bar_short_term_features,
            **self.dollar_bar_mid_term_features,
            **self.dollar_bar_long_term_features,
        }
        side_features = {k: all_features[k] for k in SIDE_ALL}
        return pd.DataFrame(side_features)

    @property
    @cached
    def side_model_pred(self) -> float:
        return self.side_model.predict(self.side_model_features)[-1]

    @property
    def meta_model_features(self) -> pd.DataFrame:
        all_features = {
            **self.dollar_bar_short_term_features,
            **self.dollar_bar_mid_term_features,
            **self.dollar_bar_long_term_features,
            "side_model_res": self.side_model_pred,
        }
        meta_features = {k: all_features[k] for k in META_ALL}
        return pd.DataFrame(meta_features)

    @property
    @cached
    def meta_model_pred(self):
        return self.meta_model.predict(self.meta_model_features)[-1]

    ############################ jesse 交易逻辑 ############################
    # def z_score_filter(self) -> bool:
    #     if not self.should_trade_main_bar:
    #         return False
    #     dollar_bar_close = helpers.get_candle_source(self.dollar_bar_mid_term, "close")
    #     res = z_score_filter_np(
    #         dollar_bar_close, mean_window=5, std_window=5, z_score=1
    #     )
    #     return res > 0.5

    # def filters(self) -> list:
    #     return [
    #         self.z_score_filter,
    #     ]

    def should_long(self) -> bool:
        if not self.should_trade_main_bar:
            return False
        if self.meta_model_pred > META_MODEL_THRESHOLD:
            if self.side_model_pred > SIDE_MODEL_THRESHOLD:
                return True
            else:
                return False
        else:
            return False

    def should_short(self) -> bool:
        if not self.should_trade_main_bar:
            return False
        if self.meta_model_pred > META_MODEL_THRESHOLD:
            if self.side_model_pred <= SIDE_MODEL_THRESHOLD:
                return True
            else:
                return False
        else:
            return False

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        return False

    def go_long(self):
        self.cancel_active_orders()
        # 打开多仓
        entry_price = self.price - 0.1
        qty = utils.size_to_qty(
            self.available_margin, entry_price, fee_rate=self.fee_rate
        )
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * (1 - STOP_LOSS_RATIO)

    def go_short(self):
        self.cancel_active_orders()
        # 打开空仓
        entry_price = self.price + 0.1
        qty = utils.size_to_qty(
            self.available_margin, entry_price, fee_rate=self.fee_rate
        )
        self.sell = qty, entry_price
        self.stop_loss = qty, entry_price * (1 + STOP_LOSS_RATIO)

    def update_position(self):
        if not self.should_trade_main_bar:
            return
        # 更新仓位
        if self.is_long:
            if not self.should_long():
                self.liquidate()
        if self.is_short:
            if not self.should_short():
                self.liquidate()

    # def before_terminate(self):
    #     # 在策略结束时执行，可以用于关闭仓位
    #     self.liquidate()
