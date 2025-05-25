import numpy as np
import pandas as pd
from jesse import helpers, utils
from jesse.strategies import Strategy, cached

from custom_indicators.all_features import FeatureCalculator
from custom_indicators.toolbox.bar.entropy_bar_v2 import EntropyBarContainer

from .config import (
    ENTROPY_THRESHOLD,
    META_ALL,
    META_FEATURES,
    SIDE_LONG,
    SIDE_SHORT,
    VOL_REF_WINDOW,
    VOL_T_WINDOW,
    WINDOW,
    get_meta_model,
    get_side_model,
)

META_MODEL_THRESHOLD = 0.5
SIDE_MODEL_THRESHOLD = 0.5
STOP_LOSS_RATIO = 0.05
ORDER_TIMEOUT = 300 * 1000


class BinanceMLV2(Strategy):
    def __init__(self):
        super().__init__()
        self.meta_model = get_meta_model(self.is_livetrading)
        self.side_model_long = get_side_model(self.is_livetrading, "long")
        self.side_model_short = get_side_model(self.is_livetrading, "short")

        self.main_bar_container = EntropyBarContainer(
            window=WINDOW,
            window_vol_t=VOL_T_WINDOW,
            window_vol_ref=VOL_REF_WINDOW,
            entropy_threshold=ENTROPY_THRESHOLD,
        )

        self.entropy_bar_fc = FeatureCalculator()

    @property
    def spot_candles(self):
        candles = self.get_candles("Binance Spot", "BTC-USDT", "1m")
        return candles

    def cancel_active_orders(self):
        # 检查超时的活跃订单，如果订单超时依然没有成交，则取消订单
        alive_orders = [o for o in self.orders if o.is_active or o.is_partially_filled]
        for order in alive_orders:
            if helpers.now_to_timestamp() - order.created_at > ORDER_TIMEOUT:
                order.cancel()

    ############################### dollar bar 预处理 ##############################
    def before(self):
        self.main_bar_container.update_with_candle(self.spot_candles)
        # 检查超时的活跃订单，如果订单超时依然没有成交，则取消订单
        self.cancel_active_orders()

    @property
    def should_trade_main_bar(self) -> bool:
        return self.main_bar_container.is_latest_bar_complete()

    @property
    def entropy_bar(self) -> np.ndarray:
        return self.main_bar_container.get_entropy_bar()

    ############################ 机器学习模型 ############################

    @property
    @cached
    def entropy_bar_features(self) -> dict:
        self.entropy_bar_fc.load(self.entropy_bar)
        feature_names = SIDE_LONG + SIDE_SHORT + META_FEATURES
        feature_names = sorted(list(set(feature_names)))
        features = self.entropy_bar_fc.get(feature_names)
        return features

    @property
    def side_model_long_features(self) -> pd.DataFrame:
        side_long = {k: self.entropy_bar_features[k] for k in SIDE_LONG}
        return pd.DataFrame(side_long)

    @property
    def side_model_short_features(self) -> pd.DataFrame:
        side_short = {k: self.entropy_bar_features[k] for k in SIDE_SHORT}
        return pd.DataFrame(side_short)

    @property
    @cached
    def side_model_long_pred(self) -> float:
        return self.side_model_long.predict(self.side_model_long_features)[-1]

    @property
    @cached
    def side_model_short_pred(self) -> float:
        return self.side_model_short.predict(self.side_model_short_features)[-1]

    @property
    def meta_model_features(self) -> pd.DataFrame:
        all_features = {
            **self.entropy_bar_features,
            "model_long": self.side_model_long_pred,
            "model_short": self.side_model_short_pred,
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
            if (
                self.side_model_long_pred
                > SIDE_MODEL_THRESHOLD
                > self.side_model_short_pred
            ):
                return True
            else:
                return False
        else:
            return False

    def should_short(self) -> bool:
        if not self.should_trade_main_bar:
            return False
        if self.meta_model_pred > META_MODEL_THRESHOLD:
            if (
                self.side_model_short_pred
                > SIDE_MODEL_THRESHOLD
                > self.side_model_long_pred
            ):
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
