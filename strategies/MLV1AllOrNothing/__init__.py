import numpy as np
import pandas as pd
from jesse import helpers, utils
from jesse.strategies import Strategy, cached

from custom_indicators.all_features import FeatureCalculator
from custom_indicators.config import (
    DOLLAR_BAR_THRESHOLD_LONG,
    DOLLAR_BAR_THRESHOLD_MID,
    DOLLAR_BAR_THRESHOLD_SHORT,
    LONG_TERM,
    META_ALL,
    META_LONG,
    META_MID,
    META_SHORT,
    MID_TERM,
    SHORT_TERM,
    SIDE_ALL,
    SIDE_LONG,
    SIDE_MID,
    SIDE_SHORT,
)
from custom_indicators.model import get_meta_model, get_side_model
from custom_indicators.toolbox.dollar_bar import DollarBarContainer, build_dollar_bar
from custom_indicators.toolbox.filters import z_score_filter_np

META_MODEL_THRESHOLD = 0.5
SIDE_MODEL_THRESHOLD = 0.5
STOP_LOSS_RATIO = 0.05


class MLV1AllOrNothing(Strategy):

    def __init__(self):
        super().__init__()
        self.meta_model = get_meta_model(self.is_livetrading)
        self.side_model = get_side_model(self.is_livetrading)

        self.dollar_bar_container = DollarBarContainer(
            DOLLAR_BAR_THRESHOLD_MID,
            max_bars=2000,
        )

        self.dollar_bar_short_term_fc = FeatureCalculator()
        self.dollar_bar_mid_term_fc = FeatureCalculator()
        self.dollar_bar_long_term_fc = FeatureCalculator()

    ############################### dollar bar 预处理 ##############################

    def before(self):
        self.dollar_bar_container.update_with_candle(
            self.get_candles("Binance Perpetual Futures", "BTC-USDT", "1m")
        )

    @property
    def should_trade_dollar_bar(self) -> bool:
        return self.dollar_bar_container.is_latest_bar_complete()

    @property
    @cached
    def dollar_bar_short_term(self) -> np.ndarray:
        return build_dollar_bar(
            self.get_candles("Binance Perpetual Futures", "BTC-USDT", "1m"),
            DOLLAR_BAR_THRESHOLD_SHORT,
            max_bars=5000,
        )

    @property
    def dollar_bar_mid_term(self) -> np.ndarray:
        return self.dollar_bar_container.get_dollar_bars()

    @property
    @cached
    def dollar_bar_long_term(self) -> np.ndarray:
        return build_dollar_bar(
            self.get_candles("Binance Perpetual Futures", "BTC-USDT", "1m"),
            DOLLAR_BAR_THRESHOLD_LONG,
            max_bars=5000,
        )

    ############################ 机器学习模型 ############################
    @property
    @cached
    def dollar_bar_short_term_features(self) -> dict:
        self.dollar_bar_short_term_fc.load(self.dollar_bar_short_term)
        feature_names = sorted(
            [i.replace(f"{SHORT_TERM}_", "") for i in set(SIDE_SHORT + META_SHORT)]
        )
        features = self.dollar_bar_short_term_fc.get(feature_names)
        return {f"{SHORT_TERM}_{k}": v for k, v in features.items()}

    @property
    @cached
    def dollar_bar_mid_term_features(self) -> dict:
        self.dollar_bar_mid_term_fc.load(self.dollar_bar_mid_term)
        feature_names = sorted(
            [i.replace(f"{MID_TERM}_", "") for i in set(SIDE_MID + META_MID)]
        )
        features = self.dollar_bar_mid_term_fc.get(feature_names)
        return {f"{MID_TERM}_{k}": v for k, v in features.items()}

    @property
    @cached
    def dollar_bar_long_term_features(self) -> dict:
        self.dollar_bar_long_term_fc.load(self.dollar_bar_long_term)
        feature_names = sorted(
            [i.replace(f"{LONG_TERM}_", "") for i in set(SIDE_LONG + META_LONG)]
        )
        features = self.dollar_bar_long_term_fc.get(feature_names)
        return {f"{LONG_TERM}_{k}": v for k, v in features.items()}

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
    #     if not self.should_trade_dollar_bar:
    #         return False
    #     dollar_bar_close = helpers.get_candle_source(self.dollar_bar_mid_term, "close")
    #     res = z_score_filter_np(
    #         dollar_bar_close, mean_window=20, std_window=20, z_score=1
    #     )[-1]
    #     return res == 1

    # def filters(self) -> list:
    #     return [
    #         self.z_score_filter,
    #     ]

    def should_long(self) -> bool:
        if not self.should_trade_dollar_bar:
            return False
        if self.meta_model_pred > META_MODEL_THRESHOLD:
            if self.side_model_pred > SIDE_MODEL_THRESHOLD:
                return True
            else:
                return False
        else:
            return False

    def should_short(self) -> bool:
        if not self.should_trade_dollar_bar:
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
        return True

    def go_long(self):
        # 打开多仓
        entry_price = self.price
        qty = utils.size_to_qty(
            self.available_margin, entry_price, fee_rate=self.fee_rate
        )
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * (1 - STOP_LOSS_RATIO)

    def go_short(self):
        # 打开空仓
        entry_price = self.price
        qty = utils.size_to_qty(
            self.available_margin, entry_price, fee_rate=self.fee_rate
        )
        self.sell = qty, entry_price
        self.stop_loss = qty, entry_price * (1 + STOP_LOSS_RATIO)

    def update_position(self):
        if not self.should_trade_dollar_bar:
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
