import numpy as np
import pandas as pd
from jesse import utils
from jesse.strategies import Strategy, cached

from src.features.all_features import FeatureCalculator
from model.config import (
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
)
from model.config import get_meta_model, get_side_model
from src.data_process.bet_sizing import discretize_position
from bar import (
    DollarBarContainer,
    build_dollar_bar,
)

DISCRETE_THRESHOLD = 0.05


def discrete_position_ratio(old_ratio, new_ratio) -> float:
    discrete_new_ratio = discretize_position(
        new_ratio,
        old_ratio,
        threshold=DISCRETE_THRESHOLD,
    )
    if discrete_new_ratio - old_ratio < DISCRETE_THRESHOLD:
        return old_ratio
    else:
        return discrete_new_ratio


class MLV1Partial(Strategy):

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

    ############################### 仓位管理 ##################################
    @property
    def position_ratio_based_on_meta_model(self):
        return max(0, 2 * self.meta_model_pred - 1)

    @property
    def used_margin(self):
        used = (self.portfolio_value - self.available_margin) / self.leverage
        return used if used > 0 else 0

    @property
    def total_margin(self):
        return self.available_margin + self.used_margin

    @property
    def current_position_ratio(self):
        return self.used_margin / self.total_margin

    @property
    def max_available_qty(self):
        return utils.size_to_qty(
            self.available_margin, self.price, fee_rate=self.fee_rate
        )

    def percentage_to_qty(self, percentage_change):
        target_margin = self.total_margin * abs(percentage_change)
        margin = min(target_margin, self.available_margin)
        qty = utils.size_to_qty(margin, self.price, fee_rate=self.fee_rate)
        return qty

    def get_final_qty(self, current_ratio) -> float:
        target_percentage = discrete_position_ratio(
            current_ratio, self.position_ratio_based_on_meta_model
        )
        if target_percentage == current_ratio:
            return 0
        sign = 1 if target_percentage > current_ratio else -1
        # 如果target_percentage为负数，说明需要减仓
        target_qty = self.percentage_to_qty(target_percentage - current_ratio)
        return sign * target_qty

    ############################ jesse 交易逻辑 ############################
    def should_long(self) -> bool:
        if not self.should_trade_dollar_bar:
            return False
        if self.meta_model_pred > 0.5:
            if self.side_model_pred > 0.5 and self.get_final_qty(0) != 0:
                return True
            else:
                return False
        else:
            return False

    def should_short(self) -> bool:
        if not self.should_trade_dollar_bar:
            return False
        if self.meta_model_pred > 0.5:
            # 预测为多头
            if self.side_model_pred < 0.5 and self.get_final_qty(0) != 0:
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
        qty = self.get_final_qty(0)
        self.buy = qty, entry_price

    def go_short(self):
        # 打开空仓
        entry_price = self.price
        qty = self.get_final_qty(0)
        self.sell = qty, entry_price

    def update_position(self):
        if not self.should_trade_dollar_bar:
            return
        # 更新仓位
        if self.is_long:
            if not self.should_long():
                self.liquidate()
            else:
                # update qty
                new_qty = self.get_final_qty(self.current_position_ratio)
                if new_qty > 0:
                    self.buy = new_qty, self.price
                if new_qty < 0:
                    self.sell = abs(new_qty), self.price
        if self.is_short:
            if not self.should_short():
                self.liquidate()
            else:
                new_qty = self.get_final_qty(self.current_position_ratio)
                if new_qty > 0:
                    self.sell = new_qty, self.price
                if new_qty < 0:
                    self.buy = abs(new_qty), self.price

    # def before_terminate(self):
    #     # 在策略结束时执行，可以用于关闭仓位
    #     self.liquidate()
