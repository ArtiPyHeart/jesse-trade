import os

import numpy as np
import pandas as pd
from jesse import utils
from jesse.strategies import Strategy, cached
from joblib._parallel_backends import LokyBackend  # 内部 API
from joblib.externals.loky import get_reusable_executor
from joblib.parallel import register_parallel_backend

from src.features.all_features import FeatureCalculator
from src.bars.fusion.v0 import FusionBarContainerV0

from .config import (
    META_ALL,
    META_FEATURES,
    SIDE,
    get_meta_model,
    get_side_model,
)

# joblib设置
# ① 主线程启动时就建好进程池
executor = get_reusable_executor(
    max_workers=os.cpu_count(), timeout=None, reuse=True
)  # 永不过期
backend = LokyBackend(executor=executor, timeout=None)
# ② 把它注册成全局 backend
register_parallel_backend("loky_reuse", lambda **kw: backend, make_default=True)

META_MODEL_THRESHOLD = 0.5
SIDE_MODEL_THRESHOLD = 0.5
STOP_LOSS_RATIO_NO_LEVERAGE = 0.05
ORDER_TIMEOUT = 600 * 1000


class BinanceBtcEntropyBarV1(Strategy):
    def __init__(self):
        super().__init__()
        self.meta_model = get_meta_model(self.is_livetrading)
        self.side_model = get_side_model(self.is_livetrading)

        self.main_bar_container = FusionBarContainerV0(max_bars=10000)

        self.fusion_bar_fc = FeatureCalculator()

    @property
    def cleaned_candles(self):
        candles = self.get_candles("Binance Perpetual Futures", "BTC-USDT", "1m")
        candles = candles[candles[:, 5] > 0]
        return candles

    @property
    def loss_ratio_with_leverage(self):
        return STOP_LOSS_RATIO_NO_LEVERAGE / self.leverage

    # def cancel_active_orders(self, with_stoploss=False):
    #     # 检查超时的活跃订单，如果订单超时依然没有成交，则取消订单
    #     if with_stoploss:
    #         alive_orders = [o for o in self.orders if o.is_cancellable]
    #     else:
    #         alive_orders = [
    #             o for o in self.orders if not o.is_stop_loss and o.is_cancellable
    #         ]
    #
    #     for order in alive_orders:
    #         if helpers.now_to_timestamp() - order.created_at > ORDER_TIMEOUT:
    #             order.cancel()

    ############################### dollar bar 预处理 ##############################
    def before(self):
        self.main_bar_container.update_with_candles(self.cleaned_candles)
        # 检查超时的活跃订单，如果订单超时依然没有成交，则取消订单
        # self.cancel_active_orders()

    @property
    def should_trade_main_bar(self) -> bool:
        return self.main_bar_container.is_latest_bar_complete

    @property
    def fusion_bar(self) -> np.ndarray:
        return self.main_bar_container.get_fusion_bars()

    ############################ 机器学习模型 ############################

    @property
    @cached
    def fusion_bar_features(self) -> dict:
        self.fusion_bar_fc.load(self.fusion_bar)
        feature_names = SIDE + META_FEATURES
        feature_names = sorted(list(set(feature_names)))
        features = self.fusion_bar_fc.get(feature_names)
        return features

    @property
    def side_model_features(self) -> pd.DataFrame:
        side_features = {k: self.fusion_bar_features[k] for k in SIDE}
        return pd.DataFrame(side_features)

    @property
    @cached
    def side_model_pred(self) -> float:
        return self.side_model.predict(self.side_model_features)[-1]

    @property
    def meta_model_features(self) -> pd.DataFrame:
        all_features = {
            **self.fusion_bar_features,
            "model": self.side_model_pred,
        }
        meta_features = {k: all_features[k] for k in META_ALL}
        return pd.DataFrame(meta_features)

    @property
    @cached
    def meta_model_pred(self):
        return self.meta_model.predict(self.meta_model_features)[-1]

    @property
    def model_shows_long(self):
        meta_pred = self.meta_model_pred > META_MODEL_THRESHOLD
        side_pred = self.side_model_pred > SIDE_MODEL_THRESHOLD
        return meta_pred & side_pred

    @property
    def model_shows_short(self):
        meta_pred = self.meta_model_pred > META_MODEL_THRESHOLD
        side_pred = self.side_model_pred < SIDE_MODEL_THRESHOLD
        return meta_pred & side_pred

    def should_long(self) -> bool:
        if not self.should_trade_main_bar:
            return False
        return self.model_shows_long

    def should_short(self) -> bool:
        if not self.should_trade_main_bar:
            return False
        return self.model_shows_short

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        if self.should_long() or self.should_short():
            return True
        return False

    def go_long(self):
        # self.cancel_active_orders(with_stoploss=True)
        # 打开多仓
        # entry_price = self.price - 0.1
        entry_price = self.price
        qty = utils.size_to_qty(
            self.leveraged_available_margin * 0.95, entry_price, fee_rate=self.fee_rate
        )
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * (1 - self.loss_ratio_with_leverage)

    def go_short(self):
        # self.cancel_active_orders(with_stoploss=True)
        # 打开空仓
        # entry_price = self.price + 0.1
        entry_price = self.price
        qty = utils.size_to_qty(
            self.leveraged_available_margin * 0.95, entry_price, fee_rate=self.fee_rate
        )
        self.sell = qty, entry_price
        self.stop_loss = qty, entry_price * (1 + self.loss_ratio_with_leverage)

    def update_position(self):
        if not self.should_trade_main_bar:
            return
        # 更新仓位
        if self.is_long:
            if not self.model_shows_long:
                self.liquidate()
        if self.is_short:
            if not self.model_shows_short:
                self.liquidate()

    # def before_terminate(self):
    #     # 在策略结束时执行，可以用于关闭仓位
    #     self.liquidate()
