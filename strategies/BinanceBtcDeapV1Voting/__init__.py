import os

import numpy as np
import pandas as pd
from jesse import utils
from jesse.strategies import Strategy, cached
from joblib._parallel_backends import LokyBackend  # 内部 API
from joblib.externals.loky import get_reusable_executor
from joblib.parallel import register_parallel_backend

from src.bars.fusion.deap_v1 import DeapBarV1
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from .models.config import (
    LGBMContainer,
    FEAT_FRACDIFF,
    DeepSSMContainer,
    LGSSMContainer,
    ALL_RAW_FEAT,
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


class BinanceBtcDeapV1Voting(Strategy):
    def __init__(self):
        super().__init__()
        self.bar_container = DeapBarV1(max_bars=2500)
        self.fc = SimpleFeatureCalculator()

        self.deep_ssm_model = DeepSSMContainer()
        self.lg_ssm_model = LGSSMContainer()

        self._init_models()

    def _init_models(self):
        self.model_c_L4_N1 = LGBMContainer(
            "c", 4, 1, is_livetrading=self.is_livetrading
        )
        self.model_c_L5_N1 = LGBMContainer(
            "c", 5, 1, is_livetrading=self.is_livetrading
        )
        self.model_c_L6_N1 = LGBMContainer(
            "c", 6, 1, is_livetrading=self.is_livetrading
        )

        self.model_c_L4_N2 = LGBMContainer(
            "c", 4, 2, is_livetrading=self.is_livetrading
        )
        self.model_c_L5_N2 = LGBMContainer(
            "c", 5, 2, is_livetrading=self.is_livetrading
        )
        self.model_c_L6_N2 = LGBMContainer(
            "c", 6, 2, is_livetrading=self.is_livetrading
        )

        self.model_c_L4_N3 = LGBMContainer(
            "c", 4, 3, is_livetrading=self.is_livetrading
        )
        self.model_c_L5_N3 = LGBMContainer(
            "c", 5, 3, is_livetrading=self.is_livetrading
        )
        self.model_c_L6_N3 = LGBMContainer(
            "c", 6, 3, is_livetrading=self.is_livetrading
        )

    @property
    def cleaned_candles(self):
        candles = self.get_candles("Binance Perpetual Futures", "BTC-USDT", "1m")
        candles = candles[candles[:, 5] > 0]
        return candles

    @property
    def loss_ratio_with_leverage(self):
        return STOP_LOSS_RATIO_NO_LEVERAGE / self.leverage

    ############################### bar 预处理 ##############################
    def before(self):
        self.bar_container.update_with_candles(self.cleaned_candles)

    @property
    def should_trade_bar(self) -> bool:
        return self.bar_container.is_latest_bar_complete

    @property
    def fusion_bar(self) -> np.ndarray:
        return self.bar_container.get_fusion_bars()

    ############################ 机器学习模型 ############################
    @property
    @cached
    def df_all_features(self) -> pd.DataFrame:
        self.fc.load(self.fusion_bar, sequential=False)
        df_feats = pd.DataFrame.from_dict(self.fc.get(ALL_RAW_FEAT))
        df_feat_fracdiff = df_feats[FEAT_FRACDIFF]
        df_feat_deep_ssm = self.deep_ssm_model.inference(df_feat_fracdiff)
        df_feat_lg_ssm = self.lg_ssm_model.inference(df_feat_fracdiff)
        df_final = pd.concat([df_feat_deep_ssm, df_feat_lg_ssm, df_feats], axis=1)
        return df_final

    @property
    @cached
    def model_c_N1_vote(self):
        preds = [
            self.model_c_L4_N1.predict_proba(self.df_all_features),
            self.model_c_L5_N1.predict_proba(self.df_all_features),
            self.model_c_L6_N1.predict_proba(self.df_all_features),
        ]
        preds = [0 if np.isnan(i) else (1 if i > 0.5 else -1) for i in preds]
        return sum(preds)

    @property
    @cached
    def model_c_N2_vote(self):
        preds = [
            self.model_c_L4_N2.predict_proba(self.df_all_features),
            self.model_c_L5_N2.predict_proba(self.df_all_features),
            self.model_c_L6_N2.predict_proba(self.df_all_features),
        ]
        preds = [0 if np.isnan(i) else (1 if i > 0.5 else -1) for i in preds]
        return sum(preds)

    @property
    @cached
    def model_c_N3_vote(self):
        preds = [
            self.model_c_L4_N3.predict_proba(self.df_all_features),
            self.model_c_L5_N3.predict_proba(self.df_all_features),
            self.model_c_L6_N3.predict_proba(self.df_all_features),
        ]
        preds = [0 if np.isnan(i) else (1 if i > 0.5 else -1) for i in preds]
        return sum(preds)

    @property
    def model_shows_long(self) -> bool:
        vote = []
        vote.append(self.model_c_N1_vote == 3)
        vote.append(self.model_c_N2_vote == 3)
        vote.append(self.model_c_N3_vote == 3)
        return all(vote)

    @property
    def model_shows_short(self) -> bool:
        vote = []
        vote.append(self.model_c_N1_vote == -3)
        vote.append(self.model_c_N2_vote == -3)
        vote.append(self.model_c_N3_vote == -3)
        return all(vote)

    def should_long(self) -> bool:
        if not self.should_trade_bar:
            return False
        return self.model_shows_long

    def should_short(self) -> bool:
        if not self.should_trade_bar:
            return False
        return self.model_shows_short

    def should_cancel_entry(self) -> bool:
        # Only for limit orders，当提交的限价单没有成交时，是否在下一个candle取消
        if self.should_long() or self.should_short():
            return True
        return False

    def go_long(self):
        entry_price = self.price
        qty = utils.size_to_qty(
            self.leveraged_available_margin * 0.95, entry_price, fee_rate=self.fee_rate
        )
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * (1 - self.loss_ratio_with_leverage)

    def go_short(self):
        entry_price = self.price
        qty = utils.size_to_qty(
            self.leveraged_available_margin * 0.95, entry_price, fee_rate=self.fee_rate
        )
        self.sell = qty, entry_price
        self.stop_loss = qty, entry_price * (1 + self.loss_ratio_with_leverage)

    def update_position(self):
        if not self.should_trade_bar:
            return
        # 更新仓位
        if self.is_long:
            if not self.model_shows_long:
                self.liquidate()
        if self.is_short:
            if not self.model_shows_short:
                self.liquidate()
