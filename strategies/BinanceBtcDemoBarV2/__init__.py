import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from jesse import utils
from jesse.strategies import Strategy, cached
from joblib._parallel_backends import LokyBackend  # 内部 API
from joblib.externals.loky import get_reusable_executor
from joblib.parallel import register_parallel_backend

from src.bars.fusion.demo import DemoBar
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from .models.config import (
    model_name_to_params,
    LGBMContainer,
    SSMContainer,
)

# joblib设置
# ① 主线程启动时就建好进程池
executor = get_reusable_executor(
    max_workers=os.cpu_count(), timeout=None, reuse=True
)  # 永不过期
backend = LokyBackend(executor=executor, idle_worker_timeout=None)
# ② 把它注册成全局 backend
register_parallel_backend("loky_reuse", lambda **kw: backend, make_default=True)

STOP_LOSS_RATIO_NO_LEVERAGE = 0.05
POSITION_SIZE_RATIO = 0.95

# 模型与特征设置
MODELS = [
    "c_L6_N1",
    "r_L5_N2",
]

path_features = Path(__file__).parent / "models" / "feature_info.json"
with open(path_features) as f:
    feature_info: dict[str, list[str]] = json.load(f)
FEAT_FRACDIFF: list[str] = feature_info["fracdiff"]
ALL_RAW_FEAT = []
ALL_RAW_FEAT.extend(FEAT_FRACDIFF)
for m in MODELS:
    ALL_RAW_FEAT.extend(feature_info[m])
ALL_RAW_FEAT = set(ALL_RAW_FEAT)
ALL_RAW_FEAT = sorted(
    [
        i
        for i in ALL_RAW_FEAT
        if not i.startswith("deep_ssm") and not i.startswith("lg_ssm")
    ]
)


class BinanceBtcDemoBarV2(Strategy):
    def __init__(self):
        super().__init__()
        self.bar_container = DemoBar(max_bars=3500, threshold=1.399)
        self.fc = SimpleFeatureCalculator()

        self.deep_ssm_model = SSMContainer("deep_ssm")
        self.lg_ssm_model = SSMContainer("lg_ssm")

        self._init_models()

    def _init_models(self):
        for m in MODELS:
            model_container = LGBMContainer(*model_name_to_params(m))
            model_container.is_livetrading = self.is_livetrading
            setattr(self, f"model_{m}", model_container)

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
    def votes(self) -> list[int]:
        preds = []
        for m in MODELS:
            mc: LGBMContainer = getattr(self, f"model_{m}")
            preds.append(
                mc.final_predict(self.df_all_features[feature_info[mc.MODEL_NAME]])
            )
        return preds

    @property
    def model_shows_long(self) -> bool:
        return all([v == 1 for v in self.votes])

    @property
    def model_shows_short(self) -> bool:
        return all([v == -1 for v in self.votes])

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
            self.leveraged_available_margin * POSITION_SIZE_RATIO,
            entry_price,
            fee_rate=self.fee_rate,
        )
        self.buy = qty, entry_price
        self.stop_loss = qty, entry_price * (1 - self.loss_ratio_with_leverage)

    def go_short(self):
        entry_price = self.price
        qty = utils.size_to_qty(
            self.leveraged_available_margin * POSITION_SIZE_RATIO,
            entry_price,
            fee_rate=self.fee_rate,
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
