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
from src.features.pipeline import FeaturePipeline
from .models.config import model_name_to_params, LGBMContainer

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

# FeaturePipeline 设置
MODEL_DIR = Path(__file__).parent / "models"
PIPELINE_NAME = "global_pipeline"
# FeaturePipeline 推理要求足够历史长度以避免 NaN
MIN_FUSION_BARS = 512

# 模型与特征设置
MODELS = [
    "c_L6_N1",
    "r_L5_N2",
]


class BinanceBtcDemoBarV2(Strategy):
    def __init__(self):
        super().__init__()
        self.bar_container = DemoBar(max_bars=3500, threshold=1.399)
        self.feature_pipeline = FeaturePipeline.load(str(MODEL_DIR), PIPELINE_NAME)
        self.feature_pipeline.verbose = False
        self._pipeline_warmed = False

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
    def _ensure_pipeline_ready(self) -> bool:
        if self._pipeline_warmed:
            return True

        fusion_bars = self.fusion_bar
        if len(fusion_bars) < MIN_FUSION_BARS or len(fusion_bars) < 2:
            return False

        self.feature_pipeline.warmup_ssm(fusion_bars[:-1])
        self._pipeline_warmed = True
        return True

    @property
    @cached
    def df_all_features(self) -> pd.DataFrame:
        assert self._ensure_pipeline_ready(), "Feature pipeline not warmed up."
        return self.feature_pipeline.inference(self.fusion_bar)

    @property
    @cached
    def votes(self) -> list[int]:
        preds = []
        df_features = self.df_all_features
        for m in MODELS:
            mc: LGBMContainer = getattr(self, f"model_{m}")
            expected_columns = mc.model.feature_name()
            if df_features.shape[1] != len(expected_columns):
                raise ValueError(
                    f"Feature count mismatch for {mc.MODEL_NAME}: "
                    f"pipeline={df_features.shape[1]}, model={len(expected_columns)}"
                )
            if list(df_features.columns) != expected_columns:
                df_features_aligned = df_features.set_axis(
                    expected_columns, axis=1, copy=False
                )
            else:
                df_features_aligned = df_features
            preds.append(mc.final_predict(df_features_aligned))
        return preds

    @property
    def model_shows_long(self) -> bool:
        return all([v == 1 for v in self.votes])

    @property
    def model_shows_short(self) -> bool:
        return all([v == -1 for v in self.votes])

    def should_long(self) -> bool:
        if not self.should_trade_bar or not self._ensure_pipeline_ready():
            return False
        return self.model_shows_long

    def should_short(self) -> bool:
        if not self.should_trade_bar or not self._ensure_pipeline_ready():
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
        if not self.should_trade_bar or not self._ensure_pipeline_ready():
            return
        # 更新仓位
        if self.is_long:
            if not self.model_shows_long:
                self.liquidate()
        if self.is_short:
            if not self.model_shows_short:
                self.liquidate()
