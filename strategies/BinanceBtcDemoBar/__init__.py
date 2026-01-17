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
from src.features.dimensionality_reduction import ARDVAE
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from .models.config import (
    model_name_to_params,
    LGBMContainer,
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

# 模型目录设置
MODEL_DIR = Path(__file__).parent / "models"
# 推理要求足够历史长度以避免 NaN
MIN_FUSION_BARS = 512

# 模型设置
MODELS = [
    "c_L6_N1",
    "r_L5_N2",
]


def _load_model_features(model_dir: Path, model_name: str) -> list[str]:
    features_path = model_dir / model_name / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.json: {features_path}")

    with open(features_path, "r") as f:
        features = json.load(f)

    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError(f"Invalid features.json format: {features_path}")

    if not features:
        raise ValueError(f"Empty features.json: {features_path}")

    seen = set()
    duplicates = []
    for feature in features:
        if feature in seen:
            duplicates.append(feature)
        else:
            seen.add(feature)

    if duplicates:
        raise ValueError(
            f"Duplicate features in {features_path}: {sorted(set(duplicates))}"
        )

    return features


def _collect_model_features(
    model_dir: Path, models: list[str]
) -> tuple[dict[str, list[str]], list[str]]:
    model_features = {}
    global_features_set: set[str] = set()

    for model_name in models:
        features = _load_model_features(model_dir, model_name)
        model_features[model_name] = features
        global_features_set.update(features)

    global_features = sorted(global_features_set)
    if not global_features:
        raise ValueError("No features found from model configs")

    return model_features, global_features


def _align_lgbm_feature_columns(
    df_features: pd.DataFrame, expected_columns: list[str]
) -> pd.DataFrame:
    if df_features.shape[1] != len(expected_columns):
        raise ValueError(
            "Feature count mismatch between reducer output and model. "
            f"reducer={df_features.shape[1]}, model={len(expected_columns)}"
        )

    if list(df_features.columns) == expected_columns:
        return df_features

    if set(df_features.columns) == set(expected_columns):
        return df_features[expected_columns]

    missing = set(expected_columns) - set(df_features.columns)
    extra = set(df_features.columns) - set(expected_columns)
    raise ValueError(
        "Feature name mismatch between reducer output and model. "
        f"missing={sorted(missing)}, extra={sorted(extra)}"
    )


class BinanceBtcDemoBar(Strategy):
    def __init__(self):
        super().__init__()
        self.bar_container = DemoBar(max_bars=3500)

        self.model_features, self.global_features = _collect_model_features(
            MODEL_DIR, MODELS
        )
        self.feature_calculator = SimpleFeatureCalculator(verbose=False)
        self.reducers = {
            model_name: ARDVAE.load(str(MODEL_DIR / model_name), model_name)
            for model_name in MODELS
        }

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
    def _compute_global_features(self, fusion_bars: np.ndarray) -> pd.DataFrame:
        if len(fusion_bars) < 2:
            raise ValueError("Fusion bars not ready for feature calculation")

        self.feature_calculator.load(fusion_bars, sequential=True)
        features_dict = self.feature_calculator.get(self.global_features)
        features_df = pd.DataFrame(features_dict)

        if len(features_df) != len(fusion_bars):
            raise ValueError(
                "Feature length mismatch. "
                f"features={len(features_df)}, bars={len(fusion_bars)}"
            )

        latest_row = features_df.iloc[-1]
        if latest_row.isna().any():
            nan_cols = latest_row.index[latest_row.isna()].tolist()
            raise ValueError(f"Latest features contain NaN: {nan_cols}")

        return features_df

    def _ensure_features_ready(self) -> bool:
        fusion_bars = self.fusion_bar
        if len(fusion_bars) < MIN_FUSION_BARS:
            return False

        try:
            _ = self.df_raw_features
        except ValueError:
            return False

        return True

    @property
    @cached
    def df_raw_features(self) -> pd.DataFrame:
        """SimpleFeatureCalculator 输出的原始特征"""
        fusion_bars = self.fusion_bar
        if len(fusion_bars) < MIN_FUSION_BARS:
            raise ValueError("Fusion bars not ready for feature calculation")
        return self._compute_global_features(fusion_bars)

    @property
    @cached
    def votes(self) -> list[int]:
        preds = []
        raw_features = self.df_raw_features
        latest_features = raw_features.iloc[[-1]]
        for m in MODELS:
            mc: LGBMContainer = getattr(self, f"model_{m}")
            model_features = self.model_features[m]
            model_features_df = latest_features[model_features]
            # ARDVAE 降维
            reduced_features = self.reducers[m].transform(model_features_df)
            # 对齐列名
            expected_columns = mc.model.feature_name()
            reduced_features = _align_lgbm_feature_columns(
                reduced_features, expected_columns
            )
            preds.append(mc.final_predict(reduced_features))
        return preds

    @property
    def model_shows_long(self) -> bool:
        return all([v == 1 for v in self.votes])

    @property
    def model_shows_short(self) -> bool:
        return all([v == -1 for v in self.votes])

    def should_long(self) -> bool:
        if not self.should_trade_bar or not self._ensure_features_ready():
            return False
        return self.model_shows_long

    def should_short(self) -> bool:
        if not self.should_trade_bar or not self._ensure_features_ready():
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
        if not self.should_trade_bar or not self._ensure_features_ready():
            return
        # 更新仓位
        if self.is_long:
            if not self.model_shows_long:
                self.liquidate()
        if self.is_short:
            if not self.model_shows_short:
                self.liquidate()
