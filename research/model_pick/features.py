import gc

import numpy as np
import pandas as pd

from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.simple_feature_calculator.buildin.feature_names import (
    BUILDIN_FEATURES,
)

WINDOW = 20
BASIC = ["bar_open_dt", "bar_high_dt", "bar_low_dt", "bar_close_dt"]

# 关键波动率和动量指标（用于极值特征）
KEY_VOLATILITY_INDICATORS = ["natr", "bekker_parkinson_vol", "corwin_schultz_estimator"]
KEY_MOMENTUM_INDICATORS = ["williams_r", "fisher", "mod_rsi", "adaptive_rsi"]
KEY_INDICATORS = BASIC + KEY_VOLATILITY_INDICATORS + KEY_MOMENTUM_INDICATORS

# ============================================================================
# 基础OHLC的高级变换特征
# ============================================================================
basic_hurst_feats = [f"{i}_hurst{WINDOW}" for i in BASIC]
basic_curv_feats = [f"{i}_curv{WINDOW}" for i in BASIC]
basic_phent_feats = [f"{i}_phent{WINDOW}" for i in BASIC]

# ============================================================================
# 所有BUILDIN_FEATURES的统计特征（一阶矩、二阶矩、高阶矩）
# ============================================================================
# 一阶矩：中心趋势
mean_feats = [f"{i}_mean{WINDOW}" for i in BUILDIN_FEATURES]
median_feats = [f"{i}_median{WINDOW}" for i in BUILDIN_FEATURES]

# 二阶矩：离散程度
std_feats = [f"{i}_std{WINDOW}" for i in BUILDIN_FEATURES]

# 三阶矩/四阶矩：分布形态
skew_feats = [f"{i}_skew{WINDOW}" for i in BUILDIN_FEATURES]
kurt_feats = [f"{i}_kurt{WINDOW}" for i in BUILDIN_FEATURES]

# ============================================================================
# 极值特征（支撑阻力、突破检测）
# ============================================================================
max_feats = [f"{i}_max{WINDOW}" for i in KEY_INDICATORS]
min_feats = [f"{i}_min{WINDOW}" for i in KEY_INDICATORS]

# ============================================================================
# 归一化特征（相对位置、超买超卖）
# ============================================================================
# 归一化到[0,1]：价格在窗口的相对位置
norm_feats = [f"{i}_norm{WINDOW}" for i in KEY_INDICATORS]

# Z-score：超买超卖信号
zscore_feats = [f"{i}_zscore{WINDOW}" for i in KEY_INDICATORS]

# ============================================================================
# 高级拓扑/分形特征
# ============================================================================
hurst_feats = [f"{i}_hurst{WINDOW}" for i in BUILDIN_FEATURES]
curv_feats = [f"{i}_curv{WINDOW}" for i in BUILDIN_FEATURES]
phent_feats = [f"{i}_phent{WINDOW}" for i in BUILDIN_FEATURES]

# ============================================================================
# 差分特征（动量）
# ============================================================================
dt_feats = [f"{i}_dt" for i in BUILDIN_FEATURES]
ddt_feats = [f"{i}_ddt" for i in BUILDIN_FEATURES]

# ============================================================================
# 组合所有非滞后特征
# ============================================================================
feats = (
    BUILDIN_FEATURES
    # 基础OHLC高级特征
    + basic_hurst_feats
    + basic_curv_feats
    + basic_phent_feats
    # 统计特征
    + mean_feats
    + median_feats
    + std_feats
    + skew_feats  # 新增
    + kurt_feats  # 新增
    # 极值特征
    + max_feats  # 新增
    + min_feats  # 新增
    # 归一化特征
    + norm_feats  # 新增
    + zscore_feats  # 新增
    # 高级特征
    + hurst_feats
    + curv_feats
    # 差分特征
    + dt_feats
    + ddt_feats
)

# ============================================================================
# 滞后特征（时序信息）
# ============================================================================
lag_feats = [f"{i}_lag{l}" for i in feats for l in range(1, 4)]

# ============================================================================
# 完整特征集（包含phent特征和滞后特征）
# ============================================================================
ALL_FEATS = feats + phent_feats + lag_feats


class FeatureLoader:
    def __init__(self, candles: np.ndarray):
        self.feature_calculator_seq = SimpleFeatureCalculator(verbose=True)
        self.feature_calculator_seq.load(candles, sequential=True)

        self._features = {}
        self._df_features = None  # 缓存 DataFrame，避免重复创建
        self._candles_index = candles[:, 0].astype(int)

    @property
    def features(self):
        if not self._features:
            self._features = self.feature_calculator_seq.get(ALL_FEATS)
        return self._features

    def _ensure_df_features(self) -> pd.DataFrame:
        """懒加载 DataFrame，避免重复从字典创建"""
        if self._df_features is None:
            # 使用 copy=False 避免额外内存分配
            self._df_features = pd.DataFrame(self.features, copy=False)
            self._df_features.index = self._candles_index
        return self._df_features

    def get_feature_label_bundle(
        self, label: np.ndarray, pred_next: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        # 使用缓存的 DataFrame
        df = self._ensure_df_features()

        len_gap = len(df) - len(label) - pred_next
        end_idx = len(df) - pred_next

        # 用 numpy 快速计算 NA 数量，避免创建临时布尔 DataFrame
        # 只计算需要切片范围内的 NA
        na_counts = np.isnan(df.values[len_gap:end_idx]).sum(axis=0)
        max_na_len = int(na_counts.max()) if na_counts.size > 0 else 0

        # 单次切片，减少内存分配
        start_idx = len_gap + max_na_len
        df_result = df.iloc[start_idx:end_idx].copy()
        label_result = label[max_na_len:]

        assert len(df_result) == len(
            label_result
        ), f"Length mismatch: df={len(df_result)}, label={len(label_result)}"

        return df_result, label_result

    def clear_features(self):
        """释放特征内存，用于任务间清理"""
        self._features = {}
        self._df_features = None
        if hasattr(self.feature_calculator_seq, "clear_cache"):
            self.feature_calculator_seq.clear_cache()
        gc.collect()
