"""
Flow Feature Select - 批量特征筛选流水线

遍历 LOG_RETURN_LAG x PRED_NEXT 组合，使用 GMMLabeler 生成标签，
通过 SimpleFeatureCalculator 计算特征，执行全量特征筛选，结果记录到 CSV。

Usage:
    python flow_feature_select.py
"""

import gc
import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from jesse import helpers, research
from research.labeler.gmm_labeler import GMMLabeler
from research.utils import align_features_labels
from src.bars.fusion.demo import DemoBar
from src.features.feature_selection import GrootCVConfig, GrootCVSelector
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.simple_feature_calculator.buildin.feature_names import (
    BUILDIN_FEATURES,
)
from src.utils.env_dates import get_env_date, load_env_values

# ============================================================================
# 配置参数
# ============================================================================
# 数据范围
ENV_VALUES = load_env_values(Path(".env"))
TRAIN_START = get_env_date("TRAIN_START_DATE", ENV_VALUES)
TRAIN_END = get_env_date("TRAIN_END_DATE", ENV_VALUES)

# 搜索参数
LOG_RETURN_LAGS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # GMMLabeler 的 lag_n
PRED_NEXT_STEPS = [1, 2, 3]  # 预测时间范围
LABEL_TYPES: list[Literal["hard", "direction", "directional_prob"]] = [
    "hard",
    "direction",
    "directional_prob",
]

# 特征筛选配置
GROOTCV_CUTOFF = 5
GROOTCV_SHAP_BACKEND = "lgb"
GROOTCV_SHAP_BATCH_SIZE = 256
GROOTCV_SHAP_MAX_SAMPLES = None
GROOTCV_FASTSHAP = False

# 输出文件
OUTPUT_FILE = "feature_selection_results.csv"

# 临时特征缓存（memmap）
TEMP_DIR = Path("temp")
FEATURE_STORE_PATH = TEMP_DIR / "feature_store.mmap"
FEATURE_STORE_META_PATH = TEMP_DIR / "feature_store_meta.json"
FEATURE_STORE_DTYPE = np.float32
FEATURE_STORE_CLEAR_CACHE_EVERY = 128
FEATURE_STORE_FORCE_REBUILD = False

# ============================================================================
# 特征列表构建
# ============================================================================
WINDOWS = [20, 40, 60]

# 基础 OHLC dt 特征
BASIC = ["bar_open_dt", "bar_high_dt", "bar_low_dt", "bar_close_dt"]

# 关键波动率和动量指标（用于极值特征）
KEY_VOLATILITY_INDICATORS = ["natr", "bekker_parkinson_vol", "corwin_schultz_estimator"]
KEY_MOMENTUM_INDICATORS = ["williams_r", "fisher", "mod_rsi", "adaptive_rsi"]
KEY_INDICATORS = BASIC + KEY_VOLATILITY_INDICATORS + KEY_MOMENTUM_INDICATORS

# 基础 OHLC 的高级变换特征（多窗口）
basic_hurst_feats = [f"{i}_hurst{w}" for i in BASIC for w in WINDOWS]
basic_curv_feats = [f"{i}_curv{w}" for i in BASIC for w in WINDOWS]
basic_phent_feats = [f"{i}_phent{w}" for i in BASIC for w in WINDOWS]

# 所有 BUILDIN_FEATURES 的统计特征（多窗口）
mean_feats = [f"{i}_mean{w}" for i in BUILDIN_FEATURES for w in WINDOWS]
median_feats = [f"{i}_median{w}" for i in BUILDIN_FEATURES for w in WINDOWS]
std_feats = [f"{i}_std{w}" for i in BUILDIN_FEATURES for w in WINDOWS]
skew_feats = [f"{i}_skew{w}" for i in BUILDIN_FEATURES for w in WINDOWS]
kurt_feats = [f"{i}_kurt{w}" for i in BUILDIN_FEATURES for w in WINDOWS]

# 极值特征（支撑阻力、突破检测）（多窗口）
max_feats = [f"{i}_max{w}" for i in KEY_INDICATORS for w in WINDOWS]
min_feats = [f"{i}_min{w}" for i in KEY_INDICATORS for w in WINDOWS]

# 归一化特征（相对位置、超买超卖）（多窗口）
norm_feats = [f"{i}_norm{w}" for i in KEY_INDICATORS for w in WINDOWS]
zscore_feats = [f"{i}_zscore{w}" for i in KEY_INDICATORS for w in WINDOWS]

# 高级拓扑/分形特征（多窗口）
hurst_feats = [f"{i}_hurst{w}" for i in BUILDIN_FEATURES for w in WINDOWS]
curv_feats = [f"{i}_curv{w}" for i in BUILDIN_FEATURES for w in WINDOWS]
phent_feats = [f"{i}_phent{w}" for i in BUILDIN_FEATURES for w in WINDOWS]

# 差分特征（动量）
dt_feats = [f"{i}_dt" for i in BUILDIN_FEATURES]
ddt_feats = [f"{i}_ddt" for i in BUILDIN_FEATURES]

# 组合所有非滞后特征
_feats = (
    list(BUILDIN_FEATURES)
    # 基础 OHLC 高级特征
    + basic_hurst_feats
    + basic_curv_feats
    + basic_phent_feats
    # 统计特征
    + mean_feats
    + median_feats
    + std_feats
    + skew_feats
    + kurt_feats
    # 极值特征
    + max_feats
    + min_feats
    # 归一化特征
    + norm_feats
    + zscore_feats
    # 高级特征
    + hurst_feats
    + curv_feats
    # 差分特征
    + dt_feats
    + ddt_feats
)

# 滞后特征（时序信息）
lag_feats = [f"{i}_lag{lag}" for i in _feats for lag in range(1, 4)]

# 完整特征集
FEATURE_NAMES = _feats + phent_feats + lag_feats


def _hash_feature_names(feature_names: list[str]) -> str:
    hasher = hashlib.sha256()
    for name in feature_names:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _load_feature_store_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_or_load_feature_store(
    calc: SimpleFeatureCalculator,
    candles: np.ndarray,
    feature_names: list[str],
) -> np.memmap:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    n_rows = len(candles)
    n_cols = len(feature_names)
    feature_hash = _hash_feature_names(feature_names)
    candles_hash = hashlib.sha256(candles.tobytes()).hexdigest()
    dtype_name = np.dtype(FEATURE_STORE_DTYPE).name

    meta = _load_feature_store_meta(FEATURE_STORE_META_PATH)
    if (
        not FEATURE_STORE_FORCE_REBUILD
        and meta is not None
        and meta.get("n_rows") == n_rows
        and meta.get("n_cols") == n_cols
        and meta.get("dtype") == dtype_name
        and meta.get("feature_hash") == feature_hash
        and meta.get("candles_hash") == candles_hash
        and FEATURE_STORE_PATH.exists()
    ):
        return np.memmap(
            FEATURE_STORE_PATH,
            dtype=FEATURE_STORE_DTYPE,
            mode="r",
            shape=(n_rows, n_cols),
        )

    print("\n[2/3] 计算全局特征并写入 memmap...")
    mmap = calc.compute_to_memmap(
        feature_names,
        FEATURE_STORE_PATH,
        dtype=FEATURE_STORE_DTYPE,
        clear_cache_every=FEATURE_STORE_CLEAR_CACHE_EVERY,
    )
    calc.clear_cache()

    meta_out = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtype": dtype_name,
        "feature_hash": feature_hash,
        "candles_hash": candles_hash,
        "start": TRAIN_START,
        "end": TRAIN_END,
    }
    with FEATURE_STORE_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta_out, f)

    return mmap


def run_single_selection(
    features_df: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
    label_type: Literal["hard", "direction", "directional_prob"],
    cutoff: float,
) -> dict:
    """
    执行单次特征筛选

    Args:
        features_df: 全局特征 DataFrame
        candles: K 线数据
        log_return_lag: GMMLabeler 的 lag_n 参数
        pred_next: 预测时间范围
        label_type: 标签类型 ("hard" 或 "direction")
        cutoff: GrootCV 筛选阈值

    Returns:
        包含筛选结果的字典
    """
    print(f"\n{'=' * 60}")
    print(
        f"[筛选] log_return_lag={log_return_lag}, pred_next={pred_next}, "
        f"label_type={label_type}"
    )
    print("=" * 60)

    # 1. 生成标签
    labeler = GMMLabeler(candles, lag_n=log_return_lag, verbose=False)
    gmm_random_state = labeler.random_state  # 记录 GMM 的 seed，供后续 build 复现
    if label_type == "hard":
        raw_labels = labeler.label_hard_state
    elif label_type == "direction":
        raw_labels = labeler.label_direction_force
    elif label_type == "directional_prob":
        raw_labels = labeler.label_directional_prob
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    print(f"标签生成完成: {len(raw_labels)} 样本, GMM seed={gmm_random_state}")

    # 2. 对齐特征和标签
    aligned_features, aligned_labels = align_features_labels(
        features_df,
        raw_labels,
        log_return_lag=log_return_lag,
        pred_next=pred_next,
    )
    print(f"对齐后: {len(aligned_features)} 样本, {aligned_features.shape[1]} 特征")

    # 3. 特征筛选
    groot_config = GrootCVConfig(
        cutoff=cutoff,
        shap_backend=GROOTCV_SHAP_BACKEND,
        shap_batch_size=GROOTCV_SHAP_BATCH_SIZE,
        shap_max_samples=GROOTCV_SHAP_MAX_SAMPLES,
        fastshap=GROOTCV_FASTSHAP,
    )
    selector = GrootCVSelector(config=groot_config, verbose=True)
    selector.fit(aligned_features, aligned_labels)

    selected = selector.selected_features_
    print(f"筛选结果: {len(selected)}/{aligned_features.shape[1]} 特征")

    return {
        "log_return_lag": log_return_lag,
        "pred_next": pred_next,
        "label_type": label_type,
        "gmm_random_state": gmm_random_state,  # GMM seed，确保 build 时标签一致
        "n_total_features": aligned_features.shape[1],
        "n_selected_features": len(selected),
        "selected_features": json.dumps(selected),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    print("=" * 60)
    print("Flow Feature Select - 批量特征筛选")
    print("=" * 60)
    print(f"数据范围: {TRAIN_START} ~ {TRAIN_END}")
    print(f"LOG_RETURN_LAGS: {LOG_RETURN_LAGS}")
    print(f"PRED_NEXT_STEPS: {PRED_NEXT_STEPS}")
    print(f"LABEL_TYPES: {LABEL_TYPES}")
    print(f"GROOTCV_CUTOFF: {GROOTCV_CUTOFF}")
    print(f"GROOTCV_SHAP_BACKEND: {GROOTCV_SHAP_BACKEND}")
    print(f"GROOTCV_SHAP_BATCH_SIZE: {GROOTCV_SHAP_BATCH_SIZE}")
    print(f"GROOTCV_SHAP_MAX_SAMPLES: {GROOTCV_SHAP_MAX_SAMPLES}")
    print(f"GROOTCV_FASTSHAP: {GROOTCV_FASTSHAP}")
    print(f"特征数量: {len(FEATURE_NAMES)}")
    print("=" * 60)

    # 1. 获取 fusion candles
    print("\n[1/3] 加载 K 线数据...")
    _, raw_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(TRAIN_START),
        helpers.date_to_timestamp(TRAIN_END),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )
    print(f"原始 K 线数据: {raw_candles.shape}")

    # 转换为 Fusion Bar
    bar = DemoBar(max_bars=-1)
    bar.update_with_candles(raw_candles)
    candles = bar.get_fusion_bars()
    print(f"Fusion K 线数据: {candles.shape}")

    # 2. 计算全局特征（只计算一次）
    print("\n[2/3] 准备全局特征...")
    calc = SimpleFeatureCalculator(verbose=True)
    calc.load(candles, sequential=True)

    feature_store = _build_or_load_feature_store(calc, candles, FEATURE_NAMES)
    features_df = pd.DataFrame(feature_store, columns=FEATURE_NAMES, copy=False)
    print(
        f"全局特征: {features_df.shape}, dtype={FEATURE_STORE_DTYPE}, "
        f"memmap={FEATURE_STORE_PATH}"
    )
    calc.clear_cache()
    del calc

    gc.collect()

    # 3. 遍历所有参数组合进行筛选
    print("\n[3/3] 开始批量特征筛选...")
    combinations = list(product(LOG_RETURN_LAGS, PRED_NEXT_STEPS, LABEL_TYPES))
    total = len(combinations)
    results = []

    for idx, (log_return_lag, pred_next, label_type) in enumerate(combinations, 1):
        print(f"\n进度: {idx}/{total}")
        try:
            result = run_single_selection(
                features_df=features_df,
                candles=candles,
                log_return_lag=log_return_lag,
                pred_next=pred_next,
                label_type=label_type,
                cutoff=GROOTCV_CUTOFF,
            )
            results.append(result)
        except Exception as e:
            print(
                f"[ERROR] log_return_lag={log_return_lag}, pred_next={pred_next}, "
                f"label_type={label_type}: {e}"
            )
            continue

    # 4. 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'=' * 60}")
        print(f"完成! 结果已保存到: {OUTPUT_FILE}")
        print(f"共 {len(results)} 条记录")
        print("=" * 60)

        # 打印汇总
        print("\n筛选结果汇总:")
        print(
            df[
                [
                    "log_return_lag",
                    "pred_next",
                    "label_type",
                    "n_total_features",
                    "n_selected_features",
                ]
            ].to_string()
        )
    else:
        print("\n[WARNING] 没有成功的筛选结果")


if __name__ == "__main__":
    main()
