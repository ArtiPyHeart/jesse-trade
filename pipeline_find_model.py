"""
Pipeline Find Model - 使用 FeaturePipeline 的模型搜索流水线

新流程：
1. 获取 fusion candles
2. 全局 FeaturePipeline（不降维）→ 计算全量特征（含 SSM）
3. 按 label 进行特征筛选 → 返回特征名称（含 SSM 如 deep_ssm_0）
4. 模型特定 FeaturePipeline（copy_ssm_from + 降维）→ 降维后特征
5. LightGBM 调参
6. CSV 记录（含降维器配置、降维前特征数量）
"""

import gc
import json
import logging
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp

from research.model_pick.candle_fetch import FusionCandles, bar_container
from research.model_pick.feature_utils import (
    align_features_and_labels,
    build_full_feature_config,
    build_model_config,
    select_features,
)
from research.model_pick.features import ALL_FEATS
from research.model_pick.labeler import PipelineLabeler
from research.model_pick.model_tuning import ModelTuning
from src.features.pipeline import FeaturePipeline

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 抑制Optuna的详细日志
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 抑制LightGBM的日志
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 用于保存deep ssm与lg ssm
MODEL_SAVE_DIR = Path("strategies/BinanceBtcDemoBarV2/models")

# 固定训练集切分点
TRAIN_TEST_SPLIT_DATE = "2025-05-31"
CANDLE_START = "2022-08-01"
CANDLE_END = "2025-07-01"
RESULTS_FILE = "model_search_results.csv"

# ARDVAE 降维器配置（固定，不进行调参）
REDUCER_CONFIG = {
    "max_latent_dim": 512,  # over-complete 设计，ARD prior 自动确定 active dims
    "kl_threshold": 0.01,  # 判断维度是否 active 的阈值
    "max_epochs": 200,
    "patience": 15,
    "seed": 42,
}


class ModelSearchTracker:
    """管理模型搜索结果的保存和进度追踪"""

    def __init__(self, results_file: str = RESULTS_FILE):
        self.results_file = results_file
        self.results_df = self._load_results()

    def _load_results(self) -> pd.DataFrame:
        """加载已有的结果文件"""
        if os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
                logger.info(
                    f"加载已有结果文件: {self.results_file}, 包含 {len(df)} 条记录"
                )
                return df
            except Exception as e:
                logger.warning(f"读取结果文件失败: {e}, 创建新文件")
                return pd.DataFrame()
        else:
            logger.info(f"创建新的结果文件: {self.results_file}")
            return pd.DataFrame()

    def is_completed(
        self, log_return_lag: int, pred_next: int, model_type: str
    ) -> bool:
        """检查某个参数组合是否已完成"""
        if self.results_df.empty:
            return False

        mask = (
            (self.results_df["log_return_lag"] == log_return_lag)
            & (self.results_df["pred_next"] == pred_next)
            & (self.results_df["model_type"] == model_type)
            & (self.results_df["status"] == "completed")
        )
        return mask.any()

    def save_result(
        self,
        log_return_lag: int,
        pred_next: int,
        model_type: str,
        best_score: float,
        best_params: dict,
        feature_count: int,
        feature_names: list[str],
        duration: float,
        reducer_config: dict,
        n_features_before_reduction: int,
        n_features_after_reduction: int,
        status: str = "completed",
    ):
        """保存单个实验结果（新增降维相关字段）"""
        result = {
            "log_return_lag": log_return_lag,
            "pred_next": pred_next,
            "model_type": model_type,
            "best_score": best_score,
            "feature_count": feature_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "duration_seconds": duration,
            "selected_features": json.dumps(feature_names),
            # 新增字段：降维器相关
            "reducer_config": json.dumps(reducer_config),
            "n_features_before_reduction": n_features_before_reduction,
            "n_features_after_reduction": n_features_after_reduction,
            # 模型最佳参数
            "best_params": json.dumps(best_params),
        }

        # 添加到DataFrame
        new_df = pd.DataFrame([result])
        self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

        # 保存到文件
        self.results_df.to_csv(self.results_file, index=False)
        logger.info(
            f"保存结果: {model_type} (lag={log_return_lag}, pred={pred_next}) -> score={best_score:.4f}"
        )

    def get_pending_tasks(self, all_lags: list, all_preds: list) -> list:
        """获取未完成的任务列表"""
        pending = []
        for lag in all_lags:
            for pred in all_preds:
                for model_type in ["regressor", "classifier"]:
                    if not self.is_completed(lag, pred, model_type):
                        pending.append((lag, pred, model_type))
        return pending

    def print_summary(self):
        """打印结果汇总"""
        if self.results_df.empty:
            logger.info("暂无结果")
            return

        print("\n" + "=" * 60)
        print("模型搜索结果汇总")
        print("=" * 60)

        # 按模型类型分组显示最佳结果
        for model_type in ["classifier", "regressor"]:
            type_df = self.results_df[self.results_df["model_type"] == model_type]
            if not type_df.empty:
                best_row = type_df.loc[type_df["best_score"].idxmax()]
                print(f"\n{model_type.upper()} 最佳模型:")
                print(f"  - Log Return Lag: {int(best_row['log_return_lag'])}")
                print(f"  - Pred Next: {int(best_row['pred_next'])}")
                print(f"  - Score: {best_row['best_score']:.4f}")
                print(
                    f"  - Features (降维前): {int(best_row.get('n_features_before_reduction', best_row['feature_count']))}"
                )
                print(
                    f"  - Features (降维后): {int(best_row.get('n_features_after_reduction', best_row['feature_count']))}"
                )

        print("\n" + "=" * 60)


def cleanup_multiprocessing_resources():
    """强制清理 multiprocessing 资源，防止累积泄漏"""
    import ctypes

    # 多轮强制 Python 垃圾回收（处理循环引用）
    for _ in range(3):
        gc.collect()

    # 清理 multiprocessing 的全局资源
    try:
        for child in multiprocessing.active_children():
            child.join(timeout=1.0)
            if child.is_alive():
                child.terminate()
                child.join(timeout=1.0)
        gc.collect()
    except Exception as e:
        logger.warning(f"清理 multiprocessing 资源时出现警告（可忽略）: {e}")

    # 尝试释放 C 库内存（macOS/Linux）
    try:
        if hasattr(ctypes, "CDLL"):
            libc = ctypes.CDLL("libc.dylib")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
    except Exception:
        pass

    gc.collect()
    logger.debug("✓ Multiprocessing 资源清理完成")


logger.info("=" * 60)
logger.info("初始化数据加载和特征处理模块")
logger.info("=" * 60)

logger.info("加载K线数据: Binance Perpetual Futures BTC-USDT 1m")
candle_container = FusionCandles(
    exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
)
logger.info(f"{bar_container.THRESHOLD = }")
candles = candle_container.get_candles(CANDLE_START, CANDLE_END)
logger.info(f"K线数据加载完成: {len(candles)} 条记录")
logger.info(
    f"时间范围: {pd.to_datetime(candles[0][0], unit='ms')} - {pd.to_datetime(candles[-1][0], unit='ms')}"
)

# 构建全局 FeaturePipeline（不降维），计算全量特征
logger.info("初始化全局 FeaturePipeline（不降维）...")
global_config = build_full_feature_config(ALL_FEATS, ssm_state_dim=5)
global_pipeline = FeaturePipeline(global_config)
logger.info(f"配置特征数: {len(global_config.feature_names)} (含 SSM 特征)")

logger.info("计算全局特征（训练 SSM 模型）...")
global_features = global_pipeline.fit_transform(candles)
logger.info(f"全局特征计算完成: {global_features.shape}")

# 初始化追踪器
tracker = ModelSearchTracker()


def evaluate_classifier(
    global_pipeline: FeaturePipeline,
    global_features: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    """
    评估分类器

    流程：
    1. 生成标签
    2. 对齐全局特征与标签
    3. 划分训练集
    4. 特征筛选
    5. 构建模型特定 Pipeline（启用降维）
    6. 计算降维后特征
    7. 模型调参
    """
    logger.info(
        f"[分类器] 开始评估 - log_return_lag={log_return_lag}, pred_next={pred_next}"
    )

    # 1. 生成标签
    logger.info(f"[分类器] 创建标签器，log_return_lag={log_return_lag}")
    labeler = PipelineLabeler(candles, log_return_lag)
    raw_label = labeler.label_hard
    logger.info(
        f"[分类器] 标签分布: {dict(zip(*np.unique(raw_label[~np.isnan(raw_label)].astype(int), return_counts=True)))}"
    )

    # 2. 对齐全局特征与标签
    logger.info("[分类器] 对齐特征和标签...")
    aligned_features, aligned_labels = align_features_and_labels(
        global_features, raw_label, pred_next, candles[:, 0]
    )
    logger.info(f"[分类器] 对齐后特征维度: {aligned_features.shape}")

    # 3. 划分训练集
    train_mask = aligned_features.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = aligned_features[train_mask]
    train_y = aligned_labels[: train_mask.sum()]
    logger.info(
        f"[分类器] 训练集大小: {train_x.shape[0]} 样本, {train_x.shape[1]} 特征"
    )

    # 4. 特征筛选
    logger.info("[分类器] 开始特征筛选...")
    selection_result = select_features(train_x, train_y)
    logger.info(
        f"[分类器] 特征筛选完成: 从 {selection_result.n_total} 个特征中选择了 {selection_result.n_selected} 个"
    )

    # 5. 构建模型特定 Pipeline（启用降维）
    logger.info("[分类器] 构建模型特定 Pipeline（启用 ARDVAE 降维）...")
    model_config = build_model_config(
        selection_result.selected_features,
        ssm_state_dim=5,
        reducer_config=REDUCER_CONFIG,
    )
    model_pipeline = FeaturePipeline(model_config)
    model_pipeline.share_raw_calculator_from(global_pipeline)
    model_pipeline.copy_ssm_from(global_pipeline)

    # 6. 计算降维后特征
    logger.info("[分类器] 计算降维后特征...")
    model_features = model_pipeline.fit_transform(candles)
    logger.info(
        f"[分类器] 降维完成: {selection_result.n_selected} -> {model_features.shape[1]} 维"
    )

    # 7. 重新对齐降维后特征
    model_aligned, _ = align_features_and_labels(
        model_features, raw_label, pred_next, candles[:, 0]
    )
    train_x_reduced = model_aligned[
        model_aligned.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    ]

    # 8. 模型调参
    logger.info("[分类器] 开始模型调参...")
    model_tuning = ModelTuning(TRAIN_TEST_SPLIT_DATE, train_x_reduced, train_y)
    params, best_score = model_tuning.tuning_classifier_direct(train_x_reduced, train_y)
    logger.info(f"[分类器] 调参完成 - 最佳得分: {best_score:.4f}")

    # 返回结果
    reducer_info = {
        "config": REDUCER_CONFIG,
        "n_before_reduction": selection_result.n_selected,
        "n_after_reduction": model_features.shape[1],
    }

    # 清理模型特定 Pipeline
    del model_pipeline
    gc.collect()

    return (
        params,
        best_score,
        selection_result.n_selected,
        selection_result.selected_features,
        reducer_info,
    )


def evaluate_regressor(
    global_pipeline: FeaturePipeline,
    global_features: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    """
    评估回归器

    流程与分类器相同，使用连续标签
    """
    logger.info(
        f"[回归器] 开始评估 - log_return_lag={log_return_lag}, pred_next={pred_next}"
    )

    # 1. 生成标签
    logger.info(f"[回归器] 创建标签器，log_return_lag={log_return_lag}")
    labeler = PipelineLabeler(candles, log_return_lag)
    raw_label = labeler.label_direction
    valid_labels = raw_label[~np.isnan(raw_label)]
    logger.info(
        f"[回归器] 标签统计: 均值={np.mean(valid_labels):.6f}, 标准差={np.std(valid_labels):.6f}"
    )

    # 2. 对齐全局特征与标签
    logger.info("[回归器] 对齐特征和标签...")
    aligned_features, aligned_labels = align_features_and_labels(
        global_features, raw_label, pred_next, candles[:, 0]
    )
    logger.info(f"[回归器] 对齐后特征维度: {aligned_features.shape}")

    # 3. 划分训练集
    train_mask = aligned_features.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = aligned_features[train_mask]
    train_y = aligned_labels[: train_mask.sum()]
    logger.info(
        f"[回归器] 训练集大小: {train_x.shape[0]} 样本, {train_x.shape[1]} 特征"
    )

    # 4. 特征筛选
    logger.info("[回归器] 开始特征筛选...")
    selection_result = select_features(train_x, train_y)
    logger.info(
        f"[回归器] 特征筛选完成: 从 {selection_result.n_total} 个特征中选择了 {selection_result.n_selected} 个"
    )

    # 5. 构建模型特定 Pipeline（启用降维）
    logger.info("[回归器] 构建模型特定 Pipeline（启用 ARDVAE 降维）...")
    model_config = build_model_config(
        selection_result.selected_features,
        ssm_state_dim=5,
        reducer_config=REDUCER_CONFIG,
    )
    model_pipeline = FeaturePipeline(model_config)
    model_pipeline.share_raw_calculator_from(global_pipeline)
    model_pipeline.copy_ssm_from(global_pipeline)

    # 6. 计算降维后特征
    logger.info("[回归器] 计算降维后特征...")
    model_features = model_pipeline.fit_transform(candles)
    logger.info(
        f"[回归器] 降维完成: {selection_result.n_selected} -> {model_features.shape[1]} 维"
    )

    # 7. 重新对齐降维后特征
    model_aligned, _ = align_features_and_labels(
        model_features, raw_label, pred_next, candles[:, 0]
    )
    train_x_reduced = model_aligned[
        model_aligned.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    ]

    # 8. 模型调参
    logger.info("[回归器] 开始模型调参...")
    model_tuning = ModelTuning(TRAIN_TEST_SPLIT_DATE, train_x_reduced, train_y)
    params, best_score = model_tuning.tuning_regressor_direct(train_x_reduced, train_y)
    logger.info(f"[回归器] 调参完成 - 最佳R²得分: {best_score:.4f}")

    # 返回结果
    reducer_info = {
        "config": REDUCER_CONFIG,
        "n_before_reduction": selection_result.n_selected,
        "n_after_reduction": model_features.shape[1],
    }

    # 清理模型特定 Pipeline
    del model_pipeline
    gc.collect()

    return (
        params,
        best_score,
        selection_result.n_selected,
        selection_result.selected_features,
        reducer_info,
    )


if __name__ == "__main__":
    # 参数配置
    log_return_lags = list(range(4, 8))
    pred_next_steps = [1, 2, 3]

    # 获取待完成的任务
    logger.info("\n" + "=" * 60)
    logger.info("任务规划")
    logger.info("=" * 60)
    logger.info("参数配置:")
    logger.info(f"  - log_return_lags: {log_return_lags}")
    logger.info(f"  - pred_next_steps: {pred_next_steps}")
    logger.info("  - 模型类型: ['classifier', 'regressor']")
    logger.info(f"  - 训练/测试分割日期: {TRAIN_TEST_SPLIT_DATE}")
    logger.info(
        f"  - 降维器配置: max_latent_dim={REDUCER_CONFIG['max_latent_dim']}, kl_threshold={REDUCER_CONFIG['kl_threshold']}"
    )

    pending_tasks = tracker.get_pending_tasks(log_return_lags, pred_next_steps)
    total_tasks = len(log_return_lags) * len(pred_next_steps) * 2
    completed_tasks = total_tasks - len(pending_tasks)

    logger.info("\n任务统计:")
    logger.info(f"  - 总任务数: {total_tasks}")
    logger.info(f"  - 已完成: {completed_tasks}")
    logger.info(f"  - 待完成: {len(pending_tasks)}")

    if pending_tasks:
        logger.info("\n待完成任务列表:")
        for i, (lag, pred, model_type) in enumerate(pending_tasks[:5], 1):
            logger.info(f"  {i}. {model_type}: lag={lag}, pred={pred}")
        if len(pending_tasks) > 5:
            logger.info(f"  ... 还有 {len(pending_tasks) - 5} 个任务")

    if len(pending_tasks) == 0:
        logger.info("所有任务已完成!")
        tracker.print_summary()
        exit(0)

    # 主循环
    logger.info("\n" + "=" * 60)
    logger.info("开始模型搜索主循环")
    logger.info("=" * 60)

    for task_idx, (lag, pred, model_type) in enumerate(pending_tasks, 1):
        # 显示进度
        overall_progress = completed_tasks + task_idx
        logger.info("\n" + "-" * 60)
        logger.info(
            f"[进度 {overall_progress}/{total_tasks}] ({(overall_progress - 1) / total_tasks * 100:.1f}%) 任务 #{task_idx}/{len(pending_tasks)}"
        )
        logger.info(
            f"开始训练: {model_type.upper()} | log_return_lag={lag} | pred_next={pred}"
        )
        logger.info("-" * 60)

        try:
            start_time = time.time()

            if model_type == "classifier":
                params, score, feature_count, feature_names, reducer_info = (
                    evaluate_classifier(
                        global_pipeline,
                        global_features.copy(),
                        candles.copy(),
                        lag,
                        pred,
                    )
                )
            else:
                params, score, feature_count, feature_names, reducer_info = (
                    evaluate_regressor(
                        global_pipeline,
                        global_features.copy(),
                        candles.copy(),
                        lag,
                        pred,
                    )
                )

            duration = time.time() - start_time

            # 保存结果
            tracker.save_result(
                log_return_lag=lag,
                pred_next=pred,
                model_type=model_type,
                best_score=score,
                best_params=params,
                feature_count=feature_count,
                feature_names=feature_names,
                duration=duration,
                reducer_config=reducer_info["config"],
                n_features_before_reduction=reducer_info["n_before_reduction"],
                n_features_after_reduction=reducer_info["n_after_reduction"],
                status="completed",
            )

            logger.info("\n" + "=" * 40)
            logger.info("✓ 任务完成！")
            logger.info(f"  - 模型类型: {model_type}")
            logger.info(f"  - 参数: lag={lag}, pred={pred}")
            logger.info(f"  - 最佳得分: {score:.4f}")
            logger.info(
                f"  - 特征数量: {reducer_info['n_before_reduction']} -> {reducer_info['n_after_reduction']} (降维后)"
            )
            logger.info(f"  - 训练耗时: {duration:.1f} 秒")
            logger.info(
                f"  - 预计剩余时间: {(len(pending_tasks) - task_idx) * duration / 60:.1f} 分钟"
            )
            logger.info("=" * 40)

            # 强制清理资源
            cleanup_multiprocessing_resources()

        except KeyboardInterrupt:
            logger.warning("\n" + "!" * 60)
            logger.warning("用户中断程序")
            logger.warning(
                f"当前进度: {overall_progress}/{total_tasks} ({overall_progress / total_tasks * 100:.1f}%)"
            )
            logger.warning("!" * 60)
            tracker.print_summary()
            exit(0)

        except Exception as e:
            logger.error("\n" + "!" * 60)
            logger.error("✗ 训练失败!")
            logger.error(f"  - 错误信息: {str(e)}")
            logger.error(f"  - 失败任务: {model_type} (lag={lag}, pred={pred})")
            logger.error(f"  - 当前进度: {overall_progress}/{total_tasks}")
            logger.error("!" * 60)
            logger.error("程序终止，显示已完成的结果：")
            tracker.print_summary()
            raise

    # 完成后显示汇总
    logger.info("\n" + "=" * 60)
    logger.info("🎉 所有任务完成!")
    logger.info("=" * 60)
    tracker.print_summary()
    logger.info("\n程序执行完毕")
