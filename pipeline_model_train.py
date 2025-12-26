"""
Pipeline Model Train - 从特征筛选结果构建和训练模型

流程：
1. 读取 feature_selection_results.csv 所有 selected_features → 去重 + 排序
2. 单一 FeaturePipeline (启用降维) → 一阶特征计算 + SSM 训练/transform + ARDVAE 降维
3. 保存 FeaturePipeline 到 MODEL_DIR/global_pipeline/
4. 对每个 (lag, pred_next, label_type): 生成标签 → 对齐 → Optuna 调参 → 训练 LightGBM → 保存

与 pipeline_build_models.py 的区别：
- 输入 CSV: feature_selection_results.csv（无 best_params）
- 需要进行 Optuna 调参（200 trials）
- Pipeline 保存目录: global_pipeline/
"""

import gc
import json
import logging
import multiprocessing
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp

# Import torch-dependent modules before any LightGBM imports to avoid OpenMP conflicts.
from src.features.dimensionality_reduction import ARDVAEConfig
from src.features.pipeline import FeaturePipeline

from research.model_pick.candle_fetch import FusionCandles, bar_container
from research.model_pick.feature_utils import (
    align_features_and_labels,
    build_model_config,
)
from research.model_pick.labeler import PipelineLabeler
from research.model_pick.model_tuning import ModelTuning

import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 抑制 Optuna 的详细日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 抑制 LightGBM 的日志
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 固定参数配置
# ============================================================
MODEL_DIR = Path("./strategies/BinanceBtcDemoBarV2/models")
PIPELINE_NAME = "global_pipeline"
TRAIN_TEST_SPLIT_DATE = "2025-06-01"
CANDLE_START = "2022-08-01"
CANDLE_END = "2025-07-01"
GLOBAL_SEED = 42
FEATURE_SELECTION_FILE = "feature_selection_results.csv"
RESULTS_FILE = "model_train_results.csv"

# ARDVAE 降维器配置（固定，与 pipeline_find_model.py 保持一致）
REDUCER_CONFIG = ARDVAEConfig(
    max_latent_dim=512,  # over-complete 设计，ARD prior 自动确定 active dims
    kl_threshold=0.01,  # 判断维度是否 active 的阈值
    max_epochs=150,
    patience=15,
    seed=GLOBAL_SEED,
)


class ModelTrainTracker:
    """管理模型训练结果的保存和进度追踪"""

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
        self, log_return_lag: int, pred_next: int, label_type: str
    ) -> bool:
        """检查某个参数组合是否已完成"""
        if self.results_df.empty:
            return False

        mask = (
            (self.results_df["log_return_lag"] == log_return_lag)
            & (self.results_df["pred_next"] == pred_next)
            & (self.results_df["label_type"] == label_type)
            & (self.results_df["status"] == "completed")
        )
        return mask.any()

    def save_result(
        self,
        log_return_lag: int,
        pred_next: int,
        label_type: str,
        best_params: dict,
        best_score: float,
        n_features: int,
        duration: float,
        status: str = "completed",
    ):
        """保存单个实验结果"""
        model_type = "c" if label_type == "hard" else "r"
        model_name = f"{model_type}_L{log_return_lag}_N{pred_next}"

        result = {
            "log_return_lag": log_return_lag,
            "pred_next": pred_next,
            "label_type": label_type,
            "model_name": model_name,
            "best_score": best_score,
            "n_features": n_features,
            "best_params": json.dumps(best_params),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "status": status,
        }

        # 添加到 DataFrame
        new_df = pd.DataFrame([result])
        self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

        # 保存到文件
        self.results_df.to_csv(self.results_file, index=False)
        logger.info(
            f"保存结果: {model_name} (lag={log_return_lag}, pred={pred_next}) -> score={best_score:.4f}"
        )

    def get_pending_tasks(self, df_selection: pd.DataFrame) -> list[tuple]:
        """
        获取未完成的任务列表

        从 feature_selection_results.csv 中提取所有 (lag, pred_next, label_type) 组合
        """
        pending = []
        for _, row in df_selection.iterrows():
            lag = int(row["log_return_lag"])
            pred_next = int(row["pred_next"])
            label_type = row["label_type"]

            if not self.is_completed(lag, pred_next, label_type):
                pending.append((lag, pred_next, label_type))

        return pending

    def print_summary(self):
        """打印结果汇总"""
        if self.results_df.empty:
            logger.info("暂无结果")
            return

        print("\n" + "=" * 60)
        print("模型训练结果汇总")
        print("=" * 60)

        # 按 label_type 分组显示最佳结果
        for label_type in ["hard", "direction"]:
            type_df = self.results_df[self.results_df["label_type"] == label_type]
            if not type_df.empty:
                best_row = type_df.loc[type_df["best_score"].idxmax()]
                model_desc = "分类器" if label_type == "hard" else "回归器"
                print(f"\n{model_desc} ({label_type}) 最佳模型:")
                print(f"  - Model: {best_row['model_name']}")
                print(f"  - Score: {best_row['best_score']:.4f}")
                print(f"  - Features: {int(best_row['n_features'])}")

        print("\n" + "=" * 60)


def collect_all_features_from_csv(df: pd.DataFrame) -> list[str]:
    """
    从 feature_selection_results.csv 收集所有 selected_features，去重并排序

    Args:
        df: 包含 selected_features 列的 DataFrame

    Returns:
        排序后的去重特征列表
    """
    all_features = set()
    for features_json in df["selected_features"]:
        features = (
            json.loads(features_json)
            if isinstance(features_json, str)
            else features_json
        )
        all_features.update(features)
    return sorted(all_features)


def build_unified_pipeline(
    candles: np.ndarray,
    all_features: list[str],
) -> tuple[FeaturePipeline, pd.DataFrame]:
    """
    构建统一的 FeaturePipeline（启用降维）

    单一 Pipeline 完成：一阶特征计算 + SSM 训练/transform + ARDVAE 降维

    Args:
        candles: K线数据
        all_features: 去重后的全量特征列表

    Returns:
        (unified_pipeline, reduced_features)
    """
    config = build_model_config(
        selected_features=all_features,
        ssm_state_dim=5,
        reducer_config=REDUCER_CONFIG,
    )
    pipeline = FeaturePipeline(config)
    reduced_features = pipeline.fit_transform(candles)

    return pipeline, reduced_features


def _f1_eval(preds: np.ndarray, eval_dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM 自定义评估函数：weighted F1 score"""
    y_true = eval_dataset.get_label()
    value = f1_score(y_true, preds > 0.5, average="weighted")
    return "f1", value, True  # (name, value, higher_is_better)


def _r2_eval(preds: np.ndarray, eval_dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM 自定义评估函数：R² score"""
    y_true = eval_dataset.get_label()
    ss_res = np.sum((y_true - preds) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return "r2", r2, True  # (name, value, higher_is_better)


def _determine_best_iterations_by_cv(
    params: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    is_regression: bool,
    n_splits: int = 5,
    seed: int = GLOBAL_SEED,
) -> tuple[int, float]:
    """
    通过 5-fold CV 确定最佳迭代轮数

    Args:
        params: LightGBM 参数
        train_x: 训练特征
        train_y: 训练标签
        is_regression: 是否回归任务
        n_splits: CV 折数
        seed: 随机种子

    Returns:
        (best_iteration, cv_score): 最佳迭代轮数和 CV 评分
    """
    # 选择 CV 策略
    if is_regression:
        cv_folds = list(
            KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x)
        )
        feval = _r2_eval
        metric_name = "r2"
    else:
        cv_folds = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
                train_x, train_y
            )
        )
        feval = _f1_eval
        metric_name = "f1"

    # 构建 Dataset
    dtrain = lgb.Dataset(train_x, train_y, free_raw_data=True, params={"max_bin": 255})

    # 执行 CV
    cv_results = lgb.cv(
        params,
        dtrain,
        num_boost_round=3000,
        folds=cv_folds,
        feval=feval,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # 获取最佳迭代轮数和评分
    best_iteration = len(cv_results[f"valid {metric_name}-mean"])
    best_score = cv_results[f"valid {metric_name}-mean"][-1]

    return best_iteration, best_score


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
    logger.debug("Multiprocessing 资源清理完成")


def train_model(
    reduced_features: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
    label_type: str,
) -> tuple[dict, float, lgb.Booster]:
    """
    训练单个模型：生成标签 → 对齐 → Optuna 调参 → 全量训练

    Args:
        reduced_features: 降维后的特征 DataFrame
        candles: K线数据
        log_return_lag: 标签计算的滞后期数
        pred_next: 预测步长
        label_type: 标签类型（"hard" 或 "direction"）

    Returns:
        (best_params, best_score, model): 最佳参数、最佳得分、训练好的模型
    """
    is_regression = label_type == "direction"
    model_type = "r" if is_regression else "c"
    model_name = f"{model_type}_L{log_return_lag}_N{pred_next}"

    logger.info(
        f"[{model_name}] 开始训练 - log_return_lag={log_return_lag}, pred_next={pred_next}, label_type={label_type}"
    )

    # 1. 生成标签
    logger.info(f"[{model_name}] 创建标签器...")
    labeler = PipelineLabeler(candles, log_return_lag)
    raw_label = labeler.label_direction if is_regression else labeler.label_hard

    if is_regression:
        valid_labels = raw_label[~np.isnan(raw_label)]
        logger.info(
            f"[{model_name}] 标签统计: 均值={np.mean(valid_labels):.6f}, 标准差={np.std(valid_labels):.6f}"
        )
    else:
        label_dist = dict(
            zip(
                *np.unique(
                    raw_label[~np.isnan(raw_label)].astype(int), return_counts=True
                )
            )
        )
        logger.info(f"[{model_name}] 标签分布: {label_dist}")

    del labeler

    # 2. 对齐特征和标签
    logger.info(f"[{model_name}] 对齐特征和标签...")
    aligned_features, aligned_labels = align_features_and_labels(
        reduced_features, raw_label, pred_next, candles[:, 0]
    )
    logger.info(f"[{model_name}] 对齐后特征维度: {aligned_features.shape}")

    # 3. 划分训练集
    train_mask = aligned_features.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = aligned_features[train_mask]
    train_y = aligned_labels[: train_mask.sum()]
    logger.info(
        f"[{model_name}] 训练集大小: {train_x.shape[0]} 样本, {train_x.shape[1]} 特征"
    )

    # 4. Optuna 调参
    logger.info(f"[{model_name}] 开始 Optuna 调参 (200 trials)...")
    model_tuning = ModelTuning(TRAIN_TEST_SPLIT_DATE, train_x, train_y)
    if is_regression:
        best_params, best_score = model_tuning.tuning_regressor_direct(train_x, train_y)
        metric_name = "R²"
    else:
        best_params, best_score = model_tuning.tuning_classifier_direct(
            train_x, train_y
        )
        metric_name = "F1"

    logger.info(f"[{model_name}] 调参完成 - 最佳 {metric_name}: {best_score:.4f}")

    # 5. 添加固定参数
    best_params.update(
        {
            "boosting": "gbdt",
            "is_unbalance": False,
            "feature_fraction": 1.0,
            "feature_pre_filter": False,
            "seed": GLOBAL_SEED,
            "feature_fraction_seed": GLOBAL_SEED,
            "bagging_seed": GLOBAL_SEED,
            "data_random_seed": GLOBAL_SEED,
        }
    )

    # 6. CV 确定最佳迭代轮数
    logger.info(f"[{model_name}] 运行 5-fold CV 确定最佳迭代轮数...")
    train_x_np = np.ascontiguousarray(train_x.to_numpy(dtype=np.float32))
    best_iteration, cv_score = _determine_best_iterations_by_cv(
        best_params, train_x_np, train_y, is_regression, n_splits=5, seed=GLOBAL_SEED
    )
    logger.info(
        f"[{model_name}] CV {metric_name}: {cv_score:.4f}, 最佳迭代轮数: {best_iteration}"
    )

    # 7. 全量训练
    logger.info(
        f"[{model_name}] 训练最终模型: {train_x.shape[1]} 特征, {best_iteration} 轮..."
    )
    model = lgb.train(
        best_params,
        lgb.Dataset(train_x_np, train_y),
        num_boost_round=best_iteration,
    )

    # 清理内存
    del aligned_features, aligned_labels, train_x, train_x_np, train_y, raw_label
    gc.collect()

    return best_params, best_score, model


# ============================================================================
# 主流程
# ============================================================================

if __name__ == "__main__":
    # 设置全局 seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    logger.info("=" * 60)
    logger.info("Pipeline Model Train - 从特征筛选结果构建和训练模型")
    logger.info("=" * 60)
    logger.info(f"Global random seed: {GLOBAL_SEED}")
    logger.info(f"训练/测试分割日期: {TRAIN_TEST_SPLIT_DATE}")
    logger.info(f"K线范围: {CANDLE_START} ~ {CANDLE_END}")
    logger.info("=" * 60 + "\n")

    # 1. 读取 feature_selection_results.csv
    logger.info("加载特征筛选结果...")
    df_selection = pd.read_csv(Path(__file__).parent / FEATURE_SELECTION_FILE)
    df_selection["selected_features"] = df_selection["selected_features"].apply(
        json.loads
    )
    logger.info(f"加载了 {len(df_selection)} 条特征筛选记录")

    # 2. 加载 candles
    logger.info("加载 K 线数据: Binance Perpetual Futures BTC-USDT 1m")
    candle_container = FusionCandles(
        exchange="Binance Perpetual Futures",
        symbol="BTC-USDT",
        timeframe="1m",
    )
    logger.info(f"{bar_container.THRESHOLD = }")
    candles = candle_container.get_candles(CANDLE_START, CANDLE_END)
    logger.info(f"K 线数据加载完成: {len(candles)} 条记录")
    logger.info(
        f"时间范围: {pd.to_datetime(candles[0][0], unit='ms')} - "
        f"{pd.to_datetime(candles[-1][0], unit='ms')}"
    )

    # 3. 收集所有特征（去重排序）
    all_features = collect_all_features_from_csv(df_selection)
    logger.info(f"收集到 {len(all_features)} 个唯一特征")

    # 4. 检查 Pipeline 是否已存在
    pipeline_path = MODEL_DIR / PIPELINE_NAME
    if pipeline_path.exists():
        logger.info(f"发现已存在的 Pipeline: {pipeline_path}")
        logger.info("加载已有 Pipeline...")
        unified_pipeline = FeaturePipeline.load(str(MODEL_DIR), PIPELINE_NAME)
        logger.info("使用已有 Pipeline 计算降维后特征...")
        reduced_features = unified_pipeline.transform(candles)
    else:
        # 5. 构建统一 Pipeline（SSM + ARDVAE）
        logger.info("\n构建统一 FeaturePipeline（SSM + ARDVAE 降维）...")
        unified_pipeline, reduced_features = build_unified_pipeline(
            candles, all_features
        )
        logger.info(f"降维后特征维度: {reduced_features.shape}")

        # 6. 保存 Pipeline
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        unified_pipeline.save(str(MODEL_DIR), PIPELINE_NAME)
        logger.info(f"Pipeline 已保存到: {pipeline_path}")

    # 7. 初始化 tracker，获取待完成任务
    tracker = ModelTrainTracker()
    pending_tasks = tracker.get_pending_tasks(df_selection)

    total_tasks = len(df_selection)
    completed_tasks = total_tasks - len(pending_tasks)

    logger.info("\n" + "=" * 60)
    logger.info("任务规划")
    logger.info("=" * 60)
    logger.info(f"  - 总任务数: {total_tasks}")
    logger.info(f"  - 已完成: {completed_tasks}")
    logger.info(f"  - 待完成: {len(pending_tasks)}")

    if pending_tasks:
        logger.info("\n待完成任务列表:")
        for i, (lag, pred_next, label_type) in enumerate(pending_tasks[:5], 1):
            model_type = "c" if label_type == "hard" else "r"
            logger.info(f"  {i}. {model_type}_L{lag}_N{pred_next} ({label_type})")
        if len(pending_tasks) > 5:
            logger.info(f"  ... 还有 {len(pending_tasks) - 5} 个任务")

    if len(pending_tasks) == 0:
        logger.info("所有任务已完成!")
        tracker.print_summary()
        exit(0)

    # 8. 遍历训练每个模型
    logger.info("\n" + "=" * 60)
    logger.info("开始模型训练主循环")
    logger.info("=" * 60)

    for task_idx, (lag, pred_next, label_type) in enumerate(pending_tasks, 1):
        model_type = "c" if label_type == "hard" else "r"
        model_name = f"{model_type}_L{lag}_N{pred_next}"
        model_path = MODEL_DIR / f"model_{model_name}.txt"

        # 显示进度
        overall_progress = completed_tasks + task_idx
        logger.info("\n" + "-" * 60)
        logger.info(
            f"[进度 {overall_progress}/{total_tasks}] "
            f"({(overall_progress - 1) / total_tasks * 100:.1f}%) "
            f"任务 #{task_idx}/{len(pending_tasks)}"
        )
        logger.info(f"开始训练: {model_name}")
        logger.info("-" * 60)

        try:
            start_time = time.time()

            best_params, best_score, model = train_model(
                reduced_features,
                candles,
                lag,
                pred_next,
                label_type,
            )

            duration = time.time() - start_time

            # 保存模型
            model.save_model(model_path.resolve().as_posix())
            logger.info(f"[{model_name}] 模型已保存到: {model_path}")

            # 记录结果
            tracker.save_result(
                log_return_lag=lag,
                pred_next=pred_next,
                label_type=label_type,
                best_params=best_params,
                best_score=best_score,
                n_features=reduced_features.shape[1],
                duration=duration,
                status="completed",
            )

            logger.info("\n" + "=" * 40)
            logger.info("任务完成!")
            logger.info(f"  - 模型: {model_name}")
            logger.info(f"  - 最佳得分: {best_score:.4f}")
            logger.info(f"  - 训练耗时: {duration:.1f} 秒")
            remaining = len(pending_tasks) - task_idx
            if remaining > 0:
                logger.info(f"  - 预计剩余时间: {remaining * duration / 60:.1f} 分钟")
            logger.info("=" * 40)

            # 强制清理资源
            del model
            cleanup_multiprocessing_resources()

        except KeyboardInterrupt:
            logger.warning("\n" + "!" * 60)
            logger.warning("用户中断程序")
            logger.warning(
                f"当前进度: {overall_progress}/{total_tasks} "
                f"({overall_progress / total_tasks * 100:.1f}%)"
            )
            logger.warning("!" * 60)
            tracker.print_summary()
            exit(0)

        except Exception as e:
            logger.error("\n" + "!" * 60)
            logger.error("训练失败!")
            logger.error(f"  - 错误信息: {str(e)}")
            logger.error(f"  - 失败任务: {model_name}")
            logger.error(f"  - 当前进度: {overall_progress}/{total_tasks}")
            logger.error("!" * 60)
            logger.error("程序终止，显示已完成的结果：")
            tracker.print_summary()
            raise

    # 完成后显示汇总
    logger.info("\n" + "=" * 60)
    logger.info("所有任务完成!")
    logger.info("=" * 60)
    tracker.print_summary()
    logger.info("\n程序执行完毕")
