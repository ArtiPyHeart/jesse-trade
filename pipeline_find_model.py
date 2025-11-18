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
from research.model_pick.feature_select import FeatureSelector
from research.model_pick.features import FeatureLoader
from research.model_pick.labeler import PipelineLabeler
from research.model_pick.model_tuning import ModelTuning

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
# 使用 path = MODEL_SAVE_DIR / "deep_ssm"
# path.resolve().as_posix()的方式生成路径
MODEL_SAVE_DIR = Path("strategies/BinanceBtcDemoBarV2/models")

# 固定训练集切分点，从而固定训练集，节约特征生成和筛选的时间。测试集主要用于回测
TRAIN_TEST_SPLIT_DATE = "2025-05-31"
CANDLE_START = "2022-08-01"
CANDLE_END = "2025-11-15"
RESULTS_FILE = "model_search_results.csv"


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
        status: str = "completed",
    ):
        """保存单个实验结果"""
        result = {
            "log_return_lag": log_return_lag,
            "pred_next": pred_next,
            "model_type": model_type,
            "best_score": best_score,
            "feature_count": feature_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "duration_seconds": duration,
            "best_params": json.dumps(best_params),
            "selected_features": json.dumps(
                feature_names
            ),  # 将特征列表保存为JSON字符串
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
                # 分类器和回归器（R²）都是越大越好
                best_row = type_df.loc[type_df["best_score"].idxmax()]
                print(f"\n{model_type.upper()} 最佳模型:")
                print(f"  - Log Return Lag: {int(best_row['log_return_lag'])}")
                print(f"  - Pred Next: {int(best_row['pred_next'])}")
                print(f"  - Score: {best_row['best_score']:.4f}")
                print(f"  - Features: {int(best_row['feature_count'])}")

        print("\n" + "=" * 60)


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

# 特征生成只关心特征名称和原始数据
logger.info("初始化特征加载器...")
feature_loader = FeatureLoader(candles)

# 由于训练集相同，selector内部的deep ssm与lg ssm只需要训练一次
logger.info("初始化特征选择器（将缓存SSM模型）...")
feature_selector = FeatureSelector(model_save_dir=MODEL_SAVE_DIR)
logger.info("初始化完成")

# 初始化追踪器
tracker = ModelSearchTracker()


def cleanup_multiprocessing_resources():
    """
    强制清理 multiprocessing 资源，防止累积泄漏

    这个函数解决的问题：
    - LightGBM + GridSearchCV 创建的 worker 进程池
    - 进程间通信的 semaphore 和 shared memory
    - 这些资源在任务结束后可能不会自动释放
    """
    # 1. 强制 Python 垃圾回收
    gc.collect()

    # 2. 清理 multiprocessing 的全局资源
    try:
        # 获取当前进程的所有子进程
        current_process = multiprocessing.current_process()

        # 如果存在活跃的子进程，等待它们结束
        for child in multiprocessing.active_children():
            child.join(timeout=0.1)  # 短暂等待
            if child.is_alive():
                child.terminate()  # 强制终止僵尸进程

        # 3. 再次垃圾回收，清理终止进程的资源
        gc.collect()

    except Exception as e:
        logger.warning(f"清理 multiprocessing 资源时出现警告（可忽略）: {e}")

    logger.debug("✓ Multiprocessing 资源清理完成")


def evaluate_classifier(
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    logger.info(
        f"[分类器] 开始评估 - log_return_lag={log_return_lag}, pred_next={pred_next}"
    )

    # 创建标签
    logger.info(f"[分类器] 创建标签器，log_return_lag={log_return_lag}")
    labeler = PipelineLabeler(candles, log_return_lag)
    label_for_classifier = labeler.label_hard
    logger.info(
        f"[分类器] 标签分布: {np.unique(label_for_classifier, return_counts=True)}"
    )

    # 获取特征和标签
    logger.info(f"[分类器] 加载特征数据，pred_next={pred_next}")
    df_feat, label_c = feature_loader.get_feature_label_bundle(
        label_for_classifier, pred_next
    )
    logger.info(f"[分类器] 特征维度: {df_feat.shape}")

    # 划分训练集
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x_all_feat = df_feat[train_mask]
    train_y = label_c[train_mask]
    logger.info(
        f"[分类器] 训练集大小: {train_x_all_feat.shape[0]} 样本, {train_x_all_feat.shape[1]} 特征"
    )
    logger.info(
        f"[分类器] 训练集标签分布: {dict(zip(*np.unique(train_y, return_counts=True)))}"
    )

    # 特征选择
    logger.info(f"[分类器] 开始特征选择...")
    feature_names = feature_selector.select_features(train_x_all_feat, train_y)
    logger.info(
        f"[分类器] 特征选择完成: 从 {train_x_all_feat.shape[1]} 个特征中选择了 {len(feature_names)} 个"
    )
    logger.debug(f"[分类器] 选中的特征: {feature_names[:10]}...")  # 只显示前10个特征

    # 模型调参
    logger.info(f"[分类器] 开始模型调参...")
    model_tuning = ModelTuning(
        TRAIN_TEST_SPLIT_DATE,
        train_x_all_feat,
        train_y,
    )

    params, best_score = model_tuning.tuning_classifier(feature_selector, feature_names)
    logger.info(f"[分类器] 调参完成 - 最佳得分: {best_score:.4f}")
    logger.info(f"[分类器] 最佳参数: {params}")

    return params, best_score, len(feature_names), feature_names


def evaluate_regressor(
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    logger.info(
        f"[回归器] 开始评估 - log_return_lag={log_return_lag}, pred_next={pred_next}"
    )

    # 创建标签
    logger.info(f"[回归器] 创建标签器，log_return_lag={log_return_lag}")
    labeler = PipelineLabeler(candles, log_return_lag)
    label_for_regressor = labeler.label_direction
    logger.info(
        f"[回归器] 标签统计: 均值={np.mean(label_for_regressor):.6f}, 标准差={np.std(label_for_regressor):.6f}"
    )

    # 获取特征和标签
    logger.info(f"[回归器] 加载特征数据，pred_next={pred_next}")
    df_feat, label_r = feature_loader.get_feature_label_bundle(
        label_for_regressor, pred_next
    )
    logger.info(f"[回归器] 特征维度: {df_feat.shape}")

    # 划分训练集
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x_all_feat = df_feat[train_mask]
    train_y = label_r[train_mask]
    logger.info(
        f"[回归器] 训练集大小: {train_x_all_feat.shape[0]} 样本, {train_x_all_feat.shape[1]} 特征"
    )
    logger.info(
        f"[回归器] 训练集标签范围: [{np.min(train_y):.6f}, {np.max(train_y):.6f}]"
    )

    # 特征选择
    logger.info(f"[回归器] 开始特征选择...")
    feature_names = feature_selector.select_features(train_x_all_feat, train_y)
    logger.info(
        f"[回归器] 特征选择完成: 从 {train_x_all_feat.shape[1]} 个特征中选择了 {len(feature_names)} 个"
    )
    logger.debug(f"[回归器] 选中的特征: {feature_names[:10]}...")  # 只显示前10个特征

    # 模型调参
    logger.info(f"[回归器] 开始模型调参...")
    model_tuning = ModelTuning(
        TRAIN_TEST_SPLIT_DATE,
        train_x_all_feat,
        train_y,
    )

    params, best_score = model_tuning.tuning_regressor(feature_selector, feature_names)
    logger.info(f"[回归器] 调参完成 - 最佳R²得分: {best_score:.4f}")
    logger.info(f"[回归器] 最佳参数: {params}")

    return params, best_score, len(feature_names), feature_names


if __name__ == "__main__":
    # 参数配置
    log_return_lags = list(range(4, 8))
    pred_next_steps = [1, 2, 3, 4]

    # 获取待完成的任务
    logger.info("\n" + "=" * 60)
    logger.info("任务规划")
    logger.info("=" * 60)
    logger.info(f"参数配置:")
    logger.info(f"  - log_return_lags: {log_return_lags}")
    logger.info(f"  - pred_next_steps: {pred_next_steps}")
    logger.info(f"  - 模型类型: ['classifier', 'regressor']")
    logger.info(f"  - 训练/测试分割日期: {TRAIN_TEST_SPLIT_DATE}")

    pending_tasks = tracker.get_pending_tasks(log_return_lags, pred_next_steps)
    total_tasks = len(log_return_lags) * len(pred_next_steps) * 2  # 2种模型类型
    completed_tasks = total_tasks - len(pending_tasks)

    logger.info(f"\n任务统计:")
    logger.info(f"  - 总任务数: {total_tasks}")
    logger.info(f"  - 已完成: {completed_tasks}")
    logger.info(f"  - 待完成: {len(pending_tasks)}")

    if pending_tasks:
        logger.info(f"\n待完成任务列表:")
        for i, (lag, pred, model_type) in enumerate(
            pending_tasks[:5], 1
        ):  # 只显示前5个
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
                params, score, feature_count, feature_names = evaluate_classifier(
                    candles.copy(), lag, pred
                )
            else:
                params, score, feature_count, feature_names = evaluate_regressor(
                    candles.copy(), lag, pred
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
                status="completed",
            )

            logger.info("\n" + "=" * 40)
            logger.info(f"✓ 任务完成！")
            logger.info(f"  - 模型类型: {model_type}")
            logger.info(f"  - 参数: lag={lag}, pred={pred}")
            logger.info(f"  - 最佳得分: {score:.4f}")
            logger.info(f"  - 特征数量: {feature_count}")
            logger.info(f"  - 训练耗时: {duration:.1f} 秒")
            logger.info(
                f"  - 预计剩余时间: {(len(pending_tasks) - task_idx) * duration / 60:.1f} 分钟"
            )
            logger.info("=" * 40)

            # 🔧 强制清理资源，防止多进程资源泄漏累积
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
            logger.error(f"✗ 训练失败!")
            logger.error(f"  - 错误信息: {str(e)}")
            logger.error(f"  - 失败任务: {model_type} (lag={lag}, pred={pred})")
            logger.error(f"  - 当前进度: {overall_progress}/{total_tasks}")
            logger.error("!" * 60)
            logger.error("程序终止，显示已完成的结果：")
            # 显示已完成的结果
            tracker.print_summary()
            raise

    # 完成后显示汇总
    logger.info("\n" + "=" * 60)
    logger.info("🎉 所有任务完成!")
    logger.info("=" * 60)
    tracker.print_summary()
    logger.info("\n程序执行完毕")
