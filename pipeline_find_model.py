import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp

from research.model_pick.candle_fetch import FusionCandles
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

# 固定训练集切分点，从而固定训练集，节约特征生成和筛选的时间。测试集主要用于回测
TRAIN_TEST_SPLIT_DATE = "2025-03-01"
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
                best_row = type_df.loc[
                    (
                        type_df["best_score"].idxmax()
                        if model_type == "classifier"
                        else type_df["best_score"].idxmin()
                    )
                ]
                print(f"\n{model_type.upper()} 最佳模型:")
                print(f"  - Log Return Lag: {int(best_row['log_return_lag'])}")
                print(f"  - Pred Next: {int(best_row['pred_next'])}")
                print(f"  - Score: {best_row['best_score']:.4f}")
                print(f"  - Features: {int(best_row['feature_count'])}")

        print("\n" + "=" * 60)


candle_container = FusionCandles(
    exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
)
candles = candle_container.get_candles("2022-07-01", "2025-09-15")
# 特征生成只关心特征名称和原始数据
feature_loader = FeatureLoader(candles)
# 由于训练集相同，selector内部的deep ssm与lg ssm只需要训练一次
feature_selector = FeatureSelector()

# 初始化追踪器
tracker = ModelSearchTracker()


def evaluate_classifier(
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    labeler = PipelineLabeler(candles, log_return_lag)
    label_for_classifier = labeler.label_hard

    df_feat, label_c = feature_loader.get_feature_label_bundle(
        label_for_classifier, pred_next
    )
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = df_feat[train_mask]
    train_y = label_c[train_mask]

    feature_names = feature_selector.select_features(train_x, train_y)

    model_tuning = ModelTuning(
        TRAIN_TEST_SPLIT_DATE,
        train_x,
        train_y,
    )

    params, best_score = model_tuning.tuning_classifier(feature_selector, feature_names)
    return params, best_score, len(feature_names)


def evaluate_regressor(
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    labeler = PipelineLabeler(candles, log_return_lag)
    label_for_regressor = labeler.label_direction

    df_feat, label_r = feature_loader.get_feature_label_bundle(
        label_for_regressor, pred_next
    )
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = df_feat[train_mask]
    train_y = label_r[train_mask]

    feature_names = feature_selector.select_features(train_x, train_y)

    model_tuning = ModelTuning(
        TRAIN_TEST_SPLIT_DATE,
        train_x,
        train_y,
    )

    params, best_score = model_tuning.tuning_regressor(feature_selector, feature_names)
    return params, best_score, len(feature_names)


if __name__ == "__main__":
    # 参数配置
    log_return_lags = [4, 5, 6, 7, 8]
    pred_next_steps = [1, 2, 3]

    # 获取待完成的任务
    pending_tasks = tracker.get_pending_tasks(log_return_lags, pred_next_steps)
    total_tasks = len(log_return_lags) * len(pred_next_steps) * 2  # 2种模型类型
    completed_tasks = total_tasks - len(pending_tasks)

    logger.info(
        f"总任务数: {total_tasks}, 已完成: {completed_tasks}, 待完成: {len(pending_tasks)}"
    )

    if len(pending_tasks) == 0:
        logger.info("所有任务已完成!")
        tracker.print_summary()
        exit(0)

    # 主循环
    for task_idx, (lag, pred, model_type) in enumerate(pending_tasks, 1):
        # 显示进度
        overall_progress = completed_tasks + task_idx
        logger.info(
            f"\n[{overall_progress}/{total_tasks}] 开始训练: {model_type} (lag={lag}, pred={pred})"
        )

        try:
            start_time = time.time()

            if model_type == "classifier":
                params, score, feature_count = evaluate_classifier(
                    candles.copy(), lag, pred
                )
            else:
                params, score, feature_count = evaluate_regressor(
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
                duration=duration,
                status="completed",
            )

            logger.info(
                f"✓ 完成训练 (耗时: {duration:.1f}秒, 得分: {score:.4f}, 特征数: {feature_count})"
            )

        except KeyboardInterrupt:
            logger.warning("用户中断程序")
            tracker.print_summary()
            exit(0)

        except Exception as e:
            logger.error(f"✗ 训练失败: {str(e)}")
            logger.error(f"失败位置: {model_type} (lag={lag}, pred={pred})")
            logger.error("程序终止")
            # 显示已完成的结果
            tracker.print_summary()
            raise

    # 完成后显示汇总
    logger.info("\n所有任务完成!")
    tracker.print_summary()
