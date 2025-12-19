"""
Pipeline Feature Selection - 纯特征筛选流水线

流程：
1. 获取 fusion candles
2. 全局 FeaturePipeline（不降维）→ 计算全量特征（含 SSM）
3. 按 label 配置进行特征筛选 → 记录到 CSV

不进行后续的降维和模型调参，目的是全量测试多个 label 配置下的特征筛选效果。
"""

import gc
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp

from research.model_pick.candle_fetch import FusionCandles, bar_container
from research.model_pick.feature_utils import (
    align_features_and_labels,
    build_full_feature_config,
    select_features,
)
from research.model_pick.features import ALL_FEATS
from research.model_pick.labeler import PipelineLabeler
from src.features.pipeline import FeaturePipeline

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 固定时间范围
TRAIN_TEST_SPLIT_DATE = "2025-05-31"
CANDLE_START = "2022-08-01"
CANDLE_END = "2025-07-01"
RESULTS_FILE = "feature_selection_results.csv"

# 搜索参数
LOG_RETURN_LAGS = list(range(4, 8))  # 4, 5, 6, 7
PRED_NEXT_STEPS = [1, 2, 3]
LABEL_TYPES = ["hard", "direction"]


class FeatureSelectionTracker:
    """管理特征筛选结果的保存和进度追踪"""

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
        n_total_features: int,
        n_selected_features: int,
        selected_features: list[str],
        duration: float,
        status: str = "completed",
    ):
        """保存单个实验结果"""
        result = {
            "log_return_lag": log_return_lag,
            "pred_next": pred_next,
            "label_type": label_type,
            "n_total_features": n_total_features,
            "n_selected_features": n_selected_features,
            "selected_features": json.dumps(selected_features),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "status": status,
        }

        # 添加到DataFrame
        new_df = pd.DataFrame([result])
        self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

        # 保存到文件
        self.results_df.to_csv(self.results_file, index=False)
        logger.info(
            f"保存结果: lag={log_return_lag}, pred={pred_next}, type={label_type} -> "
            f"{n_selected_features}/{n_total_features} 特征"
        )

    def get_pending_tasks(
        self, all_lags: list, all_preds: list, all_types: list
    ) -> list:
        """获取未完成的任务列表"""
        pending = []
        for lag in all_lags:
            for pred in all_preds:
                for label_type in all_types:
                    if not self.is_completed(lag, pred, label_type):
                        pending.append((lag, pred, label_type))
        return pending

    def print_summary(self):
        """打印结果汇总"""
        if self.results_df.empty:
            logger.info("暂无结果")
            return

        print("\n" + "=" * 60)
        print("特征筛选结果汇总")
        print("=" * 60)

        # 按 label_type 分组显示
        for label_type in LABEL_TYPES:
            type_df = self.results_df[self.results_df["label_type"] == label_type]
            if not type_df.empty:
                print(f"\n{label_type.upper()} 标签:")
                avg_selected = type_df["n_selected_features"].mean()
                avg_total = type_df["n_total_features"].mean()
                print(f"  - 平均选中特征数: {avg_selected:.1f} / {avg_total:.1f}")
                print(f"  - 选中率: {avg_selected / avg_total * 100:.1f}%")

                # 显示每个 lag 的统计
                for lag in sorted(type_df["log_return_lag"].unique()):
                    lag_df = type_df[type_df["log_return_lag"] == lag]
                    avg_sel = lag_df["n_selected_features"].mean()
                    print(f"    - lag={lag}: 平均 {avg_sel:.1f} 个特征")

        print("\n" + "=" * 60)


def run_feature_selection(
    global_features: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
    label_type: str,
) -> tuple[int, int, list[str]]:
    """
    执行单次特征筛选

    Parameters
    ----------
    global_features : pd.DataFrame
        全局特征（来自 FeaturePipeline）
    candles : np.ndarray
        K线数据
    log_return_lag : int
        标签计算的滞后期数
    pred_next : int
        预测步长
    label_type : str
        标签类型（"hard" 或 "direction"）

    Returns
    -------
    tuple[int, int, list[str]]
        (总特征数, 选中特征数, 选中特征名列表)
    """
    logger.info(f"[特征筛选] lag={log_return_lag}, pred={pred_next}, type={label_type}")

    # 1. 生成标签
    labeler = PipelineLabeler(candles, log_return_lag)
    if label_type == "hard":
        raw_label = labeler.label_hard
        label_desc = dict(
            zip(
                *np.unique(
                    raw_label[~np.isnan(raw_label)].astype(int), return_counts=True
                )
            )
        )
        logger.info(f"[特征筛选] 标签分布: {label_desc}")
    else:
        raw_label = labeler.label_direction
        valid_labels = raw_label[~np.isnan(raw_label)]
        logger.info(
            f"[特征筛选] 标签统计: 均值={np.mean(valid_labels):.6f}, "
            f"标准差={np.std(valid_labels):.6f}"
        )

    # 2. 对齐全局特征与标签
    aligned_features, aligned_labels = align_features_and_labels(
        global_features, raw_label, pred_next, candles[:, 0]
    )
    logger.info(f"[特征筛选] 对齐后特征维度: {aligned_features.shape}")

    # 3. 划分训练集
    train_mask = aligned_features.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = aligned_features[train_mask]
    train_y = aligned_labels[: train_mask.sum()]
    logger.info(
        f"[特征筛选] 训练集大小: {train_x.shape[0]} 样本, {train_x.shape[1]} 特征"
    )

    # 4. 特征筛选
    selection_result = select_features(train_x, train_y)
    logger.info(
        f"[特征筛选] 完成: 从 {selection_result.n_total} 个特征中选择了 "
        f"{selection_result.n_selected} 个"
    )

    return (
        selection_result.n_total,
        selection_result.n_selected,
        selection_result.selected_features,
    )


# ============================================================================
# 初始化
# ============================================================================
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
    f"时间范围: {pd.to_datetime(candles[0][0], unit='ms')} - "
    f"{pd.to_datetime(candles[-1][0], unit='ms')}"
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
tracker = FeatureSelectionTracker()


# ============================================================================
# 主循环
# ============================================================================
if __name__ == "__main__":
    # 获取待完成的任务
    logger.info("\n" + "=" * 60)
    logger.info("任务规划")
    logger.info("=" * 60)
    logger.info("参数配置:")
    logger.info(f"  - log_return_lags: {LOG_RETURN_LAGS}")
    logger.info(f"  - pred_next_steps: {PRED_NEXT_STEPS}")
    logger.info(f"  - label_types: {LABEL_TYPES}")
    logger.info(f"  - 训练/测试分割日期: {TRAIN_TEST_SPLIT_DATE}")

    pending_tasks = tracker.get_pending_tasks(
        LOG_RETURN_LAGS, PRED_NEXT_STEPS, LABEL_TYPES
    )
    total_tasks = len(LOG_RETURN_LAGS) * len(PRED_NEXT_STEPS) * len(LABEL_TYPES)
    completed_tasks = total_tasks - len(pending_tasks)

    logger.info("\n任务统计:")
    logger.info(f"  - 总任务数: {total_tasks}")
    logger.info(f"  - 已完成: {completed_tasks}")
    logger.info(f"  - 待完成: {len(pending_tasks)}")

    if pending_tasks:
        logger.info("\n待完成任务列表:")
        for i, (lag, pred, label_type) in enumerate(pending_tasks[:5], 1):
            logger.info(f"  {i}. lag={lag}, pred={pred}, type={label_type}")
        if len(pending_tasks) > 5:
            logger.info(f"  ... 还有 {len(pending_tasks) - 5} 个任务")

    if len(pending_tasks) == 0:
        logger.info("所有任务已完成!")
        tracker.print_summary()
        exit(0)

    # 主循环
    logger.info("\n" + "=" * 60)
    logger.info("开始特征筛选主循环")
    logger.info("=" * 60)

    for task_idx, (lag, pred, label_type) in enumerate(pending_tasks, 1):
        # 显示进度
        overall_progress = completed_tasks + task_idx
        logger.info("\n" + "-" * 60)
        logger.info(
            f"[进度 {overall_progress}/{total_tasks}] "
            f"({(overall_progress - 1) / total_tasks * 100:.1f}%) "
            f"任务 #{task_idx}/{len(pending_tasks)}"
        )
        logger.info(
            f"开始筛选: log_return_lag={lag} | pred_next={pred} | label_type={label_type}"
        )
        logger.info("-" * 60)

        try:
            start_time = time.time()

            n_total, n_selected, selected_features = run_feature_selection(
                global_features.copy(),
                candles.copy(),
                lag,
                pred,
                label_type,
            )

            duration = time.time() - start_time

            # 保存结果
            tracker.save_result(
                log_return_lag=lag,
                pred_next=pred,
                label_type=label_type,
                n_total_features=n_total,
                n_selected_features=n_selected,
                selected_features=selected_features,
                duration=duration,
                status="completed",
            )

            logger.info("\n" + "=" * 40)
            logger.info("任务完成!")
            logger.info(f"  - 参数: lag={lag}, pred={pred}, type={label_type}")
            logger.info(f"  - 选中特征: {n_selected}/{n_total}")
            logger.info(f"  - 耗时: {duration:.1f} 秒")
            remaining = len(pending_tasks) - task_idx
            if remaining > 0:
                logger.info(f"  - 预计剩余时间: {remaining * duration / 60:.1f} 分钟")
            logger.info("=" * 40)

            # 清理内存
            gc.collect()

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
            logger.error("筛选失败!")
            logger.error(f"  - 错误信息: {str(e)}")
            logger.error(f"  - 失败任务: lag={lag}, pred={pred}, type={label_type}")
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
