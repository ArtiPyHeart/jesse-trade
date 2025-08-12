"""
Optuna配置和存储管理模块
用于处理长时间优化实验的持久化存储和恢复
"""

import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import optuna

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptunaStorageManager:
    """Optuna存储管理器"""

    def __init__(self, storage_dir: str = "optuna_storage"):
        self.storage_dir = storage_dir
        self.db_path = os.path.join(storage_dir, "optuna_study.db")
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.storage_dir, exist_ok=True)

    @property
    def storage(self) -> optuna.storages.RDBStorage:
        """获取RDB存储对象"""
        return optuna.storages.RDBStorage(f"sqlite:///{self.db_path}")

    def create_or_load_study(
        self,
        study_name: Optional[str] = None,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """创建或加载研究"""

        if study_name is None:
            study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 默认采样器和剪枝器
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=500, multivariate=True
            )

        if pruner is None:
            pruner = optuna.pruners.PercentilePruner(
                percentile=10.0, n_startup_trials=50, n_warmup_steps=20
            )

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=load_if_exists,
        )

        logger.info(
            f"研究 '{study_name}' 已{'加载' if load_if_exists and self.study_exists(study_name) else '创建'}"
        )
        logger.info(f"数据库路径: {self.db_path}")
        logger.info(f"已完成试验数: {len(study.trials)}")

        return study

    def study_exists(self, study_name: str) -> bool:
        """检查研究是否存在"""
        if not os.path.exists(self.db_path):
            return False

        try:
            study_summaries = optuna.get_all_study_summaries(self.storage)
            return any(summary.study_name == study_name for summary in study_summaries)
        except Exception:
            return False

    def list_studies(self) -> List[optuna.study.StudySummary]:
        """列出所有研究"""
        if not os.path.exists(self.db_path):
            logger.warning(f"数据库文件不存在: {self.db_path}")
            return []

        try:
            return optuna.get_all_study_summaries(self.storage)
        except Exception as e:
            logger.error(f"获取研究列表时出错: {e}")
            return []

    def get_latest_study(self) -> Optional[optuna.Study]:
        """获取最新的研究"""
        studies = self.list_studies()
        if not studies:
            return None

        # 按创建时间排序，获取最新的
        latest_study = max(studies, key=lambda x: x.datetime_start)
        return optuna.load_study(
            study_name=latest_study.study_name, storage=self.storage
        )

    def backup_database(self) -> str:
        """备份数据库"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.backup_{timestamp}"

        import shutil

        shutil.copy2(self.db_path, backup_path)
        logger.info(f"数据库已备份到: {backup_path}")
        return backup_path

    def get_study_info(self, study_name: str) -> Dict[str, Any]:
        """获取研究的详细信息"""
        if not self.study_exists(study_name):
            raise ValueError(f"研究 '{study_name}' 不存在")

        study = optuna.load_study(study_name=study_name, storage=self.storage)

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        failed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]

        info = {
            "study_name": study_name,
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "failed_trials": len(failed_trials),
            "pruned_trials": len(pruned_trials),
            "best_value": study.best_value if completed_trials else None,
            "best_params": study.best_params if completed_trials else None,
            "direction": study.direction.name,
        }

        if completed_trials:
            values = [t.value for t in completed_trials]
            info.update(
                {
                    "worst_value": (
                        min(values)
                        if study.direction == optuna.study.StudyDirection.MAXIMIZE
                        else max(values)
                    ),
                    "mean_value": sum(values) / len(values),
                }
            )

        return info

    def cleanup_failed_trials(self, study_name: str) -> int:
        """清理失败的试验（注意：这会直接操作数据库）"""
        if not self.study_exists(study_name):
            raise ValueError(f"研究 '{study_name}' 不存在")

        # 这是一个高级操作，直接操作SQLite数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 获取研究ID
            cursor.execute(
                "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
            )
            study_id = cursor.fetchone()[0]

            # 删除失败的试验
            cursor.execute(
                "DELETE FROM trials WHERE study_id = ? AND state = ?",
                (study_id, optuna.trial.TrialState.FAIL.value),
            )

            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"已删除 {deleted_count} 个失败的试验")
            return deleted_count

        except Exception as e:
            conn.rollback()
            logger.error(f"清理失败试验时出错: {e}")
            raise
        finally:
            conn.close()


def create_robust_study(
    study_name: Optional[str] = None,
    storage_dir: str = "optuna_storage",
    direction: str = "maximize",
    n_startup_trials: int = 500,
    percentile_for_pruning: float = 10.0,
) -> optuna.Study:
    """创建一个具有鲁棒性配置的研究"""

    manager = OptunaStorageManager(storage_dir)

    # 配置采样器
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,
        constant_liar=True,  # 支持并行优化
        warn_independent_sampling=False,
    )

    # 配置剪枝器
    pruner = optuna.pruners.PercentilePruner(
        percentile=percentile_for_pruning,
        n_startup_trials=max(50, n_startup_trials // 10),
        n_warmup_steps=20,
    )

    return manager.create_or_load_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def safe_optimize(
    study: optuna.Study,
    objective,
    n_trials: int = 1000,
    timeout: Optional[float] = None,
    n_jobs: int = 1,
    catch: tuple = (Exception,),
    gc_after_trial: bool = True,
    show_progress_bar: bool = False,
):
    """安全的优化函数，包含错误处理和进度保存"""

    logger.info(f"开始优化，目标试验数: {n_trials}")
    logger.info(f"当前已完成试验数: {len(study.trials)}")

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=catch,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )
    except KeyboardInterrupt:
        logger.info("优化被用户中断")
    except Exception as e:
        logger.error(f"优化过程中出现错误: {e}")
        raise
    finally:
        logger.info(f"优化结束，总试验数: {len(study.trials)}")
        if study.trials:
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if completed_trials:
                logger.info(f"当前最佳值: {study.best_value:.4f}")
                logger.info(f"最佳参数: {study.best_params}")
