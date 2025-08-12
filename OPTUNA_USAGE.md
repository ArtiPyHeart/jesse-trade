# Optuna实验进度保存和恢复指南

本指南介绍如何在长时间的Optuna优化实验中保存进度到硬盘，并在意外中断后恢复实验。

## 核心特性

✅ **持久化存储**: 使用SQLite数据库自动保存所有试验结果  
✅ **自动恢复**: 实验中断后可以无缝继续  
✅ **安全优化**: 包含错误处理和进度监控  
✅ **存储管理**: 提供备份、清理等管理功能  

## 快速开始

### 1. 创建新的优化实验

```python
from tuning_pipeline import tune_pipeline
from research.optuna_config import create_robust_study, safe_optimize

# 创建研究（会自动保存到硬盘）
study = create_robust_study(
    study_name="my_experiment",  # 固定名称，便于后续继续
    storage_dir="optuna_storage",  # 存储目录
    direction="maximize"
)

# 开始优化（支持中断和恢复）
safe_optimize(
    study=study,
    objective=tune_pipeline,
    n_trials=50000,
    n_jobs=1,
    gc_after_trial=True
)
```

### 2. 继续中断的实验

```python
from research.optuna_config import OptunaStorageManager, safe_optimize

# 创建存储管理器
manager = OptunaStorageManager("optuna_storage")

# 加载现有研究
study = manager.create_or_load_study(
    study_name="my_experiment",
    load_if_exists=True
)

# 继续优化
safe_optimize(
    study=study,
    objective=tune_pipeline,
    n_trials=50000
)
```

## 存储结构

```
optuna_storage/
├── optuna_study.db          # SQLite数据库文件
├── optuna_study.db.backup_* # 自动备份文件
└── optimization_results.csv # 导出的结果文件
```

## 主要功能

### 存储管理器 (OptunaStorageManager)

```python
from research.optuna_config import OptunaStorageManager

manager = OptunaStorageManager("optuna_storage")

# 列出所有研究
studies = manager.list_studies()

# 检查研究是否存在
exists = manager.study_exists("my_experiment")

# 获取研究详细信息
info = manager.get_study_info("my_experiment")

# 备份数据库
backup_path = manager.backup_database()

# 清理失败的试验
deleted_count = manager.cleanup_failed_trials("my_experiment")

# 获取最新的研究
latest_study = manager.get_latest_study()
```

### 安全优化函数 (safe_optimize)

```python
from research.optuna_config import safe_optimize

safe_optimize(
    study=study,
    objective=objective_function,
    n_trials=1000,
    timeout=3600,  # 1小时超时
    n_jobs=1,
    catch=(Exception,),  # 捕获的异常类型
    gc_after_trial=True,  # 每次试验后垃圾回收
    show_progress_bar=False
)
```

## 实验恢复策略

### 策略1: 使用固定研究名称

```python
# 始终使用相同的研究名称
study_name = "backtest_tuning_main"

study = create_robust_study(
    study_name=study_name,
    load_if_exists=True  # 如果存在则加载，否则创建
)
```

### 策略2: 继续最新的研究

```python
manager = OptunaStorageManager()
latest_study = manager.get_latest_study()

if latest_study:
    safe_optimize(latest_study, objective, n_trials=1000)
```

### 策略3: 手动选择研究

```python
manager = OptunaStorageManager()
studies = manager.list_studies()

# 显示所有研究供用户选择
for i, summary in enumerate(studies):
    print(f"{i}: {summary.study_name} ({summary.n_trials} trials)")

# 选择要继续的研究
study_index = int(input("选择研究编号: "))
study_name = studies[study_index].study_name

study = manager.create_or_load_study(study_name, load_if_exists=True)
```

## 错误处理

### 处理KeyboardInterrupt

```python
try:
    safe_optimize(study, objective, n_trials=50000)
except KeyboardInterrupt:
    print("实验被用户中断")
    print(f"已完成试验数: {len(study.trials)}")
    if study.trials:
        print(f"当前最佳值: {study.best_value}")
```

### 处理其他异常

```python
try:
    safe_optimize(study, objective, n_trials=50000)
except Exception as e:
    print(f"实验出现错误: {e}")
    # 可以选择继续或停止
    manager.backup_database()  # 先备份
```

## 性能优化建议

### 1. 定期备份

```python
# 每1000次试验备份一次
if len(study.trials) % 1000 == 0:
    manager.backup_database()
```

### 2. 清理失败试验

```python
# 定期清理失败的试验以减少数据库大小
if len(study.trials) % 5000 == 0:
    deleted = manager.cleanup_failed_trials(study.study_name)
    print(f"清理了 {deleted} 个失败试验")
```

### 3. 监控存储空间

```python
import os

db_size = os.path.getsize(manager.db_path) / (1024 * 1024)  # MB
print(f"数据库大小: {db_size:.2f} MB")
```

## 最佳实践

1. **使用固定的研究名称**: 便于后续继续实验
2. **定期备份**: 防止数据丢失
3. **监控进度**: 使用日志记录实验状态
4. **错误处理**: 优雅地处理中断和异常
5. **资源管理**: 启用垃圾回收和内存管理

## 故障排除

### 数据库损坏

```python
# 如果数据库损坏，可以从备份恢复
import shutil

# 找到最新的备份文件
backup_files = glob.glob("optuna_storage/optuna_study.db.backup_*")
latest_backup = max(backup_files, key=os.path.getctime)

# 恢复备份
shutil.copy2(latest_backup, "optuna_storage/optuna_study.db")
```

### 研究不存在

```python
# 检查研究是否存在
if not manager.study_exists("my_experiment"):
    print("研究不存在，创建新的研究")
    study = create_robust_study("my_experiment")
```

### 权限问题

确保存储目录有写权限：

```bash
chmod 755 optuna_storage/
chmod 644 optuna_storage/optuna_study.db
```

## 总结

通过使用本配置系统，你可以：

- ✅ 安全地运行长时间优化实验
- ✅ 在任何时候中断和恢复实验
- ✅ 自动保存所有试验结果到硬盘
- ✅ 管理和分析实验数据
- ✅ 处理各种异常情况

这确保了你的量化交易策略优化实验的连续性和可靠性。 