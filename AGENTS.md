# AGENTS.md

## 定位与目标

本仓库（`extern/jesse-trade`）是 **trade-on-nautilus 迁移工程的 legacy 参考源**。其代码、策略、特征与模型流程用于理解旧系统行为，指导向 NautilusTrader 的迁移。

- **只读参考**：除非 owner 明确要求，否则不在此仓库做新增功能或大规模重构。所有新开发发生在主仓库 `trade-on-nautilus`。
- 正确性优先：计算错误可能造成重大财务损失，分析 legacy 逻辑时以正确性为第一优先级。
- 科学方法：推进建模与特征工程时，优先可证伪/可复现的方法，避免经验主义。
- 优先使用**简体中文**回复与沟通（代码标识符、技术术语保持原文）。
- 将本文件视为持续演进的经验精华：当发现新的最佳实践、多次失败教训或 owner 强调“应该记住”的指示时，应及时更新。
- 若文档约束与用户指令出现矛盾，必须先停下来说明冲突点并让用户决策，之后更新文档消除矛盾。

## AI 协作规则

### 单一目标指令

- 每次执行一个明确目标。收到多目标、含糊或“给你几个选项你自己挑”的指令时，不做选择，立即提问让 owner 澄清为单一路径。
- 执行者不替 owner 做决定：遇到“方案 A 还是 B”，只输出事实与 tradeoff，并请 owner 拍板。
- 指令不清就提问，绝不猜测意图。

### MCP 工具调用

- 调用 MCP 工具（Claude MCP、Codex 等）时应给予其**充分运行时间**。优先异步任务并通过状态查询/等待获取结果，避免因超时或频繁打断导致失败或结果不完整。
- 需要查第三方库最新用法/文档时，使用 **Context7**（`resolve-library-id` → `query-docs`），避免依赖过时信息。

### 环境与依赖

- conda 环境缺少依赖库时，直接 `conda install` 或 `pip install` 安装，无需额外确认。

## 项目结构（legacy）

> 以下描述的是 legacy jesse-trade 仓库结构，仅用于迁移参考。

- `src/`：核心代码（bars/features/indicators/utils）
- `rust_indicators/`：Rust 高性能指标（已实现 VMD/NRBO 等）
- `strategies/`：Jesse 策略（每策略独立目录），Jesse 运行入口
- `research/`：离线研究实验（不在生产导入）
- `extern/`：参考资料（不在生产导入）
- `tests/`：pytest 测试
- `archive/`：过时代码存档（除非用户指定，否则无需参考）
- `.claude/skills/`：Claude 专用 SKILL 文档（若使用 Claude Code 工作流时参考；迁移工程一般以主仓库规则为准）

### 运行入口（legacy）

- 通过 `strategies/` 下策略启动 Jesse 后运行，不能直接用 Python 执行。
- 从 Jesse 获取真实 candles 的程序/脚本/测试必须在项目根目录运行，需要读取 `.env` 配置，在其他目录会导致配置无法识别而失败。

## Jesse K 线规范（legacy）

- 格式：6 列 NumPy 数组 `[timestamp, open, close, high, low, volume]`
- 转换：`numpy_candles_to_dataframe(candles)`
- 自定义 K 线：Dollar/Range/Entropy Bar、DEAP 符号回归

## 特征到模型流程（legacy）

```text
原始 Candles -> Fusion Bars -> 特征计算 -> 模型预测
```

### 模型类型与标签

| 前缀 | label_type | 标签方法 | threshold | 说明 |
|------|-----------|---------|-----------|------|
| `c_` | hard | `label_hard_state` | 0.5 | 二分类 (0/1) |
| `r_` | direction | `label_direction_force` | 0.0 | 回归 [-1,1]，对称分布 |
| `r2_` | directional_prob | `label_directional_prob` | 0.0 | 回归，非对称概率 |

- 模型命名：`{type}_L{lag}_N{pred_next}`（如 `c_L4_N3`、`r_L4_N2`、`r2_L5_N3`）
- 特征筛选：`flow_feature_select.py`
- 模型构建：`flow_model_build.py`
- 配置解析：`strategies/BinanceBtcDemoBar/models/config.py`

### 特征计算

- 原始特征：`SimpleFeatureCalculator`（普通特征直接用，fracdiff 需进一步处理）
- SSM 推理：fracdiff 特征 → `SSM.inference()` → SSM 特征
- 特征拼接：`[SSM 特征, 原始特征]` → 完整特征 DataFrame
- 模型预测：选择 LGBM 需要的列 → `model.final_predict()` → 预测 (1/-1/0)

### Warmup vs Trading

- Warmup：批量计算 fracdiff 特征，逐行调用 `SSM.inference()` 更新状态（不保存输出）
- Trading：新 fusion bar 生成时计算最新特征 → `SSM.inference()` → 拼接 → 模型预测
- 关键：SSM 全程使用 `inference()`，不使用 `transform()`

## 指标开发（legacy）

- 位置：稳定指标放 `src/indicators/prod/`，实验指标放 `experimental/`
- 规范：`sequential=True` 返回全序列，`False` 返回最新值
- 长度：使用 `np.nan` 填充，保持与 K 线长度一致
- 类指标：继承 `_cls_ind.py` 基类

## 编码规范（legacy，迁移时参考）

> 新代码遵循主仓库 `trade-on-nautilus/AGENTS.md`。

- 内部函数使用 `_` 前缀
- 数据操作使用 NumPy/Pandas
- 除非明确说明，否则不要使用 `*args` 和 `**kwargs`，优先使用显式参数或配置对象
- 配置对象优先使用 `pydantic.BaseModel`，仅性能敏感内部结构（如高频循环）用 `dataclass`
- Fail Fast：用 `assert` 拦截非法输入，避免宽泛 `try/except`
- 异常仅用于可恢复场景，特征/指标失败应直接抛出

## 测试与代码质量（legacy）

- 简单测试可用 `if __name__ == "__main__"`，复杂测试放 `tests/`
- pytest 测试文件以 `test_` 开头，直接运行脚本避免 `test_` 前缀
- Python 代码改动后使用 ruff：
  - `ruff check <file_or_dir>`
  - `ruff format <file_or_dir>`

## 安装（legacy）

- 安装生产依赖：`./install.sh`
- 安装开发依赖：`./install.sh --dev`
