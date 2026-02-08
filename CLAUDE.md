# jesse-trade — Legacy 参考仓库执行指南

> 本仓库是 **trade-on-nautilus** 迁移工程的子模块（`extern/jesse-trade`），定位为 **只读 legacy 参考**。
> 除非用户明确要求，不在此做新增功能或大规模重构。

## 迁移工程定位与核心原则

1. **只读参考，不做新开发**：本子模块的主要任务是定位/解释 legacy 实现，并产出迁移所需的事实描述与建议，供 trade-on-nautilus 主仓库消费。
2. **算法正确性**：计算错误可能造成重大财务损失，分析 legacy 逻辑时以正确性为第一优先级。
3. **科学方法**：应用先进数学/物理概念于交易，迁移时优先保留原始算法语义。
4. **小幅修正可接受**：仅允许为辅助分析而做的最小改动（例如临时 log/print）。完成验证后应还原，避免“参考仓库”长期漂移。

## Codex / Claude MCP 分工

| 角色 | 职责 | 备注 |
|------|------|------|
| **Codex（主 Agent）** | 事前规划（迁移方案设计、架构对比）、事后验收（Review 迁移结果） | 规划阶段产出明确的“单一目标”指令 |
| **Claude MCP（执行者）** | 按单一目标指令执行：定位代码、解释逻辑、提取事实、产出迁移建议、执行修正 | **不做方案选择**；指令不清时立即提问，不自行猜测 |

## Claude MCP 执行规则

- 收到的指令必须是**单一、明确的目标**，不接受模糊的“帮我想想怎么迁移”。
- 不做方案选择；如果指令包含多个可选方向，先要求调用方改成单一路径。
- 如果指令存在歧义或缺少关键信息，**立即提问**，不自行假设。
- 调用 MCP 工具（包括 Claude/其他 MCP、以及浏览/检索类工具）时，应给予充分运行时间：优先异步任务并通过状态查询/等待获取结果，避免超时或频繁打断导致失败或结果不完整。
- 输出格式：事实陈述 + 代码引用（文件路径 + 行号） + 迁移建议（如适用）。
- 不主动发起重构，不做超出指令范围的改动。

## Legacy 项目结构

> 以下为 jesse-trade 原始目录结构，供迁移参考。

- `src/`：生产代码（bars/features/indicators/utils）—— 原始生产代码的唯一导入源
- `rust_indicators/`：Rust 高性能指标
- `strategies/`：Jesse 策略（每策略独立目录）—— 原项目运行入口（需启动 jesse 后运行）
- `research/`：离线研究实验（勿在生产导入）
- `extern/`：参考资料（勿导入）
- `tests/`：pytest 测试（文件名用 `test_` 前缀）
- `archive/`：过时代码存档（除非用户指定，否则无需参考）

## 开发环境（仅用于本地分析/验证）

> 本子模块默认只读。仅在需要本地运行 legacy 代码进行验证时使用。

```bash
ruff check <file> && ruff format <file>
pytest tests/
```

- **conda 缺库**：直接 `conda install <pkg>` 或 `pip install <pkg>` 安装，不需要事先确认
- **第三方库最新用法**：使用 **Context7 MCP** 获取最新文档与示例

## Legacy 技术参考

### Jesse K 线规范

- 格式：6 列 NumPy 数组 `[timestamp, open, close, high, low, volume]`
- 转换：`numpy_candles_to_dataframe(candles)`
- 自定义 K 线：Dollar/Range/Entropy Bar，DEAP 符号回归

### 特征→模型流程

```
原始 Candles → Fusion Bars → 特征计算 → 模型预测
```

#### 模型类型与标签

| 前缀 | label_type | 标签方法 | threshold | 说明 |
|------|-----------|---------|-----------|------|
| `c_` | hard | `label_hard_state` | 0.5 | 二分类 (0/1) |
| `r_` | direction | `label_direction_force` | 0.0 | 回归 [-1,1]，对称分布 |
| `r2_` | directional_prob | `label_directional_prob` | 0.0 | 回归，非对称概率 |

- 模型命名格式：`{type}_L{lag}_N{pred_next}`，如 `c_L4_N3`、`r_L4_N2`、`r2_L5_N3`
- 相关文件：`flow_feature_select.py`（特征筛选）、`flow_model_build.py`（模型构建）
- 配置解析：`strategies/BinanceBtcDemoBar/models/config.py` 的 `model_name_to_params()`

#### 特征计算流程

1. **计算原始特征**（SimpleFeatureCalculator）：普通特征直接用于模型，fracdiff 特征需 SSM 处理
2. **SSM 推理**：fracdiff 特征 → `SSM.inference()` → SSM 特征
3. **特征拼接**：`[SSM 特征, 原始特征]` → 完整特征 DataFrame
4. **模型预测**：选择 LGBM 需要的列 → `model.final_predict()` → 预测结果 (1/-1/0)

Warmup vs Trading：Warmup 批量计算后逐行调用 `SSM.inference()` 更新状态；Trading 时每次新 bar 生成即计算→推理→预测。SSM 全程使用 `inference()`，不使用 `transform()`。

### 指标约定

- 位置：稳定 → `src/indicators/prod/`，实验 → `experimental/`
- `sequential=True` 返回全序列，`False` 返回最新值；用 `np.nan` 填充保持与 K 线长度一致
- 类指标：继承 `_cls_ind.py` 基类

### 编码规范（迁移时参考）

- 内部函数用 `_` 前缀，数据操作用 NumPy/Pandas
- 显式参数优先，避免 `*args`/`**kwargs`
- 配置对象用 Pydantic `BaseModel`，仅高频循环用 `dataclass`
- Fail Fast：`assert` 拦截非法输入，异常立即抛出
- 简单测试用 `if __name__ == "__main__"`，复杂测试放 `tests/`
- EasyLanguage 角度 → Python 弧度：`src/utils/math_tools.py`

## 关键提醒

- **本子模块默认只读**：所有新功能开发在 trade-on-nautilus 主仓库进行
- **MCP 服务依赖**：当需要的 MCP 服务不可用时，立即提示用户检查配置，不绕过或降级
- **Context7 查文档**：需要查第三方库最新用法时，用 Context7 MCP
- **conda 缺库直接装**：`conda install <pkg>` 或 `pip install <pkg>`，无需事先确认
- 从 jesse 获取真实 candles 的脚本必须在 **jesse-trade 项目根目录**运行，需读取 `.env` 配置
- 策略间保持独立，避免交叉依赖
