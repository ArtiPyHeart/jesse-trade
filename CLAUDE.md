# jesse-trade 开发指南

## 核心原则
- **算法正确性**：计算错误可能造成重大财务损失，确保正确性优先于优化
- **生产标准**：维持生产就绪代码质量，提交前必须审查
- **科学方法**：应用先进数学/物理概念于交易
- **破坏性变更优先**：优先采用破坏性变更+变更后验证，减少技术债。除非用户明确要求，否则不考虑向后兼容

## 项目结构
- `src/`：生产代码（bars/features/indicators/utils）—— 生产代码仅从此导入
- `rust_indicators/`：Rust高性能指标（VMD/NRBO，50-100x加速）
- `strategies/`：Jesse策略（每策略独立目录）—— **项目运行入口**，需启动jesse后运行
- `research/`：离线研究实验（勿在生产导入）
- `extern/`：参考资料（勿导入）
- `tests/`：pytest测试（文件名用`test_`前缀）
- `archive/`：过时代码存档（除非用户指定，否则无需参考）
- `.claude/skills/`：Claude专用SKILL文档，用户提及相关任务时自动加载

## 开发环境
```bash
./install.sh                         # 生产依赖
./install.sh --dev                   # 开发依赖（在生产环境基础上增量安装）
ruff check <file> && ruff format <file>  # 代码质量检查
```

## Jesse K线规范
- 格式：6列NumPy数组 `[timestamp, open, close, high, low, volume]`
- 转换：`numpy_candles_to_dataframe(candles)`
- 自定义K线：Dollar/Range/Entropy Bar，DEAP符号回归

## 特征→模型流程
```
原始 Candles → Fusion Bars → 特征计算 → 模型预测
```

### 模型类型与标签
| 前缀 | label_type | 标签方法 | threshold | 说明 |
|------|-----------|---------|-----------|------|
| `c_` | hard | `label_hard_state` | 0.5 | 二分类 (0/1) |
| `r_` | direction | `label_direction_force` | 0.0 | 回归 [-1,1]，对称分布 |
| `r2_` | directional_prob | `label_directional_prob` | 0.0 | 回归，非对称概率 |

- 模型命名格式：`{type}_L{lag}_N{pred_next}`，如 `c_L4_N3`、`r_L4_N2`、`r2_L5_N3`
- 相关文件：`flow_feature_select.py`（特征筛选）、`flow_model_build.py`（模型构建）
- 配置解析：`strategies/BinanceBtcDemoBar/models/config.py` 的 `model_name_to_params()`

### 特征计算流程
1. **计算原始特征**（SimpleFeatureCalculator）：普通特征直接用于模型，fracdiff特征需SSM处理
2. **SSM 推理**：fracdiff特征 → `SSM.inference()` → SSM特征
3. **特征拼接**：`[SSM特征, 原始特征]` → 完整特征DataFrame
4. **模型预测**：选择LGBM需要的列 → `model.final_predict()` → 预测结果(1/-1/0)

**Warmup vs Trading**：Warmup批量计算后逐行调用`SSM.inference()`更新状态；Trading时每次新bar生成即计算→推理→预测。SSM全程使用`inference()`，不使用`transform()`。

## 指标开发
- 位置：稳定→`src/indicators/prod/`，实验→`experimental/`
- 规范：`sequential=True`返回全序列，`False`返回最新值；用`np.nan`填充保持与K线一致
- 类指标：继承`_cls_ind.py`基类

## 编码规范
- 内部函数用`_`前缀，数据操作用NumPy/Pandas
- **除非明确说明，否则不要使用 `*args`/`**kwargs`**：优先使用显式参数或配置对象
- **配置对象用Pydantic**：`class Config(BaseModel): field: int = Field(default=1, ge=0)`，仅高频循环用`dataclass`
- **Fail Fast**：用`assert`拦截非法输入，异常立即抛出，仅在可恢复场景（如网络重试）捕获异常
- 简单测试用`if __name__ == "__main__"`，复杂测试放`tests/`
- EasyLanguage角度→Python弧度：用`src/utils/math_tools.py`

## SKILL文档（位于 `.claude/skills/`）
- `rust-indicators-development`：Rust高性能指标开发与集成
- `find-best-fusion-bar`：自定义趋势轴（Fusion Bar）开发与优化，触发短语："构建新的趋势轴"/"寻找最佳fusion bar"/"开发自定义轴"

## 开发工具

### Codex 技术指导
遇到算法/架构问题时，通过 **Codex MCP** 调用 GPT-5 获取专业建议。

**固定配置**（jesse-trade 项目专用）：
| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `gpt-5.2-codex` | 始终使用最先进模型 |
| `model_reasoning_effort` | `xhigh` | 最高推理强度（问的都是难题） |
| `model_reasoning_summary` | `none` | 不展示推理过程 |
| `cwd` | `/Users/yangqiuyu/Github/jesse-trade` | 项目根目录 |

**调用模式**：
| 场景 | 工具 | 是否需要完整背景 | 说明 |
|------|------|------------------|------|
| 新问题 | `mcp__codex__codex` | ✅ 需要 | 首次调用必须提供完整上下文 |
| 同 thread 追问 | `mcp__codex__codex-reply` | ❌ 不需要 | Codex 保留对话历史，直接追问即可 |
| 跨 session 继续 | `mcp__codex__codex` | ✅ 需要 | threadId 可能过期，需重新提供背景 |

**关键规则**：
- **新开 thread 时提供完整背景**：项目背景、相关代码、已尝试方案等，并附带文件路径让 Codex 聚焦
- **同一 thread 内直接追问**：首次调用返回 `threadId`，后续用 `codex-reply` + `threadId` 继续对话，无需重复背景
- **跨 session 持久化**：将重要讨论结论写入 markdown 文件（如 `docs/codex_context_<topic>.md`），方便后续 session 引用

**MCP 调用示例**（基于真实开发场景）：

**场景：验证 pred_next 延迟逻辑的正确性**

```
# 1. 首次调用 - 提供完整背景和代码片段
mcp__codex__codex(
  prompt="""
项目背景：jesse-trade 量化交易系统，使用多模型投票进行交易决策。

核心逻辑：每个模型有 pred_next 参数，表示预测需要延迟多少步才生效。
- c_L4_N3 的 pred_next=3，意味着 bar[i] 的预测在 bar[i+3] 时才用于交易
- 这是为了避免使用未来信息（look-ahead bias）

当前实现（flow_backtest_vectorized.py lines 180-210）：
```python
# 应用 pred_next 延迟
delayed_preds = {}
for m in models:
    mc = model_containers[m]
    delay = mc.pred_next
    delayed = np.zeros(n_bars, dtype=np.int32)
    if delay < n_bars:
        delayed[delay:] = raw_preds[m][:-delay] if delay > 0 else raw_preds[m]
    delayed_preds[m] = delayed
```

问题：这个延迟逻辑是否正确？是否真的实现了"bar[i] 的预测在 bar[i+delay] 时生效"？
""",
  model="gpt-5.2-codex",
  cwd="/Users/yangqiuyu/Github/jesse-trade",
  config={"model_reasoning_effort": "xhigh", "model_reasoning_summary": "none"}
)
# 返回: threadId="019c07d2-3d61-..." + 详细分析

# 2. 追问 - Codex 保留上下文，直接简洁提问
mcp__codex__codex-reply(
  threadId="019c07d2-3d61-...",
  prompt="如果我想在逐步回测中实现同样的延迟逻辑，用 deque 队列是否是最佳选择？"
)

# 3. 继续追问 - 请求具体实现
mcp__codex__codex-reply(
  threadId="019c07d2-3d61-...",
  prompt="能否给出 deque 实现的代码示例？要求与向量化版本产生完全相同的结果。"
)
```

**典型使用场景**：
- 算法正确性验证（如上例的延迟逻辑、止损触发顺序）
- 数学原理解释（Hurst 指数计算、分形维度、ELBO 推导）
- 性能优化建议（NumPy 向量化、Rust 集成方案）
- 架构设计决策（SSM inference vs transform 的选择）

## 关键提醒
- **MCP服务依赖**：当需要的 MCP 服务不可用时，立即提示用户检查配置，不要绕过或降级处理
- 开发时用 context7 MCP 查看最新文档
- **浏览器操作分工**：
  - 正常网页交互（阅读网页、填表、点击等）→ 优先使用 claude-in-chrome 插件（mcp__claude-in-chrome__* 工具）
  - 开发调试相关（查看 console、network、DOM 调试等）→ 使用 chrome-devtools MCP
- 从 jesse 获取真实 candles 的程序/脚本/测试必须在项目根目录运行，需读取 .env 配置，否则会导致配置无法识别而失败
- 策略间保持独立，避免交叉依赖
- 功能实现后必须单元测试
