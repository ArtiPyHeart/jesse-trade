# jesse-trade 开发指南

## 核心原则
- **算法正确性**：计算错误可能造成重大财务损失，确保正确性优先于优化。
- **生产标准**：维持生产就绪代码质量，提交前必须审查。
- **科学方法**：应用先进数学/物理概念于交易。
- **MBTI人格化**：使用INTJ人格构思架构，思索科学问题；使用ISTJ人格执行具体开发任务。

## 项目结构
- `src/`：生产代码（bars/features/indicators/utils）
- `rust_indicators/`：rust化的高性能指标，目前实现了vmd指标的rust化
- `strategies/`：Jesse策略（每策略独立目录）- **项目运行入口**
- `research/`：离线研究实验（勿在生产导入）
- `extern/`：参考资料（勿导入）
- `tests/`：pytest测试
- `skills/`：Claude专用SKILL文档，定义特定领域的工作流程和标准

## 运行入口
该项目的运行入口为 `strategies/` 下的各项策略，需要启动jesse后运行，无法直接使用python运行

## 开发环境
```bash
./install.sh                      # 生产依赖
pip install -r requirements-dev.txt  # 开发依赖
```

## Jesse K线规范
- 格式：6列NumPy数组 `[timestamp, open, close, high, low, volume]`
- 转换：`numpy_candles_to_dataframe(candles)`
- 自定义K线：Dollar/Range/Entropy Bar，DEAP符号回归

## 特征→模型流程
完整的数据处理和预测流程（以 BinanceBtcDeapV1Voting 为例）：

```
原始 Candles → Fusion Bars → 特征计算 → 模型预测
```

### 特征计算流程
1. **计算原始特征**（SimpleFeatureCalculator）
   - 普通特征：直接用于模型
   - fracdiff 特征：需进一步处理
2. **SSM 推理**：fracdiff 特征 → `SSM.inference()` → SSM 特征
3. **特征拼接**：`[SSM特征, 原始特征]` → 完整特征 DataFrame
4. **模型预测**：从完整特征中选择 LGBM 需要的列 → `model.final_predict()` → 预测结果 (1/-1/0)

### Warmup vs Trading
- **Warmup**：批量计算 fracdiff 特征 → 逐行调用 `SSM.inference()` 更新状态（不保存输出）
- **Trading**：每次新 fusion bar 生成时，计算最新特征 → SSM.inference() → 拼接 → 模型预测
- **关键**：SSM 全程使用 `inference()`，不使用 `transform()`

## 指标开发
- 位置：稳定→`src/indicators/prod/`，实验→`experimental/`
- 规范：`sequential=True`返回全序列，`False`返回最新值
- 长度：用`np.nan`填充保持与K线一致
- 类指标：继承`_cls_ind.py`基类

### Rust高性能指标
- **快速使用**: `import _rust_indicators`，已实现VMD/NRBO（50-100x加速）
- **开发集成**: 阅读 `skills/RUST_INDICATORS_SKILL.md`

## 编码规范
- 内部函数用`_`前缀
- 数据操作用NumPy/Pandas
- **Fail Fast**：立即暴露错误而非静默处理
  - 使用`assert`拦截非法输入，避免宽泛`try/except`
  - 特征计算/指标计算失败应立即抛出异常，不记录错误后继续
  - 异常捕获仅用于明确可恢复的场景（如网络重试），不用于掩盖逻辑错误
- 简单测试用`if __name__ == "__main__"`，复杂测试放`tests/`
- EasyLanguage角度→Python弧度：用`src/utils/math_tools.py`
- **代码质量检查**：添加或修改 Python 代码后，使用 ruff 进行检查和格式化
  ```bash
  ruff check <file_or_dir>    # 代码检查（lint）
  ruff format <file_or_dir>   # 代码格式化
  ```

## SKILL文档使用
`skills/`目录包含针对特定任务的专业工作流程和标准：
- **触发条件**：当用户提及相关任务时，主动读取对应SKILL文档
- **现有SKILL**：
  - `MODEL_SCREENING_SKILL.md`：模型快速筛选与质量评估（置信度切片分析的前置步骤）
  - `CONFIDENCE_SLICE_ANALYSIS_SKILL.md`：置信度切片分析与过滤器配置
  - `RUST_INDICATORS_SKILL.md`：Rust高性能指标开发、集成与维护
- **使用原则**：严格遵循SKILL中定义的决策准则、输出格式和沟通方式

## 开发工具

### 代码检索（Auggie MCP）

**优先使用 `mcp__auggie-mcp__codebase-retrieval` 进行代码检索**，而非普通的 Grep/Glob 工具：

```python
# 使用 auggie mcp 进行语义化代码检索
mcp__auggie-mcp__codebase-retrieval(
    information_request="描述你要查找的代码功能或模式"
)
```

### 调用 Codex 获取技术指导

当遇到算法或架构问题需要专业建议时，可通过 mcp-shell-server 调用 codex：

```python
# 使用 mcp__mcp-shell-server__shell_exec 工具
mcp__mcp-shell-server__shell_exec(
    command='cd /Users/yangqiuyu/Github/jesse-trade && codex exec "你的问题"'
)
```

**使用场景**：
- 算法复杂度分析（如 Ripser 的 O(N³) 复杂度）
- 架构设计建议（如 apparent pairs 优化）
- 数学/物理概念解释
- 最佳实践咨询

**关键点**：
- 必须使用 `codex exec` 子命令（非交互模式）
- 需要在项目目录中运行（cd 到 repo）
- 从 stdout 直接获取答案
- 要求codex给出简明扼要的叙述
- **给予充分思考时间**：复杂问题需要长时间推理，应设置最长超时（600000ms）或使用后台任务（run_in_background: true）

**示例**：
```bash
cd /Users/yangqiuyu/Github/jesse-trade && codex exec "What is the time complexity of matrix reduction in persistent homology? Answer in 2 sentences."
```

## 关键提醒
- **MCP 服务依赖**：所有 MCP 工具调用（auggie、context7、chrome-devtools、mcp-shell-server 等）如果发现服务不存在或连接失败，**必须立即停止当前任务并提示用户配置对应的 MCP 服务**，不要尝试使用替代方案绕过
- 开发时使用 mcp context7 查看最新文档
- 代码检索优先使用 auggie mcp（`mcp__auggie-mcp__codebase-retrieval`），而非 Grep/Glob
- 如果WebFetch直接获取网页失败，可使用chrome devtools mcp打开网页并阅读内容（**优先使用 headless 模式**以提高效率，仅需可视化调试时使用常规模式）
- **遇到复杂算法/架构问题**：使用 codex exec 获取专业建议，避免盲目尝试
- 生产代码仅从`src/`导入
- 策略间保持独立，避免交叉依赖
- 功能实现后必须单元测试
- 测试文件命名：pytest用`test_`开头，直接运行避免`test_`前缀