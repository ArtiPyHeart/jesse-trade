# jesse-trade 开发指南

## 核心原则
- **算法正确性**：计算错误可能造成重大财务损失，确保正确性优先于优化
- **生产标准**：维持生产就绪代码质量，提交前必须审查
- **科学方法**：应用先进数学/物理概念于交易
- **破坏性变更优先**：优先采用破坏性变更+变更后验证，减少技术债。除非用户明确要求，否则不考虑向后兼容
- **中文回复**：回应用户时, 优先使用简体中文回答

## 项目结构
- `src/`：生产代码（bars/features/indicators/utils）—— 生产代码仅从此导入
- `rust_indicators/`：Rust高性能指标（VMD/NRBO，50-100x加速），详见 `skills/RUST_INDICATORS_SKILL.md`
- `strategies/`：Jesse策略（每策略独立目录）—— **项目运行入口**，需启动jesse后运行
- `research/`：离线研究实验（勿在生产导入）
- `extern/`：参考资料（勿导入）
- `tests/`：pytest测试（文件名用`test_`前缀）
- `skills/`：Claude专用SKILL文档，用户提及相关任务时主动读取

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
- **禁止 `*args`/`**kwargs`**：使用显式参数或配置对象
- **配置对象用Pydantic**：`class Config(BaseModel): field: int = Field(default=1, ge=0)`，仅高频循环用`dataclass`
- **Fail Fast**：用`assert`拦截非法输入，异常立即抛出，仅在可恢复场景（如网络重试）捕获异常
- 简单测试用`if __name__ == "__main__"`，复杂测试放`tests/`
- EasyLanguage角度→Python弧度：用`src/utils/math_tools.py`

## SKILL文档
- `MODEL_SCREENING_SKILL.md`：模型快速筛选与质量评估
- `CONFIDENCE_SLICE_ANALYSIS_SKILL.md`：置信度切片分析与过滤器配置
- `RUST_INDICATORS_SKILL.md`：Rust高性能指标开发与集成

## 开发工具

### 代码检索
**优先使用 `mcp__auggie-mcp__codebase-retrieval`** 进行语义化代码检索，而非 Grep/Glob。

### Codex 技术指导
遇到算法/架构问题时，通过 mcp-shell-server 调用 codex 获取专业建议：
```bash
cd /Users/yangqiuyu/Github/jesse-trade && codex exec "问题描述"
```

**关键规则**：
- **每次调用独立**：codex 不知道对话上下文，必须在问题中提供完整背景信息（项目背景、相关代码、已尝试方案等）
- **必须附带文件路径**：让 codex 聚焦于具体文件和问题
- 复杂问题设置长超时（600000ms）或后台运行

```bash
# 示例
codex exec "Review src/models/deep_ssm/deep_ssm.py lines 200-300
Context: This implements ELBO computation for a VAE model.
Question: Is this numerically stable?"
```

## 关键提醒
- **MCP服务依赖**：auggie/context7/chrome-devtools/mcp-shell-server等服务不可用时，立即停止并提示用户配置，不要绕过
- 开发时用 context7 MCP 查看最新文档
- WebFetch失败时可用chrome devtools MCP（优先headless模式）
- 策略间保持独立，避免交叉依赖
- 功能实现后必须单元测试
