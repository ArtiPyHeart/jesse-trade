# AGENTS.md

## 目标
- 面向生产环境的交易项目, 计算错误可能造成重大财务损失, 正确性优先于优化
- 保持生产就绪代码质量, 提交前必须审查
- 采用科学方法推进建模与特征工程, 避免经验主义
- 优先接受破坏性变更并在变更后验证, 减少技术债, 除非用户明确要求兼容
- 回应用户时, 优先使用简体中文回答
- 当AGENTS.md更新时，需同步更新CLAUDE.md

## 项目结构
- `src/`: 生产代码 (bars/features/indicators/utils)
- `rust_indicators/`: Rust 高性能指标 (已实现 VMD)
- `strategies/`: Jesse 策略 (每策略独立目录), 项目运行入口
- `research/`: 离线研究实验 (不要在生产导入)
- `extern/`: 参考资料 (不要在生产导入)
- `tests/`: pytest 测试
- `skills/`: Claude 专用 SKILL 文档 (定义任务流程与标准)

## 运行入口
- 通过 `strategies/` 下策略启动 Jesse 后运行, 不能直接用 Python 执行
- 从 jesse 获取真实 candles 的程序/脚本/测试必须在项目根目录运行, 需要读取 .env 配置, 在其他目录会导致配置无法识别而失败

## Jesse K 线规范
- 格式: 6 列 NumPy 数组 `[timestamp, open, close, high, low, volume]`
- 转换: `numpy_candles_to_dataframe(candles)`
- 自定义 K 线: Dollar/Range/Entropy Bar, DEAP 符号回归

## 特征到模型流程
```
原始 Candles -> Fusion Bars -> 特征计算 -> 模型预测
```

### 特征计算
- 原始特征: `SimpleFeatureCalculator` (普通特征直接用, fracdiff 需进一步处理)
- SSM 推理: fracdiff 特征 -> `SSM.inference()` -> SSM 特征
- 特征拼接: `[SSM 特征, 原始特征]` -> 完整特征 DataFrame
- 模型预测: 选择 LGBM 需要的列 -> `model.final_predict()` -> 预测 (1/-1/0)

### Warmup vs Trading
- Warmup: 批量计算 fracdiff 特征, 逐行调用 `SSM.inference()` 更新状态 (不保存输出)
- Trading: 新 fusion bar 生成时计算最新特征 -> `SSM.inference()` -> 拼接 -> 模型预测
- 关键: SSM 全程使用 `inference()`, 不使用 `transform()`

## 指标开发
- 位置: 稳定指标放 `src/indicators/prod/`, 实验指标放 `experimental/`
- 规范: `sequential=True` 返回全序列, `False` 返回最新值
- 长度: 使用 `np.nan` 填充, 保持与 K 线长度一致
- 类指标: 继承 `_cls_ind.py` 基类

### Rust 高性能指标
- 快速使用: `import _rust_indicators` (已实现 VMD/NRBO, 50-100x 加速)
- 开发集成: 阅读 `skills/RUST_INDICATORS_SKILL.md`

## 编码规范
- 内部函数使用 `_` 前缀
- 数据操作使用 NumPy/Pandas
- 禁止使用 `*args` 和 `**kwargs`, 使用显式参数或配置对象
- 配置对象优先使用 `pydantic.BaseModel`, 仅性能敏感内部结构 (如高频循环) 用 `dataclass`
- Fail Fast: 用 `assert` 拦截非法输入, 避免宽泛 `try/except`
- 异常仅用于可恢复场景, 特征/指标失败应直接抛出
- 简单测试可用 `if __name__ == "__main__"`, 复杂测试放 `tests/`
- EasyLanguage 角度到 Python 弧度: 使用 `src/utils/math_tools.py`
- 代码质量: Python 代码改动后使用 ruff
  - `ruff check <file_or_dir>`
  - `ruff format <file_or_dir>`

## SKILL 文档使用
- 触发条件: 用户提及相关任务时主动读取对应 SKILL
- 现有 SKILL:
  - `skills/MODEL_SCREENING_SKILL.md`
  - `skills/CONFIDENCE_SLICE_ANALYSIS_SKILL.md`
  - `skills/RUST_INDICATORS_SKILL.md`
- 使用原则: 严格遵循 SKILL 的决策准则、输出格式和沟通方式

## 开发工具
### 代码检索 (Auggie MCP)
- 优先使用 `mcp__auggie-mcp__codebase-retrieval` 进行语义化检索
- 避免使用 grep/glob 做代码检索

## 关键提醒
- MCP 工具不可用时立即停止并提示用户配置服务
- 优先使用 context7 获取最新文档
- WebFetch 获取网页失败时, 使用 chrome-devtools mcp (优先 headless)
- 生产代码仅从 `src/` 导入
- 策略间保持独立, 避免交叉依赖
- 新功能必须配套单元测试
- pytest 测试文件以 `test_` 开头, 直接运行脚本避免 `test_` 前缀

## 环境与依赖
- 安装生产依赖: `./install.sh`
- 安装开发依赖: `./install.sh --dev`
