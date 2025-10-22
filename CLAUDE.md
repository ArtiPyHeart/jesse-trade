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

### Python 运行规范
- **PYTHONPATH 设置**：在项目中运行 Python 代码时，总是设置 `PYTHONPATH=.`（项目根目录）来保证 `import src` 等的正确性
  ```bash
  # 正确做法
  PYTHONPATH=. python tests/test_example.py
  PYTHONPATH=. python -m pytest tests/

  # 错误做法（可能导致 ModuleNotFoundError）
  python tests/test_example.py
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
- 使用`assert`拦截非法输入，避免宽泛`try/except`
- 简单测试用`if __name__ == "__main__"`，复杂测试放`tests/`
- EasyLanguage角度→Python弧度：用`src/utils/math_tools.py`

## SKILL文档使用
`skills/`目录包含针对特定任务的专业工作流程和标准：
- **触发条件**：当用户提及相关任务时，主动读取对应SKILL文档
- **现有SKILL**：
  - `CONFIDENCE_SLICE_ANALYSIS_SKILL.md`：置信度切片分析与过滤器配置
  - `RUST_INDICATORS_SKILL.md`：Rust高性能指标开发、集成与维护
- **使用原则**：严格遵循SKILL中定义的决策准则、输出格式和沟通方式

## 关键提醒
- 开发时使用mcp context7 查看最新文档，如果mcp调用失败，停下来提示用户先配置
- 生产代码仅从`src/`导入
- 策略间保持独立，避免交叉依赖
- 功能实现后必须单元测试
- 测试文件命名：pytest用`test_`开头，直接运行避免`test_`前缀