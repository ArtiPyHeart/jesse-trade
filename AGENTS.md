# jesse量化交易仓库指南

## 项目结构与模块
- `src/`：生产代码。
  - `bars/`：Dollar/Range/Entropy 自定义K线；DEAP 融合。
  - `features/`：特征计算与特征选择。
  - `indicators/`：指标（`prod/` 生产可用，`experimental/` 实验性）。
  - `utils/`、`data_process/`：数学工具与变换。
- `strategies/`：Jesse 策略（每个策略一个目录）。
- `tests/` 与 `src/**/tests/`：pytest 测试。
- `docker/`：运行时栈（Jesse、Postgres、Redis、pgbouncer）。
- `research/`、`archive/`：离线笔记本/实验（勿在生产中导入）。
- `extern/`：参考/第三方代码（勿导入）。

## 环境、构建与运行
- 环境：`conda create -n jesse python=3.11 -y && conda activate jesse`。
- 安装基本依赖：`bash ./install.sh`。安装开发依赖：`pip install -r requirements-dev.txt`。
- 运行jesse服务：`bash ./run.sh`（Linux 上先重启 pgbouncer，再执行 `jesse run`）。

## Jesse K线与指标
- K线形状：二维 NumPy，6 列 `[timestamp, open, close, high, low, volume]`。
- 指标规则：`sequential=True` 返回完整序列；`False` 返回最新值。必要时用 `np.nan` 填充以保持长度一致。
- 放置位置：稳定指标放 `src/indicators/prod/`；实验性放 `src/indicators/experimental/`。
- 数学辅助：使用 `src/utils/math_tools.py`（如 `deg_sin/deg_cos/deg_tan`、`dt/lag/std`）。优先使用向量化 NumPy/Numba。

## 测试规范
- 运行：`pytest -q`（示例：`tests/test_merge.py::test_np_merge_bars_consistency`、`src/features/flexible_feature_calculator/tests`）。
- 约定：文件命名为 `test_*.py`；当缺少 `data/*.npy` 时使用可复现的模拟数据。
- 覆盖点：指标的 `sequential` 行为，以及 Bar 构建的边界情况。

## 编码风格与命名
- 遵循 PEP-8，4 空格缩进；使用类型注解与简洁 docstring。
- 命名：模块/变量 `snake_case`，类 `CamelCase`，常量 `UPPER_SNAKE_CASE`。
- 倡导早期 `assert` 暴露非法输入；避免使用宽泛的 `try/except` 掩盖错误。
- 生产代码仅从 `src/` 导入；禁止从 `research/` 或 `extern/` 导入。
- 使用mcp context7查阅各代码库最新文档。

## 提交与 Pull Request
- 提交信息：祈使句主题；可选前缀（`feat:`、`fix:`、`refactor:`）。
- PR 内容：变更说明、动机、关联 issue、测试覆盖；相关图表/产物请放入 `outputs/`。
- 本地验证：确保单元测试通过。

## 安全与配置
- 机密仅存放于 `.env`（参考 `.env.example`）。
- 修改 `docker/docker-compose.yml` 与 `pgbouncer/*` 时需谨慎，避免破坏本地开发体验。
