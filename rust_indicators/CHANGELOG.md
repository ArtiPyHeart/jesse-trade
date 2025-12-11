# Changelog

本文件记录 pyrs-indicators 项目的重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [0.6.2] - 2025-12-11

### ⚡ CWT 性能优化 - Phase 2

**核心优化**:
- **线程本地缓冲区复用**: 使用 `thread_local!` 宏为每个 Rayon 工作线程预分配 `CwtBuffers`，减少 90%+ 堆分配
- **par_bridge → par_iter**: 改用更高效的 `par_iter()` 替代 `par_bridge()`，减少调度开销
- **FFT scratch buffer 复用**: 使用 `process_with_scratch()` 替代 `process()`，避免重复分配 scratch 缓冲区

**性能提升**:
- 大规模数据 (4000x128): **10.18ms → 8.39ms** (+18% 提升)
- 中等规模数据 (2000x64): **3.18ms → 2.68ms** (+16% 提升)
- 小规模数据略有开销（thread_local 初始化），但在实际生产场景中可忽略

**数值精度**:
- 与 PyWavelets 完美对齐（误差 < 6e-13）
- 4/4 CWT 正确性测试通过

**技术细节**:
- 新增 `CwtBuffers` 结构体封装所有可复用缓冲区
- 新增 `cwt_single_scale_with_buffers()` 内部函数
- `cwt_multi_scale()` 重构为使用 `thread_local!` + `RefCell` 模式

**测试与验证**:
- 新增 `scripts/benchmark_cwt_speed.py` 性能基准测试
- 修复 `test_cwt_correctness.py` 中 dB scale 计算（log10 vs 20*log10）
- VMD、NRBO 数值一致性测试全部通过
- 新增 `tests/test_nrbo_python_rust_consistency.py` NRBO 一致性测试

---

## [0.6.0] - 2025-10-26

### 🗑️ 移除功能

**移除 Ripser 拓扑数据分析模块**
- 原因：性能远低于 giotto-ph 的 ripser_parallel（实测速度远达不到预期）
- 移除内容：
  - 完整的 Rust 实现（`src/ripser/` 目录）
  - Python 包装（`pyrs_indicators/topology/` 模块）
  - 所有相关测试（`tests/ripser/`, `test_ripser_*.py`）
  - 基准测试（`benches/ripser_benches.rs`）
  - 文档（`RIPSER_ALGORITHM.md`）
- 替代方案：使用 giotto-ph 的 `ripser_parallel` 进行拓扑数据分析

### 🔧 架构优化

**项目聚焦核心优势**
- 专注于高性能信号处理指标：VMD、NRBO、CWT、FTI
- 移除未达标的拓扑数据分析模块，保持代码库质量
- 清理项目结构，提升维护性

### ⚠️ 破坏性变更

**不再支持拓扑数据分析**
- `pyrs_indicators.topology` 模块已完全移除
- 原有代码若依赖 `from pyrs_indicators.topology import ripser` 需迁移至 giotto-ph

**迁移指南**：
```python
# 旧代码（不再可用）
from pyrs_indicators.topology import ripser
result = ripser(points, max_dim=1, threshold=2.0)

# 新代码（推荐）
from gtda.externals import ripser_parallel
result = ripser_parallel(points, maxdim=1, thresh=2.0)
```

---

## [0.5.0] - 2025-10-25

### ✨ 新增功能

**🎯 拓扑数据分析模块 - Ripser 持久同调算法**
- 实现完整的 Vietoris-Rips 持久同调计算（H_0, H_1, H_2）
- 支持点云和距离矩阵输入
- 支持多种距离度量（euclidean, manhattan, chebyshev）
- 实现零长度 pair 过滤（`collapse_edges` 参数）
- 辅助函数：`filter_persistence`, `get_betti_numbers`

**Python API**:
```python
from pyrs_indicators.topology import ripser
result = ripser(points, max_dim=1, threshold=2.0)
```

### 🐛 Bug 修复

**修复三个 Critical Bugs（在 Codex GPT-5 协助下发现）**:

1. **H_0 边遍历顺序** (`core/cohomology.rs:250`)
   - 问题：降序遍历边导致 death 值完全错误
   - 修复：改为升序 filtration 顺序
   - 影响：H_0 差异从 ~1.9 降至 < 1e-7

2. **PivotTracker 大小** (`core/cohomology.rs:571-573`)
   - 问题：使用列数而非全局 simplex 索引范围
   - 修复：改为 `max_pivot = diameters.len() - 1`
   - 影响：H_1 在稀疏 filtration 中产生虚假特征
   - 发现方式：Codex 深度代码分析

3. **维度切片逻辑** (`core/algorithm.rs:383-393`) - **最严重**
   - 问题：假设排序后维度单调，使用错误的 offset 切片
   - 修复：改用 `filter(|s| s.vertices.len() - 1 == dim)`
   - 影响：H_1/H_2 对随机点云完全不可用
   - 发现方式：Codex 复现测试，明确指出"确定性 bug"

### ✅ 验证与测试

**数值一致性**（与 giotto-ph ripser_parallel 对比）:
- H_0: max_diff < 1e-7（完美匹配）
- H_1: Triangle/Circle/Two Circles/随机点云全部完美匹配

**集成测试**: 10/10 全部通过
- Triangle, Disconnected, Threshold, Distance matrix
- Validation, Filter persistence, Betti numbers
- Different metrics, Numerical stability, Edge cases

**性能基准**（MacBook Apple Silicon, threshold=2.0）:
- 20 点: < 1ms
- 50 点: 9ms
- 100 点: 118ms

### 📚 文档

- 创建完整的 `pyrs_indicators/topology/README.md`
- 创建 `src/ripser/README.md` 包含算法详解、bug 修复记录、未来改进方向
- Codex 协助总结文档

### 🙏 致谢

感谢 OpenAI Codex (GPT-5) 的两次关键协助（每次 600 秒超时）:
- 精准定位 PivotTracker 和维度切片 bug
- 澄清"实现差异" vs "确定性错误"
- 避免了重大生产事故

---

## [0.4.0] - 2025-10-24

### ✨ 新增功能

**🎨 Python API 接口层**
- 新增 `pyrs_indicators` Python 接口层
  - 分类组织：`ind_wavelets/`, `ind_decomposition/`, `ind_trend/`
  - 完整类型提示：`numpy.typing` 支持，IDE 自动补全
  - 参数验证：Fail Fast 原则，非法参数立即抛 `ValueError`
  - 详细文档：每个函数都有完整 docstring
  - 简化返回值：默认只返回常用结果，可选返回完整输出
- 迁移所有生产代码到新接口
  - `src/indicators/prod/emd/cls_vmd_indicator.py`
  - `src/indicators/prod/wavelets/cls_cwt_swt.py`
  - `src/indicators/prod/fti.py`
- 更新文档和测试使用新接口
- 移除旧 `_rust_indicators` 直接调用文档

### ⚠️ 破坏性变更
- 推荐使用新接口，旧接口 `_rust_indicators` 仍然可用但不推荐

**使用示例**:
```python
# 新接口（推荐）
from pyrs_indicators.ind_decomposition import vmd
from pyrs_indicators.ind_wavelets import cwt
from pyrs_indicators.ind_trend import fti

signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200))
modes = vmd(signal, alpha=2000.0, K=3)
```

---

## [0.3.1] - 2025-10-24

### 🔧 依赖升级

**PyO3 0.27 + numpy 0.27**
- 升级 PyO3: 0.26.0 → 0.27.1
- 升级 numpy: 0.26.0 → 0.27.0
- 升级 rayon: 1.10 → 1.11
- 支持 Python 3.14.0 final
- 所有指标验证通过（FTI/CWT/VMD 冒烟测试）
- 新增冒烟测试套件 (`rust_indicators/tests/`)

### ✅ 测试结果
```
✅ FTI 冒烟测试通过
✅ CWT 冒烟测试通过 (cmor1.5-1.0 wavelet)
✅ VMD 冒烟测试通过 (能量守恒验证)
```

### 📊 升级影响
- 无 breaking changes，平滑升级
- 代码无需修改，干净编译（零警告）
- 性能保持不变

---

## [0.2.3] - 2025-10-22

### ⚡ VMD 内存布局优化

**Array3 维度重排**:
- **Array3 维度重排**: u_hat_plus 从 `(niter, t, k)` 改为 `(niter, k, t)`
- **内存访问优化**: 将频率维度 `t` 放在最内层，提升 cache 局部性
- **全面索引更新**: 修改约 20 处索引访问

**性能提升**:
- Rolling window w256: **2.415s → 2.104s** (12.9% 提升)
- Rolling window w128: **0.295s → 0.271s** (8.1% 提升)
- k=4/k=3 倍率改善: **3.73x → 3.39x** (9.1% 改善)

**数值精度**:
- 完美保持数值对齐（误差 0.00e+00 < 1e-13）
- 所有正确性测试通过

**技术细节**:
- 新维度顺序使内层循环遍历连续内存
- 改善 cache line 利用率，减少 cache miss
- 对 k 较大的场景效果更明显

**依赖升级**:
- criterion 0.5 → 0.7（dev 依赖）
- rand 0.8 → 0.9（API 更新）
- 零编译警告，所有测试通过

**测试工具**:
- 新增 `profile_vmd.py` 和 `analyze_vmd_timing.py`
- 通过 profiling 科学分析瓶颈

---

## [0.2.2] - 2025-10-22

### 🔧 VMD 代码质量改进

**FFT Plan 架构优化**:
- 引入 `FftPlanCache` 类型和可选 cache 参数
- `utils.rs` 重构：`fft()` 和 `ifft()` 支持 optional cache 参数
- `core.rs` 优化：VMD 主函数预计算所需的 FFT plan sizes

**技术细节**:
- 使用 `Arc<HashMap<...>>` 实现零成本共享
- 预计算 2 种 FFT sizes: t (镜像信号) 和 t/2 (输出频谱)
- 保持完美数值对齐（误差 < 1e-13）

**性能影响**:
- 单次调用内 FFT/IFFT 复用 cache，减少重复创建开销
- Rolling window 场景暂未观察到显著提升
- 为后续跨调用 cache 复用奠定架构基础

**测试**:
- 新增 `test_vmd_correctness.py` 数值正确性测试
- 新增 `bench_vmd_opt.py` 和 `bench_vmd_rolling.py` 性能基准

---

## [0.2.1] - 2025-10-22

### ⚡ CWT 性能优化 - Phase 1

**核心优化**:
- **FFT Planner 复用**: 预计算并缓存所有 FFT plan，避免重复创建（核心优化，提升 150%+）
- **Scale 采样优化**: 预计算倒数避免重复除法（微小提升 2%）
- **向量化 diff 计算**: 使用 ndarray 切片操作替代手工循环（微小提升 0.5%）

**性能提升**:
- 平均加速: **5.8x → 10.2x** （提升 75.9%）
- 最大加速: **8x → 17.4x** （提升 117.5%）
- Scale 数量越多，优势越明显（100 scales 可达 17x）

**数值精度**:
- 完美保持与 PyWavelets 对齐（误差 < 2e-14）
- 所有测试通过（正确性 + 性能）

**技术细节**:
- 使用 HashMap 缓存不同 FFT size 的 plan
- Arc 包装实现并行环境下的零成本共享
- 保持零编译警告

---

## [0.2.0] - 2025-10-22

### ✨ CWT 实现

**新增功能**:
- CWT (Continuous Wavelet Transform) 完整实现
- 支持 Complex Morlet 小波 (cmor)
- 内置对称填充 (symmetric padding)
- 可选 verbose 模式用于开发调试

**性能优化**:
- 平均 5.8x 加速（vs PyWavelets）
- 大信号场景达到 7-8x 加速
- Rayon 并行化处理多 scale 计算

**数值精度**:
- 与 PyWavelets 完美对齐（误差 < 2e-14）
- 3/3 集成测试通过
- 零编译警告

---

## [0.1.0] - 2025-10-21

### ✨ 首次发布 - 生产就绪

**核心功能**:
- VMD (Variational Mode Decomposition) 实现
- NRBO (Newton-Raphson Boundary Optimization) 实现
- 完整的数值对齐验证 (误差 ~1e-15)
- 性能基准测试 (50-100x 加速)

**技术改进**:
- 升级到 PyO3 0.26 (使用现代 Bound API)
- 升级到 numpy 0.26 (零拷贝优化)
- 零编译警告，100% 测试通过

**项目清理**:
- 整理文档结构
- 配置 .gitignore 排除临时文件
- 代码审查通过，可安全集成

---

## 版本说明

**语义化版本控制**:
- **主版本号**：不兼容的 API 变更
- **次版本号**：向后兼容的新功能
- **修订号**：向后兼容的 bug 修复

**0.5.0 为何是次版本更新**:
- 新增拓扑数据分析模块（新功能）
- 向后兼容：不影响现有 VMD/CWT/FTI 用户
- 重大里程碑：但未改变现有 API

**未来规划**:
- v1.0.0：API 稳定，生产成熟
- v1.1：性能优化（缓存、apparent pairs）
- v1.2：功能扩展（代表圈、可视化）
- v2.0：应用场景（时间序列特征提取器、ML 集成）
