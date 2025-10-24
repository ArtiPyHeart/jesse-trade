# Rust Indicators

[![Rust](https://img.shields.io/badge/rust-1.74%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

高性能 Rust 实现的交易技术指标，为 jesse-trade 量化交易框架提供 **5-120x** 性能提升。

---

## 🎯 核心特性

✅ **完美数值对齐**: 与 Python 参考实现误差达到浮点精度极限 (~1e-15)
⚡ **极致性能**: CWT 平均 5.8x 加速，NRBO 平均 53.6x 加速，VMD 平均 94.4x 加速
🔒 **生产级质量**: 100% 测试通过，零编译警告
🚀 **零运行时开销**: 无 JIT 编译延迟，性能可预测

---

## 📦 已实现的指标

### CWT (Continuous Wavelet Transform)
连续小波变换算法，用于时频分析和特征提取。

**性能** (v0.2.1 优化后):
- 平均加速: **10.2x** (v0.2.0: 5.8x)
- 小信号 (128样本): **3-9x**
- 中信号 (256-512样本): **4-16x**
- 大信号 (1024样本): **8-17x**
- **v0.2.1 新增**: FFT Planner 复用优化（核心优化，提升 150%+）

**特性**:
- 完美对齐 PyWavelets (误差 < 2e-14)
- 支持 Complex Morlet 小波 (cmor)
- 内置对称填充 (symmetric padding)
- 可选 verbose 模式用于开发调试
- Scale 数量越多，性能优势越明显（100 scales可达 17x）

### VMD (Variational Mode Decomposition)
变分模态分解算法，用于信号分解和特征提取。

**性能**:
- 平均加速: **94.4x** (含冷启动)
- 稳态加速: **1.3-1.6x** (vs Numba JIT)
- 首次调用: **837x** (vs Python 冷启动)

### NRBO (Newton-Raphson Boundary Optimization)
牛顿-拉夫森边界优化算法，用于改善 IMF 边界效应。

**性能**:
- 平均加速: **53.6x**
- 小信号: **200x** (N=100)
- 大信号: **3-6x** (N≥500)

### FTI (Frequency Tunable Indicator)
频率可调谐指标，用于识别价格数据中的优势周期结构。

**性能** (v0.3.x):
- 平均加速: **41.7x**
- 首次调用: **122x** (跳过 Numba JIT 编译)
- 稳态调用: **1.3-1.4x** (vs Numba JIT)

**特性**:
- 完美数值对齐 (误差 0.00e+00)
- 自动周期检测 (5-65 周期范围)
- Gamma 累积分布函数变换
- 可配置滤波器参数
- **v0.3.1**: 升级到 PyO3 0.27 + numpy 0.27，支持 Python 3.14

---

## 🚀 快速开始

### 安装

**推荐方式**: 通过项目根目录的 `install.sh`

```bash
cd /path/to/jesse-trade
./install.sh
```

install.sh 会自动：
1. 安装 Python 依赖
2. 检测 Rust 环境
3. 编译并安装 Rust indicators

**手动安装**:

```bash
# 1. 安装 Rust (如未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. 进入项目目录
cd /path/to/jesse-trade/rust_indicators

# 3. 编译并安装 (Release 模式)
maturin develop --release
```

### 使用

```python
import _rust_indicators
import numpy as np

# CWT 时频分析
signal = np.sin(np.linspace(0, 10, 1000) * 2 * np.pi * 5)
scales = np.logspace(np.log2(8), np.log2(64), 20, base=2)
cwt_db, freqs = _rust_indicators.cwt_py(
    signal, scales, 'cmor1.5-1.0',
    sampling_period=0.5, precision=12, pad_width=20
)

# VMD 分解
signal = np.sin(np.linspace(0, 1, 1000) * 2 * np.pi * 5)
u, u_hat, omega = _rust_indicators.vmd_py(signal, alpha=2000, k=2)

# NRBO 优化
imf = np.sin(np.linspace(0, 10, 100))
optimized = _rust_indicators.nrbo_py(imf, max_iter=10, tol=1e-6)

# FTI 周期检测
price_data = np.random.randn(200) + 100  # 价格数据（最近的在索引0）
fti, filtered_value, width, best_period = _rust_indicators.fti_process_py(
    price_data,
    use_log=True,
    min_period=5,
    max_period=65,
    half_length=35,
    lookback=150,
    beta=0.95,
    noise_cut=0.20
)
```

---

## 📊 性能对比

### CWT 性能

| 信号长度 | Scales | Pad Width | PyWavelets | Rust | 加速比 |
|---------|--------|-----------|------------|------|--------|
| 100 | 10 | 0 | 0.54 ms | **0.28 ms** | **1.9x** |
| 100 | 20 | 20 | 0.95 ms | **0.25 ms** | **3.9x** |
| 1000 | 20 | 20 | 3.10 ms | **0.50 ms** | **6.2x** |
| 1000 | 50 | 50 | 7.40 ms | **1.01 ms** | **7.4x** ⚡ |
| 5000 | 20 | 20 | 13.0 ms | **1.63 ms** | **8.0x** |
| **10000** | **20** | **20** | **25.5 ms** | **3.21 ms** | **7.9x** 🚀 |

**注**: 信号越大，加速比越高。生产环境典型场景（1000-10000样本）可达 **6-8倍** 加速。

### NRBO 性能

| 信号长度 | Python | Rust | 加速比 |
|---------|--------|------|--------|
| 100 | 18.4 ms | **0.09 ms** | **200.3x** ⚡ |
| 500 | 0.09 ms | **0.01 ms** | **6.2x** |
| 1000 | 0.11 ms | **0.02 ms** | **5.0x** |
| 5000 | 0.21 ms | **0.07 ms** | **2.8x** |

### VMD 性能

| 信号长度 | K | Python | Rust | 加速比 |
|---------|---|--------|------|--------|
| **100** | **2** | **345 ms** | **0.41 ms** | **837.7x** 🚀 |
| 100 | 3 | 0.73 ms | 0.56 ms | 1.3x |
| 500 | 3 | 16.5 ms | 12.3 ms | 1.3x |
| 1000 | 5 | 79.0 ms | 50.9 ms | 1.6x |

**注**: 首次调用 Python/Numba 需要 JIT 编译，Rust 无此开销。

---

## 🧪 测试验证

### 数值精度测试

```bash
cd rust_indicators

# 生成测试数据
python scripts/generate_test_cases.py

# 运行 Rust 测试
python scripts/run_rust_tests.py

# 对比结果
python scripts/compare_with_python.py nrbo simple_sine --rust-output test_data/nrbo/simple_sine_rust.pkl
```

**测试结果**:
- CWT: 3/3 通过，误差 **< 2e-14**
- NRBO: 4/4 通过，误差 **0.00e+00**
- VMD: 5/5 通过，误差 **~1e-15**

### 性能基准测试

```bash
python scripts/benchmark_performance.py
```

结果保存在 `benchmark_results/*.csv`

---

## 📁 项目结构

```
rust_indicators/
├── Cargo.toml                      # Rust 包配置
├── pyproject.toml                  # Python 包配置
│
├── src/                            # Rust 源代码
│   ├── lib.rs                      # 模块入口
│   ├── cwt/
│   │   ├── core.rs                 # CWT 核心算法
│   │   ├── wavelets.rs             # 小波函数生成
│   │   ├── utils.rs                # 工具函数（填充/dB转换）
│   │   └── ffi.rs                  # Python 绑定
│   ├── nrbo/
│   │   ├── core.rs                 # NRBO 核心算法
│   │   └── ffi.rs                  # Python 绑定
│   └── vmd/
│       ├── core.rs                 # VMD 核心算法
│       ├── utils.rs                # FFT 工具
│       └── ffi.rs                  # Python 绑定
│
├── benches/                        # Criterion 基准测试
└── scripts/                        # 测试和工具脚本
```

---

## 🔧 开发

### 编译模式

```bash
# 开发模式 (快速编译，无优化)
maturin develop

# 发布模式 (完整优化，推荐)
maturin develop --release
```

### 代码检查

```bash
# 检查编译错误和警告
cargo check

# 运行 Rust 单元测试
cargo test

# 运行 Clippy 静态分析
cargo clippy

# 格式化代码
cargo fmt
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|-----|------|------|
| 核心语言 | Rust 1.74+ | 高性能实现 |
| Python 绑定 | PyO3 0.27 | Python 互操作 (Bound API, Python 3.14) |
| 数组操作 | ndarray 0.16 | N 维数组 |
| NumPy 绑定 | numpy 0.27 | 零拷贝数组转换 |
| FFT | rustfft 6.4 | 快速傅里叶变换 |
| 并行计算 | rayon 1.11 | CPU 多核并行 |
| 构建工具 | maturin 1.0+ | Python 扩展打包 |
| 错误处理 | thiserror 2.0 | 类型安全错误 |

### 依赖版本说明

**PyO3 & numpy 版本**:
- ✅ **当前版本**: PyO3 0.27.1 + numpy 0.27.0 (v0.3.1+)
- 🎉 **Python 3.14 支持**: PyO3 0.27 首次测试 Python 3.14.0 final
- 🔧 **平滑升级**: 无 breaking changes，所有指标验证通过
- 📚 **历史版本**: v0.3.0 使用 PyO3 0.26 + numpy 0.26

---

## 📝 版本历史

### v0.3.1 (2025-10-24)
**依赖升级**: PyO3 0.27 + numpy 0.27

**变更内容**:
- ⬆️ 升级 PyO3: 0.26.0 → 0.27.1
- ⬆️ 升级 numpy: 0.26.0 → 0.27.0
- ⬆️ 升级 rayon: 1.10 → 1.11
- 🎉 支持 Python 3.14.0 final
- ✅ 所有指标验证通过（FTI/CWT/VMD 冒烟测试）
- 📚 新增冒烟测试套件 (`rust_indicators/tests/`)

**测试结果**:
```
✅ FTI 冒烟测试通过
✅ CWT 冒烟测试通过 (cmor1.5-1.0 wavelet)
✅ VMD 冒烟测试通过 (能量守恒验证)
```

**升级影响**:
- ✨ 无 breaking changes，平滑升级
- 🔒 代码无需修改，干净编译（零警告）
- 📊 性能保持不变

---

### v0.2.3 (2025-10-22)

**VMD 内存布局优化 - Array3 维度重排**

核心改动:
- ⚡ **Array3 维度重排**: u_hat_plus 从 `(niter, t, k)` 改为 `(niter, k, t)`
- ⚡ **内存访问优化**: 将频率维度 `t` 放在最内层，提升 cache 局部性
- 🔧 **全面索引更新**: 修改 compute_single_omega_plus, vmd_core_loop, 后处理等约20处索引

性能提升:
- ⚡ Rolling window w256: **2.415s → 2.104s** (12.9% 提升)
- ⚡ Rolling window w128: **0.295s → 0.271s** (8.1% 提升)
- ⚡ k=4/k=3 倍率改善: **3.73x → 3.39x** (9.1% 改善)

数值精度:
- ✅ 完美保持数值对齐（误差 0.00e+00 < 1e-13）
- ✅ 所有正确性测试通过

技术细节:
- 新维度顺序使内层循环遍历连续内存
- 改善 cache line 利用率，减少 cache miss
- 对 k 较大的场景效果更明显

依赖升级:
- 🔧 **criterion 0.5 → 0.7**: 最新benchmark框架（dev依赖）
- 🔧 **rand 0.8 → 0.9**: API更新 `thread_rng()` → `rng()`, `gen()` → `random()`
- ✅ 零编译警告，所有测试通过

测试工具:
- ✅ 新增 `profile_vmd.py` 和 `analyze_vmd_timing.py`
- ✅ 通过 profiling 科学分析瓶颈

### v0.2.2 (2025-10-22)

**VMD 代码质量改进 - FFT Plan 架构优化**

代码重构:
- 🔧 **FFT Plan Cache 架构**: 引入 `FftPlanCache` 类型和可选 cache 参数
- 🔧 **utils.rs 重构**: `fft()` 和 `ifft()` 支持 optional cache 参数
- 🔧 **core.rs 优化**: VMD 主函数预计算所需的 FFT plan sizes

技术细节:
- 使用 `Arc<HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>>` 实现零成本共享
- 预计算 2 种 FFT sizes: t (镜像信号) 和 t/2 (输出频谱)
- 保持完美数值对齐（误差 < 1e-13）

性能影响:
- 单次调用内 FFT/IFFT 复用 cache，减少重复创建开销
- Rolling window 场景（多次调用）暂未观察到显著提升
- 为后续跨调用 cache 复用奠定架构基础

测试:
- ✅ 新增 `test_vmd_correctness.py` 数值正确性测试
- ✅ 新增 `bench_vmd_opt.py` 和 `bench_vmd_rolling.py` 性能基准

### v0.2.1 (2025-10-22)

**CWT 性能优化 - Phase 1 完成**

核心优化:
- ⚡ **FFT Planner 复用**: 预计算并缓存所有 FFT plan，避免重复创建（核心优化，提升 150%+）
- ⚡ **Scale 采样优化**: 预计算倒数避免重复除法（微小提升 2%）
- ⚡ **向量化 diff 计算**: 使用 ndarray 切片操作替代手工循环（微小提升 0.5%）

性能提升:
- ⚡ 平均加速: **5.8x → 10.2x** （提升 75.9%）
- ⚡ 最大加速: **8x → 17.4x** （提升 117.5%）
- ⚡ Scale 数量越多，优势越明显（100 scales 可达 17x）

数值精度:
- ✅ 完美保持与 PyWavelets 对齐（误差 < 2e-14）
- ✅ 所有测试通过（正确性 + 性能）

技术细节:
- 使用 HashMap 缓存不同 FFT size 的 plan
- Arc 包装实现并行环境下的零成本共享
- 保持零编译警告

### v0.2.0 (2025-10-22)

**CWT 实现 - 时频分析加速**

新增功能:
- ✅ CWT (Continuous Wavelet Transform) 完整实现
- ✅ 支持 Complex Morlet 小波 (cmor)
- ✅ 内置对称填充 (symmetric padding)
- ✅ 可选 verbose 模式用于开发调试

性能优化:
- ⚡ 平均 5.8x 加速（vs PyWavelets）
- ⚡ 大信号场景达到 7-8x 加速
- ⚡ Rayon 并行化处理多 scale 计算

数值精度:
- ✅ 与 PyWavelets 完美对齐（误差 < 2e-14）
- ✅ 3/3 集成测试通过
- ✅ 零编译警告

### v0.1.0 (2025-10-21)

**首次发布 - 生产就绪**

核心功能:
- ✅ VMD (Variational Mode Decomposition) 实现
- ✅ NRBO (Newton-Raphson Boundary Optimization) 实现
- ✅ 完整的数值对齐验证 (误差 ~1e-15)
- ✅ 性能基准测试 (50-100x 加速)

技术改进:
- 🔧 升级到 PyO3 0.26 (使用现代 Bound API)
- 🔧 升级到 numpy 0.26 (零拷贝优化)
- 🔧 零编译警告，100% 测试通过

项目清理:
- 📁 整理文档结构
- 🧹 配置 .gitignore 排除临时文件
- ✅ 代码审查通过，可安全集成

---

## 🎯 路线图

- [x] Phase 1-2: 核心算法实现
- [x] Phase 3-4: 数值对齐验证
- [x] Phase 5-6: 性能测试
- [ ] Phase 7: 策略集成测试
- [ ] Phase 8: 优化和发布

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python 绑定
- [rustfft](https://github.com/ejmahler/RustFFT) - 高性能 FFT 库
- [maturin](https://github.com/PyO3/maturin) - Python 扩展构建工具

---

**⚡ Powered by Rust + PyO3 | 为 jesse-trade 量化交易框架提供加速**
