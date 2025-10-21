# Rust高性能指标开发SKILL

## 概述
本SKILL定义了jesse-trade项目中Rust高性能指标的开发、集成和维护标准。

## 项目结构
```
rust_indicators/
├── Cargo.toml              # Rust包配置
├── pyproject.toml          # Python绑定配置
├── README.md               # 完整文档
├── src/
│   ├── lib.rs             # 模块入口
│   ├── vmd/               # VMD指标
│   │   ├── core.rs        # 核心算法
│   │   ├── utils.rs       # FFT工具
│   │   └── ffi.rs         # Python绑定
│   └── nrbo/              # NRBO指标
│       ├── core.rs        # 核心算法
│       └── ffi.rs         # Python绑定
└── tests/                 # 数值对齐测试
```

## 技术栈
- **核心语言**: Rust 1.74+
- **Python绑定**: PyO3 0.26 (Bound API)
- **NumPy绑定**: numpy 0.26 (零拷贝)
- **数组操作**: ndarray 0.15
- **FFT**: rustfft 6.2
- **构建工具**: maturin 1.0+

## 安装与编译

### 自动安装(推荐)
```bash
./install.sh  # 项目根目录,自动编译并安装Rust指标
```

**优化说明**: install.sh会自动设置针对当前CPU的最优化编译:
- `target-cpu=native`: 利用当前CPU的所有指令集(AVX2/AVX512等)
- `lto=fat`: 完整链接时优化,跨crate内联
- `codegen-units=1`: 单编译单元,最大化优化机会

### 手动编译

**开发模式**(快速编译,无优化):
```bash
cd rust_indicators
maturin develop
```

**生产模式**(完整优化):
```bash
cd rust_indicators

# 标准release
maturin develop --release

# 最优化release(针对当前CPU,推荐)
RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1" \
  maturin develop --release
```

## 已实现指标

### VMD (Variational Mode Decomposition)
```python
import _rust_indicators

u, u_hat, omega = _rust_indicators.vmd_py(
    signal, alpha=2000, tau=0.0, k=5,
    dc=False,  # ⚠️ 必须bool类型
    init=1, tol=1e-7
)
# 返回: (u, u_hat, omega) - 分解的模态、频域表示、中心频率
# 性能: 94.4x加速
```

### NRBO (Newton-Raphson Boundary Optimization)
```python
import _rust_indicators

optimized = _rust_indicators.nrbo_py(
    imf,
    max_iter=10,  # ⚠️ 必须显式传参
    tol=1e-6
)
# 返回: 优化后的IMF
# 性能: 53.6x加速
```

## 集成到生产代码

### 原则
1. **数值对齐优先**: 集成前必须验证误差<1e-10
2. **移除并行**: Rust已足够快,移除joblib.Parallel
3. **参数转换**: 注意Python/Rust类型转换(如dc需bool)
4. **保持接口**: 替换实现,不改变外部API

### 集成步骤

#### Step 1: 创建数值对齐测试
```python
# tests/test_{indicator}_rust_integration.py
import numpy as np
import _rust_indicators
from src.indicators.prod.xxx import python_version

def test_numerical_alignment():
    signal = np.random.randn(100)

    result_python = python_version(signal, ...)
    result_rust = _rust_indicators.xxx_py(signal, ...)

    max_error = np.max(np.abs(result_python - result_rust))
    assert max_error < 1e-10, f"Error {max_error:.2e} >= 1e-10"
```

#### Step 2: 运行测试验证
```bash
PYTHONPATH=/path/to/jesse-trade pytest tests/test_{indicator}_rust_integration.py -v
```

#### Step 3: 替换生产代码
```python
# Before:
from src.indicators.prod.xxx.python_impl import slow_function
from joblib import Parallel, delayed

def calc(data):
    return slow_function(data, param1, param2)

res = Parallel()(delayed(calc)(d) for d in data_list)

# After:
import _rust_indicators

def calc(data):
    # 注意参数类型转换
    return _rust_indicators.xxx_py(data, param1=param1, param2=bool(param2))

# Rust足够快,无需并行
res = [calc(d) for d in data_list]
```

### 示例: VMD集成

**生产文件**: `src/indicators/prod/emd/cls_vmd_indicator.py`

```python
# 导入
import _rust_indicators

# 核心函数
def _calc_vmd_nrbo(src: np.ndarray):
    # VMD分解
    u, u_hat, omega = _rust_indicators.vmd_py(
        src, alpha=ALPHA, tau=TAU, k=K,
        dc=bool(DC),  # 类型转换!
        init=INIT, tol=TOL
    )
    u = u[2:]  # 跳过前2个模态

    # NRBO优化
    u_nrbo = np.zeros_like(u)
    for i in range(u.shape[0]):
        u_nrbo[i] = _rust_indicators.nrbo_py(
            u[i], max_iter=10, tol=1e-6
        )
    return u_nrbo.T

# 顺序处理(无需并行)
def _sequential_process(self):
    src_windows = [...]
    res = [_calc_vmd_nrbo(w) for w in src_windows]
    self.raw_result.extend(res)
```

## 开发新指标

### 1. Rust核心实现
```rust
// src/new_indicator/core.rs
use ndarray::Array1;

pub fn compute(signal: &Array1<f64>, param: f64) -> Array1<f64> {
    // 算法实现
}
```

### 2. Python绑定
```rust
// src/new_indicator/ffi.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
pub fn new_indicator_py<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    param: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = signal.as_array();
    let result = super::core::compute(&signal.to_owned(), param);
    Ok(result.into_pyarray_bound(py))
}
```

### 3. 注册到模块
```rust
// src/lib.rs
mod new_indicator;

#[pymodule]
fn _rust_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_indicator::new_indicator_py, m)?)?;
    Ok(())
}
```

### 4. 编译测试
```bash
cargo check              # 检查错误
cargo clippy             # 静态分析
maturin develop --release # 编译安装
```

### 5. 数值验证
创建对比测试,确保与Python版本误差<1e-10

## 关键注意事项

### 类型转换陷阱
```python
# ❌ 错误: Python int传给Rust bool
_rust_indicators.vmd_py(signal, dc=0)  # 类型错误!

# ✓ 正确: 显式转换
_rust_indicators.vmd_py(signal, dc=bool(DC))
```

### 参数必须显式
```python
# ❌ 错误: 缺少默认参数
_rust_indicators.nrbo_py(imf)  # Rust没有默认值!

# ✓ 正确: 显式传参
_rust_indicators.nrbo_py(imf, max_iter=10, tol=1e-6)
```

### 不要过度并行
```python
# ❌ 不必要: Rust已经很快
from joblib import Parallel, delayed
res = Parallel(n_jobs=-1)(delayed(rust_func)(x) for x in data)

# ✓ 简洁高效: 直接调用
res = [rust_func(x) for x in data]
```

## 测试标准

### 数值对齐
- **误差要求**: max_error < 1e-10
- **测试用例**: 多种信号长度(100, 150, 200, 500, 1000)
- **边界情况**: 小信号、大信号、含噪声信号

### 性能基准
- 对比Python/Numba版本
- 记录加速比和绝对时间
- 测试冷启动和稳态性能

## 依赖版本锁定

### 关键约束
```toml
[dependencies]
pyo3 = "0.26"        # 使用Bound API
numpy = "0.26"       # 零拷贝优化
ndarray = "0.15"     # 必须0.15!(numpy 0.26依赖)
rustfft = "6.2"
thiserror = "2.0"
```

**警告**: ndarray不能升级到0.16,会与numpy 0.26冲突!

## 故障排查

### 编译警告
```bash
cargo clippy --all-targets  # 查看所有警告
cargo fix --allow-dirty     # 自动修复部分问题
```

### 导入失败
```python
# 症状: ModuleNotFoundError: No module named '_rust_indicators'
# 解决: 重新编译安装
cd rust_indicators && maturin develop --release
```

### 数值不对齐
1. 检查参数类型转换
2. 对比中间结果
3. 使用`assert_allclose`查看具体差异

## 参考文档
- 完整文档: `rust_indicators/README.md`
- PyO3官方: https://pyo3.rs
- numpy绑定: https://github.com/PyO3/rust-numpy
