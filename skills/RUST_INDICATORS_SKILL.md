# Rust高性能指标开发SKILL

## 技术栈
- Rust 1.74+ | PyO3 0.26 | numpy 0.26 | ndarray 0.15 | rustfft 6.2 | maturin 1.0+
- **警告**: ndarray必须0.15，升级到0.16会与numpy 0.26冲突

## 快速开始
```bash
./install.sh  # 自动CPU优化编译(target-cpu=native/lto=fat)
# 或手动: cd rust_indicators && cargo clean && maturin develop --release
```

## 集成原则
1. **数值对齐**: 误差必须<1e-10
2. **移除并行**: 删除joblib.Parallel（Rust已足够快）
3. **类型转换**: `dc=bool(DC)`（Python int → Rust bool）
4. **显式传参**: Rust无默认值（如`max_iter=10, tol=1e-6`）

## 集成流程
```python
# 1. 创建测试: tests/test_{indicator}_rust_integration.py
import _rust_indicators
max_error = np.max(np.abs(result_python - result_rust))
assert max_error < 1e-10

# 2. 替换生产代码
# Before: joblib.Parallel()(delayed(slow_func)(d) for d in data)
# After:  [_rust_indicators.xxx_py(d, param=val) for d in data]
```

## 开发新指标（5步）
```rust
// 1. core.rs - 算法实现
pub fn compute(signal: &Array1<f64>, param: f64) -> Array1<f64> { ... }

// 2. ffi.rs - Python绑定
#[pyfunction]
pub fn xxx_py<'py>(py: Python<'py>, signal: PyReadonlyArray1<f64>, param: f64)
    -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(compute(&signal.as_array().to_owned(), param).into_pyarray_bound(py))
}

// 3. lib.rs - 注册模块
m.add_function(wrap_pyfunction!(xxx_py, m)?)?;
```
```bash
# 4. 编译测试
cargo check && cargo clippy && maturin develop --release

# 5. 数值验证（误差<1e-10）
```

## 常见问题
| 问题 | 解决 |
|------|------|
| ModuleNotFoundError | `cd rust_indicators && maturin develop --release` |
| 类型错误 | `dc=bool(DC)` 显式转换 |
| 数值不对齐 | 检查参数类型、对比中间结果 |

参考: `rust_indicators/README.md` | [PyO3文档](https://pyo3.rs)
