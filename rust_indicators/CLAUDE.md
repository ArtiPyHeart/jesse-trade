# Rust Indicators 开发指南

## ⚠️ 编译规范

**每次修改 Rust 代码后，必须全量干净编译：**
```bash
cargo clean && maturin develop --release
```

**原因**：FFI边界、内联优化、数值精度问题（编译需30-60秒，必要代价）

**基本流程**：
```bash
# 1. 首次安装（editable模式，IDE可识别）
pip install -e .

# 2. 修改Rust代码后，干净编译
cargo clean && maturin develop --release

# 3. 验证加载
python -c "from pyrs_indicators.ind_decomposition import vmd; print('OK')"

# 4. 运行测试
python tests/test_<indicator>_correctness.py
```

**关键配置**（pyproject.toml）：
- `[project] name = "pyrs-indicators"` 包名（用连字符）
- `[tool.maturin] module-name = "pyrs_indicators._rust_indicators"` 将 Rust 扩展作为子模块
- `[tool.maturin] python-source = "."` 指向 Python 包所在目录
- 版本号必须同步更新（当前 0.4.0）

**为什么使用子模块路径**：
- ✅ 避免 Rust 扩展与 Python 包同名冲突
- ✅ IDE 能正确识别顶层 Python 包
- ✅ 编译无警告（`PyInit_*` 符号匹配）
- ✅ 符合 Maturin 最佳实践

---

## 🎨 架构设计

**分层职责**：
```
用户代码 → pyrs_indicators (Python层) → _rust_indicators (Rust层)
          ↑ 类型提示、参数验证、文档    ↑ 高性能计算
```

- **Rust层**：纯计算，零验证（极致性能）
- **Python层**：Fail Fast验证，用户友好接口

**目录结构**：
```
pyrs_indicators/
├── __init__.py              # 版本号、导出
├── _core.py                 # Rust绑定（内部）
└── ind_<category>/          # 指标子包（如ind_wavelets/）
    └── <indicator>.py       # 单个指标
```

---

## 📝 接口规范

**核心模板**：
```python
def indicator(
    data: npt.NDArray[np.float64],
    param: float = 1.0,
    *,
    optional: bool = False
) -> npt.NDArray[np.float64]:
    """简短描述

    Args:
        data: 输入数据（1D数组）
        param: 参数说明（推荐值）

    Returns:
        输出数组（形状说明）

    Raises:
        ValueError: 非法输入
    """
    # 参数验证（Fail Fast）
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError(...)
    if param <= 0:
        raise ValueError(...)

    # 调用Rust
    result = _rust_indicators.func(data, param)

    # 结果验证
    if np.any(np.isnan(result)):
        raise RuntimeError("Computation failed")

    return result
```

**验证检查清单**：
1. 类型：`isinstance(signal, np.ndarray)`
2. 维度：`signal.ndim == 1`
3. 长度：`len(signal) >= min_length`
4. 值域：`0 < param <= max_value`
5. 输出：无NaN/Inf

---

## 🧪 测试规范

**文件命名**：`tests/test_<indicator>_correctness.py`

**必含测试**：
```python
def test_basic():
    """冒烟测试：形状、无NaN"""
    result = indicator(signal)
    assert result.shape == expected
    assert not np.any(np.isnan(result))

def test_validation():
    """参数验证：非法输入抛异常"""
    with pytest.raises(ValueError, match="must be positive"):
        indicator(signal, param=-1)

def test_typical_case():
    """典型场景：实际使用验证"""
    ...
```

---

## ✅ 开发原则

1. **用户友好优先**：完整类型提示 + 详细文档
2. **Fail Fast**：立即抛异常，不静默处理
3. **零性能开销**：Python层只验证不计算
4. **测试驱动**：新功能必须有测试
5. **干净编译**：每次修改必须`cargo clean`
