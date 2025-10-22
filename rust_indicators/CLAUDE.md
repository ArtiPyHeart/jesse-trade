# Rust Indicators 开发指南

## ⚠️ 编译规范

**每次修改 Rust 代码后，必须使用全量干净编译：**

```bash
cargo clean && maturin develop --release
```

**禁止增量编译：**
```bash
# ✗ 错误
maturin develop --release

# ✓ 正确
cargo clean && maturin develop --release
```

### 为什么必须干净编译？

1. **FFI边界问题**：增量编译可能保留旧的符号定义
2. **内联优化**：跨模块的内联函数可能不会重新优化
3. **数值精度**：编译器优化差异可能导致精度变化

### 基本开发流程

```bash
# 1. 修改代码
# 2. 干净编译
cargo clean && maturin develop --release

# 3. 验证加载
python -c "import _rust_indicators; print('OK')"

# 4. 运行测试
PYTHONPATH=. python tests/test_cwt_rust_integration.py
```

---

**注意**：完整编译需要30-60秒，这是确保正确性的必要代价。
