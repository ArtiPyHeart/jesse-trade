#!/bin/bash

# 脚本出错时立即退出
set -e

echo "=========================================="
echo "Jesse-Trade 生产环境安装"
echo "=========================================="
echo ""

# ========== 必需依赖检查 ==========
echo ">>> 步骤 1: 检查必需依赖..."

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未检测到 Python"
    echo "   请先安装 Python 3.8+"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo "✓ $PYTHON_VERSION"

# 检查 Rust (必需)
if ! command -v cargo &> /dev/null; then
    echo ""
    echo "❌ 错误: 未检测到 Rust"
    echo ""
    echo "Rust 是项目运行的必需依赖 (用于高性能 VMD/NRBO 指标)。"
    echo "请先安装 Rust,然后重新运行此脚本:"
    echo ""
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    echo "安装完成后,重启终端或运行: source \$HOME/.cargo/env"
    exit 1
fi
RUST_VERSION=$(rustc --version)
echo "✓ $RUST_VERSION"

echo ""
echo ">>> 步骤 2: 安装 Python 依赖..."

# 检查是否为 macOS (Darwin是其内核名)
if [[ "$(uname)" == "Darwin" ]]; then
  echo ">>> 检测到 macOS，将采用源码编译方式优化 NumPy..."

  # 第一步：正常安装所有依赖。
  # pip 会解决所有依赖关系，并安装一个正确版本的、但可能是预编译的 numpy。
  echo "  [2.1/3] 安装所有项目依赖..."
  python -m pip install -r requirements.txt

  # 第二步：从环境中找出已安装的 numpy 及其确切版本。
  # `pip freeze` 会列出所有包及其版本，例如 "numpy==1.26.4"。
  # `grep` 用于精确匹配以 "numpy==" 开头的行。
  echo "  [2.2/3] 检测已安装的 NumPy 版本..."
  NUMPY_SPEC=$(python -m pip freeze | grep -i '^numpy==')

  if [ -z "$NUMPY_SPEC" ]; then
    echo "❌ 错误：在环境中未找到 NumPy。请检查 requirements.txt 文件。"
    exit 1
  fi

  echo "  检测到规格为: $NUMPY_SPEC"

  # 第三步：使用检测到的确切版本号，强制从源码重新安装 numpy。
  # --force-reinstall: 卸载现有版本再安装 [3][5]。
  # --no-binary numpy: 确保 numpy 从源码编译。
  # --no-deps: 因为依赖项在第一步已安装好，此步不再重复安装，可以加速并避免潜在问题 [1]。
  echo "  [2.3/3] 强制从源码重新编译并安装 $NUMPY_SPEC..."
  python -m pip install --force-reinstall --no-deps --no-binary :all: "$NUMPY_SPEC"
  
else
  # 对于非 macOS 系统 (如 Linux, Windows)
  echo ">>> 非 macOS 系统，执行标准安装..."
  python -m pip install -r requirements.txt
fi

# 检查 maturin (在安装Python依赖后)
echo ""
echo ">>> 检查 maturin (Rust-Python 构建工具)..."
if ! command -v maturin &> /dev/null; then
    echo "❌ 错误: maturin 未安装"
    echo "   maturin 应该在 requirements.txt 中，请检查是否正确安装"
    exit 1
fi
MATURIN_VERSION=$(maturin --version)
echo "✓ $MATURIN_VERSION"

# ========== Rust Indicators 编译 (必需) ==========
echo ""
echo ">>> 步骤 3: 编译 Rust Indicators (必需)..."

# 检查 rust_indicators 目录
if [ ! -d "rust_indicators" ]; then
    echo "❌ 错误: rust_indicators 目录不存在"
    echo "   项目结构可能不完整,请检查代码仓库"
    exit 1
fi

cd rust_indicators

echo ">>> 编译 Rust 扩展 (针对当前CPU优化的release模式)..."
echo "   这可能需要几分钟,请耐心等待..."

# 设置针对当前CPU的优化标志
# target-cpu=native: 针对当前CPU架构优化
# 注意: lto和codegen-units已在Cargo.toml的[profile.release]中配置
export RUSTFLAGS="-C target-cpu=native"

# 编译,如果失败则退出
if ! maturin develop --release; then
    echo ""
    echo "❌ Rust 编译失败"
    echo "   请检查错误信息,或联系开发团队"
    exit 1
fi

# 清除环境变量
unset RUSTFLAGS

cd ..

echo "✓ Rust Indicators 编译完成 (已针对当前CPU优化)"
echo "  VMD/NRBO 可获得 50-100x 加速"

# 检查是否为 Linux 系统
if [[ "$(uname)" == "Linux" ]]; then
    systemctl restart pgbouncer
fi

echo ""
echo "=========================================="
echo "✓ 安装成功完成！"
echo "=========================================="
echo ""
echo "已安装组件:"
echo "  • Python 依赖包"
echo "  • Rust 高性能指标 (VMD/NRBO)"
echo ""
echo "可以开始使用 jesse-trade 进行回测和交易"
echo ""
