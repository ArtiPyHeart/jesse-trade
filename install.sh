#!/bin/bash

# 脚本出错时立即退出
set -e

echo ">>> 正在检查操作系统..."

# 检查是否为 macOS (Darwin是其内核名)
if [[ "$(uname)" == "Darwin" ]]; then
  echo ">>> 检测到 macOS，将采用源码编译方式优化 NumPy..."

  # 第一步：正常安装所有依赖。
  # pip 会解决所有依赖关系，并安装一个正确版本的、但可能是预编译的 numpy。
  echo ">>> 步骤 1/3: 安装所有项目依赖..."
  python -m pip install -r requirements.txt

  # 第二步：从环境中找出已安装的 numpy 及其确切版本。
  # `pip freeze` 会列出所有包及其版本，例如 "numpy==1.26.4"。
  # `grep` 用于精确匹配以 "numpy==" 开头的行。
  echo ">>> 步骤 2/3: 检测已安装的 NumPy 版本..."
  NUMPY_SPEC=$(python -m pip freeze | grep -i '^numpy==')

  if [ -z "$NUMPY_SPEC" ]; then
    echo "错误：在环境中未找到 NumPy。请检查 requirements.txt 文件。"
    exit 1
  fi
  
  echo ">>> 检测到规格为: $NUMPY_SPEC"

  # 第三步：使用检测到的确切版本号，强制从源码重新安装 numpy。
  # --force-reinstall: 卸载现有版本再安装 [3][5]。
  # --no-binary numpy: 确保 numpy 从源码编译。
  # --no-deps: 因为依赖项在第一步已安装好，此步不再重复安装，可以加速并避免潜在问题 [1]。
  echo ">>> 步骤 3/3: 强制从源码重新编译并安装 $NUMPY_SPEC..."
  python -m pip install --force-reinstall --no-deps --no-binary :all: "$NUMPY_SPEC"
  
else
  # 对于非 macOS 系统 (如 Linux, Windows)
  echo ">>> 非 macOS 系统，执行标准安装..."
  python -m pip install -r requirements.txt
fi

# ========== Rust Indicators 编译 ==========
echo ""
echo ">>> 步骤 4/4: 编译 Rust Indicators..."

# 检查 Rust 是否安装
if ! command -v cargo &> /dev/null; then
    echo "⚠️  未检测到 Rust，跳过 Rust Indicators 编译"
    echo "   如需使用 Rust 加速的 VMD/NRBO，请安装 Rust:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
else
    RUST_VERSION=$(rustc --version)
    echo ">>> 检测到 $RUST_VERSION"

    # 检查 maturin 是否已安装
    if ! command -v maturin &> /dev/null; then
        echo ">>> 安装 maturin (Rust-Python 构建工具)..."
        python -m pip install maturin
    fi

    # 进入 rust_indicators 目录
    if [ -d "rust_indicators" ]; then
        cd rust_indicators

        echo ">>> 编译 Rust 扩展 (release 模式)..."
        maturin develop --release

        cd ..

        echo ">>> ✅ Rust Indicators 编译完成！"
        echo "   VMD 和 NRBO 现在可以使用 10-20x 加速版本"
    else
        echo "⚠️  rust_indicators 目录不存在，跳过 Rust 编译"
    fi
fi

# 检查是否为 Linux 系统
if [[ "$(uname)" == "Linux" ]]; then
    systemctl restart pgbouncer
fi

echo ""
echo ">>> 安装成功完成！"
