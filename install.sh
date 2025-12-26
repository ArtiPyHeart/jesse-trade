#!/usr/bin/env bash

set -euo pipefail

print_header() {
    echo "=========================================="
    echo "Jesse-Trade 环境安装 (Conda)"
    echo "=========================================="
    echo ""
}

print_usage() {
    echo "用法:"
    echo "  ./install.sh            # 生产环境"
    echo "  ./install.sh --dev      # 开发环境 (基于生产环境增量安装)"
    echo ""
}

MODE="prod"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            print_usage
            exit 1
            ;;
    esac
done

print_header

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_ENV_FILE="$ROOT_DIR/environment.yml"
DEV_ENV_FILE="$ROOT_DIR/environment-dev.yml"

echo ">>> 步骤 1: 检查必需依赖..."

if ! command -v conda >/dev/null 2>&1; then
    echo "❌ 错误: 未检测到 conda"
    echo "   请先安装 Miniforge/Miniconda 并配置到 PATH"
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
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
echo "✓ $(rustc --version)"

if [ ! -f "$BASE_ENV_FILE" ]; then
    echo "❌ 错误: 未找到环境定义文件: $BASE_ENV_FILE"
    exit 1
fi

ENV_NAME="$(awk -F ': *' '/^name:/ {print $2; exit}' "$BASE_ENV_FILE")"
if [ -z "$ENV_NAME" ]; then
    echo "❌ 错误: 环境文件缺少 name 字段: $BASE_ENV_FILE"
    exit 1
fi

if [ "$MODE" = "dev" ]; then
    if [ ! -f "$DEV_ENV_FILE" ]; then
        echo "❌ 错误: 未找到环境定义文件: $DEV_ENV_FILE"
        exit 1
    fi

    DEV_ENV_NAME="$(awk -F ': *' '/^name:/ {print $2; exit}' "$DEV_ENV_FILE")"
    if [ -z "$DEV_ENV_NAME" ]; then
        echo "❌ 错误: 环境文件缺少 name 字段: $DEV_ENV_FILE"
        exit 1
    fi
    if [ "$DEV_ENV_NAME" != "$ENV_NAME" ]; then
        echo "❌ 错误: 生产/开发环境名称不一致: $ENV_NAME vs $DEV_ENV_NAME"
        exit 1
    fi
fi

CONDA_EXE="$(command -v conda)"
eval "$("$CONDA_EXE" shell.posix hook)"

CONDA_SOLVER="conda"
if command -v mamba >/dev/null 2>&1; then
    CONDA_SOLVER="mamba"
fi

echo ""
echo ">>> 步骤 2: 安装生产环境依赖 ($ENV_NAME)..."

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    $CONDA_SOLVER env update -n "$ENV_NAME" -f "$BASE_ENV_FILE" --prune
else
    $CONDA_SOLVER env create -n "$ENV_NAME" -f "$BASE_ENV_FILE"
fi

if [ "$MODE" = "dev" ]; then
    echo ""
    echo ">>> 步骤 3: 安装开发环境依赖 (增量更新)..."
    $CONDA_SOLVER env update -n "$ENV_NAME" -f "$DEV_ENV_FILE"
fi

echo ""
echo ">>> 步骤 4: 激活环境并检查 Python..."
conda activate "$ENV_NAME"
echo "✓ $(python --version)"

echo ""
echo ">>> 检查 maturin (Rust-Python 构建工具)..."
if ! command -v maturin >/dev/null 2>&1; then
    echo "❌ 错误: maturin 未安装"
    echo "   请检查环境文件中的依赖配置"
    exit 1
fi
echo "✓ $(maturin --version)"

echo ""
echo ">>> 步骤 5: 编译 Rust Indicators (必需)..."

if [ ! -d "$ROOT_DIR/rust_indicators" ]; then
    echo "❌ 错误: rust_indicators 目录不存在"
    echo "   项目结构可能不完整,请检查代码仓库"
    exit 1
fi

cd "$ROOT_DIR/rust_indicators"

echo ">>> 清理旧的编译产物 (确保干净构建)..."
if [ -d "target" ]; then
    rm -rf target
    echo "  ✓ 已删除 target/ 目录"
fi

if [ -f "Cargo.lock" ]; then
    rm -f Cargo.lock
    echo "  ✓ 已删除 Cargo.lock 文件"
fi

if command -v cargo >/dev/null 2>&1; then
    cargo clean 2>/dev/null || true
fi

echo ">>> 编译 Rust 扩展 (针对当前CPU优化的release模式)..."
echo "   这是完整的干净构建,可能需要几分钟,请耐心等待..."

export RUSTFLAGS="-C target-cpu=native"

if ! maturin develop --release; then
    echo ""
    echo "❌ Rust 编译失败"
    echo "   请检查错误信息,或联系开发团队"
    exit 1
fi

unset RUSTFLAGS

cd "$ROOT_DIR"

echo "✓ Rust Indicators 编译完成 (已针对当前CPU优化)"

if [ "$(uname)" = "Linux" ] && command -v systemctl >/dev/null 2>&1; then
    systemctl restart pgbouncer
fi

echo ""
echo "=========================================="
echo "✓ 安装成功完成！"
echo "=========================================="
echo ""
echo "已安装组件:"
echo "  • Conda 环境 ($ENV_NAME)"
echo "  • Rust 高性能指标 (VMD/NRBO)"
echo ""
echo "可以开始使用 jesse-trade 进行回测和交易"
echo ""
