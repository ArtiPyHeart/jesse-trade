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
    echo "  ./install.sh            # 生产环境 (默认安装 PyPI 版 jesse)"
    echo "  ./install.sh --patch    # 使用本地 jesse submodule (patch 分支)"
    echo "  ./install.sh --dev      # 开发环境 (基于生产环境增量安装)"
    echo "  ./install.sh --dev --patch  # 开发环境 + 本地 patch 分支"
    echo ""
}

strip_pip_block() {
    awk '
        /^  - pip:/ {in_pip=1; next}
        in_pip {
            if ($0 ~ /^  - /) {in_pip=0}
            else {next}
        }
        {print}
    ' "$1"
}

extract_pip_deps() {
    awk '
        /^  - pip:/ {in_pip=1; next}
        in_pip {
            if ($0 ~ /^  - /) {
                in_pip=0
                next
            }
            if ($0 ~ /^[[:space:]]+- /) {
                line=$0
                sub(/^[[:space:]]+- /, "", line)
                sub(/#.*/, "", line)
                gsub(/^[ \t]+|[ \t]+$/, "", line)
                if (line != "") print line
                next
            }
            next
        }
    ' "$1"
}

extract_conda_names() {
    awk '
        /^dependencies:/ {in_dep=1; next}
        !in_dep {next}
        /^  - pip:/ {in_pip=1; next}
        in_pip {
            if ($0 ~ /^  - /) {in_pip=0}
            else {next}
        }
        /^  - / {
            line=$0
            sub(/^  - /, "", line)
            sub(/#.*/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            gsub(/^"|"$/, "", line)
            if (line == "") next
            name=line
            sub(/[<>=!~].*$/, "", name)
            name=tolower(name)
            if (name != "") print name
        }
    ' "$1"
}

merge_conda_env_files() {
    local base_file="$1"
    local dev_file="$2"
    local out_file="$3"

    awk '
        function dep_key(spec,   name) {
            name = spec
            gsub(/\[.*\]/, "", name)
            sub(/[<>=!~].*$/, "", name)
            gsub(/^"|"$/, "", name)
            gsub(/^[ \t]+|[ \t]+$/, "", name)
            return tolower(name)
        }
        function add_channel(chan) {
            if (!(chan in channel_seen)) {
                channel_seen[chan] = 1
                channel_order[++channel_count] = chan
            }
        }
        NR == FNR {
            if ($0 ~ /^name:/) {
                if (env_name == "") {
                    env_name = $0
                    sub(/^name:[ \t]*/, "", env_name)
                }
            }
            if ($0 ~ /^channels:/) {
                in_channels = 1
                next
            }
            if (in_channels) {
                if ($0 ~ /^  - /) {
                    chan = $0
                    sub(/^  - /, "", chan)
                    gsub(/[ \t]+$/, "", chan)
                    add_channel(chan)
                    next
                }
                if ($0 !~ /^  /) {
                    in_channels = 0
                }
            }
            if ($0 ~ /^dependencies:/) {
                in_deps = 1
                next
            }
            if (in_deps) {
                if ($0 ~ /^  - /) {
                    spec = $0
                    sub(/^  - /, "", spec)
                    sub(/#.*/, "", spec)
                    gsub(/^[ \t]+|[ \t]+$/, "", spec)
                    gsub(/^"|"$/, "", spec)
                    if (spec != "") {
                        key = dep_key(spec)
                        if (key != "") {
                            if (!(key in base_seen)) {
                                base_order[++base_count] = key
                            }
                            base_seen[key] = 1
                            base_spec[key] = spec
                        }
                    }
                } else if ($0 !~ /^  /) {
                    in_deps = 0
                }
            }
            next
        }
        FNR == 1 {
            in_channels = 0
            in_deps = 0
        }
        {
            if ($0 ~ /^channels:/) {
                in_channels = 1
                next
            }
            if (in_channels) {
                if ($0 ~ /^  - /) {
                    chan = $0
                    sub(/^  - /, "", chan)
                    gsub(/[ \t]+$/, "", chan)
                    add_channel(chan)
                    next
                }
                if ($0 !~ /^  /) {
                    in_channels = 0
                }
            }
            if ($0 ~ /^dependencies:/) {
                in_deps = 1
                next
            }
            if (in_deps) {
                if ($0 ~ /^  - /) {
                    spec = $0
                    sub(/^  - /, "", spec)
                    sub(/#.*/, "", spec)
                    gsub(/^[ \t]+|[ \t]+$/, "", spec)
                    gsub(/^"|"$/, "", spec)
                    if (spec != "") {
                        key = dep_key(spec)
                        if (key != "") {
                            dev_spec[key] = spec
                            if (!(key in dev_seen)) {
                                dev_seen[key] = 1
                                if (!(key in base_seen)) {
                                    dev_only[++dev_count] = key
                                }
                            }
                        }
                    }
                } else if ($0 !~ /^  /) {
                    in_deps = 0
                }
            }
        }
        END {
            if (env_name == "") {
                env_name = "env"
            }
            print "name: " env_name
            if (channel_count > 0) {
                print "channels:"
                for (i = 1; i <= channel_count; i++) {
                    print "  - " channel_order[i]
                }
            }
            print "dependencies:"
            for (i = 1; i <= base_count; i++) {
                key = base_order[i]
                if (key in dev_spec) {
                    print "  - " dev_spec[key]
                } else {
                    print "  - " base_spec[key]
                }
            }
            for (i = 1; i <= dev_count; i++) {
                key = dev_only[i]
                print "  - " dev_spec[key]
            }
        }
    ' "$base_file" "$dev_file" > "$out_file"
}

MODE="prod"
PATCH_MODE="no"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dev)
            MODE="dev"
            shift
            ;;
        --patch)
            PATCH_MODE="yes"
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

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BASE_ENV_CONDA_FILE="$TMP_DIR/environment.base.conda.yml"
BASE_PIP_DEPS_FILE="$TMP_DIR/pip.base.txt"
CONDA_NAMES_FILE="$TMP_DIR/conda.names.txt"
ENV_CONDA_FILE="$BASE_ENV_CONDA_FILE"

strip_pip_block "$BASE_ENV_FILE" > "$BASE_ENV_CONDA_FILE"
extract_pip_deps "$BASE_ENV_FILE" > "$BASE_PIP_DEPS_FILE"
extract_conda_names "$BASE_ENV_FILE" > "$CONDA_NAMES_FILE"

DEV_ENV_CONDA_FILE=""
DEV_PIP_DEPS_FILE=""
if [ "$MODE" = "dev" ]; then
    DEV_ENV_CONDA_FILE="$TMP_DIR/environment.dev.conda.yml"
    DEV_PIP_DEPS_FILE="$TMP_DIR/pip.dev.txt"
    MERGED_ENV_CONDA_FILE="$TMP_DIR/environment.dev.merged.conda.yml"
    strip_pip_block "$DEV_ENV_FILE" > "$DEV_ENV_CONDA_FILE"
    extract_pip_deps "$DEV_ENV_FILE" > "$DEV_PIP_DEPS_FILE"
    extract_conda_names "$DEV_ENV_FILE" >> "$CONDA_NAMES_FILE"
    merge_conda_env_files "$BASE_ENV_CONDA_FILE" "$DEV_ENV_CONDA_FILE" "$MERGED_ENV_CONDA_FILE"
    ENV_CONDA_FILE="$MERGED_ENV_CONDA_FILE"
fi

sort -u "$CONDA_NAMES_FILE" | awk 'NF {if ($0 != "python" && $0 != "pip") print $0}' > "$CONDA_NAMES_FILE.sorted"
mv "$CONDA_NAMES_FILE.sorted" "$CONDA_NAMES_FILE"

CONDA_EXE="$(command -v conda)"
eval "$("$CONDA_EXE" shell.posix hook)"

CONDA_SOLVER="conda"
if command -v mamba >/dev/null 2>&1; then
    CONDA_SOLVER="mamba"
fi

echo ""
if [ "$MODE" = "dev" ]; then
    echo ">>> 步骤 2: 安装生产+开发环境依赖 ($ENV_NAME)..."
else
    echo ">>> 步骤 2: 安装生产环境依赖 ($ENV_NAME)..."
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    $CONDA_SOLVER env update -n "$ENV_NAME" -f "$ENV_CONDA_FILE" --prune
else
    $CONDA_SOLVER env create -n "$ENV_NAME" -f "$ENV_CONDA_FILE"
fi

if [ "$MODE" = "dev" ]; then
    echo ""
    echo ">>> 步骤 3: 开发依赖已合并到环境文件"
fi

echo ""
echo ">>> 步骤 4: 激活环境并检查 Python..."
set +u
conda activate "$ENV_NAME"
set -u
echo "✓ $(python --version)"

echo ""
JESSE_SPEC=""
JESSE_CONDA_SPECS_FILE="$TMP_DIR/jesse.conda.specs.txt"
JESSE_PIP_SPECS_FILE="$TMP_DIR/jesse.pip.specs.txt"
JESSE_META_FILE="$TMP_DIR/jesse.meta.env"

if [ "$PATCH_MODE" = "yes" ]; then
    echo ">>> 步骤 4.5: 更新 jesse submodule (patch 分支)..."
    JESSE_SUBMODULE_DIR="$ROOT_DIR/jesse"
    if [ ! -d "$JESSE_SUBMODULE_DIR" ]; then
        echo "❌ 错误: jesse submodule 目录不存在: $JESSE_SUBMODULE_DIR"
        echo "   请运行: git submodule update --init jesse"
        exit 1
    fi

    # 初始化 submodule（如果尚未初始化）
    if [ ! -f "$JESSE_SUBMODULE_DIR/.git" ] && [ ! -d "$JESSE_SUBMODULE_DIR/.git" ]; then
        echo "   初始化 jesse submodule..."
        git submodule update --init jesse
    fi

    # 切换到 patch 分支并拉取最新
    (
        cd "$JESSE_SUBMODULE_DIR"
        git fetch origin patch
        git checkout patch
        git reset --hard origin/patch
    )
    JESSE_COMMIT="$(cd "$JESSE_SUBMODULE_DIR" && git rev-parse --short HEAD)"
    echo "✓ jesse submodule 已同步到 origin/patch ($JESSE_COMMIT)"

    # jesse 从本地 submodule 安装
    JESSE_SPEC="$JESSE_SUBMODULE_DIR"

    echo ""
    echo ">>> 步骤 5: 解析 jesse 依赖并对齐 Conda 版本..."
    echo "   使用本地 submodule: $JESSE_SPEC"

    python -m pip install -q packaging

    JESSE_SPEC="$JESSE_SPEC" python - "$CONDA_NAMES_FILE" "$JESSE_CONDA_SPECS_FILE" "$JESSE_PIP_SPECS_FILE" "$JESSE_META_FILE" <<'PY'
import os
import re
import sys
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import shlex

conda_names_file = sys.argv[1]
conda_specs_file = sys.argv[2]
pip_specs_file = sys.argv[3]
meta_file = sys.argv[4]

jesse_spec = os.environ.get("JESSE_SPEC", "jesse")

with open(conda_names_file, "r", encoding="utf-8") as f:
    conda_names = {line.strip().lower() for line in f if line.strip()}

def _compatible_upper_bound(version: str) -> str:
    ver = Version(version)
    release = list(ver.release)
    if not release:
        return version
    if len(release) == 1:
        upper = [release[0] + 1, 0]
    else:
        upper_prefix = release[:-1]
        upper_prefix[-1] += 1
        upper = upper_prefix + [0]
    return ".".join(str(x) for x in upper)

def _conda_spec(spec_set: SpecifierSet) -> str:
    spec_str = str(spec_set)
    if not spec_str:
        return ""
    parts = [part.strip() for part in spec_str.split(",") if part.strip()]
    converted = []
    for part in parts:
        match = re.match(r"(~=|==|===|!=|<=|>=|<|>)(.+)", part)
        if not match:
            continue
        op, ver = match.groups()
        if op == "~=":
            converted.append(f">={ver}")
            converted.append(f"<{_compatible_upper_bound(ver)}")
        else:
            if op in ("==", "==="):
                op = "="
            converted.append(f"{op}{ver}")
    return ",".join(converted)

# 从本地 submodule 读取依赖
req_file = os.path.join(jesse_spec, "requirements.txt")
version_file = os.path.join(jesse_spec, "jesse", "version.py")

if not os.path.isfile(req_file):
    raise SystemExit(f"找不到 jesse requirements.txt: {req_file}")

# 读取版本号
version = ""
if os.path.isfile(version_file):
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if m:
                version = m.group(1)
                break

# 读取依赖
with open(req_file, "r", encoding="utf-8") as f:
    req_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

env = default_environment()
env["extra"] = ""

mapping = {
    "torch": "pytorch",
    "sklearn": "scikit-learn",
}

conda_specs = []
pip_specs = []

for req_str in req_lines:
    try:
        req = Requirement(req_str)
    except Exception:
        # 无法解析的行跳过
        continue
    if req.marker and not req.marker.evaluate(env):
        continue
    name = req.name.lower()
    conda_name = mapping.get(name, name)
    if conda_name in conda_names:
        spec = _conda_spec(req.specifier)
        conda_specs.append(f"{conda_name}{spec}")
    else:
        if req.url:
            pip_specs.append(req_str)
        else:
            pip_specs.append(f"{req.name}{req.specifier}")

with open(conda_specs_file, "w", encoding="utf-8") as f:
    for item in conda_specs:
        f.write(f"{item}\n")

with open(pip_specs_file, "w", encoding="utf-8") as f:
    for item in pip_specs:
        f.write(f"{item}\n")

with open(meta_file, "w", encoding="utf-8") as f:
    f.write(f"JESSE_VERSION={shlex.quote(version)}\n")
PY

    if [ -f "$JESSE_META_FILE" ]; then
        # shellcheck disable=SC1090
        source "$JESSE_META_FILE"
        if [ -n "${JESSE_VERSION:-}" ]; then
            echo "   jesse 版本: $JESSE_VERSION (patch)"
        fi
    fi

    if [ -s "$JESSE_CONDA_SPECS_FILE" ]; then
        JESSE_CONDA_ARGS="$(awk 'NF {printf "%s ", $0}' "$JESSE_CONDA_SPECS_FILE")"
        JESSE_CONDA_ARGS="${JESSE_CONDA_ARGS%" "}"
        if [ -n "$JESSE_CONDA_ARGS" ]; then
            $CONDA_SOLVER install -n "$ENV_NAME" -c conda-forge --yes $JESSE_CONDA_ARGS
        fi
    fi
else
    echo ">>> 步骤 4.5: 使用 PyPI 版本 jesse (默认)"
fi

echo ""
echo ">>> 步骤 6: 安装 pip 依赖 (含 jesse 的非 Conda 依赖)..."

PIP_INSTALL_FILE="$TMP_DIR/pip.install.txt"
EXCLUDE_JESSE="0"
if [ "$PATCH_MODE" = "yes" ]; then
    EXCLUDE_JESSE="1"
fi
{
    if [ -f "$BASE_PIP_DEPS_FILE" ]; then
        cat "$BASE_PIP_DEPS_FILE"
    fi
    if [ "$MODE" = "dev" ] && [ -f "$DEV_PIP_DEPS_FILE" ]; then
        cat "$DEV_PIP_DEPS_FILE"
    fi
    if [ -f "$JESSE_PIP_SPECS_FILE" ]; then
        cat "$JESSE_PIP_SPECS_FILE"
    fi
} | awk 'NF' | awk -v exclude_jesse="$EXCLUDE_JESSE" 'BEGIN{IGNORECASE=1}
    {
        line=$0
        name=line
        gsub(/\[.*\]/, "", name)
        sub(/[<>=!~].*$/, "", name)
        if (exclude_jesse == 1 && tolower(name) == "jesse") next
        print line
    }' | awk '!seen[tolower($0)]++' > "$PIP_INSTALL_FILE"

if [ -s "$PIP_INSTALL_FILE" ]; then
    PIP_INSTALL_ARGS="$(awk 'NF {printf "%s ", $0}' "$PIP_INSTALL_FILE")"
    PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS%" "}"
    if [ -n "$PIP_INSTALL_ARGS" ]; then
        python -m pip install $PIP_INSTALL_ARGS
    fi
fi

if [ "$PATCH_MODE" = "yes" ]; then
    # 从本地 submodule 安装 jesse（--no-deps 避免重复安装依赖）
    python -m pip install --no-deps "$JESSE_SPEC"
    echo "✓ jesse 已从本地 submodule 安装"
fi

if [ "$(uname)" = "Darwin" ]; then
    echo ""
    echo ">>> 步骤 6.5: macOS 检测到,重装 numpy 使用 Accelerate (避免 OpenBLAS/OpenMP 崩溃)..."
    $CONDA_SOLVER install -n "$ENV_NAME" -c conda-forge --yes \
        "numpy" \
        "libblas=*=*accelerate" \
        "liblapack=*=*accelerate" \
        "libcblas=*=*accelerate"
fi

echo ""
echo ">>> 步骤 7: 检查 maturin (Rust-Python 构建工具)..."
if ! command -v maturin >/dev/null 2>&1; then
    echo "❌ 错误: maturin 未安装"
    echo "   请检查环境文件中的依赖配置"
    exit 1
fi
echo "✓ $(maturin --version)"

echo ""
echo ">>> 步骤 8: 编译 Rust Indicators (必需)..."

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

echo ""
echo ">>> 步骤 9: 检查 Jesse 项目结构..."
if [ ! -d "$ROOT_DIR/storage" ]; then
    mkdir -p "$ROOT_DIR/storage"
    echo "✓ 已创建 storage/ 目录"
else
    echo "✓ storage/ 目录已存在"
fi

echo ""
echo "=========================================="
echo "✓ 安装成功完成！"
echo "=========================================="
echo ""
echo "已安装组件:"
echo "  • Conda 环境 ($ENV_NAME)"
echo "  • Rust 高性能指标 (VMD/NRBO)"
echo "  • Jesse 项目结构 (strategies/, storage/)"
echo ""
echo "可以开始使用 jesse-trade 进行回测和交易"
echo ""
