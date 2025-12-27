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

find_jesse_spec() {
    local deps_file="$1"
    if [ -z "$deps_file" ] || [ ! -f "$deps_file" ]; then
        return 0
    fi
    awk 'BEGIN{IGNORECASE=1}
        {
            line=$0
            sub(/#.*/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            if (line == "") next
            name=line
            gsub(/\[.*\]/, "", name)
            sub(/[<>=!~].*$/, "", name)
            if (tolower(name) == "jesse") {print line; exit}
        }' "$deps_file"
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

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BASE_ENV_CONDA_FILE="$TMP_DIR/environment.base.conda.yml"
BASE_PIP_DEPS_FILE="$TMP_DIR/pip.base.txt"
CONDA_NAMES_FILE="$TMP_DIR/conda.names.txt"

strip_pip_block "$BASE_ENV_FILE" > "$BASE_ENV_CONDA_FILE"
extract_pip_deps "$BASE_ENV_FILE" > "$BASE_PIP_DEPS_FILE"
extract_conda_names "$BASE_ENV_FILE" > "$CONDA_NAMES_FILE"

DEV_ENV_CONDA_FILE=""
DEV_PIP_DEPS_FILE=""
if [ "$MODE" = "dev" ]; then
    DEV_ENV_CONDA_FILE="$TMP_DIR/environment.dev.conda.yml"
    DEV_PIP_DEPS_FILE="$TMP_DIR/pip.dev.txt"
    strip_pip_block "$DEV_ENV_FILE" > "$DEV_ENV_CONDA_FILE"
    extract_pip_deps "$DEV_ENV_FILE" > "$DEV_PIP_DEPS_FILE"
    extract_conda_names "$DEV_ENV_FILE" >> "$CONDA_NAMES_FILE"
fi

sort -u "$CONDA_NAMES_FILE" | awk 'NF {if ($0 != "python" && $0 != "pip") print $0}' > "$CONDA_NAMES_FILE.sorted"
mv "$CONDA_NAMES_FILE.sorted" "$CONDA_NAMES_FILE"

JESSE_SPEC="$(find_jesse_spec "$BASE_PIP_DEPS_FILE")"
if [ -z "$JESSE_SPEC" ] && [ "$MODE" = "dev" ]; then
    JESSE_SPEC="$(find_jesse_spec "$DEV_PIP_DEPS_FILE")"
fi
if [ -z "$JESSE_SPEC" ]; then
    JESSE_SPEC="jesse"
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
    $CONDA_SOLVER env update -n "$ENV_NAME" -f "$BASE_ENV_CONDA_FILE" --prune
else
    $CONDA_SOLVER env create -n "$ENV_NAME" -f "$BASE_ENV_CONDA_FILE"
fi

if [ "$MODE" = "dev" ]; then
    echo ""
    echo ">>> 步骤 3: 安装开发环境依赖 (增量更新)..."
    $CONDA_SOLVER env update -n "$ENV_NAME" -f "$DEV_ENV_CONDA_FILE"
fi

echo ""
echo ">>> 步骤 4: 激活环境并检查 Python..."
conda activate "$ENV_NAME"
echo "✓ $(python --version)"

echo ""
echo ">>> 步骤 5: 解析 jesse 依赖并对齐 Conda 版本..."
echo "   使用 $JESSE_SPEC 作为依赖基准"

JESSE_CONDA_SPECS_FILE="$TMP_DIR/jesse.conda.specs.txt"
JESSE_PIP_SPECS_FILE="$TMP_DIR/jesse.pip.specs.txt"
JESSE_META_FILE="$TMP_DIR/jesse.meta.env"

python -m pip install -q packaging

JESSE_SPEC="$JESSE_SPEC" python - "$CONDA_NAMES_FILE" "$JESSE_CONDA_SPECS_FILE" "$JESSE_PIP_SPECS_FILE" "$JESSE_META_FILE" <<'PY'
import os
import re
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import shlex
import subprocess
import shutil

conda_names_file = sys.argv[1]
conda_specs_file = sys.argv[2]
pip_specs_file = sys.argv[3]
meta_file = sys.argv[4]

jesse_spec = os.environ.get("JESSE_SPEC", "jesse")

with open(conda_names_file, "r", encoding="utf-8") as f:
    conda_names = {line.strip().lower() for line in f if line.strip()}

tmp_dir = tempfile.mkdtemp()

def _read_metadata_from_wheel(path: str) -> str:
    with zipfile.ZipFile(path) as zf:
        meta_name = next(name for name in zf.namelist() if name.endswith(".dist-info/METADATA"))
        return zf.read(meta_name).decode("utf-8")

def _read_metadata_from_sdist(path: str) -> str:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            meta_name = next((name for name in zf.namelist() if name.endswith("PKG-INFO")), None)
            if not meta_name:
                raise RuntimeError("PKG-INFO not found in sdist zip")
            return zf.read(meta_name).decode("utf-8")
    with tarfile.open(path, "r:*") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("PKG-INFO")), None)
        if not member:
            raise RuntimeError("PKG-INFO not found in sdist")
        return tf.extractfile(member).read().decode("utf-8")

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

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--dest", tmp_dir, jesse_spec]
    )
    candidates = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
    wheel = next((f for f in candidates if f.endswith(".whl")), None)
    if wheel:
        meta_text = _read_metadata_from_wheel(wheel)
    else:
        sdist = next((f for f in candidates if f.endswith((".tar.gz", ".tgz", ".zip"))), None)
        if not sdist:
            raise RuntimeError("jesse package archive not found after download")
        meta_text = _read_metadata_from_sdist(sdist)
except Exception as exc:
    raise SystemExit(f"解析 jesse 元数据失败: {exc}") from exc
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

metadata = Parser().parsestr(meta_text)
version = metadata.get("Version", "")
requires_python = metadata.get("Requires-Python", "")
requires_dist = metadata.get_all("Requires-Dist") or []

if requires_python:
    spec = SpecifierSet(requires_python)
    current = Version(".".join(map(str, sys.version_info[:3])))
    if not spec.contains(current, prereleases=True):
        raise SystemExit(
            f"当前 Python {current} 不满足 jesse Requires-Python: {requires_python}"
        )

env = default_environment()
env["extra"] = ""

mapping = {
    "torch": "pytorch",
    "sklearn": "scikit-learn",
}

conda_specs = []
pip_specs = []

for req_str in requires_dist:
    req = Requirement(req_str)
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
    f.write(f"JESSE_REQUIRES_PYTHON={shlex.quote(requires_python)}\n")
PY

if [ -f "$JESSE_META_FILE" ]; then
    # shellcheck disable=SC1090
    source "$JESSE_META_FILE"
    if [ -n "${JESSE_VERSION:-}" ]; then
        echo "   jesse 版本: $JESSE_VERSION"
    fi
    if [ -n "${JESSE_REQUIRES_PYTHON:-}" ]; then
        echo "   Requires-Python: $JESSE_REQUIRES_PYTHON"
    fi
fi

if [ -s "$JESSE_CONDA_SPECS_FILE" ]; then
    JESSE_CONDA_ARGS="$(awk 'NF {printf "%s ", $0}' "$JESSE_CONDA_SPECS_FILE")"
    JESSE_CONDA_ARGS="${JESSE_CONDA_ARGS%" "}"
    if [ -n "$JESSE_CONDA_ARGS" ]; then
        $CONDA_SOLVER install -n "$ENV_NAME" -c conda-forge --yes $JESSE_CONDA_ARGS
    fi
fi

echo ""
echo ">>> 步骤 6: 安装 pip 依赖 (含 jesse 的非 Conda 依赖)..."

PIP_INSTALL_FILE="$TMP_DIR/pip.install.txt"
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
} | awk 'NF' | awk 'BEGIN{IGNORECASE=1}
    {
        line=$0
        name=line
        gsub(/\[.*\]/, "", name)
        sub(/[<>=!~].*$/, "", name)
        if (tolower(name) == "jesse") next
        print line
    }' | awk '!seen[tolower($0)]++' > "$PIP_INSTALL_FILE"

if [ -s "$PIP_INSTALL_FILE" ]; then
    PIP_INSTALL_ARGS="$(awk 'NF {printf "%s ", $0}' "$PIP_INSTALL_FILE")"
    PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS%" "}"
    if [ -n "$PIP_INSTALL_ARGS" ]; then
        python -m pip install $PIP_INSTALL_ARGS
    fi
fi

python -m pip install --no-deps "$JESSE_SPEC"

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
