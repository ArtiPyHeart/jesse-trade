#!/usr/bin/env bash
#
# 批量运行所有模型两两组合的回测
#
# 使用方式:
#   ./run_2_model_pairs.sh                # 运行所有组合（跳过已完成的）
#   ./run_2_model_pairs.sh -j 4           # 并行运行4个回测
#   ./run_2_model_pairs.sh --jobs 8       # 并行运行8个回测
#   ./run_2_model_pairs.sh --dry-run      # 仅显示待运行组合，不运行
#   ./run_2_model_pairs.sh --force        # 强制重新运行所有组合
#   ./run_2_model_pairs.sh --clean        # 删除 sharpe ratio <= 0 的结果
#   ./run_2_model_pairs.sh --clean --dry-run  # 预览将被删除的目录
#

set -e

MODEL_DIR="strategies/BinanceBtcDemoBar/models"
RESULTS_DIR="backtest_results"
LOG_DIR="${RESULTS_DIR}/batch_logs"

# 解析参数
DRY_RUN=false
FORCE=false
CLEAN=false
JOBS=1  # 默认单线程

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE=true; shift ;;
        --clean)   CLEAN=true; shift ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        *) shift ;;
    esac
done

# 验证 JOBS 参数
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
    echo "Error: --jobs must be a positive integer, got: $JOBS"
    exit 1
fi

# ============================================
# 清理模式：删除 sharpe ratio <= 0 的结果
# ============================================
if $CLEAN; then
    echo "=============================================="
    echo "Cleanup Mode: Remove Low-Quality Results"
    echo "=============================================="
    echo "Criteria: Sharpe Ratio <= 0"
    echo ""

    TO_DELETE=()
    TO_KEEP=()

    # 遍历所有回测结果目录（只处理2模型组合）
    for dir in "$RESULTS_DIR"/*_vectorized_*/; do
        [[ ! -d "$dir" ]] && continue

        dir_name=$(basename "$dir")
        models_part="${dir_name%_vectorized_*}"

        # 计算模型数量（通过匹配模型名模式）
        model_count=$(echo "$models_part" | grep -oE '[cr]_L[0-9]+_N[0-9]+' | wc -l | tr -d ' ')

        # 只处理2模型组合
        [[ "$model_count" -ne 2 ]] && continue

        metrics_file="${dir}metrics.json"
        if [[ ! -f "$metrics_file" ]]; then
            continue
        fi

        # 提取 sharpe_ratio（使用 python 解析 JSON）
        sharpe=$(python3 -c "
import json
with open('$metrics_file') as f:
    data = json.load(f)
    sharpe = data.get('risk', {}).get('sharpe_ratio', 0)
    print(sharpe if sharpe is not None else 0)
" 2>/dev/null || echo "0")

        # 比较 sharpe ratio
        is_bad=$(python3 -c "print(1 if float('$sharpe') <= 0 else 0)" 2>/dev/null || echo "1")

        if [[ "$is_bad" == "1" ]]; then
            TO_DELETE+=("$dir_name|$sharpe")
        else
            TO_KEEP+=("$dir_name|$sharpe")
        fi
    done

    echo "Results to KEEP (Sharpe > 0): ${#TO_KEEP[@]}"
    echo "Results to DELETE (Sharpe <= 0): ${#TO_DELETE[@]}"
    echo ""

    if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
        echo "No directories to delete."
        exit 0
    fi

    echo "Directories to delete:"
    echo "----------------------------------------------"
    for item in "${TO_DELETE[@]}"; do
        dir_name="${item%|*}"
        sharpe="${item#*|}"
        printf "  %-60s Sharpe: %s\n" "$dir_name" "$sharpe"
    done
    echo "----------------------------------------------"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] No files deleted."
        exit 0
    fi

    # 确认删除（兼容 bash 和 zsh）
    printf "Delete these ${#TO_DELETE[@]} directories? [y/N] "
    read confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi

    # 执行删除
    deleted=0
    for item in "${TO_DELETE[@]}"; do
        dir_name="${item%|*}"
        dir_path="$RESULTS_DIR/$dir_name"
        if rm -rf "$dir_path"; then
            echo "  Deleted: $dir_name"
            deleted=$((deleted + 1))
        else
            echo "  Failed:  $dir_name"
        fi
    done

    echo ""
    echo "=============================================="
    echo "Cleanup Complete"
    echo "=============================================="
    echo "Deleted: $deleted directories"
    echo "Remaining: ${#TO_KEEP[@]} directories"
    echo "=============================================="
    exit 0
fi

# ============================================
# 回测模式
# ============================================

# 检查模型组合是否已完成回测
is_pair_completed() {
    local m1="$1"
    local m2="$2"

    if [[ "$m1" > "$m2" ]]; then
        local tmp="$m1"
        m1="$m2"
        m2="$tmp"
    fi

    local pattern="${m1}_${m2}_vectorized_*"
    local match
    match=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | head -1)

    if [[ -n "$match" ]] && [[ -f "$match/metrics.json" ]]; then
        return 0
    fi

    return 1
}

# 获取所有模型名称
MODELS_FILE=$(mktemp)
ls "$MODEL_DIR" | grep -E '^[cr]_L[0-9]+_N[0-9]+$' | sort > "$MODELS_FILE"

NUM_MODELS=$(wc -l < "$MODELS_FILE" | tr -d ' ')
TOTAL_PAIRS=$((NUM_MODELS * (NUM_MODELS - 1) / 2))

# 统计已完成和待运行的组合
COMPLETED_COUNT=0
PENDING_PAIRS_FILE=$(mktemp)

LINE_NUM1=0
while IFS= read -r MODEL1; do
    LINE_NUM1=$((LINE_NUM1 + 1))

    LINE_NUM2=0
    while IFS= read -r MODEL2; do
        LINE_NUM2=$((LINE_NUM2 + 1))
        [[ $LINE_NUM2 -le $LINE_NUM1 ]] && continue

        if ! $FORCE && is_pair_completed "$MODEL1" "$MODEL2"; then
            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        else
            echo "$MODEL1 $MODEL2" >> "$PENDING_PAIRS_FILE"
        fi
    done < "$MODELS_FILE"
done < "$MODELS_FILE"

PENDING_COUNT=$(wc -l < "$PENDING_PAIRS_FILE" | tr -d ' ')

echo "=============================================="
echo "Model Pair Backtest Runner"
echo "=============================================="
echo "Models found:    ${NUM_MODELS}"
echo "Total pairs:     ${TOTAL_PAIRS}"
echo "Completed:       ${COMPLETED_COUNT}"
echo "Pending:         ${PENDING_COUNT}"
echo "Parallel jobs:   ${JOBS}"
if $FORCE; then
    echo "Mode:            FORCE (re-run all)"
fi
echo "=============================================="

if [[ $PENDING_COUNT -eq 0 ]]; then
    echo ""
    echo "All pairs already completed! Use --force to re-run."
    rm -f "$MODELS_FILE" "$PENDING_PAIRS_FILE"
    exit 0
fi

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Pending pairs to run:"
    echo ""
    PAIR_NUM=0
    while IFS=' ' read -r M1 M2; do
        PAIR_NUM=$((PAIR_NUM + 1))
        printf "[%3d/%d] %s %s\n" "$PAIR_NUM" "$PENDING_COUNT" "$M1" "$M2"
    done < "$PENDING_PAIRS_FILE"
    rm -f "$MODELS_FILE" "$PENDING_PAIRS_FILE"
    exit 0
fi

# 创建日志目录
mkdir -p "$LOG_DIR"
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="${LOG_DIR}/pairs_batch_${BATCH_TIMESTAMP}.log"
echo "Batch log: $BATCH_LOG"
echo ""

# ============================================
# 运行待处理的组合（并行版本 - 使用 xargs）
# ============================================
START_TIME=$(date +%s)

echo "Starting parallel execution with $JOBS workers..."
echo ""

# 创建临时目录存放每个任务的状态
TASK_DIR=$(mktemp -d)
trap 'rm -rf "$TASK_DIR" "$MODELS_FILE" "$PENDING_PAIRS_FILE"' EXIT

# 创建运行脚本（每个任务执行一次）
RUNNER_SCRIPT="$TASK_DIR/runner.sh"
cat > "$RUNNER_SCRIPT" << 'RUNNER_EOF'
#!/usr/bin/env bash
line="$1"
task_dir="$2"
batch_log="$3"
pending_count="$4"

# 解析参数
pair_num=$(echo "$line" | cut -d'|' -f1)
m1=$(echo "$line" | cut -d'|' -f2)
m2=$(echo "$line" | cut -d'|' -f3)

echo "[${pair_num}/${pending_count}] START: ${m1} + ${m2}"
echo "[$(date '+%H:%M:%S')] [${pair_num}/${pending_count}] Starting: ${m1} + ${m2}" >> "$batch_log"

task_log="${task_dir}/task_${pair_num}.log"

if python flow_backtest_vectorized.py "$m1" "$m2" > "$task_log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] [${pair_num}/${pending_count}] SUCCESS: ${m1} + ${m2}" >> "$batch_log"
    echo "[${pair_num}] DONE: ${m1} + ${m2}"
    echo "1" > "${task_dir}/success_${pair_num}"
else
    echo "[$(date '+%H:%M:%S')] [${pair_num}/${pending_count}] FAILED:  ${m1} + ${m2}" >> "$batch_log"
    echo "[${pair_num}] FAIL: ${m1} + ${m2}"
    echo "1" > "${task_dir}/failed_${pair_num}"
fi
RUNNER_EOF
chmod +x "$RUNNER_SCRIPT"

# 生成带编号的任务列表（格式：编号|模型1|模型2）
NUMBERED_TASKS="$TASK_DIR/numbered_tasks.txt"
PAIR_NUM=0
while IFS=' ' read -r M1 M2; do
    PAIR_NUM=$((PAIR_NUM + 1))
    echo "${PAIR_NUM}|${M1}|${M2}"
done < "$PENDING_PAIRS_FILE" > "$NUMBERED_TASKS"

# 使用 xargs 并行执行
xargs -P "$JOBS" -I {} "$RUNNER_SCRIPT" {} "$TASK_DIR" "$BATCH_LOG" "$PENDING_COUNT" < "$NUMBERED_TASKS"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# 统计成功和失败数
SUCCESS_COUNT=$(find "$TASK_DIR" -name 'success_*' 2>/dev/null | wc -l | tr -d ' ')
FAILED_COUNT=$(find "$TASK_DIR" -name 'failed_*' 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=============================================="
echo "Batch Complete"
echo "=============================================="
echo "Processed:     $PAIR_NUM"
echo "Success:       $SUCCESS_COUNT"
echo "Failed:        $FAILED_COUNT"
echo "Parallel jobs: $JOBS"
echo "Elapsed time:  ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "Log file:      $BATCH_LOG"
echo "=============================================="
