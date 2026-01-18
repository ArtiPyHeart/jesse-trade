#!/usr/bin/env bash
#
# 批量运行所有模型两两组合的回测
#
# 使用方式:
#   ./run_all_model_pairs.sh           # 运行所有组合（跳过已完成的）
#   ./run_all_model_pairs.sh --dry-run # 仅显示待运行组合，不运行
#   ./run_all_model_pairs.sh --force   # 强制重新运行所有组合
#

set -e

MODEL_DIR="strategies/BinanceBtcDemoBar/models"
RESULTS_DIR="backtest_results"
LOG_DIR="${RESULTS_DIR}/batch_logs"

# 解析参数
DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
    esac
done

# 检查模型组合是否已完成回测
# 参数: MODEL1 MODEL2
# 返回: 0 = 已完成, 1 = 未完成
is_pair_completed() {
    local m1="$1"
    local m2="$2"

    # 排序模型名称（与 create_output_dir 中的 sorted() 一致）
    if [[ "$m1" > "$m2" ]]; then
        local tmp="$m1"
        m1="$m2"
        m2="$tmp"
    fi

    local pattern="${m1}_${m2}_vectorized_*"

    # 使用 find 检查是否存在匹配的目录且包含 metrics.json
    local match
    match=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | head -1)

    if [[ -n "$match" ]] && [[ -f "$match/metrics.json" ]]; then
        return 0  # 已完成
    fi

    return 1  # 未完成
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
        printf "[%3d/%d] %s %s\n" $PAIR_NUM $PENDING_COUNT "$M1" "$M2"
    done < "$PENDING_PAIRS_FILE"
    rm -f "$MODELS_FILE" "$PENDING_PAIRS_FILE"
    exit 0
fi

# 创建日志目录
mkdir -p "$LOG_DIR"
BATCH_LOG="${LOG_DIR}/batch_$(date +%Y%m%d_%H%M%S).log"
echo "Batch log: $BATCH_LOG"
echo ""

# 运行待处理的组合
PAIR_COUNT=0
FAILED_COUNT=0
SUCCESS_COUNT=0
START_TIME=$(date +%s)

while IFS=' ' read -r MODEL1 MODEL2; do
    PAIR_COUNT=$((PAIR_COUNT + 1))

    echo ""
    echo "=============================================="
    printf "[%3d/%d] Running: %s + %s\n" $PAIR_COUNT $PENDING_COUNT "$MODEL1" "$MODEL2"
    echo "=============================================="

    if python flow_backtest_vectorized.py "$MODEL1" "$MODEL2" 2>&1 | tee -a "$BATCH_LOG"; then
        echo "[SUCCESS] $MODEL1 + $MODEL2" >> "$BATCH_LOG"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[FAILED] $MODEL1 + $MODEL2" >> "$BATCH_LOG"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done < "$PENDING_PAIRS_FILE"

rm -f "$MODELS_FILE" "$PENDING_PAIRS_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Batch Complete"
echo "=============================================="
echo "Processed:     $PAIR_COUNT"
echo "Success:       $SUCCESS_COUNT"
echo "Failed:        $FAILED_COUNT"
echo "Elapsed time:  ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "Log file:      $BATCH_LOG"
echo "=============================================="
