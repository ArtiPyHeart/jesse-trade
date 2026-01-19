#!/usr/bin/env bash
#
# 批量运行所有3模型组合的回测
# 基于已有的2模型组合结果，提取有效模型后进行3模型组合
#
# 使用方式:
#   ./run_triple_model_combos.sh              # 运行所有组合（跳过已完成的）
#   ./run_triple_model_combos.sh --dry-run    # 仅显示待运行组合，不运行
#   ./run_triple_model_combos.sh --force      # 强制重新运行所有组合
#   ./run_triple_model_combos.sh --clean      # 删除 sharpe ratio <= 0 的结果
#   ./run_triple_model_combos.sh --clean --dry-run  # 预览将被删除的目录
#

set -e

RESULTS_DIR="backtest_results"
LOG_DIR="${RESULTS_DIR}/batch_logs"

# 解析参数
DRY_RUN=false
FORCE=false
CLEAN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
        --clean)   CLEAN=true ;;
    esac
done

# ============================================
# 从2模型组合结果中提取有效模型
# ============================================
extract_models_from_pairs() {
    local models_set=$(mktemp)

    for dir in "$RESULTS_DIR"/*_vectorized_*/; do
        [[ ! -d "$dir" ]] && continue

        # 只处理2模型组合（目录名格式：model1_model2_vectorized_timestamp）
        dir_name=$(basename "$dir")

        # 提取模型名（移除 _vectorized_* 后缀，然后按 _ 分割）
        # 格式：c_L5_N2_c_L6_N1_vectorized_20260118_053409
        # 模型名格式：[cr]_L[0-9]+_N[0-9]+
        models_part="${dir_name%_vectorized_*}"

        # 使用正则提取所有模型名
        echo "$models_part" | grep -oE '[cr]_L[0-9]+_N[0-9]+' >> "$models_set"
    done

    # 去重并排序
    sort -u "$models_set"
    rm -f "$models_set"
}

# ============================================
# 清理模式：删除 sharpe ratio <= 0 的3模型结果
# ============================================
if $CLEAN; then
    echo "=============================================="
    echo "Cleanup Mode: Remove Low-Quality Triple Combos"
    echo "=============================================="
    echo "Criteria: Sharpe Ratio <= 0"
    echo ""

    TO_DELETE=()
    TO_KEEP=()

    # 遍历所有3模型组合的回测结果目录
    # 3模型组合目录名包含3个模型名，如：c_L5_N2_c_L6_N1_r_L7_N2_vectorized_*
    for dir in "$RESULTS_DIR"/*_vectorized_*/; do
        [[ ! -d "$dir" ]] && continue

        dir_name=$(basename "$dir")
        models_part="${dir_name%_vectorized_*}"

        # 计算模型数量（通过匹配模型名模式）
        model_count=$(echo "$models_part" | grep -oE '[cr]_L[0-9]+_N[0-9]+' | wc -l | tr -d ' ')

        # 只处理3模型组合
        [[ "$model_count" -ne 3 ]] && continue

        metrics_file="${dir}metrics.json"
        if [[ ! -f "$metrics_file" ]]; then
            continue
        fi

        # 提取 sharpe_ratio
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
        printf "  %-70s Sharpe: %s\n" "$dir_name" "$sharpe"
    done
    echo "----------------------------------------------"
    echo ""

    if $DRY_RUN; then
        echo "[DRY RUN] No files deleted."
        exit 0
    fi

    # 确认删除
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

echo "=============================================="
echo "Triple Model Combo Backtest Runner"
echo "=============================================="
echo ""
echo "Extracting models from existing pair results..."

# 提取有效模型
MODELS_FILE=$(mktemp)
extract_models_from_pairs > "$MODELS_FILE"
NUM_MODELS=$(wc -l < "$MODELS_FILE" | tr -d ' ')

if [[ $NUM_MODELS -lt 3 ]]; then
    echo "Error: Need at least 3 models, found $NUM_MODELS"
    rm -f "$MODELS_FILE"
    exit 1
fi

# 计算组合数 C(n,3) = n! / (3! * (n-3)!) = n*(n-1)*(n-2)/6
TOTAL_COMBOS=$((NUM_MODELS * (NUM_MODELS - 1) * (NUM_MODELS - 2) / 6))

echo "Models found:    ${NUM_MODELS}"
echo "Total combos:    ${TOTAL_COMBOS} (C($NUM_MODELS,3))"
echo ""
echo "Models:"
cat "$MODELS_FILE" | while read m; do echo "  - $m"; done
echo ""

# 检查组合是否已完成
is_combo_completed() {
    local m1="$1"
    local m2="$2"
    local m3="$3"

    # 排序模型名
    local sorted=$(printf "%s\n%s\n%s" "$m1" "$m2" "$m3" | sort)
    m1=$(echo "$sorted" | sed -n '1p')
    m2=$(echo "$sorted" | sed -n '2p')
    m3=$(echo "$sorted" | sed -n '3p')

    local pattern="${m1}_${m2}_${m3}_vectorized_*"
    local match
    match=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | head -1)

    if [[ -n "$match" ]] && [[ -f "$match/metrics.json" ]]; then
        return 0
    fi

    return 1
}

# 生成所有3模型组合
COMPLETED_COUNT=0
PENDING_COMBOS_FILE=$(mktemp)

LINE_NUM1=0
while IFS= read -r MODEL1; do
    LINE_NUM1=$((LINE_NUM1 + 1))

    LINE_NUM2=0
    while IFS= read -r MODEL2; do
        LINE_NUM2=$((LINE_NUM2 + 1))
        [[ $LINE_NUM2 -le $LINE_NUM1 ]] && continue

        LINE_NUM3=0
        while IFS= read -r MODEL3; do
            LINE_NUM3=$((LINE_NUM3 + 1))
            [[ $LINE_NUM3 -le $LINE_NUM2 ]] && continue

            if ! $FORCE && is_combo_completed "$MODEL1" "$MODEL2" "$MODEL3"; then
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            else
                echo "$MODEL1 $MODEL2 $MODEL3" >> "$PENDING_COMBOS_FILE"
            fi
        done < "$MODELS_FILE"
    done < "$MODELS_FILE"
done < "$MODELS_FILE"

PENDING_COUNT=$(wc -l < "$PENDING_COMBOS_FILE" | tr -d ' ')

echo "=============================================="
echo "Completed:       ${COMPLETED_COUNT}"
echo "Pending:         ${PENDING_COUNT}"
if $FORCE; then
    echo "Mode:            FORCE (re-run all)"
fi
echo "=============================================="

if [[ $PENDING_COUNT -eq 0 ]]; then
    echo ""
    echo "All combos already completed! Use --force to re-run."
    rm -f "$MODELS_FILE" "$PENDING_COMBOS_FILE"
    exit 0
fi

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Pending combos to run:"
    echo ""
    COMBO_NUM=0
    while IFS=' ' read -r M1 M2 M3; do
        COMBO_NUM=$((COMBO_NUM + 1))
        printf "[%3d/%d] %s %s %s\n" $COMBO_NUM $PENDING_COUNT "$M1" "$M2" "$M3"
    done < "$PENDING_COMBOS_FILE"
    rm -f "$MODELS_FILE" "$PENDING_COMBOS_FILE"
    exit 0
fi

# 创建日志目录
mkdir -p "$LOG_DIR"
BATCH_LOG="${LOG_DIR}/triple_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Batch log: $BATCH_LOG"
echo ""

# 运行待处理的组合
COMBO_COUNT=0
FAILED_COUNT=0
SUCCESS_COUNT=0
START_TIME=$(date +%s)

while IFS=' ' read -r MODEL1 MODEL2 MODEL3; do
    COMBO_COUNT=$((COMBO_COUNT + 1))

    echo ""
    echo "=============================================="
    printf "[%3d/%d] Running: %s + %s + %s\n" $COMBO_COUNT $PENDING_COUNT "$MODEL1" "$MODEL2" "$MODEL3"
    echo "=============================================="

    if python flow_backtest_vectorized.py "$MODEL1" "$MODEL2" "$MODEL3" 2>&1 | tee -a "$BATCH_LOG"; then
        echo "[SUCCESS] $MODEL1 + $MODEL2 + $MODEL3" >> "$BATCH_LOG"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[FAILED] $MODEL1 + $MODEL2 + $MODEL3" >> "$BATCH_LOG"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done < "$PENDING_COMBOS_FILE"

rm -f "$MODELS_FILE" "$PENDING_COMBOS_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Batch Complete"
echo "=============================================="
echo "Processed:     $COMBO_COUNT"
echo "Success:       $SUCCESS_COUNT"
echo "Failed:        $FAILED_COUNT"
echo "Elapsed time:  ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "Log file:      $BATCH_LOG"
echo "=============================================="
