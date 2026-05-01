#!/usr/bin/env bash
# Replication experiment: 3 seeds x 2 strategies = 6 evals on 40-mol slices.
# Caffeinate prevents sleep on long runs.
[[ -z "${CAFFEINATED}" ]] && exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

CANONICAL_CSV="$REPO/outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.8_s50.csv"
CANONICAL_TXT="$REPO/outputs/chebi20_27M_scibert_20ep_contrastive_eval_summary_cfg1.5_t0.8_s50.txt"
BACKUP_CSV="$REPO/outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.8_s50_BEST_BASELINE_v2.csv"
BACKUP_TXT="$REPO/outputs/chebi20_27M_scibert_20ep_contrastive_eval_summary_cfg1.5_t0.8_s50_BEST_BASELINE_v2.txt"

echo "============================================================"
echo "REPLICATION EXPERIMENT -- dep_aware smoke test, 3 seeds"
echo "Start: $(date)"
echo "============================================================"

# -- Pre-flight checks
echo ""
echo "[preflight] Checking required files..."
for f in \
    "$REPO/checkpoints/chebi20_27M_scibert_20ep_contrastive.pt" \
    "$REPO/checkpoints/contrastive_scibert_chembl.pt" \
    "$REPO/data/chebi20_test.csv" \
    "$REPO/outputs/stratified_analysis_best_model.csv"
do
    if [[ ! -f "$f" ]]; then
        echo "  ERROR: missing $f"
        exit 1
    fi
    echo "  OK: $(basename "$f")"
done
echo ""

# -- Build slices
echo "[phase 1] Building smoke slices for seeds 43, 44, 45 ..."
for SEED in 43 44 45; do
    echo ""
    echo "  -> seed $SEED"
    python3 "$SCRIPT_DIR/build_smoke_slice_v2.py" --seed "$SEED"
done
echo ""
echo "[phase 1] Done."
echo ""

# -- Backup canonical file (once, before any eval)
if [[ -f "$CANONICAL_CSV" && ! -f "$BACKUP_CSV" ]]; then
    echo "[backup] Saving canonical baseline before evals..."
    cp "$CANONICAL_CSV" "$BACKUP_CSV"
    [[ -f "$CANONICAL_TXT" ]] && cp "$CANONICAL_TXT" "$BACKUP_TXT"
    echo "  Saved to $(basename "$BACKUP_CSV")"
elif [[ -f "$BACKUP_CSV" ]]; then
    echo "[backup] Backup already exists, skipping."
else
    echo "[backup] No canonical CSV found yet, nothing to back up."
fi
echo ""

# -- Eval helper
run_eval() {
    local SEED=$1
    local STRATEGY=$2
    local SLICE_CSV="$REPO/data/chebi20_smoke_hard40_seed${SEED}.csv"
    local OUT_CSV="$REPO/outputs/replication_seed${SEED}_${STRATEGY}.csv"
    local OUT_TXT="$REPO/outputs/replication_seed${SEED}_${STRATEGY}_summary.txt"

    echo "--------------------------------------------------------------"
    echo "  seed=$SEED  strategy=$STRATEGY  $(date +%H:%M:%S)"
    echo "--------------------------------------------------------------"

    python3 "$REPO/finetune/train_chebi20.py" --eval_only \
        --encoder contrastive \
        --contrastive_ckpt "$REPO/checkpoints/contrastive_scibert_chembl.pt" \
        --model_size 27M_scibert_20ep \
        --cfg 1.5 --temp 0.8 --steps 50 \
        --test_csv "$SLICE_CSV" \
        --commit_strategy "$STRATEGY"

    if [[ -f "$CANONICAL_CSV" ]]; then
        mv "$CANONICAL_CSV" "$OUT_CSV"
        echo "  -> renamed: $(basename "$OUT_CSV")"
    else
        echo "  ERROR: expected output not found: $CANONICAL_CSV"
        exit 1
    fi
    if [[ -f "$CANONICAL_TXT" ]]; then
        mv "$CANONICAL_TXT" "$OUT_TXT"
        echo "  -> renamed: $(basename "$OUT_TXT")"
    fi
    echo ""
}

echo "[phase 2] Running 6 evals (est. 3-5 min each) ..."
echo ""
for SEED in 43 44 45; do
    run_eval "$SEED" standard
    run_eval "$SEED" dep_aware
done

# -- Restore canonical baseline
if [[ -f "$BACKUP_CSV" ]]; then
    echo "[restore] Restoring canonical baseline..."
    cp "$BACKUP_CSV" "$CANONICAL_CSV"
    [[ -f "$BACKUP_TXT" ]] && cp "$BACKUP_TXT" "$CANONICAL_TXT"
    echo "  Restored $(basename "$CANONICAL_CSV")"
fi
echo ""

# -- Quick side-by-side from summary .txt files
echo "============================================================"
echo "QUICK SUMMARY (from summary files)"
echo "============================================================"
for SEED in 43 44 45; do
    for STRAT in standard dep_aware; do
        SFILE="$REPO/outputs/replication_seed${SEED}_${STRAT}_summary.txt"
        if [[ -f "$SFILE" ]]; then
            echo "  seed=$SEED  $STRAT:"
            grep -E "morgan|maccs|bleu" "$SFILE" | head -3 | sed "s/^/    /"
        fi
    done
done
echo ""
echo "============================================================"
echo "End: $(date)"
echo "All 6 evals complete."
echo "Run: python3 scripts/compare_replication_results.py"
echo "============================================================"
