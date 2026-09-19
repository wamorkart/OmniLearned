#!/bin/bash
# Finish the DeepSets top-tagging width scan: train + evaluate the three
# missing points (nano, micro, tiny) on the confirmed a05_T4 KD recipe, so
# they line up with the already-done small (298,887 params, 93.30%) and
# off-curve distillnet (10,981, 92.85%).
#
#   screen -dmS deepsets_size_scan bash scripts/size_scan_deepsets_queue.sh
#
# Sequential, ONE interactive slot at a time (the other is held by the live
# atlas_flav ftag v2 loop) -- for each width:
#   1. bash scripts/distill_loop_top.sh top_deepsets_<w>   (salloc-resubmit
#      loop, 4 nodes, exits 0 after 50 epochs)
#   2. bash scripts/run_eval_deepsets.sh <tag> <w>         (salloc 1 node,
#      writes per-rank npz)
#   3. compute_metrics_top.py --tag <tag>                  (append to RESULTS)
# A width whose loop hits MAX_LOOPS is logged and skipped, scan continues.
#
# All three configs are one-variable (SIZE) copies of top_deepsets_a05:
# KD alpha=beta=0.5 / T=4, teacher fine_tune_top_l, wd 0.5, batch 128,
# 1000 iters x 50 epochs, lr 5e-4.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
CKPT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/size_scan_deepsets
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/results.log"
DRIVER_LOG="$LOG_DIR/driver.log"

# Which widths this driver runs (sequential, one interactive slot). Override
# with SCAN_WIDTHS="nano micro" when another width is being run elsewhere
# (e.g. tiny on the regular queue via train_deepsets_tiny_reg.sbatch).
# shellcheck disable=SC2206
WIDTHS=(${SCAN_WIDTHS:-nano micro tiny})
declare -A CFG=(
    [nano]=top_deepsets_nano
    [micro]=top_deepsets_micro
    [tiny]=top_deepsets_tiny
)
declare -A TAG=(
    [nano]=distill_top_deepsets_nano_scratch_a05_T4
    [micro]=distill_top_deepsets_micro_scratch_a05_T4
    [tiny]=distill_top_deepsets_tiny_scratch_a05_T4
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"; }

log "=== DeepSets width-scan driver start: ${WIDTHS[*]} ==="
{
    echo "===================================================================="
    echo "  DeepSets top-tagging width scan  --  started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  recipe: a05_T4 KD (alpha=beta=0.5, T=4, teacher fine_tune_top_l,"
    echo "          wd 0.5, batch 128, 1000 it x 50 ep, lr 5e-4)"
    echo "  reference points already on disk:"
    echo "    distillnet (10,981 params, off-curve)  92.85% / AUC 0.9800 / rej50 199.1"
    echo "    small      (298,887 params)            93.30% / AUC 0.9828 / rej50 283.2"
    echo "===================================================================="
} | tee -a "$RESULTS"

for w in "${WIDTHS[@]}"; do
    cfg="${CFG[$w]}"
    tag="${TAG[$w]}"
    train_json="$CKPT_DIR/training_${tag}.json"

    log "---- width=$w  config=$cfg  tag=$tag ----"

    if [ -f "$train_json" ]; then
        log "$train_json already present -- skipping training for $w"
    else
        log "launching training loop: distill_loop_top.sh $cfg"
        bash "$SCRIPT_DIR/distill_loop_top.sh" "$cfg" \
            2>&1 | tee -a "$LOG_DIR/train_${w}.out"
        rc="${PIPESTATUS[0]}"
        log "training loop for $w exited $rc"
        if [ "$rc" -ne 0 ] || [ ! -f "$train_json" ]; then
            log "SKIP $w: training did not complete (no $train_json). Continuing."
            echo "  [$w] TRAINING INCOMPLETE -- not evaluated" | tee -a "$RESULTS"
            continue
        fi
    fi

    # settle so the final best_model_*.pt write is flushed
    sleep 20
    log "evaluating $w on the test split"
    bash "$SCRIPT_DIR/run_eval_deepsets.sh" "$tag" "$w" \
        2>&1 | tee -a "$LOG_DIR/eval_${w}.out"
    log "eval salloc for $w exited ${PIPESTATUS[0]}"

    npz=$(ls "$EVAL_DIR"/outputs_"${tag}"_top_test_rank*.npz 2>/dev/null | wc -l)
    log "found $npz per-rank npz for $tag"
    if [ "$npz" -eq 0 ]; then
        log "SKIP scoring $w: no eval npz."
        echo "  [$w] EVAL PRODUCED NO NPZ" | tee -a "$RESULTS"
        continue
    fi

    {
        echo
        echo "==== width=$w  ($tag)  $(date '+%Y-%m-%d %H:%M:%S') ===="
    } | tee -a "$RESULTS"
    "$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$tag" \
        2>&1 | tee -a "$RESULTS"
done

log "=== width-scan driver done. Full table in $RESULTS ==="
{
    echo
    echo "==== scan complete $(date '+%Y-%m-%d %H:%M:%S') -- full width ladder ===="
} | tee -a "$RESULTS"
for t in \
    distill_top_deepsets_nano_scratch_a05_T4 \
    distill_top_deepsets_distillnet_scratch_a05_T4 \
    distill_top_deepsets_micro_scratch_a05_T4 \
    distill_top_deepsets_tiny_scratch_a05_T4 \
    distill_top_deepsets_small_scratch_a05_T4_archfix0804
do
    "$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$t" \
        2>&1 | tee -a "$RESULTS" || true
done
