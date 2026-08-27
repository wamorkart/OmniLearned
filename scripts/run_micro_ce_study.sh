#!/bin/bash
# End-to-end orchestrator for the CE-only micro-student baseline study:
#   1. Train 5 independent CE-only micro reps (queue_top_micro_ce_reps.sh,
#      2 concurrent -q interactive slots, auto-resubmit on walltime).
#   2. Evaluate each rep on the top-tagging test split (2 concurrent slots,
#      retry once on transient failure).
#   3. Compute metrics and print the final comparison table against the
#      5-rep KD-micro study and the small/teacher benchmarks.
#
# Run inside a screen session so it survives disconnects:
#   screen -S micro_ce_study bash run_micro_ce_study.sh
#
# Progress/results land in:
#   /pscratch/sd/t/twamorka/omnilearned/logs/train_top_micro_ce/   (training logs)
#   /pscratch/sd/t/twamorka/omnilearned/eval/top_micro_ce/         (eval outputs)
#   /pscratch/sd/t/twamorka/omnilearned/results/micro_ce_vs_kd.txt (final table)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/train_top_micro_ce
RESULTS_DIR=/pscratch/sd/t/twamorka/omnilearned/results
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/orchestrator.log"
}

REPS=(r1 r2 r3 r4 r5)

# ---------------------------------------------------------------------------
# Stage 1: training
# ---------------------------------------------------------------------------
log "=== Stage 1: training 5 CE-only micro reps ==="
bash "$SCRIPT_DIR/queue_top_micro_ce_reps.sh"
log "=== Stage 1 done: all reps trained ==="

# ---------------------------------------------------------------------------
# Stage 2: evaluation (2 concurrent slots, retry once on failure)
# ---------------------------------------------------------------------------
log "=== Stage 2: evaluating 5 CE-only micro reps ==="

eval_one() {
    local rep="$1"
    local tag="train_top_micro_ce_scratch_${rep}"
    local attempt
    for attempt in 1 2; do
        log "Evaluating ${tag} (attempt ${attempt})"
        SAVE_TAG="$tag" salloc -C gpu -q interactive -t 60 --nodes 4 \
            --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
            bash -c "SAVE_TAG=$tag bash $SCRIPT_DIR/evaluate_top_micro_ce.sh" \
            > "$LOG_DIR/${tag}_eval_attempt${attempt}.log" 2>&1
        local status=$?
        local n
        n=$(ls /pscratch/sd/t/twamorka/omnilearned/eval/top_micro_ce/outputs_${tag}_top_test_rank*.npz 2>/dev/null | wc -l)
        if [ "$status" -eq 0 ] && [ "$n" -eq 16 ]; then
            log "${tag} eval OK (16/16 rank files)"
            return 0
        fi
        log "${tag} eval attempt ${attempt} failed (status=${status}, ${n}/16 rank files)"
    done
    log "${tag} eval FAILED after 2 attempts"
    return 1
}

# 2 concurrent, matching the interactive-queue slot limit
eval_one r1 & P1=$!
eval_one r2 & P2=$!
wait "$P1"; wait "$P2"

eval_one r3 & P1=$!
eval_one r4 & P2=$!
wait "$P1"; wait "$P2"

eval_one r5
log "=== Stage 2 done: evaluation complete ==="

# ---------------------------------------------------------------------------
# Stage 3: metrics + comparison table
# ---------------------------------------------------------------------------
log "=== Stage 3: computing metrics and comparison table ==="

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env

python "$SCRIPT_DIR/../compare_micro_ce_vs_kd.py" | tee "$RESULTS_DIR/micro_ce_vs_kd.txt"

log "=== Stage 3 done. Results: $RESULTS_DIR/micro_ce_vs_kd.txt ==="
log "=== STUDY COMPLETE ==="
