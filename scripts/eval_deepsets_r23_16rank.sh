#!/bin/bash
# One-off (2026-08-27): re-run the DeepSets r2/r3 top-tagging test-split evals
# at 16 ranks (4 nodes x 4 GPU), two evals in parallel across BOTH interactive
# slots, then compute the base+r2+r3 accuracy/AUC spread for the ce and teachS
# recipes.
#
#   screen -dmS deepsets_r23_eval bash scripts/eval_deepsets_r23_16rank.sh
#   screen -r deepsets_r23_eval          # to reattach
#
# Each eval is verified by fresh per-rank npz files, not salloc's exit code:
# salloc can sit PENDING, get revoked, and STILL exit 0 (caught 2026-08-25).
# A tag is CONFIRMED only once >= MIN_FRESH rank files newer than the attempt
# start exist; otherwise it retries a fresh salloc up to MAX_RETRIES times.
# On confirm, any older rank files for that tag are purged so the metrics
# glob sees only this run's output.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/scripts/evaluate_top_distill_deepsets.sh"
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_deepsets_r23_16rank
PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
mkdir -p "$LOG_DIR"

# All size=small. r2/r3 of both recipes the user has replicates for.
R23_TAGS=(
    distill_top_deepsets_small_scratch_teachS_a05_T4_r2
    distill_top_deepsets_small_scratch_teachS_a05_T4_r3
    train_top_deepsets_small_ce_scratch_r2
    train_top_deepsets_small_ce_scratch_r3
)
SIZE=small
MAX_CONCURRENT=2      # user asked to use both interactive slots
MAX_RETRIES=30        # high: the GPU partition was DOWN at launch (maintenance);
RETRY_SLEEP=180       # a revoked salloc should back off and keep trying for hours
MIN_FRESH=14          # a good run writes 16 (4 nodes x 4 tasks); allow slack
POLL_SECONDS=30

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/driver.log"; }

eval_one() {
    local tag="$1" attempt=1 start nfresh
    while :; do
        start=$(date +%s)
        log "$tag: eval attempt ${attempt}/${MAX_RETRIES} (16 ranks)"
        SAVE_TAG="$tag" SIZE="$SIZE" salloc \
            -C gpu -q interactive -t 60 \
            --nodes 4 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
            bash "$EVAL_SCRIPT" \
            > "$LOG_DIR/${tag}_attempt${attempt}.out" 2>&1
        nfresh=$(find "$OUTPUT_DIR" -maxdepth 1 \
                    -name "outputs_${tag}_top_test_rank*.npz" \
                    -newermt "@${start}" 2>/dev/null | wc -l)
        if [ "$nfresh" -ge "$MIN_FRESH" ]; then
            find "$OUTPUT_DIR" -maxdepth 1 \
                 -name "outputs_${tag}_top_test_rank*.npz" \
                 ! -newermt "@${start}" -delete 2>/dev/null
            log "$tag: CONFIRMED (${nfresh} fresh rank files)"
            return 0
        fi
        log "$tag: attempt ${attempt} produced ${nfresh} fresh files (<${MIN_FRESH}) -- salloc likely failed to allocate; see ${tag}_attempt${attempt}.out"
        if [ "$attempt" -ge "$MAX_RETRIES" ]; then
            log "$tag: FAILED after ${MAX_RETRIES} attempts -- giving up (spread will skip it)"
            return 1
        fi
        attempt=$((attempt + 1))
        sleep "$RETRY_SLEEP"
    done
}

log "=== start: ${#R23_TAGS[@]} evals, ${MAX_CONCURRENT} concurrent (both interactive slots) ==="

declare -A PID_TAG
idx=0
launch_next() {
    [ "$idx" -lt "${#R23_TAGS[@]}" ] || return 1
    local tag="${R23_TAGS[$idx]}"
    idx=$((idx + 1))
    eval_one "$tag" &
    PID_TAG[$!]="$tag"
    log "launched $tag (pid $!)"
    sleep 20     # let this salloc register before the next slot is filled
}

for ((i = 0; i < MAX_CONCURRENT; i++)); do launch_next; done

while [ "${#PID_TAG[@]}" -gt 0 ]; do
    sleep "$POLL_SECONDS"
    for pid in "${!PID_TAG[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        wait "$pid"; rc=$?
        log "${PID_TAG[$pid]}: worker exited rc=${rc}"
        unset "PID_TAG[$pid]"
        launch_next
    done
done

log "=== all evals attempted; computing spreads ==="

log "--- teachS spread (base + r2 + r3) ---"
"$PY" "$REPO_ROOT/compute_metrics_top.py" --indir "$OUTPUT_DIR" \
    --tag distill_top_deepsets_small_scratch_teachS_a05_T4 \
    --tag distill_top_deepsets_small_scratch_teachS_a05_T4_r2 \
    --tag distill_top_deepsets_small_scratch_teachS_a05_T4_r3 \
    2>&1 | tee "$LOG_DIR/spread_teachS.log"

log "--- ce spread (base + r2 + r3) ---"
"$PY" "$REPO_ROOT/compute_metrics_top.py" --indir "$OUTPUT_DIR" \
    --tag train_top_deepsets_small_ce_scratch \
    --tag train_top_deepsets_small_ce_scratch_r2 \
    --tag train_top_deepsets_small_ce_scratch_r3 \
    2>&1 | tee "$LOG_DIR/spread_ce.log"

log "=== done. spreads in ${LOG_DIR}/spread_{teachS,ce}.log ==="
