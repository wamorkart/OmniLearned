#!/bin/bash
# Queue manager for the 5 CE-only micro-student replicate runs (r1-r5).
#
# CE-only counterpart to distill_queue_top_micro_reps.sh: keeps exactly 2
# concurrent train_top_micro_ce_loop_rep.sh jobs running at all times
# (matching the 2 concurrent `-q interactive` salloc slots available),
# pulling the next queued tag as soon as a slot frees up.
#
# Run inside a screen/tmux session so it survives disconnects:
#   screen -S micro_ce_reps bash queue_top_micro_ce_reps.sh
#
# Each train_top_micro_ce_loop_rep.sh internally resubmits on timeout and
# exits 0 once its 50-epoch run completes, so this manager just needs to
# keep 2 of them alive concurrently until the queue drains.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/train_top_micro_ce
mkdir -p "$LOG_DIR"

QUEUE=(
    train_top_micro_ce_scratch_r1
    train_top_micro_ce_scratch_r2
    train_top_micro_ce_scratch_r3
    train_top_micro_ce_scratch_r4
    train_top_micro_ce_scratch_r5
)
MAX_CONCURRENT=2

declare -A PID_TO_TAG
QUEUE_IDX=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/queue_manager.log"
}

launch_next() {
    if [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
        return 1
    fi
    TAG="${QUEUE[$QUEUE_IDX]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))
    log "Launching ${TAG} (slot free)"
    bash "$SCRIPT_DIR/train_top_micro_ce_loop_rep.sh" "$TAG" \
        > "$LOG_DIR/${TAG}_loop.out" 2>&1 &
    PID_TO_TAG[$!]="$TAG"
    return 0
}

log "=== Queue manager starting: ${#QUEUE[@]} reps, ${MAX_CONCURRENT} concurrent slots ==="

for _ in $(seq 1 "$MAX_CONCURRENT"); do
    launch_next
done

while [ "${#PID_TO_TAG[@]}" -gt 0 ]; do
    if wait -n 2>/dev/null; then
        STATUS=0
    else
        STATUS=$?
    fi

    for PID in "${!PID_TO_TAG[@]}"; do
        if ! kill -0 "$PID" 2>/dev/null; then
            TAG="${PID_TO_TAG[$PID]}"
            log "${TAG} (pid $PID) finished with status ${STATUS}"
            unset "PID_TO_TAG[$PID]"
            launch_next
        fi
    done
done

log "=== All 5 CE reps complete. Queue manager exiting. ==="
