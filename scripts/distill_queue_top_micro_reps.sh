#!/bin/bash
# Queue manager for the 4 micro-student replicate runs (r1-r4).
#
# Keeps exactly 2 concurrent distill_loop_top_micro_rep.sh jobs running at
# all times (matching the 2 concurrent `-q interactive` salloc slots
# available), pulling the next queued tag as soon as a slot frees up.
#
# Run inside a screen/tmux session so it survives disconnects:
#   screen -S micro_reps bash distill_queue_top_micro_reps.sh
#
# Each distill_loop_top_micro_rep.sh internally resubmits on timeout and
# exits 0 once its 50-epoch run completes, so this manager just needs to
# keep 2 of them alive concurrently until the queue drains.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_micro
mkdir -p "$LOG_DIR"

QUEUE=(
    distill_top_micro_scratch_a05_T4_r1
    distill_top_micro_scratch_a05_T4_r2
    distill_top_micro_scratch_a05_T4_r3
    distill_top_micro_scratch_a05_T4_r4
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
    bash "$SCRIPT_DIR/distill_loop_top_micro_rep.sh" "$TAG" \
        > "$LOG_DIR/${TAG}_loop.out" 2>&1 &
    PID_TO_TAG[$!]="$TAG"
    return 0
}

log "=== Queue manager starting: ${#QUEUE[@]} reps, ${MAX_CONCURRENT} concurrent slots ==="

# Fill both slots initially
for _ in $(seq 1 "$MAX_CONCURRENT"); do
    launch_next
done

# As each background job exits, backfill from the queue until it's empty
# and no jobs remain running.
while [ "${#PID_TO_TAG[@]}" -gt 0 ]; do
    if wait -n 2>/dev/null; then
        STATUS=0
    else
        STATUS=$?
    fi

    # Find which PID(s) just finished
    for PID in "${!PID_TO_TAG[@]}"; do
        if ! kill -0 "$PID" 2>/dev/null; then
            TAG="${PID_TO_TAG[$PID]}"
            log "${TAG} (pid $PID) finished with status ${STATUS}"
            unset "PID_TO_TAG[$PID]"
            launch_next
        fi
    done
done

log "=== All 4 reps complete. Queue manager exiting. ==="
