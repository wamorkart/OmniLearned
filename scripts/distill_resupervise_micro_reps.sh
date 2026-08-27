#!/bin/bash
# Resupervisor for the micro-student replicate runs (r2/r3/r4) after the
# original distill_queue_top_micro_reps.sh queue manager died mid-run
# (2026-07-18): its screen/process was gone, but r3/r4's SLURM jobs kept
# running as orphans since salloc/srun don't need a live parent once
# dispatched. This script takes over: treats the still-running r3/r4 jobs
# as occupying the 2 concurrent -q interactive slots, queues r2 to launch
# as soon as a slot frees, and re-queues r3/r4 too if they get killed by
# walltime before reaching epoch 50/50 (checkpoints make resubmission via
# --resuming cheap - each rep only needs a few more epochs).
#
# Run inside a screen session so it survives disconnects:
#   screen -S micro_reps_resupervise bash distill_resupervise_micro_reps.sh

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_micro
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/resupervise.log"
}

declare -A TAG=(
    [r2]=distill_top_micro_scratch_a05_T4_r2
    [r3]=distill_top_micro_scratch_a05_T4_r3
    [r4]=distill_top_micro_scratch_a05_T4_r4
)

# SLURM jobs already running for r3/r4 at the moment this script starts,
# whose supervising bash loop is dead. Occupy slots until they leave squeue.
declare -A ORPHAN_JOB=( [56107349]=r3 [56107115]=r4 )

QUEUE=(r2)

declare -A PID_TO_KEY
MAX_CONCURRENT=2

is_complete() {
    local full="$1"
    local lf="$LOG_DIR/${full}_summary.log"
    if [ -f "$lf" ] && grep -q "Training completed" "$lf"; then
        return 0
    fi
    local latest
    latest=$(ls -t "$LOG_DIR/${full}_session_"*.out 2>/dev/null | head -1)
    if [ -n "$latest" ] && grep -q "Epoch \[50/50\]" "$latest"; then
        return 0
    fi
    return 1
}

occupied() {
    echo $(( ${#ORPHAN_JOB[@]} + ${#PID_TO_KEY[@]} ))
}

launch() {
    local key="$1"
    local full="${TAG[$key]}"
    log "Launching ${full} (slot free)"
    bash "$SCRIPT_DIR/distill_loop_top_micro_rep.sh" "$full" \
        > "$LOG_DIR/${full}_loop.out" 2>&1 &
    PID_TO_KEY[$!]="$key"
}

log "=== Resupervisor starting. Orphan jobs: ${!ORPHAN_JOB[*]}. Queue: ${QUEUE[*]:-<empty>} ==="

while :; do
    for pid in "${!PID_TO_KEY[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            key="${PID_TO_KEY[$pid]}"
            full="${TAG[$key]}"
            wait "$pid"; status=$?
            log "${full} (pid $pid) loop exited status ${status}"
            unset "PID_TO_KEY[$pid]"
            if is_complete "$full"; then
                log "${full} confirmed complete."
            else
                log "${full} not complete -> re-queueing"
                QUEUE+=("$key")
            fi
        fi
    done

    for jid in "${!ORPHAN_JOB[@]}"; do
        if ! squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; then
            key="${ORPHAN_JOB[$jid]}"
            full="${TAG[$key]}"
            log "Orphan job ${jid} (${full}) left the queue."
            unset "ORPHAN_JOB[$jid]"
            if is_complete "$full"; then
                log "${full} confirmed complete."
            else
                log "${full} not complete -> re-queueing"
                QUEUE+=("$key")
            fi
        fi
    done

    while [ "$(occupied)" -lt "$MAX_CONCURRENT" ] && [ "${#QUEUE[@]}" -gt 0 ]; do
        next="${QUEUE[0]}"
        QUEUE=("${QUEUE[@]:1}")
        launch "$next"
    done

    if [ "${#ORPHAN_JOB[@]}" -eq 0 ] && [ "${#PID_TO_KEY[@]}" -eq 0 ] && [ "${#QUEUE[@]}" -eq 0 ]; then
        log "=== All reps (r2, r3, r4) complete. Resupervisor exiting. ==="
        break
    fi

    sleep 30
done
