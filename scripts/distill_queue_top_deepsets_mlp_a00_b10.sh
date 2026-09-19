#!/bin/bash
# Queue manager for the two pure-KD (alpha=0/beta=1) architecture-ablation
# reruns: DeepSets and minimal-MLP KD students on top tagging.
#
# Rationale: the existing DeepSets/MLP KD checkpoints were both trained at
# alpha=0.5/beta=0.5 (the old default), never at alpha=0/beta=1 -- the
# operating point this project's own T-sweep found was the outright winner
# for the PET2 student (AUC 0.9879 vs 0.9875), and the operating point used
# by the DistillNet reference (arXiv:2311.12551, pure teacher-transfer loss,
# no ground-truth term). Rerunning both isolates "architecture cost" from
# "loss-weighting cost", and doubles as an A/B check on why the MLP student's
# a05/b05 eval came out near-random (AUC 0.626).
#
# IMPORTANT (learned the hard way 2026-08-13): NERSC's gpu_interactive QOS
# enforces MaxSubmitJobPerUserLimit=2 on *submitted* jobs (running OR
# pending), not just running ones. A 3rd salloc call doesn't queue behind
# the first two -- it's rejected outright at submit time
# ("QOSMaxSubmitJobPerUserLimit"). The first version of this script assumed
# rejected submissions would gracefully queue like a normal SLURM pending
# job; instead the 2nd loop script's own salloc-retry-every-15s logic just
# hammered the scheduler with rejected submissions until it exhausted
# MAX_LOOPS and gave up for good. Fixed by gating each launch on the
# account-wide submitted-job count via squeue (counts ALL of this user's
# gpu_interactive jobs, including ones this manager didn't launch, e.g. an
# unrelated medium-model run already using a slot) before calling salloc,
# so a launch is only attempted once it can actually succeed.
#
# Run inside a screen/tmux session so it survives disconnects:
#   screen -S deepsets_mlp_a00_b10 bash scripts/distill_queue_top_deepsets_mlp_a00_b10.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_queue_top_deepsets_mlp_a00_b10
mkdir -p "$LOG_DIR"

# "loop_script[:CONFIG]" -- CONFIG (if present) is exported for the loop script.
# Both runs are configs of distill_loop_top.sh + configs/train/: deepsets
# a00_b10 via top_deepsets_a00_b10.sh, mlp a00_b10 via top_mlp_a00_b10.sh.
QUEUE=(
    "distill_loop_top.sh:top_deepsets_a00_b10"
    "distill_loop_top.sh:top_mlp_a00_b10"
)
MAX_CONCURRENT=2
MAX_SUBMIT_PER_USER=2
POLL_SECONDS=30

declare -A PID_TO_SCRIPT
QUEUE_IDX=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/queue_manager.log"
}

# Blocks until the user's account-wide gpu_interactive submitted-job count
# (running+pending, from ALL sources, not just this manager) leaves room for
# one more submission.
wait_for_slot() {
    while true; do
        COUNT=$(squeue -u "$USER" -h -o "%q" 2>/dev/null | grep -c '^gpu_interactive$')
        if [ "$COUNT" -lt "$MAX_SUBMIT_PER_USER" ]; then
            return 0
        fi
        log "No free gpu_interactive submit slot (${COUNT}/${MAX_SUBMIT_PER_USER} in use account-wide) -- waiting ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
    done
}

launch_next() {
    if [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
        return 1
    fi
    ENTRY="${QUEUE[$QUEUE_IDX]}"
    SCRIPT="${ENTRY%%:*}"
    CONFIG="${ENTRY#*:}"; [ "$CONFIG" = "$ENTRY" ] && CONFIG=""
    QUEUE_IDX=$((QUEUE_IDX + 1))
    wait_for_slot
    log "Launching ${SCRIPT}${CONFIG:+ (CONFIG=$CONFIG)}"
    CONFIG="$CONFIG" bash "$SCRIPT_DIR/${SCRIPT}" \
        > "$LOG_DIR/${SCRIPT%.sh}${CONFIG:+_$CONFIG}_loop.out" 2>&1 &
    PID_TO_SCRIPT[$!]="$SCRIPT"
    # Give the just-launched loop script's salloc call time to actually
    # register in squeue before the next launch_next's wait_for_slot checks
    # the count again -- without this delay, two launch_next calls in quick
    # succession can both see the pre-submission count and both proceed,
    # and the 2nd salloc gets rejected by QOSMaxSubmitJobPerUserLimit
    # (hit this exact race 2026-08-13).
    sleep 20
    return 0
}

log "=== Queue manager starting: ${#QUEUE[@]} runs, ${MAX_CONCURRENT} concurrent slots ==="

for _ in $(seq 1 "$MAX_CONCURRENT"); do
    launch_next
done

while [ "${#PID_TO_SCRIPT[@]}" -gt 0 ]; do
    if wait -n 2>/dev/null; then
        STATUS=0
    else
        STATUS=$?
    fi

    for PID in "${!PID_TO_SCRIPT[@]}"; do
        if ! kill -0 "$PID" 2>/dev/null; then
            SCRIPT="${PID_TO_SCRIPT[$PID]}"
            log "${SCRIPT} (pid $PID) finished with status ${STATUS}"
            unset "PID_TO_SCRIPT[$PID]"
            launch_next
        fi
    done
done

log "=== Both runs complete. Queue manager exiting. ==="
