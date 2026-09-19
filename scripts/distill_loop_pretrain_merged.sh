#!/bin/bash
# Loop wrapper for distill_train_pretrain_merged.sh -- tag
# distill_pretrain_s_scratch_a05_b05_T4_full500, reading the merged datasets on
# $SCRATCH. Target --epoch 500 (the real full-pass target). At ~438s/epoch-unit
# measured throughput and a 240-min interactive session cap, a session packs
# ~33 epochs ideally but realistically more like 10-25 once queue
# wait/DDP-init/checkpoint-restore overhead and occasional zero-progress
# sessions (observed ~20% hang-on-resume rate pre-wandb-fix) are factored in --
# pessimistic case needs 50+ resubmits for the full 500 epoch-units. MAX_LOOPS
# set well above that since running out just stalls the loop until someone
# notices, at no other cost (it exits 0 on its own the moment --epoch 500 is
# reached, regardless of how high this is set).
#
# NOTE: a 48-GPU/-q regular variant (12 nodes, --epoch 167) was drafted and
# reverted 2026-08-09 -- it briefly overwrote this file while job 56540200
# (this exact 16-GPU/interactive config, launched 11:03) was already live, an
# accident, not intentional. If revisiting 48-GPU scaling, use a NEW script
# filename, never edit this one while its own loop might still be running.
#
# Same set +e/-e guard around the salloc|tee pipeline as distill_loop.sh and
# distill_loop_pretrain_relaunch.sh: under `set -e` + `pipefail`, a non-zero
# salloc exit would otherwise kill the script before the resubmit logic runs.
#
# Run inside a screen session so the loop survives terminal disconnects:
#   screen -dmS distill_pretrain_merged bash distill_loop_pretrain_merged.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_pretrain_merged
mkdir -p "$LOG_DIR"

MAX_LOOPS=100
LOOP=0

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === Session ${LOOP} / ${MAX_LOOPS} ===" | tee -a "$LOG_DIR/summary.log"

    # Rank-0 stall watcher (2026-08-08): waits for this session's salloc jobid
    # to show up in the log, then hands off to watch_rank0_stall.sh, which
    # py-spy-dumps rank 0's live stack if the log goes quiet -- catches the
    # recurring hang in the act instead of only reconstructing it afterward.
    # Self-terminates once the job leaves the queue; see that script for detail.
    (
        set +e
        for _ in $(seq 1 60); do
            WATCH_JOBID=$(grep -oP 'Granted job allocation \K[0-9]+' "$LOG_FILE" 2>/dev/null | head -1)
            if [ -n "$WATCH_JOBID" ]; then
                bash "$SCRIPT_DIR/watch_rank0_stall.sh" "$LOG_FILE" "$WATCH_JOBID" \
                    >> "$LOG_DIR/watcher_session_${LOOP}.log" 2>&1
                break
            fi
            sleep 5
        done
    ) &
    WATCHER_BGPID=$!

    set +e
    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/distill_train_pretrain_merged.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"
    set -e
    kill "$WATCHER_BGPID" 2>/dev/null || true

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Run completed (target epoch reached). Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit, preemption, or crash). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS} without a clean exit -- check logs in $LOG_DIR." \
    | tee -a "$LOG_DIR/summary.log"
exit 1
