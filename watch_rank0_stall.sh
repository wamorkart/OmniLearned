#!/bin/bash
# Stall watcher for the recurring rank-0 NCCL hang (see
# distill-lazy-teacher-progress memory, 2026-08-08 entries). Flight-recorder
# dumps already proved rank 0 is stuck in ordinary Python/CPU code (it never
# even enqueues the stalling collective), not a GPU/NCCL-level fault -- but
# nobody has caught its live stack mid-stall yet. This tails a training
# session's log file and, if it goes quiet for longer than is normal between
# heartbeat prints (every 100 iters, ~45-120s in observed throughput), SSHes
# to rank 0's node and takes a py-spy snapshot before the DDP watchdog kills
# the job, so we can see exactly what rank 0 is blocked on.
#
# Usage: watch_rank0_stall.sh <LOG_FILE> <JOBID>
#   LOG_FILE -- the session's tee'd stdout log (same file the training loop writes)
#   JOBID    -- the salloc job ID (used to find rank 0's node and to know when
#               the job has ended, so the watcher can exit on its own)
#
# Designed to be launched in the background alongside the training loop, e.g.:
#   bash watch_rank0_stall.sh "$LOG_FILE" "$JOBID" &

set -u

LOG_FILE="$1"
JOBID="$2"
PYSPY=/global/homes/t/twamorka/omnilearned-clean/env/bin/py-spy
DUMP_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/rank0_stall_dumps
mkdir -p "$DUMP_DIR"

STALL_THRESHOLD_SEC=240   # normal heartbeat cadence observed ~45-120s; well above that
POLL_INTERVAL_SEC=30
COOLDOWN_SEC=120          # avoid spamming dumps for the same ongoing stall

last_dump_time=0

echo "[watcher] watching $LOG_FILE for job $JOBID (stall threshold ${STALL_THRESHOLD_SEC}s)"

while true; do
    STATE=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
    if [ -z "$STATE" ]; then
        echo "[watcher] job $JOBID no longer in queue -- exiting"
        exit 0
    fi
    if [ "$STATE" != "RUNNING" ]; then
        sleep "$POLL_INTERVAL_SEC"
        continue
    fi

    if [ ! -f "$LOG_FILE" ]; then
        sleep "$POLL_INTERVAL_SEC"
        continue
    fi

    now=$(date +%s)
    mtime=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo "$now")
    gap=$((now - mtime))

    if [ "$gap" -ge "$STALL_THRESHOLD_SEC" ] && [ $((now - last_dump_time)) -ge "$COOLDOWN_SEC" ]; then
        echo "[watcher] $(date '+%Y-%m-%d %H:%M:%S') log quiet for ${gap}s -- capturing rank 0 stack"
        last_dump_time=$now

        NODE=$(scontrol show hostnames "$(squeue -j "$JOBID" -h -o "%N")" 2>/dev/null | head -1)
        if [ -z "$NODE" ]; then
            echo "[watcher] could not resolve rank-0 node, skipping this attempt"
            sleep "$POLL_INTERVAL_SEC"
            continue
        fi

        DUMP_FILE="$DUMP_DIR/stall_${JOBID}_$(date '+%Y%m%d_%H%M%S').txt"
        ssh "$NODE" "
            for pid in \$(pgrep -u \$USER -f omnilearned); do
                if tr '\0' '\n' < /proc/\$pid/environ 2>/dev/null | grep -q '^SLURM_PROCID=0\$'; then
                    echo \"=== rank0 pid \$pid on $NODE ===\"
                    $PYSPY dump --pid \$pid --locals 2>&1
                fi
            done
        " > "$DUMP_FILE" 2>&1

        echo "[watcher] dump written to $DUMP_FILE"
        if [ -s "$DUMP_FILE" ]; then
            head -40 "$DUMP_FILE"
        else
            echo "[watcher] dump file empty -- rank0 pid not found or ssh/py-spy failed"
        fi
    fi

    sleep "$POLL_INTERVAL_SEC"
done
