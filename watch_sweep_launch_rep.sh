#!/bin/bash
# Watches the two sweep runs (a025_b075 and a00_b10) and launches r1 and r2
# in new screen sessions once both are done.
#
# Run in its own screen session on login25:
#   screen -dmS watcher bash /path/to/watch_sweep_launch_rep.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_SWEEP=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep

LOG_A025="$LOG_SWEEP/distill_top_small_scratch_a025_b075_T4/summary.log"
LOG_A00="$LOG_SWEEP/distill_top_small_scratch_a00_b10_T4/summary.log"

WATCHER_LOG="$LOG_SWEEP/watcher_rep_r1r2.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"; }

log "Watcher started. Monitoring:"
log "  $LOG_A025"
log "  $LOG_A00"
log "Will launch r1 and r2 when both complete."

check_done() {
    local logfile="$1"
    [ -f "$logfile" ] && grep -q "Training completed:" "$logfile"
}

while true; do
    done_a025=false; done_a00=false
    check_done "$LOG_A025" && done_a025=true
    check_done "$LOG_A00"  && done_a00=true

    log "a025_b075 done: $done_a025 | a00_b10 done: $done_a00"

    if $done_a025 && $done_a00; then
        log "Both sweep runs finished. Launching r1 and r2..."
        screen -dmS top_r1 bash -c "cd '$SCRIPT_DIR' && bash distill_loop_top_rep.sh distill_top_small_scratch_a05_T4_r1"
        screen -dmS top_r2 bash -c "cd '$SCRIPT_DIR' && bash distill_loop_top_rep.sh distill_top_small_scratch_a05_T4_r2"
        log "Launched screen sessions: top_r1, top_r2"
        log "Watcher done. Exiting."
        exit 0
    fi

    sleep 300  # poll every 5 minutes
done
