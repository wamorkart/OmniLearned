#!/bin/bash
# Watcher / manager for the temperature sweep on top-tagging distillation.
#
# Manages training in three phases (2 concurrent runs each), then runs eval+metrics.
#
#   Phase 1: T=1  +  T=2
#   Phase 2: T=8  +  T=16   (auto-launched when phase 1 done)
#   Phase 3: T=4 r1 + T=4 r2  (auto-launched when phase 2 done)
#   Eval:    evaluate_metrics_top_T_sweep.sh  (auto-launched when phase 3 done)
#
# Run in a screen session on a login node:
#   screen -S watch_T_sweep bash /path/to/watch_T_sweep.sh
#
# The watcher respects already-done runs (reads summary.log for "Training completed:").
# Check distill_status_T_sweep.sh at any time for a live status snapshot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_SWEEP=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep
RESULTS_DIR=/pscratch/sd/t/twamorka/omnilearned/results
EVAL_LOG="$RESULTS_DIR/top_T_sweep_eval.log"
WATCHER_LOG="$LOG_SWEEP/watch_T_sweep.log"
mkdir -p "$LOG_SWEEP" "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"; }

is_done() {
    local logfile="$1"
    [ -f "$logfile" ] && grep -q "Training completed:" "$logfile"
}

# Launch a T-sweep run in a named screen session, unless already done or already running.
# Args: session_name  T  [rep_suffix]
launch() {
    local session="$1"
    local T="$2"
    local suffix="${3:-}"
    local tag="distill_top_small_scratch_a00_b10_T${T}${suffix:+_$suffix}"
    local logfile="$LOG_SWEEP/$tag/summary.log"

    if is_done "$logfile"; then
        return 0
    fi
    if screen -ls 2>/dev/null | grep -q "\.$session "; then
        return 0
    fi
    log "$tag: (re)launching screen session '$session'..."
    mkdir -p "$LOG_SWEEP/$tag"
    screen -dmS "$session" bash -c \
        "cd '$SCRIPT_DIR' && bash distill_loop_top_T_sweep.sh '$T' '$suffix' \
         2>&1 | tee '$LOG_SWEEP/$tag/watch_launch.log'"
    log "$tag: screen session '$session' launched"
}

# Poll every 5 minutes until both runs are done.
# Re-calls launch() on every tick for any run that isn't done — this auto-recovers
# from crashed screen sessions (launch is idempotent: no-op if already running/done).
# Args: sess1 T1 suf1  sess2 T2 suf2
wait_both() {
    local sess1="$1" T1="$2" suf1="$3"
    local sess2="$4" T2="$5" suf2="$6"
    local tag1="distill_top_small_scratch_a00_b10_T${T1}${suf1:+_$suf1}"
    local tag2="distill_top_small_scratch_a00_b10_T${T2}${suf2:+_$suf2}"
    local log1="$LOG_SWEEP/$tag1/summary.log"
    local log2="$LOG_SWEEP/$tag2/summary.log"

    while true; do
        local d1=false d2=false
        is_done "$log1" && d1=true
        is_done "$log2" && d2=true
        log "$tag1 done=$d1  |  $tag2 done=$d2"
        $d1 && $d2 && return 0

        # Relaunch any session that died without completing
        $d1 || launch "$sess1" "$T1" "$suf1"
        $d2 || launch "$sess2" "$T2" "$suf2"

        sleep 300
    done
}

log "=== T-sweep watcher started ==="
log "Phases: [T1+T2] → [T8+T16] → [T4_r1+T4_r2] → eval"

# ── Phase 1: T=1 and T=2 ──────────────────────────────────────────────────────
log "--- Phase 1: T=1 and T=2 ---"
wait_both "top_T1" 1 "" "top_T2" 2 ""
log "Phase 1 complete."

# ── Phase 2: T=8 and T=16 ────────────────────────────────────────────────────
log "--- Phase 2: T=8 and T=16 ---"
wait_both "top_T8" 8 "" "top_T16" 16 ""
log "Phase 2 complete."

# ── Phase 3: T=4 r1 and T=4 r2 ───────────────────────────────────────────────
log "--- Phase 3: T=4 r1 and T=4 r2 ---"
wait_both "top_T4r1" 4 r1 "top_T4r2" 4 r2
log "Phase 3 complete."

# ── Eval + metrics ────────────────────────────────────────────────────────────
log "--- All training done. Grabbing salloc for eval+metrics... ---"
salloc \
    -C gpu -q interactive -t 120 \
    --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 \
    -A m3246 \
    bash "$SCRIPT_DIR/evaluate_metrics_top_T_sweep.sh" \
    2>&1 | tee "$EVAL_LOG"

log "Eval complete. Results saved to $EVAL_LOG"
log "=== T-sweep watcher done ==="
