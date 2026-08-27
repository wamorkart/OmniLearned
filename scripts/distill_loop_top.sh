#!/bin/bash
# Resubmit loop for any top-tagging KD config under configs/train/.
#
#   bash scripts/distill_loop_top.sh [<config>]          # default: top_small_a05
#   MAX_LOOPS=30 bash scripts/distill_loop_top.sh top_micro_a05
#   screen -dmS distill_top bash scripts/distill_loop_top.sh top_medium_a00_b10
#
# Thin shim over lib/resubmit_loop.sh + run_train.sh: looks up the per-config
# log dir and resubmit cap in the table below (override either with the
# LOOP_LOG_DIR / MAX_LOOPS env vars), then hands off.
#
# Replaces the old one-per-config shims distill_loop_top{,_medium,
# _medium_a05_b05_T4,_micro,_mlp,_mlp_a00_b10,_small_via_medium}.sh. The
# arg-taking variants (_rep, _micro_rep, _micro_noint_rep, _sweep) are still
# their own scripts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Config comes from $1, or from $CONFIG for distill_queue_*'s "script:CONFIG"
# queue entries.
CONFIG="${1:-${CONFIG:-top_small_a05}}"

LOG_BASE=/pscratch/sd/t/twamorka/omnilearned/logs
SWEEP="$LOG_BASE/distill_loop_top_sweep"
# config -> "<log dir>|<max loops>"
declare -A LOOP_CFG=(
    [top_small_a05]="$LOG_BASE/distill_loop_top|20"
    [top_micro_a05]="$LOG_BASE/distill_loop_top_micro|20"
    [top_micro_noint_a05]="$LOG_BASE/distill_loop_top_micro_noint|20"
    [top_mlp_a05]="$LOG_BASE/distill_loop_top_mlp|8"
    [top_mlp_a00_b10]="$LOG_BASE/distill_loop_top_mlp_a00_b10|8"
    [top_medium_a00_b10]="$SWEEP/distill_top_medium_scratch_a00_b10_T4|50"
    [top_medium_a05_b05]="$SWEEP/distill_top_medium_scratch_a05_b05_T4|50"
    [top_small_via_medium_a00_b10]="$SWEEP/distill_top_small_via_medium_a00_b10_T4|50"
)

entry="${LOOP_CFG[$CONFIG]:-}"
if [ -n "$entry" ]; then
    export LOOP_LOG_DIR="${LOOP_LOG_DIR:-${entry%|*}}"
    export MAX_LOOPS="${MAX_LOOPS:-${entry#*|}}"
else
    # Unknown config (e.g. a new configs/train/top_*.sh with no row yet):
    # sensible defaults, still overridable via env.
    echo "distill_loop_top.sh: no table row for '$CONFIG', using defaults" >&2
    export LOOP_LOG_DIR="${LOOP_LOG_DIR:-$LOG_BASE/distill_loop_$CONFIG}"
    export MAX_LOOPS="${MAX_LOOPS:-20}"
fi

exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" "$CONFIG"
