#!/bin/bash
# Chains the CE-only and teachS DeepSets spread evaluations sequentially
# (one interactive salloc slot at a time) so both spreads get computed
# without exceeding the default single-interactive-slot budget.
#
# Run inside a screen session:
#   screen -dmS deepsets_spreads bash run_deepsets_spreads.sh
#   screen -r deepsets_spreads   # to reattach

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%F %T')] Starting CE-only spread eval"
bash "$SCRIPT_DIR/eval_deepsets_ce_spread.sh"

echo "[$(date '+%F %T')] Starting teachS spread eval"
bash "$SCRIPT_DIR/eval_deepsets_teachs_spread.sh"

echo "[$(date '+%F %T')] Both spreads done."
