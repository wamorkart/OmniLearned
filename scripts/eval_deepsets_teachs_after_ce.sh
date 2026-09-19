#!/bin/bash
# Chain wrapper: wait for the eval_deepsets_ce_spread screen session to
# finish (it's already using the single interactive slot per this project's
# default -- see feedback-default-single-interactive-slot memory), then run
# eval_deepsets_teachs_spread.sh on that same slot.
#
# Run inside a screen session:
#   screen -dmS eval_deepsets_teachs_after_ce bash eval_deepsets_teachs_after_ce.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%F %T')] Waiting for eval_deepsets_ce_spread screen session to finish..."
while screen -ls 2>/dev/null | grep -q eval_deepsets_ce_spread; do
    sleep 30
done
echo "[$(date '+%F %T')] eval_deepsets_ce_spread done. Starting teachS eval."

exec bash "$SCRIPT_DIR/eval_deepsets_teachs_spread.sh"
