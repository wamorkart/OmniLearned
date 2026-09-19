#!/bin/bash
# Resubmit loop for the alpha/beta (+ T) sweep on PET2-small top KD.
# Args: ALPHA BETA [T]   (T defaults to 4). Overrides the a/b/T of
# config top_small_a05 and derives a matching save-tag.
#   tmux new-session -s sweep_a025_b075 'bash scripts/distill_loop_top_sweep.sh 0.25 0.75'
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHA="${1:?Usage: $0 ALPHA BETA [T]}"
BETA="${2:?Usage: $0 ALPHA BETA [T]}"
DISTILL_T="${3:-4}"
SAVE_TAG=$(python3 -c "
a, b, t = '$ALPHA', '$BETA', '$DISTILL_T'
f = lambda x: str(float(x)).replace('.', '')
print(f'distill_top_small_scratch_a{f(a)}_b{f(b)}_T{int(float(t))}')
")
export ALPHA BETA DISTILL_T SAVE_TAG
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/$SAVE_TAG
export MAX_LOOPS="${MAX_LOOPS:-50}"
echo "Starting sweep run: ALPHA=$ALPHA BETA=$BETA T=$DISTILL_T SAVE_TAG=$SAVE_TAG"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_small_a05
