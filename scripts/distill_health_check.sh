#!/bin/bash
# Periodic health-check watcher for the pretrain-distillation training loop.
# Runs alongside distill_loop.sh (separate screen session) and every
# CHECK_INTERVAL seconds writes a status snapshot to HEALTH_LOG:
#   - squeue state for the training job(s)
#   - summary.log tail (session count / exit codes)
#   - latest session log tail (grep for tracebacks)
#   - latest wandb loss / loss_kd (via wandb API) with a NaN check
#   - checkpoint file mtime/size (proof of progress)
#
# Log-only: does not attempt any outbound notification. Read HEALTH_LOG
# when you're back online.
#
# Run in a screen session:
#   screen -dmS distill_health bash /path/to/distill_health_check.sh

set -uo pipefail

SAVE_TAG=distill_pretrain_s_scratch_a05_b05_T4
WANDB_PROJECT=OmniBoone
CHECKPOINT=/pscratch/sd/t/twamorka/omnilearned/checkpoints/best_model_${SAVE_TAG}.pt
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop
HEALTH_LOG=/pscratch/sd/t/twamorka/omnilearned/results/distill_health_check.log
mkdir -p "$(dirname "$HEALTH_LOG")"

CHECK_INTERVAL=21600   # 6 hours
MAX_CHECKS=45          # ~11.25 days of coverage, matching the training loop's own margin

module load conda >/dev/null 2>&1
conda activate /global/homes/t/twamorka/omnilearned-clean/env >/dev/null 2>&1

check_once() {
    {
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="

        echo "--- squeue ---"
        squeue -u "$USER" 2>&1

        echo "--- summary.log (last 5) ---"
        tail -5 "$LOG_DIR/summary.log" 2>&1

        echo "--- latest session log: traceback/error grep ---"
        LATEST=$(ls -t "$LOG_DIR"/session_*.out 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "file: $LATEST"
            MATCHES=$(grep -iE "error|traceback|nan|exception" "$LATEST")
            if [ -n "$MATCHES" ]; then
                echo "$MATCHES" | tail -10
            else
                echo "(none found)"
            fi
        else
            echo "(no session logs yet)"
        fi

        echo "--- checkpoint file ---"
        if [ -f "$CHECKPOINT" ]; then
            stat --format="path=%n size=%s bytes mtime=%y" "$CHECKPOINT"
        else
            echo "checkpoint not found yet: $CHECKPOINT"
        fi

        echo "--- wandb latest loss ---"
        python3 - <<PYEOF
import wandb
try:
    api = wandb.Api()
    runs = api.runs("$WANDB_PROJECT", filters={"display_name": "$SAVE_TAG"})
    if len(runs) == 0:
        print("no wandb run found with name $SAVE_TAG")
    else:
        run = runs[0]
        hist = run.history(keys=["loss", "loss_kd", "epoch"], pandas=False)
        if not hist:
            print("wandb run found but no history rows yet")
        else:
            last = hist[-1]
            print(f"latest logged step: {last}")
            loss = last.get("loss")
            loss_kd = last.get("loss_kd")
            if loss is not None and (loss != loss):
                print("WARNING: loss is NaN")
            if loss_kd is not None and (loss_kd != loss_kd):
                print("WARNING: loss_kd is NaN")
except Exception as e:
    print(f"wandb check failed: {e}")
PYEOF

        echo
    } >> "$HEALTH_LOG" 2>&1
}

for i in $(seq 1 "$MAX_CHECKS"); do
    check_once
    sleep "$CHECK_INTERVAL"
done
