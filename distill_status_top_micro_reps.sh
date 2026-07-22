#!/bin/bash
# Quick status dashboard for the micro-student replicate runs (r1-r4). No GPU
# needed. Run any time to see training/eval progress and queue-manager state:
#   bash distill_status_top_micro_reps.sh
#
# Claude can also run this via the Bash tool to monitor on your behalf.

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_micro
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_micro

echo "=== Micro-student rep status [$(date '+%Y-%m-%d %H:%M:%S')] ==="
echo ""

python3 - <<PYEOF
import json, glob, os

CHECKPOINT_DIR = "$CHECKPOINT_DIR"
EVAL_DIR       = "$EVAL_DIR"
LOG_DIR        = "$LOG_DIR"

TAGS = [
    ("distill_top_micro_scratch_a05_T4",    "orig [done]"),
    ("distill_top_micro_scratch_a05_T4_r1", "r1"),
    ("distill_top_micro_scratch_a05_T4_r2", "r2"),
    ("distill_top_micro_scratch_a05_T4_r3", "r3"),
    ("distill_top_micro_scratch_a05_T4_r4", "r4"),
]

def epochs_done(tag):
    p = f"{CHECKPOINT_DIR}/training_{tag}.json"
    if not os.path.exists(p):
        return 0
    try:
        d = json.load(open(p))
        return len(d.get("train_loss", []))
    except Exception:
        return 0

def train_complete(tag):
    lf = f"{LOG_DIR}/{tag}_summary.log"
    return os.path.exists(lf) and "Training completed" in open(lf).read()

def eval_done(tag):
    return bool(glob.glob(f"{EVAL_DIR}/outputs_{tag}_top_test_rank0.npz"))

def ckpt_exists(tag):
    return os.path.exists(f"{CHECKPOINT_DIR}/best_model_{tag}.pt")

print(f"  {'Label':<12}  {'Tag':<40}  Status")
print(f"  {'-'*12}  {'-'*40}  {'-'*26}")

for tag, label in TAGS:
    ep   = epochs_done(tag)
    done = train_complete(tag) or ep >= 50
    evl  = eval_done(tag)
    ckpt = ckpt_exists(tag)

    if evl:
        status = "DONE (eval complete)"
    elif done:
        status = "NEEDS_EVAL"
    elif ep > 0 or ckpt:
        status = f"TRAINING  ({ep}/50 ep)"
    else:
        status = "NOT_STARTED"

    print(f"  {label:<12}  {tag:<40}  {status}")

print()
PYEOF

echo "Queue manager log (last 10 lines):"
tail -n 10 "$LOG_DIR/queue_manager.log" 2>/dev/null || echo "  (not started yet)"

echo ""
echo "Screen sessions (micro_reps):"
screen -ls 2>/dev/null | grep "micro_reps" || echo "  (none)"

echo ""
echo "Squeue:"
squeue -u twamorka --format="  %.10i %-30j %8T %.10M %4D %R" 2>/dev/null \
    || echo "  (squeue unavailable)"
