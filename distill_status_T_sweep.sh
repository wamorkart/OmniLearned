#!/bin/bash
# Quick status dashboard for the T-sweep. No GPU needed.
#
# Run any time to see training + eval progress:
#   bash distill_status_T_sweep.sh
#
# Claude can also run this via the Bash tool to monitor on your behalf.

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep

echo "=== T-Sweep Status [$(date '+%Y-%m-%d %H:%M:%S')] ==="
echo ""

python3 - <<PYEOF
import json, glob, os

CHECKPOINT_DIR = "$CHECKPOINT_DIR"
EVAL_DIR       = "$EVAL_DIR"
LOG_DIR        = "$LOG_DIR"

TAGS = [
    ("distill_top_small_scratch_a00_b10_T1",    "T=1"),
    ("distill_top_small_scratch_a00_b10_T2",    "T=2"),
    ("distill_top_small_scratch_a00_b10_T4",    "T=4  [done]"),
    ("distill_top_small_scratch_a00_b10_T8",    "T=8"),
    ("distill_top_small_scratch_a00_b10_T16",   "T=16"),
    ("distill_top_small_scratch_a00_b10_T4_r1", "T=4 r1"),
    ("distill_top_small_scratch_a00_b10_T4_r2", "T=4 r2"),
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
    lf = f"{LOG_DIR}/{tag}/summary.log"
    return os.path.exists(lf) and "Training completed:" in open(lf).read()

def eval_done(tag):
    return bool(glob.glob(f"{EVAL_DIR}/outputs_{tag}_top_test_rank0.npz"))

def ckpt_exists(tag):
    return os.path.exists(f"{CHECKPOINT_DIR}/best_model_{tag}.pt")

print(f"  {'Label':<12}  {'Tag':<46}  Status")
print(f"  {'-'*12}  {'-'*46}  {'-'*26}")

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

    print(f"  {label:<12}  {tag:<46}  {status}")

print()
PYEOF

echo "Screen sessions (top_T*):"
screen -ls 2>/dev/null | grep "top_T" || echo "  (none)"

echo ""
echo "Squeue:"
squeue -u twamorka --format="  %.10i %-22j %8T %.10M %4D %R" 2>/dev/null \
    || echo "  (squeue unavailable)"
