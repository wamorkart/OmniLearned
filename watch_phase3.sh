#!/bin/bash
# Phase 3 watcher — starts after phase 2 finishes.
#
# 1. Waits for watch_phase2.sh to complete.
# 2. Reads the T-sweep eval results to find the best temperature (by 1/FPR@30%).
# 3. If best T = 4: skip top-tagging reps (3 seeds already exist) and exit.
#    If best T ≠ 4: run 2 reps of best T → eval → metrics.
#
# Run in a screen session before going offline:
#   screen -S watch_phase3 bash /path/to/watch_phase3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR=/pscratch/sd/t/twamorka/omnilearned/results
WATCHER_LOG="$RESULTS_DIR/watch_phase3.log"
PHASE3_EVAL_LOG="$RESULTS_DIR/phase3_eval.log"
LOG_SWEEP=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"; }

session_alive() { screen -ls 2>/dev/null | grep -q "\.$1 "; }

training_done() {
    local logfile="$1"
    [ -f "$logfile" ] && grep -q "Training completed:" "$logfile"
}

launch_rep() {
    local session="$1" T="$2" suffix="$3"
    local tag="distill_top_small_scratch_a00_b10_T${T}_${suffix}"
    local logfile="$LOG_SWEEP/${tag}/summary.log"

    if training_done "$logfile"; then
        log "$tag: already done, skipping launch"
        return 0
    fi
    if session_alive "$session"; then
        return 0
    fi
    log "$tag: (re)launching screen session '$session'..."
    mkdir -p "$LOG_SWEEP/$tag"
    screen -dmS "$session" bash -c \
        "cd '$SCRIPT_DIR' && bash distill_loop_top_T_sweep.sh '$T' '$suffix'"
    log "$tag: launched"
}

wait_rep() {
    local sess1="$1" T1="$2" suf1="$3"
    local sess2="$4" T2="$5" suf2="$6"
    local log1="$LOG_SWEEP/distill_top_small_scratch_a00_b10_T${T1}_${suf1}/summary.log"
    local log2="$LOG_SWEEP/distill_top_small_scratch_a00_b10_T${T2}_${suf2}/summary.log"
    while true; do
        local d1=false d2=false
        training_done "$log1" && d1=true
        training_done "$log2" && d2=true
        log "rep1 done=$d1  |  rep2 done=$d2"
        $d1 && $d2 && return 0
        $d1 || launch_rep "$sess1" "$T1" "$suf1"
        $d2 || launch_rep "$sess2" "$T2" "$suf2"
        sleep 300
    done
}

# ── Gate: wait for phase 2 to complete ───────────────────────────────────────

log "=== Phase 3 watcher started — waiting for phase 2 to complete ==="

while true; do
    if [ -f "$RESULTS_DIR/watch_phase2.log" ] && \
       grep -q "Phase 2 watcher done" "$RESULTS_DIR/watch_phase2.log"; then
        log "Phase 2 complete. Proceeding."
        break
    fi
    log "Phase 2 still running — waiting..."
    sleep 300
done

# ── Find the best T from T-sweep eval results ─────────────────────────────────

log "Reading T-sweep eval results to find best T..."

BEST_T=$(python3 - <<'PYEOF'
import glob, os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_DIR = "/pscratch/sd/t/twamorka/omnilearned/eval/top_distill"
SIGNAL_CLASS = 1

T_TAGS = {
    1:  "distill_top_small_scratch_a00_b10_T1",
    2:  "distill_top_small_scratch_a00_b10_T2",
    4:  "distill_top_small_scratch_a00_b10_T4",
    8:  "distill_top_small_scratch_a00_b10_T8",
    16: "distill_top_small_scratch_a00_b10_T16",
}

def rej30(tag):
    paths = sorted(glob.glob(f"{EVAL_DIR}/outputs_{tag}_top_test_rank*.npz"))
    if not paths:
        return None
    preds = np.concatenate([np.load(p)["prediction"].astype(np.float32) for p in paths])
    labels = np.concatenate([np.load(p)["pid"] for p in paths])
    auc = roc_auc_score(labels, preds[:, SIGNAL_CLASS])
    fpr, tpr, _ = roc_curve(labels, preds[:, SIGNAL_CLASS])
    idx = np.searchsorted(tpr, 0.30)
    r30 = 1.0 / fpr[idx] if idx < len(fpr) and fpr[idx] > 0 else 0.0
    return auc, r30

results = {}
for T, tag in T_TAGS.items():
    r = rej30(tag)
    if r:
        results[T] = r

if not results:
    print(4)  # fallback
else:
    # rank by 1/FPR@30%, break ties by AUC
    best = max(results.keys(), key=lambda t: (results[t][1], results[t][0]))
    print(best)
    import sys
    print(f"T ranking (1/FPR@30%): " +
          "  ".join(f"T={t}: {results[t][1]:.0f}" for t in sorted(results)), file=sys.stderr)
PYEOF
)

log "Best T = ${BEST_T}"

# ── Decide whether to run reps ────────────────────────────────────────────────

if [ "$BEST_T" -eq 4 ]; then
    log "Best T is 4 — already have 3 seeds (base + r1 + r2). No reps needed."
    log "=== Phase 3 watcher done (no new runs) ==="
    exit 0
fi

log "Best T = ${BEST_T} (≠ 4). Running 2 reps..."

# ── Run 2 reps of best T ─────────────────────────────────────────────────────

R1_SESS="top_T${BEST_T}_r1"
R2_SESS="top_T${BEST_T}_r2"

wait_rep "$R1_SESS" "$BEST_T" "r1" "$R2_SESS" "$BEST_T" "r2"

log "Reps complete. Running eval+metrics..."

# ── Eval + metrics ────────────────────────────────────────────────────────────

salloc \
    -C gpu -q interactive -t 120 \
    --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 \
    -A m3246 \
    bash -c "
set -euo pipefail
SCRIPT_DIR='$SCRIPT_DIR'
BEST_T='$BEST_T'

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=\$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill
mkdir -p \"\$EVAL_DIR\"

for SUFFIX in r1 r2; do
    TAG=\"distill_top_small_scratch_a00_b10_T\${BEST_T}_\${SUFFIX}\"
    SENTINEL=\"\$EVAL_DIR/outputs_\${TAG}_top_test_rank0.npz\"
    CKPT=\"\$CHECKPOINT_DIR/best_model_\${TAG}.pt\"
    if [ ! -f \"\$CKPT\" ]; then
        echo \"[SKIP] \$TAG — checkpoint missing\"
        continue
    fi
    if [ -f \"\$SENTINEL\" ]; then
        echo \"[SKIP] \$TAG — eval already done\"
        continue
    fi
    echo \"=== Evaluating \$TAG ===\"
    srun -l -u bash -c \"
    source '\$SCRIPT_DIR/export_ddp.sh'
    omnilearned evaluate \\
        -i \$CHECKPOINT_DIR -o \$EVAL_DIR \\
        --save-tag \$TAG \\
        --dataset top --path /global/cfs/cdirs/m4567/www/ \\
        --size small --interaction --local-interaction \\
        --num-classes 2 --batch 128 --num-workers 4 --dataset-type test
    \"
done

python3 - <<'PYEOF'
import glob, os, numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sys

EVAL_DIR = '/pscratch/sd/t/twamorka/omnilearned/eval/top_distill'
BEST_T = int(os.environ.get('BEST_T', '1'))
SIGNAL_CLASS = 1

ALL_T = [1, 2, 4, 8, 16]
ALL_RUNS = [(T, '') for T in ALL_T] + [(BEST_T, 'r1'), (BEST_T, 'r2'), (4, 'r1'), (4, 'r2')]

def tag(T, suffix=''):
    return f'distill_top_small_scratch_a00_b10_T{T}' + (f'_{suffix}' if suffix else '')

def metrics(t):
    paths = sorted(glob.glob(f'{EVAL_DIR}/outputs_{t}_top_test_rank*.npz'))
    if not paths: return None
    preds = np.concatenate([np.load(p)[\"prediction\"].astype(np.float32) for p in paths])
    labels = np.concatenate([np.load(p)[\"pid\"] for p in paths])
    auc = roc_auc_score(labels, preds[:, SIGNAL_CLASS])
    fpr, tpr, _ = roc_curve(labels, preds[:, SIGNAL_CLASS])
    r50 = 1./fpr[np.searchsorted(tpr, 0.50)] if fpr[np.searchsorted(tpr, 0.50)] > 0 else float('inf')
    r30 = 1./fpr[np.searchsorted(tpr, 0.30)] if fpr[np.searchsorted(tpr, 0.30)] > 0 else float('inf')
    return (preds.argmax(1) == labels).mean(), auc, r50, r30

print(f\"\nPhase 3 results — all T-sweep runs (α=0 β=1)\")
print(f'  {\"Tag\":<48} {\"Acc\":>7}  {\"AUC\":>7}  {\"1/FPR@50%\":>10}  {\"1/FPR@30%\":>10}')
print('  ' + '-'*92)
for T in ALL_T:
    t = tag(T)
    r = metrics(t)
    label = f'T={T}' + (' ← BEST' if T == BEST_T else '')
    if r: print(f'  {t:<48} {r[0]*100:>6.2f}%  {r[1]:>7.4f}  {r[2]:>10.0f}  {r[3]:>10.0f}   {label}')
    else: print(f'  {t:<48} (no outputs)')
for suffix in ['r1', 'r2']:
    t = tag(BEST_T, suffix)
    r = metrics(t)
    if r: print(f'  {t:<48} {r[0]*100:>6.2f}%  {r[1]:>7.4f}  {r[2]:>10.0f}  {r[3]:>10.0f}   T={BEST_T} rep')
    else: print(f'  {t:<48} (no outputs)')
for suffix in ['r1', 'r2']:
    if BEST_T != 4:
        t = tag(4, suffix)
        r = metrics(t)
        if r: print(f'  {t:<48} {r[0]*100:>6.2f}%  {r[1]:>7.4f}  {r[2]:>10.0f}  {r[3]:>10.0f}   T=4 rep')
PYEOF
" 2>&1 | tee "$PHASE3_EVAL_LOG"

log "Phase 3 eval+metrics complete. Results in $PHASE3_EVAL_LOG"
log "=== Phase 3 watcher done ==="
