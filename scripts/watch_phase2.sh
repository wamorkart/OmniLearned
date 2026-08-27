#!/bin/bash
# Phase 2 watcher — starts after the T-sweep finishes and runs two parallel pipelines:
#
#   Pipeline A: fine-tune KD-pretrained student on top tagging (1 node × 4 GPUs)
#               → evaluate → compute metrics vs fine_tune_top_s and T-sweep winner
#
#   Pipeline B: JetClass distillation α=0 β=1 T=4 (4 nodes × 16 GPUs)
#               → evaluate → compute metrics vs α=0.5 baseline
#
# Run in a screen session on login25, BEFORE going offline:
#   screen -S watch_phase2 bash /path/to/watch_phase2.sh
#
# It will wait silently until watch_T_sweep.sh finishes (polls every 5 min),
# then launches both pipelines in new screen sessions and monitors them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_SWEEP=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep
RESULTS_DIR=/pscratch/sd/t/twamorka/omnilearned/results
WATCHER_LOG="$RESULTS_DIR/watch_phase2.log"
PHASE2_EVAL_LOG="$RESULTS_DIR/phase2_eval.log"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

training_done() {
    local tag="$1"
    local logfile="$LOG_SWEEP/${tag}/summary.log"
    [ -f "$logfile" ] && grep -q "Training completed:" "$logfile"
}

loop_done() {
    local logfile="$1"
    [ -f "$logfile" ] && grep -q "Training completed:" "$logfile"
}

session_alive() {
    screen -ls 2>/dev/null | grep -q "\.$1 "
}

launch_if_needed() {
    local session="$1" script="$2"
    local logfile="$3"   # summary.log to poll for completion

    if loop_done "$logfile"; then
        log "($session) already done, skipping launch"
        return 0
    fi
    if session_alive "$session"; then
        return 0
    fi
    log "($session) launching..."
    screen -dmS "$session" bash -c "cd '$SCRIPT_DIR' && bash '$script'"
    log "($session) launched"
}

wait_both_phase2() {
    local sess_a="$1" log_a="$2"
    local sess_b="$3" log_b="$4"
    local script_a="$5" script_b="$6"

    while true; do
        local da=false db=false
        loop_done "$log_a" && da=true
        loop_done "$log_b" && db=true
        log "pipeline_A done=$da  |  pipeline_B done=$db"
        $da && $db && return 0

        $da || launch_if_needed "$sess_a" "$script_a" "$log_a"
        $db || launch_if_needed "$sess_b" "$script_b" "$log_b"
        sleep 300
    done
}

# ── Phase 1 gate: wait for T-sweep to finish ─────────────────────────────────

T_SWEEP_TAGS=(
    distill_top_small_scratch_a00_b10_T1
    distill_top_small_scratch_a00_b10_T2
    distill_top_small_scratch_a00_b10_T8
    distill_top_small_scratch_a00_b10_T16
    distill_top_small_scratch_a00_b10_T4_r1
    distill_top_small_scratch_a00_b10_T4_r2
)

log "=== Phase 2 watcher started — waiting for T-sweep to complete ==="

while true; do
    all_done=true
    for tag in "${T_SWEEP_TAGS[@]}"; do
        if ! training_done "$tag"; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        log "All T-sweep training runs complete. Starting phase 2."
        break
    fi

    # Count how many are done for status
    ndone=0
    for tag in "${T_SWEEP_TAGS[@]}"; do
        training_done "$tag" && ndone=$((ndone+1))
    done
    log "T-sweep gate: ${ndone}/${#T_SWEEP_TAGS[@]} training runs done — still waiting"
    sleep 300
done

# ── Phase 2: launch both pipelines ───────────────────────────────────────────

log "--- Launching pipeline A: fine-tune KD-pretrained top student ---"
log "--- Launching pipeline B: JetClass α=0 distillation ---"

LOG_FT=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_pretrain_finetune/summary.log
LOG_JC=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_jetclass_a00/summary.log

wait_both_phase2 \
    "top_ft_pretrain" "$LOG_FT" \
    "jc_a00"          "$LOG_JC" \
    "distill_loop_top_pretrain_finetune.sh" \
    "distill_loop_jetclass_a00.sh"

log "Both phase 2 pipelines complete. Running eval+metrics..."

# ── Eval + metrics ────────────────────────────────────────────────────────────

salloc \
    -C gpu -q interactive -t 120 \
    --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 \
    -A m3246 \
    bash -c "
set -euo pipefail
SCRIPT_DIR='$SCRIPT_DIR'

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=\$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
TOP_PRETRAIN_EVAL=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_pretrain
JC_EVAL=/pscratch/sd/t/twamorka/omnilearned/eval/jetclass_distill
mkdir -p \"\$TOP_PRETRAIN_EVAL\" \"\$JC_EVAL\"

# ── Pipeline A: top fine-tune eval ───────────────────────────────────────────
echo '=== Pipeline A: evaluate fine_tune_top_distill_pretrain_s_a05_T4 ==='
SENTINEL=\"\$TOP_PRETRAIN_EVAL/outputs_fine_tune_top_distill_pretrain_s_a05_T4_top_test_rank0.npz\"
if [ ! -f \"\$SENTINEL\" ]; then
    srun -l -u bash -c \"
    source '\$SCRIPT_DIR/export_ddp.sh'
    omnilearned evaluate \\
        -i \$CHECKPOINT_DIR \\
        -o \$TOP_PRETRAIN_EVAL \\
        --save-tag fine_tune_top_distill_pretrain_s_a05_T4 \\
        --dataset top \\
        --path /global/cfs/cdirs/m4567/www/ \\
        --size small --interaction --local-interaction \\
        --num-classes 2 --batch 128 --num-workers 4 --dataset-type test
    \"
else
    echo '[SKIP] pipeline A eval already done'
fi

# ── Pipeline B: JetClass α=0 eval ────────────────────────────────────────────
echo '=== Pipeline B: evaluate distill_jetclass_s_pretrain_l_a00_b10_T4_100epochs ==='
SENTINEL=\"\$JC_EVAL/outputs_distill_jetclass_s_pretrain_l_a00_b10_T4_100epochs_jetclass_test_rank0.npz\"
if [ ! -f \"\$SENTINEL\" ]; then
    srun -l -u bash -c \"
    source '\$SCRIPT_DIR/export_ddp.sh'
    omnilearned evaluate \\
        -i \$CHECKPOINT_DIR \\
        -o \$JC_EVAL \\
        --save-tag distill_jetclass_s_pretrain_l_a00_b10_T4_100epochs \\
        --dataset jetclass \\
        --path /global/cfs/cdirs/m4567/www/ \\
        --size small --num-feat 9 --interaction --local-interaction \\
        --num-classes 10 --batch 128 --num-workers 4 --dataset-type test
    \"
else
    echo '[SKIP] pipeline B eval already done'
fi

# ── Metrics ───────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import glob, os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

SIGNAL_CLASS = 1
TOP_EVAL = '/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_pretrain'
JC_EVAL  = '/pscratch/sd/t/twamorka/omnilearned/eval/jetclass_distill'

def rej_at_eff(fpr, tpr, eff):
    idx = np.searchsorted(tpr, eff)
    return float('inf') if idx >= len(fpr) or fpr[idx] == 0 else 1.0 / fpr[idx]

def top_metrics(tag, indir):
    paths = sorted(glob.glob(f'{indir}/outputs_{tag}_top_test_rank*.npz'))
    if not paths: return None
    preds = np.concatenate([np.load(p)['prediction'].astype(np.float32) for p in paths])
    labels = np.concatenate([np.load(p)['pid'] for p in paths])
    scores = preds[:, SIGNAL_CLASS]
    acc = (preds.argmax(1) == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return acc, auc, rej_at_eff(fpr, tpr, 0.50), rej_at_eff(fpr, tpr, 0.30)

print('\n=== Phase 2 Results ===\n')
print('Top-tagging  (pretrain KD → fine-tune):')
W = 44
print(f'  {\"Tag\":<{W}} {\"Acc\":>7}  {\"AUC\":>7}  {\"1/FPR@50%\":>10}  {\"1/FPR@30%\":>10}')
print(f'  {\"-\"*W}  {\"-\"*7}  {\"-\"*7}  {\"-\"*10}  {\"-\"*10}')
for tag, label in [
    ('fine_tune_top_distill_pretrain_s_a05_T4', 'distill_pretrain → fine-tune'),
]:
    r = top_metrics(tag, TOP_EVAL)
    if r:
        acc, auc, r50, r30 = r
        print(f'  {label:<{W}} {acc*100:>6.2f}%  {auc:>7.4f}  {r50:>10.0f}  {r30:>10.0f}')
    else:
        print(f'  {label:<{W}} (no outputs)')
print('  Reference:')
print(f'  {\"fine_tune_top_s  (CE-only pretrain)\":<{W}} {\"94.38%\":>7}  {\"0.9875\":>7}  {\"577\":>10}  {\"2556\":>10}')
print(f'  {\"distill_top_s_scratch_a05_T4\":<{W}} {\"94.32%\":>7}  {\"0.9875\":>7}  {\"580\":>10}  {\"2804\":>10}')
print(f'  {\"distill_top_s_scratch_a00_b10_T4\":<{W}} {\"94.43%\":>7}  {\"0.9879\":>7}  {\"621\":>10}  {\"3205\":>10}')

from sklearn.metrics import roc_auc_score, roc_curve
CLASS_NAMES = ['QCD','Hbb','Hcc','Hgg','H4q','Hqql','Zqq','Wqq','Tbqq','Tbl']

def jc_metrics(tag, indir):
    paths = sorted(glob.glob(f'{indir}/outputs_{tag}_jetclass_test_rank*.npz'))
    if not paths: return None
    preds = np.concatenate([np.load(p)['prediction'].astype(np.float32) for p in paths])
    labels = np.concatenate([np.load(p)['pid'] for p in paths])
    acc = (preds.argmax(1) == labels).mean()
    auc_macro = roc_auc_score(labels, preds, multi_class='ovr', average='macro')
    return acc, auc_macro

print('\nJetClass  (pretrain_l teacher, α sweep):')
print(f'  {\"Tag\":<{W}} {\"Acc\":>7}  {\"AUC macro\":>10}')
print(f'  {\"-\"*W}  {\"-\"*7}  {\"-\"*10}')
for tag, label in [
    ('distill_jetclass_s_pretrain_l_a00_b10_T4_100epochs', 'α=0.0 β=1.0 T=4'),
    ('distill_jetclass_s_pretrain_l_a05_T4_100epochs',     'α=0.5 β=0.5 T=4  [done]'),
]:
    r = jc_metrics(tag, JC_EVAL)
    if r:
        acc, auc = r
        print(f'  {label:<{W}} {acc*100:>6.2f}%  {auc:>10.4f}')
    else:
        print(f'  {label:<{W}} (no outputs)')
print(f'  {\"fine_tune_jetclass_s  (CE-only, no KD)\":<{W}} {\"85.34%\":>7}  {\"0.9865\":>10}')
PYEOF
" 2>&1 | tee "$PHASE2_EVAL_LOG"

log "Phase 2 eval+metrics complete. Results in $PHASE2_EVAL_LOG"
log "=== Phase 2 watcher done ==="
