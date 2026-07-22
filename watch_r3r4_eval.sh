#!/bin/bash
# Watches r3 and r4 training summary logs. Once both complete, grabs an
# salloc and runs evaluate_metrics_top_sweep.sh, then saves the metrics
# table as a PDF.
#
# Run in a new screen session on login25:
#   screen -S eval_watcher bash watch_r3r4_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top
LOG_R3="$LOG_DIR/distill_top_small_scratch_a05_T4_r3_summary.log"
LOG_R4="$LOG_DIR/distill_top_small_scratch_a05_T4_r4_summary.log"

RESULTS_DIR=/pscratch/sd/t/twamorka/omnilearned/results
EVAL_LOG="$RESULTS_DIR/top_distill_sweep_eval.log"
PDF_OUT="$RESULTS_DIR/top_distill_sweep_metrics.pdf"
mkdir -p "$RESULTS_DIR"

WATCHER_LOG="$LOG_DIR/watch_r3r4_eval.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"; }

log "Watcher started."
log "  r3 log: $LOG_R3"
log "  r4 log: $LOG_R4"

check_done() { [ -f "$1" ] && grep -q "Training completed" "$1"; }

while true; do
    r3_done=false; r4_done=false
    check_done "$LOG_R3" && r3_done=true
    check_done "$LOG_R4" && r4_done=true
    log "r3 done: $r3_done | r4 done: $r4_done"

    if $r3_done && $r4_done; then
        log "Both r3 and r4 finished. Starting eval..."
        break
    fi
    sleep 300
done

# ── Evaluate ─────────────────────────────────────────────────────────────────

log "Grabbing salloc for evaluation..."
salloc \
    -C gpu -q interactive -t 120 \
    --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 \
    -A m3246 \
    bash "$SCRIPT_DIR/evaluate_metrics_top_sweep.sh" \
    2>&1 | tee "$EVAL_LOG"

log "Eval complete. Log saved to $EVAL_LOG"

# ── PDF ───────────────────────────────────────────────────────────────────────

log "Generating PDF..."

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

python3 - "$EVAL_LOG" "$PDF_OUT" <<'PYEOF'
import sys, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

eval_log, pdf_out = sys.argv[1], sys.argv[2]

with open(eval_log) as f:
    text = f.read()

# ── Parse data rows ───────────────────────────────────────────────────────────
# Matches lines like:  "  a=0.00 b=1.00  [pure KD]   94.32%  0.9875     577     2804"
DATA_RE = re.compile(
    r"^\s{2}([\w=\.\s\[\]]+?)\s{2,}"   # label (2+ space separator)
    r"(\d+\.\d+)%\s+"                   # Acc
    r"(0\.\d+)\s+"                      # AUC
    r"([\d.]+|inf)\s+"                  # 1/FPR@50%
    r"([\d.]+|inf)",                    # 1/FPR@30%
    re.MULTILINE,
)
VARIANCE_RE = re.compile(
    r"Rep variance \(n=(\d+)\):\s+AUC ([\d.]+) ± ([\d.]+)\s+1/FPR@30% ([\d.]+) ± ([\d.]+)"
)
REF_RE = re.compile(
    r"^\s{2}(fine_tune_top_\S+.*?)\s{2,}"
    r"(\d+\.\d+)%\s+(0\.\d+)\s+([\d.]+)\s+([\d.]+)",
    re.MULTILINE,
)

data_rows = DATA_RE.findall(text)
ref_rows  = REF_RE.findall(text)
var_match = VARIANCE_RE.search(text)

def fmt_rej(v):
    try:
        return f"{float(v):,.0f}" if v != "inf" else "∞"
    except ValueError:
        return v

col_headers = ["Run", "Acc (%)", "AUC", "1/FPR\n@50%", "1/FPR\n@30%"]
sweep_section = [r for r in data_rows if "r1" not in r[0] and "r2" not in r[0]
                 and "r3" not in r[0] and "r4" not in r[0]]
rep_section   = [r for r in data_rows if any(x in r[0] for x in ["r1","r2","r3","r4"])]

def make_table_data(rows):
    out = []
    for label, acc, auc, r50, r30 in rows:
        out.append([label.strip(), f"{float(acc):.2f}", auc,
                    fmt_rej(r50), fmt_rej(r30)])
    return out

sweep_data = make_table_data(sweep_section)
rep_data   = make_table_data(rep_section)
ref_data   = make_table_data(ref_rows)

# ── Build figure ──────────────────────────────────────────────────────────────
BLUE  = "#1f4e79"
LGRAY = "#f2f2f2"
DGRAY = "#d9d9d9"
RED   = "#c00000"
GREEN = "#375623"

fig = plt.figure(figsize=(11, 8.5))
fig.patch.set_facecolor("white")

title_ax = fig.add_axes([0.05, 0.90, 0.90, 0.08])
title_ax.axis("off")
title_ax.text(0.5, 0.7, "Top-Tagging Distillation — Sweep & Repetition Metrics",
              ha="center", va="center", fontsize=14, fontweight="bold", color=BLUE)
title_ax.text(0.5, 0.1, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              ha="center", va="center", fontsize=8, color="gray")

def draw_table(ax, title, table_data, col_w=None, color_auc=True):
    ax.axis("off")
    ax.text(0.0, 1.02, title, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=BLUE, va="bottom")
    if not table_data:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    all_rows = [col_headers] + table_data
    if col_w is None:
        col_w = [0.38, 0.12, 0.12, 0.19, 0.19]

    tbl = ax.table(
        cellText=all_rows[1:],
        colLabels=all_rows[0],
        colWidths=col_w,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)

    n_cols = len(col_headers)
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")

    if color_auc and table_data:
        try:
            aucs = [float(r[2]) for r in table_data]
            best_i = int(np.argmax(aucs))
        except (ValueError, IndexError):
            best_i = -1
    else:
        best_i = -1

    for i, row in enumerate(table_data):
        bg = LGRAY if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            if i == best_i and j in (2, 4) and color_auc:
                cell.set_text_props(color=GREEN, fontweight="bold")
    tbl[0, 0].set_text_props(ha="left")
    for i in range(len(table_data)):
        tbl[i + 1, 0].set_text_props(ha="left")

# Sweep table
ax1 = fig.add_axes([0.05, 0.58, 0.90, 0.30])
draw_table(ax1, "Alpha / Beta Sweep  (T=4, small student trained from scratch on top tagging)",
           sweep_data)

# Rep table
ax2 = fig.add_axes([0.05, 0.30, 0.90, 0.26])
draw_table(ax2, "Repetitions of a=0.50, b=0.50  (variance / consistency check)",
           rep_data)

# Variance annotation
if var_match and rep_data:
    n, auc_m, auc_s, r30_m, r30_s = var_match.groups()
    fig.text(0.07, 0.295,
             f"Run-to-run variance (n={n}):  AUC {auc_m} ± {auc_s}   "
             f"1/FPR@30% {r30_m} ± {r30_s}",
             fontsize=8, color="dimgray", style="italic")

# Reference table
ax3 = fig.add_axes([0.05, 0.08, 0.90, 0.20])
draw_table(ax3, "Reference baselines  (CE-only, no distillation)",
           ref_data, color_auc=False)

with PdfPages(pdf_out) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
    d = pdf.infodict()
    d["Title"]  = "Top-Tagging Distillation Sweep Metrics"
    d["Author"] = "OmniLearned"

plt.close(fig)
print(f"PDF saved: {pdf_out}")
PYEOF

log "PDF saved to $PDF_OUT"
log "Done."
