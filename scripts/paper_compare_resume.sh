#!/bin/bash
# Resume the arXiv:2512.17011 comparison after the 2026-09-01 10:38 screen
# session died ~18 min in (mid step #4). Steps #2 (flops_probe) and #3 (the
# four PTQ runs) already produced their outputs under $OUTDIR / the repo, so
# this only does what is left:
#   - figures/tables first (CPU, no GPU deps) so we always get artifacts
#   - then the #4 nconst evals (deepsets first: fast), appending nconst_results.tsv
#   - then make_tables again to fold in whatever nconst rows landed
# Meant to run on the still-live idle allocation:
#   srun --jobid=57828802 --overlap -N1 -n1 bash scripts/paper_compare_resume.sh
set -uo pipefail

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
PC=$REPO/analysis/paper_compare
QZ=$REPO/tools/quantize
OUTDIR=/pscratch/sd/t/twamorka/omnilearned/results/paper_compare
mkdir -p "$OUTDIR"

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$PC:$QZ:${PYTHONPATH:-}"

DS_SMALL_TAG=distill_top_deepsets_small_scratch_a05_T4_archfix0804
DS_DNET_TAG=distill_top_deepsets_distillnet_scratch_a05_T4

echo "[$(date '+%F %T')] === paper_compare RESUME start ==="

# fresh #4 accumulator
rm -f "$OUTDIR/nconst_results.tsv"

# ---- #1 + #2 + #3 : figures and tables (CPU, deps already present) ----
cd "$PC"
echo "[$(date '+%F %T')] fig_params_vs_rejection"
$PY fig_params_vs_rejection.py --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/fig1.log"
echo "[$(date '+%F %T')] energy_model"
$PY energy_model.py --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/energy.log"
echo "[$(date '+%F %T')] make_tables (pre-nconst)"
$PY make_tables.py --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/tables.log"

# ---- #4: constituent-count robustness (full-N vs N=30) ----------------
#   tag                                          arch     size
NCONST_TARGETS=(
  "${DS_SMALL_TAG}                               deepsets small"
  "${DS_DNET_TAG}                                deepsets distillnet"
  "distill_top_micro_scratch_a05_T4              pet2     micro"
  "distill_top_small_scratch_a05_T4              pet2     small"
  "distill_top_small_scratch_a00_b10_T4          pet2     small"
)
for row in "${NCONST_TARGETS[@]}"; do
    set -- $row; tag=$1; arch=$2; size=$3
    for N in 0 30; do
        echo "[$(date '+%F %T')] nconst  $tag  N=$N"
        $PY "$PC/eval_nconst_top.py" --tag "$tag" --arch "$arch" --size "$size" \
            --nconst "$N" --outdir "$OUTDIR" 2>&1 | tail -n 12
    done
done

echo "[$(date '+%F %T')] make_tables (final)"
$PY "$PC/make_tables.py" --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/tables.log"

echo "[$(date '+%F %T')] === paper_compare RESUME done ==="
ls -la "$OUTDIR"
