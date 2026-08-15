#!/bin/bash
# One-time copy of 5 of the 7 "pretrain" mixture datasets (atlas, aspen, h1,
# cms_qcd, cms_bsm) from CFS to $SCRATCH. jetclass and jetclass2 are excluded
# here -- they're already being transferred separately by hand (jetclass done,
# jetclass2 in progress as of 2026-08-05); this script would only race that
# transfer for no benefit. Once that manual transfer finishes, no further
# action needed for those two -- they'll already be at the destination below.
#
# Motivation: the distill_pretrain_s_scratch_a05_b05_T4_relaunch0805
# validation run stalled 3x on a single-rank hang reading from CFS (see
# distill_loop_pretrain_relaunch logs, session_3: rank 0 went silent after the
# epoch-2 checkpoint, other 15 ranks timed out waiting on it 90 min later).
# $SCRATCH is faster/less contended than CFS for this random-small-read access
# pattern, and the teacher companion-logit files already live there, so moving
# the source data there too puts both required reads on the same fast fs.
#
# Each dataset has train/test/val subdirectories (not flat) -- mirrored here.
#
# Runs on NERSC's dedicated xfer QOS (data-transfer nodes, not compute
# allocation) so it doesn't eat GPU node-hours and can run long/unattended.
#
# Submit any time -- it's a background transfer, no need to wait:
#   sbatch copy_pretrain_data_to_scratch.sbatch
#
# To copy only a subset of datasets (e.g. try h1 first before the rest),
# override DATASETS via --export instead of editing this file:
#   sbatch --export=ALL,DATASETS_OVERRIDE=h1 copy_pretrain_data_to_scratch.sbatch
#
# Check progress:
#   squeue -u $USER -q xfer
#   du -sh /pscratch/sd/t/twamorka/omnilearned/datasets/*
#
# After this AND the manual jetclass/jetclass2 transfer both finish, update
# --path in distill_train_pretrain_relaunch.sh from
# /global/cfs/cdirs/m4567/www/ to /pscratch/sd/t/twamorka/omnilearned/datasets/
# before relaunching distill_loop_pretrain_relaunch.sh.

#SBATCH -A m3246
#SBATCH -q xfer
#SBATCH --time=12:00:00
#SBATCH -J copy_pretrain_data
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/copy_pretrain_data/slurm_%j.out

set -euo pipefail

SRC=/global/cfs/cdirs/m4567/www
DST=/pscratch/sd/t/twamorka/omnilearned/datasets

mkdir -p /pscratch/sd/t/twamorka/omnilearned/logs/copy_pretrain_data

DATASETS="${DATASETS_OVERRIDE:-atlas aspen h1 cms_qcd cms_bsm}"
SPLITS="train test val"

for d in $DATASETS; do
    for s in $SPLITS; do
        [ -d "$SRC/$d/$s" ] || continue
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Copying $d/$s ==="
        mkdir -p "$DST/$d/$s"
        # -P 16: parallel file-level copies -- both CFS (GPFS) and $SCRATCH
        # (Lustre) are parallel filesystems, many concurrent single-file
        # copies go much faster than one serial rsync/cp -r.
        # cp -n: skip files that already exist at the destination, so a
        # resubmit after a timeout/preemption doesn't redo completed work.
        find "$SRC/$d/$s" -maxdepth 1 \( -name "*.h5" -o -name "*.hdf5" \) -print0 \
            | xargs -0 -P 16 -I{} cp -n {} "$DST/$d/$s/"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Done with $d/$s ==="
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All datasets copied."

# Sanity check: per-split file counts should match the source.
for d in $DATASETS; do
    for s in $SPLITS; do
        [ -d "$SRC/$d/$s" ] || continue
        n_src=$(find "$SRC/$d/$s" -maxdepth 1 \( -name "*.h5" -o -name "*.hdf5" \) | wc -l)
        n_dst=$(find "$DST/$d/$s" -maxdepth 1 \( -name "*.h5" -o -name "*.hdf5" \) | wc -l)
        echo "$d/$s: src=$n_src dst=$n_dst"
        if [ "$n_src" -ne "$n_dst" ]; then
            echo "WARNING: file count mismatch for $d/$s"
        fi
    done
done
