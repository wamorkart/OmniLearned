#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A m3246
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -J preprocess_lhco_idealized
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/preprocess_lhco_idealized_%j.out

# Build the LHCO idealized signal-vs-background classifier dataset in
# OmniLearned's HEPDataset schema, following docs/LHCO_ANOMALY_DETECTION_PROCEDURE.md
# (Pipeline A/B). Writes <out>/custom/{train,val,test}/*.h5 with data (N,320,6)
# = [d_eta, d_phi, log(pT_rel), log(pT), is_jet1, is_jet2], pid 0/1, global (N,5),
# mjj (N,). idealized mode never touches the 15 GB *_extended files, so peak RAM
# stays well under the 32 GB ask; batch job only to keep it off the login node.
#
# After this finishes, build the index cache:
#   module load conda && conda activate ol_env
#   PYTHONPATH=$PWD/src python -m omnilearned.cli dataloader -d custom \
#     -f /pscratch/sd/t/twamorka/omnilearned/datasets

mkdir -p /pscratch/sd/t/twamorka/omnilearned/logs

module load conda
conda activate ol_env

cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
python -u anomalydetection/preprocess_lhco.py \
  --data-dir /global/cfs/cdirs/m3246/vmikuni/PET/LHCO \
  --out-dir /pscratch/sd/t/twamorka/omnilearned/datasets \
  --mode idealized
