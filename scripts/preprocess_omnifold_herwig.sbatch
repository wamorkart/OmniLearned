#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A m3246
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -J preprocess_omnifold_herwig
#SBATCH -o /pscratch/sd/t/twamorka/unfolding/logs/preprocess_herwig_%j.out

# Preprocess the full Herwig Z+jets Delphes sample into train_herwig.h5 /
# test_herwig.h5, matching the existing train_pythia.h5 / test_pythia.h5
# already on disk (same script, same schema). Running as a batch job because
# doing this directly on a login-node shell OOM-killed at full scale (~30GB+
# peak for the final concatenated reco/gen arrays plus working copies) --
# login nodes enforce a per-user memory cap well below that even though the
# node itself has hundreds of GB free.

mkdir -p /pscratch/sd/t/twamorka/unfolding/logs

module load conda
conda activate energyflow

cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
python -u tools/preprocess/preprocess_omnifold.py \
  --folder /pscratch/sd/t/twamorka/unfolding --sample herwig --num-data -1 --npoints 101
