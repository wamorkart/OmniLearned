# TAKD stage 1 variant: PET2-medium assistant at alpha=beta=0.5, T=4.
# batch 64 (not 128): medium + --local-interaction OOMs at 128 on 40GB.
# (was distill_train_top_medium_a05_b05_T4.sh)
SIZE=medium
BATCH=64
SAVE_TAG=distill_top_medium_scratch_a05_b05_T4
