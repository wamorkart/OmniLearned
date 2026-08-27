# Minimal MLP student (--arch mlp: masked mean-pool -> 1 hidden -> linear),
# lower-bound baseline. KD alpha=beta=0.5, T=4. --wandb OFF: the multi-node
# wandb fork corrupts NCCL state and hangs the first all-reduce (root-caused
# 2026-08-04, see EXPERIMENTS_deepsets_kd.md). (was distill_train_top_mlp.sh)
ARCH=mlp
INTERACTION=0
LOCAL_INTERACTION=0
WANDB=0
SAVE_TAG=distill_top_mlp_small_scratch_a05_T4
