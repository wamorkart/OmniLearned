# Fine-tuned qg teacher (fine_tune_qg_pretrain_l) -> DeepSets/PFN student on
# qg. KD alpha=beta=0.5, T=4, size=small. The reference / confirmed
# best DeepSets-KD recipe.
#
# DeepSets is Phi-embed + masked pool + rho-MLP, no attention, so the
# interaction flags are turned off (the CLI would ignore them anyway).
#
# Everything else -- batch 128, iterations 1000, epoch 50, lr 5e-4, wd 0.5,
# teacher companion dir/tag, data path -- comes from _defaults.sh unchanged.
#
# Was: distill_train_top_deepsets.sh CONFIG=a05 (tag suffix _archfix0804).
# This config uses a clean SAVE_TAG for a fresh run; to continue the existing
# checkpoint instead, set SAVE_TAG=distill_top_deepsets_small_scratch_a05_T4_archfix0804
# at call time.
#
# Note: --wandb is ON here (inherited from _defaults.sh). The old script
# defaulted it off for this recipe; the multi-node wandb/NCCL caution was
# retired 2026-08-25.
ARCH=deep-sets
DATASET=qg
INTERACTION=1
LOCAL_INTERACTION=0
TEACHER_TAG=fine_tune_qg_pretrain_l
SAVE_TAG=distill_qg_deepsets_small_scratch_a05_T4
