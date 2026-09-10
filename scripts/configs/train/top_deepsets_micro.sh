# DeepSets/PFN top-tagging student at "micro" width (base_dim=32, phi=3, rho=2,
# ~19,431 params). Width-scan point between "distillnet" and "tiny". One-variable
# change from top_deepsets_a05 (SIZE); KD a0.5/b0.5/T4, teacher fine_tune_top_l,
# wd 0.5, batch 128, 1000 iters x 50 epochs, lr 5e-4 -- all from _defaults.sh.
#
# Part of the DeepSets width scan: nano | distillnet(off-curve) | micro | tiny |
# small. small (298,887 params) = 93.30%; distillnet (10,981) = 92.85%.
# NOTE: unrelated to the PET2 "micro" student (tags distill_top_micro_*); this
# is a Deep Sets body (tag distill_top_deepsets_micro_*).
#
# --wandb ON (inherited from _defaults.sh).
ARCH=deep-sets
SIZE=micro
INTERACTION=0
LOCAL_INTERACTION=0
SAVE_TAG=distill_top_deepsets_micro_scratch_a05_T4
