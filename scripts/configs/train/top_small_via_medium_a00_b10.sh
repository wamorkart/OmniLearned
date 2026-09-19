# TAKD stage 3: PET2-medium assistant -> PET2-small student, pure KD
# alpha=0, beta=1, T=4. Needs companion_distill_top_medium logits from
# save_teacher_logits_top_medium.sh (stage 2) first.
# (was distill_train_top_small_via_medium.sh)
ALPHA=0.0
BETA=1.0
TEACHER_TAG=distill_top_medium_scratch_a00_b10_T4
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_distill_top_medium
SAVE_TAG=distill_top_small_via_medium_a00_b10_T4
