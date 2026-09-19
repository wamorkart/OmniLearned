# Same recipe as top_small_a05 (large top teacher -> PET2-small student, KD
# alpha=beta=0.5, T=4) but pointed at the from-scratch large teacher
# (top_l_scratch) instead of the JetClass-pretrained-then-fine-tuned one
# (fine_tune_top_l). Requires teacher logits already built via:
#   TAG=top_l_scratch bash scripts/save_teacher_logits_top.sh
TEACHER_TAG=top_l_scratch
SAVE_TAG=distill_top_small_scratch_a05_T4_teacherscratch
