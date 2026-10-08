# DeepSets "distillnet"-width student + one GNN message-passing block on
# ATLAS flavour tagging (4 classes: 0=light 1=c 2=b 3=tau), KD from the v7
# ftag fine-tune (docs/ATLAS_FLAV_RESULTS.md). Recipe is the top-tagging
# top_deepsets_distillnet_gnn.sh one (a0.5/b0.5/T4, wd 0.5, K=64) with only
# the dataset swapped. Classifier mode: the student has no per-track origin
# head, so only the jet-flavour logits are distilled.
#
# Requires teacher logits built via scripts/save_teacher_logits_atlas_flav.sh.
# atlas_flav jets hold at most 40 tracks, so K=64 never truncates.
DATASET=atlas_flav
DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/
NUM_CLASSES=4
NUM_FEAT=4
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
TEACHER_TAG=fine_tune_atlas_flav_m_localint_v7
EXTRA_FLAGS="--use-add --num-add 17 --conditional --num-cond 4 --num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=distill_atlas_flav_deepsets_distillnet_a05_T4_gnn1_k64
