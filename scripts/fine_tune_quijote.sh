module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


export BATCH=4




export NEVT=100


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_quijote_s_$NEVT --dataset quijote --epoch 20 --lr 5e-6 --size small --wd 0.1 --num-classes 2 --num-feat 3 --batch $BATCH --mode regression --nevts 6  --lr-factor 2 --fine-tune --pretrain-tag pretrain_s --interaction-type astro --local-interaction --num-coord 3 --k 20 --iterations 60"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_quijote_s_$NEVT --dataset quijote --epoch 30 --lr 1e-5 --size small --wd 1.0 --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --nevts 6  --lr-factor 10 --fine-tune --pretrain-tag pretrain_s  --interaction-type astro  --local-interaction --num-coord 3 --k 20 --iterations 100"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


export VEPOCHS=300
export REPOCHS=100
export LR=1e-5
export WD=0.0
export LRFACTOR=10.
export VLRFACTOR=20.



export NEVT=1000



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_quijote_s_$NEVT --dataset quijote --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression --nevts 62  --lr-factor $LRFACTOR --fine-tune --pretrain-tag pretrain_s  --interaction-type astro  --local-interaction --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_quijote_s_$NEVT --dataset quijote --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --nevts 62  --lr-factor $VLRFACTOR --fine-tune --pretrain-tag pretrain_s  --interaction-type astro  --local-interaction --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



export NEVT=19651



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_quijote_s_$NEVT --dataset quijote --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression   --lr-factor $LRFACTOR --fine-tune --pretrain-tag pretrain_s  --interaction-type astro  --local-interaction --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_quijote_s_$NEVT --dataset quijote --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3  --lr-factor $VLRFACTOR --fine-tune --pretrain-tag pretrain_s  --interaction-type astro  --local-interaction --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "





# export NEVT=19651



# cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_quijote_s_$NEVT --dataset quijote --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression  --lr-factor $LRFACTOR --fine-tune --pretrain-tag pretrain_s --wandb  --interaction-type astro  --local-interaction --num-coord 3 --k 20"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_quijote_s_$NEVT --dataset quijote --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3  --lr-factor $VLRFACTOR --fine-tune --pretrain-tag pretrain_s  --interaction-type astro --local-interaction --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



