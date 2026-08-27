module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


export BATCH=4
export VEPOCHS=15
export REPOCHS=400
export LR=5e-4
export WD=1.0


export NEVT=100

cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_noint_s_$NEVT --dataset camels --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression  --nevts 6 --num-coord 3"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_camels_s_$NEVT --dataset camels --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --nevts 6   --interaction-type astro  --num-coord 3 --local-interaction --iterations 100"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


export NEVT=300

cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_noint_s_$NEVT --dataset camels --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression  --nevts 19   --interaction-type astro  --num-coord 3"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_camels_noint_s_$NEVT --dataset camels --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --nevts 19   --interaction-type astro  --num-coord 3"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "

export NEVT=600

cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_noint_s_$NEVT --dataset camels --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression --num-coord 3"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_camels_noint_s_$NEVT --dataset camels --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --num-coord 3"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


