module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)



export BATCH=2



export NEVT=100

cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag quijote_noint_s_$NEVT --dataset quijote --epoch 20 --lr 5e-4 --size small --wd 0.0 --num-classes 2 --num-feat 3 --batch $BATCH --mode regression --nevts 6   --interaction-type astro  --num-coord 3 --k 20  --iterations 100"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_quijote_noint_s_$NEVT --dataset quijote --epoch 20 --lr 5e-4 --size small --wd 0.0 --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --nevts 6  --interaction-type astro  --num-coord 3 --k 20 --iterations 100"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


export VEPOCHS=300
export REPOCHS=250
export LR=1e-4
export WD=1.0


export NEVT=1000



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag quijote_noint_s_$NEVT --dataset quijote --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression --nevts 62    --interaction-type astro   --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_quijote_noint_s_$NEVT --dataset quijote --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3 --nevts 62    --interaction-type astro  --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "




export NEVT=19651



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag quijote_noint_s_$NEVT --dataset quijote --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH --mode regression   --interaction-type astro  --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_quijote_noint_s_$NEVT --dataset quijote --epoch $VEPOCHS --lr $LR --size small --wd $WD --num-classes 2 --num-feat 3 --batch $BATCH  --mode segmentation --num-gen-classes 3   --interaction-type astro  --num-coord 3 --k 20"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



