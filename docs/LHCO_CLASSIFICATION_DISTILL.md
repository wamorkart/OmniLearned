# LHCO Classification from the Already-Distilled Pretrain Checkpoint

Goal: pure supervised classification (signal vs. background) on LHCO,
starting from the small student already distilled (KD) from the large
teacher on the whole pretrain dataset — no CATHODE-style anomaly detection
pipeline, and **no new distillation step**.

## What we're actually doing

The KD already happened at pretraining time (see
[[distill-lazy-teacher-progress]] — e.g.
`best_model_distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2.pt`).
That checkpoint is not yet task-specific. Adapting it to LHCO is a plain
**fine-tune**, exactly like
`scripts/configs/train/ft_top_pretrain_full500_reg52_a00_b10_v2.sh` adapts it
to top-tagging — just swap the dataset. No `--distill` / teacher-logits
machinery is needed at this stage.

No code changes are needed either way. The `custom` dataset slot already
exists in `src/omnilearned/dataloader.py` and is dataset-agnostic once your
files match the expected schema.

## 1. Preprocess LHCO into OmniLearned's h5 schema

Place files under `<path>/custom/{train,test,val}/`. Each `.h5` needs:

- `data`: shape `(N, M, 4)` — per-particle `[Δη, Δφ, log(pT), log(E)]`
  relative to the jet axis (add PID/vertex features later via `--use-pid`
  / `--use-add` if desired)
- `pid`: integer label per event — signal=1 / background=0 for classification
- `global` (optional for classification): jet-level features like mass,
  multiplicity

You'll need a `preprocess_lhco.py` to convert the raw LHCO R&D delphes h5
(flat px/py/pz/E constituents) into this schema — nothing in the repo does
this yet. `tools/preprocess/preprocess_omnifold.py` is a reasonable structural
template, even though it targets a different (unfold) schema.

After preprocessing:

```bash
omnilearned dataloader -d custom -f <path>
```

to build the index file.

## 2. Fine-tune the distilled checkpoint as an LHCO classifier

Direct adaptation of `scripts/configs/train/ft_top_pretrain_full500_reg52_a00_b10_v2.sh`
(run via `scripts/run_train.sh`) — same `FINETUNE=1`/`PRETRAIN_TAG` pattern,
`DATASET=custom` instead of `top`, `DISTILL=0`:

```bash
cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_lhco_distill_pretrain_s_a00_b10_T4_full500_reg52_v2 \
  --pretrain-tag distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2 \
  --fine-tune \
  --dataset custom --mode classifier --num-classes 2 \
  --path /YOUR/LHCO/DATA/ROOT \
  --size small \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \  ∏
  --lr 5e-5 --lr-factor 10 --wd 0.5 \
  --num-workers 4 \
  --wandb --resuming"
```

`--pretrain-tag` must match the tag of your existing distilled checkpoint
exactly (whatever `best_model_distill_pretrain_s_...pt` you already have),
and `--size` must match that checkpoint's model size.

## Closest existing template

- `scripts/configs/train/ft_top_pretrain_full500_reg52_a00_b10_v2.sh` — copy it
  to a new config, set `DATASET=custom`, adjust `SAVE_TAG`; run with
  `scripts/run_train.sh <new-config>`.

## Reminder

Before editing any live `fine_tune_*` / `distill_loop_*` script in place,
check `squeue` first in case that exact script is mid-run — safer to `cp`
it to a new `*_lhco.sh` name and edit the copy.
