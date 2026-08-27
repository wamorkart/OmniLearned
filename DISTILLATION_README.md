# Knowledge Distillation — how the pieces fit together

This document explains the offline knowledge-distillation (KD) machinery added on
top of the base OmniLearned repo: what each script does, the order they run in,
and why the pipeline is shaped the way it is. For the day-by-day experiment
schedule and current run status, see `PROJECT_PLAN.md`. For the original design
rationale, see `DISTILLATION_PLAN.md` (early exploration) and
`DISTILLATION_IMPLEMENTATION_PLAN.md` (the lazy-companion-h5 rewrite actually
implemented on this branch).

## The idea, in one paragraph

A large teacher model (`size=large`) has already been trained. We want a small
student model (`size=small`) that matches more of the teacher's accuracy than a
small model trained from scratch would. This is done with **offline KD**: run
the teacher once over the data, save its logits to disk, then train the student
against a loss that mixes cross-entropy on the true label with a KL term toward
the teacher's (softened) output. "Offline" means the teacher never runs during
student training — only its saved logits are read.

Two distillation targets exist on this branch:
- **Pretrain-head KD** (210-class, the multi-dataset self-supervised pretrain task).
- **Top-tagging KD** (2-class, top vs. QCD jets), plus a similar JetClass line.

Both share the same machinery; only `--dataset`, `--num-classes`, `--mode`, and
the teacher tag differ.

## Why a "lazy companion HDF5" instead of just loading the teacher logits?

The repo's dataloader (`HEPDataset`) is built to stream: it keeps only a list of
`(file_idx, sample_idx)` pairs in RAM and reads one row at a time straight from
the source `.h5` files — this is what makes 100M+ event pretrain runs feasible.

An earlier version of the teacher-logit plumbing broke that pattern: it
concatenated *all* teacher logits into one big in-RAM array plus a Python dict
lookup. At pretrain scale (100M+ events × 210 classes) that's tens of GB per
process, replicated on every DDP rank — not viable, and it didn't handle
`--dataset pretrain` (7 constituent datasets, global vs. local file indices)
correctly at all.

The fix implemented here: teacher logits are stored as **one companion `.h5` per
source data file**, holding a `teacher_logits` dataset indexed by `sample_idx`,
living at `<companion_dir>/<dataset>/<split>/<source_stem>.h5`. `HEPDataset`
opens the companion lazily (same pattern as the data file itself) and reads one
row on demand — no in-RAM array, no dict, no dependence on shuffling order.

## Pipeline stages

```
teacher checkpoint
      │  omnilearned evaluate  (evaluate*.sh / evaluate_batch.sbatch)
      ▼
sharded teacher logits (outputs_<tag>_<dataset>_<split>_chunk<K>of<N>_rank<r>.npz)
      │  checks/validate_chunks.py, checks/sanity_logits.py   (sanity gate)
      │  concat_logits.py                                     (optional: merge to one .npz/split)
      ▼
build_teacher_h5.py
      ▼
per-source-file companion .h5  (<companion_dir>/<dataset>/<split>/<stem>.h5, teacher_logits[N_f, C])
      │  distill_smoke.sh                                      (1-2 iter smoke test)
      ▼
omnilearned train --distill ...  (distill_train*.sh)
      │  distill_loop*.sh                                      (salloc wrapper, auto-resubmit)
      │  distill_health_check.sh                                (background monitor)
      ▼
distilled student checkpoint (best_model_<save-tag>.pt)
      │  omnilearned evaluate  (evaluate_*_distill.sh)
      ▼
compute_metrics_top.py / compute_metrics_jetclass.py            (AUC, accuracy, 1/FPR)
```

### 1. Generate teacher logits (`evaluate*`)

Run the *teacher* checkpoint through `omnilearned evaluate` over whichever
split(s) the student will train on. Each rank writes its own shard, so a run
produces many small files:

- `evaluate.sh` — simple single-`srun` evaluate (small pretrain-head runs, or
  quick top-tagging evaluate).
- `evaluate_top_for_distill.sh` — evaluates `fine_tune_top_l` (large model
  fine-tuned on top tagging) on train/val of the `top` dataset; this produces
  the teacher signal that top-tagging KD trains against.
- `evaluate_batch.sbatch` — batch-queue (`sbatch`) chunked evaluate for the big
  pretrain-scale runs (`--num-chunks`/`--chunk-idx`), needed because a 100M+
  event split can't finish inside one interactive session. Submitted per
  dataset/split/chunk; expects `DATASET`, `DATASET_TYPE`, `NUM_CHUNKS`,
  `CHUNK_IDX`, etc. exported by a submission wrapper.
- `run_eval.sh <config>` + `configs/eval/<config>.sh` — config-driven
  "evaluate one student checkpoint on a split" (top/jetclass distill, micro,
  mlp, pretrain-KD, ptq lineages). Replaces the old per-checkpoint
  `evaluate_{top_distill*,jetclass_*,top_micro_ce}.sh`. See `scripts/README.md`.
- `evaluate_chunk.sh`, `evaluate_top_large.sh`,
  `evaluate_metrics_top_sweep.sh`, `evaluate_metrics_top_T_sweep.sh` —
  structurally different (sharded/chunked dumps, multi-tag metrics tables),
  still their own scripts.

Output filenames encode tag/dataset/split/chunk/rank, e.g.
`outputs_pretrain_l_atlas_train_chunk3of24_rank5.npz`, each holding `logits`
`(N, C)` fp16 and `sample_keys` `(N, 2)` int64 = `(file_idx, sample_idx)` (local,
per-dataset file index).

```bash
# Quick single-srun evaluate of the top-tagging teacher (fine_tune_top_l), val split.
DATASET_TYPE=val bash evaluate_top_for_distill.sh

# Chunked batch-queue evaluate of the pretrain teacher (pretrain_l) over train,
# one sbatch job per (dataset, chunk) — submitted directly with --export:
sbatch --export=ALL,DATASET=atlas,DATASET_TYPE=train,NUM_CHUNKS=24,CHUNK_IDX=3,\
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST,\
SAVE_TAG=pretrain_l,CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/,\
DATA_PATH=/global/cfs/cdirs/m4567/www/,SIZE=large \
  evaluate_batch.sbatch
# (repeat --export=...,CHUNK_IDX=0..23 for every chunk, or loop over it in a
# submission wrapper)
```

### 2. Sanity-check the shards before spending compute on conversion

- `checks/validate_chunks.py` — read-only pass over one dataset/split's chunk
  files: confirms the expected rank count per chunk, consistent keys/shapes
  across files, no NaN/Inf, and estimates the peak RAM `concat_logits.py` would
  need for a full in-RAM merge (informational — the actual pipeline no longer
  needs that merge for training, only `build_teacher_h5.py` does, and it streams).
- `checks/sanity_logits.py` — semantic check on a sample of files: confirms
  `softmax(logits)` reproduces the saved `prediction`/`event_prediction` arrays,
  and that the teacher's predicted classes aren't collapsed onto one class.
  This is the check that would catch a broken/truncated inference run before it
  poisons a multi-day distillation job.
- `checks/ab_precision.sh` / `checks/ab_compare.py` — A/B precision comparison
  (e.g. fp32 vs. default AMP) on teacher inference, to confirm evaluate
  precision settings don't materially change the saved logits.
- `checks/compare_models.py` — compare two checkpoints' predictions directly.

```bash
# Edit OUT/TAG/DATASET/SPLIT/NUM_CHUNKS/EXPECTED_RANKS at the top of the script
# for the run you just produced, then:
python checks/validate_chunks.py

# Semantic check (edit OUT/PAT at the top for your tag/dataset/split first):
python checks/sanity_logits.py
```

### 3. `concat_logits.py` (optional merge)

Groups the per-chunk/per-rank `.npz` shards by split and concatenates them into
one `<tag>_<dataset>_<split>.npz`. Useful for quick sanity checks (e.g. loading
one file to spot-check teacher accuracy) but **not** required for training —
`build_teacher_h5.py` reads the raw sharded files directly, streaming, so it
never needs the concatenated array in RAM.

```bash
python concat_logits.py \
  --indir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST \
  --tag pretrain_l --dataset atlas
# -> pretrain_l_atlas_test.npz (or _train.npz/_val.npz, one per split found)
```

### 4. `build_teacher_h5.py` (the key conversion step)

One-time, per dataset/split: converts the sharded `.npz` teacher logits into the
lazy companion `.h5` files training actually reads.

- Enumerates source files with the **exact same ordering** the dataloader uses
  (`sorted(glob("*.h5") + glob("*.hdf5"))` per dataset/split) so `file_idx` in
  the saved `sample_keys` lines up with the right source file.
- Pre-creates each companion, shape `(N_f, C)` fp16, **initialized to NaN** as a
  sentinel, then streams each npz shard in and scatter-writes rows by
  `(file_idx, sample_idx)` — bounded RAM regardless of dataset size.
- Re-scans every companion afterward and **fails loudly** if any row is still
  NaN (incomplete coverage) or if a cell was written twice with differing
  values (duplicate/stray npz shard) — both are silent-corruption risks
  otherwise.
- `--dataset pretrain` expands to the 7 pretrain constituents
  (`atlas, aspen, jetclass, jetclass2, h1, cms_qcd, cms_bsm`) in one invocation.
- `--skip-existing` lets you re-run safely after an interrupted conversion job
  without re-truncating datasets that already converted cleanly.

```bash
# Top-tagging teacher (2-class), train+val, single dataset:
python build_teacher_h5.py \
  --npz-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST \
  --tag fine_tune_top_l \
  --data-path /global/cfs/cdirs/m4567/www/ \
  --out-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l \
  --dataset top --split train,val

# Pretrain-head teacher (210-class), all 7 constituents, resumable:
python build_teacher_h5.py \
  --npz-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST \
  --tag pretrain_l \
  --data-path /global/cfs/cdirs/m4567/www/ \
  --out-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion \
  --dataset pretrain --split train,val --skip-existing
```

### 5. Smoke test (`distill_smoke.sh`)

Before committing a multi-day allocation: 1 epoch × 2 iterations on one small
dataset (`atlas`), single interactive allocation. Confirms companions load, no
missing-companion errors, `loss_kd` is finite and non-zero, and forward/backward
works end to end. Always run this after any change to the dataloader or a fresh
companion conversion, before launching the full loop.

```bash
salloc -C gpu -q interactive -t 30 --nodes 1 --ntasks-per-node 4 \
       --gpus-per-node 4 -A m3246 \
       bash -c "./distill_smoke.sh 2>&1 | tee distill_smoke.log"
# check the log for: "loss_kd" finite/non-zero, no "missing companion" error
```

### 6. Training drivers (`distill_train*.sh`)

Each is a single `srun` invocation of `omnilearned train --distill ...`, meant to
run *inside* an existing SLURM allocation (interactive or batch). Common KD
flags:

| Flag | Meaning |
|---|---|
| `--distill` | turn on the KD loss path |
| `--teacher-labels-dir` | companion root, e.g. `.../teacher_logits/companion` |
| `--teacher-tag` | just a label now (companion path is keyed by dataset/split/filename, not by tag) |
| `--distill-alpha`, `--distill-beta` | CE-vs-KD loss mixing weights |
| `--distill-t` | softmax temperature |

Variants:
- `distill_train.sh` — pretrain-head KD, student trained from scratch, teacher =
  `pretrain_l`.
- `distill_train_top.sh` — top-tagging KD, student from scratch, teacher =
  `fine_tune_top_l` (large model fine-tuned on top tagging).
- `distill_train_top_sweep.sh` / `distill_train_top_rep.sh` — same as above but
  read `$ALPHA`/`$BETA`/`$DISTILL_T`/`$SAVE_TAG` from the environment, so a sweep
  loop can parameterize a family of runs without editing the script.
- `distill_train_jetclass*.sh`, `distill_train_jetclass_a00.sh`,
  `distill_train_jetclass_pretrain_l.sh` — JetClass-specific analogues (student
  init variants: scratch vs. init from a pretrain checkpoint).
- `run_train.sh ft_*` (configs `configs/train/ft_*.sh`) — **not KD**: plain
  `--fine-tune` runs (`FINETUNE=1 DISTILL=0 PRETRAIN_TAG=…`), either a CE-only
  baseline or a fine-tune of a KD-pretrained checkpoint (e.g.
  `distill_pretrain_s_scratch_a05_T4` warm-started, then fine-tuned on top
  tagging) — the "pretrain-KD-init" arm of the init ablation. Replaces the old
  per-run `fine_tune_{jetclass,top_distill_pretrain*,dctr_*}.sh`.

```bash
# Run a driver directly inside an existing allocation (e.g. after salloc):
bash distill_train_top.sh          # fixed-config top-tagging KD
bash distill_train.sh              # pretrain-head KD

# Sweep driver: reads ALPHA/BETA/DISTILL_T/SAVE_TAG from the environment
# instead of hardcoding them (used internally by the loop wrappers, but can
# be run standalone inside an allocation too):
ALPHA=0.3 BETA=0.7 DISTILL_T=4 SAVE_TAG=distill_top_small_scratch_a03_b07_T4 \
  bash distill_train_top_sweep.sh

# The underlying omnilearned CLI call any of the above wrap, spelled out:
omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_top_small_scratch_a05_T4 \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ --size small \
  --interaction --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 --lr 5e-4 --wd 0.5 --num-workers 4 \
  --distill \
  --teacher-labels-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb --resuming
```

### 7. Loop wrappers (`distill_loop*.sh`)

Perlmutter's interactive queue only grants 4h slots. These wrap the training
driver in a loop of `salloc` → run → check exit code → resubmit, so a
multi-day training run survives repeated 4h allocation expirations without
manual babysitting. Exit 0 (all epochs done) stops the loop; non-zero (timeout/
preemption) resubmits after a short sleep, up to `MAX_LOOPS`. Meant to run
inside `tmux`/`screen` so they survive SSH disconnects.

- `distill_loop.sh` — pretrain-head KD (`MAX_LOOPS=90`, sized for the ~500-epoch
  pretrain run).
- `distill_loop_top.sh` — top-tagging KD, fixed α/β/T (`MAX_LOOPS=20`).
- `distill_loop_top_sweep.sh ALPHA BETA [T]` / `distill_loop_top_T_sweep.sh T
  [rep]` / `distill_loop_top_T_sweep_below1.sh` — parameterized sweep launchers;
  each derives a unique `SAVE_TAG` from its arguments (so wandb run names and
  checkpoint files don't collide) and calls `distill_train_top_sweep.sh` with
  those env vars set.
- `distill_loop_top_rep.sh` — repeat-seed run of the winning config, to estimate
  run-to-run variance.
- `distill_loop_jetclass*.sh`, `fine_tune_loop_jetclass.sh` — JetClass analogues.
- `distill_loop_top_pretrain_finetune.sh` — loop wrapper for the
  pretrain-KD-init → top-tagging fine-tune arm.

```bash
# Always inside tmux/screen so it survives disconnects:
tmux new-session -s distill_top 'bash distill_loop_top.sh'

# Fixed pretrain-head KD loop (long-running, up to 90 sessions):
screen -dmS distill_pretrain bash distill_loop.sh

# Alpha/beta sweep point: alpha=0.25, beta=0.75, T=4 (own log dir + SAVE_TAG):
tmux new-session -s distill_a025_b075 'bash distill_loop_top_sweep.sh 0.25 0.75 4'

# Pure-KD temperature sweep point: T=8, first replicate:
tmux new-session -s distill_T8 'bash distill_loop_top_T_sweep.sh 8'
# ...second replicate of the same T, tagged distinctly:
tmux new-session -s distill_T8_r1 'bash distill_loop_top_T_sweep.sh 8 r1'
```

### 8. Monitoring (`distill_health_check.sh`)

Run as a second, independent background process (separate `screen`/`tmux`
session) alongside a loop script. Every `CHECK_INTERVAL` it appends a snapshot
to a log: `squeue` state, the loop's own `summary.log` tail, a grep for
errors/NaN in the latest session log, checkpoint file mtime/size (proof of
progress), and the latest wandb `loss`/`loss_kd` via the wandb API with an
explicit NaN check. Log-only — it doesn't page anyone, so check the log
manually when returning to a run.

```bash
screen -dmS distill_health bash distill_health_check.sh
# ... later:
tail -f /pscratch/sd/t/twamorka/omnilearned/results/distill_health_check.log
```

### 9. Evaluate the distilled student + compute metrics

Same `omnilearned evaluate` machinery as step 1, pointed at the **student**
checkpoint (`run_eval.sh top_distill`, `run_eval.sh top_distill_pretrain`,
`run_eval.sh jetclass_distill`, `run_eval.sh jetclass_finetune`; or
`evaluate_train.sh` for the large teacher), writing
`outputs_<save-tag>_<dataset>_test_*.npz`.

Then:
- `compute_metrics_top.py --indir <dir> --tag <save-tag>` — binary top-tagging
  metrics: accuracy, AUC, and 1/FPR (background rejection) at 50%/30% signal
  efficiency.
- `compute_metrics_jetclass.py` — JetClass analogue (multi-class).

Compare against two references for every distilled checkpoint: the CE-only
baseline of the same size (no KD, same data) and the teacher (large model) —
the whole point is to land closer to the teacher than the baseline does.

```bash
bash run_eval.sh top_distill   # writes outputs_<save-tag>_top_test_rank*.npz

python compute_metrics_top.py \
  --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill/ \
  --tag distill_top_small_scratch_a05_T4
# => Accuracy, AUC, 1/FPR @ 50%/30% signal efficiency

python compute_metrics_jetclass.py \
  --indir /pscratch/sd/t/twamorka/omnilearned/eval/jetclass_distill/ \
  --tag <jetclass-save-tag>
# or, from a concat_logits.py output:
python compute_metrics_jetclass.py --file <tag>_jetclass_test.npz
```

## Recipe: launching a brand-new KD line

1. Make sure the teacher checkpoint exists and evaluate it on the splits you'll
   train on (`evaluate_top_for_distill.sh`-style script, once per split).
2. `checks/validate_chunks.py` then `checks/sanity_logits.py` on the resulting
   shards — do not proceed if either fails.
3. `build_teacher_h5.py --npz-dir ... --tag <teacher_tag> --dataset ... --split
   train,val --out-dir <companion_root>` — watch for the final "zero NaN"
   summary per dataset/split.
4. `distill_smoke.sh` inside a small interactive allocation — confirm finite,
   non-zero `loss_kd`.
5. Write/adapt a `distill_train_*.sh` (teacher dir/tag, `--distill-alpha/beta/t`,
   `--save-tag`) and a `distill_loop_*.sh` wrapper with a fresh log directory.
6. Launch the loop in `tmux`/`screen`; optionally start
   `distill_health_check.sh` in a second session.
7. Once converged: evaluate the student checkpoint, run the matching
   `compute_metrics_*.py`, and compare to baseline + teacher.

## Current experiment status

Status of individual runs (which configs are done/partial/pending) changes
daily and is tracked in `PROJECT_PLAN.md`, not duplicated here — that file has
the live ablation matrix (`save_tag` × α × T × init) and the week-by-week
schedule.
