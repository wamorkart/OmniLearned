# LHCO Anomaly Detection — Classifier Procedure for Pipeline A & B

Scope: idealized/fully-supervised classification (signal vs. background) on LHCO,
using truth labels directly — **no CATHODE-style background generation, no CWoLa
mixing**. Assumes the preprocessed LHCO classifier data already exists in this
repo's `HEPDataset` schema under `<lhco_path>/custom/{train,val,test}/*.h5`
(`data` = per-particle point cloud, `pid` = 0/1 background/signal label). See
`LHCO_CLASSIFICATION_DISTILL.md` for the preprocessing step itself (not yet
written) if that data doesn't exist yet.

Pipeline definitions: **A = distill → fine-tune** (student distilled from an
upstream, not-yet-task-specific teacher, then fine-tuned on the target task).
**B = fine-tune → distill** (teacher fine-tuned on the target task first, student
then distilled directly from it, same task). See `pipeline-ab-definitions`
project memory for the full history of why these are named this way.

---

## Pipeline A (distill → fine-tune) on LHCO

Reuses the already-distilled 210-class pretrain checkpoint — same pattern as the
top-tagging fine-tune, just swap the dataset. No new distillation step.

1. Confirm the preprocessed LHCO classifier data is laid out as
   `<lhco_path>/custom/{train,val,test}/*.h5`, each with `data` (per-particle
   4-feature point cloud), `pid` (0 = background, 1 = signal), matching
   `HEPDataset`'s expected schema.
2. Build the index cache: `omnilearned dataloader -d custom -f <lhco_path>`.
3. Pick the pretrain-distilled checkpoint to start from (e.g.
   `distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2`, the pure-KD
   pretrain run — or the `a05_b05` variant, since the two tied on top tagging).
4. Fine-tune it directly on LHCO signal-vs-background — no `--distill*` flags:
   ```bash
   omnilearned train \
     --pretrain-tag distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2 \
     --fine-tune \
     --save-tag fine_tune_lhco_distill_pretrain_s_a00_b10_T4_full500_reg52_v2 \
     --dataset custom --mode classifier --num-classes 2 \
     --path <lhco_path> --size small --interaction --local-interaction \
     --batch 128 --iterations 1000 --epoch 50 --lr 5e-5 --lr-factor 10 --wd 0.5
   ```
5. Launch via a resubmit-loop script inside a named `screen` session (not bare
   `nohup`), same pattern as every other training run in this project.
6. Evaluate on the LHCO test split:
   ```bash
   omnilearned evaluate --save-tag fine_tune_lhco_distill_pretrain_s_a00_b10_T4_full500_reg52_v2 \
     --dataset custom --dataset-type test --path <lhco_path> \
     --size small --interaction --local-interaction --num-classes 2
   ```
7. Compute metrics (accuracy, AUC, $1/\mathrm{FPR}$ at fixed signal efficiency)
   from the eval `.npz` outputs — same script pattern as `tools/metrics/compute_metrics_top.py`,
   adapted for the LHCO label convention.

## Pipeline B (fine-tune → distill) on LHCO

Not yet run on any task besides top tagging — this is new. Needs a task-specific
teacher fine-tuned on LHCO first, since none exists yet.

1. Fine-tune the **large CE-only pretrain checkpoint** (`pretrain_l`, i.e.
   `best_model_pretrain_l.pt`) on LHCO signal-vs-background — this becomes the
   LHCO teacher (`fine_tune_lhco_l`), built the same way `fine_tune_top_l` was
   for top tagging (fine-tuned from the pretrained backbone, not trained from
   scratch). Keeping this construction identical to `fine_tune_top_l` matters:
   it's what makes the LHCO result comparable to the top-tagging one rather
   than confounding "new task" with "differently-built teacher." Starting from
   scratch would also likely underfit here — LHCO's signal region has far
   fewer events (~60k) than top tagging's training set.
   ```bash
   omnilearned train \
     --pretrain-tag pretrain_l \
     --fine-tune \
     --save-tag fine_tune_lhco_l \
     --dataset custom --mode classifier --num-classes 2 \
     --path <lhco_path> --size large --interaction --local-interaction \
     --batch 128 --iterations 1000 --epoch 50
   ```
2. Evaluate the teacher standalone on the LHCO test split — sanity-check its
   accuracy/AUC before trusting it as a KD source.
3. Generate teacher logits for the LHCO train/val split (companion `.h5` files
   via `tools/preprocess/build_teacher_h5.py`, or merge logits directly into the LHCO `custom`
   files) so the student's `--distill` path has something to read.
4. Distill a small student **directly from that fine-tuned teacher, on the same
   task** — start with pure KD ($\alpha=0,\beta=1,T=4$), the config that won
   the top-tagging sweep:
   ```bash
   omnilearned train \
     --save-tag distill_lhco_small_scratch_a00_b10_T4 \
     --dataset custom --mode classifier --num-classes 2 \
     --path <lhco_path> --size small --interaction --local-interaction \
     --distill --distill-alpha 0.0 --distill-beta 1.0 --distill-temp 4 \
     --teacher-labels-dir <teacher_logits_path> --teacher-tag fine_tune_lhco_l \
     --batch 128 --iterations 1000 --epoch 50
   ```
5. Launch via a resubmit-loop script inside a named `screen` session.
6. Evaluate the distilled student on the LHCO test split.
7. Compute metrics and compare against: (a) the `fine_tune_lhco_l` teacher,
   (b) a CE-only small baseline (no KD, same architecture) to isolate the KD
   gain, and (c) Pipeline A's result above — this last comparison is the point:
   it tests whether Pipeline B's advantage over Pipeline A (seen on top
   tagging) generalizes to a second, structurally different task.

## Baseline (no distillation) — shared reference for both pipelines

One run, used as the "no KD" anchor in both pipelines' comparison tables — same
role `fine_tune_top_s` played throughout the top-tagging results. Fine-tune the
**small** CE-only pretrain checkpoint (`pretrain_s`) directly on LHCO, no
`--distill*` flags — same architecture size as the KD student in both
pipelines, so it isolates the effect of distillation specifically, not model
size:

```bash
omnilearned train \
  --pretrain-tag pretrain_s \
  --fine-tune \
  --save-tag fine_tune_lhco_s \
  --dataset custom --mode classifier --num-classes 2 \
  --path <lhco_path> --size small --interaction --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 --lr 5e-5 --lr-factor 10 --wd 0.5
```

Evaluate and compute metrics on it the same way as the other runs. Final
comparison table should have four rows: `fine_tune_lhco_s` (no KD),
`fine_tune_lhco_l` (teacher), Pipeline A result, Pipeline B result.

---

## Open items noted during planning (2026-08-18), not yet resolved

- The only LHCO data currently on `$SCRATCH`
  (`/pscratch/sd/t/twamorka/omnilearned/datasets/lhco_converted/LHCO/`) is in
  **dijet** form (`data`: shape `(N, 2, 150, 7)`, two jets per event) with a
  Keras `.weights.h5` checkpoint alongside it — signature of the upstream
  TF/Keras OmniLearn LHCO example, not this repo's PyTorch `HEPDataset` (which
  expects a single-jet point cloud per sample). **Not plug-compatible as-is.**
  Its `"pid"` field also holds $m_{jj}$, not a class label.
- Raw LHCO R&D data is available at
  `/pscratch/sd/t/twamorka/omnilearned/datasets/lhco/events_anomalydetection.h5`
  (full constituents) and `events_anomalydetection_v2_features.h5` (15
  high-level features, 1.1M events) if preprocessing from scratch is preferred
  over adapting the dijet-converted files.
- This procedure assumes that mismatch is already resolved and
  `<lhco_path>/custom/{train,val,test}/*.h5` exists in the correct schema —
  the preprocessing step itself (single-jet vs. dijet decision, writing
  `preprocess_lhco.py`) is still open and not covered here.
