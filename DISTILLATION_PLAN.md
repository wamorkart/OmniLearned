# Distillation plan: large → small pretrain model

The good news: the dataloader already has half the KD plumbing. The painful
part is that **test split alone isn't enough — you need teacher labels on
the train split too**, which is the largest data volume by far.

## Stocktake — what's done vs. what's missing

| Component | Status |
|---|---|
| Large model trained | ✅ checkpoint exists |
| Inference on **test** (via `run_all_chunks.sh`) | 🟡 in progress |
| Inference on **train** | ✅ all 7 datasets complete (raw rank files in `/pscratch/.../teacher_logits/TEST/`) |
| Inference on **val** | 🟡 missing `cms_bsm`, `h1` |
| Dataloader can read teacher logits | ✅ reads raw `outputs_*.npz` directly via glob (`dataloader.py:467-493`), builds `(file_idx,sample_idx)` lookup — no concat step needed |
| Training loss with KD term | ✅ wired in `train.py` (train `117-133`, val `221-234`); loss `get_distill_loss` in `utils.py:288` |
| CLI flags `--distill / --teacher-labels-dir / --teacher-tag / --distill-alpha / --distill-beta / --distill-T` | ✅ in `cli.py:96-107`, forwarded to `run_training` |
| Distill driver script | ❌ doesn't exist (just a `omnilearned train --distill ...` invocation) |

## Phase 0 (now): finish the test inference + sanity-check

1. Let `run_all_chunks.sh` finish all 6 chunks of `DATASET_TYPE=test`.
2. `python concat_logits.py --indir … --tag pretrain_l --dataset atlas` → produces `pretrain_l_atlas_test.npz`.
3. Sanity check: `softmax(logits).argmax()` should match the recorded `pid` ground truth at a believable accuracy (matches the test accuracy you reported during pretrain). If not, the inference run is broken — fix before continuing.

This is the dry run for the real cost in Phase 1.

## Phase 1 (the big compute): teacher labels on train + val

The pretrain mode loads **multiple constituent datasets** (`atlas, aspen,
jetclass, jetclass2, h1, cms_qcd, cms_bsm` per `load_data:307`). You need
teacher labels for **every constituent × every split you want to train
on**. That's a lot.

For each dataset name listed:

```bash
DATASET=<one of atlas/aspen/jetclass/…>
NUM_CHUNKS=<sized by dataset size>
DATASET_TYPE=train  bash run_all_chunks.sh   # the big one
DATASET_TYPE=val    bash run_all_chunks.sh   # smaller, monitoring set
```

Sizing decisions to make here:

- **Storage budget.** Each event's teacher logits = 210 × fp16 = 420 bytes +
  8 bytes for the (file_idx, sample_idx) key. For the JetClass train set
  (~100M events) that's **~43 GB per split per dataset**. Atlas train is
  bigger. Hundreds of GB of teacher logits across 7 datasets — pick a
  scratch location with room.
- **Time budget.** Train is ~10× test for most datasets. 4h interactive
  sessions × `run_all_chunks.sh` works but takes days of wall clock.
  **Switch to `sbatch -q regular`** here for unattended overnight runs —
  same script body, just `sbatch --array=0-N` over chunk index. Worth
  sketching the array-job variant when you go this route.
- **Which datasets do you actually need?** If distilling the *pretrain head*,
  all 7. If only one downstream task, you could skip irrelevant ones — but
  then it's not really pretrain distillation.

This is the decision to make first. The Phase-1 compute is much larger
than what's already sized for test.

## Inference progress checklist

7 datasets × 3 splits = 21 inference runs. Tick each box once
`run_all_chunks.sh` completes (i.e. all chunks present, all 16 rank
files per chunk, concat_logits.py merged the final `.npz`).

Use exact dataset names (matches `--dataset` flag).

### atlas
- [x] test
- [x] train
- [x] val

### aspen
- [x] test
- [x] train
- [x] val

### jetclass
- [x] test
- [x] train
- [x] val

### jetclass2
- [x] test
- [x] train
- [x] val

### cms_qcd
- [x] test
- [x] train
- [x] val

### cms_bsm
- [ ] test (only 3/24 chunks present)
- [x] train
- [ ] val

### h1
- [ ] test
- [x] train (⚠️ remove stray `chunk*of5` rank files before use — see note below)
- [ ] val

## Phase 2: KD loss in `train.py` (~50 lines of code)

The dataloader already returns `batch["teacher_logits"]` (shape
`(B, num_classes)`) when teacher files exist, else `None`. The training
loop just needs:

```python
# in train.py, where the loss is computed
if batch.get("teacher_logits") is not None:
    T = kd_temperature
    t_logp = F.log_softmax(student_logits / T, dim=-1)
    s_p     = F.softmax(batch["teacher_logits"].float() / T, dim=-1)
    kd_loss = F.kl_div(t_logp, s_p, reduction="batchmean") * (T * T)
    loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
else:
    loss = ce_loss
```

Plus four new CLI flags (`--teacher-labels-dir`, `--teacher-tag`,
`--kd-alpha`, `--kd-temperature`) forwarded into `load_data` and the loss.

Doable in one PR.

## Phase 3: distill driver

A `distill.sh` mirroring the existing train script with:

```
--pretrain-tag pretrain_s    # init small model from existing small pretrain
--teacher-labels-dir /pscratch/.../teacher_logits_pretrain/
--teacher-tag pretrain_l
--kd-alpha 0.7
--kd-temperature 4
```

Two design decisions:

- **Init from existing small pretrain, or train from scratch?** Init is
  faster to converge and usually better; scratch tests "pure KD signal."
  Recommend init.
- **Mix CE + KD, or pure KD?** Mix (α=0.5–0.7, T=2–4) is standard and
  safer. Pure KD has theoretical appeal but is more sensitive.

## Phase 4: evaluation

Re-run the distilled small checkpoint through `evaluate` on every
downstream task you care about (top tagging, etc.), comparing to:
- Baseline small model (no distillation, same data).
- Teacher large model (upper bound).

If the distilled small isn't beating the baseline small by a meaningful
margin on at least one downstream, the distillation isn't paying off —
common fix: higher α, lower T, or train longer.

## Recommended execution order

1. **This week:** finish test inference (Phase 0). Verify the teacher logits look sane.
2. **Decide:** which constituent datasets to distill on, and storage location for ~hundreds of GB of teacher logits.
3. **Switch from interactive to batch** for Phase 1 — interactive 4h slots can't realistically do ~100M-event train splits. Build `sbatch_chunks.sh` (array job) when ready.
4. **One PR** to wire KD into `train.py` + CLI. Land before Phase 1 finishes, so distillation can launch the moment teacher labels are ready.
5. Distill, then evaluate.

## Open questions to resolve before writing more code

- Are you distilling the pretrain task (210-class classifier on the union of all 7 datasets), or fine-tuning the small model on one downstream (e.g. top tagging) with the large model as teacher?
- Target metric? Beat-small-by-X% on top tagging? Match-large-within-X%? Drives loss choices.
- Is the small model the existing `size=small` from the codebase, or a different architecture?
- Storage location for teacher logits — `/pscratch` or `$CFS`?

