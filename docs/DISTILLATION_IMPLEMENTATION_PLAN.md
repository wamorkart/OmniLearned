# Distillation implementation plan — lazy companion teacher logits

## Context

Goal: distill the large 210-class **pretrain head** into the small model using
offline knowledge distillation. The teacher (large model) logits are already
computed and saved as sharded `.npz` files.

The blocker is *how training consumes them*. The repo's whole data philosophy is
**lazy streaming**: `HEPDataset` holds only `file_indices` (the list of
`(file_idx, sample_idx)` pairs) in RAM and reads one row per sample straight from
the h5 files (`f["data"][sample_idx]`, `dataloader.py:185`). The teacher-logit
plumbing that currently exists (committed) breaks that pattern — it
`np.concatenate`s **all** teacher logits into one in-RAM array and builds a Python
`dict` lookup (`dataloader.py:465-493`). At pretrain-train scale (~100M+ events ×
210 × fp16) that is tens of GB per rank for the array plus ~10-15 GB for the dict,
on every DDP rank. It also doesn't work for `--dataset pretrain` at all (globs the
literal name `"pretrain"`, and saved keys use per-dataset *local* `file_idx` while
the combined loader uses *global shifted* `file_idx`).

**Decision (repo-faithful):** make teacher logits a lazily-read **companion
HDF5**, one per source h5 file, holding a `teacher_logits` dataset indexed by
`sample_idx`. Then `__getitem__` reads one row on demand — exactly like
`f["data"][sample_idx]` — using h5py (which the repo already trusts for concurrent
DDP reads; mmap is explicitly distrusted on Lustre, see `dataloader.py:368-372`).
This removes the in-RAM array, the dict, and the need for the int64-key / fp16 /
sharding optimizations. Reads are keyed by sample identity, so shuffling is
irrelevant.

All work happens on a **branch** so it can never collide with the live evaluate
pipeline running on the cms_bsm/h1 val splits.

## Already done — no work needed

- CLI flags `--distill / --teacher-labels-dir / --teacher-tag / --distill-alpha /
  --distill-beta / --distill-T` exist and are forwarded (`cli.py:96-162`).
- KD loss `get_distill_loss` (`utils.py:288`, correct Hinton KL) wired into the
  train loop (`train.py:117-133`), val loop (`221-234`), logging, and `run()`
  (`439-444`, `671-674`).
- `collate_point_cloud` already stacks per-sample `teacher_logits` and yields
  `None` if any sample lacks it (`dataloader.py:52-58`) — works unchanged.
- Teacher logits saved on `/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/`:
  train complete for all 7 constituents; cms_bsm/h1 **val** in progress.

## To-do

### 1 — Branch & data hygiene
- [ ] `git checkout -b distill-lazy-teacher`.
- [ ] Remove the aborted h1 stray files (different chunking, would corrupt a
  positional scatter):
  `rm /pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/outputs_pretrain_l_h1_train_chunk*of5_rank*.npz`
- [ ] Confirm cms_bsm/h1 **val** evaluates finished (all chunk×rank `.npz` present)
  before running conversion for those.

### 2 — New conversion script `build_teacher_h5.py` (repo root, next to `concat_logits.py`)
One-time: sharded `.npz` → per-source-h5 companion `.h5`. Detailed steps:
- [ ] Args: `--npz-dir` (e.g. `.../teacher_logits/TEST`), `--tag` (e.g.
  `pretrain_l`), `--data-path` (`/global/cfs/cdirs/m4567/www/`), `--out-dir`
  (companion root, default `.../teacher_logits/companion`), `--dataset` (single
  name, comma-list, or `pretrain` to expand to the 7), `--split` (train/val/test).
- [ ] For each dataset×split, enumerate source h5 **using the exact same ordering
  as `load_data`**: `sorted(glob("*.h5") + glob("*.hdf5"))` over
  `<data-path>/<dataset>/<split>/`. This fixes `file_idx ↔ filename`.
- [ ] Read `num_classes` (= 210) from the first npz's `logits.shape[1]`; read each
  source file's row count `N_f` from `len(f["data"])`.
- [ ] Pre-create each companion `<out-dir>/<dataset>/<split>/<stem>.h5` with a
  `teacher_logits` dataset shape `(N_f, C)` fp16, **initialized to NaN** (sentinel
  for coverage check).
- [ ] Stream the matching npz one at a time (bounded RAM): for each, group rows by
  `file_idx`; for each group do a sorted fancy-index write
  `companion[fid]["teacher_logits"][sorted_sidx] = logits[rows]`.
- [ ] Coverage check: re-scan each companion in chunks (~1M rows) for remaining
  NaN rows; **fail loudly** listing any file/positions still unfilled (a NaN
  teacher logit would poison the KD loss). Also warn on any cell written twice
  with differing values (detects stray/duplicate npz).
- [ ] Print per-dataset summary: files written, total rows, C.

### 3 — `HEPDataset`: lazy companion reads (`dataloader.py`)
- [ ] Replace `__init__` params `teacher_logits_arr` / `teacher_lookup` with a
  single `teacher_file_paths=None` (a list **parallel to `file_paths`**; entry =
  companion `.h5` path or `None`). Update docstring.
- [ ] Add `self._teacher_cache = {}` and a `_get_teacher_file(file_idx)` helper
  that opens the companion lazily per worker — mirror `_get_file`
  (`dataloader.py:169-174`).
- [ ] In `__getitem__`, replace the dict-lookup block (`:228-236`) with:
  if `teacher_file_paths` is set and the entry isn't `None`,
  `sample["teacher_logits"] = torch.from_numpy(tf["teacher_logits"][sample_idx].astype(np.float32))`,
  else `None`.
- [ ] Extend `__del__` (`:240-246`) to also close `_teacher_cache` handles.

### 4 — `load_data`: build `teacher_file_paths` (`dataloader.py`)
- [ ] Replace the post-loop in-RAM block (`:465-493`) — no glob/concat/dict.
- [ ] Set `teacher_on = teacher_labels_dir is not None and teacher_tag is not None`
  and `teacher_file_list = [] if teacher_on else None` before the file loop.
- [ ] Inside the loop, right after `file_list.extend(map(str, h5_files))`
  (`:337`), append one companion path per h5:
  `<teacher_labels_dir>/<names[iname]>/<dataset_type>/<h5.stem>.h5`. Raise a clear
  per-constituent error naming the dataset/split if a companion is missing.
- [ ] Pass `teacher_file_paths=teacher_file_list` to `HEPDataset` (`:495-509`);
  drop the old `teacher_logits_arr=` / `teacher_lookup=` kwargs.
- [ ] Works for both `--dataset pretrain` (7 constituents, companion path uses
  `names[iname]`) and single-dataset distillation (one name) — no `index_shift`
  math needed, since companions are keyed by source-file identity, not by global
  row index.

### 5 — Confirm no `train.py` / `cli.py` changes
- [ ] Verify `run()` already forwards `teacher_labels_dir`/`teacher_tag` into both
  `load_data` calls (`train.py:504-528`) — it does; nothing to change.

### 6 — Run the conversion (decide storage; size the job)
- [ ] Pick `--out-dir` on scratch with room (~same size as the existing fp16 npz
  logits; old npz can be deleted afterward to reclaim space).
- [ ] val + small datasets: run interactively per dataset.
- [ ] pretrain **train** is large I/O (reads all train npz, hundreds of GB) —
  run as `sbatch` (one job per dataset, or an array over the 7) rather than a 4h
  interactive slot.

### 7 — Coverage verification (pre-launch gate)
- [ ] Confirm every constituent × {train, val} has companions and the conversion's
  coverage check passed with **zero** NaN rows. Incomplete coverage = guaranteed
  bad batch (collate nulls the batch → train loop raises).

### 8 — Small-scale end-to-end smoke test
- [ ] On one small constituent (e.g. `--dataset atlas`, val) run a 1-2 iteration
  `omnilearned train --distill ...` and confirm: startup loads companions, the
  `loss_kd` log line is finite and non-zero, no missing-teacher errors.

### 9 — Launch full pretrain distillation
- [ ] `omnilearned train --distill --dataset pretrain --mode pretrain
  --num-classes 210 --size small --teacher-labels-dir <out-dir>
  --teacher-tag pretrain_l --distill-alpha 0.5 --distill-beta 0.5 --distill-T 4`
  plus the usual pretrain flags (interaction/local-interaction/use-pid/use-add/
  use-event-loss/batch/epochs).
- [ ] Launch decision to confirm at this point: **init small from the existing
  small pretrain checkpoint** (`--pretrain-tag`, recommended, faster convergence)
  **vs. from scratch** (pure-KD signal). Mix CE+KD (current 0.5/0.5, T=4) is the
  safe default.

### 10 — Monitor
- [ ] Watch the startup teacher-load lines (one per constituent) and the
  `KD Loss / KD Val Loss` log (`train.py:351`). Stop early if KD loss is NaN or
  flat.

### 11 — Phase 4 evaluation
- [ ] Evaluate the distilled small checkpoint on the downstream task(s) you care
  about and compare to: baseline small (no KD, same data) and the large teacher
  (upper bound). If distilled-small doesn't beat baseline-small meaningfully,
  tune α↑ / T↓ / train longer.

## Verification

- **Unit-ish:** after step 2, open a companion h5 and confirm
  `softmax(teacher_logits[i]).argmax()` vs the recorded `pid` reproduces the
  teacher's known accuracy on a sample of rows (sanity that scatter alignment is
  correct, not transposed/misindexed).
- **Integration:** the step-8 smoke test (finite `loss_kd`, no missing-teacher
  raise) on a single small dataset before the big run.
- **Static:** `python3 -c "import ast; ast.parse(open('src/omnilearned/dataloader.py').read())"`
  and run the repo linter (ruff, available inside the `pytorch` module) before
  committing.

## Files

- New: `build_teacher_h5.py` (repo root).
- Edit: `src/omnilearned/dataloader.py` (`HEPDataset.__init__`, `_get_teacher_file`,
  `__getitem__`, `__del__`, and the teacher section of `load_data`).
- No changes: `train.py`, `cli.py`, `utils.py` (KD already wired).
