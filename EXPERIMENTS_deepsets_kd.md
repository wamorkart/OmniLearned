# DeepSets top-tagging KD ablation — experiment log

Rationale and history for each `CONFIG` of `distill_train_top_deepsets.sh`
(+ `distill_loop_top_deepsets.sh` to run it, `run_eval_deepsets.sh` to
evaluate). Consolidated 2026-08-27 from the per-script comment blocks of
~26 near-duplicate launcher scripts.

Student: DeepSets / PFN (Phi-embed → pooled → rho-MLP, no attention), random
init ("scratch" in the tags = random-init *student*, not "no teacher").
DeepSets ignores `--interaction` / `--local-interaction`.

Base recipe (all configs unless noted): `--batch 128 --iterations 1000
--epoch 50 --lr 5e-4 --wd 0.5`, KD `alpha=0.5 / beta=0.5 / T=4`, teacher
`fine_tune_top_l` (large PET2, 373.7M params), companion logits under
`/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l/`.
4 nodes × 4 GPUs, Perlmutter interactive queue (240-min walltime cap → the
loop resubmits with `--resuming`; ~2h46m for 50 epochs at "small").

## The `--arch deep-sets` forwarding bug (2026-08-04)

`cli.py`'s `train` command declared `--arch` but never forwarded it into
`run_training(...)`, so every `--arch deep-sets` run before commit `455edc2`
silently trained a plain PET2-small model. Fixed (arch now passed through),
confirmed via `deepsets_smoke.sh` (0.30M params = DeepSets small). The
reference config's tag carries an `_archfix0804` suffix so `--resuming`
cannot pick up a mislabeled pre-fix checkpoint.

## The `--wandb` / multi-node NCCL fork-hang (2026-08-05)

On multi-node DDP, `--wandb` was root-caused (live py-spy stack dump) to a
hang inside the first `loss.backward()` NCCL all-reduce: `wandb.init()` forks
a `wandb-core` service subprocess *after* `DDP(model)` has established the
NCCL communicator, and that fork corrupts NCCL state.
`WANDB_START_METHOD=thread` does not fix it (non-functional in wandb 0.27.2).
Never fixed in code. `train.py` writes `epoch_log_<tag>.csv` every epoch
regardless of wandb, so a hang costs a restart, not progress.

Configs `a00_b10`, `ewpool`, `teachS`, `ce` default `WANDB=1` (enabled at
user request; the a00_b10_wandb and ewpool runs both completed 50 epochs
cleanly). Configs `a05`, `wd005`, `distillnet` default `WANDB=0`. Override
either way with `WANDB=0/1`. Symptom to watch for: a mid-epoch stall with no
heartbeat.

## Configs

### `a05` — reference / confirmed best
Tag `distill_top_deepsets_small_scratch_a05_T4_archfix0804`. The baseline
every other config is a controlled one-variable change from. KD a0.5/b0.5/T4,
wd 0.5. Result: **93.30%** acc (vs PET2-small teacher-quality baselines
fine_tune_top_s 94.38%, fine_tune_top_l 94.44%).

### `a00_b10` — pure KD (DistillNet-style)
Tag `distill_top_deepsets_small_scratch_a00_b10_T4`. `alpha=0, beta=1`: drop
the ground-truth CE term, train purely to mimic the teacher (matches
DistillNet, arXiv:2311.12551, and this project's PET2-student T-sweep winner:
AUC 0.9879 @ a00/b10 vs 0.9875 @ a05/b05).
History: the original a00_b10 checkpoint (2026-08-13) was a mislabeled PET2
model (PET2 state-dict keys, trained before the arch-forwarding fix landed),
archived to `checkpoints/broken_mislabeled_pet2/`. Retrained under the same
tag with `--wandb` on. Pipeline-A pretrain-KD α/β ablation closed out
2026-08-17 (a00_b10_v2 vs a05_b05 fine-tune tie) — see
`distill-lazy-teacher-progress` memory.

### `wd005` — weight-decay ablation
Tag `distill_top_deepsets_small_scratch_a05_wd005_T4`. `--wd 0.05` instead of
0.5. Rationale: wd 0.5 was tuned for 5M–100M+ param PET2 configs;
DeepSets-small is ~0.30M and may be underfitting under that much decay.
Isolates wd from the arch/loss questions. Finished 2026-08-20 (50/50);
underperformed the a05 baseline.

### `ewpool` — energy-weighted pooling
Tag `distill_top_deepsets_small_scratch_a05_ewpool_T4`. Adds
`--energy-weighted-pool`: pool per-particle phi embeddings weighted by raw
pT instead of a plain masked mean, so the student can emphasize hard
constituents the way PET2's attention does — no new params, still
attention-free / FPGA-friendly. Finished 2026-08-21 (50/50);
underperformed the a05 baseline.

### `distillnet` — size scan
Tag `distill_top_deepsets_distillnet_scratch_a05_T4`. `--size distillnet`
(10,981 params: base_dim=32 / phi=2 / rho=1, ~DistillNet-paper parity — see
`get_deepsets_parameters` in `utils.py`) vs "small" (298,887 params). Same
KD recipe. Isolates architecture size from the recipe ablations. Result:
**92.85%** — close 2nd to a05 despite being ~27× smaller. Evaluate with
`run_eval_deepsets.sh <tag> distillnet`.

### `teachS` — teacher-capacity ablation
Tag `distill_top_deepsets_small_scratch_teachS_a05_T4`. Distill the *small*
teacher `fine_tune_top_s` (PET2-small, 2.71M params) instead of the large
one. Hypothesis: student↔teacher capacity gap is ~1250× against
fine_tune_top_l but only ~9× against fine_tune_top_s; a teacher far more
capable than the student can distill *worse* (its softened outputs encode
structure the student cannot represent), which would explain why every
loss/regularization ablation failed to close the 93.30% vs 94.38% gap. The
two teachers are near-equal in accuracy (94.38% vs 94.44%), so this isolates
teacher *capacity* almost cleanly from teacher *quality*. Teacher companions
built 2026-08-24 by `build_teacher_h5_top_s.sh` from the pre-existing 2026-05
npz shards (no GPU re-inference): 1,211,000 train / 403,000 val rows, zero
NaN.

### `ce` — CE-only no-teacher baseline
Tag `train_top_deepsets_small_ce_scratch`. `--distill` and all teacher flags
dropped (`train.py` gates the whole KD path behind `if distill:`), so plain
cross-entropy, no teacher companion files touched. The control every DeepSets
KD result was missing: the reported 93.30% vs 94.38% gap conflates (a)
architecture capacity — DeepSets 0.30M vs PET2-small 2.71M — with (b) KD vs
plain CE. This run isolates (b).

## Replicate / seed-spread runs

No `--seed` flag exists in the CLI, so a fresh `SAVE_TAG` + a fresh process =
a true independent replicate (fresh random init + shuffle order from system
entropy). Run any config as a replicate with:

    SAVE_TAG=<tag>_r2 CONFIG=<name> bash distill_loop_top_deepsets.sh

## Results summary

| config     | params  | teacher         | test acc | notes                    |
|------------|---------|-----------------|----------|--------------------------|
| a05        | 298,887 | fine_tune_top_l | 93.30%   | reference / best         |
| distillnet | 10,981  | fine_tune_top_l | 92.85%   | close 2nd, ~27× smaller  |
| a00_b10    | 298,887 | fine_tune_top_l | —        | pure KD                  |
| wd005      | 298,887 | fine_tune_top_l | < a05    | wd 0.05                  |
| ewpool     | 298,887 | fine_tune_top_l | < a05    | pT-weighted pooling      |
| teachS     | 298,887 | fine_tune_top_s | —        | teacher-capacity test    |
| ce         | 298,887 | (none)          | —        | CE-only control          |

See the `fpga-deepsets-distillation-progress` memory for the current
best-of-breed status and the PTQ/QAT follow-on work.
