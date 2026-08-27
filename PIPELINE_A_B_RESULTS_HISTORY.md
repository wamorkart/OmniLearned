# Pipeline A & B — Full Results History

Definitions (corrected):

- **Pipeline A = distill → fine-tune.** A student is distilled from an upstream teacher
  (not yet specialized to the downstream task — either the full 210-class pretrain model, or
  a JetClass-level distillation) and only *afterward* fine-tuned on the actual downstream
  task (top tagging).
- **Pipeline B = fine-tune → distill.** A teacher is fine-tuned on the downstream task
  *first*, and a student is distilled directly from that already-fine-tuned, task-specific
  teacher, on the same task. Tested **only on top tagging so far**.

Everything below is drawn from `distill-lazy-teacher-progress.md` (project memory) and
verified against files currently on disk (`/pscratch/sd/t/twamorka/omnilearned/`) and in
this repo. Numbers are only included where an actual eval/metrics run produced them —
sections that were scripted but never executed are marked as such, not filled in.

---

## Pipeline A — distill → fine-tune

### Status: **COMPLETE** for the pretrain-level variant; JetClass-level variant complete with a negative result; a pure-KD JetClass sub-variant never launched.

### Variant 1: pretrain-level distill → top-tagging fine-tune

**Status: COMPLETE (closed out 2026-08-17).**

#### Two dead runs at small scale (Jun–Jul 2026)

The first two attempts to KD-distill the full 210-class pretrain head never converged:

- `distill_pretrain_s_scratch_a05_T4` (launched Jun 15): reached epoch 30/500 (6%), crashed
  repeatedly, resubmit loop went silent for 2.5 weeks.
- `distill_pretrain_s_scratch_a05_b05_T4` (restarted Jul 2): reached epoch 2/500 (0.4%),
  killed by an NCCL collective timeout, never resumed (dead resubmit loop).

Both left unconverged checkpoints on disk (epoch-30 and epoch-2 states) — never usable for a
real result. A planned fine-tune/eval off the epoch-30 checkpoint
(`fine_tune_top_distill_pretrain_s_a05_T4`) was scripted but **never actually run** — no
output files exist under `eval/top_distill_pretrain/` on disk.

#### Root-causing and infra fixes (Jul 14 – Aug 8)

The run was revived starting 2026-08-05 after several real bugs were found and fixed:

1. **Root cause of both original crashes**: `HEPDataset`'s per-worker `_LRUFileCache` held
   open file handles for every file touched (no real cap) on the ~7,000-file full pretrain
   mixture read from CFS/Lustre → burst of file-locking stalls at epoch-0 cold start → one
   straggler rank → NCCL watchdog abort. Also, the default 30-min PyTorch DDP watchdog
   timeout was never actually being widened by the env var the scripts thought controlled it.
2. **Fixes**: `HDF5_USE_FILE_LOCKING=FALSE`, a real `DDP_TIMEOUT` wired into
   `init_process_group`, and a `set +e` fix for a `pipefail` bug that was silently preventing
   the resubmit loop from ever firing after a crash.
3. **Data migration**: moved the full 7-dataset pretrain mixture (source data + merged
   teacher-logit companions) from CFS to `$SCRATCH`, removing the Lustre file-locking
   bottleneck. Also fixed a startup perf bug (only rank 0 now runs index validation).
4. **Recurring NCCL BROADCAST hang** (`SeqNum=7929`, always ~34–38 min into a session):
   root-caused to `wandb`'s console-capture pipe deadlocking after its objects got forked
   into DataLoader worker processes. **Fixed** by forcing DataLoader workers to fork before
   `wandb.init()` is called. Validated: a 40-epoch run completed cleanly end-to-end with zero
   hangs post-fix (2026-08-09).

#### Scale-up and the completed runs (Aug 9–16)

Scaled to the real target, ultimately run at **52 GPUs / 13 nodes, `-q regular`,
`--epoch 154`** (≈ one full pass over the ~1.06B-sample pretrain mixture at that GPU count).

- **`distill_pretrain_s_scratch_a05_b05_T4_full500_reg52`** (mixed α=0.5/β=0.5): job
  56540549, **COMPLETE 2026-08-10 19:44**, clean exit, all 154/154 epochs, monotonic loss
  decrease throughout.
- **`distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2`** (pure-KD α=0/β=1): first
  attempt crashed 3× under a stale conda-env checkout; re-run as `_v2`, **COMPLETE
  2026-08-16 13:48**, job 56998456, all 154/154 epochs. Surfaced a real anomaly: best val
  loss only at epoch 69/154 (24.58), then climbed back to 25.37 by epoch 154 — overfitting
  in the second half, generative-loss component ~4× higher than the a05_b05 run's.
  (Downstream fine-tune correctly loads the pre-degradation checkpoint, not the overfit one.)

#### Final top-tagging fine-tune comparison — COMPLETE 2026-08-17

404,000 test events:

| Model | Acc | AUC | 1/FPR@50% | 1/FPR@30% |
|---|---|---|---|---|
| a00_b10_v2 (pure-KD pretrain → top fine-tune) | 94.17% | 0.9867 | 478.5 | 2148.0 |
| a05_b05 (mixed pretrain → top fine-tune) | 94.17% | 0.9866 | 484.2 | 2148.0 |

**Essentially a tie** between α/β weightings at the pretrain level — unlike Pipeline B
(below), where the weighting choice makes a clear difference.

**Both underperform Pipeline B's direct in-task result:**

| Model | Acc | AUC | 1/FPR@30% |
|---|---|---|---|
| `fine_tune_top_s` (CE-only baseline, no KD) | 94.38% | 0.9875 | 2556 |
| a05_b05 pretrain-distill → fine-tune (Pipeline A) | 94.17% | 0.9866 | 2148 |
| a00_b10_v2 pretrain-distill → fine-tune (Pipeline A) | 94.17% | 0.9867 | 2148 |
| Pipeline B, mixed α=0.5 (fine-tune → distill, in-task) | 94.32% | 0.9875 | 2804 |
| Pipeline B, pure α=0 (fine-tune → distill, in-task) | 94.43% | **0.9879** | **3205** |

Pipeline A (distill-then-fine-tune) categorically underperforms Pipeline B
(fine-tune-then-distill, same task) — the headline comparison between the two pipelines.

#### Loose end

An eval output set exists on disk (`eval/top_distill_pretrain_full500_reg52_matchedhp/`,
checkpoint `..._a05_b05_T4_full500_reg52_matchedhp.pt`) for an apparent "matched
hyperparameters" fine-tune variant. **Not documented anywhere in project memory and no
metrics were ever computed for it.** Worth running `compute_metrics_top.py` on it or
cleaning it up.

### Variant 2: JetClass-level distill → top-tagging fine-tune

**Status: COMPLETE, NEGATIVE result (2026-07-21).**

Student distilled from `best_model_pretrain_l.pt` (210-class teacher, sliced) on JetClass
(10-class, α=0.5/β=0.5/T=4, 100 epochs, completed Jun 24), then fine-tuned on top tagging
(completed Jul 17), evaluated on 404k test events:

| Model | Acc | AUC | 1/FPR@50% | 1/FPR@30% |
|---|---|---|---|---|
| `fine_tune_top_s` (CE-only baseline) | 94.38% | 0.9875 | 577 | 2556 |
| `fine_tune_top_l` (large teacher) | 94.44% | 0.9880 | 645 | 3365 |
| Pipeline B, mixed α=0.5 (direct in-task KD) | 94.32% | 0.9875 | 580 | 2804 |
| **JetClass-distill → top fine-tune (Pipeline A)** | **94.13%** | **0.9865** | **467** | **2019** |

Worst of every config tested — routing distillation through JetClass as an intermediate task
before fine-tuning on top tagging underperforms both CE-only (-21% on 1/FPR@30%) and
in-task Pipeline B KD (-28%).

**Note**: a pure-KD (α=0/β=1) analogue of this JetClass distillation step was scripted
(`distill_train_jetclass_a00.sh` / `distill_loop_jetclass_a00.sh`) but **never launched** —
no checkpoint, training log, or eval output exists on disk, and no job is in `squeue`. If
wanted, it can be launched via `screen -dmS jc_a00 bash distill_loop_jetclass_a00.sh`.

---

## Pipeline B — fine-tune → distill

### Status: tested **only on top tagging**, fully complete there; not yet attempted on any other downstream task (JetClass, DCTR, etc.)

All of these distill from `fine_tune_top_l` / `fine_tune_top_s` — teachers already
fine-tuned on top tagging — directly on the top-tagging task itself (same-task KD):

- **α/β + T sweep — COMPLETE (2026-07-01)**: pure KD (α=0/β=1) wins outright, insensitive to
  T∈[1,16] — AUC 0.9879 (matches the large teacher's 0.9880), 1/FPR@30%=3205 vs the CE-only
  baseline's 2556 (+25%). Adding CE weight monotonically hurts (α=0→0.25→0.5 degrades AUC
  0.9879→0.9877→0.9875 and 1/FPR@30% 3205→3205→2804).
- **T<1 extension — COMPLETE (2026-07-14)**: the T≥1 plateau does not extend below T=1;
  clean monotonic collapse (AUC 0.9879→0.9860, 1/FPR@30% 3257→1374 from T=1 to T=0.125). By
  T=0.125, KD underperforms CE-only on both metrics.
- **Micro-student KD — COMPLETE (2026-07-18)**, extended to **n=5 reps — COMPLETE
  (2026-07-21)**: a 4-layer/base_dim=64 micro student matches/slightly beats the small KD
  student despite being much smaller, and micro-KD beats micro-CE-only by +0.22pp acc,
  +0.0013 AUC, and nearly 2× on 1/FPR@30% (3021 vs 1639, +84%) — the strongest KD-compression
  result in the whole project. At micro scale, CE-only training is genuinely
  capacity-limited (worse than even the small CE-only baseline).
- **No-interaction ablation** (drops the pairwise interaction matrix, 5 reps): launched
  2026-07-21, completion status not confirmed in memory — check
  `distill_status_top_micro_noint_reps.sh` if needed for the write-up.

Full reference table (top tagging, 404k test events):

| Model | Acc | AUC | 1/FPR@50% | 1/FPR@30% |
|---|---|---|---|---|
| `fine_tune_top_s` (CE-only, no KD) | 94.38% | 0.9875 | 577 | 2556 |
| `fine_tune_top_l` (large teacher) | 94.44% | 0.9880 | 645 | 3365 |
| Pipeline B, α=0.5/β=0.5/T=4 (small) | 94.32% | 0.9875 | 580 | 2804 |
| Pipeline B, α=0/β=1, T∈[1,16] (small) | 94.43% | **0.9879** | 621 | **3205** |
| Pipeline B, α=0.5/β=0.5/T=4 (micro, n=5) | 94.34% ± 0.01 | 0.9876 ± 0.0001 | 605 ± 7 | 3021 ± 251 |
| Pipeline B, CE-only (micro, n=5, no KD) | 94.12% ± 0.05 | 0.9863 ± 0.0002 | 442 ± 14 | 1639 ± 71 |

**This is the gap the user flagged**: Pipeline B has a complete, positive result on top
tagging, but has never been run on any other downstream task. Repeating it — fine-tune a
teacher on JetClass (or another task), then distill a same-task student from that teacher —
would be the natural next step to see whether Pipeline B's advantage over Pipeline A
generalizes beyond top tagging.

---

## Bottom line for the paper

- **Pipeline A (distill → fine-tune) is done for both tested intermediates** (full pretrain
  mixture, and JetClass). Both underperform Pipeline B, and JetClass-as-intermediate is
  actively worse than not distilling at all.
- **Pipeline B (fine-tune → distill) has a strong, complete positive result — but only on top
  tagging.** Pure-KD in-task distillation (α=0) beats CE-only by +25% on 1/FPR@30% and holds
  up down to micro student scale. This has not been validated on a second task yet.
- **Recommended next step**: run Pipeline B on at least one more downstream task (e.g.
  JetClass, or DCTR) to test whether the fine-tune→distill advantage generalizes, before
  treating the top-tagging result as the general claim.
