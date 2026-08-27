# scripts/

Job launchers, training/eval loops, queue managers, and one-shot helpers for
running OmniLearned on Perlmutter. Everything that was previously loose in the
repo root now lives here.

## Run them from the repo root

```
bash scripts/<name>.sh
salloc -C gpu -q interactive ... bash scripts/<name>.sh
sbatch scripts/<name>.sbatch
screen -dmS <sess> bash scripts/<name>.sh
```

The scripts assume the current working directory is the **repo root**, not
`scripts/`. In particular many training scripts do `source export_ddp.sh`
from within `srun bash -c "..."`, which resolves against the submission CWD.

## Config-driven training (`run_train.sh` + `configs/train/`)

The `top`-tagging distillation family used to be ~22 near-copy scripts
(`distill_train_top*.sh` + `distill_loop_top*.sh`). It is now:

- `run_train.sh <config> [--dry-run]` — assembles and runs one
  `omnilearned train` invocation. `--dry-run` prints the command and exits.
- `configs/train/_defaults.sh` — every value shared across the family, in one
  place. `configs/train/<config>.sh` — only what that run changes.
- `lib/common.sh` — module loads, NCCL env, the `srun … source export_ddp.sh`
  wrapper (`run_ddp`).
- `lib/resubmit_loop.sh` — the generic salloc/tee/resubmit loop
  (`LOOP_LOG_DIR`, `LOOP_TAG`, `NODES`, `WALLTIME`, `MAX_LOOPS` via env).
- `distill_loop_top.sh [<config>]` — one resubmit-loop driver for every
  top config: picks the per-config log dir + `MAX_LOOPS` from a table
  (override via env), then wraps `resubmit_loop.sh` + `run_train.sh`.

```
scripts/run_train.sh top_micro_a05 --dry-run          # inspect
salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
       --gpus-per-node 4 -A m3246 bash scripts/run_train.sh top_micro_a05
SAVE_TAG=..._r3 scripts/run_train.sh top_micro_a05    # replicate (env override)
screen -dmS distill_top bash scripts/distill_loop_top.sh top_medium_a00_b10
```

The per-config `distill_loop_top_*.sh` shims are gone, folded into
`distill_loop_top.sh <config>`. Still their own scripts: the arg-taking
`distill_loop_top_{rep,micro_rep,micro_noint_rep,sweep}.sh`. `distill_queue_*`
entries now name `distill_loop_top.sh:<config>`. `configs/train/` each note
which old script they replace.

## Config-driven eval (`run_eval.sh` + `configs/eval/`)

Same shape for `omnilearned evaluate`: `run_eval.sh <config> [--dry-run]`
sources `configs/eval/_defaults.sh` + `configs/eval/<config>.sh`, assembles
the CLI, and runs it via `lib/common.sh`'s `run_ddp`. Output lands in
`$EVAL_ROOT/<config>/outputs_<save-tag>_<dataset>_<split>_rank*.npz`.

```
scripts/run_eval.sh top_distill --dry-run
SAVE_TAG=..._r2 scripts/run_eval.sh top_micro_ce      # env override
DATASET_TYPE=val scripts/run_eval.sh top_distill      # pick a split
```

Replaces the old per-checkpoint `evaluate_{top_distill*,jetclass_*,
top_micro_ce}.sh` (13 scripts). Still their own scripts (structurally
different): `evaluate_chunk.sh` / `evaluate_top_large.sh` (sharded/chunked
teacher dumps), `evaluate_metrics_top_{,T_}sweep.sh` (multi-tag metrics
tables), `evaluate_top_distill_deepsets.sh` + `run_eval_deepsets.sh`,
`evaluate_top_for_distill.sh`, `evaluate_train.sh`, `evaluate.sh`.

Not yet migrated: `distill_train_top_deepsets.sh` (already config-driven via
its own `CONFIG` table), `distill_loop_top_T_sweep*.sh`, and the
`fine_tune_*` / `jetclass` / `dctr` training families.

## `export_ddp.sh`

The real file is `../export_ddp.sh` (repo root); `scripts/export_ddp.sh` is a
symlink to it. This keeps both reference styles working:

- bare `source export_ddp.sh` (CWD = repo root) — used inside `srun bash -c`
- `source "$SCRIPT_DIR/export_ddp.sh"` — used by a few generator/watch scripts

## Cross-references

- Loop/queue scripts call their sibling launchers via
  `bash "$SCRIPT_DIR/<sibling>.sh"`, so they resolve regardless of CWD.
- Scripts that call a Python entry point (`compute_metrics_top.py`,
  `compare_micro_ce_vs_kd.py`, …) reference it at the repo root — either by
  bare name after a `cd` to the root, or as `"$SCRIPT_DIR/../<file>.py"`.

## Notable entry points

- `distill_train_top_deepsets.sh` / `distill_loop_top_deepsets.sh` /
  `run_eval_deepsets.sh` — config-driven DeepSets-KD family
  (see `../EXPERIMENTS_deepsets_kd.md`)
- `run_all_chunks.sh` / `submit_datasets.sh` — chunked teacher-logit generation
- `train_omnifold_pythia_herwig.sh` / `distill_loop_omnifold.sh` — OmniFold unfolding
- `qat_train_deepsets_distillnet_8bit.sh` — QAT for the FPGA student
