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

```
scripts/run_train.sh top_micro_a05 --dry-run          # inspect
salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
       --gpus-per-node 4 -A m3246 bash scripts/run_train.sh top_micro_a05
SAVE_TAG=..._r3 scripts/run_train.sh top_micro_a05    # replicate (env override)
```

The `distill_loop_top*.sh` names are kept as one-line shims over
`resubmit_loop.sh` + `run_train.sh` (unchanged CLI, log dirs, screen
workflow), so `distill_queue_*`, `watch_sweep_launch_rep.sh`, and
`distill_resupervise_micro_reps.sh` still work untouched. `configs/train/`
each note which old script they replace.

Not yet migrated: `distill_train_top_deepsets.sh` (already config-driven via
its own `CONFIG` table), `distill_loop_top_T_sweep*.sh`, and the
`evaluate_*` / `fine_tune_*` / `jetclass` / `dctr` families.

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
