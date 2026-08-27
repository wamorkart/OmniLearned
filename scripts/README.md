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
