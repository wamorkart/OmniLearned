# Project Plan: Knowledge Distillation Ablation Study
**June 23 – July 23, 2026**

## State of Play

| Experiment | Status | Epochs done |
|---|---|---|
| Pretrain distillation (210-class, scratch, α=0.5, T=4) | **Not launched** | 0 / 500 |
| Top KD scratch α=0.5 T=4 (`distill_top_small_scratch_a05_T4`) | Converged | 50 / 50 |
| Top KD scratch (`distill_top_s`) | Converged | 50 / 50 |
| Top KD v2 (`distill_top_s_v2`) | Partial | 20 / 50 |
| Top KD T4-mix (`distill_top_s_T4_mix`) | Partial | 10 / 50 |
| Top KD from-pretrain-s init (`distill_top_s_from_pretrain_s`) | Partial | 10 / 50 |
| CE-only small fine-tune (`fine_tune_top_s`) | Partial | 10 / 50 |
| CE-only large fine-tune (`fine_tune_top_l`) | Partial | 5 / 50 |

**Biggest risk:** pretrain distillation at 500 epochs on the full pretrain dataset may take 10–15+ days of wall clock (16 GPUs, 4h interactive slots, ~100M+ events). Treated as a background job throughout — ablation on direct top-tagging KD runs in parallel while it trains.

---

## Week 1 (Jun 23–29): Launch everything + top ablation foundation

**Tue Jun 23**
- [ ] Start `distill_loop.sh` in a tmux session (4 nodes × 4 GPUs, 4h slots, auto-resubmit). Most time-sensitive action — every day it's not running is a lost day.
- [ ] Resume CE-only baseline: extend `fine_tune_top_s` to 50 epochs (use `distill_loop_top.sh` pattern with `--save-tag fine_tune_top_s`, no `--distill`).
- [ ] Document the exact α/T/init config of each existing top checkpoint (via wandb or training JSON loss curves).

**Wed Jun 24**
- [ ] Check pretrain distillation W&B: `loss_kd` finite and decreasing? If NaN or flat, stop and debug before it wastes more node-hours.
- [ ] Resume `distill_top_s_v2` to 50 epochs (understand what config it was before touching it).
- [ ] **New run:** top KD α=0.3, T=4, scratch (`distill_top_s_a03_T4`).

**Thu Jun 25**
- [ ] **New run:** top KD α=0.7, T=4, scratch (`distill_top_s_a07_T4`).
- [ ] Resume `distill_top_s_T4_mix` to 50 epochs (clarify its config first).

**Fri Jun 26**
- [ ] **New run:** top KD α=0.5, T=2, scratch (`distill_top_s_a05_T2`).
- [ ] **New run:** top KD α=0.5, T=8, scratch (`distill_top_s_a05_T8`).
- [ ] Resume `distill_top_s_from_pretrain_s` to 50 epochs (pretrain_s warmstart, α=0.5, T=4).

**Sat Jun 27**
- [ ] **New run:** top KD α=1.0 (pure KD, no CE), T=4, scratch (`distill_top_s_a10_T4`).
- [ ] Check CE baseline convergence; if done, run `omnilearned evaluate` on `fine_tune_top_s` → save AUC.
- [ ] Run evaluate on the two already-converged top KD checkpoints as a sanity check.

**Sun–Mon Jun 28–29**
- [ ] Keep pretrain distillation sessions going (2 concurrent interactive slots).
- [ ] Collect val losses from all top runs into a table to identify the best configs so far.

---

## Week 2 (Jun 30 – Jul 6): Finish top ablation + evaluate

**Tue Jul 1**
- [ ] All α/T top ablation runs should be at 50 epochs or close. Run `evaluate` on each converged checkpoint → get accuracy/AUC on the test set.
- [ ] Check pretrain distillation: how many epochs in? Update estimated finish date.

**Wed Jul 2**
- [ ] Run full evaluation sweep: for each checkpoint in `{fine_tune_top_l, fine_tune_top_s, distill_top_s_*, distill_top_small_scratch_a05_T4}`, run `omnilearned evaluate` on top test set. Collect all numbers into Table 1.
- [ ] Build initial α-sweep plot (AUC vs α, fixed T=4) and T-sweep plot (AUC vs T, fixed α=0.5).

**Thu Jul 3**
- [ ] Identify winner from α/T sweep. Run that config once more with a different random seed to estimate variance.
- [ ] Check whether `distill_top_s_v2` has a different config from `distill_top_s` — if duplicates, pick one.

**Fri Jul 4**
- [ ] **Decision gate:** Is the best distilled top model better than `fine_tune_top_s` baseline? By how much relative to the teacher large model?
  - If yes: KD pipeline works — proceed to pretrain-initialized fine-tune.
  - If no: investigate (higher α? lower T? more epochs? different LR?). Run one fix iteration.
- [ ] Start drafting Table 1 in final format: baseline (small CE) / distilled-small (best) / teacher large.

**Sat–Mon Jul 5–7** *(Mon Jul 7 = start of week 3)*
- [ ] Continue pretrain distillation sessions.
- [ ] Finalize top-tagging ablation section (α/T/init comparison table complete, plots drafted).
- [ ] Run evaluate on teacher (`fine_tune_top_l`) on top test set to confirm upper-bound numbers.

---

## Week 3 (Jul 7–13): Pretrain-distilled warmstart + synthesis

**Tue Jul 8**
- [ ] **Gate:** Check pretrain distillation status. If converged or plateaued (200+ epochs, loss not improving), treat the current checkpoint as final.
- [ ] If pretrain checkpoint is ready: fine-tune `distill_pretrain_s_scratch_a05_T4.pt` on top tagging with KD from `fine_tune_top_l` teacher (`distill_top_s_from_distill_pretrain` — pretrain-KD warmstart + top-KD fine-tune).
- [ ] If not ready: keep sessions running; spend today on analysis of existing results.

**Wed Jul 9**
- [ ] Run fine-tune sessions (pretrain-KD-init): `distill_loop_top.sh` adapted for `--fine-tune` / `--pretrain-tag distill_pretrain_s_scratch_a05_T4`.
- [ ] Compare val loss curves: pretrain-KD-init vs. pretrain-CE-init vs. scratch.

**Thu Jul 10**
- [ ] Evaluate pretrain-KD-init fine-tune checkpoint on top test set.
- [ ] Build the "init comparison" sub-table: scratch / pretrain-CE-init / pretrain-KD-init, each with their best α/T.
- [ ] Generate ROC curves for the main comparison (baseline, best-KD, teacher).

**Fri Jul 11**
- [ ] Identify any remaining gaps in the ablation. Anything missing from {α, T, init} × {top tagging}.
- [ ] Run any fill-in experiments.

**Sat Jul 12**
- [ ] Full result compilation: every number that goes in the paper/report is on disk.
- [ ] Generate final versions of all plots: loss curves, AUC table, ROC curves.

**Sun–Mon Jul 13–14** *(Mon Jul 14 = start of week 4)*
- [ ] Review consistency of numbers. Check units/conventions (accuracy vs AUC vs 1/εB).
- [ ] Clean up branch: commit all training scripts; add any missing configs to distill_train_top.sh variants.

---

## Week 4 (Jul 14–20): Writeup / report

**Tue Jul 15**
- [ ] Complete "Experiments" section draft: dataset, baselines, distillation setup, ablation results.
- [ ] Complete Table 1 and Table 2 (ablation) in final form.

**Wed Jul 16**
- [ ] Draft "Method" section: offline KD, lazy companion h5, training objective.
- [ ] Figure 1: architecture/pipeline diagram. Figure 2: top-tagging ROC. Figure 3: ablation heatmap or bar chart.

**Thu Jul 17**
- [ ] Draft Introduction and Related Work.
- [ ] Internal consistency check: every claim in text backed by a number in a table.

**Fri Jul 18**
- [ ] Draft Conclusion and Abstract.
- [ ] Full readthrough pass.

**Sat Jul 19**
- [ ] Share with collaborators for feedback.

**Sun–Mon Jul 20–21**
- [ ] Revisions based on feedback.

---

## Buffer (Jul 21–23): Final checks + submission

- [ ] **Tue Jul 22:** Final proofread, figure quality check (font sizes, axis labels).
- [ ] **Wed Jul 23:** Fix any remaining issues, freeze code / tag git commit. Submit / hand off.

---

## Critical Path

The single biggest risk is the **pretrain distillation** taking longer than 2 weeks on the interactive queue. Mitigation:
- Use `sbatch -q regular` for overnight pretrain sessions rather than relying entirely on the 4h interactive loop.
- If pretrain distillation isn't converged by **Jul 7**, the pretrain-KD-warmstart experiment gets cut and the ablation focuses only on direct top-tagging KD (α, T, init from CE pretrain). The paper is still completable — the pretrain-KD warmstart is a nice-to-have, not a must-have.

The **direct top-tagging KD ablation** (α/T/init from CE pretrain) is the core of the study and is fully independent of the pretrain distillation run. That work can and should be finished by end of Week 2 regardless.

---

## Ablation Matrix

| Config | save_tag | α | T | Init | Status |
|---|---|---|---|---|---|
| CE baseline (small) | `fine_tune_top_s` | — | — | scratch | partial (10 ep) |
| CE baseline (large) | `fine_tune_top_l` | — | — | scratch | partial (5 ep) |
| KD α=0.5 T=4 scratch | `distill_top_small_scratch_a05_T4` | 0.5 | 4 | scratch | **done** (50 ep) |
| KD α=0.5 T=4 scratch v2 | `distill_top_s_v2` | 0.5? | 4? | scratch? | partial (20 ep) |
| KD T4-mix | `distill_top_s_T4_mix` | ? | 4 | ? | partial (10 ep) |
| KD from pretrain-s | `distill_top_s_from_pretrain_s` | 0.5 | 4 | pretrain_s | partial (10 ep) |
| KD α=0.3 T=4 scratch | `distill_top_s_a03_T4` | 0.3 | 4 | scratch | **TODO** |
| KD α=0.7 T=4 scratch | `distill_top_s_a07_T4` | 0.7 | 4 | scratch | **TODO** |
| KD α=1.0 T=4 scratch | `distill_top_s_a10_T4` | 1.0 | 4 | scratch | **TODO** |
| KD α=0.5 T=2 scratch | `distill_top_s_a05_T2` | 0.5 | 2 | scratch | **TODO** |
| KD α=0.5 T=8 scratch | `distill_top_s_a05_T8` | 0.5 | 8 | scratch | **TODO** |
| KD from distill-pretrain | `distill_top_s_from_distill_pretrain` | 0.5 | 4 | distill_pretrain_s | **TODO** (gated on pretrain conv.) |
