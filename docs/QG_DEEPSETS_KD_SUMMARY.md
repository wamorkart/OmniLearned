# Quark–gluon tagging: knowledge distillation into Deep Sets — results summary

Prepared 2026-10-04 as source material for a group presentation. Self-contained:
every number below was measured in this repo, and the provenance section at the
end says which config and command produced each one.

---

## 1. The question

Can a large pretrained transformer teacher be distilled into a small, cheap
Deep Sets student without losing quark–gluon tagging performance — and does
distillation actually beat just training the same small student on hard labels?

The motivation is deployment: Deep Sets has no attention, so it is the
architecture that fits on an FPGA for trigger-level inference. The transformer
is the accuracy reference we would like to approach.

## 2. Setup

**Dataset.** `qg` quark–gluon tagging, binary classification. Test split is
200,000 events, identical for every run below.

**Student.** Deep Sets / PFN — per-particle φ MLP, masked mean pool, ρ MLP, no
attention. `SIZE=small` = base_dim 128, 3 φ layers, 2 ρ layers, **0.30 M
parameters**. Random init in every run ("scratch" in the run tags means the
*student* is randomly initialised; it does not mean "no teacher").

**Teacher.** `fine_tune_qg_pretrain_l` — large PET2 transformer, pretrained then
fine-tuned on qg. Large PET2 is 373.7 M parameters, i.e. ~1250× the student (param count
quoted for the `top` teacher of the same `SIZE=large` architecture in
`docs/EXPERIMENTS_deepsets_kd.md`; the qg teacher was not separately counted).
Teacher logits are precomputed once and stored, so the teacher is never run
during student training.

**Recipe** (identical across all runs): `--batch 128 --iterations 1000 --epoch 50
--lr 5e-4 --wd 0.5 --use-pid`, 4 nodes × 4 GPUs on Perlmutter.

**KD loss.** Total loss is

```
loss = alpha * CE(student, hard_labels)
     + beta  * T^2 * KL( softmax(student/T) || softmax(teacher/T) )
```

The `T^2` factor matters for interpreting high T: as T → ∞ the KD term does not
vanish, it approaches logit matching (Hinton's asymptotic limit). Two α/β
settings were run:

| Name | alpha | beta | Meaning |
|---|---|---|---|
| `a05` | 0.5 | 0.5 | Equal mix of hard labels and teacher |
| `a0` | 0.0 | 1.0 | Pure KD, teacher only, no hard labels |

## 3. Results — test split, 200k events

Higher is better for all four metrics. `rej@50` / `rej@30` are 1/FPR at 50% and
30% signal efficiency. Quoted ± are standard deviations over independent
training replicates (different random init and data order: `run_train.sh`
never passes `--seed`, and the CLI default of `-1` leaves runs unseeded, so
replicates differ genuinely).

| Run | Reps | Accuracy | AUC | rej@50 | rej@30 |
|---|---|---|---|---|---|
| **CE only** (no teacher) | 3 | 0.8308 ± 0.0004 | 0.9046 ± 0.0002 | 37.33 ± 0.21 | 95.3 ± 1.9 |
| **KD a05, T=1** | 1 | 0.8326 | 0.9054 | 37.5 | 91.7 |
| **KD a05, T=4** | 3 | 0.8325 ± 0.0013 | 0.9057 ± 0.0007 | 37.24 ± 0.17 | 90.8 ± 4.9 |
| **KD a05, T=16** | 1 | 0.8334 | 0.9064 | 37.1 | 91.6 |
| **KD a05, T=100** | 1 | 0.8334 | **0.9072** | **38.5** | 96.9 |
| **KD a0, T=4** (pure KD) | 1 | 0.8279 | 0.9045 | 36.6 | 92.4 |

### Finding 1 — the hard-label term is doing the work

Pure KD (`a0`, α=0) is the **worst** configuration on accuracy, AUC and rej@50 —
it is the only run that fails to beat the CE-only baseline. Dropping the hard
labels costs ~0.003 in accuracy relative to the α=0.5 mix. Whatever benefit
distillation provides on this task, it comes from *combining* teacher and hard
labels, not from the teacher alone.

### Finding 2 — AUC rises monotonically with temperature

AUC across the sweep: T=1 → 0.9054, T=4 → 0.9057, T=16 → 0.9064, T=100 → 0.9072.
Fitting AUC against log₂(T) gives a slope of +0.00028 per doubling with
R² = 0.98.

**The R² is misleading and should not be presented without the caveat.** It is
computed from the fit residual of 4 points that happen to lie nearly on a line.
Measured against the independently determined per-run scatter (σ = 0.0007, from
the three T=4 replicates), the slope is only **≈ 2σ**. Three of the four points
are single runs. The trend is suggestive, not established.

### Finding 3 — T=100 is the best run, at ~3.7σ over CE

T=100 is the only configuration that beats the CE baseline on *every* metric
simultaneously, including both rejection metrics. AUC gap vs CE is +0.0026,
which against the single-run σ is ≈ 3.7σ.

This was originally run as a **sanity check** — the expectation was that T=100
would be indistinguishable from T=16, because the teacher's soft targets stop
changing once T greatly exceeds the logit scale (see Finding 4). It was not
indistinguishable, which is the most interesting open result in the study.

For context, the significance of each KD point against the CE baseline:

| Config | ΔAUC vs CE | Significance |
|---|---|---|
| T=1 | +0.0008 | 1.1σ |
| T=4 (3 reps) | +0.0011 | 2.6σ (Welch p = 0.10) |
| T=16 | +0.0018 | 2.5σ |
| T=100 | +0.0026 | 3.7σ |

### Finding 4 — the teacher is not very confident, which compresses the sweep

Measured over all 1.6 M training events, the teacher's logit gap
Δz = z₁ − z₀ has:

| Statistic | Value |
|---|---|
| mean \|Δz\| | 0.95 |
| **median \|Δz\|** | **0.25** |
| p99 \|Δz\| | 3.94 |

A median gap of 0.25 means the teacher's median predicted probability is ≈ 0.56 —
on half the events it is nearly 50/50 *before any temperature softening*. Since
the softened teacher is `p₁ = sigmoid(Δz/T)`, the targets are already soft at
T=1, and by T=64 every event satisfies Δz/T ≪ 1:

| T | mean \|p₁ − 0.5\| |
|---|---|
| 4 | 0.057 |
| 16 | 0.015 |
| 64 | 0.0037 |
| 100 | 0.0024 |

This predicts the T≥16 region should be flat, which makes the T=100 result
genuinely puzzling and worth more replicates.

### Finding 5 — accuracy/AUC and rejection metrics disagree below T=100

For T=1, T=4 and T=16, accuracy and AUC favour KD while **both rejection metrics
favour CE**. The T=4 three-rep comparison illustrates this clearly:

| Metric | KD T=4 | CE | Welch p | Favours |
|---|---|---|---|---|
| Accuracy | 0.8325 | 0.8308 | 0.13 | KD |
| AUC | 0.9057 | 0.9046 | 0.10 | KD |
| rej@50 | 37.24 | 37.33 | 0.62 | CE |
| rej@30 | 90.8 | 95.3 | 0.25 | CE |

No metric reaches p < 0.05, and metrics pointing in opposite directions is the
signature of no real effect. **The defensible statement for T=4 is that KD and
CE are statistically indistinguishable.** T=100 is the only point where this
disagreement resolves.

## 4. Caveats to state explicitly

- **Three of five KD configurations are single runs.** Only CE and KD T=4 have
  the 3 replicates needed for an error bar. The headline T=100 result rests on
  one training run.
- **rej@30 is the noisiest metric by far.** At rej ≈ 95 only ~1050 background
  events survive the cut, so Poisson counting alone contributes ~3% (≈ ±2.9),
  comparable to the entire observed replicate spread. KD T=4's σ of 4.9 is
  driven almost entirely by one replicate at 96.4 against two at ~88.
- **Quoted error bars capture training variance only.** All runs are scored on
  the same 200k test events, so test-set statistical uncertainty is common-mode
  and absent from the replicate standard deviations. True uncertainty on any
  KD−CE difference is larger than the table implies.
- **The teacher's own test performance has never been measured on this branch.**
  There is no measured ceiling, so "how much of the teacher's performance did
  the student retain" cannot currently be answered.
- **Effect sizes are small in absolute terms.** The best KD−CE AUC gap is
  0.0026, i.e. 0.3%. Worth stating plainly so the audience calibrates.

## 5. Open work

1. **Two more T=100 replicates.** This is the highest-value next step — it takes
   the headline result from 3.7σ on one run to a proper spread, and tests
   whether 0.9072 regresses toward the T=16 value.
2. **T=16 replicates.** One run (`..._T16_r2`, val loss 0.4216) is trained but
   never evaluated; a third (`_r3`) never ran.
3. **Evaluate the teacher** to establish the performance ceiling.
4. **GNN variants** (planned, not yet run): add one EdgeConv / Interaction-
   Network message-passing block to the Deep Sets body via
   `--num-interaction-layers 1 --interaction-k 64`, for both KD (a05/T4) and
   CE-only, to test whether giving the student pairwise context closes more of
   the gap to the transformer than KD does.

## 6. Provenance

| Result | Train config | Eval config | Run tag |
|---|---|---|---|
| CE ×3 | `qg_deepsets_ce` | `qg_deepsets_ce` | `train_qg_deepsets_small_ce_scratch{,_r2,_r3}` |
| KD T=4 ×3 | `qg_deepsets_a05` | `qg_deepsets_a05` | `distill_qg_deepsets_small_scratch_a05_T4{,_r2,_r3}` |
| KD T=1 | `qg_deepsets_a05_T1` | `qg_deepsets_a05_T1` | `..._a05_T1` |
| KD T=16 | `qg_deepsets_a05_T16` | `qg_deepsets_a05_T16` | `..._a05_T16` |
| KD T=100 | `qg_deepsets_a05_T100` | `qg_deepsets_a05_T100` | `..._a05_T100` |
| KD a0 | `qg_deepsets_a0` | `qg_deepsets_a0` | `..._a0_T4` |

Configs live under `scripts/configs/{train,eval}/`. Metrics come from
`tools/metrics/compute_metrics_qg.py --indir <eval dir> --tag <tag> [--tag ...]`,
which prints a mean ± std block when given more than one tag. Teacher logit
statistics were computed directly from
`/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_qg_pretrain_l/qg/train/train_qg.h5`
(a `(1600000, 2)` float16 array; 14 events have `|Δz| > 100`, up to ±223.5,
which is 0.0009% and too rare to affect the loss).

The KD loss is `get_distill_loss` in `src/omnilearned/utils.py`; the α/β
combination is in `src/omnilearned/train.py`. Note that `--interaction` /
`--local-interaction` are **no-ops for `--arch deep-sets`** — they are only
consumed by PET2 — so their presence in the qg configs has no effect.
