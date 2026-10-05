# LHCO Idealized-CWoLa Anomaly Detection on OmniLearned — Work Summary

Status snapshot for group presentation prep. Covers the port of OmniLearn's
idealized-CWoLa weakly-supervised anomaly detection (the "Max SIC vs injected
signal events" scan) onto OmniLearned's unified PyTorch architecture, plus the
debugging history behind each result. Scope deliberately does **not** cover
the separate, unrelated Pipeline A/B distillation-comparison LHCO work in
`anomalydetection/` / `docs/LHCO_ANOMALY_DETECTION_PROCEDURE.md` (different
project, different author, never actually run).

## 1. Goal

Reproduce OmniLearn's reference figure: max significance-improvement
characteristic (SIC = TPR/sqrt(FPR)) as a function of the number of true
signal events injected into a background-only "data" sample, for an idealized
CWoLa setup (background sampled from a known distribution, no generative
model). Two reference curves from the original OmniLearn paper/code:

- **Idealized OmniLearn** (pretrained-then-fine-tuned): max-SIC ≈ 1.46 → 48.5
  across nsig = 500 → 10000, 10 seeds per point, median + 16/84% band.
- **Idealized PET** (trained from scratch, same architecture): stays ≈1.2
  (random) until nsig ≈ 2000, then rises similarly.

## 2. Pipeline built

All in `tools/preprocess/` and `tools/metrics/` of this repo
(`OmniLearned_distillation`), launched via bash scripts in `scripts/lhco/`.

**Preprocessing** (`convert_lhco.py`): merges both jets' particles into one
point cloud per event (with a 2-column one-hot marking jet origin), builds
dataset `lhco_ad`:
- `data.h5` — all of `background_SR` + up to `--nsig` injected signal events,
  `pid=1`.
- `bkg.h5` — the independent `background_SR_extended` sample, `pid=0`, kept
  fully disjoint from `data.h5`.
- Independent 80/10/10 train/val/test split per file.
- `global` (11 features: mjj + per-jet log pT/eta/phi/log mass/mult) is
  z-scored using **background_SR's train-slice only**, applied uniformly to
  every population (signal, background_SR, background_SR_extended) — fixed a
  latent bug where this fit was accidentally coupled to `--nsig` via shared
  RNG consumption order; reordered so it's invariant to `--nsig`.
- Per-particle features kept: `delta_eta, delta_phi, log_pt, log_e` (4 of
  OmniLearn's original 7 — see §5.3).

**Truth-labeled evaluation** (`build_lhco_eval_signal.py`): `data.h5`'s `pid`
can't be used as ground truth for computing a real ROC — it's `1` for both
injected signal *and* background_SR rows. Fix: reconstruct, for a given
`--nsig`, exactly which signal events were *never* injected (by replaying
`convert_lhco.py`'s RNG stream) and build a small pure-signal file from them.
Paired with the already-held-out, already-pure `bkg.h5` test split (reused
from an existing evaluate run rather than re-evaluated), this gives a
genuinely truth-labeled background-vs-signal set — without any retraining.

**Training/eval scripts** (`scripts/lhco/`): `fine_tune_lhco_ad.sh`,
`evaluate_lhco_ad.sh` (real test split, only its `pid==0` rows get used),
`evaluate_lhco_eval_signal.sh` (the pure leftover-signal file).

**Metrics** (`tools/metrics/`):
- `compute_roc_sic_lhco.py` — combines the two evaluate outputs' truth-labeled
  scores into AUC/max-SIC, with an `FPR_FLOOR` (matching OmniLearn's own
  `evaluate_classifiers_lhco.py`) to avoid the SIC ratio being dominated by
  the last few surviving background events. Also reports the actual
  background-event count and Poisson uncertainty at the operating point where
  max-SIC was attained (see §5.2) — not just the ROC-point count, which is a
  much larger and misleading number. Results persist to `lhco_sic_results.json`
  keyed by save-tag, so nothing is pasted-in by hand.
- `plot_lhco_sic_curve.py` — reads that file, plots nsig vs. max-SIC (median +
  16/84% band once ≥2 seeds share an nsig; a single-seed point otherwise, with
  no fake band).

## 3. Results so far (single seed per nsig — see §6 on why that's a caveat)

| nsig | max-SIC (ours) | reference (OmniLearn paper, approx.) |
|---|---|---|
| 500 | ≈1.0 | ≈1.46 |
| 1000 | ≈1.08 | ≈38 (!) — see §5.1, likely seed variance |
| 2000 | ≈15.3 | between the two reference curves' 1-2k rise |
| 10000 | ≈60.0 (±~20% Poisson, see §5.2) | ≈48.5 |

AUC at nsig=10000: 0.9763. At nsig=500: 0.5983 (barely above random).

Shape is qualitatively consistent with the reference — near-random at low
nsig, steep rise, large SIC at high nsig — and the endpoints roughly bracket
the published curve. The nsig=1000 point is the clearest outlier and is
discussed below.

## 4. Debugging history (why this took a while)

- **Loss stuck at ln(2)≈0.693** across many early configurations (the
  classic "model isn't learning anything" signature). Root-caused to a
  mix of: (a) checkpoint-loading shape mismatches during fine-tuning
  (`add_embed`, and — before `--interaction`/`--local-interaction` were
  re-enabled — `local_physics` too) silently reinitializing those layers
  randomly instead of warm-starting them; (b) learning rate too low for
  those freshly-initialized layers even with `--lr-factor`; (c) batch
  size / sequence-length memory limits forcing `--size small`, `--batch 16`.
- **Pure-signal-vs-pure-background sanity check**: built a no-mixing
  diagnostic dataset (100% signal vs. 100% background_SR_extended, no CWoLa
  injection at all). Loss dropped cleanly to 0.106 — confirmed the
  architecture and pipeline *can* learn; the real CWoLa task's difficulty is
  about weak signal fraction / hyperparameters, not a broken model.
- **From-scratch vs. fine-tuned comparisons** at several nsig values: mixed
  results — sometimes scratch plateaus at the same level as fine-tuned,
  raising the open question (currently being tested again at nsig=2000,
  run in progress) of whether the fine-tuning hyperparameters themselves
  need rethinking, independent of the architecture-mismatch issues.
- **Eventually got real, non-plateaued training** at nsig=10000 (lower LR,
  `--interaction --local-interaction` re-enabled, more epochs): loss fell
  steadily from 0.688→0.658 over 20 epochs, still falling at the end.

## 5. Specific findings worth flagging to the group

### 5.1 Single-seed noise is real and large in this regime
The original methodology runs **10 independent seeds per nsig** and reports
the median — our curve currently has exactly 1 seed per nsig. The published
quantile bands at low-to-moderate nsig are enormous (e.g. at nsig=700:
1.99–29.6 across seeds), so a single run landing far from the "expected"
value (as nsig=1000 did) isn't necessarily a bug — it's consistent with this
being a genuinely unstable training regime. Confirmed the nsig=1000 run
really did train for its full epoch budget (checked via wandb across its
three resumed sessions) before concluding this.

### 5.2 The nsig=10000 SIC=60 result is statistically fragile
Only **6 raw background events** survive at the exact FPR where max-SIC was
attained — `fpr_at_max ≈ n_bkg⁻¹ × 6`. Poisson uncertainty on that count is
±41%, which propagates to roughly ±20% on the SIC value itself (since
SIC∝1/√N). That ±20% band comfortably contains the reference paper's 48.5.
On top of that, max-SIC is a *maximum* over a noisy curve — a known upward
bias that a single run doesn't get the benefit of averaging away, unlike the
paper's 10-seed median. Net: the apparent "overshoot" vs. the reference is
plausibly fully explained by statistics, not a real difference in model
quality. `compute_roc_sic_lhco.py` now prints this event count and Poisson
% directly for every run.

### 5.3 We are not feeding the model the same particle features OmniLearn used
Checked directly against both the conversation history and the actual
reference code:

| | per-jet | per-particle | 
|---|---|---|
| OmniLearn (original) | pt, eta, phi, mass, multiplicity | delta_eta, delta_phi, **log(1-pT_rel), log_pt, log(1-E_rel)**, log_e, **deltaR** |
| OmniLearned (ours) | pt, eta, phi, mass, multiplicity | delta_eta, delta_phi, log_pt, log_e |

We drop 3 of the original 7 particle features: `log(1-pT_rel)`,
`log(1-E_rel)`, `deltaR` — exactly the kind of jet-substructure variables a
resonance search leans on. Traced the cause: an early-session simplification
(relayed as "mentor suggested 6 total features per merged particle row")
that doesn't actually match the cited reference script (`torch_lhco.py`),
which keeps all 7 (confirmed via its `mean_part`/`std_part` normalization
arrays each having 7 entries, and its own `make_omnilearned_data()` not
slicing anything down). Likely real driver of at least part of the
performance gap vs. the reference curves — not yet resolved, pending
discussion with the mentor since restoring 4→7 features trades off against
the specific 4-feature pretrained-checkpoint compatibility reasoning below.

That said, the specific **choice of which 4** (and their order) is not
arbitrary — verified directly against `network.py`/`layers.py`: feature
column index 2 is hardcoded in three separate places (pairwise invariant
mass via `get_mass`, energy-weighted pooling, and the padding/validity mask
check) as `log(pT)`. Our column selection `(0,1,3,5)` → `[delta_eta,
delta_phi, log_pt, log_e]` is specifically what slides `log_pt` into that
required slot. `deltaR` is also provably redundant as an *input* — `get_dr`
recomputes it internally from columns 0-1 whenever needed.

### 5.4 A colleague's (Vini's) reported better result at the same nsig values
Investigated a discrepancy where a colleague reported a lower (better)
validation loss using different CLI flags (`--num-feat 6`, no `--use-add`,
no explicit `--path`). Checked the actual pretrained checkpoint's stored
weight shapes directly: his flag choice actually keeps *less* of the
pretrained architecture intact than ours (mismatches both the primary
per-particle embedding and the local-interaction block; ours only mismatches
the smaller add-info pathway) — so it's not an obvious explanation on its
own. Separately, re-examining the actual numbers showed the "gap" was mostly
illusory: our own best epoch (~0.6775) and his reported number (0.6778) are
essentially the same, not meaningfully different.

Also investigated whether he might be running a genuinely different (his
own, non-forked upstream `ViniciusMikuni/OmniLearned`) codebase from the
`wamorkart/OmniLearned` fork this repo is built from — found substantial
real code differences (hundreds of changed lines in `network.py`/`train.py`/
`cli.py`/`dataloader.py`) between the two, but also found his upstream repo
(even after pulling latest) doesn't recognize the `lhco_ad` dataset name his
command used — suggesting he's likely *not* running his own separate repo at
all, probably the same fork from a different checkout. Pending his direct
confirmation.

## 6. Open items / next steps

- Awaiting colleague confirmation on which codebase he's actually running.
- From-scratch vs. fine-tuned comparison at nsig=2000 (same hyperparameters
  otherwise) — run in progress, to check whether fine-tuning hyperparameters
  need revisiting independent of the architecture-mismatch findings.
- Decide whether to restore the 3 dropped particle features (§5.3) — real
  tradeoff against pretrained-checkpoint input-embedding compatibility.
- Need multiple independent seeds per nsig to get a trustworthy median+band
  curve rather than single noisy points (§5.1 especially motivates this).
- Consider enlarging the held-out background test split (currently only 10%
  of ~595k events) to reduce the Poisson-noise problem in §5.2 at any given
  FPR, independent of seed-averaging.
