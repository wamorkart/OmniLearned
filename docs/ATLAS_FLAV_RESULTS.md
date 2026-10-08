# ATLAS flavour tagging (`atlas_flav`, `--mode ftag`) results

Scored with `tools/metrics/compute_metrics_flav.py`, which reproduces the
notebook used for the OmniLearned paper numbers
(https://gist.github.com/ViniciusMikuni/dbcfda9da5f082194bf996298dedff0b):

- jets with pT = exp(cond[:, 0]) > 20 GeV
- D_b = log( p_b / (f_c p_c + f_tau p_tau + (1 - f_c - f_tau) p_light) )
- D_c = log( p_c / (f_b p_b + f_tau p_tau + (1 - f_b - f_tau) p_light) )
- f_c = 0.2, f_b = 0.2, f_tau = 0.01 (the GN2 reference row uses f_b = 0.3)
- log of a non-positive ratio set to 0; rejection read off the signal-vs-single-
  background ROC curve at the first point with tpr > working point

Label map (NOT the GN2 convention): 0 = light, 1 = c, 2 = b, 3 = tau.

Reference rows are copied from the notebook outputs (full test set). Unless
marked otherwise, our rows use the full test split too: 24,967,722 jets after
the pT cut (of 25,637,537).

Outputs and `metrics.txt` for every row:
`/pscratch/sd/t/twamorka/omnilearned/eval/<tag>_full/` (teacher:
`atlas_flav_ftag_m_localint_v7_full/`).

## Teacher: `fine_tune_atlas_flav_m_localint_v7`

### b-tagging, rejection at 70% b-efficiency

| Model                          | light  | c    | tau   |
|--------------------------------|--------|------|-------|
| GN2 (open data)                | 1097.9 | 45.5 | 245.8 |
| PET v2-s (scratch)             | 1512.3 | 55.7 | 409.2 |
| OmniLearned-s                  | 1726.2 | 61.4 | 459.9 |
| PET v2-m (scratch)             | 1772.8 | 63.6 | 494.2 |
| OmniLearned-m                  | 1853.2 | 66.5 | 493.1 |
| ours `localint_v7`             | 1756.0 | 62.5 | 456.9 |
| v7 / OmniLearned-m             | 0.95   | 0.94 | 0.93  |

### c-tagging, rejection at 30% c-efficiency

| Model                          | light | b    | tau  |
|--------------------------------|-------|------|------|
| GN2 (open data, f_b = 0.3)     | 166.0 | 21.1 | 21.4 |
| PET v2-s (scratch)             | 206.1 | 22.4 | 24.2 |
| OmniLearned-s                  | 221.2 | 23.6 | 26.1 |
| PET v2-m (scratch)             | 233.3 | 24.0 | 26.8 |
| OmniLearned-m                  | 235.4 | 24.8 | 28.5 |
| ours `localint_v7`             | 226.9 | 23.8 | 26.3 |
| v7 / OmniLearned-m             | 0.96  | 0.96 | 0.92 |

### Full working-point scan for `localint_v7`

| b-eff | rej light | rej c | rej tau |
|-------|-----------|-------|---------|
| 60%   | 9041.1    | 322.8 | 2639.4  |
| 70%   | 1756.0    | 62.5  | 456.9   |
| 77%   | 485.4     | 21.3  | 125.9   |
| 85%   | 102.8     | 6.9   | 23.9    |

| c-eff | rej light | rej b | rej tau |
|-------|-----------|-------|---------|
| 20%   | 844.5     | 57.1  | 87.2    |
| 30%   | 226.9     | 23.8  | 26.3    |
| 40%   | 77.4      | 12.4  | 11.2    |

Accuracy 79.14%; one-vs-rest AUC light 0.9475 / c 0.8341 / b 0.9664 /
tau 0.9759.

### Earlier fine-tunes (test chunk 0 of 6, 4,161,194 jets)

| Model                          | b@70 light | c    | tau   | c@30 light | b    | tau  |
|--------------------------------|------------|------|-------|------------|------|------|
| ours `localint_v2` (epoch 15)  | 204.6      | 10.4 | 56.6  | 47.9       | 7.3  | 8.5  |
| ours `localint_v3`             | 1190.0     | 35.9 | 184.5 | 142.6      | 19.1 | 23.4 |
| ours `localint_v5`             | 1266.2     | 46.2 | 311.5 | 183.2      | 20.9 | 21.5 |
| ours `localint_v7`             | 1807.7     | 61.2 | 440.9 | 232.8      | 23.9 | 26.6 |

`atlas_flav_ftag_m` (the first fine-tune) collapsed (49% accuracy, no tau
predictions) and is omitted.

## Distilled students

Teacher logits from `localint_v7` (`scripts/save_teacher_logits_atlas_flav.sh`).
All students train in `--mode ftag` on the atlas_flav schedule (bs 512,
2000 it, 30 ep, wd 0.1; lr 5e-5 for PET2-small, 5e-4 for DeepSets), so the
loss is `a * (0.1*CE + track_origin) + b * KD` with a = b = 0.5, T = 4.
"no KD" rows are the same model and schedule trained on labels only.
Parameter counts include the training-only track-origin head.

| Student (config)                                         | params | KD  | acc   | b@70 light | c    | tau   | c@30 light | b    | tau  |
|----------------------------------------------------------|--------|-----|-------|------------|------|-------|------------|------|------|
| v7 teacher                                               |        |     | 79.14 | 1756.0     | 62.5 | 456.9 | 226.9      | 23.8 | 26.3 |
| PET2-small (`atlas_flav_small_a05_T4`, tag `_r2`)        | 2.76M  | yes | 78.96 | 1370.2     | 50.8 | 371.9 | 186.1      | 21.7 | 21.9 |
| PET2-small (`atlas_flav_small_ce`)                       | 2.76M  | no  | 77.61 | 945.9      | 36.8 | 255.8 | 140.2      | 17.8 | 17.7 |
| DeepSets distillnet + GNN k64 (`..._gnn_ftag`)           | 23k    | yes | 77.87 | 626.8      | 20.7 | 157.2 | 72.4       | 14.1 | 10.5 |
| DeepSets distillnet + GNN k64 (`..._gnn_ftag_ce`)        | 23k    | no  | 75.66 | 497.4      | 17.5 | 127.4 | 73.8       | 11.7 | 11.0 |
| DeepSets distillnet (`..._relu_ftag`)                    | 17k    | yes | 77.21 | 431.0      | 12.9 | 86.5  | 41.6       | 11.3 | 7.4  |
| DeepSets distillnet (`..._relu_ftag_ce`)                 | 17k    | no  | 74.75 | 374.6      | 13.3 | 86.1  | 57.1       | 9.9  | 8.4  |

DeepSets configs are `scripts/configs/train/atlas_flav_deepsets_distillnet_*`
(ReLU, padding mask kept, no fixed N); they are scored with `--mode classifier`.

Takeaways:

- PET2-small: KD gives ~1.4x b-tag and ~1.25-1.35x c-tag rejection over no
  KD, and recovers ~90% of the teacher's accuracy gap and ~half of its
  b@70 light-rejection gap (0.78x of the teacher).
- DeepSets + GNN: KD gives +2.2 pt accuracy and ~1.2-1.26x b-tag rejection;
  c-tag light rejection is unchanged (0.98x).
- Plain DeepSets: KD gives +2.5 pt accuracy and ~1.15x b@70 light rejection,
  but c-tag light rejection drops to 0.73x. The GNN block removes that
  regression and adds 1.45x b@70 light rejection over plain DeepSets with KD.

The first PET2-small KD run (tag `distill_atlas_flav_small_a05_T4`: acc
78.65, b@70 1200.2 / 44.5 / 333.4, c@30 164.1 / 19.8 / 20.5) had every loss
rise after its epoch-16 resume. Neither a replay from its epoch-15
checkpoint nor the from-scratch `_r2` rerun reproduced that, so `_r2`
replaces it.

### Full working-point scans (students with KD)

PET2-small `_r2`:

| b-eff | rej light | rej c | rej tau |   | c-eff | rej light | rej b | rej tau |
|-------|-----------|-------|---------|---|-------|-----------|-------|---------|
| 60%   | 6863.3    | 239.0 | 2038.0  |   | 20%   | 669.4     | 51.0  | 69.1    |
| 70%   | 1370.2    | 50.8  | 371.9   |   | 30%   | 186.1     | 21.7  | 21.9    |
| 77%   | 408.1     | 18.4  | 111.2   |   | 40%   | 65.4      | 11.6  | 9.8     |
| 85%   | 91.7      | 6.3   | 22.2    |   |       |           |       |         |

DeepSets distillnet + GNN k64:

| b-eff | rej light | rej c | rej tau |   | c-eff | rej light | rej b | rej tau |
|-------|-----------|-------|---------|---|-------|-----------|-------|---------|
| 60%   | 2234.2    | 62.5  | 537.1   |   | 20%   | 204.1     | 29.5  | 25.9    |
| 70%   | 626.8     | 20.7  | 157.2   |   | 30%   | 72.4      | 14.1  | 10.5    |
| 77%   | 228.4     | 9.9   | 59.8    |   | 40%   | 30.9      | 8.2   | 5.8     |
| 85%   | 61.7      | 4.4   | 15.8    |   |       |           |       |         |
