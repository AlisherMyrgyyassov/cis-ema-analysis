# Reproducing the paper

Everything here is a shell command you can paste. Paths assume you are in the repository
root and that `paths.toml` points at the corpora (`docs/DATA.md`).

## 0. Before anything: is the checkout consistent?

```bash
python verify_results.py            # 64 claims, no corpus needed
```

This reads only `expected/`. If it does not print `64/64 checks passed`, the repository
itself is damaged — do not debug your data setup yet.

```bash
python verify_results.py --paths    # what each path key resolves to
```

## 1. Smoke test (~1 minute)

One fold, one seed, one epoch. Proves torch, the loader, the folds and the writer all work
in your environment before you spend GPU hours.

```bash
python scripts/run_exp1.py --fold 0 --seeds 0 --epochs 1 --out-dir runs/smoke
```

Expect a deliberately poor accuracy (one epoch) and, importantly, the pooled baselines,
which do not depend on training at all:

```
LogReg (pooled)  88.93 +- 1.65      # ledger: 88.93 +- 1.65
RF (pooled)      87.03 +- 1.22      # ledger: 87.03 +- 1.22
```

If those two differ by more than ~0.1 points, your `features.csv` is not the paper's.

## 2. Full run order

Each step writes into `runs/paper/<exp>/`; later steps read earlier ones, so keep the order.

```bash
python scripts/run_exp1.py --seeds 0,1              # Table I          -> exp1/
python scripts/run_exp2.py                          # Fig. 4           <- exp1/
python scripts/run_exp3.py                          # §IV-D probing    <- exp1/
python scripts/run_exp4.py                          # Tables III-IV, Fig. 5  <- exp1/
python scripts/run_exp5.py --lmbdas 0.3,1.0         # §IV-E aux loss
python scripts/run_exp5_oracle.py                    # §IV-E oracle
python scripts/run_exp6_sweep3.py                    # §IV-F same-modality KD
python scripts/run_exp6_crossmodal_sweep.py          # §IV-F cross-modal KD
python scripts/exp6_paired_test.py                   # §IV-F KD paired tests
python scripts/exp6_subject_stats.py                 # §IV-F KD subject-level CI
python scripts/within_syllable_tone.py               # §IV-D tone       <- exp1/
python scripts/f0_tone_baseline.py                   # §IV-D F0 ceiling
python scripts/sensor_shapley.py                     # Fig. 7a          <- exp1/
python scripts/run_montage_subset.py --sets midsag4,midsag3,min2,td1   # Fig. 7b
python scripts/run_arch.py                           # §IV-A arch
python scripts/run_scaling.py                        # Fig. 2
python scripts/run_fair_deploy.py --mode center_only --seeds 0,1   # Fig. 3
python scripts/run_fair_deploy.py --mode robust_z --seeds 0,1      # Fig. 3
```

Then regenerate the tables and compare:

```bash
python -m src.analysis_stats        # paired tests  -> paper/results/paired_tests.{json,md}
python -m src.report                # summary table -> paper/results/summary.md
python verify_results.py --runs runs/paper
```

`--runs runs/paper` switches `verify_results.py` from "is the ledger self-consistent" to
"does a fresh run reproduce the paper", matching artifacts by filename.

## 3. Figures

```bash
python -m src.paper_figures --out paper/figures       # Fig. 5 + the Fig. 1 panels
python scripts/make_scaling_figure.py                 # Fig. 2
python scripts/make_montage_figure.py                 # Fig. 7
python scripts/make_taslp_figures.py                  # Figs. 3 and 4 (+ fig_crosscorpus.png, not submitted)
```

**One quirk (Fig. 1).** The corpus figures use the *geometric* frame
(`variants/scaled6`), while `manifest.csv` lives in the main corpus, and the code reads both
from one directory. Give it one:

```bash
python scripts/build_scaled6.py                       # writes variants/scaled6/features.csv
mkdir -p runs/_fig && cp "$(python -c 'from src.paths import get;print(get("cis_dataset"))')/manifest.csv" runs/_fig/
cp "$(python -c 'from src.paths import get;print(get("cis_dataset"))')/variants/scaled6/features.csv" runs/_fig/
CIS_EMA_DATASET="$PWD/runs/_fig" python -m src.paper_figures --out paper/figures \
    --only fig_montage_daai_sequence.png fig_trajectories.png
```

Fig. 1 as printed in the manuscript was additionally composed by hand from those panels;
Fig. 6's coil outlines were drawn by hand too. Nothing in the repository regenerates the
assembled versions.

## 4. The public arm (SAIT-EMA)

No restricted data involved — this is the path a reviewer can follow end to end.

```bash
python -m preprocessing.sait_ema.build_manifest  --root data/SAIT-EMA --out data/SAIT-EMA/Dataset
python -m preprocessing.sait_ema.prepare_dataset --root data/SAIT-EMA --out data/SAIT-EMA/Dataset
python -m preprocessing.sait_ema.build_features  --out data/SAIT-EMA/Dataset --norm robust_z

# normalization ablation (Table VI denominators)
python -m preprocessing.sait_ema.build_features --out data/SAIT-EMA/Dataset --norm raw
python -m preprocessing.sait_ema.build_features --out data/SAIT-EMA/Dataset --norm center_only

python scripts/sait_classify.py --features data/SAIT-EMA/Dataset/features_mono.csv \
    --view ema --task syllable --folds 6 --seeds 2 --native-only --out runs/sait/robust_z.json
python scripts/sait_shapley.py      # group Shapley, rho = 0.74 vs CIS-EMA
python scripts/sait_domain.py       # native -> L2 as a domain shift
python scripts/sait_calib.py        # calibration k-curve, saturates near k = 50
```

Targets: `robust_z` **39.31 ± 8.17**, `center_only` **44.44 ± 5.16**, `raw` **23.93 ± 12.28**,
full 106-channel **74.83 ± 12.47**.

Re-running the first of those here (12 fold-seeds, 405 classes) gave **39.45 ± 7.92** against
the published 39.31 ± 8.17 — including the intrinsic difficulty of fold 2 (24.5 → 23.5) and
fold 5. This is the arm to trust if you want to confirm the pipeline works without any
restricted data:

```bash
python verify_results.py --runs runs/sait --tol 0.5     # 1/1 passed, 56 skipped
```

## What a deviation means

Re-runs are **not bit-identical.** CUDA kernels and multi-threaded BLAS reorder
floating-point reductions, and a different torch release changes the optimisation path
outright. The reference artifacts were produced on a 2025-era stack; this repository was
re-verified on torch 2.11 (2026-11).

The artifacts in `expected/` **are** the CIS-EMA-v2 production run, so there is no
paper-vs-reproduction gap to report for this corpus: the ledger, the shipped artifacts and the
manuscript are three views of one run. What can be checked without retraining is the
non-learned part of Table I, which is deterministic given the feature table:

| | ledger | reproduced here |
|---|---|---|
| LogReg, 424 pooled stats | 88.93 ± 1.65 | **88.93 ± 1.65** |
| Random forest, 424 pooled stats | 87.03 ± 1.22 | **87.03 ± 1.22** |

That is the check that matters for your setup: `features.csv`, the folds and the label maps are
the ones the paper used. The neural rows cannot be reproduced to the decimal on a different
day — CUDA kernels and multi-threaded BLAS reorder floating-point reductions — so treat them
with the tolerance table below; re-evaluating the shipped checkpoints does return 84.93 and
92.67 exactly, because it reuses the stored weights.

```bash
python verify_results.py --runs runs/verify_exp1 --tol 0.5
```

The v2 summary is committed as `expected/verification/table1_reproduced.json`
(per-model mean and std), so you can compare without rerunning anything.

Every figure is passed through `src/figcheck.py` before it is written: it asks the renderer
where each label actually landed and refuses to save a figure in which two of them overlap.
If a layout change introduces a collision you get the offending strings and their overlap in
pixels rather than a silently broken PNG; `FIGCHECK=warn` downgrades it to a warning while you
iterate.

Run against the reproduction above, that reports the two means as passing and the two
EMA-only std values as off; read it that way rather than as a failure.

| comparison | accept | investigate | wrong |
|---|---|---|---|
| pooled LogReg / RF (no training) | ≤ 0.1 | 0.1 – 0.5 | > 0.5 |
| single fold-seed | ≤ 2 points | 2 – 4.5 | > 4.5 |
| 12-run mean (Table I) | ≤ 0.5 | 0.5 – 1.5 | > 1.5 |
| a paired test's sign and p < 0.05 | holds | p within 0.02 of 0.05 | flips |

A deviation larger than the "investigate" band is almost always an input problem, not a
randomness problem — check `features.csv` row count and columns first
(`verify_results.py --paths`, then the four asserts in `docs/DATA.md`).

## Troubleshooting

| symptom | cause |
|---|---|
| `FileNotFoundError: ... cis_dataset ...` | `paths.toml` missing or wrong; run `verify_results.py --paths` |
| pooled baselines off by > 0.5 points | wrong `features.csv` (5-coil table, a truncated `variants/` build, or a different normalization) |
| `no speaker has usable /baa/, /daai/ and /gu/` | Fig. 1 needs the `scaled6` frame — see the quirk above |
| `ModuleNotFoundError: parselmouth / librosa` | only the figure and preprocessing stages need `requirements-extras.txt` |
| every figure script fails with a missing `runs/...` file | figure builders read artifacts, not data; run the experiments first, or populate `runs/paper/` from `expected/cis/` to redraw the paper's figures without retraining |
| a builder rewrote a file when you only wanted `--help` | `build_scaled6.py`, `build_center_calib.py` and `f0_tone_baseline.py` have no argument parser: they run on invocation. Read them before running them |
