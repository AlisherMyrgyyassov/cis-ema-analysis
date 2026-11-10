# Reproducing the paper

Everything here is a shell command you can paste. Paths assume you are in the repository
root and that `paths.toml` points at the corpora (`docs/DATA.md`).

## 0. Before anything: is the checkout consistent?

```bash
python verify_results.py            # 63 claims, no corpus needed
```

This reads only `expected/`. If it does not print `63/63 checks passed`, the repository
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
LogReg (pooled)  84.03 +- 3.23      # paper: 84.04 +- 3.25
RF (pooled)      82.78 +- 4.49      # paper: 82.77 +- 4.45
```

If those two differ by more than ~0.1 points, your `features.csv` is not the paper's.

## 2. Full run order

Each step writes into `runs/paper/<exp>/`; later steps read earlier ones, so keep the order.

```bash
python scripts/run_exp1.py --seeds 0,1              # Table I          -> exp1/
python scripts/run_exp2.py                          # Fig. 4           <- exp1/
python scripts/run_exp3.py                          # Fig. 5, §IV-D    <- exp1/
python scripts/run_exp4.py                          # Tables III-IV    <- exp1/
python scripts/run_exp5.py --lmbdas 0.3,1.0         # §IV-E aux loss
python scripts/run_exp5_oracle.py                    # §IV-E oracle
python scripts/run_exp6_sweep3.py                    # §IV-F KD
python scripts/within_syllable_tone.py               # §IV-D tone       <- exp1/
python scripts/f0_tone_baseline.py                   # §IV-D F0 ceiling
python scripts/sensor_shapley.py                     # Fig. 7a          <- exp1/
python scripts/run_montage_subset.py --sets midsag4,midsag3,min2,td1   # Fig. 7b
python scripts/run_arch.py                           # §IV-A arch
python scripts/run_scaling.py                        # Fig. 2
python scripts/run_calib_sweep.py                    # Fig. 3
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
python -m src.paper_figures --out paper/figures       # Figs. 3, 4, 5 (+ montage panels)
python scripts/make_scaling_figure.py                 # Fig. 2
python scripts/make_montage_figure.py                 # Fig. 7
python scripts/make_taslp_figures.py                  # Figs. 3, 4, 6 as submitted
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
python scripts/sait_probe.py        # onset / final / tone selectivity
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

Measured by re-running Table I here, 12 fold-seeds per model:

| | paper | reproduced | Δ mean | Δ std |
|---|---|---|---|---|
| LogReg, 424 pooled stats | 84.04 ± 3.25 | **84.03 ± 3.23** | −0.01 | −0.02 |
| Random forest, 424 pooled stats | 82.77 ± 4.45 | **82.78 ± 4.49** | +0.01 | +0.04 |
| BiLSTM EMA-only | 78.71 ± 4.41 | **78.22 ± 3.72** | −0.49 | −0.69 |
| BiLSTM full | 88.28 ± 3.36 | **88.16 ± 3.34** | −0.12 | −0.02 |

Two things this tells you:

* **The non-learned baselines reproduce to 0.01**, which is the check that matters for your
  setup: `features.csv`, the folds and the label maps are the paper's.
* **The neural rows scatter around the published mean** with deltas of both signs
  (+2.16, −0.65, +1.62, −2.05 …), largest single fold-seed |Δ| ≈ 4 points, mean −0.49. That is
  training noise on a different stack, not a shifted pipeline. The EMA-only *dispersion* is
  also smaller here (3.72 vs 4.41), so the same tolerance cannot apply to means and standard
  deviations — check means at ±0.5 and treat std as indicative.

```bash
python verify_results.py --runs runs/verify_exp1 --tol 0.5
```

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
