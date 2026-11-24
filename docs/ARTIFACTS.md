# Artifacts: which script produces which number

Every claim in the paper, the artifact that holds it, and the value quoted in the
manuscript. `expected/claims.json` is the machine-readable form of this table and
`verify_results.py` checks all of it.

**64 claims.**

## Claims

| claim id | where | artifact | value |
|---|---|---|---|
| `t1_chance` | Table III, row 1 | `cis/results/exp1_ceilings.json` | 2.33 |
| `t1_majority` | Table III, row 2 | `cis/results/exp1_ceilings.json` | 3.37 ± 0.198 |
| `t1_logreg` | Table III, row 3 | `cis/results/exp1_ceilings.json` | 88.93 ± 1.65 |
| `t1_rf` | Table III, row 4 | `cis/results/exp1_ceilings.json` | 87.03 ± 1.22 |
| `t1_ema` | Table III, row 5 | `cis/exp1/ema_only_results.json` | 84.93 ± 2.14 |
| `t1_full` | Table III, row 6 | `cis/exp1/full_results.json` | 92.67 ± 0.73 |
| `t1_hubert` | Table III, row 7 | `cis/exp1_ssl_hubert/summary.json` | 94.13 ± 1.11 |
| `t1_xlsr` | Table III, row 8 | `cis/exp1_ssl_xlsr/summary.json` | 92.8 ± 0.801 |
| `t1_wav2vec` | Table III, row 9 | `cis/exp1_ssl/summary.json` | 85.0 ± 1.98 |
| `t2_raw` | Table IV | `cis/variant_ema_6coil/raw.json` | 73.79 ± 7.39 |
| `t2_pu_z` | Table IV | `cis/variant_ema_6coil/pu_z.json` | 77.75 ± 1.33 |
| `t2_proc` | Table IV | `cis/variant_ema_6coil/proc.json` | 80.35 ± 4.12 |
| `t2_proc_ps_z` | Table IV | `cis/variant_ema_6coil/proc_ps_z.json` | 82.4 ± 3.53 |
| `t2_center` | Table IV | `cis/variant_ema_6coil/center_only.json` | 84.29 ± 2.96 |
| `t2_ps_z` | Table IV | `cis/variant_ema_6coil/ps_z.json` | 84.99 ± 2.89 |
| `t2_robust` | Table IV | `cis/exp1/ema_only_results.json` | 84.93 ± 2.14 |
| `tm_f0` | Table V, row F0 | `cis/exp4/correlation.json` | 0.158 |
| `tm_f2` | Table V, row F2 | `cis/exp4/correlation.json` | 0.576 |
| `tm_f3` | Table V, row F3 | `cis/exp4/correlation.json` | 0.05 |
| `sh_g7` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 33.36 |
| `sh_g6` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 7.81 |
| `sh_g1` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 20.65 |
| `sh_g4` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 0.193 |
| `sh_g5` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 3.29 |
| `pr_vowel_ema` | Section IV-D | `cis/exp3/probing_ema.json` | 97.12 |
| `pr_vowel_full` | Section IV-D | `cis/exp3/probing_full.json` | 97.9 |
| `pr_onset_ema` | Section IV-D | `cis/exp3/probing_ema.json` | 86.38 |
| `pr_onset_full` | Section IV-D | `cis/exp3/probing_full.json` | 93.94 |
| `tone_ema` | Section IV-D | `cis/tone/tone_within_syllable.json` | 77.64 |
| `tone_prior` | Section IV-D | `cis/tone/tone_within_syllable.json` | 65.0 |
| `tone_full` | Section IV-D | `cis/tone/tone_within_syllable.json` | 78.11 |
| `mo_all` | Section V-C | `cis/montage_subset/all.json` | 84.93 ± 2.14 |
| `mo_midsag4` | Section V-C / Fig. 7b | `cis/montage_subset/midsag4.json` | 84.96 |
| `mo_td1` | Section V-C / Fig. 7b | `cis/montage_subset/td1.json` | 61.42 |
| `ar_tcn_ema` | Section IV-A | `cis/arch/tcn_ema.json` | 86.61 |
| `ar_tcn_full` | Section IV-A | `cis/arch/tcn_full.json` | 94.54 |
| `ar_tr_ema` | Section IV-A | `cis/arch/transformer_ema.json` | 85.97 |
| `ar_tr_full` | Section IV-A | `cis/arch/transformer_full.json` | 93.4 |
| `sc_k06` | Section IV-A / Fig. 2 | `cis/scaling/spk_k06.json` | 73.13 |
| `sc_k12` | Section IV-A / Fig. 2 | `cis/scaling/spk_k12.json` | 80.46 |
| `sc_k18` | Section IV-A / Fig. 2 | `cis/scaling/spk_k18.json` | 84.05 |
| `kd_ens` | Section IV-F | `cis/exp6/sweep3_summary.json` | 86.36 |
| `kd_diff` | Section IV-F | `cis/exp6/paired_vs_baseline.json` | 1.43 |
| `cal_center_k1` | Section IV-B / Fig. 3 | `cis/fair_deploy_center_only/fair_deploy.json` | 56.93 |
| `cal_center_k5` | Section IV-B / Fig. 3 | `cis/fair_deploy_center_only/fair_deploy.json` | 73.24 |
| `cal_center_k10` | Section IV-B / Fig. 3 | `cis/fair_deploy_center_only/fair_deploy.json` | 75.78 |
| `sait_raw` | Section IV-H | `sait/sait_norm_raw_2seed.json` | 23.93 ± 12.28 |
| `sait_center` | Section IV-H | `sait/sait_norm_center_only_2seed.json` | 44.44 ± 5.16 |
| `sait_robust` | Section IV-H | `sait/sait_norm_robust_z_2seed.json` | 39.31 ± 8.17 |
| `sait_center_dropy` | Section IV-H | `sait/sait_syllable_native_2seed.json` | 47.36 ± 7.41 |
| `sait_full` | Section IV-H | `sait/sait_full_2seed.json` | 74.83 ± 12.47 |
| `sait_sh_g7` | Section IV-H | `sait/shapley_full.json` | 38.19 |
| `sait_dom_n2l` | Section IV-H | `sait/domain_native_l2.json` | 42.47 |
| `sait_dom_l2n` | Section IV-H | `sait/domain_native_l2.json` | 34.19 |
| `sait_cal_k1` | Section IV-H | `sait/calib_fair.json` | 23.76 |
| `sait_cal_k50` | Section IV-H | `sait/calib_fair.json` | 46.18 |
| `tone_oracle_ema` | Section IV-D | `cis/f0_baseline/tone_syllable_oracle.json` | 77.91 |
| `tone_oracle_full` | Section IV-D | `cis/f0_baseline/tone_syllable_oracle.json` | 78.02 |
| `f0_psz_nosyl` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 65.91 |
| `f0_psz_prior` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 65.03 |
| `f0_psz_syl` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 82.39 |
| `f0_mean_only` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 81.62 |
| `mo_midsag3` | Section V-C | `cis/montage_subset/midsag3.json` | 81.18 |
| `y_noy` | Section IV-B | `cis/y_analysis/robust_z_noy.json` | 86.71 ± 2.37 |

### The `y_analysis` artifact (Section IV-B)

`robust_z_noy.json` — the EMA-only model retrained on **41 channels** with the medial–lateral
($y$) position, velocity and acceleration channels of all six coils removed (the five geometry
terms and every $x/z$ channel are kept). It is produced by `scripts/run_y_analysis.py`, which
replaces an earlier hand-run whose values could not be checked:

```bash
python scripts/run_y_analysis.py          # 6 folds x 2 seeds
```

It writes `runs/y_analysis/robust_z_noy.json` and mirrors it to
`expected/cis/y_analysis/robust_z_noy.json`.

The paper quotes the *paired* difference between that run and the six-coil model,
**+1.78 points, t = 3.09, p = 0.027, d = 1.26**. `verify_results.py` cannot express a
cross-artifact difference, so this one is documented rather than ledgered. To recompute:

```python
import json, statistics as st
from scipy import stats
base = json.load(open("expected/cis/exp1/ema_only_results.json"))
noy  = json.load(open("expected/cis/y_analysis/robust_z_noy.json"))["runs"]
fold = lambda rs: {f: st.mean(r["test_acc"] * 100 for r in rs if r["fold"] == f)
                   for f in range(6)}
a, b = fold(base), fold(noy)
print(sum(b[f] - a[f] for f in range(6)) / 6,
      stats.ttest_rel([b[f] for f in range(6)], [a[f] for f in range(6)]))
```

`y_recoverability.json` is still shipped but **no longer supports a manuscript claim**: the
sentence it backed now cites Table V's best-correlate column instead. It has no producer script
in this repository and should be treated as a historical note.

## Figures

**Shared style.** Every generated figure is drawn through `src/figstyle.py`, which sets one
typographic and colour scheme: a Times stack with STIX maths so figure labels sit in the same
face as the IEEEtran body text, hairline open frames, and a palette where navy is articulatory
data and amber is the acoustic reference. Import it and call `apply()` at the top of a
generator; `figstyle.chip()` gives the translucent rounded plate used behind a value that
overlaps data. Figures 1 and 6 are hand-assembled and are not covered.


| figure | what it shows | regenerate with | inputs |
|---|---|---|---|
| Fig. 1 | montage panels: six coils, /daai/ sequence | `python -m src.paper_figures --only fig_montage_daai_sequence.png` (needs the `scaled6` frame — see REPRODUCING) | — |
| Fig. 2 | training-set-size curve: every fold x seed run, mean +/- SD, step gains | `python scripts/make_scaling_figure.py [--out DIR]` | `runs/scaling/spk_k*.json` (+ `_s1` seeds) |
| Fig. 3 | deployment calibration k-curve | `python scripts/make_taslp_figures.py` | `runs/fair_deploy_*/fair_deploy.json` |
| Fig. 4 | exact group Shapley | `python scripts/run_exp2.py` -> `python scripts/make_taslp_figures.py` | `runs/paper/exp2/shapley.json` |
| Fig. 5 | mechanistic taxonomy | `python -m src.paper_figures` | `paper/results/exp4_correlation.json` |
| Fig. 6 | cross-corpus coil importance | no script: assembled by hand from the retrained leave-one-coil-out numbers (`make_taslp_figures.py` emits the group-Shapley comparison `fig_crosscorpus.png`, which is not in the submission) | `runs/montage_subset/drop_*`, SAIT `sait_loo_*` |
| Fig. 7 | which coils are needed | `python scripts/sensor_shapley.py`, `run_montage_subset.py`, `make_montage_figure.py` | `runs/paper/exp8_robust_z/`, `runs/montage_subset/` |
| Table I | performance ceilings | `python scripts/run_exp1.py --seeds 0,1` | `runs/paper/exp1/`, `exp1_ssl_*/`, `paper/results/exp1_ceilings.json` |
| Table IV | normalization ablation | `python scripts/build_zscore_features.py --mode {raw,pu_z,proc,proc_ps_z,center_only,ps_z}`, then train each variant table (`scripts/run_exp1.py --data-dir <variant dir>`, with `manifest.csv` copied beside its `features.csv`; the thin wrapper used for the paper is not shipped) | `runs/paper/variant_ema_6coil/` |
| Table III | mechanistic correlation | `python scripts/run_exp4.py` | `runs/paper/exp4/correlation.json` |
| Table IV | linear vs nonlinear recoverability | `python scripts/nonlinear_recoverability.py` | `runs/paper/exp4/nonlinear_recoverability.json` |
| Table VI | cross-corpus comparison | CIS: above + SAIT arm | `runs/paper/*` + SAIT `Dataset/runs/*` |
| Table VII | place-of-articulation errors | `python scripts/grouped_confusion.py` | `paper/results/confusion_grouped.json` |
| README hero | the montage and trajectories of one /daai/ utterance | `python -m src.paper_figures --only fig_montage_daai_sequence.png` (needs the `scaled6` frame) | recordings |
| README gallery | mean trajectories by place of articulation | `python -m src.paper_figures --only fig_trajectories.png` (needs the `scaled6` frame) | recordings |

## Regenerating without retraining

`expected/cis/` mirrors the `runs/` layout of the original project, so the figure builders can
redraw every manuscript figure from the shipped artifacts alone:

```bash
mkdir -p runs/paper paper/results
cp -r expected/cis/exp* expected/cis/tone expected/cis/variant_ema_6coil runs/paper/
cp -r expected/cis/arch expected/cis/montage_subset expected/cis/scaling \
      expected/cis/f0_baseline expected/cis/fair_deploy_* runs/
cp expected/cis/results/*.json paper/results/
python -m src.paper_figures --out paper/figures
python scripts/make_scaling_figure.py && python scripts/make_montage_figure.py
python scripts/make_taslp_figures.py
```

`expected/cis/exp4/correlation.json` and `expected/cis/results/exp4_correlation.json` are the same
file at two paths, on purpose: `verify_results.py` reads the run-mirror path and the Fig. 5
builder reads the `paper/results/` path. Keep both.

All eight generated figures match the manuscript's files at pixel-dimension level
(two differ by a few pixels of text layout from a matplotlib version change).
