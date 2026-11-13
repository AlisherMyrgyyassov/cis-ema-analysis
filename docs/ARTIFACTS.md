# Artifacts: which script produces which number

Every claim in the paper, the artifact that holds it, and the value quoted in the
manuscript. `expected/claims.json` is the machine-readable form of this table and
`verify_results.py` checks all of it.

**63 claims.**

## Claims

| claim id | where | artifact | value |
|---|---|---|---|
| `t1_chance` | Table III, row 1 | `cis/results/exp1_ceilings.json` | 2.33 |
| `t1_majority` | Table III, row 2 | `cis/results/exp1_ceilings.json` | 3.33 ± 0.13 |
| `t1_logreg` | Table III, row 3 | `cis/results/exp1_ceilings.json` | 84.04 ± 3.25 |
| `t1_rf` | Table III, row 4 | `cis/results/exp1_ceilings.json` | 82.77 ± 4.45 |
| `t1_ema` | Table III, row 5 | `cis/exp1/ema_only_results.json` | 78.71 ± 4.41 |
| `t1_full` | Table III, row 6 | `cis/exp1/full_results.json` | 88.28 ± 3.36 |
| `t1_hubert` | Table III, row 7 | `cis/exp1_ssl_hubert/summary.json` | 94.13 ± 1.11 |
| `t1_xlsr` | Table III, row 8 | `cis/exp1_ssl_xlsr/summary.json` | 92.8 ± 0.8 |
| `t1_wav2vec` | Table III, row 9 | `cis/exp1_ssl/summary.json` | 85 ± 1.98 |
| `t2_raw` | Table IV | `cis/variant_ema_6coil/raw.json` | 68.92 ± 8.82 |
| `t2_pu_z` | Table IV | `cis/variant_ema_6coil/pu_z.json` | 72.01 ± 3.01 |
| `t2_proc` | Table IV | `cis/variant_ema_6coil/proc.json` | 73.93 ± 5.97 |
| `t2_proc_ps_z` | Table IV | `cis/variant_ema_6coil/proc_ps_z.json` | 76.02 ± 5.43 |
| `t2_center` | Table IV | `cis/variant_ema_6coil/center_only.json` | 77.71 ± 4.24 |
| `t2_ps_z` | Table IV | `cis/variant_ema_6coil/ps_z.json` | 78.39 ± 4.08 |
| `t2_robust` | Table IV | `cis/exp1/ema_only_results.json` | 78.71 ± 4.41 |
| `tm_f0` | Table V, row F0 | `cis/exp4/correlation.json` | 0.14 |
| `tm_f2` | Table V, row F2 | `cis/exp4/correlation.json` | 0.54 |
| `tm_f3` | Table V, row F3 | `cis/exp4/correlation.json` | 0.04 |
| `sh_g7` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 49.4 |
| `sh_g6` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 12.7 |
| `sh_g1` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 6.5 |
| `sh_g4` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | -0.2 |
| `sh_g5` | Section IV-C / Fig. 4 | `cis/exp2/shapley.json` | 3.2 |
| `pr_vowel_ema` | Section IV-D | `cis/exp3/probing_ema.json` | 90.8 |
| `pr_vowel_full` | Section IV-D | `cis/exp3/probing_full.json` | 95.9 |
| `pr_onset_ema` | Section IV-D | `cis/exp3/probing_ema.json` | 79.7 |
| `pr_onset_full` | Section IV-D | `cis/exp3/probing_full.json` | 90.1 |
| `tone_ema` | Section IV-D | `cis/tone/tone_within_syllable.json` | 73.03 |
| `tone_prior` | Section IV-D | `cis/tone/tone_within_syllable.json` | 64.83 |
| `tone_full` | Section IV-D | `cis/tone/tone_within_syllable.json` | 73.93 |
| `mo_all` | Section V-C | `cis/montage_subset/all.json` | 78.71 ± 4.41 |
| `mo_midsag4` | Section V-C / Fig. 7b | `cis/montage_subset/midsag4.json` | 78.37 |
| `mo_td1` | Section V-C / Fig. 7b | `cis/montage_subset/td1.json` | 55.68 |
| `ar_tcn_ema` | Section IV-A | `cis/arch/tcn_ema.json` | 79.02 |
| `ar_tcn_full` | Section IV-A | `cis/arch/tcn_full.json` | 90.61 |
| `ar_tr_ema` | Section IV-A | `cis/arch/transformer_ema.json` | 79.09 |
| `ar_tr_full` | Section IV-A | `cis/arch/transformer_full.json` | 89.47 |
| `sc_k06` | Section IV-A / Fig. 2 | `cis/scaling/spk_k06.json` | 61.02 |
| `sc_k12` | Section IV-A / Fig. 2 | `cis/scaling/spk_k12.json` | 72.29 |
| `sc_k18` | Section IV-A / Fig. 2 | `cis/scaling/spk_k18.json` | 76.03 |
| `kd_ens` | Section IV-F | `cis/exp6/sweep3_summary.json` | 80.26 |
| `kd_single` | Section IV-F | `cis/exp6/sweep3_summary.json` | 80.14 |
| `kd_diff` | Section IV-F | `cis/exp6/paired_vs_baseline.json` | 1.55 |
| `cal_center_k1` | Section IV-B / Fig. 3 | `cis/fair_deploy_center_only/fair_deploy.json` | 57 |
| `cal_center_k5` | Section IV-B / Fig. 3 | `cis/fair_deploy_center_only/fair_deploy.json` | 72.5 |
| `cal_center_k10` | Section IV-B / Fig. 3 | `cis/fair_deploy_center_only/fair_deploy.json` | 75.9 |
| `sait_raw` | Section IV-H | `sait/sait_norm_raw_2seed.json` | 23.93 ± 12.28 |
| `sait_center` | Section IV-H | `sait/sait_norm_center_only_2seed.json` | 44.44 ± 5.16 |
| `sait_robust` | Section IV-H | `sait/sait_norm_robust_z_2seed.json` | 39.31 ± 8.17 |
| `sait_center_dropy` | Section IV-H | `sait/sait_syllable_native_2seed.json` | 47.36 ± 7.41 |
| `sait_full` | Section IV-H | `sait/sait_full_2seed.json` | 74.83 ± 12.47 |
| `sait_sh_g7` | Section IV-H | `sait/shapley_full.json` | 38.2 |
| `sait_dom_n2l` | Section IV-H | `sait/domain_native_l2.json` | 42.47 |
| `sait_dom_l2n` | Section IV-H | `sait/domain_native_l2.json` | 34.19 |
| `sait_cal_k1` | Section IV-H | `sait/calib_fair.json` | 23.76 |
| `sait_cal_k50` | Section IV-H | `sait/calib_fair.json` | 46.18 |
| `tone_oracle_ema` | Section IV-D | `cis/f0_baseline/tone_syllable_oracle.json` | 73.94 |
| `tone_oracle_full` | Section IV-D | `cis/f0_baseline/tone_syllable_oracle.json` | 73.97 |
| `y_rec_f3` | Section IV-B | `cis/y_analysis/y_recoverability.json` | 0.005 |
| `f0_psz_nosyl` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 65.69 |
| `f0_psz_prior` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 64.98 |
| `f0_psz_syl` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 82.11 |
| `f0_mean_only` | Section IV-D | `cis/f0_baseline/f0_tone_baseline.json` | 81.25 |

### The two `y_analysis` artifacts (Section IV-B)

Both were recovered from the source project's `runs/paper/exp9_calib/`, produced by
`scripts/y_recoverability.py` and by the y-dropped classification run. They are shipped as
summaries because neither needs the corpus to be read:

- `y_recoverability.json` — ridge $R^2$ for F0/F1/F2/F3 recoved from the $y$ channels alone,
  from $x/z$ alone, and from all channels. The paper quotes **F3 = 0.00454** as "≈0.005"
  (claim `y_rec_f3`). The comparison is the useful part: F3 is 0.0045 from $y$ alone,
  0.0347 from $x/z$, and 0.0354 from all 59 channels — the medial–lateral axis carries
  essentially none of it.
- `robust_z_noy.json` — the EMA-only model retrained on 41 channels with the
  medial–lateral channels removed: **80.20 ± 3.87** over 12 runs.

The paper quotes the *paired* difference between that run and the six-coil model,
**+1.49 points, t = 3.23, p = 0.0232, d = 1.32**. `verify_results.py` cannot express a
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
