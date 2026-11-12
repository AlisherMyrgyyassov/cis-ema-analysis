# Provenance

Where this repository came from, what was changed to make it runnable, and what was
deliberately left out. Written so that a reader — or a reviewer, or you in two years — can
tell the difference between "the paper did this" and "the cleanup did this".

## Source

| | |
|---|---|
| source repository | `ema-syllables` (private working repository) |
| commit | `7d87075048eeb9b0bfa82517502bec9395bd14a8` |
| snapshotted | 2026-11-10 |
| reference artifacts | copied from that commit's `runs/` and `paper/results/` into `expected/` |

## What changed, and nothing else

The guiding rule was **minimise the diff, not maximise the tidiness**. Every refactor is an
opportunity to change a number silently, so the modules and script names are the ones that
produced the paper.

Three things were edited:

1. **Paths.** `src/paths.py` is new and resolves every dataset location from an environment
   variable, then `paths.toml`, then a repository-relative default. `src/config.py` keeps its
   public names (`DATA_DIR`, `RAW_DIR`, `STIMULI_PATH`, `RUNS_DIR`, `RESULTS_DIR`) and now
   derives them from `src/paths.py`. Thirteen scripts and six preprocessing modules that had
   absolute Windows paths now call the same resolver; their argparse signatures are otherwise
   untouched.
2. **Folds.** `folds/cis_folds.json` was exported once via `scripts/export_folds.py` and is
   now loaded by `src/data/cv.py`. The generator is kept as `compute_folds()` so the file can
   be re-derived and audited, but experiments read the frozen file. Reason: the split used to
   depend on `numpy.random.RandomState` behaviour, and a numpy upgrade must not be able to
   reshuffle the test speakers without anyone noticing.
3. **Output locations.** Run artifacts and tables are written under `runs/paper/` and
   `paper/results/` relative to the repository root, as before, but now through the resolver.

4. **One import bug fixed.** `scripts/within_syllable_tone.py` imported `src.*` without the
   `sys.path` bootstrap every other script has, so it failed with `ModuleNotFoundError` when
   invoked as `python scripts/within_syllable_tone.py` — the form the upstream README
   documents. Three lines were added. It changes no numeric result: the script now runs and
   reproduces the paper's tone numbers (below).

5. **Figure labels.** Axis labels were title-cased to the convention used in the manuscript
   ("Tongue Height", "Lip Aperture", "Normalized Time", "Front–Back", "Left–Right"), and
   `src/figcheck.py` was added: every figure is checked for overlapping text before it is
   written, and the save is refused if two labels collide. Running it over the existing
   figures found two real defects: the place-of-articulation labels on the grouped-confusion
   matrix overlapped each other at 45° once the panels were narrowed to a third of the figure
   width (now 90°), and the G5/G8 group labels in the cross-corpus scatter printed into the
   same pixels because the two points nearly coincide (labels now stagger above/below). **The manuscript's copies of that figure and of the
   corpus figures should be regenerated from this code before the next submission.**

No model, no preprocessing step, no metric and no hyperparameter was modified.

## Verification performed when building this repository

| check | result |
|---|---|
| every script imports and exposes `--help` | pass; 6 scripts (`build_scaled6`, `build_center_calib`, `f0_tone_baseline`, `make_taslp_figures`, `make_scaling_figure`, `within_syllable_tone`) have no argument parser and do work on invocation, so they were checked by running them against the reference artifacts instead |
| `python scripts/run_exp1.py --fold 0 --seeds 0 --epochs 1` | pass; pooled baselines reproduced to 0.01 (LogReg 84.03 vs 84.04, RF 82.78 vs 82.77) |
| full Table I re-run (6 folds × 2 seeds, both models) | EMA-only **78.22** vs 78.71 published; full **88.16** vs 88.28; mean Δ −0.49 / −0.12, per-run scatter ±4 with both signs — training noise on a newer torch, not a shifted pipeline (numbers in `REPRODUCING.md`) |
| `python verify_results.py` (ledger vs `expected/`) | 63/63 claims pass |
| figure builders on the shipped artifacts | all eight generated figures match the manuscript's at pixel-dimension level |
| `python -m src.analysis_stats` on the shipped artifacts | reproduces the paper's paired tests exactly (+4.14 p=0.001, +1.47 p=0.011, +0.97 p=0.087, +8.20 p=0.010, +9.10 p=0.001) |
| `scripts/within_syllable_tone.py` after the import fix | runs; reproduces §IV-D tone (73.07 EMA / 73.77 full vs 73.03 / 73.93, probed with sklearn 1.9 instead of the original version) |
| frozen folds vs the subject lists recorded inside the run artifacts | identical for all 6 folds |
| SAIT-EMA 405-way re-run from the public feature table (12 fold-seeds) | **39.45 ± 7.92** vs 39.31 ± 8.17 published; per-fold pattern reproduced, including the intrinsically hard fold 2 |

`verify_results.py --runs <dir>` additionally compares a fresh run against the shipped
artifacts; see `REPRODUCING.md` for the measured deviation.

## Deliberately not included

| excluded | why |
|---|---|
| `legacy/`, `paper/archive_v0/`, `docs/archive/` | superseded 5-coil / V0 pipelines and old drafts; keeping them invites the wrong file being used |
| `runs/` checkpoints, `variants/` feature tables (15 GB) | regenerable; the summaries that matter are the ones in `expected/` |
| ~45 exploratory scripts (`build_anchor_centroid_features`, `rederive_psz`, `y_perclass_compare`, …) | variant exploration and one-off diagnostics, not on any path to a reported number |
| internal documentation (`docs/STATUS.md`, `decisions.md`) | working notes containing superseded numbers; `docs/` here was rewritten for a reader |
| the HuBERT fine-tuning runs | fold-level summaries are shipped; retraining needs the restricted audio and multi-GB weights, and reproduces only the 94.13 ceiling |

Every script that remains maps onto a table, a figure, or a step that produces one, with two
documented exceptions kept for provenance: `src/tone_analysis.py` (the naive tone probe the
paper reports as confounded with syllable identity) and `scripts/export_coil_panels.py` (panel
export for the hand-assembled Fig. 1). If you need a script that is missing, it is in the
source commit above.

## Quirks you will trip over

These are real inconsistencies in the original project. They are documented rather than
"fixed", because fixing them would change the artifacts.

* **`sait_syllable_native_2seed.json` is the `center_only` − y (41-channel) run.** Its `tag`
  says `sait_syllable_native`, its `n_features` is 41, and it holds the 47.36 ± 7.41 figure
  the paper reports for the best EMA-only Mandarin model. The filename predates the naming
  scheme; `verify_results.py` claim `sait_center_dropy` points at it.
* **Scaling curves are split across two files per point.** `spk_k06.json` and
  `spk_k06_s1.json` are seed 0 and seed 1; the paper quotes the two-seed mean, so the claim
  merges both (`"artifacts": [...]` in `expected/claims.json`). The architecture artifacts
  (`arch/*.json`) already contain both seeds.
* **406-class SAIT artifacts are superseded.** One unlabelled item was counted as an extra
  class before the fix; anything tagged 406 is archived upstream and is not referenced here.
* **Fig. 1 (montage) and Fig. 6 (cross-corpus coils) are partly hand-made.** Fig. 1's panels
  are generated by `src/paper_figures.py` from the `scaled6` frame (see `REPRODUCING.md` for
  the one quirk in getting it that frame), but the printed figure was composed by hand from
  them; Fig. 6's coil outlines are hand-drawn schematics. No command regenerates the
  assembled versions.
* **`scripts/build_zscore_features.py` writes its CSV in place.** An interrupted run truncates
  the variant table. Check the row count after any interrupted build.
* **Item 114 of the stimulus list is excluded** from tone analyses (the character has two
  readings across speakers). `EXCLUDED_TONE_ITEMS = (114,)` in `src/config.py`.
* **Tone-5 is rare** (32 of 4,884 utterances). The syllable-controlled probe is reported with
  and without it upstream; only the main variant is shipped here.
* **`src/tone_analysis.py` is the naive 6-way tone probe** that the paper reports as
  *confounded* with syllable identity. It is included for completeness, not because its
  numbers should be read as tone encoding; `scripts/within_syllable_tone.py` is the
  interpretable version.

## If you re-run and the numbers move

Small deviations are expected, not a bug — see the tolerance discussion in
`REPRODUCING.md`. The largest single source is the torch version: the reference artifacts
were produced on a 2025-era stack and this repository was re-verified on torch 2.11. Per-run
accuracy moves by roughly ±1 point, fold means by less, and the 12-run mean by less again.
