# Data

Two corpora, two very different access situations. Read this before opening an issue
about a missing file.

```
data/
├── CIS-EMA/                 restricted — not distributed with this repository
│   ├── features.csv         361,835 frames x 106 features
│   ├── manifest.csv         4,884 utterances
│   ├── meta.json            column-name lists, inventories
│   └── utterances/sub_XX/*.npz
├── raw/CIS-EMA/             restricted — the original recordings
│   └── sub_XX/{posR,wav,TG}/
└── SAIT-EMA/                public — download it yourself (CC BY 4.0)
    ├── Dataset/             build this from the download (below)
    └── <speaker archives>/  as distributed
```

Point the code at any of these with `paths.toml` or an environment variable; nothing
else in the repository contains an absolute path. See `paths.example.toml`.

---

## CIS-EMA — Cantonese Isolated-Syllable EMA corpus

33 healthy adult speakers (17 F / 16 M), 4,884 isolated Cantonese syllables, 43 distinct
syllables, 155 scripted street-name stimuli, six articulatory coils on an AG501 at 250 Hz
with synchronized audio at 48 kHz. Coordinates are already head-corrected (the AG501
`posR` frame).

| | |
|---|---|
| Coils | TD (tongue dorsum), TL (tongue lateral), TT (tongue tip), UL (upper lip), LL (lower lip), LC (lip corner) |
| Axes | x = front–back, z = height, y = left–right |
| Labels | 43 tone-less Jyutping syllables; 15 onsets, 7 vowels, 6 lexical tones derived from them |
| Chunking | one utterance per recording; median ~74 frames at 250 Hz |

**Not publicly released.** The recordings were collected under an institutional ethics
approval (The Hong Kong Polytechnic University HSESC, reference HSEARS20230531001) whose
information sheet restricts access to the researchers involved in the study. Publishing the
corpus requires an approval amendment that has not yet been granted, so the data are not in
this repository and are not downloadable elsewhere.

Consequences, stated plainly:

* every experiment in this repository can be **read** and **run**, given the corpus;
* the paper's CIS-EMA numbers cannot be re-derived without it;
* the method itself **can** be re-derived by anyone, because it also runs on SAIT-EMA
  (public) — see section IV-G of the paper and the SAIT-EMA arm below.

A data descriptor for CIS-EMA is in preparation. If you need access for a specific
scientific purpose, write to the corresponding author of the paper; any future release will
follow the terms the ethics amendment permits, which may separate open articulatory
kinematics from controlled-access audio.

### Files the code expects

`features.csv` — long format, one row per frame, 361,835 rows:

| column group | columns |
|---|---|
| metadata | `source`, `subj`, `sex`, `time`, `syllable`, `vowel`, `onset` |
| articulatory (59) | G1 position (18), G2 velocity (18), G3 acceleration (18), G4 geometry (5) |
| acoustic (47) | G5 voicing (2), G6 formants (3), G7 MFCC + Δ + ΔΔ (39), G8 energy (3) |

`source` is `sub_<id:02d>_<item:04d>`, so the item number — and hence the tone label — is
recoverable from the utterance id. `manifest.csv` carries one row per utterance
(`source, subj, sex, file_num, syllable, vowel, onset, n_frames, duration, path`).

After a valid local setup, this must hold:

```python
import pandas as pd
df = pd.read_csv("data/CIS-EMA/features.csv")
assert len(df) == 361_835
assert df.subj.nunique() == 33
assert df.source.nunique() == 4_884
assert df.syllable.nunique() == 43
```

### Rebuilding it from raw recordings

Only needed if you have the raw corpus. The acoustic stage needs
`requirements-extras.txt` (Praat via parselmouth, librosa).

```bash
python preprocessing/prepare_dataset.py --raw-dir data/raw/CIS-EMA --out-dir data/CIS-EMA
python preprocessing/build_features.py  --out-dir data/CIS-EMA
python scripts/build_zscore_features.py --variant robust_z   # the paper's representation
```

The canonical `features.csv` is the `robust_z` variant: per-speaker median/IQR on the
head-corrected coordinates. The `variants/` tables used for the normalization ablation
(Table II) come from `scripts/build_variant_features.py` and `scripts/build_center_calib.py`.

The tone labels are **not** in the corpus; they are joined from the 155-item stimulus list
(`stimuli.xlsx`, item number = `file_num`). That join, including the one excluded item (114,
ambiguous tonal reading), is documented in the paper's Section III-A and implemented in
`src/data/tone.py`.

---

## SAIT-EMA — Mandarin, public

18 speakers (12 native Mandarin, 6 L2), 6,885 monosyllabic items, 405 distinct tone-less
syllables, AG501 at 250 Hz. Six coils, a different sensor configuration from CIS-EMA: TT,
TB (tongue blade), TD, LI (lower incisor, fixed to the lower teeth), UL, LL — no lip corner,
no tongue lateral.

> Jia, H., Xie, Y., Fang, Q., and Sun, Y. *SAIT-EMA: A Tridimensional Electromagnetic
> Articulography Database for Mandarin with Diverse Language Backgrounds.* Scientific Data
> 13:899, 2026. Data: https://doi.org/10.5281/zenodo.17506802 (CC BY 4.0).

Download the archive from that record and extract it under `data/SAIT-EMA/`, then build the
derived tables this repository reads:

```bash
python -m preprocessing.sait_ema.build_manifest --root data/SAIT-EMA --out data/SAIT-EMA/Dataset
python -m preprocessing.sait_ema.prepare_dataset --root data/SAIT-EMA --out data/SAIT-EMA/Dataset
python -m preprocessing.sait_ema.build_features --out data/SAIT-EMA/Dataset --norm robust_z
```

That produces `Dataset/features_mono.csv`, `Dataset/manifest.csv`, and the per-speaker
arrays. For the normalization ablation in Section IV-G rebuild with `--norm raw` and
`--norm center_only` as well, and drop the medial–lateral channels with the `dropy` variant.

Two data facts that matter when reading the numbers:

* **Only the 12 native speakers are used** in the paper (`--native-only` on the
  classification scripts), because the L2 speakers are analysed separately as a domain shift.
* **405 classes, not 406.** One monosyllabic item is unlabelled for all 18 speakers and was
  previously counted as a spurious class; every 406-class artifact from before that fix is
  superseded and archived under `runs/archive_406class/` in the original project.

Absolute accuracy is **not comparable** between the corpora: 405 classes with one token per
stimulus is a harder task than 43 classes with ~113 tokens each. The paper compares deltas
and orderings only.

---

## What is gated and what is open, and why

| payload | status | reason |
|---|---|---|
| CIS-EMA articulatory kinematics, annotations, folds, code | not published | ethics approval does not cover disclosure |
| CIS-EMA audio and voice-derived acoustic features | not published | voice is not anonymisable; same approval issue |
| SAIT-EMA everything | public, CC BY 4.0 | upstream authors released it |

### What this repository does disclose

`folds/cis_folds.json` lists the 33 subject identifiers used in the paper, their sex, and
their fold membership. These are pseudonymous integers (`1`–`33`); **no mapping from them to
a person exists anywhere in this repository**, and the paper itself reports sex counts and
per-fold results at the same level of detail. No names, audio, trajectories, transcripts of
individual speakers, or any recorded signal is present — the only tracked file types are
`.py`, `.json`, `.md`, `.txt`, `.toml`, `.cff`, and `LICENSE`.

Every figure, table and claim in the paper can therefore be *checked* two ways: against the
reference artifacts in `expected/` (no data needed), and end-to-end against public data for
the whole method (SAIT-EMA).

## Licence

The **code** in this repository is MIT (`LICENSE`). That licence does not extend to any
research data: CIS-EMA is not redistributed here, and SAIT-EMA remains under CC BY 4.0 from
its own Zenodo record. The reference artifacts under `expected/` are result summaries
produced by this code and carry the same MIT terms as the code.
