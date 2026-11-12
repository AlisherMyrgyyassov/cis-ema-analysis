"""Build the public CIS-EMA release payload (the folder that goes to Zenodo).

The release is *derived*, never hand-assembled: every file is produced from the
internal dataset here, so the published copy can be rebuilt and diffed.

What is released (approved scope):

* head-corrected articulator positions in millimetres (G1)
* their velocity and acceleration (G2, G3) and the derived geometry (G4)
* the per-speaker normalised table the paper trains on (``features_robust_z``)
* annotations: syllable, onset, vowel, lexical tone, and the interval times
* the frozen 6-fold subject-disjoint split

What is not: audio, and the voice-derived acoustic channels (F0, formants,
MFCC, energy). Those stay gated behind a data-use agreement.

Usage::

    python scripts/export_release.py --out "C:/path/to/CIS-EMA-1"
    python scripts/export_release.py --out /tmp/probe --limit 3 --skip-npz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import EMA_COLUMNS, FEATURE_GROUPS, EXCLUDED_TONE_ITEMS
from src.data.tone import load_stimuli
from src.paths import get as path, require

import preprocessing.build_features as bf

VERSION = "1.0.0"
EPS = 1e-6
POS_COLS = FEATURE_GROUPS["G1"]["columns"]
VEL_COLS = FEATURE_GROUPS["G2"]["columns"]
ACC_COLS = FEATURE_GROUPS["G3"]["columns"]
GEO_COLS = FEATURE_GROUPS["G4"]["columns"]
META = ["source", "subj", "sex", "time", "syllable", "vowel", "onset"]


# --------------------------------------------------------------------------- #
# TextGrid intervals (same parser and clock rules as prepare_dataset.py)
# --------------------------------------------------------------------------- #
def parse_textgrid(tg_path):
    tiers = {}
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    for block in re.split(r"item \[\d+\]:", text):
        m = re.search(r'name = "([^"]+)"', block)
        if not m:
            continue
        ivs = re.findall(
            r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"',
            block,
        )
        tiers[m.group(1)] = [(float(a), float(b), t) for a, b, t in ivs]
    return tiers


def interval_row(tg_path, offset):
    """Return (syl_start, syl_end, onset_start, onset_end, vowel_start, vowel_end)."""
    tiers = parse_textgrid(tg_path)
    word = None
    for name in ("word", "syllable", "sound", "sonud"):
        if name in tiers:
            word = tiers[name]
            break
    if word is None:
        for name, ivs in tiers.items():
            if name != "segment":
                word = ivs
                break
    if not word:
        return (np.nan,) * 6
    words = [(a, b, t) for a, b, t in word if t.strip()]
    if not words:
        return (np.nan,) * 6
    s, e, _ = words[0]
    segs = [(a, b, t) for a, b, t in tiers.get("segment", [])
            if t.strip() and a >= s - 1e-6 and b <= e + 1e-6]
    o_s, o_e = (segs[0][0], segs[0][1]) if segs else (np.nan, np.nan)
    if len(segs) > 1:
        v_s, v_e = segs[1][0], segs[-1][1]
    else:
        v_s, v_e = np.nan, np.nan
    return (s + offset, e + offset, o_s + offset if segs else np.nan,
            o_e + offset if segs else np.nan,
            v_s + offset if len(segs) > 1 else np.nan,
            v_e + offset if len(segs) > 1 else np.nan)


def textgrid_index(raw_dir):
    """(subject_id, item_number) -> (path, clock offset in seconds)."""
    idx = {}
    for folder in sorted(os.listdir(raw_dir)):
        m = re.match(r"sub_(\d+)_", folder)
        if not m or not os.path.isdir(os.path.join(raw_dir, folder)):
            continue
        sid = int(m.group(1))
        for sub, ext, off in (("TG", ".TextGrid", 0.0),
                              ("TG_edgeTrimmed", ".TextGrid", 0.004)):
            d = os.path.join(raw_dir, folder, sub)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.endswith(ext):
                    continue
                num = os.path.splitext(f)[0]
                if num.isdigit() and (sid, int(num)) not in idx:
                    idx[(sid, int(num))] = (os.path.join(d, f), off)
    return idx


# --------------------------------------------------------------------------- #
# normalisation statistics, exactly as build_zscore_features.py computed them
# --------------------------------------------------------------------------- #
def normalization_stats(manifest, npz_dir):
    """Per-subject median and IQR of the outlier-fixed, *unsmoothed* positions.

    The paper normalises the smoothed trajectory with statistics taken over the
    unsmoothed frames (build_zscore_features.py, lines 86-116). Reproducing that
    order is what makes (raw - median) / IQR reproduce the shipped table.
    """
    per_subj = {}
    for _, r in manifest.iterrows():
        d = np.load(os.path.join(npz_dir, r["path"]))
        e = bf.fix_outliers(d["ema"].astype(np.float64))
        per_subj.setdefault(int(r["subj"]), []).append(e)

    rows = []
    full_index = {c: i for i, c in enumerate(EMA_COLUMNS)}
    for s in sorted(per_subj):
        e = np.concatenate(per_subj[s], 0)
        med = np.median(e, 0)
        q1, q3 = np.percentile(e, [25, 75], axis=0)
        iqr = (q3 - q1) + EPS
        for c in POS_COLS:
            i = full_index[c]
            rows.append({"subj": s, "column": c, "median": med[i], "iqr": iqr[i]})
    return pd.DataFrame(rows)



def readme_rest(**ctx):
    return f"""
## Loading it

```python
import pandas as pd
man = pd.read_csv("manifest.csv")                                  # {ctx['n_utt']} rows
raw = pd.read_csv("kinematics/features_raw.csv.gz")                # {ctx['n_frames']} frames, mm
```

```python
import numpy as np
z = np.load("kinematics/utterances/sub_10/sub_10_0002.npz")
z["time"].shape, z["pos"].shape    # (T,), (T, 18)
```

`manifest.csv` joins everything on `source`. To train a sequence model, group the long
table by `source`, or read the per-utterance `.npz` files directly.

## Coordinate frame and units

Positions are in **millimetres** in the head-corrected frame of the articulograph
(the AG501 `posR` output, already rotated out of the sensor frame):

| axis | meaning |
|---|---|
| x | front–back |
| y | left–right |
| z | height |

Six coils: **TD** tongue dorsum, **TL** tongue lateral, **TT** tongue tip, **UL** upper lip,
**LL** lower lip, **LC** lip corner. Three channels per coil (`TDx`, `TDy`, `TDz`, …),
18 in total.

## The two kinematic tables

`features_raw.csv.gz` holds smoothed positions in millimetres plus their derivatives.
`features_robust_z.csv.gz` holds the same frames after per-speaker normalisation, which is
the table the accompanying paper trains on:

    x_norm = (x - median_speaker) / (IQR_speaker + 1e-6)

`normalization_stats.csv` gives that median and IQR per speaker per position channel, so the
normalised table is reproducible from the raw one. Two details matter:

1. **The statistics are taken over unsmoothed frames and applied to smoothed ones**, exactly
   as in the original pipeline. Recomputing them from `features_raw.csv.gz` therefore agrees
   with `features_robust_z.csv.gz` to about 1e-3, not bit-exactly. Use the shipped statistics
   for an exact match.
2. **In the normalised table, the derived channels are computed after normalisation.** Its
   velocity, acceleration and geometry are not rescaled copies of the raw ones: distances in
   particular are measured between normalised coordinates and are dimensionless. If you want
   physical geometry, use `features_raw.csv.gz`.

## What is not included

* **Audio.** The recordings are not distributed.
* **Voice-derived acoustic measurements** — F0, formants, MFCCs, energy. These are
  voice-derived and can support speaker identification, so they are held back under the
  ethics approval governing the recordings. They are available to named researchers under a
  data-use agreement from the corresponding author; write to the contact on the record.

Consequently the acoustic-model results in the accompanying paper (and its HuBERT ceiling
and cross-modal distillation arms) cannot be reproduced from this record alone. Everything
based on articulation — the EMA-only classifier, normalisation, Shapley attribution,
probing, calibration and sensor removal — can.

## Ethics

Recorded under approval from the Human Subjects Ethics Sub-Committee of The Hong Kong
Polytechnic University (HSESC reference HSEARS20230531001). Participants gave written
informed consent. Subject identifiers are sequential integers (`sub_01`–`sub_{ctx['n_subj']:02d}`)
with sex; no name, date, age or other identifying field is present in this record.

## Licence and citation

Data: **CC BY 4.0** — see `LICENSE`. Please cite the dataset and the paper (see
`CITATION.cff`). If you use these data, cite the record's DOI, which is minted on
publication.

## Integrity

`check_dataset.py` verifies the file inventory, the row and column counts, that the
normalised table reconstructs from the raw table plus the shipped statistics, and that the
geometry columns follow from the normalised positions. Run it first if something looks wrong:

```bash
python check_dataset.py
```

`SHA256SUMS` lists a SHA-256 for every file in the record.
"""


SCHEMA = """# Column schema

## Encoding

`annotations/stimuli.csv` and `annotations/inventory.csv` are **UTF-8 with BOM**, because
they carry Chinese stimulus names and Excel misreads plain UTF-8 as ANSI; `pd.read_csv`
and R's `read.csv` handle the BOM transparently. Everything else is plain UTF-8 (the
kinematic tables are pure ASCII).

## manifest.csv — one row per utterance

| column | type | meaning |
|---|---|---|
| `source` | str | utterance id, `sub_XX_<item:04d>`; the join key for everything else |
| `subj` | int | speaker id, 1–33 |
| `sex` | str | `F` or `M` |
| `item` | int | stimulus item number, 2–156 |
| `syllable` | str | tone-less Jyutping syllable, one of 43 classes |
| `onset` | str | onset consonant |
| `vowel` | str | vowel (7 classes) |
| `tone` | int | lexical tone 1–6, joined from the stimulus list (NaN for item 114, excluded as ambiguous) |
| `n_frames` | int | frames in the aligned 250 Hz window |
| `duration` | float | syllable duration, seconds |
| `t_first`, `t_last` | float | first and last frame time, seconds |
| `t_syl_start`, `t_syl_end` | float | syllable interval from the TextGrid |
| `t_onset_start`, `t_onset_end` | float | onset interval |
| `t_vowel_start`, `t_vowel_end` | float | vowel interval |

Times are in the section clock of the original recording; subtract `t_first` for a
0-based utterance. Interval times come from the Praat TextGrids, converted to the EMA
clock (edge-trimmed files carry a +0.004 s offset).

## kinematics/features_raw.csv.gz — long table, one row per frame

66 columns: the 7 manifest identity columns (`source`, `subj`, `sex`, `time`, `syllable`,
`vowel`, `onset`) plus 59 kinematic channels.

| group | columns | unit | how it was made |
|---|---|---|---|
| **G1 position** (18) | `TDx…LCz` | mm | AG501 head-corrected `posR`, outlier-fixed, Savitzky–Golay smoothed |
| **G2 velocity** (18) | `<coil><axis>_velocity` | mm/s | first derivative of G1, 20 Hz Butterworth low-pass, winsorised at 6σ |
| **G3 acceleration** (18) | `<coil><axis>_acceleration` | mm/s² | second derivative, same filtering and winsorisation |
| **G4 geometry** (5) | `TT_TD_distance`, `UL_LL_distance`, `TT_LL_distance` | mm | Euclidean distance between coils |
| | `TT_TD_LL_angle`, `UL_LL_LC_angle` | degrees | angle at the middle coil, `arccos` of the normalised dot product |

G2–G4 are **derived**, not measured. They are reproducible from G1 with the parameters
above, and a user who wants a different smoothing or derivative scheme should ignore them
and recompute.

## kinematics/features_robust_z.csv.gz

Same 66 columns, same frame order, after per-speaker normalisation:

    x_norm = (x - median_speaker) / (IQR_speaker + 1e-6)

Position channels are normalised directly. Velocity, acceleration and geometry are then
recomputed **from the normalised positions**, so they are in normalised units rather than
mm/s, mm/s², mm and degrees. Note that the geometry here is measured between normalised
coordinates and is dimensionless.

## kinematics/normalization_stats.csv

| column | meaning |
|---|---|
| `subj` | speaker id |
| `column` | position channel name (`TDx`, …) |
| `median` | median of that channel over the speaker's frames, computed on unsmoothed positions |
| `iqr` | interquartile range of the same, plus 1e-6 |

## kinematics/utterances/sub_XX/sub_XX_<item>.npz

| array | shape | unit |
|---|---|---|
| `time` | (T,) | seconds |
| `pos` | (T, 18) | mm, G1 order |
| `vel` | (T, 18) | mm/s |
| `acc` | (T, 18) | mm/s² |
| `geom` | (T, 5) | mm, mm, mm, degrees, degrees |

Arrays are float32 and follow the `features_raw` values exactly.

## folds/cis_folds.json

The subject-disjoint, sex-stratified six-fold split used in the paper: `folds[i].train`,
`.val`, `.test` list subject ids. Reproduce the paper's protocol by training on `.train`,
early-stopping on `.val`, and reporting on `.test`, with two seeds per fold.
"""


LICENSE = """CIS-EMA: Cantonese Isolated-Syllable Electromagnetic Articulography corpus
Copyright (c) 2026 Bruce Xiao Wang, Alisher Myrgyyassov, Lei He, Min Ney Wong

This dataset is licensed under the Creative Commons Attribution 4.0 International
Licence (CC BY 4.0). You are free to share and adapt the material for any purpose,
including commercially, provided you give appropriate credit, provide a link to the
licence, and indicate if changes were made.

Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode

Suggested attribution:

  Wang, B. X., Myrgyyassov, A., He, L., and Wong, M. N. (2026).
  CIS-EMA: Cantonese Isolated-Syllable Electromagnetic Articulography corpus.
  Zenodo. https://doi.org/<DOI>

This licence covers the data files in this record. It does not grant access to the
original audio recordings, which are not distributed, nor to the voice-derived
acoustic measurements, which are available separately under a data-use agreement.
"""


CITATION = """cff-version: 1.2.0
message: "If you use this corpus, please cite it as below."
title: "CIS-EMA: Cantonese Isolated-Syllable Electromagnetic Articulography corpus"
abstract: >-
  Articulatory kinematics (six electromagnetic articulography coils, 250 Hz) and
  phonetic annotation for {n_utt} isolated Cantonese syllables from {n_subj} speakers,
  with a frozen subject-disjoint cross-validation split.
type: dataset
version: 1.0.0
date-released: "2026"
license: CC-BY-4.0
authors:
  - family-names: Wang
    given-names: Bruce Xiao
    affiliation: "Department of English and Communication, The Hong Kong Polytechnic University"
  - family-names: Myrgyyassov
    given-names: Alisher
    affiliation: "Department of Biomedical Engineering and Sports Technology, The Hong Kong Polytechnic University"
  - family-names: He
    given-names: Lei
    affiliation: "Institute of Modern Languages and Linguistics, Fudan University"
  - family-names: Wong
    given-names: Min Ney
    affiliation: "Department of Language Science and Technology, The Hong Kong Polytechnic University"
keywords:
  - electromagnetic articulography
  - speech production
  - Cantonese
  - articulatory kinematics
  - speech recognition
"""


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_csv_gz(df, path_out):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df.to_csv(path_out, index=False, float_format="%.6g", compression="gzip")
    return os.path.getsize(path_out)


def sha256(path_in):
    h = hashlib.sha256()
    with open(path_in, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# documentation written into the record
# --------------------------------------------------------------------------- #
README = """# CIS-EMA: Cantonese Isolated-Syllable Electromagnetic Articulography corpus

Synchronized articulatory kinematics and phonetic annotation for **{n_utt} isolated
Cantonese syllables** produced by **{n_subj} healthy adults** ({n_f} female, {n_m} male),
recorded with an AG501 electromagnetic articulograph (Carstens Medizinelektronik) at **250 Hz**.

This record contains the articulator trajectories and the annotations. It does **not**
contain the audio recordings, and it does not contain the voice-derived acoustic
measurements (F0, formants, MFCCs, energy); see *What is not included* below.

## What is included

```
manifest.csv                        one row per utterance ({n_utt} rows)
annotations/stimuli.csv             the {n_stim} stimulus items, tone-marked
annotations/inventory.csv           the {n_syl} syllables with onset, vowel and tone
kinematics/features_raw.csv.gz      {n_frames} frames x 59 channels, millimetres
kinematics/features_robust_z.csv.gz the same frames, per-speaker normalised
kinematics/normalization_stats.csv  per-speaker median and IQR per position channel
kinematics/utterances/sub_XX/*.npz  per-utterance arrays, one file per utterance
folds/cis_folds.json                the subject-disjoint 6-fold split
check_dataset.py                    verifies this record is complete and consistent
SCHEMA.md                           every column, its unit, and how it was computed
"""

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="release directory to create/populate")
    ap.add_argument("--limit", type=int, default=0, help="rows (debug)")
    ap.add_argument("--skip-npz", action="store_true")
    args = ap.parse_args()

    ds = require("cis_dataset")
    raw_dir = require("cis_raw")
    out = args.out
    os.makedirs(out, exist_ok=True)

    manifest = pd.read_csv(os.path.join(ds, "manifest.csv"))
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    print(f"utterances: {len(manifest)}  subjects: {manifest.subj.nunique()}")

    # ---- two kinematic tables -------------------------------------------
    cols = META + POS_COLS + VEL_COLS + ACC_COLS + GEO_COLS
    raw = pd.read_csv(os.path.join(ds, "variants", "raw", "features.csv"), usecols=cols)
    rob = pd.read_csv(os.path.join(ds, "features.csv"), usecols=cols)
    if args.limit:
        keep = set(manifest.source)
        raw = raw[raw.source.isin(keep)]
        rob = rob[rob.source.isin(keep)]
    print(f"raw frames: {len(raw)}   normalised frames: {len(rob)}")

    # ---- annotations ----------------------------------------------------
    stim = load_stimuli()
    tone_of = {int(r["Item No."]): int(r["tone"]) for _, r in stim.iterrows()}
    idx = textgrid_index(raw_dir)
    print(f"textgrids indexed: {len(idx)}")

    rows, speak = [], 0
    for _, r in manifest.iterrows():
        num = int(r["source"].split("_")[-1])
        sid = int(r["subj"])
        hit = idx.get((sid, num))
        if hit:
            iv = interval_row(hit[0], hit[1])
        else:
            iv = (np.nan,) * 6
        t0 = raw.loc[raw.source == r["source"], "time"]
        rows.append({
            "source": r["source"], "subj": sid, "sex": r["sex"], "item": num,
            "syllable": r["syllable"], "onset": r["onset"], "vowel": r["vowel"],
            "tone": (np.nan if num in EXCLUDED_TONE_ITEMS
                     else tone_of.get(num, np.nan)),
            "n_frames": int(r["n_frames"]), "duration": float(r["duration"]),
            "t_first": float(t0.min()) if len(t0) else np.nan,
            "t_last": float(t0.max()) if len(t0) else np.nan,
            "t_syl_start": iv[0], "t_syl_end": iv[1],
            "t_onset_start": iv[2], "t_onset_end": iv[3],
            "t_vowel_start": iv[4], "t_vowel_end": iv[5],
        })
        if not np.isnan(iv[0]) and len(t0):
            inside = (t0.min() >= iv[0] - 1e-3) and (t0.max() <= iv[1] + 1e-3)
            speak += int(inside)
    man = pd.DataFrame(rows)
    have_iv = man.t_syl_start.notna().sum()
    print(f"intervals recovered for {have_iv}/{len(man)} utterances; "
          f"frames inside their syllable interval for {speak}/{len(man)}")
    if have_iv and speak < 0.95 * len(man):
        print("  WARNING: clock mismatch between TextGrid intervals and frame times")

    # ---- normalisation statistics ---------------------------------------
    # manifest["path"] already reads "utterances/sub_XX/0002.npz", relative to the corpus root
    stats = normalization_stats(manifest, str(ds))
    print(f"normalisation stats: {len(stats)} rows ({stats.subj.nunique()} subjects)")

    # ---- write the payload ----------------------------------------------
    os.makedirs(os.path.join(out, "kinematics", "utterances"), exist_ok=True)
    os.makedirs(os.path.join(out, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(out, "folds"), exist_ok=True)

    man.to_csv(os.path.join(out, "manifest.csv"), index=False)
    sz_raw = write_csv_gz(raw, os.path.join(out, "kinematics", "features_raw.csv.gz"))
    sz_z = write_csv_gz(rob, os.path.join(out, "kinematics", "features_robust_z.csv.gz"))
    stats.to_csv(os.path.join(out, "kinematics", "normalization_stats.csv"),
                 index=False, float_format="%.10g")
    print(f"features_raw.csv.gz {sz_raw/1e6:.1f} MB   features_robust_z.csv.gz {sz_z/1e6:.1f} MB")

    stim_out = stim[["Item No.", "Chinese.Name", "Transcription", "Transcription1",
                     "Tone", "cons", "vowels"]].rename(columns={
        "Item No.": "item", "Chinese.Name": "chinese_name",
        "Transcription": "transcription", "Transcription1": "target_syllable",
        "Tone": "tone", "cons": "onset", "vowels": "vowel"})
    # UTF-8 *with BOM*: without it Excel and most Windows tools read the Chinese
    # stimulus names as ANSI and render them as mojibake. Pandas and R read the BOM
    # transparently. The kinematic tables carry no non-ASCII, so they stay plain UTF-8.
    for _col in ("transcription", "target_syllable", "chinese_name", "onset", "vowel"):
        stim_out[_col] = stim_out[_col].astype(str).str.strip()
    stim_out.to_csv(os.path.join(out, "annotations", "stimuli.csv"), index=False,
                    encoding="utf-8-sig")

    inv = (man.groupby(["syllable", "onset", "vowel"], dropna=False)
              .agg(n_utterances=("source", "size"),
                   tones=("tone", lambda x: ",".join(str(int(t)) for t in sorted(set(x.dropna())))))
              .reset_index().sort_values("syllable"))
    inv.to_csv(os.path.join(out, "annotations", "inventory.csv"), index=False,
               encoding="utf-8-sig")
    print(f"syllables: {len(inv)}   stimuli: {len(stim_out)}")

    folds_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "folds", "cis_folds.json")
    if os.path.isfile(folds_src):
        shutil.copy2(folds_src, os.path.join(out, "folds", "cis_folds.json"))

    # ---- per-utterance arrays -------------------------------------------
    if not args.skip_npz:
        all_cols = POS_COLS + VEL_COLS + ACC_COLS + GEO_COLS
        n = 0
        for source, g in raw.groupby("source", sort=False):
            subj = f"sub_{int(g.subj.iloc[0]):02d}"
            d = os.path.join(out, "kinematics", "utterances", subj)
            os.makedirs(d, exist_ok=True)
            arr = g[all_cols].to_numpy(np.float32)
            np.savez_compressed(
                os.path.join(d, f"{source}.npz"),
                time=g["time"].to_numpy(np.float32),
                pos=arr[:, :18], vel=arr[:, 18:36],
                acc=arr[:, 36:54], geom=arr[:, 54:59])
            n += 1
        print(f"wrote {n} per-utterance files")

    # ---- documentation ---------------------------------------------------
    per_speaker = man.drop_duplicates("subj").sex.value_counts()
    n_f, n_m = int(per_speaker.get("F", 0)), int(per_speaker.get("M", 0))
    ctx = dict(n_utt=f"{len(man):,}", n_subj=man.subj.nunique(), n_f=n_f, n_m=n_m,
               n_frames=f"{len(raw):,}", n_syl=len(inv), n_stim=len(stim_out))
    (Path(out) / "README.md").write_text(README.format(**ctx) + readme_rest(**ctx),
                                         encoding="utf-8")
    (Path(out) / "SCHEMA.md").write_text(SCHEMA, encoding="utf-8")
    (Path(out) / "LICENSE").write_text(LICENSE, encoding="utf-8")
    (Path(out) / "CITATION.cff").write_text(CITATION.format(**ctx), encoding="utf-8")
    tmpl = Path(__file__).resolve().parent / "release_templates"
    (Path(out) / "check_dataset.py").write_bytes(
        (tmpl / "check_dataset.py").read_bytes())

    # ---- checksums -------------------------------------------------------
    lines = []
    for root, _, files in os.walk(out):
        for f in sorted(files):
            if f == "SHA256SUMS":
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, out).replace("\\", "/")
            lines.append(f"{sha256(fp)}  {rel}")
    (Path(out) / "SHA256SUMS").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    print(f"record size: {total/1e6:.1f} MB, {len(lines)} files checksummed")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
