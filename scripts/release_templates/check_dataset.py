"""Verify that this CIS-EMA record is complete and internally consistent.

    python check_dataset.py

Checks the file inventory, the row and column counts, that the per-speaker
normalised table reconstructs from the raw table plus the shipped statistics,
that the derived geometry follows from the normalised positions, and that the
per-utterance arrays agree with the manifest. Exit status is 0 on success.

Needs only numpy and pandas.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
POS = [f"{c}{a}" for c in ("TD", "TL", "TT", "UL", "LL", "LC") for a in "xyz"]
GEO = ["TT_TD_distance", "UL_LL_distance", "TT_LL_distance",
       "TT_TD_LL_angle", "UL_LL_LC_angle"]
META = ["source", "subj", "sex", "time", "syllable", "vowel", "onset"]

fails = []
checks = 0


def ok(cond, label, detail=""):
    global checks
    checks += 1
    if cond:
        print(f"  ok    {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
        fails.append(label)


print("CIS-EMA record check")
print(f"  root: {ROOT}\n")

need = ["README.md", "SCHEMA.md", "LICENSE", "DATA_USE_TERMS.md", "CITATION.cff",
        "manifest.csv",
        "annotations/stimuli.csv", "annotations/inventory.csv",
        "kinematics/features_raw.csv.gz", "kinematics/features_robust_z.csv.gz",
        "kinematics/normalization_stats.csv", "folds/cis_folds.json", "SHA256SUMS"]
for rel in need:
    ok(os.path.isfile(os.path.join(ROOT, rel)), f"present: {rel}")

man = pd.read_csv(os.path.join(ROOT, "manifest.csv"))
raw = pd.read_csv(os.path.join(ROOT, "kinematics", "features_raw.csv.gz"))
z = pd.read_csv(os.path.join(ROOT, "kinematics", "features_robust_z.csv.gz"))
stats = pd.read_csv(os.path.join(ROOT, "kinematics", "normalization_stats.csv"))

print()
ok(len(man) == 4884, "manifest rows", f"{len(man)} (expected 4884)")
ok(man.subj.nunique() == 33, "subjects", str(man.subj.nunique()))
ok(man.syllable.nunique() == 43, "syllable classes", str(man.syllable.nunique()))
ok(set(man.sex.unique()) <= {"F", "M"}, "sex values in {F, M}")
ok("qc_position_outlier" in man.columns and int(man.qc_position_outlier.sum()) == 2,
   "quality flag present", "2 utterances marked")
n114 = int((man["item"] == 114).sum())
ok(int(man.tone.notna().sum()) == len(man) - n114, "tone labels joined, item 114 excluded",
   f"{int(man.tone.notna().sum())} of {len(man)} ({n114} excluded)")

ok(len(raw) == len(z), "same frame count in both tables", f"{len(raw)}")
ok(list(raw.columns) == list(z.columns), "identical column sets", f"{len(raw.columns)} columns")
ok(list(raw.columns) == META + POS + [c + "_velocity" for c in POS]
   + [c + "_acceleration" for c in POS] + GEO, "column order as documented")
ok((raw["source"] == z["source"]).all() and np.allclose(raw["time"], z["time"]),
   "frame identity aligned across the two tables")
ok(raw.source.nunique() == len(man), "utterances in the tables match the manifest",
   str(raw.source.nunique()))

print()
med = stats.pivot(index="subj", columns="column", values="median")
iqr = stats.pivot(index="subj", columns="column", values="iqr")
err = 0.0
for s_id, g in raw.groupby("subj"):
    if int(s_id) not in med.index:
        continue
    rec = ((g[POS].to_numpy() - med.loc[int(s_id), POS].to_numpy())
           / iqr.loc[int(s_id), POS].to_numpy())
    err = max(err, float(np.abs(rec - z.loc[g.index, POS].to_numpy()).max()))
ok(err < 5e-3, "normalised positions reconstruct from raw + statistics",
   f"max |err| = {err:.2e}")

gpos = z[POS].to_numpy().reshape(len(z), 6, 3).astype(np.float64)


def dist(a, b):
    return np.sqrt(((a - b) ** 2).sum(1))


geo_err = max(
    float(np.abs(dist(gpos[:, 2], gpos[:, 0]) - z["TT_TD_distance"].to_numpy()).max()),
    float(np.abs(dist(gpos[:, 3], gpos[:, 4]) - z["UL_LL_distance"].to_numpy()).max()),
    float(np.abs(dist(gpos[:, 2], gpos[:, 4]) - z["TT_LL_distance"].to_numpy()).max()),
)
ok(geo_err < 1e-3, "geometry follows from the normalised positions",
   f"max |err| = {geo_err:.2e}")

print()
npz_dir = os.path.join(ROOT, "kinematics", "utterances")
files = [os.path.join(r, f) for r, _, fs in os.walk(npz_dir)
         for f in fs if f.endswith(".npz")]
ok(len(files) == len(man), "one npz per utterance", f"{len(files)}")
nframes = dict(zip(man.source, man.n_frames))
bad = []
rng = np.random.RandomState(0)
for i in rng.choice(len(files), size=min(80, len(files)), replace=False):
    d = np.load(files[i])
    src = os.path.splitext(os.path.basename(files[i]))[0]
    want = int(nframes.get(src, -1))
    if not {"time", "pos", "vel", "acc", "geom"} <= set(d.files):
        bad.append((src, "keys"))
    elif d["pos"].shape != (want, 18) or d["geom"].shape != (want, 5):
        bad.append((src, f"shape {d['pos'].shape} vs {want}"))
ok(not bad, "sampled arrays match the manifest", f"{len(bad)} bad")

folds = json.load(open(os.path.join(ROOT, "folds", "cis_folds.json")))
test_ids = sorted(s for f in folds["folds"] for s in f["test"])
ok(test_ids == sorted(man.subj.unique().tolist()),
   "folds cover every subject exactly once", f"{len(folds['folds'])} folds")

print()
print(f"{checks - len(fails)}/{checks} checks passed")
if fails:
    print("failed: " + ", ".join(fails))
sys.exit(0 if not fails else 1)
