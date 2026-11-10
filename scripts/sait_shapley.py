"""Exact group Shapley on SAIT-EMA -- the Exp-2 replication (parked-plan pillar 1).

The SAIT-EMA montage has the same feature *structure* as Bruce (59 EMA = 18
position + 18 velocity + 18 acceleration + 5 geometry, plus 47 acoustic
columns), so the paper's G1-G8 taxonomy transfers directly:

    G1 EMA-position   G2 EMA-velocity   G3 EMA-acceleration   G4 EMA-geometry
    G5 Acoustic-voicing (F0, voiced)
    G6 Acoustic-formant (F1-F3)
    G7 Acoustic-spectral (MFCC 1-13 + deltas)
    G8 Acoustic-energy

What differs is *which coils* exist (SAIT: TT TB TD LJ UL LL; Bruce: TD TT LL
LC TL UL), so per-channel comparison -- not per-group comparison -- is the
correct cross-corpus reading. Geometry features reference different coils and
are kept as their own group.

Exact Shapley over all 2^8 = 256 group subsets, using the same inference-time
feature masking as ``src/exp2_shapley.py`` (imported, so the estimator is
identical). Requires a *full* (106-feature) SAIT model.

Usage:
  python scripts/sait_shapley.py --features <features_mono.csv> \
      --ckpt-dir <runs>/ckpt --tag sait_full --native-only --out <runs>/shapley_full.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from math import comb

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import sait_classify as sc  # noqa: E402
from src.exp2_shapley import eval_masked  # noqa: E402
from src.models.bilstm import SyllableBiLSTM  # noqa: E402

COORD = re.compile(r"^[A-Z]{2}[xyz](_velocity|_acceleration)?$")


def feature_groups(cols):
    aco = [c for c in cols if c.startswith(sc.ACO_PREFIX)]
    ema = [c for c in cols if c not in aco]
    pos = [c for c in ema if re.match(r"^[A-Z]{2}[xyz]$", c)]
    vel = [c for c in ema if re.match(r"^[A-Z]{2}[xyz]_velocity$", c)]
    acc = [c for c in ema if re.match(r"^[A-Z]{2}[xyz]_acceleration$", c)]
    geom = [c for c in ema if c not in pos and c not in vel and c not in acc]
    groups = {
        "G1": pos, "G2": vel, "G3": acc, "G4": geom,
        "G5": [c for c in aco if c in ("F0", "voiced")],
        "G6": [c for c in aco if c in ("F1", "F2", "F3")],
        "G7": [c for c in aco if c.startswith("MFCC")],
        "G8": [c for c in aco if c.startswith("energy")],
    }
    assert sum(len(v) for v in groups.values()) == len(cols), \
        (sum(len(v) for v in groups.values()), len(cols))
    return groups


def shapley_weights(n):
    """w[k] = k!(n-k-1)!/n! for a subset of size k."""
    return np.array([1.0 / (n * comb(n - 1, k)) for k in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--native-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = df.dropna(subset=["syllable"]).copy()
    if args.native_only:
        df = df[df["group"] == "native"]
    meta_cols = {"source", "speaker", "group", "sex", "l1", "item", "section",
                 "syllable", "tone", "time"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    groups = feature_groups(feat_cols)
    gids = list(groups)
    n_g = len(gids)
    group_idx = {g: np.array([feat_cols.index(c) for c in groups[g]], dtype=int) for g in gids}

    X, y_lab, spk, sex = [], [], [], []
    classes = {}
    for _, g in df.groupby("source", sort=False):
        g = g.sort_values("time")
        X.append(g[feat_cols].to_numpy(dtype=np.float32))
        lab = g["syllable"].iloc[0]
        if lab not in classes:
            classes[lab] = len(classes)
        y_lab.append(classes[lab]); spk.append(g["speaker"].iloc[0]); sex.append(g["sex"].iloc[0])
    y = np.array(y_lab); spk = np.array(spk); sex = np.array(sex)
    lengths = np.array([len(x) for x in X])
    n_cls = len(classes)
    fold_of = sc.make_folds(spk, sex, args.folds)
    print(f"features={len(feat_cols)} utterances={len(X)} classes={n_cls} groups={n_g}")
    for g in gids:
        print(f"  {g}: {len(groups[g])} cols")

    weights = shapley_weights(n_g)
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.tag + "_f*_s*.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints matching {args.tag}_f*_s*.pt")
    per_run = []
    for path in ckpts:
        m = re.search(r"_f(\d+)_s(\d+)\.pt$", path)
        fold, seed = int(m.group(1)), int(m.group(2))
        ck = torch.load(path, map_location="cpu")
        model = SyllableBiLSTM(ck["input_dim"], ck["num_classes"])
        model.load_state_dict(ck["model"]); model.to(sc.DEV).eval()

        te = np.array([fold_of[s] == fold for s in spk])
        tr = ~te
        tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
        mu = np.concatenate([X[i] for i in tr_idx], axis=0).mean(0)
        sd = np.concatenate([X[i] for i in tr_idx], axis=0).std(0) + 1e-9
        Xs = [(x - mu) / sd for x in X]
        te_seqs = [Xs[i] for i in te_idx]
        te_y = [int(y[i]) for i in te_idx]

        F = len(feat_cols)
        v = np.zeros(2 ** n_g)
        for mask_i in range(2 ** n_g):
            keep = np.ones(F, dtype=bool)
            for b, g in enumerate(gids):
                if not (mask_i >> b) & 1:
                    keep[group_idx[g]] = False
            acc, _ = eval_masked(model, te_seqs, te_y, keep, sc.DEV)
            v[mask_i] = acc
        phi = {}
        full = v[2 ** n_g - 1]
        for b, g in enumerate(gids):
            subset = 0.0
            for mask_i in range(2 ** n_g):
                if (mask_i >> b) & 1:
                    continue
                k = bin(mask_i).count("1")
                subset += weights[k] * (v[mask_i | (1 << b)] - v[mask_i])
            phi[g] = float(subset)
        per_run.append({"fold": fold, "seed": seed, "full_acc": float(full), "phi": phi})
        print(f"  fold {fold} seed {seed}: full={100*full:.1f}  " +
              " ".join(f"{g}{100*phi[g]:+.1f}" for g in gids))

    agg = {g: {"mean": float(np.mean([r["phi"][g] for r in per_run])),
               "std": float(np.std([r["phi"][g] for r in per_run]))} for g in gids}
    order = sorted(gids, key=lambda g: -agg[g]["mean"])
    out = {"corpus": "SAIT-EMA", "scope": "native" if args.native_only else "all",
           "n_groups": n_g, "group_sizes": {g: len(groups[g]) for g in gids},
           "full_acc_mean": float(np.mean([r["full_acc"] for r in per_run])),
           "shapley": agg, "order": order, "runs": per_run}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nordering:", " > ".join(f"{g} ({100*agg[g]['mean']:+.1f})" for g in order))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
