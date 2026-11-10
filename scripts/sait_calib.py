"""Fair-deployment calibration curve on SAIT-EMA (parked-plan pillar 3).

Regime (same idea as ``scripts/run_fair_deploy.py`` for Bruce): the model is
trained on *training* speakers whose offset was estimated from all their data;
a NEW test speaker estimates only their own offset, from k unlabeled utterances.
Deployment-realistic -- no labels, no extra sensors, just k utterances.

Implementation notes (both verified against the released feature tables):

1. The offset is a per-channel **median** over that speaker's frames, not a mean:
   ``center_only = ema - median_s`` (``build_features.apply_norm``). Checked by
   rebuilding from ``features_mono_raw.csv`` -- median reproduces
   ``features_mono_center_only.csv`` to 0.0077 (float32 rounding), mean is off by
   up to 79.6 mm.
2. The offset is per *channel*, so each coil shifts differently and the five
   geometry features are NOT invariant (max diff 45 mm on sd 4 mm). Velocity and
   acceleration effectively are (d/dt of a constant shift). So geometry is
   recomputed from the re-centred positions using the builder's own formulas.

Consequently k = all must reproduce the canonical ``center_only`` run; that is
the validation to check first.

Usage:
  python scripts/sait_calib.py --raw <features_mono_raw.csv> \
      --ckpt-dir <runs>/ckpt --tag sait_syllable_native --native-only --drop-y \
      --out <runs>/calib_fair.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import sait_classify as sc  # noqa: E402
from preprocessing.sait_ema.build_features import (EMA_COLS, GEOM_ANG,  # noqa: E402
                                                   GEOM_DIST)
from preprocessing.sait_ema.reader import AXES  # noqa: E402
from src.models.bilstm import SyllableBiLSTM  # noqa: E402

VEL = [c + "_velocity" for c in EMA_COLS]
ACC = [c + "_acceleration" for c in EMA_COLS]
GEOM = ([f"{a}_{b}_distance" for a, b in GEOM_DIST]
        + [f"{a}_{b}_{c}_angle" for a, b, c in GEOM_ANG])
ALL_EMA = EMA_COLS + VEL + ACC + GEOM           # the builder's column order
_IDX = {c: i for i, c in enumerate(EMA_COLS)}
DEFAULT_KS = "1,2,3,5,10,20,50,100,0"           # 0 = all


def geom_features(pos18):
    """The builder's 5 geometry features, from 18 position channels."""
    def P(s):
        return pos18[:, [_IDX[f"{s}{a}"] for a in AXES]]

    def dist(a, b):
        return np.sqrt(((a - b) ** 2).sum(1))

    def angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        n1 = np.linalg.norm(v1, axis=1) + 1e-9
        n2 = np.linalg.norm(v2, axis=1) + 1e-9
        return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)))

    cols = [dist(P(a), P(b)) for a, b in GEOM_DIST]
    cols += [angle(P(a), P(b), P(c)) for a, b, c in GEOM_ANG]
    return np.stack(cols, axis=1).astype(np.float32)


def speaker_offsets(by_spk, Xpos, k, rng):
    """Per-channel median over k utterances per speaker (k=0 -> all)."""
    out = {}
    for s, idxs in by_spk.items():
        pick = idxs if k == 0 or k >= len(idxs) else list(rng.choice(idxs, size=k, replace=False))
        out[s] = np.median(np.concatenate([Xpos[i] for i in pick], axis=0), axis=0)
    return out


def evaluate(model, seqs, labels, device, batch_size=64):
    model.eval()
    ds = sc.SeqDataset(seqs, labels, np.array([len(s) for s in seqs]))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=sc.collate)
    preds, trues = [], []
    with torch.no_grad():
        for x, y, l in loader:
            preds.append(model(x.to(device), l.to(device)).argmax(1).cpu()); trues.append(y)
    preds = torch.cat(preds).numpy(); trues = torch.cat(trues).numpy()
    return float((preds == trues).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--native-only", action="store_true")
    ap.add_argument("--drop-y", action="store_true")
    ap.add_argument("--ks", default=DEFAULT_KS)
    ap.add_argument("--reps", type=int, default=3, help="random draws per k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    meta = ["source", "speaker", "group", "sex", "syllable", "tone", "time"]
    df = pd.read_csv(args.raw, usecols=meta + EMA_COLS + VEL + ACC)
    df = df.dropna(subset=["syllable"]).copy()
    if args.native_only:
        df = df[df["group"] == "native"]
    keep = [ALL_EMA.index(c) for c in
            sc.select_cols(list(ALL_EMA), "ema", args.drop_y)]

    Xpos, Xvel, Xacc, syl, spk, sex = [], [], [], [], [], []
    for _, g in df.groupby("source", sort=False):
        g = g.sort_values("time")
        Xpos.append(g[EMA_COLS].to_numpy(dtype=np.float32))
        Xvel.append(g[VEL].to_numpy(dtype=np.float32))
        Xacc.append(g[ACC].to_numpy(dtype=np.float32))
        syl.append(g["syllable"].iloc[0]); spk.append(g["speaker"].iloc[0]); sex.append(g["sex"].iloc[0])
    del df
    classes = {}
    for s in syl:
        if s not in classes:
            classes[s] = len(classes)
    y = np.array([classes[s] for s in syl])
    spk = np.array(spk); sex = np.array(sex)
    n_cls = len(classes)
    fold_of = sc.make_folds(spk, sex, args.folds)
    by_spk = {}
    for i, s in enumerate(spk):
        by_spk.setdefault(s, []).append(i)
    print(f"utterances={len(Xpos)} classes={n_cls} speakers={len(by_spk)} "
          f"kept_features={len(keep)}")

    def build(offsets):
        """(Xpos - offset) + vel + acc + geometry(from centred pos), kept cols."""
        out = []
        for pos, vel, acc, s in zip(Xpos, Xvel, Xacc, spk):
            pc = pos - offsets[s]
            full = np.concatenate([pc, vel, acc, geom_features(pc)], axis=1)
            out.append(full[:, keep])
        return out

    all_off = speaker_offsets(by_spk, Xpos, 0, np.random.RandomState(0))
    Xc = build(all_off)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.tag + "_f*_s*.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints matching {args.tag}_f*_s*.pt")
    rows = []
    for path in ckpts:
        m = re.search(r"_f(\d+)_s(\d+)\.pt$", path)
        fold, seed = int(m.group(1)), int(m.group(2))
        ck = torch.load(path, map_location="cpu")
        model = SyllableBiLSTM(ck["input_dim"], ck["num_classes"])
        model.load_state_dict(ck["model"]); model.to(sc.DEV).eval()

        te = np.array([fold_of[s] == fold for s in spk])
        rest = ~te
        va = np.array([fold_of[s] == (fold + 1) % args.folds for s in spk]) & rest
        tr = rest & ~va
        tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        mu = np.concatenate([Xc[i] for i in tr_idx], axis=0).mean(0)
        sd = np.concatenate([Xc[i] for i in tr_idx], axis=0).std(0) + 1e-9
        te_y = [int(y[i]) for i in te_idx]

        for k in ks:
            accs = []
            reps = 1 if k == 0 else args.reps
            for rep in range(reps):
                if k == 0:
                    seqs = [(Xc[i] - mu) / sd for i in te_idx]
                else:
                    off = speaker_offsets(by_spk, Xpos, k, np.random.RandomState(1000 * rep + 7))
                    Xk = build(off)
                    seqs = [(Xk[i] - mu) / sd for i in te_idx]
                accs.append(evaluate(model, seqs, te_y, sc.DEV))
            rows.append({"fold": fold, "seed": seed, "k": k,
                         "acc": round(100 * float(np.mean(accs)), 2),
                         "acc_std": round(100 * float(np.std(accs)), 2),
                         "n_reps": reps})
        am = {r["k"]: r["acc"] for r in rows if r["fold"] == fold and r["seed"] == seed}
        print(f"  fold {fold} seed {seed}: " +
              " ".join(f"k={k if k else 'all'}:{am[k]:.1f}" for k in ks))

    curve = {str(k): {"mean": round(float(np.mean([r["acc"] for r in rows if r["k"] == k])), 2),
                      "std": round(float(np.std([r["acc"] for r in rows if r["k"] == k])), 2),
                      "n": len([r for r in rows if r["k"] == k])} for k in ks}
    out = {"corpus": "SAIT-EMA", "n_features": len(keep), "n_classes": n_cls,
           "scope": "native" if args.native_only else "all", "drop_y": bool(args.drop_y),
           "ks": ks, "curve": curve, "runs": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nfair-deployment curve: " +
          " ".join(f"k={k if k else 'all'} {curve[str(k)]['mean']}" for k in ks))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
