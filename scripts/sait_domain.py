"""SAIT-EMA domain shift: train on native Mandarin, test on L2 learners (and back).

The weak version of this question is the ``all`` scope in ``sait_classify.py``,
which pools native and L2 speakers into the same folds -- that measures "is more
data helpful", not domain shift. This script measures the real thing: hold out
an entire language background, train on the other, and evaluate.

Both directions are run so the asymmetry is visible:

    native -> L2   (12 native train/val, 6 L2 test)
    L2 -> native   (6 L2 train/val, 12 native test)

Class inventory is built from the union of both groups, so the output layer (and
therefore the two accuracies) are directly comparable; a test label absent from
training simply can never be predicted.

Usage:
  python scripts/sait_domain.py --features <features_mono_center_only.csv> \
      --native-only-y-dropped ... --out <runs>/domain.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import sait_classify as sc  # noqa: E402


def pick_val(speakers, sex_of, n_val):
    """Sex-stratified deterministic validation speakers (last n per sex)."""
    val = []
    for s in ("F", "M"):
        sp = sorted([x for x in speakers if sex_of[x] == s])
        val.extend(sp[-n_val:])
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--view", default="ema", choices=["ema", "acoustic", "full"])
    ap.add_argument("--drop-y", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = df.dropna(subset=["syllable"]).copy()
    meta_cols = {"source", "speaker", "group", "sex", "l1", "item", "section",
                 "syllable", "tone", "time"}
    feat_cols = sc.select_cols([c for c in df.columns if c not in meta_cols],
                               args.view, args.drop_y)

    X, syl, grp, spk, sex = [], [], [], [], []
    for _, g in df.groupby("source", sort=False):
        g = g.sort_values("time")
        X.append(g[feat_cols].to_numpy(dtype=np.float32))
        syl.append(g["syllable"].iloc[0]); grp.append(g["group"].iloc[0])
        spk.append(g["speaker"].iloc[0]); sex.append(g["sex"].iloc[0])
    syl = np.array(syl, dtype=object); grp = np.array(grp, dtype=object)
    spk = np.array(spk); sex = np.array(sex)
    lengths = np.array([len(x) for x in X])
    # class inventory from the union, so both directions share the output layer
    classes = {}
    for s in syl:
        if s not in classes:
            classes[s] = len(classes)
    y = np.array([classes[s] for s in syl])
    n_cls = len(classes)
    sex_of = {s: x for s, x in zip(spk, sex)}
    native = sorted({s for s, g in zip(spk, grp) if g == "native"})
    l2 = sorted({s for s, g in zip(spk, grp) if g == "l2"})
    print(f"features={len(feat_cols)} utterances={len(X)} classes={n_cls} "
          f"native={len(native)} l2={len(l2)}")

    def run(tr_speakers, va_speakers, te_speakers, seed, name):
        tr = np.array([s in tr_speakers for s in spk])
        va = np.array([s in va_speakers for s in spk])
        te = np.array([s in te_speakers for s in spk])
        tr_idx, va_idx, te_idx = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
        mu = np.concatenate([X[i] for i in tr_idx], axis=0).mean(0)
        sd = np.concatenate([X[i] for i in tr_idx], axis=0).std(0) + 1e-9
        Xs = [(x - mu) / sd for x in X]
        acc, f1, _, _ = sc.run_fold([Xs[i] for i in tr_idx], y[tr_idx], lengths[tr_idx],
                                    [Xs[i] for i in va_idx], y[va_idx], lengths[va_idx],
                                    [Xs[i] for i in te_idx], y[te_idx], lengths[te_idx],
                                    n_cls, seed, args.epochs)
        maj = int(np.bincount(y[tr_idx], minlength=n_cls).argmax())
        maj_acc = float((y[te_idx] == maj).mean())
        print(f"  [{name}] seed {seed}: acc {100*acc:.2f} f1 {100*f1:.2f} "
              f"maj {100*maj_acc:.2f}  (ntr={len(tr_idx)} nte={len(te_idx)})")
        return {"direction": name, "seed": seed, "acc": round(100 * acc, 2),
                "f1": round(100 * f1, 2), "maj": round(100 * maj_acc, 2),
                "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                "train_speakers": sorted(tr_speakers), "test_speakers": sorted(te_speakers)}

    val_native = pick_val(native, sex_of, 1)
    val_l2 = pick_val(l2, sex_of, 1)
    train_native = [s for s in native if s not in val_native]
    train_l2 = [s for s in l2 if s not in val_l2]
    print(f"native train={len(train_native)} val={len(val_native)}; "
          f"l2 train={len(train_l2)} val={len(val_l2)}")

    runs = []
    for seed in range(args.seeds):
        runs.append(run(train_native, val_native, l2, seed, "native->l2"))
        runs.append(run(train_l2, val_l2, native, seed, "l2->native"))

    summary = {"corpus": "SAIT-EMA", "n_features": len(feat_cols), "n_classes": n_cls,
               "directions": {}}
    for d in ("native->l2", "l2->native"):
        sel = [r for r in runs if r["direction"] == d]
        summary["directions"][d] = {
            "acc_mean": round(float(np.mean([r["acc"] for r in sel])), 2),
            "acc_std": round(float(np.std([r["acc"] for r in sel])), 2),
            "f1_mean": round(float(np.mean([r["f1"] for r in sel])), 2),
            "maj_mean": round(float(np.mean([r["maj"] for r in sel])), 2)}
    summary["runs"] = runs
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    for d, v in summary["directions"].items():
        print(f"{d:14s} {v['acc_mean']} +- {v['acc_std']} (maj {v['maj_mean']})")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
