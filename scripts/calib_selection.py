"""Calibration SELECTION study.

Same fair-deployment setup as calib_deploy (train on full per-speaker stats,
calibrate the test speaker with k unlabeled utterances), but compare how the k
utterances are CHOSEN:
  random       - uniform random (label-free)
  coverage     - greedy max-coverage of (vowel, onset) pairs (content-known)
  vowel_sweep  - one utterance per vowel, then random fill (fixed protocol-ish)
Train once per fold, evaluate every (strategy, k).
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src.config import RUNS_DIR
from src.pipeline import prepare_data
from src.data.dataset import fit_stats, standardize
from src.models.bilstm import SyllableBiLSTM
from src.training import make_loader, train_model, evaluate_syl

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bz", os.path.join(HERE, "build_zscore_features.py"))
bz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bz)
KEEP, AXES, EMA_COLS = bz.KEEP, bz.AXES, bz.EMA_COLS

DATA = str(path("cis_dataset"))
KS = [5, 10, 20]
STRATS = ["random", "coverage", "vowel_sweep"]


def feats(norm18, t):
    keep_idx = [EMA_COLS.index(f"{c}{a}") for c in KEEP for a in AXES]
    pos = norm18[:, keep_idx]
    dt = float(np.median(np.diff(t)))
    vel, acc = bz.bf.compute_vel_accel(pos, dt)
    vel, acc = bz.bf.winsorize(vel), bz.bf.winsorize(acc)

    def P(c):
        return norm18[:, [EMA_COLS.index(f"{c}{a}") for a in AXES]]
    TT, TD, UL, LL, LC = P("TT"), P("TD"), P("UL"), P("LL"), P("LC")
    dist = lambda a, b: np.sqrt(((a - b) ** 2).sum(1))
    tt_td, ul_ll, tt_ll = dist(TT, TD), dist(UL, LL), dist(TT, LL)

    def ang(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        n1 = np.linalg.norm(v1, axis=1) + 1e-9
        n2 = np.linalg.norm(v2, axis=1) + 1e-9
        return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)))
    geom = np.stack([tt_td, ul_ll, tt_ll, ang(TT, TD, LL), ang(UL, LL, LC)], 1)
    return np.hstack([pos, vel, acc, geom]).astype(np.float32)


def stats(cat):
    med = np.median(cat, 0)
    q1, q3 = np.percentile(cat, [25, 75], axis=0)
    return med, (q3 - q1) + 1e-6


def select(recs, k, strat, seed):
    n = len(recs)
    if k >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    if strat == "random":
        return list(rng.choice(n, k, replace=False))
    if strat == "vowel_sweep":
        chosen = []
        for v in sorted({r["vowel"] for r in recs}):
            idx = [i for i, r in enumerate(recs) if r["vowel"] == v]
            chosen.append(int(rng.choice(idx)))
            if len(chosen) >= k:
                return chosen[:k]
        pool = [i for i in range(n) if i not in chosen]
        rng.shuffle(pool)
        return (chosen + pool)[:k]
    if strat == "coverage":
        covered, chosen, pool = set(), [], list(range(n))
        while len(chosen) < k and pool:
            best, bestgain = None, -1
            for i in pool:
                gain = 1 if (recs[i]["vowel"], recs[i]["onset"]) not in covered else 0
                if gain > bestgain or (gain == bestgain and rng.rand() < 0.3):
                    best, bestgain = i, gain
            chosen.append(best)
            covered.add((recs[best]["vowel"], recs[best]["onset"]))
            pool.remove(best)
        return chosen
    raise ValueError(strat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(RUNS_DIR, "exp9_calib", "calib_selection.json"))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    records, _, subjects, _, folds = prepare_data(os.path.join(DATA, "variants", "robust_z"))
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    path = {str(row["source"]): row["path"] for _, row in man.iterrows()}
    fx, sm, tt = {}, {}, {}
    for r in records:
        d = np.load(os.path.join(DATA, path[r["source"]]))
        t = d["time"].astype(np.float64)
        f = bz.bf.fix_outliers(d["ema"].astype(np.float64))
        fx[r["source"]] = f; sm[r["source"]] = bz.smooth(f, t); tt[r["source"]] = t
    by_subj = defaultdict(list)
    for r in records:
        by_subj[r["subj"]].append(r)

    curve = {(s, k): [] for s in STRATS for k in KS}
    for fd in folds:
        tr = [r for r in records if r["subj"] in set(fd["train"])]
        va = [r for r in records if r["subj"] in set(fd["val"])]
        te = [r for r in records if r["subj"] in set(fd["test"])]
        le = LabelEncoder().fit([r["syllable"] for r in tr])
        ncls = len(le.classes_)
        y_tr = le.transform([r["syllable"] for r in tr]); y_va = le.transform([r["syllable"] for r in va])
        st = {}
        for s in set(fd["train"]) | set(fd["val"]):
            st[s] = stats(np.concatenate([fx[x["source"]] for x in by_subj[s]]))
        tr_seqs = [feats((sm[r["source"]] - st[r["subj"]][0]) / st[r["subj"]][1], tt[r["source"]]) for r in tr]
        va_seqs = [feats((sm[r["source"]] - st[r["subj"]][0]) / st[r["subj"]][1], tt[r["source"]]) for r in va]
        mu, sd = fit_stats(tr_seqs)
        tr_seqs = standardize(tr_seqs, mu, sd); va_seqs = standardize(va_seqs, mu, sd)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            counts = np.bincount(y_tr, minlength=ncls).astype(np.float32)
            cw = torch.tensor(counts.sum() / (ncls * counts), dtype=torch.float32).to(device)
            model, _ = train_model(
                SyllableBiLSTM(tr_seqs[0].shape[1], ncls).to(device),
                make_loader(tr_seqs, y_tr, 32, True), make_loader(va_seqs, y_va, 32, False),
                device, epochs=args.epochs, class_weight=cw, verbose=False)
            for strat in STRATS:
                for k in KS:
                    preds_all, true_all = [], []
                    for subj in sorted({r["subj"] for r in te}):
                        recs = by_subj[subj]
                        idx = select(recs, k, strat, 42 + subj)
                        m, s = stats(np.concatenate([fx[recs[i]["source"]] for i in idx]))
                        seqs = [feats((sm[x["source"]] - m) / s, tt[x["source"]]) for x in recs]
                        seqs = standardize(seqs, mu, sd)
                        ys = le.transform([x["syllable"] for x in recs])
                        p, t = evaluate_syl(model, make_loader(seqs, ys, 32, False), device)
                        preds_all.append(p); true_all.append(t)
                    p = np.concatenate(preds_all); t = np.concatenate(true_all)
                    curve[(strat, k)].append(float((p == t).mean()))
    out = {f"{s}|k{k}": [float(np.mean(curve[(s, k)])), float(np.std(curve[(s, k)]))] for s in STRATS for k in KS}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("calibration selection (train full, test with k selected utterances):")
    print(f"{'strategy':14s} " + " ".join(f"k={k:>2}" for k in KS))
    for s in STRATS:
        print(f"{s:14s} " + " ".join(f"{out[f'{s}|k{k}'][0]*100:5.1f}" for k in KS))


if __name__ == "__main__":
    main()
