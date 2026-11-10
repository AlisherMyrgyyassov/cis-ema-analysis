"""Fair-deployment calibration curve for a given normalization mode.

Realistic deployment: train the model on training speakers with FULL per-speaker
stats; a NEW test speaker supplies k utterances from which their own stats are
estimated. Mode `center_only` subtracts the per-speaker mean; `robust_z`
subtracts median and divides by IQR.

Usage:
  python scripts/run_fair_deploy.py --mode center_only --seeds 0,1
"""
import argparse
import importlib.util
import json
import os
import sys

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.config import RUNS_DIR
from src.pipeline import prepare_data
from src.data.dataset import fit_stats, standardize
from src.models.bilstm import SyllableBiLSTM
from src.training import make_loader, train_model, evaluate_syl

spec = importlib.util.spec_from_file_location("bz", os.path.join(ROOT, "scripts", "build_zscore_features.py"))
bz = importlib.util.module_from_spec(spec); spec.loader.exec_module(bz)
KEEP, AXES, EMA_COLS = bz.KEEP, bz.AXES, bz.EMA_COLS
DATA = str(path("cis_dataset"))
KS = [1, 2, 3, 5, 10, 20, 0]  # 0 = all


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


def apply_mode(sm, cat, mode):
    if mode == "center_only":
        return sm - cat.mean(0)
    med = np.median(cat, 0)
    q1, q3 = np.percentile(cat, [25, 75], axis=0)
    return (sm - med) / ((q3 - q1) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="center_only")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    out = args.out or os.path.join(ROOT, "runs", f"fair_deploy_{args.mode}")
    os.makedirs(out, exist_ok=True)

    import pandas as pd
    records, _, _, _, folds = prepare_data(os.path.join(DATA, "variants", args.mode))
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    path = {str(r["source"]): r["path"] for _, r in man.iterrows()}
    fx, sm, tt = {}, {}, {}
    for r in records:
        d = np.load(os.path.join(DATA, path[r["source"]]))
        t = d["time"].astype(np.float64)
        f = bz.bf.fix_outliers(d["ema"].astype(np.float64))
        fx[r["source"]] = f; sm[r["source"]] = bz.smooth(f, t); tt[r["source"]] = t
    by_subj = defaultdict(list)
    for r in records:
        by_subj[r["subj"]].append(r)

    curve = {k: [] for k in KS}
    for fd in folds:
        tr = [r for r in records if r["subj"] in set(fd["train"])]
        va = [r for r in records if r["subj"] in set(fd["val"])]
        te = [r for r in records if r["subj"] in set(fd["test"])]
        le = LabelEncoder().fit([r["syllable"] for r in tr]); ncls = len(le.classes_)
        y_tr = le.transform([r["syllable"] for r in tr]); y_va = le.transform([r["syllable"] for r in va])
        tr_seqs = [feats(apply_mode(sm[r["source"]], np.concatenate([fx[x["source"]] for x in by_subj[r["subj"]]]), args.mode), tt[r["source"]]) for r in tr]
        va_seqs = [feats(apply_mode(sm[r["source"]], np.concatenate([fx[x["source"]] for x in by_subj[r["subj"]]]), args.mode), tt[r["source"]]) for r in va]
        mu, sd = fit_stats(tr_seqs)
        tr_seqs, va_seqs = standardize(tr_seqs, mu, sd), standardize(va_seqs, mu, sd)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            cw = np.bincount(y_tr, minlength=ncls).astype(np.float32)
            cw = torch.tensor(cw.sum() / (ncls * cw), dtype=torch.float32).to(device)
            model, _ = train_model(SyllableBiLSTM(tr_seqs[0].shape[1], ncls).to(device),
                                   make_loader(tr_seqs, y_tr, 32, True),
                                   make_loader(va_seqs, y_va, 32, False), device,
                                   epochs=args.epochs, class_weight=cw, verbose=False)
            for k in KS:
                ps, ts = [], []
                for subj in sorted({r["subj"] for r in te}):
                    recs = by_subj[subj]
                    idx = (np.random.RandomState(42 + subj).choice(len(recs), min(k, len(recs)), replace=False)
                           if k else np.arange(len(recs)))
                    cat = np.concatenate([fx[recs[i]["source"]] for i in idx])
                    seqs = [feats(apply_mode(sm[x["source"]], cat, args.mode), tt[x["source"]]) for x in recs]
                    seqs = standardize(seqs, mu, sd)
                    ys = le.transform([x["syllable"] for x in recs])
                    p, t = evaluate_syl(model, make_loader(seqs, ys, 32, False), device)
                    ps.append(p); ts.append(t)
                p = np.concatenate(ps); t = np.concatenate(ts)
                curve[k].append(float((p == t).mean()))
        partial = {str(k or "all"): [float(np.mean(curve[k])), float(np.std(curve[k]))] for k in KS if curve[k]}
        with open(os.path.join(out, "fair_deploy.json"), "w") as f:   # incremental
            json.dump({"folds_done": len(curve[KS[0]]), "curve": partial}, f, indent=2)
        print(f"fold {len(curve[KS[0]])} done", flush=True)
    res = {str(k or "all"): [float(np.mean(curve[k])), float(np.std(curve[k]))] for k in KS}
    with open(os.path.join(out, "fair_deploy.json"), "w") as f:
        json.dump({"folds_done": len(curve[KS[0]]), "curve": res}, f, indent=2)
    print(f"fair deployment ({args.mode}):")
    for k in KS:
        v = res[str(k or "all")]; print(f"  k={str(k or 'all'):>3}: {v[0]*100:.2f} +- {v[1]*100:.2f}")


if __name__ == "__main__":
    main()
