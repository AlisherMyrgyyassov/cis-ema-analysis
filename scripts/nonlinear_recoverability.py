"""Nonlinear recoverability check for the Exp-4 acoustic targets.

Exp-4 measures EMA -> acoustic recoverability with a *global ridge* under the
locked subject-disjoint CV, both in-sample and cross-subject, with both sides
z-scored within each utterance (time-course tracking). This script repeats the
exact same protocol with two nonlinear regressors of controlled capacity:

  * ``mlp``     -- small MLP (128, 64), early stopping;
  * ``histgbr`` -- histogram gradient-boosted trees.

To keep the linear/nonlinear comparison matched, every model is fit on the same
random subsample of the (concatenated) training frames per fold. The in-sample
row fits on all frames and evaluates on all frames (the generous ceiling Exp-4
reports); the cross-subject row fits on train+val subjects and evaluates on the
held-out test subjects.

Run (base env has sklearn/numpy)::

    python scripts/nonlinear_recoverability.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (DATA_DIR, EMA_COLUMNS, N_FOLDS, N_VAL_SUBJECTS, FOLD_SEED)
from src.data.dataset import load_dataset
from src.data.cv import make_folds

TARGETS = ["F0", "F1", "F2", "F3", "MFCC_2", "MFCC_5", "MFCC_6"]
VOICED = {"F0", "F1", "F2", "F3"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "runs", "paper", "exp4", "nonlinear_recoverability.json")


def frames(rec, x_idx, aco_idx, voiced_idx, voiced_only):
    f = rec["frame"]
    m = (f[:, voiced_idx] > 0.5) if voiced_only else np.ones(len(f), dtype=bool)
    if m.sum() < 5:
        return None, None
    x = f[m][:, x_idx].astype(np.float64)
    y = f[m][:, aco_idx].astype(np.float64)
    ys = y.std()
    if ys < 1e-9:
        return None, None
    y = (y - y.mean()) / ys
    xs = x.std(0); xs[xs < 1e-9] = 1.0
    x = (x - x.mean(0, keepdims=True)) / xs
    return x, y


def make_model(kind):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                            solver="adam", batch_size=8192, learning_rate_init=1e-3,
                            alpha=1e-4, early_stopping=True, n_iter_no_change=5,
                            max_iter=200, random_state=0)
    if kind == "histgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                             early_stopping=True, n_iter_no_change=10,
                                             random_state=0)
    raise ValueError(kind)


def r2_per_utt(model, te_x, te_y):
    out = []
    for x, y in zip(te_x, te_y):
        yhat = model.predict(x)
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float((y ** 2).sum())
        if ss_tot > 1e-9:
            out.append(1.0 - ss_res / ss_tot)
    return out


def summarize(r2):
    a = np.array(r2)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "n": int(len(a))}


def run(models, targets, n_train, seed=0):
    records, col_index = load_dataset(DATA_DIR)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)
    x_idx = [col_index[c] for c in EMA_COLUMNS]
    voiced_idx = col_index["voiced"]
    rng = np.random.RandomState(seed)

    results = {m: {t: {"insample": [], "cross": []} for t in targets} for m in models}

    def gather(recs, aco_idx, vo):
        xs, ys = [], []
        for r in recs:
            x, y = frames(r, x_idx, aco_idx, voiced_idx, vo)
            if x is not None:
                xs.append(x); ys.append(y)
        return xs, ys

    def subsample(X, Y, r):
        if X.shape[0] > n_train:
            sel = r.choice(X.shape[0], n_train, replace=False)
            return X[sel], Y[sel]
        return X, Y

    for t in targets:
        aco_idx = col_index[t]
        vo = t in VOICED
        # ---- in-sample: fit once on all frames, evaluate on all frames ----
        all_x, all_y = gather(records, aco_idx, vo)
        ax, ay = subsample(np.concatenate(all_x), np.concatenate(all_y),
                           np.random.RandomState(seed))
        for m in models:
            mdl = make_model(m).fit(ax, ay)
            results[m][t]["insample"].extend(r2_per_utt(mdl, all_x, all_y))
        # ---- cross-subject: per fold ----
        for fd in folds:
            tr_recs = [r for r in records if r["subj"] in set(fd["train"] + fd["val"])]
            te_recs = [r for r in records if r["subj"] in set(fd["test"])]
            Xtr, Ytr = gather(tr_recs, aco_idx, vo)
            if not Xtr:
                continue
            Xs, Ys = subsample(np.concatenate(Xtr), np.concatenate(Ytr), rng)
            te_x, te_y = gather(te_recs, aco_idx, vo)
            for m in models:
                mdl = make_model(m).fit(Xs, Ys)
                results[m][t]["cross"].extend(r2_per_utt(mdl, te_x, te_y))
        print(f"target {t} done", flush=True)

    out = {m: {t: {k: summarize(v) for k, v in d.items()} for t, d in td.items()}
           for m, td in results.items()}
    # reference: Exp-4 linear (full-data) cross-subject medians, for context
    ref = os.path.join(os.path.dirname(OUT), "correlation.json")
    if os.path.exists(ref):
        j = json.load(open(ref))
        out["_exp4_linear_full"] = {row["acoustic_feature"]: {
            "joint_ridge_r2_in": row["joint_ridge_r2_in"],
            "joint_ridge_r2_xs": row["joint_ridge_r2_xs"]} for row in j["table"]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge,mlp,histgbr")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-train", type=int, default=150000)
    ap.add_argument("--folds", type=int, default=0, help="0 = all folds")
    a = ap.parse_args()
    t0 = time.time()
    run(a.models.split(","), a.targets.split(","), a.n_train)
    print(f"elapsed {time.time() - t0:.0f}s")
