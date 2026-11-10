"""SAIT-EMA syllable-controlled tone analysis (42 syllables x 4 tones).

Design mirrors the Cantonese tone study: restrict to the multi-tone paradigm so
syllable identity cannot determine tone; train subject-disjoint (leave-one-
speaker-out); baseline = per-syllable majority tone from the training speakers;
controls = duration-only and a permutation null (tone shuffled within syllable in
training). Probes are pooled per-utterance statistics (mean/std) of the EMA vs
acoustic feature blocks.

Usage:
  python scripts/sait_tone_analysis.py --features <features_mono.csv> --out <dir> \
      --native-only
"""
from __future__ import annotations

import argparse
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EMA_BLOCKS = ("_velocity", "_acceleration")
ACO_PREFIX = ("F0", "F1", "F2", "F3", "MFCC", "energy", "voiced")


def pooled(df, cols):
    g = df.groupby("source", sort=False)
    stats = g[cols].agg(["mean", "std"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    meta = g[["speaker", "syllable", "tone", "group", "sex", "l1", "item"]].first()
    return stats.join(meta).reset_index()


def split_cols(meta, feature_cols):
    ema = [c for c in feature_cols if not c.startswith(ACO_PREFIX)]
    aco = [c for c in feature_cols if c.startswith(ACO_PREFIX)]
    return ema, aco


def loso(X, y, syl, speakers, dur=None, seed=0):
    """Leave-one-speaker-out. If dur is given, X is residualized against duration
    (fit on the training speakers) before probing."""
    res = []
    for spk in np.unique(speakers):
        tr = speakers != spk
        te = speakers == spk
        if te.sum() == 0:
            continue
        Xtr, Xte = X[tr], X[te]
        if dur is not None:
            A = np.column_stack([np.ones(tr.sum()), dur[tr]])
            beta, *_ = np.linalg.lstsq(A, Xtr, rcond=None)
            Xtr = Xtr - A @ beta
            Ate = np.column_stack([np.ones(te.sum()), dur[te]])
            Xte = Xte - Ate @ beta
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(sc.transform(Xtr), y[tr])
        acc = clf.score(sc.transform(Xte), y[te])
        prior = {}
        for s in np.unique(syl[tr]):
            prior[s] = Counter(y[tr][syl[tr] == s]).most_common(1)[0][0]
        base = np.mean([prior.get(s, -1) == yy for s, yy in zip(syl[te], y[te])])
        res.append((spk, acc, base, int(te.sum())))
    return res


def perm_null(X, y, syl, speakers, n=20, seed=0):
    rng = np.random.RandomState(seed)
    accs = []
    for rep in range(n):
        yp = y.copy()
        # shuffle tone within syllable (across all speakers)
        for s in np.unique(syl):
            idx = np.where(syl == s)[0]
            yp[idx] = rng.permutation(yp[idx])
        r = loso(X, yp, syl, speakers, seed=rep)
        accs.append(np.mean([a for _, a, _, _ in r]))
    return float(np.mean(accs)), float(np.std(accs))


def summarize(name, res):
    acc = np.mean([a for _, a, _, _ in res])
    base = np.mean([b for _, _, b, _ in res])
    n = sum(n for _, _, _, n in res)
    return {"probe": name, "acc": round(100 * acc, 2), "baseline": round(100 * base, 2),
            "delta": round(100 * (acc - base), 2), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(path("sait_dataset") / "features_mono.csv"))
    ap.add_argument("--out", default=str(path("sait_dataset") / "tone"))
    ap.add_argument("--native-only", action="store_true")
    ap.add_argument("--perm", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = df.dropna(subset=["syllable", "tone"]).copy()
    df["tone"] = df["tone"].astype(int)
    df = df[df["section"] == "mono_tones"].copy()
    if args.native_only:
        df = df[df["group"] == "native"]
    feat_cols = [c for c in df.columns if c not in
                 ("source", "speaker", "group", "sex", "l1", "item", "section", "syllable", "tone", "time")]
    ema_cols, aco_cols = split_cols(df, feat_cols)
    # duration per utterance
    dur = df.groupby("source")["time"].agg(lambda x: x.iloc[-1] - x.iloc[0])

    P = pooled(df, ema_cols + aco_cols)
    P["duration"] = dur.reindex(P["source"]).to_numpy()
    ema_p = [f"{c}_{s}" for c in ema_cols for s in ("mean", "std")]
    aco_p = [f"{c}_{s}" for c in aco_cols for s in ("mean", "std")]
    y = P["tone"].to_numpy()
    syl = P["syllable"].to_numpy()
    spk = P["speaker"].to_numpy()

    print(f"multi-tone utterances: {len(P)}  syllables: {P['syllable'].nunique()}  "
          f"speakers: {P['speaker'].nunique()}  tones: {sorted(set(y))}")

    results = []
    results.append(summarize("EMA pooled (pos+vel+acc+geom)", loso(P[ema_p].to_numpy(), y, syl, spk)))
    results.append(summarize("EMA residualized on duration", loso(P[ema_p].to_numpy(), y, syl, spk, dur=P["duration"].to_numpy())))
    results.append(summarize("EMA + duration", loso(np.column_stack([P[ema_p].to_numpy(), P["duration"].to_numpy()]), y, syl, spk)))
    results.append(summarize("Acoustic pooled (F0+F1-3+MFCC+energy)", loso(P[aco_p].to_numpy(), y, syl, spk)))
    results.append(summarize("Duration only", loso(P[["duration"]].to_numpy(), y, syl, spk)))

    if args.perm:
        m, s = perm_null(P[ema_p].to_numpy(), y, syl, spk, n=args.perm)
        print(f"EMA permutation null: {100*m:.2f} +- {100*s:.2f}")

    os.makedirs(args.out, exist_ok=True)
    # per-speaker detail for the EMA probe
    res = loso(P[ema_p].to_numpy(), y, syl, spk)
    detail = [{"speaker": s_, "acc": round(100 * a, 2), "baseline": round(100 * b, 2), "n": n}
              for s_, a, b, n in res]
    out = {"scope": "native" if args.native_only else "all",
           "n_utterances": int(len(P)), "n_syllables": int(P["syllable"].nunique()),
           "results": results, "ema_per_speaker": detail}
    if args.perm:
        out["ema_perm_null"] = {"mean": round(100 * m, 2), "std": round(100 * s, 2), "reps": args.perm}
    with open(os.path.join(args.out, f"tone_{'native' if args.native_only else 'all'}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n%-40s %8s %9s %7s" % ("probe", "acc", "baseline", "delta"))
    for r in results:
        print("%-40s %8.2f %9.2f %+7.2f" % (r["probe"], r["acc"], r["baseline"], r["delta"]))


if __name__ == "__main__":
    main()
