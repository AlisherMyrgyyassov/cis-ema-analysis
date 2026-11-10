"""Direct-F0 tone ceiling for the syllable-controlled tone comparison.

The paper reports tone from the *classifier's pooled embedding* (EMA 73.0, full
73.9 on the ten multi-tone syllables). Because the full model sees F0 as an input,
a reader can ask why its probe is barely above EMA. This script measures the
ceiling directly: a logistic probe on utterance-level F0 statistics, same ten
syllables and the same folds, with the same per-syllable majority baseline.

Features (voiced frames only): mean, std, min, max, range of log2(F0), the
within-utterance slope of log2(F0) over normalized time, and the voiced fraction.
A mean-only variant separates pitch level from contour.

Usage:
  python scripts/f0_tone_baseline.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import DATA_DIR                                   # noqa: E402
from src.data.dataset import load_dataset                         # noqa: E402
from src.data.tone import add_tone_to_records                     # noqa: E402
from src.data.cv import make_folds                                # noqa: E402
from src.config import N_FOLDS, N_VAL_SUBJECTS, FOLD_SEED         # noqa: E402

FEATURES = ["mean", "std", "min", "max", "range", "slope", "voiced_frac"]


def utt_features(frame, col_index):
    f0 = np.log2(np.maximum(frame[:, col_index["F0"]], 1e-3))
    voiced = frame[:, col_index["voiced"]] > 0.5
    if voiced.sum() < 2:
        return None
    v = f0[voiced]
    t = np.linspace(0.0, 1.0, len(v))
    slope = float(np.polyfit(t, v, 1)[0]) if len(v) > 1 else 0.0
    return {
        "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
        "max": float(v.max()), "range": float(v.max() - v.min()),
        "slope": slope, "voiced_frac": float(voiced.mean()),
    }


def main():
    records, col_index = load_dataset(DATA_DIR)
    add_tone_to_records(records)

    tone_by_syl = {}
    for r in records:
        if r.get("tone") is not None:
            tone_by_syl.setdefault(r["syllable"], set()).add(r["tone"])
    multi = sorted(s for s, ts in tone_by_syl.items() if len(ts) >= 2)

    kept, feats = [], []
    for r in records:
        if r.get("tone") is None or r["syllable"] not in multi:
            continue
        f = utt_features(r["frame"], col_index)
        if f is None:
            continue
        kept.append(r)
        feats.append([f[k] for k in FEATURES])
    X = np.asarray(feats, dtype=float)
    spk = np.array([r["subj"] for r in kept])
    print(f"multi-tone syllables: {len(multi)}  utterances: {len(kept)}")

    # Per-speaker robust normalization (label-free, same transductive setting as
    # the EMA features): the paper normalizes articulation per speaker, so a raw
    # cross-speaker F0 probe would not be a comparable ceiling.
    X_psz = X.copy()
    for s in sorted(set(spk)):
        m = spk == s
        med = np.median(X[m], axis=0)
        iqr = np.percentile(X[m], 75, axis=0) - np.percentile(X[m], 25, axis=0)
        X_psz[m] = (X[m] - med) / (iqr + 1e-9)

    # Given-syllable variant: one-hot syllable identity concatenated to the F0
    # statistics. The EMA probe reads an embedding trained on syllable labels, so
    # it has syllable identity available; this makes the two probes comparable.
    syls = sorted({r["syllable"] for r in kept})
    S = np.zeros((len(kept), len(syls)))
    for i, r in enumerate(kept):
        S[i, syls.index(r["syllable"])] = 1.0
    X_psz_syl = np.hstack([X_psz, S])
    K = X_psz.shape[1]

    subjects = sorted({r["subj"] for r in kept})
    sex_map = {r["subj"]: r["sex"] for r in kept}
    # same fold construction as the rest of the paper (fold seed 0, 6 folds)
    all_subjects = sorted({r["subj"] for r in records})
    all_sex = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(all_subjects, all_sex, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)

    variants = {
        "raw_full": (X, list(range(len(FEATURES)))),
        "raw_mean_only": (X, [0]),
        "psz_full": (X_psz, list(range(len(FEATURES)))),
        "psz_mean_only": (X_psz, [0]),
        "psz_syl_full": (X_psz_syl, list(range(K + len(syls)))),
        "psz_syl_mean_only": (X_psz_syl, [0] + list(range(K, K + len(syls)))),
    }
    out = {"features": FEATURES, "multi_tone_syllables": multi,
           "n_utterances": len(kept),
           "note": "psz = per-speaker robust z (median/IQR), label-free",
           "variants": {}}

    for name, (Xv, cols) in variants.items():
        rows = []
        for fid, fd in enumerate(folds):
            tr = [i for i, r in enumerate(kept) if r["subj"] in set(fd["train"])]
            te = [i for i, r in enumerate(kept) if r["subj"] in set(fd["test"])]
            if not tr or not te:
                continue
            Xtr, Xte = Xv[tr][:, cols], Xv[te][:, cols]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            ytr = np.array([kept[i]["tone"] for i in tr])
            yte = np.array([kept[i]["tone"] for i in te])
            clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(Xtr, ytr)
            acc = float((clf.predict(Xte) == yte).mean())
            maj = {}
            for i in tr:
                r = kept[i]
                maj.setdefault(r["syllable"], {}).setdefault(r["tone"], 0)
                maj[r["syllable"]][r["tone"]] += 1
            maj = {s: max(d, key=d.get) for s, d in maj.items()}
            base = float(np.mean([maj.get(kept[i]["syllable"], 1) == kept[i]["tone"]
                                  for i in te]))
            rows.append({"fold": fid, "probe": acc, "baseline": base,
                         "n_test": len(te)})
            print(f"  [{name}] fold {fid}: probe {acc*100:.1f} base {base*100:.1f}")

        fa = np.array([r["probe"] for r in rows])
        fb = np.array([r["baseline"] for r in rows])
        t, p = stats.ttest_rel(fa, fb)
        d = float((fa - fb).mean() / (fa - fb).std(ddof=1))
        out["variants"][name] = {
            "probe_mean": float(fa.mean()), "probe_std": float(fa.std(ddof=1)),
            "baseline_mean": float(fb.mean()), "diff": float((fa - fb).mean()),
            "t": float(t), "p": float(p), "cohens_d": d,
            "per_fold_probe": fa.tolist(), "per_fold_baseline": fb.tolist(),
            "folds": rows,
        }

    os.makedirs(os.path.join(ROOT, "runs", "f0_baseline"), exist_ok=True)
    path = os.path.join(ROOT, "runs", "f0_baseline", "f0_tone_baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nF0 psz full-set probe {100*out['variants']['psz_full']['probe_mean']:.2f} "
          f"(baseline {100*out['variants']['psz_full']['baseline_mean']:.2f}, "
          f"gain {100*out['variants']['psz_full']['diff']:+.2f}, "
          f"p={out['variants']['psz_full']['p']:.4f})")
    print(f"F0 psz given-syllable probe "
          f"{100*out['variants']['psz_syl_full']['probe_mean']:.2f} "
          f"(gain {100*out['variants']['psz_syl_full']['diff']:+.2f}, "
          f"p={out['variants']['psz_syl_full']['p']:.4f})")
    print(f"F0 raw full-set probe {100*out['variants']['raw_full']['probe_mean']:.2f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
