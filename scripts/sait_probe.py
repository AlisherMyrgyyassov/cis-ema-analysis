"""Experiment 3 (probing) on SAIT-EMA.

Mirrors ``src/exp3_probing.py`` exactly -- same four stages (proj / layer0 /
layer1 / pooled), same logistic-regression probe, same shuffled-label control
and the same ``selectivity = acc - control`` metric -- so the SAIT numbers are
directly comparable with the Cantonese Exp-3 table
(``runs/paper/exp3/probing_{ema,full}.json``).

Properties probed:

    onset  (22)  = pinyin initial, "-" for zero onset
    final  (57)  = pinyin final after the initial (see the note in
                   ``docs/findings/sait-ema.md``: 37 after normalisation)
    tone    (4)  = lexical tone, restricted to the balanced 42x4 subset
                   (``section == mono_tones``: every syllable occurs in all
                   four tones), so the tone probe is syllable-controlled by
                   construction.

The SAIT syllable target is *tone-less* (405 syllable types), so any tone
decodability is incidental to the task -- the same situation as the Cantonese
model, where tone labels come from ``stimuli.xlsx`` rather than the training
objective. That is what makes the tone probe a replication rather than a
tautology.

Usage:
  python scripts/sait_probe.py --features <features_mono_center_only.csv> \
      --ckpt-dir <runs>/ckpt --tag sait_syllable_native --native-only --drop-y \
      --out <runs>/probe_native.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import sait_classify as sc  # noqa: E402
from src.models.bilstm import SyllableBiLSTM  # noqa: E402
from preprocessing.sait_ema.reader import split_syllable  # noqa: E402

STAGES = ["proj", "layer0", "layer1", "pooled"]
TONE_SECTION = "mono_tones"


def extract(model, seqs, device, batch_size=256):
    """Mean-pooled utterance representations at each depth (as Exp 3)."""
    model.eval()
    out = {s: [] for s in STAGES}
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            lengths = torch.tensor([len(s) for s in chunk], dtype=torch.long)
            max_len = int(lengths.max())
            dim = chunk[0].shape[1]
            x = torch.zeros(len(chunk), max_len, dim)
            for b, s in enumerate(chunk):
                x[b, :s.shape[0]] = torch.from_numpy(s)
            _, hidden = model(x.to(device), lengths.to(device), return_hidden=True)
            mask = (torch.arange(max_len, device=device).unsqueeze(0)
                    < lengths.to(device).unsqueeze(1)).float()
            lf = lengths.to(device).unsqueeze(1).float()
            for stage in ("proj", "layer0", "layer1"):
                h = hidden[stage]
                out[stage].append(((h * mask.unsqueeze(-1)).sum(1) / lf).cpu().numpy())
            out["pooled"].append(hidden["pooled"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def probe(Xtr, ytr, Xte, yte, control=False, n_jobs=2):
    """Logistic-regression probe; ``control`` shuffles the training labels."""
    if control:
        ytr = np.random.RandomState(0).permutation(ytr)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=n_jobs)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def encode(ytr, yte):
    """Label-encode on the union so an unseen test label cannot crash the probe."""
    le = LabelEncoder().fit(np.concatenate([ytr, yte]))
    return le.transform(ytr), le.transform(yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tag", required=True, help="checkpoint filename prefix")
    ap.add_argument("--view", default="ema", choices=["ema", "acoustic", "full"])
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--native-only", action="store_true")
    ap.add_argument("--drop-y", action="store_true")
    ap.add_argument("--props", default="onset,final,tone")
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="logistic-regression threads (keep low while trainers run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    props = [p.strip() for p in args.props.split(",") if p.strip()]

    df = pd.read_csv(args.features)
    df = df.dropna(subset=["syllable"]).copy()
    if args.native_only:
        df = df[df["group"] == "native"]
    meta_cols = {"source", "speaker", "group", "sex", "l1", "item", "section",
                 "syllable", "tone", "time"}
    feat_cols = sc.select_cols([c for c in df.columns if c not in meta_cols],
                               args.view, args.drop_y)

    X, syl, tone, section, spk, sex = [], [], [], [], [], []
    for _, g in df.groupby("source", sort=False):
        g = g.sort_values("time")
        X.append(g[feat_cols].to_numpy(dtype=np.float32))
        syl.append(g["syllable"].iloc[0])
        tone.append(g["tone"].iloc[0])
        section.append(g["section"].iloc[0])
        spk.append(g["speaker"].iloc[0]); sex.append(g["sex"].iloc[0])
    syl = np.array(syl, dtype=object); tone = np.array(tone, dtype=object)
    section = np.array(section, dtype=object)
    spk = np.array(spk); sex = np.array(sex)
    lengths = np.array([len(x) for x in X])
    fold_of = sc.make_folds(spk, sex, args.folds)

    # property label vectors
    labels = {}
    if "onset" in props:
        labels["onset"] = np.array([split_syllable(s)[0] or "-" for s in syl], dtype=object)
    if "final" in props:
        labels["final"] = np.array([split_syllable(s)[1] for s in syl], dtype=object)
    if "tone" in props:
        labels["tone"] = tone

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.tag + "_f*_s*.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints matching {args.tag}_f*_s*.pt in {args.ckpt_dir}")
    print(f"features={len(feat_cols)} utterances={len(X)} checkpoints={len(ckpts)} "
          f"device={sc.DEV}")
    for p, v in labels.items():
        print(f"  {p}: {len(set(v))} distinct labels")

    per_run = []
    for path in ckpts:
        m = re.search(r"_f(\d+)_s(\d+)\.pt$", path)
        fold, seed = int(m.group(1)), int(m.group(2))
        ck = torch.load(path, map_location="cpu")
        model = SyllableBiLSTM(ck["input_dim"], ck["num_classes"])
        model.load_state_dict(ck["model"])
        model.to(sc.DEV).eval()

        te = np.array([fold_of[s] == fold for s in spk])
        rest = ~te
        va = np.array([fold_of[s] == (fold + 1) % args.folds for s in spk]) & rest
        tr = rest & ~va

        mu = np.concatenate([X[i] for i in np.where(tr)[0]], axis=0).mean(0)
        sd = np.concatenate([X[i] for i in np.where(tr)[0]], axis=0).std(0) + 1e-9
        Xs = [(x - mu) / sd for x in X]

        tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        R_tr = extract(model, [Xs[i] for i in tr_idx], sc.DEV)
        R_te = extract(model, [Xs[i] for i in te_idx], sc.DEV)

        run = {"fold": fold, "seed": seed}
        for prop, vals in labels.items():
            y_tr_all, y_te_all = vals[tr_idx], vals[te_idx]
            if prop == "tone":  # balanced 42x4 subset only
                keep_tr = np.array([section[i] == TONE_SECTION for i in tr_idx])
                keep_te = np.array([section[i] == TONE_SECTION for i in te_idx])
                y_tr_all, y_te_all = y_tr_all[keep_tr], y_te_all[keep_te]
                idx_tr = np.where(keep_tr)[0]; idx_te = np.where(keep_te)[0]
            else:
                idx_tr = np.arange(len(tr_idx)); idx_te = np.arange(len(te_idx))
            y_tr, y_te = encode(y_tr_all, y_te_all)
            for stage in STAGES:
                Xp_tr, Xp_te = R_tr[stage][idx_tr], R_te[stage][idx_te]
                acc = probe(Xp_tr, y_tr, Xp_te, y_te, n_jobs=args.n_jobs)
                ctrl = probe(Xp_tr, y_tr, Xp_te, y_te, control=True, n_jobs=args.n_jobs)
                run[f"{prop}__{stage}"] = {"acc": acc, "control": ctrl,
                                           "selectivity": acc - ctrl,
                                           "n_train": int(len(y_tr)),
                                           "n_test": int(len(y_te)),
                                           "n_classes": int(len(set(y_tr_all)))}
        per_run.append(run)
        print(f"  fold {fold} seed {seed}: " + " ".join(
            f"{p}={100*run[f'{p}__pooled']['acc']:.1f}/sel{100*run[f'{p}__pooled']['selectivity']:.1f}"
            for p in labels))

    agg = {}
    for prop in labels:
        for stage in STAGES:
            k = f"{prop}__{stage}"
            accs = [r[k]["acc"] for r in per_run]
            ctrls = [r[k]["control"] for r in per_run]
            agg[k] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                      "control_mean": float(np.mean(ctrls)),
                      "selectivity_mean": float(np.mean([a - c for a, c in zip(accs, ctrls)])),
                      "n_classes": per_run[0][k]["n_classes"]}

    summary = {"name": args.tag, "corpus": "SAIT-EMA", "scope": "native" if args.native_only else "all",
               "stages": STAGES, "props": list(labels), "n_features": len(feat_cols),
               "aggregate": agg, "runs": per_run}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n{args.tag}: selectivity (pooled) " + ", ".join(
        f"{p} {100*agg[f'{p}__pooled']['selectivity_mean']:.1f}" for p in labels))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
