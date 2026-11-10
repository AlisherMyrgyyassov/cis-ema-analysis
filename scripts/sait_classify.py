"""SAIT-EMA subject-disjoint syllable/onset/vowel classification for the
normalization study, using the paper's canonical BiLSTM.

Reuses `src.models.bilstm.SyllableBiLSTM`. Runs sex-stratified subject-disjoint
folds (6 by default) on the monosyllable feature table; the feature table already
encodes the coordinate representation, so pass different `--features` files built
with `build_features.py --norm {raw,center_only,robust_z}` to compare.

Usage:
  python scripts/sait_classify.py --features <features_mono.csv> --view ema \
      --task syllable --folds 6 --seeds 1 --native-only --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.bilstm import SyllableBiLSTM  # noqa: E402
from preprocessing.sait_ema.reader import split_syllable  # noqa: E402

ACO_PREFIX = ("F0", "F1", "F2", "F3", "MFCC", "energy", "voiced")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


GEOM_DEPS = {
    "TT_TD_distance": ("TT", "TD"),
    "UL_LL_distance": ("UL", "LL"),
    "TT_LL_distance": ("TT", "LL"),
    "TT_TD_LL_angle": ("TT", "TD", "LL"),
    "UL_LL_LJ_angle": ("UL", "LL", "LJ"),
}


def select_cols(cols, view, drop_y=False, drop_coils=()):
    aco = [c for c in cols if c.startswith(ACO_PREFIX)]
    ema = [c for c in cols if c not in aco]
    if drop_y:
        ema = [c for c in ema if not re.match(r"^[A-Z]{2}y(_velocity|_acceleration)?$", c)]
    if drop_coils:
        # Drop a coil's x/y/z channels AND every geometry term that involves it --
        # the same rule scripts/run_montage_subset.py uses for Bruce, so the two
        # corpora are compared on the same manipulation.
        keep = []
        for c in ema:
            if re.match(r"^[A-Z]{2}[xyz](_velocity|_acceleration)?$", c):
                if c[:2] not in drop_coils:
                    keep.append(c)
            elif c in GEOM_DEPS and any(d in drop_coils for d in GEOM_DEPS[c]):
                continue
            else:
                keep.append(c)
        ema = keep
    if view == "ema":
        return ema
    if view == "acoustic":
        return aco
    return ema + aco


def task_label(syl, task):
    if task == "onset":
        return split_syllable(syl)[0] or "-"
    if task == "vowel":
        return split_syllable(syl)[1]
    return syl


def make_folds(speakers, sex, n_folds):
    """Sex-stratified subject-disjoint folds (deterministic round-robin)."""
    order = []
    for s in ("F", "M"):
        sp = sorted([spk for spk, x in zip(speakers, sex) if x == s])
        order += [sp[i % n_folds] for i in range(len(sp))]  # placeholder
    # assign speaker -> fold: round-robin within sex
    fold_of = {}
    for s in ("F", "M"):
        sp = sorted(set(spk for spk, x in zip(speakers, sex) if x == s))
        for i, spk in enumerate(sp):
            fold_of[spk] = i % n_folds
    return fold_of


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, lengths):
        self.X, self.y, self.lengths = X, y, lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.lengths[i]


def collate(batch):
    xs, ys, ls = zip(*batch)
    T = max(len(x) for x in xs)
    D = xs[0].shape[1]
    out = torch.zeros(len(xs), T, D)
    for i, x in enumerate(xs):
        out[i, :len(x)] = torch.as_tensor(x)
    return out, torch.tensor(ys, dtype=torch.long), torch.tensor(ls, dtype=torch.long)


def run_fold(Xtr, ytr, ltr, Xva, yva, lva, Xte, yte, lte, n_cls, seed, epochs,
             return_state=False, return_preds=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SyllableBiLSTM(Xtr[0].shape[1], n_cls).to(DEV)
    # class weights
    cnt = np.bincount(ytr, minlength=n_cls).astype(float)
    w = torch.tensor((cnt.sum() / (cnt + 1e-6)), dtype=torch.float32, device=DEV)
    w = w / w.mean()
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    tr = torch.utils.data.DataLoader(SeqDataset(Xtr, ytr, ltr), batch_size=32,
                                     shuffle=True, collate_fn=collate)
    va = torch.utils.data.DataLoader(SeqDataset(Xva, yva, lva), batch_size=64, collate_fn=collate)
    te = torch.utils.data.DataLoader(SeqDataset(Xte, yte, lte), batch_size=64, collate_fn=collate)

    def evaluate(loader):
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for x, y, l in loader:
                x = x.to(DEV); l = l.to(DEV)
                preds.append(model(x, l).argmax(1).cpu()); trues.append(y)
        preds = torch.cat(preds).numpy(); trues = torch.cat(trues).numpy()
        return (float((preds == trues).mean()),
                f1_score(trues, preds, average="macro", zero_division=0), preds, trues)

    best, best_state, bad = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y, l in tr:
            x = x.to(DEV); y = y.to(DEV); l = l.to(DEV)
            opt.zero_grad()
            loss = lossf(model(x, l), y)
            loss.backward(); opt.step()
        acc = evaluate(va)[0]
        sched.step(1 - acc)
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    acc, f1, preds, trues = evaluate(te)
    extras = {}
    if return_state:
        extras["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if return_preds:
        extras["preds"], extras["trues"] = preds, trues
    return acc, f1, best, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--view", default="ema", choices=["ema", "acoustic", "full"])
    ap.add_argument("--task", default="syllable", choices=["syllable", "onset", "vowel"])
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--only-folds", default=None,
                    help="comma list of fold indices to run (default: all); the "
                         "fold structure is still built with --folds.")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index; with --seeds N runs seeds "
                         "[seed-start, N). Lets extra seeds be appended to an "
                         "existing run without redoing seed 0.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--native-only", action="store_true")
    ap.add_argument("--sections", default=None, help="comma-separated section filter")
    ap.add_argument("--drop-y", action="store_true", help="drop the medial-lateral (y) channels")
    ap.add_argument("--drop-coils", default="",
                    help="comma-separated coils whose channels to remove, e.g. TD,LJ")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save one checkpoint per fold/seed (for probing)")
    ap.add_argument("--save-preds", action="store_true",
                    help="save per-utterance predictions per fold/seed")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = df.dropna(subset=["syllable"]).copy()
    if args.sections:
        df = df[df["section"].isin(args.sections.split(","))].copy()
    if args.native_only:
        df = df[df["group"] == "native"]
    meta_cols = {"source", "speaker", "group", "sex", "l1", "item", "section",
                 "syllable", "tone", "time"}
    feat_cols = select_cols([c for c in df.columns if c not in meta_cols], args.view,
                            args.drop_y,
                            drop_coils=tuple(s for s in args.drop_coils.split(",") if s))

    # group into sequences
    X, y, spk, sex, sources = [], [], [], [], []
    classes = {}
    for src, g in df.groupby("source", sort=False):
        g = g.sort_values("time")
        X.append(g[feat_cols].to_numpy(dtype=np.float32))
        lab = task_label(g["syllable"].iloc[0], args.task)
        if lab not in classes:
            classes[lab] = len(classes)
        y.append(classes[lab])
        spk.append(g["speaker"].iloc[0]); sex.append(g["sex"].iloc[0])
        sources.append(src)
    y = np.array(y); spk = np.array(spk); sex = np.array(sex)
    lengths = np.array([len(x) for x in X])
    n_cls = len(classes)
    print(f"task={args.task} view={args.view} utt={len(X)} classes={n_cls} features={len(feat_cols)} dev={DEV}")

    fold_of = make_folds(spk, sex, args.folds)
    results = []
    per_spk = {}
    ckpt_dir = preds_dir = None
    if args.save_ckpt or args.save_preds:
        base = os.path.dirname(os.path.abspath(args.out)) if args.out else os.getcwd()
        if args.save_ckpt:
            ckpt_dir = os.path.join(base, "ckpt"); os.makedirs(ckpt_dir, exist_ok=True)
        if args.save_preds:
            preds_dir = os.path.join(base, "preds"); os.makedirs(preds_dir, exist_ok=True)
    for seed in range(args.seed_start, args.seeds):
        fold_ids = ([int(x) for x in args.only_folds.split(",")]
                    if args.only_folds else list(range(args.folds)))
        for f in fold_ids:
            te = np.array([fold_of[s] == f for s in spk])
            rest = ~te
            # validation: rotate one fold out of the rest
            va = np.array([fold_of[s] == (f + 1) % args.folds for s in spk]) & rest
            tr = rest & ~va

            def fit_stats(sub):
                allf = np.concatenate([X[i] for i in range(len(X)) if sub[i]], axis=0)
                mu, sd = allf.mean(0), allf.std(0) + 1e-9
                return mu, sd
            mu, sd = fit_stats(tr)
            Xs = [(x - mu) / sd for x in X]

            acc, f1, vaacc, extras = run_fold(
                [Xs[i] for i in np.where(tr)[0]], y[tr], lengths[tr],
                [Xs[i] for i in np.where(va)[0]], y[va], lengths[va],
                [Xs[i] for i in np.where(te)[0]], y[te], lengths[te],
                n_cls, seed * 100 + f, args.epochs,
                return_state=args.save_ckpt, return_preds=args.save_preds)
            stem = f"{args.tag or args.task}_f{f}_s{seed}"
            if ckpt_dir:
                torch.save({"model": extras["state"], "input_dim": X[0].shape[1],
                            "num_classes": n_cls, "classes": list(classes),
                            "fold": f, "seed": seed, "task": args.task,
                            "drop_y": bool(args.drop_y), "features": args.features,
                            "scope": "native" if args.native_only else "all"},
                           os.path.join(ckpt_dir, stem + ".pt"))
            if preds_dir:
                te_idx = np.where(te)[0]
                np.savez(os.path.join(preds_dir, stem + ".npz"),
                         source=np.array([str(sources[i]) for i in te_idx]),
                         true=extras["trues"], pred=extras["preds"])
            maj = int(np.bincount(y[tr], minlength=n_cls).argmax())
            maj_acc = float((y[te] == maj).mean())
            results.append({"fold": f, "seed": seed, "test_acc": round(100 * acc, 2),
                            "test_f1": round(100 * f1, 2), "maj_acc": round(100 * maj_acc, 2),
                            "val_acc": round(100 * vaacc, 2), "n_test": int(te.sum())})
            print(f"  fold {f} seed {seed}: test {100*acc:.1f} f1 {100*f1:.1f} "
                  f"maj {100*maj_acc:.1f} val {100*vaacc:.1f}")

    accs = [r["test_acc"] for r in results]
    summary = {"tag": args.tag, "task": args.task, "view": args.view,
               "scope": "native" if args.native_only else "all",
               "n_classes": n_cls, "n_features": len(feat_cols),
               "acc_mean": round(float(np.mean(accs)), 2),
               "acc_std": round(float(np.std(accs)), 2),
               "f1_mean": round(float(np.mean([r["test_f1"] for r in results])), 2),
               "maj_mean": round(float(np.mean([r["maj_acc"] for r in results])), 2),
               "folds": results}
    print(f"\n{args.tag} {args.task}/{args.view}: {summary['acc_mean']} +- {summary['acc_std']} "
          f"(f1 {summary['f1_mean']}, maj {summary['maj_mean']})")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
