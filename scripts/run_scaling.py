"""Capacity and training-set-size curves for the EMA-only BiLSTM (CIS-EMA).

Two sweeps under the locked subject-disjoint 6-fold protocol, EMA-only (59-D),
main representation ``variants/robust_z``:

  capacity : hidden in {64, 128, 256}, all training subjects
  speakers : k in {6, 12, 18, all} training speakers, hidden=128

For ``speakers`` the test and validation subjects of each fold are held fixed
and only the training subjects are subsampled, sex-balanced and deterministic
(same selection for every fold given ``--subj-seed``). Accuracy is always
computed over all 43 syllable classes (fixed label space), so a class absent
from the training subsample is scored as an error; the number of classes
covered by the training subsample is recorded next to every run.

Usage:
  python scripts/run_scaling.py --mode both --seeds 0
"""
import argparse
import json
import os
import sys

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import EMA_GROUPS, N_SYLLABLES               # noqa: E402
from src.pipeline import prepare_data                        # noqa: E402
from src.data.dataset import view, fit_stats, standardize, records_for  # noqa: E402
from src.eval import classification_metrics                   # noqa: E402
from src.training import make_loader, train_model, evaluate_syl  # noqa: E402
from src.models.bilstm import SyllableBiLSTM                  # noqa: E402

DATA = str(path("cis_dataset"))


def pick_subjects(train, sex_map, k, seed=0):
    """Deterministic sex-balanced subset of ``k`` training subjects."""
    train = sorted(train)
    if k >= len(train):
        return train
    rng = np.random.RandomState(seed)
    by_sex = {"F": [s for s in train if sex_map.get(s) == "F"],
              "M": [s for s in train if sex_map.get(s) != "F"]}
    for v in by_sex.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < k:
        for sex in ("F", "M"):
            if len(picked) < k and by_sex[sex]:
                picked.append(by_sex[sex].pop(0))
    return sorted(picked)


def run(records, col_index, folds, sex_map, label_enc, seeds, hidden, k_subj,
        subj_seed, device, epochs):
    out = []
    for fold_id, fd in enumerate(folds):
        tr_subj = pick_subjects(fd["train"], sex_map, k_subj, subj_seed)
        tr = records_for(records, tr_subj)
        va = records_for(records, fd["val"])
        te = records_for(records, fd["test"])
        y_tr = label_enc.transform([r["syllable"] for r in tr])
        y_va = label_enc.transform([r["syllable"] for r in va])
        y_te = label_enc.transform([r["syllable"] for r in te])
        tr_s = view(tr, col_index, EMA_GROUPS)
        va_s = view(va, col_index, EMA_GROUPS)
        te_s = view(te, col_index, EMA_GROUPS)
        mu, sd = fit_stats(tr_s)
        tr_s = standardize(tr_s, mu, sd)
        va_s = standardize(va_s, mu, sd)
        te_s = standardize(te_s, mu, sd)
        cw = np.bincount(y_tr, minlength=N_SYLLABLES).astype(np.float32)
        cw = torch.tensor(cw.sum() / (N_SYLLABLES * np.maximum(cw, 1e-6)),
                          dtype=torch.float32).to(device)
        covered = int(len(np.unique(y_tr)))
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = SyllableBiLSTM(tr_s[0].shape[1], N_SYLLABLES, hidden=hidden).to(device)
            model, best_val = train_model(model, make_loader(tr_s, y_tr, 32, True),
                                          make_loader(va_s, y_va, 32, False), device,
                                          epochs=epochs, class_weight=cw, verbose=False)
            p, t = evaluate_syl(model, make_loader(te_s, y_te, 32, False), device)
            m = classification_metrics(t, p, classes=list(label_enc.classes_))
            out.append({"fold": fold_id, "seed": seed,
                        "hidden": hidden, "n_train_subj": len(tr_subj),
                        "n_train_utt": len(tr), "n_classes_covered": covered,
                        "val_acc": float(best_val),
                        "test_acc": float(m["accuracy"]),
                        "test_f1": float(m["macro_f1"])})
            print(f"  [{fold_id}/{seed}] subj {len(tr_subj):2d} cls {covered:2d} "
                  f"val {best_val:.4f} test {m['accuracy']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["cap", "spk", "both"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", default="robust_z")
    ap.add_argument("--subj-seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="debug: run only the first N folds (0 = all)")
    ap.add_argument("--suffix", default="",
                    help="appended to each output filename, e.g. _s1 when running "
                         "one extra seed into separate files for later merging")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "scaling"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    records, col_index, _, sex_map, folds = prepare_data(
        os.path.join(DATA, "variants", args.variant))
    if args.max_folds:
        folds = folds[:args.max_folds]
    # Fixed 43-class label space: the class inventory is part of the task
    # definition, and it keeps folds comparable when training subjects are
    # subsampled below full coverage.
    label_enc = LabelEncoder().fit(sorted({r["syllable"] for r in records}))
    n_train_full = len(folds[0]["train"])

    configs = []
    if args.mode in ("cap", "both"):
        for h in (64, 128, 256):
            configs.append((f"cap_h{h}", h, n_train_full))
    if args.mode in ("spk", "both"):
        for k in (6, 12, 18, n_train_full):
            configs.append((f"spk_k{k:02d}", 128, k))

    for name, hidden, k in configs:
        p = os.path.join(args.out, f"{name}{args.suffix}.json")
        if os.path.exists(p):
            print(f"skip {name}"); continue
        print(f"=== {name} (hidden={hidden}, train_subj={k}) ===", flush=True)
        res = run(records, col_index, folds, sex_map, label_enc, seeds, hidden, k,
                  args.subj_seed, device, args.epochs)
        acc = np.mean([r["test_acc"] for r in res]) * 100
        print(f"{name}: {acc:.2f} +- {np.std([r['test_acc'] for r in res]) * 100:.2f}", flush=True)
        with open(p, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
