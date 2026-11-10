"""Experiment 5 - oracle: is the *true* utterance-mean MFCC_2/5/6 target
class-informative beyond EMA-only?

Exp 5's auxiliary interventions can only help the EMA-only classifier if the
target they approximate carries syllable-discriminative information the EMA
channels do not already provide. This module tests that premise directly by
giving the model the *true* target values (from the microphone, i.e. an oracle
- never meant to be an SSI operating point):

  1. pooled oracle: EMA-only pooled features (mean/std/min/max) with and
     without the true utterance-mean MFCC_2/5/6 appended, fitted with logistic
     regression / random forest under the paper's CV (paired arms).
  2. bilstm oracle: the canonical BiLSTM on EMA-only features *plus* the true
     utterance-mean MFCC_2/5/6 broadcast as constant per-frame channels.

Reading the result:
  - oracle ~ 0     -> a flat Exp-5/5b accuracy is fully explained: the target
                      is redundant with what EMA already encodes, regardless of
                      whether the aux head learned it.
  - oracle > 0     -> the target carries signal EMA lacks; then the Exp-5 null
                      must be read as "the aux head failed to recover it"
                      (check its aux R^2) rather than as redundancy.
"""
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import (RUNS_DIR, DATA_DIR, N_FOLDS, N_VAL_SUBJECTS, FOLD_SEED,
                        SEEDS, EMA_GROUPS, columns_for)
from src.data.dataset import load_dataset, records_for, view, fit_stats, standardize, pooled
from src.data.cv import make_folds
from src.eval import classification_metrics
from src.pipeline import _labels
from src.training import make_loader, train_model, evaluate_syl
from src.models.bilstm import SyllableBiLSTM

AUX_COLS = ["MFCC_2", "MFCC_5", "MFCC_6"]


def _utt_means(records, col_index):
    idx = [col_index[c] for c in AUX_COLS]
    return {r["source"]: r["frame"][:, idx].astype(np.float64).mean(0)
            for r in records}


def _broadcast_seqs(records, means, col_index):
    ema_idx = [col_index[c] for c in columns_for(EMA_GROUPS)]
    out = []
    for r in records:
        ema = r["frame"][:, ema_idx].astype(np.float64)
        m = means[r["source"]]
        add = np.repeat(m[None, :], len(ema), axis=0)
        out.append(np.concatenate([ema, add], axis=1).astype(np.float32))
    return out


# --------------------------------------------------------------------------- #
# 1. pooled oracle (LR / RF)
# --------------------------------------------------------------------------- #
def pooled_oracle(records, col_index, folds, seeds):
    means = _utt_means(records, col_index)
    out = {"lr": [], "rf": [], "lr_mean": [], "rf_mean": []}
    for fold_id, fd in enumerate(folds):
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])
        le = LabelEncoder().fit([r["syllable"] for r in tr])
        y_tr = le.transform([r["syllable"] for r in tr])
        y_te = le.transform([r["syllable"] for r in te])

        X_tr = pooled(view(tr, col_index, EMA_GROUPS))
        X_te = pooled(view(te, col_index, EMA_GROUPS))
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6

        m_tr = np.stack([means[r["source"]] for r in tr])
        m_te = np.stack([means[r["source"]] for r in te])
        mmu, msd = m_tr.mean(0), m_tr.std(0) + 1e-6
        Z_tr = np.concatenate([(X_tr - mu) / sd, (m_tr - mmu) / msd], axis=1)
        Z_te = np.concatenate([(X_te - mu) / sd, (m_te - mmu) / msd], axis=1)
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        for seed in seeds:
            base = LogisticRegression(C=1.0, max_iter=2000)
            base.fit(X_tr, y_tr)
            m = classification_metrics(y_te, base.predict(X_te), classes=le.classes_.tolist())
            out["lr"].append({"fold": fold_id, "seed": seed,
                              "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
            plus = LogisticRegression(C=1.0, max_iter=2000)
            plus.fit(Z_tr, y_tr)
            m = classification_metrics(y_te, plus.predict(Z_te), classes=le.classes_.tolist())
            out["lr_mean"].append({"fold": fold_id, "seed": seed,
                                   "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
            base_rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
            base_rf.fit(X_tr, y_tr)
            m = classification_metrics(y_te, base_rf.predict(X_te), classes=le.classes_.tolist())
            out["rf"].append({"fold": fold_id, "seed": seed,
                              "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
            plus_rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
            plus_rf.fit(Z_tr, y_tr)
            m = classification_metrics(y_te, plus_rf.predict(Z_te), classes=le.classes_.tolist())
            out["rf_mean"].append({"fold": fold_id, "seed": seed,
                                   "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
    return out


# --------------------------------------------------------------------------- #
# 2. BiLSTM oracle (broadcast true means as extra input channels)
# --------------------------------------------------------------------------- #
def bilstm_oracle(records, col_index, folds, seeds, out_dir, device="cuda",
                  fold=None, epochs=80, verbose=True):
    device = torch.device(device)
    means = _utt_means(records, col_index)
    results = []
    for fold_id, fd in enumerate(folds):
        if fold is not None and fold_id != fold:
            continue
        tr = records_for(records, fd["train"])
        va = records_for(records, fd["val"])
        te = records_for(records, fd["test"])

        le, tr_y = _labels(tr)
        va_y = le.transform([r["syllable"] for r in va])
        te_y = le.transform([r["syllable"] for r in te])
        num_classes = len(le.classes_)

        tr_seqs = _broadcast_seqs(tr, means, col_index)
        va_seqs = _broadcast_seqs(va, means, col_index)
        te_seqs = _broadcast_seqs(te, means, col_index)
        mu, sd = fit_stats(tr_seqs)
        tr_std = standardize(tr_seqs, mu, sd)
        va_std = standardize(va_seqs, mu, sd)
        te_std = standardize(te_seqs, mu, sd)
        input_dim = tr_std[0].shape[1]

        counts = np.bincount(tr_y, minlength=num_classes).astype(np.float32)
        cw = torch.tensor(counts.sum() / (num_classes * counts)).to(device)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            tr_loader = make_loader(tr_std, tr_y, 32, True)
            va_loader = make_loader(va_std, va_y, 32, False)
            te_loader = make_loader(te_std, te_y, 32, False)
            model = SyllableBiLSTM(input_dim, num_classes).to(device)
            model, best_val = train_model(model, tr_loader, va_loader, device,
                                          epochs=epochs, class_weight=cw,
                                          verbose=verbose)
            te_preds, te_truths = evaluate_syl(model, te_loader, device)
            m = classification_metrics(te_truths, te_preds,
                                       classes=le.classes_.tolist())
            results.append({"fold": fold_id, "seed": seed,
                            "input_dim": input_dim,
                            "val_acc": float(best_val),
                            "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
            print(f"[oracle bilstm fold {fold_id} seed {seed}] "
                  f"val {best_val:.4f} test_acc {m['accuracy']:.4f}")
    return results


def summarize(results):
    accs = [r["test_acc"] for r in results]
    f1s = [r["test_f1"] for r in results]
    return {"test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std": float(np.std(f1s)),
            "n": len(accs)}


def run(data_dir=None, out_dir=None, device="cuda", pooled_only=False,
        seeds=SEEDS, fold=None, epochs=80):
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or os.path.join(RUNS_DIR, "exp5_oracle")
    os.makedirs(out_dir, exist_ok=True)

    records, col_index = load_dataset(data_dir)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)

    po = pooled_oracle(records, col_index, folds, list(seeds))
    summary = {
        "pooled_lr_ema_only": summarize(po["lr"]),
        "pooled_lr_ema_plus_means": summarize(po["lr_mean"]),
        "pooled_rf_ema_only": summarize(po["rf"]),
        "pooled_rf_ema_plus_means": summarize(po["rf_mean"]),
        "runs_pooled": po,
    }
    if not pooled_only:
        bo = bilstm_oracle(records, col_index, folds, list(seeds), out_dir,
                           device=device, fold=fold, epochs=epochs)
        summary["bilstm_ema_broadcast_means"] = summarize(bo)
        summary["runs_bilstm"] = bo

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== Exp 5 oracle (does the true target mean add signal?) ====")
    p = summary["pooled_lr_ema_only"]["test_acc_mean"]
    pm = summary["pooled_lr_ema_plus_means"]["test_acc_mean"]
    print(f"  pooled-LR EMA-only           {p*100:.2f} +- "
          f"{summary['pooled_lr_ema_only']['test_acc_std']*100:.2f}")
    print(f"  pooled-LR EMA + true means    {pm*100:.2f} +- "
          f"{summary['pooled_lr_ema_plus_means']['test_acc_std']*100:.2f}")
    r = summary["pooled_rf_ema_only"]["test_acc_mean"]
    rm = summary["pooled_rf_ema_plus_means"]["test_acc_mean"]
    print(f"  pooled-RF EMA-only           {r*100:.2f} +- "
          f"{summary['pooled_rf_ema_only']['test_acc_std']*100:.2f}")
    print(f"  pooled-RF EMA + true means    {rm*100:.2f} +- "
          f"{summary['pooled_rf_ema_plus_means']['test_acc_std']*100:.2f}")
    if "bilstm_ema_broadcast_means" in summary:
        b = summary["bilstm_ema_broadcast_means"]["test_acc_mean"]
        print(f"  BiLSTM EMA + broadcast means {b*100:.2f} +- "
              f"{summary['bilstm_ema_broadcast_means']['test_acc_std']*100:.2f}"
              f"  (vs EMA-only baseline 69.21)")
    return summary
