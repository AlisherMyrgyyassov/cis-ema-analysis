"""Experiment 1: honest ceilings.

Produces the Exp 1 master table:
  1.1 chance + majority
  1.2 pooled LogisticRegression / RandomForest (full 97-feature view)
  1.3 EMA-only BiLSTM (G1-G4)
  1.4 full hand-crafted BiLSTM (G1-G8)

The SSL audio ceiling (1.5) lives in ``src/exp1_ssl.py``.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import (RUNS_DIR, EMA_GROUPS, GROUP_IDS, SEEDS, N_FOLDS,
                        N_VAL_SUBJECTS, FOLD_SEED)
from src.data.dataset import (load_dataset, records_for, view, fit_stats,
                              standardize, pooled)
from src.data.cv import make_folds
from src.eval import classification_metrics, summarize
from src.pipeline import train_biLSTM_cv, save_summary


def chance_majority(records, folds):
    n_syl = len({r["syllable"] for r in records})
    chance = 1.0 / n_syl
    majors = []
    for fd in folds:
        te = records_for(records, fd["test"])
        counts = pd.Series([r["syllable"] for r in te]).value_counts()
        majors.append(counts.max() / counts.sum())
    return {"chance_acc": chance, "majority_acc_mean": float(np.mean(majors)),
            "majority_acc_std": float(np.std(majors))}


def pooled_baselines(records, col_index, folds, groups=GROUP_IDS):
    le = None
    results = {"lr": [], "rf": []}
    for fd in folds:
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])
        le = LabelEncoder().fit([r["syllable"] for r in tr])
        y_tr = le.transform([r["syllable"] for r in tr])
        y_te = le.transform([r["syllable"] for r in te])

        X_tr = pooled(view(tr, col_index, groups))
        X_te = pooled(view(te, col_index, groups))
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        for seed in SEEDS:
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(X_tr, y_tr)
            m = classification_metrics(y_te, lr.predict(X_te),
                                       classes=le.classes_.tolist())
            results["lr"].append({"fold": None, "seed": seed,
                                  "test_acc": m["accuracy"],
                                  "test_f1": m["macro_f1"]})

            rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                        random_state=seed)
            rf.fit(X_tr, y_tr)
            m = classification_metrics(y_te, rf.predict(X_te),
                                       classes=le.classes_.tolist())
            results["rf"].append({"fold": None, "seed": seed,
                                  "test_acc": m["accuracy"],
                                  "test_f1": m["macro_f1"]})
    return results


def run(data_dir, out_dir=None, device="cuda", fold=None, seeds=SEEDS,
       epochs=80, num_layers=2, hidden=128, verbose=True):
    records, col_index, subjects, sex_map, folds = prepare(data_dir)

    chance = chance_majority(records, folds)
    pooled_res = pooled_baselines(records, col_index, folds)

    out_dir = out_dir or os.path.join(RUNS_DIR, "exp1")
    os.makedirs(out_dir, exist_ok=True)

    ema_only = train_biLSTM_cv(records, col_index, folds, EMA_GROUPS, list(seeds),
                               out_dir, name="ema_only", device=device,
                               num_layers=num_layers, hidden=hidden,
                               epochs=epochs, fold=fold, verbose=verbose)
    full = train_biLSTM_cv(records, col_index, folds, GROUP_IDS, list(seeds),
                           out_dir, name="full", device=device,
                           num_layers=num_layers, hidden=hidden,
                           epochs=epochs, fold=fold, verbose=verbose)

    master = {
        "chance": chance,
        "logistic_regression": summarize(pooled_res["lr"]),
        "random_forest": summarize(pooled_res["rf"]),
        "bilstm_ema_only": summarize(ema_only),
        "bilstm_full": summarize(full),
    }
    return master, {"ema_only": ema_only, "full": full,
                    "pooled": pooled_res}


def prepare(data_dir):
    records, col_index = load_dataset(data_dir)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)
    return records, col_index, subjects, sex_map, folds
