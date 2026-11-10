"""High-level orchestration shared by the experiments.

``train_biLSTM_cv`` trains the canonical BiLSTM over a chosen feature-group set
under the fixed subject-disjoint 6-fold CV (seeds x folds), saving per-run
checkpoints and returning fold-level results. Experiments 2 and 3 reload these
checkpoints so that Shapley masking and probing are applied to the *same*
trained full models.
"""
import json
import os

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.config import SEEDS, N_FOLDS, N_VAL_SUBJECTS, FOLD_SEED
from src.data.dataset import (load_dataset, records_for, view, fit_stats,
                              standardize)
from src.data.cv import make_folds
from src.eval import classification_metrics
from src.models.bilstm import SyllableBiLSTM
from src.training import make_loader, train_model, evaluate_syl


def prepare_data(data_dir):
    records, col_index = load_dataset(data_dir)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS, n_val=N_VAL_SUBJECTS,
                       seed=FOLD_SEED)
    return records, col_index, subjects, sex_map, folds


def _labels(recs):
    le = LabelEncoder()
    y = le.fit_transform([r["syllable"] for r in recs])
    return le, y


def train_biLSTM_cv(records, col_index, folds, groups, seeds, out_dir, name,
                    device="cuda", hidden=128, num_layers=2, dropout=0.4,
                    lr=3e-3, weight_decay=1e-4, epochs=80, patience=15,
                    batch_size=32, label_smoothing=0.0, class_weight=True,
                    fold=None, verbose=True):
    """Train/evaluate the BiLSTM on ``groups`` under subject CV.

    Returns list of per-run result dicts (one per fold x seed), each including
    the checkpoint path, test accuracy/F1 and per-class recall.
    """
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for fold_id, fd in enumerate(folds):
        if fold is not None and fold_id != fold:
            continue
        tr = records_for(records, fd["train"])
        va = records_for(records, fd["val"])
        te = records_for(records, fd["test"])

        le_syl, tr_y = _labels(tr)
        va_y = le_syl.transform([r["syllable"] for r in va])
        te_y = le_syl.transform([r["syllable"] for r in te])
        num_classes = len(le_syl.classes_)

        tr_seqs = view(tr, col_index, groups)
        va_seqs = view(va, col_index, groups)
        te_seqs = view(te, col_index, groups)

        mu, sd = fit_stats(tr_seqs)
        tr_std = standardize(tr_seqs, mu, sd)
        va_std = standardize(va_seqs, mu, sd)
        te_std = standardize(te_seqs, mu, sd)

        input_dim = tr_std[0].shape[1]

        cw = None
        if class_weight:
            counts = np.bincount(tr_y, minlength=num_classes).astype(np.float32)
            w = counts.sum() / (num_classes * counts)
            cw = torch.tensor(w, dtype=torch.float32).to(device)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            tr_loader = make_loader(tr_std, tr_y, batch_size, True)
            va_loader = make_loader(va_std, va_y, batch_size, False)
            te_loader = make_loader(te_std, te_y, batch_size, False)

            model = SyllableBiLSTM(input_dim, num_classes, hidden=hidden,
                                   num_layers=num_layers, dropout=dropout).to(device)
            model, best_val = train_model(
                model, tr_loader, va_loader, device, lr=lr,
                weight_decay=weight_decay, epochs=epochs, patience=patience,
                label_smoothing=label_smoothing, class_weight=cw, verbose=verbose)

            te_preds, te_truths = evaluate_syl(model, te_loader, device)
            m = classification_metrics(te_truths, te_preds,
                                       classes=le_syl.classes_.tolist())

            ckpt_dir = os.path.join(out_dir, name)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"fold{fold_id}_seed{seed}_model.pth")
            torch.save({
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "input_dim": input_dim, "num_classes": num_classes,
                "classes": le_syl.classes_.tolist(),
                "groups": groups, "fold": fold_id, "seed": seed,
            }, ckpt_path)

            results.append({
                "fold": fold_id, "seed": seed,
                "train_n": len(tr), "val_n": len(va), "test_n": len(te),
                "train_subj": fd["train"], "val_subj": fd["val"],
                "test_subj": fd["test"],
                "num_classes": num_classes,
                "val_acc": float(best_val),
                "test_acc": m["accuracy"],
                "test_f1": m["macro_f1"],
                "per_class_recall": m["per_class_recall"],
                "ckpt_path": ckpt_path,
            })
            if verbose:
                print(f"[fold {fold_id} seed {seed}] val {best_val:.4f} "
                      f"test_acc {m['accuracy']:.4f} test_f1 {m['macro_f1']:.4f}")

    return results


def save_summary(results, out_dir, name, extra=None):
    accs = [r["test_acc"] for r in results]
    f1s = [r["test_f1"] for r in results]
    summary = {
        "name": name,
        "n_runs": len(results),
        "test_acc_mean": float(np.mean(accs)),
        "test_acc_std": float(np.std(accs)),
        "test_f1_mean": float(np.mean(f1s)),
        "test_f1_std": float(np.std(f1s)),
        "extra": extra or {},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
