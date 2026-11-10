"""Experiment 3b: faithfulness validation of the Exp-2 ranking.

Deletion curves (progressively remove groups in Shapley order, reverse order,
and a fixed random order) and an insertion curve (start from EMA-only, add
groups back in Shapley order), all on the frozen full model with group masking.

If the Shapley-order deletion curve drops fastest and the reverse-order curve
drops slowest, the importance ranking is causally meaningful (faithfulness).
"""
import os
import json

import numpy as np
import torch

from src.config import (FEATURE_GROUPS, GROUP_IDS, EMA_GROUPS, RUNS_DIR)
from src.models.bilstm import SyllableBiLSTM
from src.exp2_shapley import _group_columns, eval_masked


def _rank_order(shapley_summary, key="shapley_acc", ascending=False):
    agg = shapley_summary["aggregate"][key]
    groups = sorted(GROUP_IDS, key=lambda g: agg[g]["mean"], reverse=not ascending)
    return groups


def _deletion_curve(model, seqs, labels, device, order):
    """Accuracy after removing groups one by one in ``order``."""
    n = len(order)
    F = seqs[0].shape[1]
    group_idx = _group_columns(GROUP_IDS, F)
    removed = set()
    accs = []
    mask = np.ones(F, dtype=bool)
    acc, _ = eval_masked(model, seqs, labels, mask, device)
    accs.append(acc)  # nothing removed
    for g in order:
        removed.add(g)
        mask = np.ones(F, dtype=bool)
        for gr in GROUP_IDS:
            if gr in removed:
                mask[group_idx[gr]] = False
        acc, _ = eval_masked(model, seqs, labels, mask, device)
        accs.append(acc)
    return np.array(accs)


def _insertion_curve(model, seqs, labels, device, order):
    """Accuracy starting from EMA-only, adding groups in ``order``."""
    F = seqs[0].shape[1]
    group_idx = _group_columns(GROUP_IDS, F)
    present = set(EMA_GROUPS)
    mask = np.ones(F, dtype=bool)
    for gr in GROUP_IDS:
        if gr not in present:
            mask[group_idx[gr]] = False
    acc, _ = eval_masked(model, seqs, labels, mask, device)
    accs = [acc]
    for g in order:
        if g in EMA_GROUPS:
            continue
        present.add(g)
        mask = np.ones(F, dtype=bool)
        for gr in GROUP_IDS:
            if gr not in present:
                mask[group_idx[gr]] = False
        acc, _ = eval_masked(model, seqs, labels, mask, device)
        accs.append(acc)
    return np.array(accs)


def run(records, col_index, folds, full_results, shapley_summary, out_dir=None,
        device="cuda", num_layers=2, hidden=128, dropout=0.4):
    device = torch.device(device)
    out_dir = out_dir or os.path.join(RUNS_DIR, "exp3")
    os.makedirs(out_dir, exist_ok=True)

    from src.data.dataset import records_for, view, fit_stats, standardize
    from src.config import GROUP_IDS as GI

    shap_order = _rank_order(shapley_summary, "shapley_acc")
    rev_order = list(reversed(shap_order))
    rng = np.random.RandomState(123)
    rand_order = list(shap_order)
    rng.shuffle(rand_order)

    orders = {"shapley": shap_order, "reverse": rev_order, "random": rand_order}
    deletion = {k: [] for k in orders}
    insertion = []

    for r in full_results:
        fold_id, seed = r["fold"], r["seed"]
        fd = folds[fold_id]
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])

        tr_seqs = view(tr, col_index, GI)
        te_seqs = view(te, col_index, GI)
        mu, sd = fit_stats(tr_seqs)
        te_std = standardize(te_seqs, mu, sd)

        ckpt = torch.load(r["ckpt_path"], map_location="cpu")
        classes = ckpt["classes"]
        cls2i = {c: i for i, c in enumerate(classes)}
        te_labels = [cls2i[rec["syllable"]] for rec in te]

        model = SyllableBiLSTM(ckpt["input_dim"], ckpt["num_classes"],
                               num_layers=num_layers, hidden=hidden, dropout=dropout)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        for name, order in orders.items():
            deletion[name].append(_deletion_curve(model, te_std, te_labels, device, order))
        insertion.append(_insertion_curve(model, te_std, te_labels, device, shap_order))

        print(f"[fold {fold_id} seed {seed}] faithfulness curves done")

    summary = {
        "shapley_order": shap_order,
        "reverse_order": rev_order,
        "random_order": rand_order,
        "deletion": {k: {"mean": np.mean(np.stack(v), 0).tolist(),
                         "std": np.std(np.stack(v), 0).tolist()}
                     for k, v in deletion.items()},
        "insertion": {"mean": np.mean(np.stack(insertion), 0).tolist(),
                      "std": np.std(np.stack(insertion), 0).tolist()},
        "insertion_groups_added": [g for g in shap_order if g not in EMA_GROUPS],
    }
    with open(os.path.join(out_dir, "faithfulness.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
