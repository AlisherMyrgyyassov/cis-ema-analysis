"""Experiment 2: SAGE feature-group Shapley decomposition.

Computes the game-theoretically fair, order-independent contribution of each of
the 8 feature groups (G1-G8) to syllable classification, using *inference-time
feature masking* on the frozen full (97-feature) model trained in Exp 1.4.

Because inference-time masking is cheap, we evaluate **all 2^8 = 256 group
subsets** (zeroing excluded groups = mean-imputation after z-scoring) and compute
the *exact* Shapley value per group from the subset values, removing Monte-Carlo
sampling variance entirely. The additivity property
(sum of Shapley = v(full) - v(empty)) is checked explicitly.
"""
import math
import os
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import (FEATURE_GROUPS, GROUP_IDS, ALL_COLUMNS, RUNS_DIR)
from src.models.bilstm import SyllableBiLSTM


def _group_columns(groups, n_features):
    """Map group -> column positions within a canonical full-feature vector."""
    return {g: [ALL_COLUMNS.index(c) for c in FEATURE_GROUPS[g]["columns"]]
            for g in groups}


def eval_masked(model, seqs, labels, mask, device, batch_size=256):
    """Accuracy + macro-F1 with only the ``mask``-selected columns present.

    The input always keeps the full feature dimension (masked columns are
    zeroed = mean-imputed), so the empty subset is a valid (all-zero) input.
    """
    model.eval()
    preds, truths = [], []
    F = seqs[0].shape[1]
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            lengths = torch.tensor([s.shape[0] for s in chunk], dtype=torch.long)
            max_len = int(lengths.max())
            x = torch.zeros(len(chunk), max_len, F)
            for b, s in enumerate(chunk):
                x[b, :s.shape[0], mask] = torch.from_numpy(s[:, mask])
            logits = model(x.to(device), lengths.to(device))
            preds.append(logits.argmax(1).cpu())
            truths.extend(labels[i:i + batch_size])
    preds = torch.cat(preds).numpy()
    truths = np.asarray(truths)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro", zero_division=0)
    return acc, f1


def _shapley_weights(n):
    fact = [math.factorial(k) for k in range(n + 1)]
    return [fact[s] * fact[n - 1 - s] / fact[n] for s in range(n)]


def shapley_for_run(model, seqs, labels, device, groups=GROUP_IDS):
    """Exact Shapley values from all 2^n subset evaluations."""
    n = len(groups)
    F = seqs[0].shape[1]
    group_idx = _group_columns(groups, F)

    values_acc = np.zeros(2 ** n)
    values_f1 = np.zeros(2 ** n)
    for s in range(2 ** n):
        mask = np.ones(F, dtype=bool)
        for i, g in enumerate(groups):
            if not (s >> i) & 1:
                mask[group_idx[g]] = False
        acc, f1 = eval_masked(model, seqs, labels, mask, device)
        values_acc[s] = acc
        values_f1[s] = f1

    weights = _shapley_weights(n)
    shap_acc = np.zeros(n)
    shap_f1 = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        for s in range(2 ** n):
            if s & bit:
                continue
            w = weights[bin(s).count("1")]
            shap_acc[i] += w * (values_acc[s | bit] - values_acc[s])
            shap_f1[i] += w * (values_f1[s | bit] - values_f1[s])

    return {
        "groups": groups,
        "shapley_acc": {g: float(shap_acc[i]) for i, g in enumerate(groups)},
        "shapley_f1": {g: float(shap_f1[i]) for i, g in enumerate(groups)},
        "full_acc": float(values_acc[-1]),
        "empty_acc": float(values_acc[0]),
        "full_f1": float(values_f1[-1]),
        "empty_f1": float(values_f1[0]),
        "additivity_acc": float(shap_acc.sum()),
        "additivity_residual_acc": float(shap_acc.sum() - (values_acc[-1] - values_acc[0])),
        "additivity_residual_f1": float(shap_f1.sum() - (values_f1[-1] - values_f1[0])),
        "subset_values_acc": values_acc.tolist(),
    }


def run(records, col_index, folds, full_results, out_dir=None, device="cuda",
        num_layers=2, hidden=128, dropout=0.4):
    """Compute per-fold/seed Shapley values for the Exp-1.4 full models."""
    device = torch.device(device)
    out_dir = out_dir or os.path.join(RUNS_DIR, "exp2")
    os.makedirs(out_dir, exist_ok=True)

    from src.data.dataset import records_for, view, fit_stats, standardize
    from src.config import GROUP_IDS as GI

    per_run = []
    for r in full_results:
        fold_id, seed = r["fold"], r["seed"]
        fd = folds[fold_id]
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])

        tr_seqs = view(tr, col_index, GI)
        te_seqs = view(te, col_index, GI)
        mu, sd = fit_stats(tr_seqs)
        te_std = standardize(te_seqs, mu, sd)
        te_labels = [rec["syllable"] for rec in te]

        # labels as ints via the checkpoint's class list
        ckpt = torch.load(r["ckpt_path"], map_location="cpu")
        classes = ckpt["classes"]
        cls2i = {c: i for i, c in enumerate(classes)}
        te_labels_int = [cls2i[lb] for lb in te_labels]

        model = SyllableBiLSTM(ckpt["input_dim"], ckpt["num_classes"],
                               num_layers=num_layers, hidden=hidden,
                               dropout=dropout)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        res = shapley_for_run(model, te_std, te_labels_int, device)
        res["fold"] = fold_id
        res["seed"] = seed
        per_run.append(res)
        print(f"[fold {fold_id} seed {seed}] additivity resid acc "
              f"{res['additivity_residual_acc']:.4f}  f1 {res['additivity_residual_f1']:.4f}")

    # aggregate across runs
    agg = {}
    for metric in ("shapley_acc", "shapley_f1"):
        per_group = {}
        for g in GI:
            vals = [r[metric][g] for r in per_run]
            per_group[g] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                            "values": vals}
        agg[metric] = per_group

    summary = {
        "groups": GI,
        "names": {g: FEATURE_GROUPS[g]["name"] for g in GI},
        "kind": {g: FEATURE_GROUPS[g]["kind"] for g in GI},
        "aggregate": agg,
        "runs": per_run,
    }
    with open(os.path.join(out_dir, "shapley.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
