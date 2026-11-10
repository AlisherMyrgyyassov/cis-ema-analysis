"""Evaluation metrics and aggregation for the paper pipeline.

All "mean +- std" numbers are computed across the fold x seed runs (12 per
config), which is what enables paired fold-wise statistical comparisons.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(truths, preds, classes=None):
    """Accuracy, macro-F1, per-class recall, and confusion matrix."""
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro", zero_division=0)
    cm = confusion_matrix(truths, preds)
    per_class_recall = {}
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        per_class_recall[classes[i] if classes is not None else i] = (
            cm[i, i] / denom if denom > 0 else 0.0)
    return {"accuracy": acc, "macro_f1": f1,
            "per_class_recall": per_class_recall, "confusion": cm}


def aggregate_runs(results, key):
    """Mean +- std of ``key`` across fold x seed runs."""
    vals = np.array([r[key] for r in results], dtype=float)
    return float(vals.mean()), float(vals.std()), vals.tolist()


def summarize(results, keys=("test_acc", "test_f1")):
    out = {}
    for k in keys:
        m, s, _ = aggregate_runs(results, k)
        out[k + "_mean"] = m
        out[k + "_std"] = s
    return out
