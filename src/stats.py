"""Paired statistical tests used throughout the paper.

All pairwise comparisons are made on *fold-level paired differences* (same fold,
different model / feature set), never on independent samples. Returns both the
paired t-test and Wilcoxon signed-rank test, plus Cohen's d on the paired
differences as an effect size.
"""
import numpy as np
from scipy import stats


def paired_compare(a, b):
    """Compare two aligned arrays (e.g. fold-level accuracies).

    a, b: equal-length sequences of matched (paired) measurements.

    Returns a dict with paired t-test (t, p), Wilcoxon (W, p), and Cohen's d.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or len(a) < 2:
        return {"n": int(len(a)), "note": "fewer than 2 paired samples"}

    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="zsplit")
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))

    return {
        "n_pairs": int(len(a)),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_W": float(w_stat) if np.isfinite(w_stat) else None,
        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else None,
        "cohens_d": d,
    }
