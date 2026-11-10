"""Pairwise significance for a distillation student against the EMA-only baseline.

``scripts/exp6_subject_stats.py`` reports two different estimands: the subject-level
delta is computed from per-seed accuracies, while its fold-level delta is computed
from the 2-seed logit ensemble of each model. Those answer different questions
(average single model vs ensemble), so this script reports the estimand the paper
quotes: a paired fold-level test (n=6, per-seed values averaged within the fold,
matching ``src/analysis_stats.py``) plus the seed-matched run-level test (n=12).
No training and no checkpoint loading: it reads the saved run summaries only.

Usage::

    python scripts/exp6_paired_test.py
    python scripts/exp6_paired_test.py --student-dir runs/paper/exp6/ensC_a09_ck
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "paper")


def load_runs(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["runs"] if isinstance(data, dict) else data


def fold_means(runs):
    by_fold = {}
    for r in runs:
        by_fold.setdefault(r["fold"], []).append(r["test_acc"])
    return {f: float(np.mean(v)) for f, v in by_fold.items()}


def paired(a, b, label):
    """a = candidate, b = baseline; both arrays of paired accuracies."""
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        w = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:  # identical vectors
        w = 1.0
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return {
        "comparison": label, "n_pairs": int(len(a)),
        "candidate_mean": float(a.mean()), "baseline_mean": float(b.mean()),
        "mean_diff": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t": float(t), "paired_t_p": float(p),
        "wilcoxon_p": w, "cohens_d": d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-dir", default=os.path.join(RUNS, "exp6", "ensC_a09_ck"),
                    help="directory containing the student summary.json")
    ap.add_argument("--baseline", default=os.path.join(RUNS, "exp1", "ema_only_results.json"),
                    help="baseline results JSON (Exp-1 EMA-only runs)")
    ap.add_argument("--out", default=os.path.join(RUNS, "exp6", "paired_vs_baseline"))
    args = ap.parse_args()

    student = load_runs(os.path.join(args.student_dir, "summary.json"))
    baseline = load_runs(args.baseline)

    # (1) fold-level: per-seed values averaged within fold (paper protocol)
    fb, fc = fold_means(baseline), fold_means(student)
    keys = sorted(set(fb) & set(fc))
    fold = paired(np.array([fc[k] for k in keys]), np.array([fb[k] for k in keys]),
                  "student vs EMA-only, fold-level (n=6)")

    # (2) run-level: the same (fold, seed) pairing
    ba = {(r["fold"], r["seed"]): r["test_acc"] for r in baseline}
    ca = {(r["fold"], r["seed"]): r["test_acc"] for r in student}
    k2 = sorted(set(ba) & set(ca))
    run = paired(np.array([ca[k] for k in k2]), np.array([ba[k] for k in k2]),
                 "student vs EMA-only, run-level (n=12, seed-matched)")

    res = {"student": os.path.basename(os.path.normpath(args.student_dir)),
           "baseline": os.path.basename(args.baseline),
           "phase_offsets_note": "per-run accuracy, not the 2-seed logit ensemble",
           "fold_level": fold, "run_level": run}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    lines = [
        "# Distillation student vs EMA-only baseline (paired tests)",
        "",
        f"Student: `{res['student']}`; baseline: `{res['baseline']}`. "
        "Accuracy per run; fold-level row averages the two seeds within each fold "
        "(the paper's protocol), the run-level row pairs the same `(fold, seed)` cells.",
        "",
        "| comparison | n | baseline | student | diff | t | p | Wilcoxon p | d |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in (fold, run):
        lines.append(
            f"| {row['comparison']} | {row['n_pairs']} | "
            f"{row['baseline_mean']*100:.2f} | {row['candidate_mean']*100:.2f} | "
            f"{row['mean_diff']*100:+.2f} | {row['paired_t']:.3f} | "
            f"{row['paired_t_p']:.4f} | {row['wilcoxon_p']:.4f} | {row['cohens_d']:.2f} |")
    lines += ["", "Per-fold differences (points): " + ", ".join(
        f"f{k}: {(fc[k]-fb[k])*100:+.2f}" for k in keys), ""]
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
