"""Paired statistical tests for the key paper comparisons.

Aggregates each configuration to fold-level values (mean of its seeds), then
runs paired t-test / Wilcoxon / Cohen's d across the 6 matched folds. Because
n = 6 folds, the paired t-test is the primary statistic and Cohen's d (effect
size) is reported for every contrast; the Wilcoxon p-value floors at 0.031 for
n = 6, so it is reported but not over-interpreted.
"""
import json
import os

import numpy as np

from src.config import RESULTS_DIR, RUNS_DIR
from src.stats import paired_compare


def _fold_level(results, key):
    folds = {}
    for r in results:
        folds.setdefault(r["fold"], []).append(r[key])
    return {f: float(np.mean(v)) for f, v in folds.items()}


def _paired(results_a, results_b, key):
    fa = _fold_level(results_a, key)
    fb = _fold_level(results_b, key)
    folds = sorted(set(fa) & set(fb))
    return paired_compare([fa[f] for f in folds], [fb[f] for f in folds])


def load(path):
    with open(path) as f:
        return json.load(f)


def _fmt(t, scale=100):
    if not t or t.get("n_pairs", 0) < 2:
        return "n/a"
    return (f"{t['mean_diff'] * scale:+.2f} | t={t['paired_t']:.2f} | "
            f"p={t['paired_t_p']:.3f} | W_p={t['wilcoxon_p']:.3f} | d={t['cohens_d']:.2f}")


def main():
    full = load(os.path.join(RUNS_DIR, "exp1", "full_results.json"))
    ema = load(os.path.join(RUNS_DIR, "exp1", "ema_only_results.json"))
    _pooled_path = os.path.join(RUNS_DIR, "exp1", "pooled_results.json")
    pooled = load(_pooled_path) if os.path.isfile(_pooled_path) else None
    hubert = load(os.path.join(RUNS_DIR, "exp1_ssl_hubert", "summary.json"))["folds"]
    xlsr = load(os.path.join(RUNS_DIR, "exp1_ssl_xlsr", "summary.json"))["folds"]

    tests = {
        "full vs EMA-only (acc)": _paired(full, ema, "test_acc"),
        "full vs EMA-only (F1)": _paired(full, ema, "test_f1"),
        "full vs HuBERT (acc)": _paired(full, hubert, "test_acc"),
        "full vs XLSR-53 (acc)": _paired(full, xlsr, "test_acc"),
        "EMA-only vs HuBERT (acc)": _paired(ema, hubert, "test_acc"),
    }
    if pooled:
        tests["full vs pooled-LR (acc)"] = _paired(full, pooled["full"]["lr"], "test_acc")
        tests["full vs pooled-RF (acc)"] = _paired(full, pooled["full"]["rf"], "test_acc")
        tests["EMA-only vs pooled-LR-EMA (acc)"] = _paired(ema, pooled["ema"]["lr"], "test_acc")
        tests["EMA-only vs pooled-RF-EMA (acc)"] = _paired(ema, pooled["ema"]["rf"], "test_acc")

    # probing: full vs EMA-only per property at the pooled stage
    pf = load(os.path.join(RUNS_DIR, "exp3", "probing_full.json"))
    pe = load(os.path.join(RUNS_DIR, "exp3", "probing_ema.json"))
    stage = "pooled"
    for prop in pf["props"]:
        if prop == "tone":  # confounded with syllable identity; see tone_within_syllable
            continue
        a = {r["fold"]: r[f"{prop}__{stage}"]["acc"] for r in pf["runs"]}
        b = {r["fold"]: r[f"{prop}__{stage}"]["acc"] for r in pe["runs"]}
        folds = sorted(set(a) & set(b))
        tests[f"probe {prop} (pooled) full vs EMA"] = paired_compare(
            [a[f] for f in folds], [b[f] for f in folds])

    # Experiment 5: MFCC-aux (readout 5a: all/voiced frames; encoder 5b) vs EMA-only
    exp5_cfgs = {
        "EMA + MFCC-aux uttmean(all)": "exp5_uttmean_all",
        "EMA + MFCC-aux uttmean(voiced)": "exp5_uttmean_voiced",
        "EMA + MFCC-aux perframe": "exp5_perframe",
    }
    for label, cfg_dir in exp5_cfgs.items():
        exp5 = load(os.path.join(RUNS_DIR, cfg_dir, "summary.json"))
        if not exp5:
            continue
        for lmbda in ("0.3", "1.0"):
            if str(lmbda) not in exp5["summary"]:
                continue
            aux_runs = [r for r in exp5["runs"] if r["lmbda"] == float(lmbda)]
            tests[f"{label} (lambda={lmbda}) vs EMA-only (acc)"] = \
                _paired(aux_runs, ema, "test_acc")

    # Experiment 5 oracle: does the *true* utterance-mean target add signal?
    oracle = load(os.path.join(RUNS_DIR, "exp5_oracle", "summary.json"))
    if oracle and "runs_pooled" in oracle:
        rp = oracle["runs_pooled"]
        tests["pooled-LR +true means vs EMA-only (acc)"] = \
            _paired(rp["lr_mean"], rp["lr"], "test_acc")
        tests["pooled-RF +true means vs EMA-only (acc)"] = \
            _paired(rp["rf_mean"], rp["rf"], "test_acc")
        if oracle.get("runs_bilstm"):
            tests["BiLSTM +true means(broadcast) vs EMA-only (acc)"] = \
                _paired(oracle["runs_bilstm"], ema, "test_acc")

    # Tone, syllable-controlled: EMA/full probe vs the per-syllable majority baseline
    tw = load(os.path.join(RUNS_DIR, "tone", "tone_within_syllable.json"))
    if tw:
        for key, label in (("within_ema", "tone within-syllable EMA"),
                           ("within_full", "tone within-syllable full")):
            d = tw[key]
            tests[f"{label} vs per-syllable majority"] = paired_compare(
                d["per_fold_probe"], d["per_fold_baseline"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "paired_tests.json"), "w") as f:
        json.dump(tests, f, indent=2)

    lines = ["# Paired fold-wise statistical tests",
             "",
             "n = 6 folds (per-seed values averaged within fold). Paired t-test is "
             "primary; Wilcoxon p floors at 0.031 for n=6. `d` = Cohen's d on paired "
             "differences; `mean_diff` in the units of the metric (percentage points "
             "for accuracy/F1).",
             "",
             "| Comparison | mean diff | paired t | p | Wilcoxon p | Cohen's d |",
             "|---|---|---|---|---|---|"]
    for name, t in tests.items():
        if not t or t.get("n_pairs", 0) < 2:
            lines.append(f"| {name} | {t.get('note', 'n/a')} | | | | |")
            continue
        lines.append(f"| {name} | {t['mean_diff'] * 100:+.2f} | {t['paired_t']:.2f} "
                     f"| {t['paired_t_p']:.3f} | {t['wilcoxon_p']:.3f} | {t['cohens_d']:.2f} |")
    with open(os.path.join(RESULTS_DIR, "paired_tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Paired fold-wise tests (mean_diff in %points; t / p / W_p / d):")
    for name, t in tests.items():
        print(f"  {name:42s} {_fmt(t)}")
    return tests


if __name__ == "__main__":
    main()
