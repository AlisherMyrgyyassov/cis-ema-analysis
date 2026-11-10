"""Aggregate all experiment outputs into paper/results tables + a summary.md."""
import json
import os

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, RUNS_DIR, FEATURE_GROUPS, GROUP_IDS


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x):
    return f"{x * 100:.2f}"


def build_master_table():
    """Exp 1 master table (7 rows) + SSL."""
    rows = []
    exp1 = _load(os.path.join(RESULTS_DIR, "exp1_ceilings.json"))
    ssl = _load(os.path.join(RESULTS_DIR, "exp1_ssl.json"))

    if exp1:
        rows.append(["Chance", "—", pct(exp1["chance"]["chance_acc"]), "—",
                     "1/43"])
        rows.append(["Majority", "—",
                     f"{pct(exp1['chance']['majority_acc_mean'])} +- {pct(exp1['chance']['majority_acc_std'])}",
                     "—", ""])
        rows.append(["Logistic Regression (pooled)", "424",
                     f"{pct(exp1['logistic_regression']['test_acc_mean'])} +- {pct(exp1['logistic_regression']['test_acc_std'])}",
                     f"{pct(exp1['logistic_regression']['test_f1_mean'])} +- {pct(exp1['logistic_regression']['test_f1_std'])}",
                     "mean/std/min/max per feature"])
        rows.append(["Random Forest (pooled)", "424",
                     f"{pct(exp1['random_forest']['test_acc_mean'])} +- {pct(exp1['random_forest']['test_acc_std'])}",
                     f"{pct(exp1['random_forest']['test_f1_mean'])} +- {pct(exp1['random_forest']['test_f1_std'])}",
                     ""])
        rows.append(["BiLSTM EMA-only", "59",
                     f"{pct(exp1['bilstm_ema_only']['test_acc_mean'])} +- {pct(exp1['bilstm_ema_only']['test_acc_std'])}",
                     f"{pct(exp1['bilstm_ema_only']['test_f1_mean'])} +- {pct(exp1['bilstm_ema_only']['test_f1_std'])}",
                     "G1-G4"])
        rows.append(["BiLSTM full (hand-crafted)", "106",
                     f"{pct(exp1['bilstm_full']['test_acc_mean'])} +- {pct(exp1['bilstm_full']['test_acc_std'])}",
                     f"{pct(exp1['bilstm_full']['test_f1_mean'])} +- {pct(exp1['bilstm_full']['test_f1_std'])}",
                     "G1-G8"])
    if ssl:
        primary = ssl.get("primary", None)
        pri = ssl.get(primary) if primary else None
        if pri:
            rows.append(["Fine-tuned SSL audio", "raw audio",
                         f"{pct(pri['test_acc_mean'])} +- {pct(pri['test_acc_std'])}",
                         f"{pct(pri['test_f1_mean'])} +- {pct(pri['test_f1_std'])}",
                         pri.get("model", primary)])
            others = [k for k in ("xlsr_53", "wav2vec2_reference")
                      if k in ssl and k != primary]
            if others:
                for k in others:
                    o = ssl[k]
                    rows.append([f"  — SSL ({k})", "raw audio",
                                 f"{pct(o['test_acc_mean'])} +- {pct(o['test_acc_std'])}",
                                 f"{pct(o['test_f1_mean'])} +- {pct(o['test_f1_std'])}",
                                 o.get("model", k)])

    df = pd.DataFrame(rows, columns=["Model", "Features", "Test acc (mean +- std)",
                                     "Test macro-F1 (mean +- std)", "Notes"])
    return df


def build_shapley_table():
    s = _load(os.path.join(RUNS_DIR, "exp2", "shapley.json"))
    if not s:
        return None
    agg = s["aggregate"]["shapley_acc"]
    agg_f1 = s["aggregate"]["shapley_f1"]
    rows = []
    for g in sorted(GROUP_IDS, key=lambda g: -agg[g]["mean"]):
        rows.append([g, FEATURE_GROUPS[g]["name"],
                     f"{agg[g]['mean'] * 100:+.2f} +- {agg[g]['std'] * 100:.2f}",
                     f"{agg_f1[g]['mean'] * 100:+.2f} +- {agg_f1[g]['std'] * 100:.2f}"])
    return pd.DataFrame(rows, columns=["Group", "Name", "Shapley acc (pts)",
                                       "Shapley macro-F1 (pts)"])


def build_probe_table(name="full"):
    s = _load(os.path.join(RUNS_DIR, "exp3", f"probing_{name}.json"))
    if not s:
        return None
    stages = s["stages"]
    rows = []
    for p in s["props"]:
        if p == "tone":  # confounded with syllable identity; see tone_within_syllable
            continue
        row = [p]
        for st in stages:
            a = s["aggregate"][f"{p}__{st}"]
            row.append(f"{a['acc_mean'] * 100:.2f} ({a['control_mean'] * 100:.2f})")
        rows.append(row)
    return pd.DataFrame(rows, columns=["Property"] + stages)


def build_correlation_table():
    s = _load(os.path.join(RUNS_DIR, "exp4", "correlation.json"))
    if not s:
        return None
    rows = [[r["acoustic_feature"], f"{r['lofo_importance'] * 100:.2f}",
             r["best_ema"], f"{r['best_abs_r']:.3f}", f"{r['best_r2']:.3f}",
             f"{r['single_ridge_r2_in']:.3f}", f"{r['joint_ridge_r2_in']:.3f}",
             f"{r['joint_ridge_r2_xs']:.3f}"]
            for r in s["table"]]
    return pd.DataFrame(rows, columns=[
        "Acoustic feature", "LOFO (pts)", "Best EMA feature", "|r| (per-utt)",
        "|r|^2", "ridge R^2 1-sensor (global)", "ridge R^2 joint (global)",
        "ridge R^2 joint (cross-subject)"])


def build_tone_table():
    """Per-tone recall for the ground-truth 6-way tone probe (Exp 3a).

    NOTE: this naive probe is confounded with syllable identity; see
    ``build_tone_within_table`` for the syllable-controlled analysis.
    """
    s = _load(os.path.join(RUNS_DIR, "tone", "tone_analysis.json"))
    if not s:
        return None
    rows = []
    for k in ("1", "2", "3", "4", "5", "6"):
        f = s["full"]["per_tone_recall"][k]
        e = s["ema"]["per_tone_recall"][k]
        rows.append([k, s["tone_distribution"].get(k, 0),
                     f"{f[0] * 100:.1f} +- {f[1] * 100:.1f}",
                     f"{e[0] * 100:.1f} +- {e[1] * 100:.1f}"])
    return pd.DataFrame(rows, columns=["Tone", "n utterances",
                                       "Full recall (%)", "EMA recall (%)"])


def build_tone_within_table():
    """Syllable-controlled tone decoding (naive vs multi-tone syllables only)."""
    s = _load(os.path.join(RUNS_DIR, "tone", "tone_within_syllable.json"))
    if not s:
        return None
    rows = []
    for label, key in (("All utterances (naive)", "naive"),
                       ("Multi-tone syllables only", "within")):
        ema, full = s[f"{key}_ema"], s[f"{key}_full"]
        rows.append([label,
                     f"{ema['probe_mean'] * 100:.1f} +- {ema['probe_std'] * 100:.1f}",
                     f"{full['probe_mean'] * 100:.1f} +- {full['probe_std'] * 100:.1f}",
                     f"{full['baseline_mean'] * 100:.1f}",
                     f"{ema['diff'] * 100:+.1f} (p={ema['p']:.3f}, d={ema['cohens_d']:+.2f})",
                     f"{full['diff'] * 100:+.1f} (p={full['p']:.3f}, d={full['cohens_d']:+.2f})"])
    return pd.DataFrame(rows, columns=[
        "Probe set", "EMA acc (%)", "Full acc (%)", "Baseline (%)",
        "EMA vs baseline", "Full vs baseline"])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    parts = ["# Experiment results (auto-generated)\n",
             "> Generated by `python -m src.report` from `runs/paper/*` "
             "(6-coil, 2026-10-17). Edit `src/report.py`, not this file.\n"]

    t = build_master_table()
    if t is not None:
        parts.append("## Experiment 1 — honest ceilings\n")
        parts.append(t.to_markdown(index=False))
        parts.append("")

    t = build_shapley_table()
    if t is not None:
        parts.append("## Experiment 2 — SAGE Shapley decomposition\n")
        parts.append(t.to_markdown(index=False))
        parts.append("")

    t = build_probe_table("full")
    if t is not None:
        parts.append("## Experiment 3a — probing, full model (acc %, control in parens)\n")
        parts.append(t.to_markdown(index=False))
        parts.append("")

    t = build_probe_table("ema")
    if t is not None:
        parts.append("## Experiment 3a — probing, EMA-only model (acc %, control in parens)\n")
        parts.append(t.to_markdown(index=False))
        parts.append("")

    t = build_correlation_table()
    if t is not None:
        parts.append("## Experiment 4 — mechanistic correlation\n")
        parts.append(t.to_markdown(index=False))
        parts.append("")
        parts.append(
            "**Interpretation.** `|r|^2` is the per-utterance single-sensor ceiling; "
            "`ridge R^2 joint` is the variance of the acoustic feature recovered by a "
            "*global* linear model over all 59 EMA features jointly (within-utterance "
            "standardised, so speaker level/scale is removed); `cross-subject` uses the "
            "paper's subject CV. Three results follow: (1) for several MFCC bands the "
            "joint model beats the single-sensor ridge probe (MFCC_2 0.51 vs 0.38, "
            "MFCC_5 0.47 vs 0.43, MFCC_6 0.35 vs 0.27), so the single-sensor estimate "
            "*undersells* the spectral information that is jointly present in EMA and "
            "part of the EMA-only BiLSTM's spectral gap is an architecture/optimisation "
            "limitation, not a structural one; (2) F2 is well recovered (joint 0.54, "
            "single 0.51, i.e. tongue front-back is essentially one-dimensional); "
            "(3) F0 and F3 stay near zero even jointly (0.14 and 0.04) — voicing and "
            "lip-rounding are genuinely absent from the sensor set, a structural gap. "
            "The remaining features (F1, MFCC_1/3/4) sit on a continuum between these "
            "poles — moderate single-sensor recovery with a modest joint gain — rather "
            "than forming a fourth mechanism of their own. See `exp4_mechanistic.png`.\n")
        parts.append("")

    # Experiment 5 (auxiliary-loss interventions: 5a readout / 5b encoder) + oracle
    exp5_cfgs = [
        ("Exp 5a readout aux — utt-mean, all frames", "exp5_uttmean_all", "all"),
        ("Exp 5a readout aux — utt-mean, voiced frames", "exp5_uttmean_voiced", "voiced"),
        ("Exp 5b encoder aux — per-frame trajectory", "exp5_perframe", None),
    ]
    available = {d: _load(os.path.join(RUNS_DIR, d, "summary.json"))
                 for _, d, _ in exp5_cfgs}
    if any(available.values()):
        base = _load(os.path.join(RESULTS_DIR, "exp1_ceilings.json"))
        parts.append("## Experiment 5 — MFCC-band auxiliary losses (5a readout, 5b encoder)\n")
        parts.append(
            "Auxiliary regression targets are MFCC_2/5/6. **5a** predicts the "
            "per-utterance mean from the pooled embedding that feeds the classifier "
            "(all-frames headline; voiced-frames variant as robustness against "
            "silence/onset dilution). **5b** predicts the within-utterance z-scored "
            "per-frame trajectory from the pre-pooling LSTM states (the Exp-4 ridge "
            "quantity, now through a neural bottleneck). Loss = CE + lambda*MSE. "
            "`auxR2 val/test` = mean-over-channels R^2 of the auxiliary head on "
            "held-out splits, i.e. whether the intervention was actually learned.\n")
        rows = []
        for label, d, _ in exp5_cfgs:
            s = available[d]
            if not s:
                continue
            for lmbda in ("0.3", "1.0"):
                if str(lmbda) not in s["summary"]:
                    continue
                x = s["summary"][str(lmbda)]
                rows.append([label, lmbda,
                             f"{x['test_acc_mean'] * 100:.2f} +- {x['test_acc_std'] * 100:.2f}",
                             f"{x['test_f1_mean'] * 100:.2f} +- {x['test_f1_std'] * 100:.2f}",
                             f"{x['aux_r2_val_mean']:.3f}",
                             f"{x['aux_r2_test_mean']:.3f}"])
        if base:
            b = base["bilstm_ema_only"]
            rows.append(["EMA-only baseline (Exp 1)", "—",
                         f"{b['test_acc_mean'] * 100:.2f} +- {b['test_acc_std'] * 100:.2f}",
                         f"{b['test_f1_mean'] * 100:.2f} +- {b['test_f1_std'] * 100:.2f}",
                         "—", "—"])
        parts.append(pd.DataFrame(rows, columns=["Model", "lambda", "Test acc",
                                                 "Test macro-F1", "auxR2 val",
                                                 "auxR2 test"]).to_markdown(index=False))
        parts.append("")
        parts.append(
            "**Interpretation.** The `auxR2` columns decide how a flat accuracy is "
            "read: if `auxR2` is near zero the auxiliary head never learned the "
            "target and the null is uninformative; if `auxR2` is moderate-to-high but "
            "accuracy does not move, forcing the target into the readout/encoder does "
            "not sharpen syllable decisions beyond what the classifier already "
            "exploits. If the oracle below is positive, that flatness is *not* because "
            "the target is uninformative — it means a multi-task nudge cannot turn the "
            "jointly-recoverable statistic into discriminative signal, even though "
            "injecting the true value can. Comparing 5b's per-frame R^2 with 5a's "
            "pooled R^2 localizes where information is lost: encoder capacity (5b "
            "high) vs the mean-pooled readout (5a low). Paired fold-wise tests vs the "
            "Exp-1 EMA-only baseline are in `paired_tests.md`.\n")
        parts.append("")

        subj = _load(os.path.join(RESULTS_DIR, "exp5_subject_tests.json"))
        if subj and subj.get("rows"):
            n = subj["n_subjects"]
            parts.append("**Subject-level paired inference (re-analysis, "
                         f"n = {n} test subjects).** Under the subject-disjoint CV the "
                         "test *subject* is the independent unit; pairing per subject "
                         "(mean over the 2 seeds) gives the maximal legitimate sample. "
                         "95% CI is a subject-cluster bootstrap. Fold-level (n=6) rows "
                         "retained for protocol consistency.\n")
            srows = []
            for r in subj["rows"]:
                srows.append([
                    f"{r['label']} (λ={r['lmbda']})",
                    f"{r['subj_delta'] * 100:+.2f}",
                    f"{r['subj_p']:.3f}",
                    f"[{r['subj_ci_low'] * 100:+.2f}, {r['subj_ci_high'] * 100:+.2f}]",
                    f"{r['fold_delta'] * 100:+.2f}", f"{r['fold_p']:.3f}",
                    f"{r['fold_d']:.2f}"])
            parts.append(pd.DataFrame(srows, columns=[
                "Cell", "Δ pts (n=33)", "p", "95% CI",
                "fold Δ (n=6)", "fold p", "fold d"]).to_markdown(index=False))
            parts.append("")
            parts.append(
                "Interpretation: at the full subject-level sample the two most "
                "positive cells (voiced λ=1.0, per-frame λ=1.0) sit at p ≈ 0.06–0.08 "
                "with CIs that include 0 at the lower bound and bound any plausible "
                "upside at ≈ +2 pts; the all-frame λ=1.0 cell is negative. The sign "
                "instability across frame-sets and λ means the results do not support "
                "a coherent auxiliary effect; the CIs bound what a simple MFCC "
                "auxiliary target could add for this task.\n")
            parts.append("")

    # Exp-5 oracle (does the true target mean carry signal beyond EMA?)
    oracle = _load(os.path.join(RUNS_DIR, "exp5_oracle", "summary.json"))
    if oracle and "runs_pooled" in oracle:
        parts.append("## Experiment 5 oracle — is the true utterance-mean MFCC_2/5/6 target class-informative beyond EMA-only?\n")
        o = oracle
        rows = []
        _acc = lambda k: f"{o[k]['test_acc_mean'] * 100:.2f} +- {o[k]['test_acc_std'] * 100:.2f}"
        rows.append(["Pooled LR (EMA-only)", _acc("pooled_lr_ema_only"), ""])
        rows.append(["Pooled LR (EMA + true means)", _acc("pooled_lr_ema_plus_means"),
                     "oracle (microphone)"])
        rows.append(["Pooled RF (EMA-only)", _acc("pooled_rf_ema_only"), ""])
        rows.append(["Pooled RF (EMA + true means)", _acc("pooled_rf_ema_plus_means"),
                     "oracle (microphone)"])
        if "bilstm_ema_broadcast_means" in o:
            rows.append(["BiLSTM EMA + broadcast true means",
                         _acc("bilstm_ema_broadcast_means"),
                         "vs EMA-only baseline 78.71% (Exp 1)"])
        parts.append(pd.DataFrame(rows, columns=["Model", "Test acc", "Notes"]).to_markdown(index=False))
        parts.append("")
        parts.append(
            "**Interpretation.** Giving the model the *true* target mean shows whether "
            "an auxiliary signal of this kind could plausibly help at all: a positive "
            "oracle means the target carries syllable-discriminative spectral "
            "information EMA does not already provide; a null oracle would make a flat "
            "Exp-5 accuracy fully explained (redundant target) independent of the aux "
            "head. The BiLSTM row is the temporally-matched analogue of the pooled "
            "rows.\n")
        parts.append("")

    tone_within = _load(os.path.join(RUNS_DIR, "tone", "tone_within_syllable.json"))
    if tone_within is not None:
        n_syl = len(tone_within["multi_tone_syllables"])
        parts.append("## Tone, syllable-controlled\n")
        parts.append(
            "The naive 6-way tone probe is **confounded with syllable identity**: only "
            f"{n_syl} of the 43 syllables occur with two or more tones "
            f"({tone_within['n_multi_utterances']} of {tone_within['n_all_utterances']} "
            "utterances), and predicting tone from the syllable alone reaches "
            f"{tone_within['naive_full']['baseline_mean'] * 100:.1f}%. The naive probes "
            "sit below that ceiling, so they mostly measure syllable decoding, not tone. "
            "Restricted to the multi-tone syllables, both the EMA probe and the full "
            "model exceed the per-syllable majority baseline.\n")
        tw = build_tone_within_table()
        if tw is not None:
            parts.append(tw.to_markdown(index=False))
            parts.append("")

    t = build_tone_table()
    if t is not None:
        ta = _load(os.path.join(RUNS_DIR, "tone", "tone_analysis.json"))
        parts.append("## Naive tone probe (confounded — interpret with care)\n")
        parts.append(
            f"Tone probe accuracy (pooled, all utterances): **full {ta['full']['acc_mean'] * 100:.1f} +- "
            f"{ta['full']['acc_std'] * 100:.1f}%** vs **EMA-only {ta['ema']['acc_mean'] * 100:.1f} +- "
            f"{ta['ema']['acc_std'] * 100:.1f}%** (chance 16.7%, majority {ta['majority'] * 100:.1f}%). "
            f"Tone labels are ground-truth from `stimuli.xlsx` (item 114 excluded).\n")
        parts.append(t.to_markdown(index=False))
        parts.append("")

    parts.append("## Notes and caveats\n")
    parts.append(
        "- **Tone.** Lexical tone labels (1-6) come from `stimuli.xlsx` "
        "(`Item No.` = the utterance `file_num`), joined on the item number; item 114 "
        "(陶域道, ambiguous tonal reading) is excluded. The naive 6-way tone probe is "
        "confounded with syllable identity (33 of 43 syllables occur with a single "
        "tone; syllable-to-tone majority = 90.2%), so it must not be read as tone "
        "recovery. The syllable-controlled analysis (`tone_within_syllable.json`, "
        "`scripts/within_syllable_tone.py`) shows both EMA and full tone decoding "
        "exceed the per-syllable majority baseline (+8.2 and +9.1 points, "
        "p=0.010/0.001). The earlier "
        "F0-based pseudo-clustering was invalid (silhouette 0.28, AMI with syllable "
        "identity 0.05) and has been superseded.\n"
        "- **Masked-model insertion curve.** The insertion curve's 'EMA-only' start "
        "point (~20%) is the frozen *full* model with acoustic groups zeroed, not the "
        "true EMA-only-trained accuracy (78.71%). This is a known SAGE/Shapley masking "
        "artifact; the *ordering* of the deletion/insertion curves remains valid.\n"
        "- **Exp 5 aux alignment fix.** The earlier Exp-5 code attached the auxiliary "
        "MSE target by sequential batch chunking, which mismatches the bucketed "
        "sampler's shuffled order, so the aux head was trained against (effectively) "
        "random utterance targets and the null was uninterpretable. The rerun carries "
        "the target inside the dataset (index-aligned with each utterance), records "
        "train/val/test aux R^2, and splits the intervention into a readout-level "
        "(5a) and an encoder-level per-frame (5b) head.\n"
        "- **Exp 5 oracle.** Adding the *true* (microphone-derived) utterance-mean "
        "MFCC_2/5/6 target to an EMA-only model measures whether such a target is "
        "class-informative at all; it is an analysis oracle, not an SSI operating "
        "point (it needs the microphone).\n"
        "- **Wilcoxon floor.** With n=6 folds the Wilcoxon signed-rank p-value floors "
        "at 0.031; the paired t-test is primary and Cohen's d is reported for every "
        "contrast (see `paired_tests.md`).\n"
        "- **SSL ceiling.** HuBERT-large (94.1%) exceeds the hand-crafted full model "
        "(88.3%), so hand-crafted features leave ~5.8 points on the table; the SSL "
        "ceiling is itself well below 100%, confirming the task is sample-size / "
        "subject-disjoint-generalisation limited.\n")

    out = os.path.join(RESULTS_DIR, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    main()
