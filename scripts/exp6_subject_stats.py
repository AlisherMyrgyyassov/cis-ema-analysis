"""Experiment 6: subject-level paired inference for a KD student vs the EMA-only
baseline. Re-scores saved checkpoints per test subject (each subject is in
exactly one test fold; accuracy averaged over the 2 seeds), then reports a paired
t-test, a subject-cluster bootstrap 95% CI, and the fold-level test for protocol
consistency. No training here.

Two fold-level estimands are reported, because they answer different questions:
``fold_delta_ens`` compares the 2-seed logit ensemble of each model (the
deployment view), while ``fold_delta_seed`` averages the per-seed fold accuracies
(the estimand the paper quotes for the single-model KD gain, computed independently
in ``scripts/exp6_paired_test.py``)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from src.config import DATA_DIR, RUNS_DIR, RESULTS_DIR, EMA_GROUPS
from src.pipeline import prepare_data
from src import exp6_distill as E6

BOOT_SEED = 0
N_BOOT = 5000


def subject_accs(records, col_index, folds, ckpt_dir, device, tag="_model",
                 seeds=(0, 1), groups=EMA_GROUPS):
    counts = {r["subj"]: [0.0, 0] for r in records}
    fold_seed, fold_ens = {}, {}
    for fold_id, fd in enumerate(folds):
        pf = E6.prepare_fold(records, col_index, fd, groups)
        cls2i = {c: i for i, c in enumerate(pf["classes"])}
        ens_logits = []
        for seed in seeds:
            path = os.path.join(ckpt_dir, f"fold{fold_id}_seed{seed}{tag}.pth")
            if not os.path.isfile(path):
                continue
            model, ckpt = E6.load_ckpt(path, device)
            logits = E6.logits_for(model, pf["te_std"], device)
            ens_logits.append(logits)
        y = np.array([cls2i[r["syllable"]] for r in pf["te"]])
        # per-seed subject accuracy
        seed_accs = []
        for logits in ens_logits:
            preds = logits.argmax(1)
            seed_accs.append(float((preds == y).mean()))
            for j, rec in enumerate(pf["te"]):
                counts[rec["subj"]][0] += int(preds[j] == y[j])
                counts[rec["subj"]][1] += 1
        if ens_logits:
            # per-seed mean (the paper's single-model estimand) and the 2-seed
            # logit ensemble (the deployment estimand)
            fold_seed[fold_id] = float(np.mean(seed_accs))
            ens = np.mean(ens_logits, axis=0)
            fold_ens[fold_id] = float((ens.argmax(1) == y).mean())
    subj = {s: c[0] / c[1] for s, c in counts.items() if c[1] > 0}
    return subj, fold_seed, fold_ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--cand-data-dir", default=None,
                    help="feature table the candidate was trained on (default: data-dir)")
    ap.add_argument("--base-data-dir", default=None,
                    help="feature table the baseline was trained on (default: data-dir)")
    ap.add_argument("--cand-dir", default=os.path.join(RUNS_DIR, "exp6", "ensC_a09_ck"))
    ap.add_argument("--cand-tag", default="_model")
    ap.add_argument("--base-dir", default=os.path.join(RUNS_DIR, "exp1", "ema_only"))
    ap.add_argument("--base-tag", default="_model")
    ap.add_argument("--out", default=os.path.join(RUNS_DIR, "exp6", "exp6_subject_stats.md"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cand_dir = args.cand_data_dir or args.data_dir
    base_dir = args.base_data_dir or args.data_dir
    records_c, col_index_c, subjects, sex_map, folds = prepare_data(cand_dir)
    records_b, col_index_b, _, _, _ = prepare_data(base_dir)

    cand, cand_seed, cand_ens = subject_accs(records_c, col_index_c, folds, args.cand_dir, device, args.cand_tag)
    base, base_seed, base_ens = subject_accs(records_b, col_index_b, folds, args.base_dir, device, args.base_tag)
    keys = sorted(set(cand) & set(base))
    delta = np.array([cand[s] - base[s] for s in keys])
    t, p = stats.ttest_rel([cand[s] for s in keys], [base[s] for s in keys])
    rng = np.random.RandomState(BOOT_SEED)
    boot = np.array([np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(N_BOOT)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    ff = sorted(set(cand_seed) & set(base_seed))
    fdelta = np.array([cand_seed[f] - base_seed[f] for f in ff])
    ft, fp = stats.ttest_rel([cand_seed[f] for f in ff], [base_seed[f] for f in ff])
    fd = float(fdelta.mean() / (fdelta.std(ddof=1) + 1e-12))

    fe = sorted(set(cand_ens) & set(base_ens))
    edelta = np.array([cand_ens[f] - base_ens[f] for f in fe])
    et, ep = stats.ttest_rel([cand_ens[f] for f in fe], [base_ens[f] for f in fe])
    ed = float(edelta.mean() / (edelta.std(ddof=1) + 1e-12))

    res = {"n_subjects": len(keys),
           "base_subject_mean": float(np.mean([base[s] for s in keys])),
           "cand_subject_mean": float(np.mean([cand[s] for s in keys])),
           "subj_delta": float(delta.mean()), "subj_p": float(p),
           "subj_ci_low": ci[0], "subj_ci_high": ci[1],
           "fold_delta": float(fdelta.mean()), "fold_t": float(ft),
           "fold_p": float(fp), "fold_d": fd,
           "fold_ens_delta": float(edelta.mean()), "fold_ens_t": float(et),
           "fold_ens_p": float(ep), "fold_ens_d": ed}
    lines = ["# Experiment 6 - subject-level paired inference (KD vs EMA-only)", "",
             f"candidate = `{os.path.basename(args.cand_dir)}`", f"baseline = `{os.path.basename(args.base_dir)}`", "",
             f"n = {len(keys)} test subjects (each in exactly one fold; per-seed averaged).", "",
             "| metric | value |", "|---|---|",
             f"| baseline subject mean | {res['base_subject_mean']*100:.2f} |",
             f"| candidate subject mean | {res['cand_subject_mean']*100:.2f} |",
             f"| delta (n={len(keys)}) | {res['subj_delta']*100:+.2f} |",
             f"| paired t p (n={len(keys)}) | {res['subj_p']:.4f} |",
             f"| bootstrap 95% CI | [{res['subj_ci_low']*100:+.2f}, {res['subj_ci_high']*100:+.2f}] |",
             f"| fold delta, per-seed mean (n=6) | {res['fold_delta']*100:+.2f} |",
             f"| fold t / p / d, per-seed mean | {res['fold_t']:.2f} / {res['fold_p']:.3f} / {res['fold_d']:.2f} |",
             f"| fold delta, 2-seed ensemble (n=6) | {res['fold_ens_delta']*100:+.2f} |",
             f"| fold t / p / d, 2-seed ensemble | {res['fold_ens_t']:.2f} / {res['fold_ens_p']:.3f} / {res['fold_ens_d']:.2f} |", ""]
    out = "\n".join(lines)
    print(out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()
