"""Experiment 5 - subject-level paired inference (post-hoc re-analysis).

The fold-level paired tests used elsewhere in the paper are capped at n=6
(one fold-level value per test-subject *group*). Under the subject-disjoint CV,
the independent sampling unit is the test *subject*: each of the 33 subjects
appears in exactly one fold's test set, and every model variant is evaluated on
the same subjects' utterances. Pairing at the subject level therefore gives the
maximal legitimate sample for the Exp-5 comparisons without any retraining.

This module re-scores the saved checkpoints of the EMA-only baseline and of the
six Exp-5 auxiliary configurations (5a uttmean all/voiced x lambda, 5b perframe
x lambda), aggregates per-test-subject accuracy over the 2 training seeds, and
reports, for every cell vs the baseline:
  - per-subject paired mean delta (n=33) with a paired t-test,
  - a subject-cluster bootstrap 95% CI on the mean delta,
  - the fold-level (n=6) delta / p / Cohen's d for protocol consistency.

No model is trained here; this is a deterministic re-scoring of existing
checkpoints. Output: paper/results/exp5_subject_tests.{json,md}.
"""
import json
import os

import numpy as np
import torch
from scipy import stats

from src.config import (RUNS_DIR, RESULTS_DIR, DATA_DIR, N_FOLDS,
                        N_VAL_SUBJECTS, FOLD_SEED, EMA_GROUPS, columns_for)
from src.data.dataset import load_dataset, records_for, view, fit_stats, standardize
from src.data.cv import make_folds
from src.models.bilstm import SyllableBiLSTM, SyllableBiLSTMAux, SyllableBiLSTMFrameAux

BOOT_SEED = 0
N_BOOT = 5000
MODEL_KINDS = {"base": SyllableBiLSTM, "aux": SyllableBiLSTMAux,
               "frame": SyllableBiLSTMFrameAux}

BASELINE = {"label": "EMA-only baseline",
            "ckpt_dir": os.path.join(RUNS_DIR, "exp1", "ema_only"),
            "tag": "_model", "kind": "base", "lmbda": None}

CELLS = [
    {"label": "5a uttmean all", "dir": "exp5_uttmean_all", "kind": "aux"},
    {"label": "5a uttmean voiced", "dir": "exp5_uttmean_voiced", "kind": "aux"},
    {"label": "5b perframe", "dir": "exp5_perframe", "kind": "frame"},
]


def _make_model(kind, input_dim, num_classes):
    cls = MODEL_KINDS[kind]
    if kind == "base":
        return cls(input_dim, num_classes)
    return cls(input_dim, num_classes, aux_dim=3)


def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _subject_accs(records, col_index, folds, ckpt_dir, tag, kind, device):
    """Per-test-subject accuracy (mean over the two training seeds)."""
    counts = {r["subj"]: [0.0, 0] for r in records}
    for fold_id, fd in enumerate(folds):
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])
        mu, sd = fit_stats(view(tr, col_index, EMA_GROUPS))
        te_std = standardize(view(te, col_index, EMA_GROUPS), mu, sd)
        for seed in (0, 1):
            ckpt = _load_ckpt(os.path.join(ckpt_dir,
                                           f"fold{fold_id}_seed{seed}{tag}.pth"))
            model = _make_model(kind, ckpt["input_dim"], ckpt["num_classes"])
            model.load_state_dict(ckpt["model"])
            model.to(device).eval()
            cls2i = {c: i for i, c in enumerate(ckpt["classes"])}
            with torch.no_grad():
                for i in range(0, len(te_std), 256):
                    chunk = te_std[i:i + 256]
                    lens = torch.tensor([s.shape[0] for s in chunk], dtype=torch.long)
                    mx = int(lens.max())
                    F = chunk[0].shape[1]
                    x = torch.zeros(len(chunk), mx, F)
                    for b, s in enumerate(chunk):
                        x[b, :s.shape[0]] = torch.from_numpy(s)
                    out = model(x.to(device), lens.to(device))
                    logits = out[0] if isinstance(out, tuple) else out
                    preds = logits.argmax(1).cpu().numpy()
                    for j, rec in enumerate(te[i:i + 256]):
                        counts[rec["subj"]][0] += int(preds[j] == cls2i[rec["syllable"]])
                        counts[rec["subj"]][1] += 1
    return {s: c[0] / c[1] for s, c in counts.items() if c[1] > 0}


def _fold_level(cell_summary, lmbda):
    """Per-fold mean test acc over seeds from a cell's summary.json runs."""
    out = {}
    for r in cell_summary["runs"]:
        if r["lmbda"] != lmbda:
            continue
        out.setdefault(r["fold"], []).append(r["test_acc"])
    return {f: float(np.mean(v)) for f, v in out.items()}


def run(data_dir=None, out_dir=None, device="cuda", n_boot=N_BOOT):
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or RESULTS_DIR
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    records, col_index = load_dataset(data_dir)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)

    base_acc = _subject_accs(records, col_index, folds, BASELINE["ckpt_dir"],
                             BASELINE["tag"], BASELINE["kind"], device)
    fold_base_raw = json.load(open(os.path.join(RUNS_DIR, "exp1",
                                                "ema_only_results.json")))
    fb = {}
    for r in fold_base_raw:
        fb.setdefault(r["fold"], []).append(r["test_acc"])
    fold_base = {f: float(np.mean(v)) for f, v in fb.items()}
    keys = sorted(set(base_acc))

    rows = []
    for cell in CELLS:
        ckpt_dir = os.path.join(RUNS_DIR, cell["dir"], "models")
        summary = json.load(open(os.path.join(RUNS_DIR, cell["dir"], "summary.json")))
        for lmbda in (0.3, 1.0):
            tag = f"_lmbda{lmbda}"
            acc = _subject_accs(records, col_index, folds, ckpt_dir, tag,
                                cell["kind"], device)
            k2 = sorted(set(acc) & set(keys))
            delta = np.array([acc[s] - base_acc[s] for s in k2])

            _, p = stats.ttest_rel([acc[s] for s in k2],
                                   [base_acc[s] for s in k2])
            rng = np.random.RandomState(BOOT_SEED)
            boot = np.array([np.mean(rng.choice(delta, size=len(delta),
                                                replace=True))
                             for _ in range(n_boot)])
            ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

            # fold-level (n=6), seed-averaged, for protocol consistency
            fold_cell = _fold_level(summary, lmbda)
            ff = sorted(set(fold_cell) & set(fold_base))
            fdelta = np.array([fold_cell[f] - fold_base[f] for f in ff])
            _, fp = stats.ttest_rel([fold_cell[f] for f in ff],
                                    [fold_base[f] for f in ff])
            d = float(fdelta.mean() / (fdelta.std(ddof=1) + 1e-12))

            rows.append({
                "cell": f"{cell['label']} (lambda={lmbda})",
                "label": cell["label"], "lmbda": lmbda,
                "fold_delta": float(fdelta.mean()), "fold_p": float(fp),
                "fold_d": d,
                "subj_delta": float(delta.mean()), "subj_p": float(p),
                "subj_ci_low": ci[0], "subj_ci_high": ci[1], "n_subjects": len(k2),
            })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp5_subject_tests.json"), "w") as f:
        json.dump({"n_subjects": len(keys),
                   "mean_baseline_subj_acc": float(np.mean([base_acc[s] for s in keys])),
                   "rows": rows}, f, indent=2)

    lines = ["# Experiment 5 — subject-level paired inference (re-analysis)",
             "",
             f"n = {len(keys)} test subjects (each in exactly one test fold; "
             "per-subject accuracy averaged over the 2 training seeds). "
             "Per-subject paired t + subject-cluster bootstrap 95% CI. "
             "Fold-level (n=6) rows kept for protocol consistency.",
             "",
             "| Cell | Δ pts (n=33) | p | 95% CI | fold Δ (n=6) | fold p | fold d |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['cell']} | {r['subj_delta']*100:+.2f} | "
                     f"{r['subj_p']:.3f} | [{r['subj_ci_low']*100:+.2f}, "
                     f"{r['subj_ci_high']*100:+.2f}] | "
                     f"{r['fold_delta']*100:+.2f} | {r['fold_p']:.3f} | "
                     f"{r['fold_d']:.2f} |")
    with open(os.path.join(out_dir, "exp5_subject_tests.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("==== Exp 5 subject-level paired inference (n=%d) ====" % len(keys))
    for r in rows:
        print(f"  {r['cell']:32s} delta={r['subj_delta']*100:+.2f}  "
              f"p={r['subj_p']:.3f}  CI=[{r['subj_ci_low']*100:+.2f}, "
              f"{r['subj_ci_high']*100:+.2f}]  fold p={r['fold_p']:.3f} d={r['fold_d']:.2f}")
    return rows


if __name__ == "__main__":
    run()
