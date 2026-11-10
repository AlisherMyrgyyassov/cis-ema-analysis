"""Naive 6-way tone probe (ground-truth tones from stimuli.xlsx).

WARNING: this probe is confounded with syllable identity. Tone is heavily
confounded with the syllable in this corpus (33 of 43 syllables occur with a
single tone; the syllable-to-tone majority ceiling is ~90%), so the accuracies
computed here (full ~80%, EMA ~65%) sit *below* the confound ceiling and mostly
measure syllable decoding, not tone. The interpretable, syllable-controlled
analysis is ``scripts/within_syllable_tone.py`` (output:
``runs/paper/tone/tone_within_syllable.json``). This module is retained only for
comparison.

With real Cantonese tone labels (1-6, item 114 excluded), we can go beyond a
single 6-way probe accuracy and report:

  * per-tone recall for the full (acoustic) vs EMA-only models at the pooled
    stage, averaged over folds/seeds,
  * an aggregated tone confusion matrix,
  * the tone distribution and chance/majority baselines.

This is the tonal counterpart to the Exp-3a probing table and grounds the
"how much of lexical tone is recoverable from articulatory motion alone" claim.
"""
import os
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from src.config import RUNS_DIR, RESULTS_DIR, GROUP_IDS, EMA_GROUPS, CANTONESE_TONES
from src.pipeline import prepare_data
from src.data.dataset import records_for, view, fit_stats, standardize
from src.data.tone import add_tone_to_records
from src.models.bilstm import SyllableBiLSTM
from src.exp3_probing import extract_representations


def _tone_probe_pooled(model, seqs, labels, device):
    X = extract_representations(model, seqs, device)
    return X["pooled"]


def run(data_dir=None, full_results=None, ema_results=None, out_dir=None,
        device="cuda", num_layers=2, hidden=128, dropout=0.4):
    from src.config import DATA_DIR
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or os.path.join(RUNS_DIR, "tone")
    os.makedirs(out_dir, exist_ok=True)

    records, col_index, subjects, sex_map, folds = prepare_data(data_dir)
    add_tone_to_records(records)

    full_results = full_results or os.path.join(RUNS_DIR, "exp1", "full_results.json")
    ema_results = ema_results or os.path.join(RUNS_DIR, "exp1", "ema_only_results.json")
    with open(full_results) as f:
        full_res = json.load(f)
    with open(ema_results) as f:
        ema_res = json.load(f)

    device = torch.device(device)

    def tone_eval(model_results, groups):
        recalls = {k: [] for k in CANTONESE_TONES}
        accs = []
        cms = []
        for r in model_results:
            fold_id, seed = r["fold"], r["seed"]
            fd = folds[fold_id]
            tr = [rec for rec in records_for(records, fd["train"]) if rec.get("tone") is not None]
            te = [rec for rec in records_for(records, fd["test"]) if rec.get("tone") is not None]

            tr_seqs = view(tr, col_index, groups)
            mu, sd = fit_stats(tr_seqs)
            tr_std = standardize(tr_seqs, mu, sd)
            te_std = standardize(view(te, col_index, groups), mu, sd)

            ckpt = torch.load(r["ckpt_path"], map_location="cpu")
            model = SyllableBiLSTM(ckpt["input_dim"], ckpt["num_classes"],
                                   num_layers=num_layers, hidden=hidden, dropout=dropout)
            model.load_state_dict(ckpt["model"])
            model.to(device).eval()

            X_tr = _tone_probe_pooled(model, tr_std, [x["tone"] for x in tr], device)
            X_te = _tone_probe_pooled(model, te_std, [x["tone"] for x in te], device)
            y_tr = np.array([x["tone"] for x in tr])
            y_te = np.array([x["tone"] for x in te])

            clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            accs.append(float((pred == y_te).mean()))
            cm = confusion_matrix(y_te, pred, labels=list(CANTONESE_TONES))
            cms.append(cm)
            for i, k in enumerate(CANTONESE_TONES):
                denom = cm[i].sum()
                recalls[k].append(cm[i, i] / denom if denom > 0 else 0.0)
        return accs, {k: (float(np.mean(v)), float(np.std(v))) for k, v in recalls.items()}, cms

    acc_f, rec_f, cms_f = tone_eval(full_res, GROUP_IDS)
    acc_e, rec_e, cms_e = tone_eval(ema_res, EMA_GROUPS)

    # tone distribution
    dist = {}
    for r in records:
        if r.get("tone") is not None:
            dist[str(r["tone"])] = dist.get(str(r["tone"]), 0) + 1

    summary = {
        "full": {"acc_mean": float(np.mean(acc_f)), "acc_std": float(np.std(acc_f)),
                 "per_tone_recall": {str(k): list(v) for k, v in rec_f.items()}},
        "ema": {"acc_mean": float(np.mean(acc_e)), "acc_std": float(np.std(acc_e)),
                "per_tone_recall": {str(k): list(v) for k, v in rec_e.items()}},
        "chance": 1 / 6,
        "majority": max(dist.values()) / sum(dist.values()),
        "tone_distribution": dist,
        "confusion_full": np.sum(cms_f, axis=0).astype(int).tolist(),
        "confusion_ema": np.sum(cms_e, axis=0).astype(int).tolist(),
    }
    with open(os.path.join(out_dir, "tone_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_table(summary):
    print(f"\n6-way tone probe (pooled, ground-truth): "
          f"full {summary['full']['acc_mean']*100:.2f} +- {summary['full']['acc_std']*100:.2f} | "
          f"EMA {summary['ema']['acc_mean']*100:.2f} +- {summary['ema']['acc_std']*100:.2f} | "
          f"chance {summary['chance']*100:.2f} | majority {summary['majority']*100:.2f}")
    print(f"\n{'tone':>5s} {'n':>5s} {'full recall':>13s} {'EMA recall':>13s}")
    for k in CANTONESE_TONES:
        ks = str(k)
        f = summary["full"]["per_tone_recall"][ks]
        e = summary["ema"]["per_tone_recall"][ks]
        n = summary["tone_distribution"].get(ks, 0)
        print(f"{k:>5d} {n:>5d} {f[0]*100:>12.1f}% {e[0]*100:>12.1f}%")


if __name__ == "__main__":
    print_table(run())
