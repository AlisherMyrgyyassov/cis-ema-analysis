"""Tone analysis with the syllable confound controlled.

Motivation
----------
A naive 6-way tone probe is not interpretable on this corpus. Tone is heavily
confounded with syllable identity: 33 of the 43 syllables occur with a single
tone, so predicting tone from the syllable alone reaches ~90%. The naive probe
(full 80.3%, EMA 64.8%) therefore sits *below* the syllable-to-tone ceiling and
mostly measures syllable decoding, not tone.

This script reports, for the frozen Exp-1 full and EMA-only models:

  * naive 6-way tone probe (all utterances) and the syllable-to-tone ceiling;
  * a syllable-controlled probe restricted to the syllables that occur with
    >=2 tones, where syllable identity cannot determine the tone, together with
    the per-syllable majority baseline and paired fold-wise tests.

Writes ``runs/paper/tone/tone_within_syllable.json``.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.config import RUNS_DIR
from src.pipeline import prepare_data
from src.data.dataset import records_for, view, fit_stats, standardize, select
from src.data.tone import item_to_tone, file_num_from_source
from src.models.bilstm import SyllableBiLSTM
from src.exp3_probing import extract_representations

EXCLUDE = {114}
FULL = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
EMA = ["G1", "G2", "G3", "G4"]


def _attach_tones(records):
    tmap = item_to_tone()
    for r in records:
        fn = file_num_from_source(r["source"])
        r["tone"] = None if fn in EXCLUDE else tmap.get(fn)


def _probe_fold(records, col_index, folds, groups, subset, run, device,
                columns=None):
    fd = folds[run["fold"]]
    tr = [x for x in records_for(records, fd["train"])
          if x["tone"] is not None and (subset is None or x["syllable"] in subset)]
    te = [x for x in records_for(records, fd["test"])
          if x["tone"] is not None and (subset is None or x["syllable"] in subset)]
    if columns is not None:
        seqs = lambda rs: [select(r, col_index, columns) for r in rs]  # noqa: E731
    else:
        seqs = lambda rs: view(rs, col_index, groups)                  # noqa: E731
    mu, sd = fit_stats(seqs(tr))
    trs = standardize(seqs(tr), mu, sd)
    tes = standardize(seqs(te), mu, sd)
    ck = torch.load(run["ckpt_path"], map_location="cpu")
    model = SyllableBiLSTM(ck["input_dim"], ck["num_classes"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    X_tr = extract_representations(model, trs, device)["pooled"]
    X_te = extract_representations(model, tes, device)["pooled"]
    y_tr = np.array([x["tone"] for x in tr])
    y_te = np.array([x["tone"] for x in te])
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(X_tr, y_tr)
    acc = float((clf.predict(X_te) == y_te).mean())
    maj = {}
    for x in tr:
        maj.setdefault(x["syllable"], {}).setdefault(x["tone"], 0)
        maj[x["syllable"]][x["tone"]] += 1
    maj = {s: max(d, key=d.get) for s, d in maj.items()}
    base = float((np.array([maj.get(x["syllable"], 1) for x in te]) == y_te).mean())
    return run["fold"], acc, base


def _summarise(records, col_index, folds, groups, subset, res, device,
               columns=None):
    rows = [_probe_fold(records, col_index, folds, groups, subset, r, device,
                        columns=columns)
            for r in res]
    acc = {f: [] for f in range(6)}
    base = {f: [] for f in range(6)}
    for f, a, b in rows:
        acc[f].append(a)
        base[f].append(b)
    fa = np.array([np.mean(acc[f]) for f in range(6)])
    fb = np.array([np.mean(base[f]) for f in range(6)])
    t, p = stats.ttest_rel(fa, fb)
    d = (fa - fb).mean() / (fa - fb).std(ddof=1)
    return {
        "probe_mean": float(fa.mean()), "probe_std": float(fa.std(ddof=1)),
        "baseline_mean": float(fb.mean()),
        "diff": float((fa - fb).mean()), "t": float(t), "p": float(p),
        "cohens_d": float(d), "per_fold_probe": fa.tolist(),
        "per_fold_baseline": fb.tolist(),
    }


def run(data_dir=None, out_dir=None, device=None, full_res_path=None,
        ema_res_path=None):
    from src.config import DATA_DIR
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or os.path.join(RUNS_DIR, "tone")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    records, col_index, _, _, folds = prepare_data(data_dir)
    _attach_tones(records)

    tone_by_syl = {}
    for r in records:
        if r["tone"] is not None:
            tone_by_syl.setdefault(r["syllable"], set()).add(r["tone"])
    multi = sorted(s for s, ts in tone_by_syl.items() if len(ts) >= 2)
    n_multi = sum(1 for r in records if r["tone"] is not None and r["syllable"] in multi)

    full_res_path = full_res_path or os.path.join(RUNS_DIR, "exp1", "full_results.json")
    ema_res_path = ema_res_path or os.path.join(RUNS_DIR, "exp1", "ema_only_results.json")
    with open(full_res_path) as f:
        full_res = json.load(f)
    with open(ema_res_path) as f:
        ema_res = json.load(f)
    if isinstance(full_res, dict):
        full_res = full_res["runs"]
    if isinstance(ema_res, dict):
        ema_res = ema_res["runs"]

    summary = {
        "multi_tone_syllables": multi,
        "n_multi_utterances": n_multi,
        "n_all_utterances": sum(1 for r in records if r["tone"] is not None),
        "naive_full": _summarise(records, col_index, folds, FULL, None, full_res, device),
        "naive_ema": _summarise(records, col_index, folds, EMA, None, ema_res, device),
        "within_full": _summarise(records, col_index, folds, FULL, set(multi), full_res, device),
        "within_ema": _summarise(records, col_index, folds, EMA, set(multi), ema_res, device),
    }
    with open(os.path.join(out_dir, "tone_within_syllable.json"), "w") as f:
        json.dump(summary, f, indent=2)

    n = summary["naive_full"]
    wf, we = summary["within_full"], summary["within_ema"]
    print(f"naive   full {n['probe_mean']*100:.2f}  ema {summary['naive_ema']['probe_mean']*100:.2f}  "
          f"syllable-to-tone ceiling {n['baseline_mean']*100:.2f}")
    print(f"within  full {wf['probe_mean']*100:.2f} (d={wf['cohens_d']:+.2f}, p={wf['p']:.3f})  "
          f"ema {we['probe_mean']*100:.2f} (d={we['cohens_d']:+.2f}, p={we['p']:.3f})  "
          f"per-syllable baseline {wf['baseline_mean']*100:.2f}")
    return summary


if __name__ == "__main__":
    run()
