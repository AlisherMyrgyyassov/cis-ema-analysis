"""Experiment 3a: probing classifiers across model layers (+ controls).

Freezes the trained full-feature BiLSTM (Exp 1.4) and the EMA-only BiLSTM
(Exp 1.3), extracts mean-pooled utterance representations at each depth (input
projection, BiLSTM layer 0, layer 1, final pooled embedding), and trains linear
probes (logistic regression) to decode phonetic properties:

    onset (15)  vowel (7)  place-of-articulation (~6)  manner (sonorant/obstruent, 2)
    tone (6, ground-truth from stimuli.xlsx, item 114 excluded)

The layer names in the plan (GraphNet / TempConv / ArticFusion / BiLSTM) refer
to the deprecated v1 architecture; the canonical Exp-0.4 model is the V2
BiLSTM, so the four stages here are proj / layer0 / layer1 / pooled.

Control task: probes trained on randomly shuffled labels (~chance) validate
that probe accuracy reflects encoded information, not probe overfitting.
"""
import os
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import (ONSET_PLACE, ONSET_SONORANT, RUNS_DIR)
from src.models.bilstm import SyllableBiLSTM
from src.data.tone import add_tone_to_records


def extract_representations(model, seqs, device, batch_size=256):
    """Mean-pooled utterance representations at each depth."""
    model.eval()
    out = {"proj": [], "layer0": [], "layer1": [], "pooled": []}
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            lengths = torch.tensor([s.shape[0] for s in chunk], dtype=torch.long)
            max_len = int(lengths.max())
            F = chunk[0].shape[1]
            x = torch.zeros(len(chunk), max_len, F)
            for b, s in enumerate(chunk):
                x[b, :s.shape[0]] = torch.from_numpy(s)
            _, hidden = model(x.to(device), lengths.to(device), return_hidden=True)
            mask = (torch.arange(max_len, device=device).unsqueeze(0)
                    < lengths.to(device).unsqueeze(1)).float()
            for stage in ("proj", "layer0", "layer1"):
                h = hidden[stage]
                pooled = (h * mask.unsqueeze(-1)).sum(1) / lengths.to(device).unsqueeze(1).float()
                out[stage].append(pooled.cpu().numpy())
            out["pooled"].append(hidden["pooled"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _property_labels(records, prop):
    if prop == "tone":
        recs = [r for r in records if r.get("tone") is not None]
        return [r["tone"] for r in recs], recs
    if prop == "onset":
        vals = [r["onset"] for r in records]
    elif prop == "vowel":
        vals = [r["vowel"] for r in records]
    elif prop == "place":
        vals = [ONSET_PLACE.get(r["onset"], "other") for r in records]
    elif prop == "manner":
        vals = [int(ONSET_SONORANT.get(r["onset"], False)) for r in records]
    else:
        raise ValueError(prop)
    return vals, records


def _probe(X_tr, y_tr, X_te, y_te, control=False):
    if control:
        rng = np.random.RandomState(0)
        y_tr = rng.permutation(y_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def run(records, col_index, folds, model_results, out_dir=None, device="cuda",
        num_layers=2, hidden=128, dropout=0.4, name="full", groups=None,
        columns=None):
    device = torch.device(device)
    out_dir = out_dir or os.path.join(RUNS_DIR, "exp3")
    os.makedirs(out_dir, exist_ok=True)

    from src.data.dataset import records_for, view, fit_stats, standardize, select
    from src.config import GROUP_IDS
    if groups is None:
        groups = GROUP_IDS

    def _seqs(recs):
        """Feature tensors by explicit channel list, else by group ids."""
        if columns is not None:
            return [select(r, col_index, columns) for r in recs]
        return view(recs, col_index, groups)

    add_tone_to_records(records)   # ground-truth tone from stimuli.xlsx
    props = ["onset", "vowel", "place", "manner", "tone"]

    stages = ["proj", "layer0", "layer1", "pooled"]
    per_run = []
    for r in model_results:
        fold_id, seed = r["fold"], r["seed"]
        fd = folds[fold_id]
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])

        tr_seqs = _seqs(tr)
        te_seqs = _seqs(te)
        mu, sd = fit_stats(tr_seqs)
        tr_std = standardize(tr_seqs, mu, sd)
        te_std = standardize(te_seqs, mu, sd)

        ckpt = torch.load(r["ckpt_path"], map_location="cpu")
        model = SyllableBiLSTM(ckpt["input_dim"], ckpt["num_classes"],
                               num_layers=num_layers, hidden=hidden, dropout=dropout)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        X_tr = extract_representations(model, tr_std, device)
        X_te = extract_representations(model, te_std, device)

        # tone probing uses only utterances with a (non-excluded) tone label
        tr_tone = [rec for rec in tr if rec.get("tone") is not None]
        te_tone = [rec for rec in te if rec.get("tone") is not None]
        tr_std_tone = standardize(_seqs(tr_tone), mu, sd)
        te_std_tone = standardize(_seqs(te_tone), mu, sd)
        X_tr_tone = extract_representations(model, tr_std_tone, device)
        X_te_tone = extract_representations(model, te_std_tone, device)

        run_result = {"fold": fold_id, "seed": seed}
        for prop in props:
            if prop == "tone":
                y_tr = [rec["tone"] for rec in tr_tone]
                y_te = [rec["tone"] for rec in te_tone]
                Xp_tr, Xp_te = X_tr_tone, X_te_tone
            else:
                y_tr, _ = _property_labels(tr, prop)
                y_te, _ = _property_labels(te, prop)
                Xp_tr, Xp_te = X_tr, X_te
            le = LabelEncoder().fit(y_tr)
            y_tr_i = le.transform(y_tr)
            y_te_i = le.transform(y_te)
            for stage in stages:
                acc = _probe(Xp_tr[stage], y_tr_i, Xp_te[stage], y_te_i)
                ctrl = _probe(Xp_tr[stage], y_tr_i, Xp_te[stage], y_te_i, control=True)
                run_result[f"{prop}__{stage}"] = {"acc": acc, "control": ctrl,
                                                  "selectivity": acc - ctrl}
        per_run.append(run_result)
        print(f"[fold {fold_id} seed {seed}] probe done")

    # aggregate
    agg = {}
    for prop in props:
        for stage in stages:
            accs = [r[f"{prop}__{stage}"]["acc"] for r in per_run]
            ctrls = [r[f"{prop}__{stage}"]["control"] for r in per_run]
            agg[f"{prop}__{stage}"] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "control_mean": float(np.mean(ctrls)),
                "selectivity_mean": float(np.mean([a - c for a, c in zip(accs, ctrls)])),
            }

    summary = {
        "name": name,
        "stages": stages, "props": props,
        "tone": {"source": "stimuli.xlsx (ground truth, item 114 excluded)",
                 "n_classes": 6},
        "aggregate": agg, "runs": per_run,
    }
    with open(os.path.join(out_dir, f"probing_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
