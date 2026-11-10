"""Does the pooled representation retain tone once the syllable is known?

The direct-F0 ceiling (`scripts/f0_tone_baseline.py`) gives its probe the true
syllable identity, which the representation probes do not receive. This script
runs the matched comparison: the frozen pooled embedding of the EMA-only and full
models, concatenated with a one-hot syllable identity, on the same ten multi-tone
syllables and folds. If the representation is tone-poor even with the oracle, the
"tone is present in F0 but not in the pooled embedding" reading holds; if the
score jumps toward the F0 ceiling, the information is present but not linearly
accessible without the syllable.

Usage:
  python scripts/tone_syllable_oracle.py
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.config import RUNS_DIR, DATA_DIR                        # noqa: E402
from src.pipeline import prepare_data                            # noqa: E402
from src.data.dataset import records_for, view, fit_stats, standardize  # noqa: E402
from src.models.bilstm import SyllableBiLSTM                     # noqa: E402
from src.exp3_probing import extract_representations             # noqa: E402
import within_syllable_tone as wst                               # noqa: E402


def run_model(records, col_index, folds, groups, res, device, syls):
    keep = set(syls)
    rows = []
    for r in res:
        fd = folds[r["fold"]]
        tr = [x for x in records_for(records, fd["train"])
              if x["tone"] is not None and x["syllable"] in keep]
        te = [x for x in records_for(records, fd["test"])
              if x["tone"] is not None and x["syllable"] in keep]
        mu, sd = fit_stats(view(tr, col_index, groups))
        trs = standardize(view(tr, col_index, groups), mu, sd)
        tes = standardize(view(te, col_index, groups), mu, sd)
        ck = torch.load(r["ckpt_path"], map_location="cpu")
        model = SyllableBiLSTM(ck["input_dim"], ck["num_classes"])
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        xtr = extract_representations(model, trs, device)["pooled"]
        xte = extract_representations(model, tes, device)["pooled"]

        def onehot(rs):
            m = np.zeros((len(rs), len(syls)))
            for i, x in enumerate(rs):
                m[i, syls.index(x["syllable"])] = 1.0
            return m

        xtr = np.hstack([xtr, onehot(tr)])
        xte = np.hstack([xte, onehot(te)])
        ytr = np.array([x["tone"] for x in tr])
        yte = np.array([x["tone"] for x in te])
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(xtr, ytr)
        acc = float((clf.predict(xte) == yte).mean())
        maj = {}
        for x in tr:
            maj.setdefault(x["syllable"], {}).setdefault(x["tone"], 0)
            maj[x["syllable"]][x["tone"]] += 1
        maj = {s: max(d, key=d.get) for s, d in maj.items()}
        base = float(np.mean([maj.get(x["syllable"], 1) == x["tone"] for x in te]))
        rows.append({"fold": r["fold"], "seed": r["seed"], "probe": acc, "baseline": base})
        print(f"  [fold {r['fold']} seed {r['seed']}] {100*acc:.1f}")
    fa = np.array([x["probe"] for x in rows])
    fb = np.array([x["baseline"] for x in rows])
    return {"probe_mean": float(fa.mean()), "probe_std": float(fa.std(ddof=1)),
            "baseline_mean": float(fb.mean()), "diff": float((fa - fb).mean()),
            "runs": rows}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, col_index, _, _, folds = prepare_data(DATA_DIR)
    wst._attach_tones(records)
    tone_by_syl = {}
    for r in records:
        if r.get("tone") is not None:
            tone_by_syl.setdefault(r["syllable"], set()).add(r["tone"])
    syls = sorted(s for s, t in tone_by_syl.items() if len(t) >= 2)

    ema = json.load(open(os.path.join(RUNS_DIR, "exp1", "ema_only_results.json")))
    full = json.load(open(os.path.join(RUNS_DIR, "exp1", "full_results.json")))
    ema = ema["runs"] if isinstance(ema, dict) else ema
    full = full["runs"] if isinstance(full, dict) else full

    out = {"n_syllables": len(syls),
           "ema_syl": run_model(records, col_index, folds, ["G1", "G2", "G3", "G4"],
                                ema, device, syls),
           "full_syl": run_model(records, col_index, folds,
                                 ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"],
                                 full, device, syls)}
    path = os.path.join(ROOT, "runs", "f0_baseline", "tone_syllable_oracle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nEMA pooled + syllable  {100*out['ema_syl']['probe_mean']:.2f}")
    print(f"Full pooled + syllable {100*out['full_syl']['probe_mean']:.2f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
