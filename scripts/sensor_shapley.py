"""Sensor-level exact Shapley on the EMA-only model (montage design).

Groups the 59 EMA channels by SENSOR (TD, TL, TT, UL, LL, LC) + geometry, and
computes exact Shapley values (2^7 subsets) on the robust_z EMA-only checkpoints.
Answers: which coils carry the predictive credit?
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import torch

from src.config import RUNS_DIR, EMA_COLUMNS, SENSORS, FEATURE_GROUPS
from src.pipeline import prepare_data
from src.data.dataset import records_for, view, fit_stats, standardize
from src.models.bilstm import SyllableBiLSTM
from src.exp2_shapley import eval_masked, _shapley_weights

DATA = os.path.join(str(path("cis_dataset")), "variants", "robust_z")
GROUPS = ["G1", "G2", "G3", "G4"]


def sensor_groups():
    g = {}
    for s in SENSORS:
        g[s] = [c for c in EMA_COLUMNS if c.startswith(s)]
    g["geometry"] = list(FEATURE_GROUPS["G4"]["columns"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="robust_z")
    ap.add_argument("--out", default=os.path.join(RUNS_DIR, "exp8_robust_z", "sensor_shapley.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, col_index, subjects, sex_map, folds = prepare_data(
        os.path.join(str(path("cis_dataset")), "variants", args.variant))
    runs = json.load(open(os.path.join(RUNS_DIR, "exp7", args.variant, "summary.json")))["runs"]
    sg = sensor_groups()
    names = list(sg.keys())
    # column positions within the 50-dim canonical EMA vector (config order)
    pos = {n: [EMA_COLUMNS.index(c) for c in sg[n]] for n in names}
    weights = _shapley_weights(len(names))
    per_run = []
    for run in runs:
        fd = folds[run["fold"]]
        tr = records_for(records, fd["train"]); te = records_for(records, fd["test"])
        tr_seqs = view(tr, col_index, GROUPS); te_seqs = view(te, col_index, GROUPS)
        mu, sd = fit_stats(tr_seqs)
        te_std = standardize(te_seqs, mu, sd)
        ck = torch.load(run["ckpt_path"], map_location="cpu", weights_only=False)
        cls2i = {c: i for i, c in enumerate(ck["classes"])}
        y = [cls2i[r["syllable"]] for r in te]
        model = SyllableBiLSTM(ck["input_dim"], ck["num_classes"])
        model.load_state_dict(ck["model"]); model.to(device).eval()
        n = len(names); F = te_std[0].shape[1]
        vals = np.zeros(2 ** n)
        for s in range(2 ** n):
            mask = np.ones(F, bool)
            for i, nm in enumerate(names):
                if not (s >> i) & 1:
                    mask[pos[nm]] = False
            acc, _ = eval_masked(model, te_std, y, mask, device)
            vals[s] = acc
        shap = np.zeros(n)
        for i in range(n):
            bit = 1 << i
            for s in range(2 ** n):
                if s & bit:
                    continue
                shap[i] += weights[bin(s).count("1")] * (vals[s | bit] - vals[s])
        per_run.append({"fold": run["fold"], "seed": run["seed"],
                        "full": float(vals[-1]), "empty": float(vals[0]),
                        "shap": {nm: float(shap[i]) for i, nm in enumerate(names)}})
    agg = {nm: {"mean": float(np.mean([r["shap"][nm] for r in per_run])),
                "std": float(np.std([r["shap"][nm] for r in per_run]))} for nm in names}
    out = {"variant": args.variant, "groups": names, "aggregate": agg, "runs": per_run}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("sensor-level Shapley (accuracy points):")
    for nm in sorted(agg, key=lambda x: -agg[x]["mean"]):
        print(f"  {nm:9s} {agg[nm]['mean']*100:+6.2f} +- {agg[nm]['std']*100:.2f}")


if __name__ == "__main__":
    main()
