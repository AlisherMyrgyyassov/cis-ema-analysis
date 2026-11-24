"""Medial-lateral (y-axis) ablation and y-only recoverability.

Reproduces ``expected/cis/y_analysis/robust_z_noy.json``. The original artifact shipped in
this repository had no producer script, so the values could not be checked; this script
regenerates it under the same schema.

The y-position, y-velocity and y-acceleration channels of all six coils (18 channels) are
removed, keeping the five geometry terms and every x/z channel: 59 -> 41 input dimensions.
Folds, seeds, architecture and training recipe are those of the six-coil EMA-only model.

    python scripts/run_y_analysis.py            # 6 folds x 2 seeds
    python scripts/run_y_analysis.py --fold 0   # smoke test

Outputs
-------
``runs/y_analysis/robust_z_noy.json``
``expected/cis/y_analysis/robust_z_noy.json``
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import DATA_DIR, FEATURE_GROUPS, RUNS_DIR, SEEDS
from src.exp1_ceilings import prepare
from src.pipeline import train_biLSTM_cv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = os.path.join(REPO, "expected", "cis", "y_analysis")


def noy_groups():
    """Register y-free variants of G1-G3 so columns_for() can resolve them."""
    out = []
    for g, tag in (("G1", "G1_noy"), ("G2", "G2_noy"), ("G3", "G3_noy")):
        cols = [c for c in FEATURE_GROUPS[g]["columns"] if not c.split("_")[0].endswith("y")]
        FEATURE_GROUPS[tag] = {"columns": cols, "label": f"{g} without y"}
        out.append(tag)
    return out + ["G4"]  # geometry kept: distances and angles, not y coordinates themselves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--fold", type=int, default=-1, help="-1 = all folds (smoke: 0)")
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "y_analysis"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    groups = noy_groups()
    n_cols = sum(len(FEATURE_GROUPS[g]["columns"]) for g in groups)
    print(f"y-free groups {groups} -> {n_cols} input channels (six-coil model has 59)")

    records, col_index, _subj, _sex, folds = prepare(args.data_dir)
    runs = train_biLSTM_cv(records, col_index, folds, groups,
                           [int(s) for s in args.seeds.split(",")],
                           args.out_dir, "robust_z_noy", device=args.device,
                           fold=None if args.fold < 0 else args.fold)

    acc = [100 * r["test_acc"] for r in runs]
    f1 = [100 * r["test_f1"] for r in runs]
    out = {
        "name": "robust_z_noy",
        "preset": "ema_noy",
        "n_channels": n_cols,
        "test_acc_mean": float(np.mean(acc)) / 100,
        "test_acc_std": float(np.std(acc)) / 100,
        "test_f1_mean": float(np.mean(f1)) / 100,
        "test_f1_std": float(np.std(f1)) / 100,
        "runs": runs,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for path in (os.path.join(args.out_dir, "robust_z_noy.json"),
                 os.path.join(EXPECTED, "robust_z_noy.json")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    print(f"robust_z_noy: {np.mean(acc):.2f} +- {np.std(acc):.2f} on {n_cols} channels")
    print("wrote", os.path.join(EXPECTED, "robust_z_noy.json"))


if __name__ == "__main__":
    main()
