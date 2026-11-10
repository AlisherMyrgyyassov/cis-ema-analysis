"""Calibration-size sweep for per-speaker normalization.

How many UNLABELED utterances from a speaker are needed for the robust_z
normalization stats to recover the full gain? Builds robust_z with stats from k
utterances/speaker (k = 1,2,3,5,10,20) and trains EMA-only under the locked CV.
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import torch

from src.config import RUNS_DIR, EMA_GROUPS
from src.pipeline import prepare_data, train_biLSTM_cv
from src.eval import summarize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = str(path("cis_dataset"))
KS = [1, 2, 3, 5, 10, 20]

spec = importlib.util.spec_from_file_location(
    "bz", os.path.join(HERE, "build_zscore_features.py"))
bz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", default=",".join(map(str, KS)),
                    help="comma list of calibration sizes; a subset may be run as "
                         "its own process (each writes calib<k>.json). "
                         "calib_summary.json is written once all of KS is present.")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    manifest = os.path.join(DATA, "manifest.csv")
    root = os.path.join(RUNS_DIR, "exp9_calib")
    os.makedirs(root, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x != ""]

    rows = []
    for k in ks:
        vdir = os.path.join(DATA, "variants", f"robust_z_calib{k}")
        if not os.path.isfile(os.path.join(vdir, "features.csv")):
            bz.build(manifest, DATA, vdir, "robust_z", 1, k)
        records, col_index, subjects, sex_map, folds = prepare_data(vdir)
        res = train_biLSTM_cv(records, col_index, folds, EMA_GROUPS, seeds,
                              root, f"calib{k}", device=device, epochs=args.epochs,
                              verbose=False)
        s = summarize(res)
        row = {"k": k, "acc": s["test_acc_mean"], "std": s["test_acc_std"]}
        rows.append(row)
        with open(os.path.join(root, f"calib{k}.json"), "w") as f:
            json.dump(row, f, indent=2)
        print(f"k={k:2d}: acc {s['test_acc_mean']*100:.2f} +- {s['test_acc_std']*100:.2f}", flush=True)

    # aggregate only from a complete sweep (a partial --ks run must not satisfy it)
    have = [json.load(open(os.path.join(root, f"calib{k}.json")))
            for k in KS if os.path.isfile(os.path.join(root, f"calib{k}.json"))]
    if len(have) == len(KS):
        have.sort(key=lambda r: r["k"])
        with open(os.path.join(root, "calib_summary.json"), "w") as f:
            json.dump(have, f, indent=2)
    print("\n===== CALIBRATION SWEEP (this process) =====")
    for r in rows:
        print(f"  k={r['k']:2d}  acc {r['acc']*100:.2f} +- {r['std']*100:.2f}")


if __name__ == "__main__":
    main()
