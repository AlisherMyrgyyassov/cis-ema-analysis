"""Run Experiment 5 (MFCC-band auxiliary losses).

Modes:
  uttmean  (5a)  - pooled-embedding head predicts the per-utterance mean of
                   MFCC_2/5/6.  --frame-set all (headline) or voiced
                   (robustness against silence/onset dilution).
  perframe (5b)  - per-frame head on pre-pooling LSTM states predicts the
                   within-utterance z-scored MFCC_2/5/6 trajectory.

Every run records train/val/test aux R^2 and saves checkpoints.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RUNS_DIR, DATA_DIR, SEEDS
from src.exp5_mfcc_aux import run as run_exp5


def default_out_dir(mode, frame_set):
    if mode == "uttmean":
        return os.path.join(RUNS_DIR, f"exp5_{mode}_{frame_set}")
    return os.path.join(RUNS_DIR, "exp5_perframe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="uttmean", choices=["uttmean", "perframe"])
    ap.add_argument("--frame-set", default="all", choices=["all", "voiced"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lmbdas", default="0.3,1.0")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    lmbdas = [float(x) for x in args.lmbdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    fold = None if args.fold < 0 else args.fold
    out_dir = args.out_dir or default_out_dir(args.mode, args.frame_set)

    run_exp5(data_dir=args.data_dir, out_dir=out_dir, device=args.device,
             mode=args.mode, frame_set=args.frame_set, lmbdas=lmbdas,
             seeds=seeds, fold=fold, epochs=args.epochs)


if __name__ == "__main__":
    main()
