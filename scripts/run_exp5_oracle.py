"""Run the Exp-5 oracle: does the *true* utterance-mean MFCC_2/5/6 target add
syllable-discriminative signal beyond EMA-only?

  1. pooled oracle (LR/RF, EMA-only vs EMA + true means)
  2. BiLSTM oracle (EMA-only + true means broadcast as input channels)
     unless --pooled-only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RUNS_DIR, DATA_DIR, SEEDS
from src.exp5_oracle import run as run_oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "exp5_oracle"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--pooled-only", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    fold = None if args.fold < 0 else args.fold

    run_oracle(data_dir=args.data_dir, out_dir=args.out_dir, device=args.device,
               pooled_only=args.pooled_only, seeds=seeds, fold=fold,
               epochs=args.epochs)


if __name__ == "__main__":
    main()
