"""Run the Exp-5 subject-level paired re-analysis (no retraining)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR
from src.exp5_subject_stats import run as run_subject_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_subject_stats(data_dir=args.data_dir, out_dir=args.out_dir,
                      device=args.device)


if __name__ == "__main__":
    main()
