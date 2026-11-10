"""Run Experiment 2 (SAGE Shapley) on the Exp-1.4 full models."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RUNS_DIR, RESULTS_DIR
from src.pipeline import prepare_data
from src.exp2_shapley import run as run_shapley
from src.plotting import plot_shapley


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--full-results", default=os.path.join(RUNS_DIR, "exp1", "full_results.json"))
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "exp2"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from src.config import DATA_DIR
    data_dir = args.data_dir or DATA_DIR

    records, col_index, subjects, sex_map, folds = prepare_data(data_dir)
    with open(args.full_results) as f:
        full_results = json.load(f)

    summary = run_shapley(records, col_index, folds, full_results,
                          out_dir=args.out_dir, device=args.device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig = plot_shapley(summary, os.path.join(RESULTS_DIR, "exp2_shapley.png"))

    print("\n==== Experiment 2 Shapley (accuracy points) ====")
    agg = summary["aggregate"]["shapley_acc"]
    for g in sorted(agg, key=lambda g: -agg[g]["mean"]):
        print(f"  {g:2s} {summary['names'][g]:22s} {agg[g]['mean']*100:+.2f} +- {agg[g]['std']*100:.2f}")
    print(f"figure -> {fig}")


if __name__ == "__main__":
    main()
