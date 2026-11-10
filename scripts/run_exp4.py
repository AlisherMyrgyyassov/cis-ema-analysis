"""Run Experiment 4 (mechanistic correlation)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RUNS_DIR, RESULTS_DIR
from src.pipeline import prepare_data
from src.exp4_correlation import run as run_correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--full-results", default=os.path.join(RUNS_DIR, "exp1", "full_results.json"))
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "exp4"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    data_dir = args.data_dir or DATA_DIR
    records, col_index, subjects, sex_map, folds = prepare_data(data_dir)
    with open(args.full_results) as f:
        full_results = json.load(f)

    summary = run_correlation(records, col_index, folds, full_results,
                              out_dir=args.out_dir, device=args.device,
                              top_k=args.top_k)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "exp4_correlation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    from src.plotting import plot_mechanistic_scatter
    fig = plot_mechanistic_scatter(summary, os.path.join(RESULTS_DIR, "exp4_mechanistic.png"))

    print("\n==== Experiment 4 mechanistic correlation ====")
    print(f"{'acoustic':12s} {'LOFO%':>6s} {'best EMA':>16s} {'|r|':>7s} {'|r|^2':>7s} "
          f"{'1senR2':>7s} {'jointR2':>8s} {'jointXS':>8s}")
    for row in summary["table"]:
        print(f"{row['acoustic_feature']:12s} {row['lofo_importance']*100:>5.2f}% "
              f"{row['best_ema']:>16s} {row['best_abs_r']:>7.3f} {row['best_r2']:>7.3f} "
              f"{row['single_ridge_r2_in']:>7.3f} {row['joint_ridge_r2_in']:>8.3f} "
              f"{row['joint_ridge_r2_xs']:>8.3f}")
    print(f"figure -> {fig}")


if __name__ == "__main__":
    main()
