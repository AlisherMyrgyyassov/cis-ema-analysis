"""Run Experiment 3 (probing + faithfulness) on the Exp-1 full + EMA-only models."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RUNS_DIR, RESULTS_DIR, GROUP_IDS, EMA_GROUPS
from src.pipeline import prepare_data
from src.exp3_probing import run as run_probing
from src.exp3_faithfulness import run as run_faithfulness
from src.plotting import plot_probing, plot_faithfulness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--full-results", default=os.path.join(RUNS_DIR, "exp1", "full_results.json"))
    ap.add_argument("--ema-results", default=os.path.join(RUNS_DIR, "exp1", "ema_only_results.json"))
    ap.add_argument("--shapley", default=os.path.join(RUNS_DIR, "exp2", "shapley.json"))
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "exp3"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = args.data_dir or DATA_DIR
    records, col_index, subjects, sex_map, folds = prepare_data(data_dir)
    with open(args.full_results) as f:
        full_results = json.load(f)
    with open(args.ema_results) as f:
        ema_results = json.load(f)
    with open(args.shapley) as f:
        shapley_summary = json.load(f)

    probe_full = run_probing(records, col_index, folds, full_results,
                             out_dir=args.out_dir, device=args.device,
                             name="full", groups=GROUP_IDS)
    probe_ema = run_probing(records, col_index, folds, ema_results,
                            out_dir=args.out_dir, device=args.device,
                            name="ema", groups=EMA_GROUPS)
    faith_summary = run_faithfulness(records, col_index, folds, full_results,
                                     shapley_summary, out_dir=args.out_dir,
                                     device=args.device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig1 = plot_probing(probe_full, os.path.join(RESULTS_DIR, "exp3_probing_full.png"))
    fig2 = plot_probing(probe_ema, os.path.join(RESULTS_DIR, "exp3_probing_ema.png"))
    fig3 = plot_faithfulness(faith_summary, os.path.join(RESULTS_DIR, "exp3_faithfulness.png"))

    print("\n==== Experiment 3a probing: FULL model (accuracy %) ====")
    _print_probing(probe_full)
    print("\n==== Experiment 3a probing: EMA-only model (accuracy %) ====")
    _print_probing(probe_ema)
    print(f"\nfigures -> {fig1}, {fig2}, {fig3}")


def _print_probing(s):
    props = s["props"]
    stages = s["stages"]
    print("property      " + " ".join(f"{st:>10s}" for st in stages))
    for p in props:
        row = f"{p:14s} " + " ".join(
            f"{s['aggregate'][f'{p}__{st}']['acc_mean']*100:>10.2f}" for st in stages)
        print(row)
    ctrl = " ".join(f"{s['aggregate'][f'{props[0]}__{st}']['control_mean']*100:>10.2f}" for st in stages)
    print(f"{'(control)':14s} {ctrl}")


if __name__ == "__main__":
    main()
