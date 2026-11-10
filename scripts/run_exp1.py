"""Run Experiment 1 ceilings (classical + EMA-only + full BiLSTM)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RUNS_DIR, RESULTS_DIR, SEEDS
import src.exp1_ceilings as exp1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "exp1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--fold", type=int, default=-1, help="-1 = all folds (smoke: 0)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    fold = None if args.fold < 0 else args.fold

    master, details = exp1.run(
        args.data_dir, out_dir=args.out_dir, device=args.device,
        fold=fold, seeds=seeds, epochs=args.epochs,
        num_layers=args.layers, hidden=args.hidden)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "exp1_ceilings.json"), "w") as f:
        json.dump(master, f, indent=2)

    # save full-model run details (ckpt paths) for Exp 2/3/4
    with open(os.path.join(args.out_dir, "full_results.json"), "w") as f:
        json.dump(details["full"], f, indent=2)
    with open(os.path.join(args.out_dir, "ema_only_results.json"), "w") as f:
        json.dump(details["ema_only"], f, indent=2)

    print("\n==== Experiment 1 ceilings ====")
    print(f"chance           {master['chance']['chance_acc']*100:.2f}%")
    print(f"majority         {master['chance']['majority_acc_mean']*100:.2f}%")
    print(f"LogReg (pooled)  {master['logistic_regression']['test_acc_mean']*100:.2f} +- {master['logistic_regression']['test_acc_std']*100:.2f}")
    print(f"RF (pooled)      {master['random_forest']['test_acc_mean']*100:.2f} +- {master['random_forest']['test_acc_std']*100:.2f}")
    print(f"BiLSTM EMA-only  {master['bilstm_ema_only']['test_acc_mean']*100:.2f} +- {master['bilstm_ema_only']['test_acc_std']*100:.2f}")
    print(f"BiLSTM full      {master['bilstm_full']['test_acc_mean']*100:.2f} +- {master['bilstm_full']['test_acc_std']*100:.2f}")


if __name__ == "__main__":
    main()
