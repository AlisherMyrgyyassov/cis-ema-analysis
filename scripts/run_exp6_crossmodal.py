"""Run Experiment 6B (cross-modal distillation from HuBERT into EMA-only)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import DATA_DIR, RUNS_DIR
from src.pipeline import prepare_data
from src import exp6_crossmodal as X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--hubert-dir", default=os.path.join(RUNS_DIR, "exp1_ssl_hubert"))
    ap.add_argument("--cache-dir", default=os.path.join(RUNS_DIR, "exp6", "hubert_cache"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default="xmodal")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lambda-feat", type=float, default=0.0)
    ap.add_argument("--lambda-con", type=float, default=0.0)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--pc-seed", type=int, default=0, help="HuBERT checkpoint seed for targets")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, col_index, subjects, sex_map, folds = prepare_data(args.data_dir)

    cache = X.precompute_hubert(records, folds, args.hubert_dir, args.cache_dir,
                                device, seed=args.pc_seed)

    out_dir = args.out_dir or os.path.join(RUNS_DIR, "exp6", args.name)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    summary, runs = X.train_crossmodal(
        records, col_index, folds, cache, out_dir, args.name, device=device,
        seed_list=seeds, temperature=args.temperature, alpha=args.alpha,
        lambda_feat=args.lambda_feat, lambda_con=args.lambda_con, epochs=args.epochs)
    print("\n==== Exp6B cross-modal summary ====")
    print(f"  T={args.temperature} a={args.alpha} lf={args.lambda_feat} lc={args.lambda_con}")
    print(f"  test acc {summary['test_acc_mean']*100:.2f} +- {summary['test_acc_std']*100:.2f} "
          f"f1 {summary['test_f1_mean']*100:.2f} n={summary['n']}")


if __name__ == "__main__":
    main()
