"""Experiment 6B sweep: cross-modal HuBERT KD variants into EMA-only.
Precomputes HuBERT targets once, then trains several configs."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import DATA_DIR, RUNS_DIR
from src.pipeline import prepare_data
from src import exp6_crossmodal as X

CONFIGS = [
    {"name": "xkd_a05",  "T": 4.0, "alpha": 0.5, "lf": 0.0, "lcon": 0.0},  # logit KD
    {"name": "xkd_a09",  "T": 4.0, "alpha": 0.9, "lf": 0.0, "lcon": 0.0},  # logit KD
    {"name": "xkd_feat", "T": 4.0, "alpha": 0.0, "lf": 1.0, "lcon": 0.0},  # feature MSE
    {"name": "xkd_con",  "T": 4.0, "alpha": 0.5, "lf": 0.0, "lcon": 0.5},  # contrastive
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--hubert-dir", default=os.path.join(RUNS_DIR, "exp1_ssl_hubert"))
    ap.add_argument("--cache-dir", default=os.path.join(RUNS_DIR, "exp6", "hubert_cache"))
    ap.add_argument("--root", default=os.path.join(RUNS_DIR, "exp6"))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, col_index, subjects, sex_map, folds = prepare_data(args.data_dir)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    wanted = None if not args.only else set(args.only.split(","))

    cache = X.precompute_hubert(records, folds, args.hubert_dir, args.cache_dir, device, seed=0)

    rows = []
    for cfg in CONFIGS:
        if wanted and cfg["name"] not in wanted:
            continue
        out_dir = os.path.join(args.root, cfg["name"])
        print(f"\n########## {cfg['name']} T={cfg['T']} a={cfg['alpha']} "
              f"lf={cfg['lf']} lcon={cfg['lcon']} ##########", flush=True)
        sj = os.path.join(out_dir, "summary.json")
        if os.path.isfile(sj):
            with open(sj) as f:
                summary = json.load(f)["summary"]
            print("  [skip] done", flush=True)
        else:
            try:
                summary, runs = X.train_crossmodal(
                    records, col_index, folds, cache, out_dir, cfg["name"], device=device,
                    seed_list=seeds, temperature=cfg["T"], alpha=cfg["alpha"],
                    lambda_feat=cfg["lf"], lambda_con=cfg["lcon"], epochs=args.epochs)
            except Exception as e:
                print(f"  !! FAILED: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
        row = {"name": cfg["name"], **cfg, "acc": summary.get("test_acc_mean"),
               "acc_std": summary.get("test_acc_std"), "f1": summary.get("test_f1_mean")}
        rows.append(row)
        print(f"  -> {cfg['name']}: acc {row['acc']*100:.2f} +/- {row['acc_std']*100:.2f}", flush=True)

    with open(os.path.join(args.root, "sweep_xkd_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n===== SWEEP XKD =====")
    for r in rows:
        print(f"  {r['name']:10s} T={r['T']} a={r['alpha']} lf={r['lf']} lcon={r['lcon']} "
              f"acc {r['acc']*100:.2f} +/- {r['acc_std']*100:.2f}")


if __name__ == "__main__":
    main()
