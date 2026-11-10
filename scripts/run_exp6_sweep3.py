"""Experiment 6 sweep, batch 3: controls + checkpoint-saving winner.

  ensC_a09_ck     - winning config (EMA-ensemble teacher, logit KD a=0.9) with
                    saved checkpoints, for subject-level paired inference.
  ensC_single_a09 - ablation: single EMA teacher (no ensemble).
  ensC_shuf_a09   - control: ensemble teacher with logits shuffled across
                    utterances (breaks pairing; tests regularization vs transfer).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import DATA_DIR, RUNS_DIR
from src.pipeline import prepare_data
from src import exp6_distill as E6

EMA = os.path.join(RUNS_DIR, "exp1", "ema_only")

CONFIGS = [
    {"name": "ensC_a09_ck",     "teacher_seeds": (0, 1), "T": 4.0, "alpha": 0.9, "lf": 0.0, "save_ckpts": True,  "shuffle": False},
    {"name": "ensC_single_a09", "teacher_seeds": (0,),   "T": 4.0, "alpha": 0.9, "lf": 0.0, "save_ckpts": False, "shuffle": False},
    {"name": "ensC_shuf_a09",   "teacher_seeds": (0, 1), "T": 4.0, "alpha": 0.9, "lf": 0.0, "save_ckpts": False, "shuffle": True},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
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

    rows = []
    for cfg in CONFIGS:
        if wanted and cfg["name"] not in wanted:
            continue
        out_dir = os.path.join(args.root, cfg["name"])
        print(f"\n########## {cfg['name']} tseeds={cfg['teacher_seeds']} "
              f"T={cfg['T']} a={cfg['alpha']} lf={cfg['lf']} shuf={cfg['shuffle']} ##########",
              flush=True)
        sj = os.path.join(out_dir, "summary.json")
        if os.path.isfile(sj):
            with open(sj) as f:
                summary = json.load(f)["summary"]
            print(f"  [skip] already done", flush=True)
        else:
            try:
                summary, runs = E6.train_kd_cv(
                    records, col_index, folds, EMA, out_dir, cfg["name"],
                    device=device, seed_list=seeds, temperature=cfg["T"],
                    alpha=cfg["alpha"], lambda_feat=cfg["lf"],
                    teacher_seeds=cfg["teacher_seeds"], epochs=args.epochs,
                    save_ckpts=cfg["save_ckpts"], shuffle_teacher=cfg["shuffle"])
            except Exception as e:
                print(f"  !! FAILED: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
        row = {"name": cfg["name"], **{k: cfg[k] for k in ("teacher_seeds", "T", "alpha", "lf", "shuffle")},
               "acc": summary.get("test_acc_mean"), "acc_std": summary.get("test_acc_std")}
        rows.append(row)
        print(f"  -> {cfg['name']}: acc {row['acc']*100:.2f} +/- {row['acc_std']*100:.2f}", flush=True)

    with open(os.path.join(args.root, "sweep3_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n===== SWEEP 3 =====")
    for r in rows:
        print(f"  {r['name']:18s} tseeds={r['teacher_seeds']} a={r['alpha']} shuf={r['shuffle']} "
              f"acc {r['acc']*100:.2f} +/- {r['acc_std']*100:.2f}")


if __name__ == "__main__":
    main()
