"""Run Experiment 1.5 SSL audio ceiling (wav2vec2 / HuBERT / XLSR)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RAW_DIR, RUNS_DIR, RESULTS_DIR
import src.exp1_ssl as ssl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=os.path.join(RUNS_DIR, "exp1_ssl"))
    ap.add_argument("--model", default="auto",
                    help="'auto' tries chinese-hubert-large -> xlsr-53 -> mandarin-w2v2 -> w2v2-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    results = ssl.run(
        data_dir=args.data_dir, raw_dir=args.raw_dir, out_dir=args.out_dir,
        model_arg=args.model, seeds=[int(s) for s in args.seeds.split(",")],
        fold=args.fold, device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, patience=args.patience)

    if results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "exp1_ssl.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
