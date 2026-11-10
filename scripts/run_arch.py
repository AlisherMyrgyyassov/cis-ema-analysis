"""Train architecture baselines (BiLSTM / Transformer / TCN) under the locked
subject-disjoint 6-fold CV, for EMA-only and full feature sets, on the main
representation (variants/robust_z). Resumable: skips (model,view) already saved.

Usage:
  python scripts/run_arch.py --models transformer,tcn --views ema,full --seeds 0,1
"""
import argparse
import json
import os
import sys

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import EMA_GROUPS, ACOUSTIC_GROUPS          # noqa: E402
from src.pipeline import prepare_data                        # noqa: E402
from src.data.dataset import view, fit_stats, standardize, records_for  # noqa: E402
from src.eval import classification_metrics                  # noqa: E402
from src.training import make_loader, train_model, evaluate_syl  # noqa: E402
from src.models.bilstm import SyllableBiLSTM                 # noqa: E402
from src.models.transformer import TransformerClassifier     # noqa: E402
from src.models.tcn import TCNClassifier                     # noqa: E402

DATA = str(path("cis_dataset"))
MODELS = {"bilstm": SyllableBiLSTM, "transformer": TransformerClassifier, "tcn": TCNClassifier}
VIEWS = {"ema": EMA_GROUPS, "full": EMA_GROUPS + ACOUSTIC_GROUPS}


def run(records, col_index, folds, groups, model_cls, seeds, device, epochs,
        save_ckpt=False, ckpt_dir=None, tag=""):
    results = []
    for fold_id, fd in enumerate(folds):
        tr = records_for(records, fd["train"]); va = records_for(records, fd["val"]); te = records_for(records, fd["test"])
        le = LabelEncoder().fit([r["syllable"] for r in tr])
        y_tr = le.transform([r["syllable"] for r in tr])
        y_va = le.transform([r["syllable"] for r in va])
        y_te = le.transform([r["syllable"] for r in te])
        tr_s = view(tr, col_index, groups); va_s = view(va, col_index, groups); te_s = view(te, col_index, groups)
        mu, sd = fit_stats(tr_s)
        tr_s, va_s, te_s = standardize(tr_s, mu, sd), standardize(va_s, mu, sd), standardize(te_s, mu, sd)
        cw = np.bincount(y_tr, minlength=len(le.classes_)).astype(np.float32)
        cw = torch.tensor(cw.sum() / (len(le.classes_) * cw), dtype=torch.float32).to(device)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = model_cls(tr_s[0].shape[1], len(le.classes_)).to(device)
            model, best_val = train_model(model, make_loader(tr_s, y_tr, 32, True),
                                          make_loader(va_s, y_va, 32, False), device,
                                          epochs=epochs, class_weight=cw, verbose=False)
            p, t = evaluate_syl(model, make_loader(te_s, y_te, 32, False), device)
            m = classification_metrics(t, p, classes=le.classes_.tolist())
            ckpt_path = None
            if save_ckpt:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{tag}_fold{fold_id}_seed{seed}.pth")
                torch.save({"model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                            "input_dim": int(tr_s[0].shape[1]), "num_classes": len(le.classes_),
                            "classes": le.classes_.tolist(), "fold": fold_id, "seed": seed}, ckpt_path)
            results.append({"fold": fold_id, "seed": seed, "val_acc": float(best_val),
                            "test_acc": float(m["accuracy"]), "test_f1": float(m["macro_f1"]),
                            "ckpt_path": ckpt_path})
            print(f"  [{fold_id}/{seed}] val {best_val:.4f} test {m['accuracy']:.4f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="transformer,tcn")
    ap.add_argument("--views", default="ema,full")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", default="robust_z")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "arch"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, col_index, _, _, folds = prepare_data(os.path.join(DATA, "variants", args.variant))

    for model_name in args.models.split(","):
        for vname in args.views.split(","):
            tag = f"{model_name}_{vname}"
            out_json = os.path.join(args.out, f"{tag}.json")
            if os.path.exists(out_json) and not args.force:
                print(f"skip {tag} (exists)"); continue
            print(f"=== {tag} ===", flush=True)
            res = run(records, col_index, folds, VIEWS[vname], MODELS[model_name], seeds, device,
                      args.epochs, save_ckpt=args.save_ckpt,
                      ckpt_dir=os.path.join(args.out, "ckpt"), tag=tag)
            acc = np.mean([r["test_acc"] for r in res])
            print(f"{tag}: {acc*100:.2f} +- {np.std([r['test_acc'] for r in res])*100:.2f}", flush=True)
            with open(out_json, "w") as f:
                json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
