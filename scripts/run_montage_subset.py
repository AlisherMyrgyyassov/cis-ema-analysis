"""Retrained montage-subset sweep on the EMA-only task.

Unlike the frozen-model sensor Shapley, this RETRAINS the EMA-only BiLSTM with
coils removed (their position/velocity/acceleration channels, plus any geometry
term that involves them), to measure the accuracy cost of physically dropping
sensors. `all` = the full six-coil EMA set (59 channels).

Modes:
  (default) leave-one-coil-out: ``all`` + ``drop_<COIL>`` for the six coils.
  --sets     named multi-coil subsets (the minimal-montage question):
             ``midsag4`` {TD,TT,UL,LL}, ``midsag3`` {TD,TT,UL},
             ``min2`` {TD,TT}, ``td1`` {TD}.

Usage:
  python scripts/run_montage_subset.py --seeds 0,1
  python scripts/run_montage_subset.py --sets midsag4,midsag3,min2,td1
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

from src.config import AXES, SENSORS, GEOMETRY                # noqa: E402
from src.pipeline import prepare_data                         # noqa: E402
from src.data.dataset import fit_stats, standardize, records_for, select  # noqa: E402
from src.eval import classification_metrics                   # noqa: E402
from src.training import make_loader, train_model, evaluate_syl  # noqa: E402
from src.models.bilstm import SyllableBiLSTM                  # noqa: E402

DATA = str(path("cis_dataset"))
GEOM_DEPS = {
    "TT_TD_distance": ["TT", "TD"], "UL_LL_distance": ["UL", "LL"],
    "TT_LL_distance": ["TT", "LL"], "TT_TD_LL_angle": ["TT", "TD", "LL"],
    "UL_LL_LC_angle": ["UL", "LL", "LC"],
}


def columns_keep(coils):
    """Channels kept when only ``coils`` are mounted (geometry needs all deps)."""
    keep = set(coils)
    cols = []
    for s in SENSORS:
        if s not in keep:
            continue
        for a in AXES:
            cols += [f"{s}{a}", f"{s}{a}_velocity", f"{s}{a}_acceleration"]
    for g in GEOMETRY:
        if all(d in keep for d in GEOM_DEPS[g]):
            cols.append(g)
    return cols


def columns_drop(coil):
    return columns_keep([s for s in SENSORS if s != coil])


# Named multi-coil subsets, pre-specified rather than searched (see docs).
MINIMAL_SUBSETS = {
    "midsag4": ["TD", "TT", "UL", "LL"],
    "midsag3": ["TD", "TT", "UL"],
    "min2": ["TD", "TT"],
    "td1": ["TD"],
}


def run(records, col_index, folds, cols, seeds, device, epochs,
        save_ckpt=False, save_preds=False, out_dir=None, tag=""):
    out = []
    for fid, fd in enumerate(folds):
        tr = records_for(records, fd["train"]); va = records_for(records, fd["val"]); te = records_for(records, fd["test"])
        le = LabelEncoder().fit([r["syllable"] for r in tr])
        y = {k: le.transform([r["syllable"] for r in rs]) for k, rs in (("tr", tr), ("va", va), ("te", te))}
        v = {k: [select(r, col_index, cols) for r in rs] for k, rs in (("tr", tr), ("va", va), ("te", te))}
        mu, sd = fit_stats(v["tr"])
        v = {k: standardize(s, mu, sd) for k, s in v.items()}
        cw = np.bincount(y["tr"], minlength=len(le.classes_)).astype(np.float32)
        cw = torch.tensor(cw.sum() / (len(le.classes_) * cw), dtype=torch.float32).to(device)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = SyllableBiLSTM(v["tr"][0].shape[1], len(le.classes_)).to(device)
            model, bv = train_model(model, make_loader(v["tr"], y["tr"], 32, True),
                                    make_loader(v["va"], y["va"], 32, False), device,
                                    epochs=epochs, class_weight=cw, verbose=False)
            p, t = evaluate_syl(model, make_loader(v["te"], y["te"], 32, False), device)
            rec = {"fold": fid, "seed": seed, "test_acc": float((p == t).mean()),
                   "n_features": len(cols)}
            stem = f"{tag}_f{fid}_s{seed}"
            if save_ckpt:
                ckpt_dir = os.path.join(out_dir, "ckpt"); os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, stem + ".pt")
                torch.save({"model": {k: v_.cpu().clone() for k, v_ in model.state_dict().items()},
                            "input_dim": int(v["tr"][0].shape[1]),
                            "num_classes": int(len(le.classes_)),
                            "classes": le.classes_.tolist(),
                            "fold": fid, "seed": seed, "columns": cols}, ckpt_path)
                rec["ckpt_path"] = ckpt_path
            if save_preds:
                preds_dir = os.path.join(out_dir, "preds"); os.makedirs(preds_dir, exist_ok=True)
                pred_path = os.path.join(preds_dir, stem + ".npz")
                te_idx = np.where([r["subj"] in set(fd["test"]) for r in records])[0]
                np.savez(pred_path,
                         source=np.array([records[i]["source"] for i in te_idx]),
                         syllable=np.array([records[i]["syllable"] for i in te_idx]),
                         classes=np.array(le.classes_),
                         true=t, pred=p)
                rec["pred_path"] = pred_path
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "montage_subset"))
    ap.add_argument("--only", default=None,
                    help="comma list of subsets to run (default: all of them). "
                         "Lets the 7 leave-one-coil-out subsets run as parallel jobs.")
    ap.add_argument("--sets", default=None,
                    help="comma list of MINIMAL_SUBSETS names; when given, runs only "
                         "those multi-coil subsets instead of the LOO sweep.")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save one checkpoint per fold/seed for the probing analyses")
    ap.add_argument("--save-preds", action="store_true",
                    help="save per-utterance test predictions per fold/seed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    records, col_index, _, _, folds = prepare_data(os.path.join(DATA, "variants", "robust_z"))

    subsets = {"all": [c for c in col_index if not c.startswith(("F0", "F1", "F2", "F3", "MFCC", "energy", "voiced"))]}
    for coil in SENSORS:
        subsets[f"drop_{coil}"] = columns_drop(coil)

    if args.sets:
        wanted = [s for s in args.sets.split(",") if s]
        unknown = [s for s in wanted if s not in MINIMAL_SUBSETS]
        if unknown:
            ap.error(f"unknown subset(s): {unknown}; choose from {list(MINIMAL_SUBSETS)}")
        subsets = {name: columns_keep(MINIMAL_SUBSETS[name]) for name in wanted}

    for name, cols in subsets.items():
        if args.only and name not in set(args.only.split(",")):
            continue
        p = os.path.join(args.out, f"{name}.json")
        if os.path.exists(p):
            print(f"skip {name}"); continue
        print(f"=== {name} ({len(cols)} cols) ===", flush=True)
        res = run(records, col_index, folds, cols, seeds, device, args.epochs,
                  save_ckpt=args.save_ckpt, save_preds=args.save_preds,
                  out_dir=args.out, tag=name)
        a = np.mean([r["test_acc"] for r in res])
        print(f"{name}: {a*100:.2f} +- {np.std([r['test_acc'] for r in res])*100:.2f}", flush=True)
        with open(p, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
