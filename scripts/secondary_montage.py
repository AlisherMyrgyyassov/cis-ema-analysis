"""Secondary-task check for the minimal montage (midsag4 vs all six coils).

The minimal-montage result (``runs/montage_subset/midsag4.json``) only measures
syllable classification. A device recommendation also needs to know whether the
dropped coils (TL, LC) carried information used by the secondary analyses:

  * phonetic probing (onset, vowel) at the pooled stage, with shuffled controls;
  * the syllable-controlled lexical-tone probe (same 10 multi-tone syllables);
  * a per-place error breakdown (bilabial, labiodental, ...), which is where a
    lip-corner coil would plausibly matter.

The four-coil checkpoints must have been produced with
``scripts/run_montage_subset.py --sets midsag4 --save-ckpt --save-preds``.
Baselines are the existing six-coil artifacts (same folds, same probe recipe).

Usage:
  python scripts/run_montage_subset.py --sets midsag4 --seeds 0,1 \
      --save-ckpt --save-preds --out runs/montage_probe
  python scripts/secondary_montage.py --subset midsag4
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.config import DATA_DIR, ONSET_PLACE, RUNS_DIR          # noqa: E402
from src.pipeline import prepare_data                          # noqa: E402
from src.stats import paired_compare                           # noqa: E402
from src.exp3_probing import run as run_probing                # noqa: E402
import within_syllable_tone as wst                             # noqa: E402
from run_montage_subset import columns_keep, MINIMAL_SUBSETS   # noqa: E402

PLACES = ["bilabial", "labiodental", "alveolar", "velar", "glottal", "glide"]


def fold_means(runs, getter):
    d = {}
    for r in runs:
        d.setdefault(r["fold"], []).append(getter(r))
    return {f: float(np.mean(v)) for f, v in d.items()}


def compare(means_a, means_b):
    folds = sorted(set(means_a) & set(means_b))
    return paired_compare([means_a[f] for f in folds],
                          [means_b[f] for f in folds])


def place_table(pred_paths, records):
    """Aggregate the 12 prediction files into one confusion matrix."""
    classes = sorted({r["syllable"] for r in records})
    idx = {c: i for i, c in enumerate(classes)}
    onset = {}
    for r in records:
        onset.setdefault(r["syllable"], r["onset"])
    cm = np.zeros((len(classes), len(classes)), dtype=float)
    for path in pred_paths:
        z = np.load(path)
        cl = [str(c) for c in z["classes"]]
        for t, p in zip(z["true"], z["pred"]):
            cm[idx[cl[int(t)]], idx[cl[int(p)]]] += 1
    out = []
    for place in PLACES:
        cls = [c for c in classes if ONSET_PLACE.get(onset[c]) == place]
        tot = sum(cm[idx[c]].sum() for c in cls)
        # place-level accuracy: any class within the same place counts as correct
        corr = sum(cm[idx[c], idx[d]] for c in cls for d in cls)
        out.append({"place": place, "n_syllables": len(cls),
                    "acc": round(100 * corr / tot, 1) if tot else None})
    return out, round(100 * np.trace(cm) / cm.sum(), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="midsag4")
    ap.add_argument("--runs", default=None,
                    help="results json with ckpt_path/pred_path (default: "
                         "runs/montage_probe/<subset>.json)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "secondary"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-probing", action="store_true",
                    help="redo the probing even if its json already exists")
    args = ap.parse_args()

    runs_path = args.runs or os.path.join(ROOT, "runs", "montage_probe",
                                          f"{args.subset}.json")
    with open(runs_path, encoding="utf-8") as f:
        runs4 = json.load(f)
    if not runs4 or "ckpt_path" not in runs4[0]:
        raise SystemExit("no checkpoints in the results json; rerun "
                         "run_montage_subset.py with --save-ckpt --save-preds")

    cols = columns_keep(MINIMAL_SUBSETS[args.subset])
    os.makedirs(args.out, exist_ok=True)
    records, col_index, _, _, folds = prepare_data(DATA_DIR)
    wst._attach_tones(records)

    # ---- probing (onset/vowel, pooled stage is the one we use) -----------
    probe_path = os.path.join(args.out, f"probing_{args.subset}.json")
    if os.path.exists(probe_path) and not args.force_probing:
        with open(probe_path, encoding="utf-8") as f:
            probe4 = json.load(f)
        print(f"reusing {probe_path}")
    else:
        probe4 = run_probing(records, col_index, folds, runs4, out_dir=args.out,
                             name=args.subset, columns=cols, device=args.device)

    # ---- syllable-controlled tone ---------------------------------------
    tone_by_syl = {}
    for r in records:
        if r.get("tone") is not None:
            tone_by_syl.setdefault(r["syllable"], set()).add(r["tone"])
    multi = sorted(s for s, ts in tone_by_syl.items() if len(ts) >= 2)
    within4 = wst._summarise(records, col_index, folds, None, set(multi),
                             runs4, args.device, columns=cols)
    naive4 = wst._summarise(records, col_index, folds, None, None,
                            runs4, args.device, columns=cols)

    # ---- per-place errors ------------------------------------------------
    pred_paths = [r["pred_path"] for r in runs4 if "pred_path" in r]
    places4, overall4 = place_table(pred_paths, records)

    # ---- six-coil baselines ---------------------------------------------
    base_probe_path = os.path.join(RUNS_DIR, "exp3", "probing_ema.json")
    base_tone_path = os.path.join(RUNS_DIR, "tone", "tone_within_syllable.json")
    base_group_path = os.path.join(ROOT, "paper", "results", "confusion_grouped.json")
    out = {"subset": args.subset, "channels": len(cols),
           "n_runs": len(runs4), "overall_acc": overall4,
           "per_place": places4, "tone": {"within": within4, "naive": naive4}}

    if os.path.exists(base_probe_path):
        with open(base_probe_path, encoding="utf-8") as f:
            base_probe = json.load(f)
        for prop in ("onset", "vowel"):
            key = f"{prop}__pooled"
            m4 = fold_means(probe4["runs"], lambda r, k=key: r[k]["acc"])
            m6 = fold_means(base_probe["runs"], lambda r, k=key: r[k]["acc"])
            out[f"{prop}_pooled"] = {
                "four_coil": round(100 * np.mean(list(m4.values())), 2),
                "six_coil": round(100 * np.mean(list(m6.values())), 2),
                "delta": round(100 * (np.mean(list(m4.values())) - np.mean(list(m6.values()))), 2),
                "test": compare(m4, m6),
            }

    if os.path.exists(base_tone_path):
        with open(base_tone_path, encoding="utf-8") as f:
            base_tone = json.load(f)["within_ema"]
        m4 = dict(enumerate(within4["per_fold_probe"]))
        m6 = dict(enumerate(base_tone["per_fold_probe"]))
        out["tone"]["four_coil"] = round(100 * within4["probe_mean"], 2)
        out["tone"]["six_coil"] = round(100 * base_tone["probe_mean"], 2)
        out["tone"]["delta"] = round(100 * (within4["probe_mean"] - base_tone["probe_mean"]), 2)
        out["tone"]["test_vs_six_coil"] = compare(m4, m6)

    if os.path.exists(base_group_path):
        with open(base_group_path, encoding="utf-8") as f:
            base_group = json.load(f)["models"]["ema_only"]
        out["per_place_six_coil"] = [
            {"place": p["place"], "acc": p["within_place_acc"]}
            for p in base_group["per_place"]]
        out["overall_acc_six_coil"] = base_group["overall_acc"]

    with open(os.path.join(args.out, f"{args.subset}_secondary.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---- console summary -------------------------------------------------
    print(f"\n{args.subset}: {len(cols)} channels, {len(runs4)} runs, "
          f"overall {overall4:.2f}% (six coils "
          f"{out.get('overall_acc_six_coil', float('nan')):.2f}%)")
    for prop in ("onset", "vowel"):
        if f"{prop}_pooled" in out:
            d = out[f"{prop}_pooled"]
            t = d["test"]
            print(f"  {prop:6s} 4-coil {d['four_coil']:.1f}  6-coil {d['six_coil']:.1f}  "
                  f"delta {d['delta']:+.1f} (p={t.get('paired_t_p', float('nan')):.3f})")
    tr = out["tone"]
    print(f"  tone (within-syllable) 4-coil {tr['four_coil']:.1f}  "
          f"6-coil {tr['six_coil']:.1f}  delta {tr['delta']:+.1f}  "
          f"(4-coil gain over baseline {100*tr['within']['diff']:+.1f}, "
          f"p={tr['within']['p']:.3f})")
    print("  per-place:")
    for p4, p6 in zip(places4, out.get("per_place_six_coil", [])):
        print(f"    {p4['place']:12s} 4-coil {p4['acc']:5.1f}  6-coil {p6['acc']:5.1f}  "
              f"delta {p4['acc'] - p6['acc']:+.1f}")
    print(f"wrote {os.path.join(args.out, f'{args.subset}_secondary.json')}")


if __name__ == "__main__":
    main()
