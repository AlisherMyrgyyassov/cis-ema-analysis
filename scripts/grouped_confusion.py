"""Grouped confusion analysis for the Exp-1 comparison (no retraining).

Reads the cached confusion matrices (paper/results/confusions.json), groups the
43 syllable classes by onset place of articulation (bilabial / labiodental /
alveolar / velar / glottal / glide), and reports, for the selected models:

  1. a 6x6 place-block confusion matrix (rows = true place, columns = predicted
     place, row-normalized %),
  2. per-place summary rows: within-place accuracy (block diagonal), the share
     of errors that stay within the same place, and the dominant *internal*
     syllable-pair confusion (e.g. bui<->pui, daai<->taai),
  3. the top global syllable confusions per model.

Also renders a grouped heatmap figure.

Outputs:
  paper/results/confusion_grouped.json
  paper/results/confusion_grouped.md
  paper/illustrations/confusion-matrix/confusion_grouped_places.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (DATA_DIR, RESULTS_DIR, ONSET_PLACE)
from src.data.dataset import load_dataset

CACHE = os.path.join(RESULTS_DIR, "confusions.json")
OUT_MD = os.path.join(RESULTS_DIR, "confusion_grouped.md")
OUT_JSON = os.path.join(RESULTS_DIR, "confusion_grouped.json")
OUT_ILLUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "paper", "illustrations", "confusion-matrix")
OUT_PNG = os.path.join(OUT_ILLUS, "confusion_grouped_places.png")

PLACES = ["bilabial", "labiodental", "alveolar", "velar", "glottal", "glide"]
MODELS = [
    ("ema_only", "BiLSTM EMA-only"),
    ("full", "BiLSTM full"),
    ("hubert", "HuBERT-large (SSL audio)"),
]


def class_onset_map():
    records, _ = load_dataset(DATA_DIR)
    m = {}
    for r in records:
        if r["syllable"] not in m:
            m[r["syllable"]] = r["onset"]
    return m


def grouped_cm(cm, names, idx, onset):
    """Return (G, tot) where G[p_true, p_pred] sums utterance counts."""
    G = np.zeros((len(PLACES), len(PLACES)))
    tot = np.zeros(len(PLACES))
    for c in names:
        p = ONSET_PLACE.get(onset[c], None)
        if p is None or p not in PLACES:
            continue
        i = idx[c]
        for cc in names:
            q = ONSET_PLACE.get(onset[cc], None)
            if q is None or q not in PLACES:
                continue
            G[PLACES.index(p), PLACES.index(q)] += cm[i, idx[cc]]
        tot[PLACES.index(p)] += cm[i].sum()
    return G, tot


def top_internal_pairs(cm, names, idx, onset, place):
    """Best i->j (i != j, both in place) and rate % of row i's utterances."""
    best = (0.0, None, None)
    for c in names:
        if ONSET_PLACE.get(onset[c]) != place:
            continue
        i = idx[c]
        row_tot = cm[i].sum()
        if row_tot <= 0:
            continue
        for cc in names:
            if cc == c or ONSET_PLACE.get(onset[cc]) != place:
                continue
            rate = cm[i, idx[cc]] / row_tot
            if rate > best[0]:
                best = (rate, c, cc)
    if best[1] is None:
        return None
    return best


def top_global(cm, names, idx, k=5):
    out = []
    for c in names:
        i = idx[c]
        row_tot = cm[i].sum()
        for cc in names:
            if cc == c:
                continue
            rate = cm[i, idx[cc]] / row_tot
            out.append((rate, f"{c}->{cc}"))
    out.sort(reverse=True)
    return out[:k]


def fmt_pair(p):
    rate, a, b = p
    return f"{a}↔{b} ({rate * 100:.0f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--models", default="ema_only,full,hubert")
    args = ap.parse_args()

    matrices = json.load(open(args.cache))
    onset = class_onset_map()

    result = {"places": PLACES, "n_places": len(PLACES),
              "class_to_onset": onset, "models": {}}
    for key, name in MODELS:
        if key not in matrices or key not in args.models.split(","):
            continue
        v = matrices[key]
        names = v["classes"]
        cm = np.asarray(v["cm"], dtype=float)
        idx = {c: i for i, c in enumerate(names)}
        G, tot = grouped_cm(cm, names, idx, onset)
        norm = G / np.maximum(tot, 1)[:, None] * 100.0
        acc = np.trace(cm) / cm.sum() * 100.0

        per_place = []
        for p in PLACES:
            i = PLACES.index(p)
            n_syl = sum(1 for c in names if ONSET_PLACE.get(onset[c]) == p)
            within = norm[i, i]
            # total errors of this place's syllables = row total minus the
            # utterances that were classified *correctly* (class-level diagonal)
            correct_here = 0.0
            for c in names:
                if ONSET_PLACE.get(onset[c]) == p:
                    correct_here += cm[idx[c], idx[c]]
            err_count = tot[i] - correct_here
            # ... of which how many land on another syllable *of the same place*?
            internal_err_count = 0.0
            for c in names:
                if ONSET_PLACE.get(onset[c]) != p:
                    continue
                ic = idx[c]
                for cc in names:
                    if cc == c or ONSET_PLACE.get(onset[cc]) != p:
                        continue
                    internal_err_count += cm[ic, idx[cc]]
            err_frac = (internal_err_count / err_count * 100.0) if err_count > 0 else 0.0
            dom = top_internal_pairs(cm, names, idx, onset, p)
            per_place.append({
                "place": p, "n_syllables": int(n_syl),
                "within_place_acc": float(within),
                "err_share_internal": float(err_frac),
                "dominant_internal_pair": fmt_pair(dom) if dom else None,
                "dominant_internal_rate": dom[0] if dom else None,
            })
        result["models"][key] = {
            "name": name, "overall_acc": float(acc),
            "grouped_matrix": G.tolist(),
            "grouped_norm_pct": norm.tolist(),
            "per_place": per_place,
            "top_global": [(r, pair) for r, pair in top_global(cm, names, idx)],
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # ---- Markdown table artifact ----
    header = ("| Place (n syllables) | EMA-only within-place acc % | "
              "EMA-only dominant internal pair | Full within-place acc % | "
              "Full dominant internal pair | HuBERT within-place acc % | "
              "HuBERT dominant internal pair |")
    lines = ["# Grouped confusion by place of articulation (rows = true place)",
             "", header, "|---|---|---|---|---|---|---|"]
    mk = result["models"]
    for p in PLACES:
        row = [p]
        for key in ("ema_only", "full", "hubert"):
            if key not in mk:
                continue
            d = next(x for x in mk[key]["per_place"] if x["place"] == p)
            row.append(f"{d['within_place_acc']:.1f}")
            row.append(d["dominant_internal_pair"] or "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Overall matrix accuracy: "
                 + ", ".join(f"{mk[k]['name']} {mk[k]['overall_acc']:.1f}%"
                             for k in ("ema_only", "full", "hubert") if k in mk))
    lines.append("")
    lines.append("## Top global confusions per model")
    for k in ("ema_only", "full", "hubert"):
        if k not in mk:
            continue
        lines.append("")
        lines.append(f"### {mk[k]['name']}")
        for r, pair in mk[k]["top_global"]:
            lines.append(f"- `{pair}` ({r * 100:.1f}%)")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- Figure ----
    os.makedirs(OUT_ILLUS, exist_ok=True)
    fig, axes = plt.subplots(1, len(mk), figsize=(6.5 * len(mk), 5.6),
                             constrained_layout=True)
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    im = None
    for ax, key in zip(axes, ("ema_only", "full", "hubert")):
        if key not in mk:
            continue
        norm = np.asarray(mk[key]["grouped_norm_pct"])
        acc = mk[key]["overall_acc"]
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(PLACES)))
        ax.set_xticklabels([p[:7] for p in PLACES], rotation=45, fontsize=8)
        ax.set_yticks(range(len(PLACES)))
        ax.set_yticklabels([p[:7] for p in PLACES], fontsize=8)
        for i in range(len(PLACES)):
            for j in range(len(PLACES)):
                val = norm[i, j]
                if val >= 3:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            color="white" if val > 55 else "#333333", fontsize=8)
        ax.set_title(f"{mk[key]['name']} — acc {acc:.1f}%", fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("row-normalized % (recall)", fontsize=10)
    fig.suptitle("Confusion grouped by onset place of articulation "
                 "(EMA-only / full / HuBERT)", fontsize=13)
    plt.savefig(OUT_PNG, dpi=170)
    plt.close()

    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    print("wrote", OUT_PNG)
    print()
    for key in ("ema_only", "full", "hubert"):
        if key not in mk:
            continue
        print(f"== {mk[key]['name']} (overall {mk[key]['overall_acc']:.2f}%)")
        for d in mk[key]["per_place"]:
            print(f"  {d['place']:11s} n={d['n_syllables']:2d} within={d['within_place_acc']:5.1f}%  "
                  f"internal-err-share={d['err_share_internal']:4.1f}%  "
                  f"top={d['dominant_internal_pair']}")
        print("  top global:", ", ".join(f"`{p}` ({r*100:.0f}%)"
                                         for r, p in mk[key]["top_global"][:3]))


if __name__ == "__main__":
    main()
