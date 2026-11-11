"""Training-set-size curve for the TASLP manuscript (single panel).

Reads the EMA-only BiLSTM speaker sweep produced by ``scripts/run_scaling.py``
(``runs/scaling/spk_k*.json`` plus optional ``_s1`` extra-seed files) and writes
a single-column figure in the same style as the other data panels.

Usage::

    python scripts/make_scaling_figure.py
    python scripts/make_scaling_figure.py --out <Figures dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import figcheck  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs", "scaling")
OUT_DEFAULT = os.path.join(REPO, "paper", "illustrations")
ACOUSTIC = 88.28            # full 106-D model, 2 seeds (docs/STATUS.md)

EMA_C = "#3B6FB6"
ACO_C = "#D07A3A"
GREY = "#7A7A7A"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})


def load(name):
    with open(os.path.join(RUNS, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def stats_cfg(name):
    """Config stats, including an optional ``_s1`` extra-seed file."""
    runs = load(name)
    extra = os.path.join(RUNS, f"{name}_s1.json")
    if os.path.exists(extra):
        with open(extra, encoding="utf-8") as f:
            runs = runs + json.load(f)
    acc = np.array([r["test_acc"] for r in runs]) * 100
    return acc.mean(), acc.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ks = sorted(int(n[5:-5]) for n in os.listdir(RUNS)
                if n.startswith("spk_k") and n.endswith(".json")
                and n[5:-5].isdigit())
    m, s = zip(*[stats_cfg(f"spk_k{k:02d}") for k in ks])
    x = np.arange(len(ks))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axhline(ACOUSTIC, color=ACO_C, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.text(-0.45, ACOUSTIC + 0.6, "with acoustics (88.3)",
            color=ACO_C, fontsize=7, ha="left", va="bottom")

    ax.errorbar(x, m, yerr=s, fmt="-o", color=EMA_C, ms=4, lw=1.2,
                capsize=2.5, elinewidth=1.0, zorder=3)
    for xi, mi, si in zip(x, m, s):
        ax.annotate(f"{mi:.1f}", (xi, mi), textcoords="offset points",
                    xytext=(0, 7 + si * 4.4), ha="center", va="bottom",
                    fontsize=8, color=EMA_C)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.55, len(ks) - 0.45)
    ax.set_xlabel("Training speakers")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 92)
    ax.set_yticks([60, 70, 80, 90])

    fig.tight_layout(pad=0.3)
    p = os.path.join(args.out, "fig_scaling.png")
    figcheck.save(fig, p, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
