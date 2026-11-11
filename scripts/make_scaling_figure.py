"""Training-set-size curve for the TASLP manuscript (Fig. 2).

Reads the EMA-only BiLSTM speaker sweep produced by ``scripts/run_scaling.py``
(``runs/scaling/spk_k*.json`` plus optional ``_s1`` extra-seed files).

Drawn in the same language as the other data figures: two colours, no grid, no
in-plot title, capped error bars, open frame. What fills the panel is content
rather than decoration: the acoustic model is a band carrying its own spread,
the deficit between the two is shaded, and the remaining gap is measured once at
the right-hand end.

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

# Full 106-D acoustic model on the same folds: 88.28 +- 3.36 over 6 folds x 2 seeds
# (paper/results/exp1_ceilings.json).
ACOUSTIC, ACOUSTIC_SD = 88.28, 3.36

EMA_C = "#3B6FB6"
ACO_C = "#D07A3A"
INK = "#333333"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 0.8,
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

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Acoustic model as a band, so the reference carries the spread of its own runs.
    ax.axhspan(ACOUSTIC - ACOUSTIC_SD, ACOUSTIC + ACOUSTIC_SD,
               color=ACO_C, alpha=0.13, lw=0, zorder=0)
    ax.axhline(ACOUSTIC, color=ACO_C, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.fill_between(x, m, ACOUSTIC, color=ACO_C, alpha=0.07, lw=0, zorder=0)

    ax.errorbar(x, m, yerr=s, fmt="-o", color=EMA_C, ms=4.6, lw=1.4,
                capsize=2.8, elinewidth=1.0, capthick=1.0, zorder=3)
    for xi, mi, si in zip(x, m, s):
        ax.annotate(f"{mi:.1f}", (xi, mi + si), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom",
                    fontsize=7.6, color=EMA_C)

    # The remaining deficit, measured once at the last condition. The caption states
    # the same number, so this is a pointer rather than a second claim.
    gap = ACOUSTIC - m[-1]
    xa = x[-1] + 0.30
    ax.annotate("", xy=(xa, ACOUSTIC), xytext=(xa, m[-1]),
                arrowprops=dict(arrowstyle="|-|", color=INK, lw=0.9,
                                shrinkA=1.5, shrinkB=1.5))
    ax.text(xa + 0.09, (ACOUSTIC + m[-1]) / 2, f"{gap:.1f}",
            ha="left", va="center", fontsize=7.4, color=INK)

    ax.text(-0.44, ACOUSTIC + ACOUSTIC_SD + 0.5,
            f"Acoustic Model ({ACOUSTIC:.1f} $\\pm$ {ACOUSTIC_SD:.1f})",
            color=ACO_C, fontsize=7.2, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.62, len(ks) + 0.12)
    ax.set_xlabel("Training Speakers")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(56, 93.6)
    ax.set_yticks([60, 70, 80, 90])

    fig.tight_layout(pad=0.35)
    p = os.path.join(args.out, "fig_scaling.png")
    figcheck.save(fig, p, bbox_inches="tight")
    print("wrote", p)
    print("  " + "  ".join(f"{k} spk {mi:.2f}+-{si:.2f}" for k, mi, si in zip(ks, m, s)))
    print(f"  remaining gap at {ks[-1]} speakers: {gap:.2f} points")
    plt.close(fig)


if __name__ == "__main__":
    main()
