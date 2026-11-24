"""Training-set size for the EMA-only model (Fig. 2 of the manuscript).

Reads the speaker sweep produced by ``scripts/run_scaling.py``
(``runs/scaling/spk_k*.json`` plus optional ``_s1`` extra-seed files).

The panel shows the runs, not just their means: the twelve fold x seed results at
each speaker count as jittered dots, the mean as a diamond with its standard
deviation, and the gain from each step as a small secondary annotation. Four
points alone leave a column of white space and hide the overlap between the 18-
and 23-speaker conditions, which is what makes the last gain (+2.4) unremarkable.

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

from src import figstyle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.environ.get("EMA_RUNS_DIR") or os.path.join(REPO, "runs"),
                    "scaling")
OUT_DEFAULT = os.path.join(REPO, "paper", "illustrations")

#: Full 106-D acoustic model, 6 folds x 2 seeds (paper/results/exp1_ceilings.json).
ACOUSTIC = 88.28

figstyle.apply()


def load(name):
    with open(os.path.join(RUNS, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def runs_of(k):
    """All fold x seed accuracies (%) for one speaker count."""
    runs = load(f"spk_k{k:02d}")
    extra = os.path.join(RUNS, f"spk_k{k:02d}_s1.json")
    if os.path.exists(extra):
        with open(extra, encoding="utf-8") as f:
            runs = runs + json.load(f)
    return np.array([r["test_acc"] for r in runs]) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ks = sorted(int(n[5:-5]) for n in os.listdir(RUNS)
                if n.startswith("spk_k") and n.endswith(".json")
                and n[5:-5].isdigit())
    runs = {k: runs_of(k) for k in ks}
    m = np.array([runs[k].mean() for k in ks])
    s = np.array([runs[k].std() for k in ks])
    x = np.arange(len(ks))

    fig, ax = plt.subplots(figsize=(3.5, 2.75))

    # The ceiling a claim is measured against.
    ax.axhline(ACOUSTIC, color=figstyle.AMBER, ls=(0, (3, 2.4)), lw=0.8, zorder=1)
    ax.text(-0.44, ACOUSTIC + 0.75, f"Acoustic Model ({ACOUSTIC:.1f})",
            color=figstyle.AMBER, fontsize=6.8, ha="left", va="bottom")

    # The data: every run, then the summary on top of it.
    rng = np.random.RandomState(7)
    for i, k in enumerate(ks):
        v = runs[k]
        ax.scatter(np.full(v.size, i) + rng.uniform(-0.16, 0.16, size=v.size), v,
                   s=7.5, color=figstyle.NAVY, alpha=0.62, linewidth=0, zorder=2)
    ax.plot(x, m, ls=(0, (4, 3)), color=figstyle.LIGHT, lw=0.8, zorder=1.6)
    ax.errorbar(x, m, yerr=s, fmt="D", color=figstyle.INK, ms=3.4, capsize=1.8,
                elinewidth=0.8, capthick=0.8, zorder=5)

    for xi, mi, si in zip(x, m, s):
        ax.annotate(f"{mi:.1f}", (xi, mi + si), textcoords="offset points",
                    xytext=(0, 4.2), ha="center", va="bottom", fontsize=7.0,
                    color=figstyle.INK, fontweight="bold", zorder=6,
                    bbox=figstyle.chip())
    for i in range(len(x) - 1):
        g = m[i + 1] - m[i]
        ax.annotate(f"+{g:.1f}", ((x[i] + x[i + 1]) / 2, (m[i] + m[i + 1]) / 2),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    va="bottom", fontsize=7.0, color="#40474F", style="italic",
                    zorder=6, bbox=dict(facecolor="white", edgecolor="none",
                                        pad=0.8, alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.52, len(ks) - 0.42)
    ax.set_ylim(55, 96)
    ax.set_yticks([60, 70, 80, 90])
    ax.set_xlabel("Training Speakers")
    ax.set_ylabel("Accuracy (%)")

    fig.tight_layout(pad=0.3)
    p = os.path.join(args.out, "fig_scaling.png")
    figstyle.save(fig, p, bbox_inches="tight")
    print("wrote", p)
    print("  " + "  ".join(f"{k} spk {mi:.2f}+-{si:.2f} (n={runs[k].size})"
                           for k, mi, si in zip(ks, m, s)))
    print("  gains " + "  ".join(f"+{g:.2f}" for g in np.diff(m)))
    plt.close(fig)


if __name__ == "__main__":
    main()
