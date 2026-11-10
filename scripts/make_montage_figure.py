"""Figure: which coils are needed (credit vs cost, then the montage ladder).

Panel (a) puts the frozen-model Shapley credit of each coil beside the accuracy
cost of removing that coil and retraining (positive = removal hurts). The two
horizontal scales differ on purpose: the removal costs of LL, TL and LC are
within half a point of one another, so a shared axis would merge them.

Panel (b) drops coils in groups, six down to one, and annotates the paired
fold-level change at each step; whiskers span the six fold means.

Data (read, not hard-coded): ``runs/paper/exp8_robust_z/sensor_shapley.json`` and
``runs/montage_subset/*.json``. Statistics are fold-level (seeds averaged within
fold, n = 6), t-based 95% CI, matching ``src/analysis_stats.py``.

The script self-checks for overlapping text and coincident markers and prints any
collision it finds.

Usage::

    python scripts/make_montage_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "runs", "montage_subset")
SHAP = os.path.join(ROOT, "runs", "paper", "exp8_robust_z", "sensor_shapley.json")
OUT = os.path.join(ROOT, "paper", "illustrations", "fig_montage_necessity")

TONGUE = "#2F6DB5"
LIP = "#C36B4A"

COILS = ["TD", "TT", "LL", "LC", "TL", "UL"]
LADDER = [("all", "6 coils\n59 ch"), ("midsag4", "4 coils\n40 ch"),
          ("midsag3", "3 coils\n28 ch"), ("min2", "2 coils\n19 ch"),
          ("td1", "1 coil\n9 ch")]


def load(name):
    with open(os.path.join(MS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def fold_means(runs):
    d = {}
    for r in runs:
        d.setdefault(r["fold"], []).append(r["test_acc"])
    return {f: float(np.mean(v)) for f, v in d.items()}


def ci95(x):
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="store_true",
                    help="clean canvas for manual annotation: coil counts only on "
                         "the x axis, no step labels and no step numbers")
    args = ap.parse_args()
    clean = args.base                     # clean canvas for manual annotation
    out = OUT + ("_base" if clean else "")

    shap = json.load(open(SHAP, encoding="utf-8"))["aggregate"]
    base = fold_means(load("all"))

    credit = {c: shap[c]["mean"] * 100 for c in COILS}
    credit_sd = {c: shap[c]["std"] * 100 for c in COILS}
    cost, cost_lo, cost_hi = {}, {}, {}
    for c in COILS:
        fd = fold_means(load("drop_" + c))
        keys = sorted(set(base) & set(fd))
        delta = np.array([base[k] - fd[k] for k in keys]) * 100
        cost[c] = float(delta.mean())
        half = ci95(delta)
        cost_lo[c] = cost[c] - half
        cost_hi[c] = cost[c] + half

    rungs = []
    for name, label in LADDER:
        fm = fold_means(load(name))
        vals = np.array([fm[k] for k in sorted(fm)]) * 100
        rungs.append({"name": name, "label": label, "folds": vals,
                      "mean": float(vals.mean())})
    removed = [("TL", "LC"), ("LL",), ("UL",), ("TT",)]

    plt.rcParams.update({
        "font.size": 8.4, "axes.labelsize": 8.6, "axes.titlesize": 8.6,
        "xtick.labelsize": 7.6, "ytick.labelsize": 8.0,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "legend.fontsize": 7.6,
        # TrueType (Type 42) instead of the default Type 3 glyph procedures:
        # Type 3 renders thin and lossy when the PDF is embedded in LaTeX.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(3.5, 4.2 if clean else 4.5), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.22, 1.0],
                          hspace=0.46, wspace=0.20, left=0.19, right=0.97,
                          top=0.87, bottom=0.10 if clean else 0.125)

    # ---------------- panel (a): credit vs cost ----------------
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    y = np.arange(len(COILS))[::-1]          # TD at the top

    for i, c in enumerate(COILS):
        col = TONGUE if c in ("TD", "TT", "TL") else LIP
        axL.errorbar(credit[c], y[i], xerr=credit_sd[c], fmt="o", ms=4.0,
                     mfc="white", mec=col, mew=1.25, ecolor=col, elinewidth=0.7,
                     capsize=0, zorder=3)
        axR.errorbar(cost[c], y[i], xerr=[[cost[c] - cost_lo[c]],
                                          [cost_hi[c] - cost[c]]],
                     fmt="s", ms=4.2, mfc=col, mec=col, ecolor=col,
                     elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=3)
        axL.text(credit[c], y[i] + 0.30, f"{credit[c]:.1f}", ha="center",
                 va="bottom", fontsize=6.3, color="0.3", zorder=4)
        c_txt = "0.0" if abs(cost[c]) < 0.05 else f"{cost[c]:+.1f}"
        axR.text(cost[c], y[i] + 0.30, c_txt, ha="center", va="bottom",
                 fontsize=6.3, color="0.3", zorder=4)

    axL.set_xlim(0, 25)
    axL.set_xticks([0, 5, 10, 15, 20])
    axL.set_ylim(-0.70, len(COILS) - 0.30)
    axL.set_yticks(y)
    axL.set_yticklabels(COILS)
    axL.set_xlabel("Frozen credit (pts)", labelpad=2, fontsize=8.0)
    axL.tick_params(axis="y", length=0, pad=2)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.grid(axis="x", color="0.88", lw=0.5, zorder=0)

    axR.set_xlim(-2.0, 6.9)
    axR.set_xticks([0, 2, 4, 6])
    axR.axvline(0, color="0.35", lw=0.8, zorder=1)
    axR.set_xlabel("Retrained cost (pts)", labelpad=2, fontsize=8.0)
    axR.tick_params(axis="y", left=False, labelleft=False)
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.grid(axis="x", color="0.88", lw=0.5, zorder=0)

    axL.text(-0.02, 1.08, "(a)", transform=axL.transAxes, fontweight="bold",
             fontsize=8.5, va="bottom")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="white", mec="0.25", ms=4.2,
               mew=1.1, label="frozen Shapley credit"),
        Line2D([], [], marker="s", ls="none", mfc="0.35", mec="0.35", ms=4.0,
               label="retrained removal cost (95% CI)")],
        loc="upper center", bbox_to_anchor=(0.60, 0.955), ncol=1, frameon=False,
        handletextpad=0.5, labelspacing=0.25)

    # ---------------- panel (b): the ladder ----------------
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(rungs))
    for i, r in enumerate(rungs):
        ax.vlines(x[i], float(r["folds"].min()), float(r["folds"].max()),
                  color="0.78", lw=1.2, zorder=1)
    ax.plot(x, [r["mean"] for r in rungs], "-", color="0.25", lw=1.0, zorder=3)
    ax.scatter(x, [r["mean"] for r in rungs], s=34, color=TONGUE,
               edgecolor="white", linewidth=0.6, zorder=4)

    if not clean:
        # fold-mean accuracy above each point, white pad so the whisker does not
        # run through the glyphs
        for i, r in enumerate(rungs):
            ax.annotate(f"{r['mean']:.1f}", (x[i], r["mean"]),
                        xytext=(7, 4), textcoords="offset points",
                        ha="left", va="center",
                        fontsize=6.3, color="0.3", zorder=5)

    ax.axhline(rungs[0]["mean"], color="0.45", ls="--", lw=0.7, zorder=1)
    ax.set_xticks(x)
    tags = ["all six"] + ["$-$" + " $-$".join(rem) for rem in removed]
    if clean:
        ax.set_xticklabels(["6 coils", "4 coils", "3 coils", "2 coils", "1 coil"])
    else:
        ax.set_xticklabels([f"{r['label']}\n{tag}" for r, tag in zip(rungs, tags)],
                           linespacing=1.35)
    ax.set_xlim(-0.5, len(rungs) - 0.38)
    ax.set_ylim(47, 90)
    ax.set_ylabel("Accuracy (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax.text(-0.02, 1.05, "(b)", transform=ax.transAxes, fontweight="bold",
            fontsize=8.5, va="bottom")

    fig.savefig(out + ".png", facecolor="white")
    fig.savefig(out + ".pdf", facecolor="white")

    # ---------------- collision self-check ----------------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [(t.get_text(), t.get_window_extent(rend))
             for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    bad = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 1.0 and h > 1.0:
                bad.append((texts[i][0].replace("\n", "|"),
                            texts[j][0].replace("\n", "|"), round(w * h, 1)))

    pts = []
    for axis in (axL, axR):
        for line in axis.lines:
            if line.get_marker() in ("o", "s"):
                xy = axis.transData.transform(line.get_xydata())[0]
                pts.append(np.asarray(xy, dtype=float))
    mind = min(float(np.hypot(*(pts[i] - pts[j])))
               for i in range(len(pts)) for j in range(i + 1, len(pts)))
    mdia = 4.2 / 72 * fig.dpi

    mpts = [ax.transData.transform(p) for p in
            np.column_stack([x, [r["mean"] for r in rungs]])]
    mmin = min(float(np.hypot(*(mpts[i] - mpts[j])))
               for i in range(len(mpts)) for j in range(i + 1, len(mpts)))

    print(f"wrote {out}.png and {out}.pdf")
    print(f"text-overlap pairs: {len(bad)}")
    for b in bad:
        print("   overlap:", b)
    print(f"panel (a): {mind:.1f} px between marker centres, diameter {mdia:.1f} px, "
          f"edge gap {mind - mdia:.1f} px")
    print(f"panel (b): {mmin:.1f} px between mean markers")


if __name__ == "__main__":
    main()
