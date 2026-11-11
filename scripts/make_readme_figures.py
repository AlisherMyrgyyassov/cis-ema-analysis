"""README figures: what the paper measured, and whether the code reproduces it.

Reads only files shipped in this repository (``expected/claims.json`` and
``expected/verification/table1_reproduced.json``), so the pictures can be
regenerated without the corpus:

    python scripts/make_readme_figures.py --out docs/figures

Panel (a) is the accuracy ladder: the full model is drawn as the EMA-only result
plus the acoustic contribution, so the 9.6-point gap is the orange segment.
Panel (b) compares the published numbers with a re-run in the released
environment; its x-axis is deliberately zoomed and says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Palette: articulatory, acoustic, audio-only, neutral.
EMA = "#2b6cb0"
ACO = "#dd6b20"
AUD = "#4a5568"
GREY = "#a0aec0"
INK = "#2d3748"


def _claim(claims, cid):
    for c in claims:
        if c["id"] == cid:
            v = c["paper"]
            return float(v[0]), (float(v[1]) if len(v) > 1 else None)
    raise KeyError(cid)


def panel_ladder(ax, claims):
    ema, ema_sd = _claim(claims, "t1_ema")
    full, full_sd = _claim(claims, "t1_full")
    hub, hub_sd = _claim(claims, "t1_hubert")
    chance, _ = _claim(claims, "t1_chance")
    maj, _ = _claim(claims, "t1_majority")

    # HuBERT: the audio ceiling
    ax.barh(2.0, hub, height=0.46, color=AUD, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.errorbar(hub, 2.0, xerr=hub_sd, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5)
    ax.text(hub - 1.8, 2.0, f"{hub:.1f}", ha="right", va="center", color="white",
            fontsize=9.0, fontweight="bold")

    # full model: EMA-only plus the acoustic contribution, drawn as one bar
    ax.barh(1.0, ema, height=0.46, color=EMA, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.barh(1.0, full - ema, left=ema, height=0.46, color=ACO, alpha=0.95,
            edgecolor="white", linewidth=0.8)
    ax.errorbar(full, 1.0, xerr=full_sd, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5)
    ax.text(ema - 1.8, 1.0, f"{ema:.1f}", ha="right", va="center", color="white",
            fontsize=9.0, fontweight="bold")
    ax.text(full + 1.6, 1.0, f"{full:.1f}", ha="left", va="center", color=INK,
            fontsize=9.0, fontweight="bold")
    ax.annotate(f"+{full - ema:.1f} from acoustics", xy=(ema + (full - ema) / 2, 1.26),
                xytext=(ema + (full - ema) / 2, 1.48), ha="center", va="bottom",
                fontsize=7.6, color=ACO, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACO, linewidth=0.8))

    # EMA-only
    ax.barh(0.0, ema, height=0.46, color=EMA, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.errorbar(ema, 0.0, xerr=ema_sd, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5)
    ax.text(ema - 1.8, 0.0, f"{ema:.1f}", ha="right", va="center", color="white",
            fontsize=9.0, fontweight="bold")

    # distance to the audio ceiling
    ax.annotate("", xy=(hub, 2.30), xytext=(full, 2.30),
                arrowprops=dict(arrowstyle="|-|", color=AUD, linewidth=1.0,
                                shrinkA=2, shrinkB=2))
    ax.text((full + hub) / 2, 2.36, f"{hub - full:.1f}", ha="center", va="bottom",
            fontsize=7.8, color=AUD, fontweight="bold", clip_on=True)

    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["EMA-only\n59 channels", "full\n+ 47 acoustics",
                        "HuBERT-large\naudio"], fontsize=8.4, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.62, 2.78)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("accuracy, 43-way syllables (%)", fontsize=8.4, color=INK)

    ax.axvline(chance, color=GREY, linestyle=":", linewidth=1.0)
    ax.text(chance + 1.0, -0.46, f"chance {chance:.1f}  \u00b7  majority {maj:.1f}",
            fontsize=7.2, color="#718096", va="center")
    ax.set_title("(a) what the paper measures", fontsize=9.4, color=INK, loc="left", pad=5)
    _clean(ax)



def panel_repro(ax, claims, repro):
    rows = [
        ("logistic regr., 424 stats", "t1_logreg", "logreg"),
        ("random forest, 424 stats", "t1_rf", "rf"),
        ("BiLSTM, EMA-only", "t1_ema", "ema_only"),
        ("BiLSTM, full", "t1_full", "full"),
    ]
    ys = [3, 2, 1, 0]
    for yy, (label, cid, key) in zip(ys, rows):
        published, _ = _claim(claims, cid)
        got = repro[key]["mean"]
        ax.plot([published, got], [yy, yy], color=GREY, linewidth=1.2, zorder=1)
        ax.scatter([published], [yy], s=34, color=INK, zorder=3,
                   label="published" if yy == ys[0] else None)
        ax.scatter([got], [yy], s=52, facecolor="none", edgecolor=ACO, linewidth=1.6,
                   zorder=4, label="reproduced" if yy == ys[0] else None)
        d = got - published
        ax.text(90.0, yy, f"{d:+.2f}", va="center", ha="right", fontsize=7.6,
                color=INK if abs(d) > 0.05 else GREY)

    ax.text(90.0, 3.45, "\u0394", va="center", ha="right", fontsize=7.6, color=INK,
            fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.4, color=INK)
    ax.set_xlim(77.0, 90.4)
    ax.set_ylim(-0.78, 3.68)
    ax.set_xticks([78, 80, 82, 84, 86, 88])
    ax.set_xlabel("accuracy (%) \u00b7 axis zoomed \u00b7 \u0394 = re-run \u2212 published",
                  fontsize=8.0, color=INK)
    ax.legend(loc="lower center", handletextpad=0.5, borderaxespad=0.1, ncol=2,
              fontsize=7.4, frameon=False, columnspacing=1.4)
    ax.set_title("(b) re-run here", fontsize=9.4, color=INK, loc="left", pad=5)
    _clean(ax)


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8.0, length=3, width=0.8, pad=2.5)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    claims = json.load(open(os.path.join(REPO, "expected", "claims.json")))["claims"]
    repro = json.load(open(os.path.join(REPO, "expected", "verification",
                                        "table1_reproduced.json")))

    os.makedirs(args.out, exist_ok=True)
    fig = plt.figure(figsize=(7.6, 3.15))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.34], wspace=0.46,
                  left=0.125, right=0.975, top=0.87, bottom=0.175)
    panel_ladder(fig.add_subplot(gs[0, 0]), claims)
    panel_repro(fig.add_subplot(gs[0, 1]), claims, repro)

    path = os.path.join(args.out, "fig_overview.png")
    fig.savefig(path, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
