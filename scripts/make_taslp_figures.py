"""Generate the 6-coil data figures for the TASLP manuscript.

Reads the locked artifacts under ``runs/`` and SAIT-EMA runs, and writes the
figures into the manuscript's ``Figures/`` directory. These are the data panels;
the anatomical schematics are drawn separately in Illustrator.

Usage::

    python scripts/make_taslp_figures.py

Figures produced
----------------
- ``fig_gap_norm_calib.png`` : ceilings / normalization variants / calibration
- ``fig_shapley.png``        : exact group Shapley (accuracy points)
- ``fig_crosscorpus.png``    : Bruce vs SAIT group Shapley + replication gains
"""
import json
import os
import statistics as st

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import figcheck
from src.paths import get as path, REPO_ROOT  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = REPO_ROOT
RUNS = os.path.join(REPO, "runs")
SAIT = str(path("sait_dataset") / "runs")
OUT = str(path("figures"))

EMA_C, ACO_C, SSL_C = "#3B6FB6", "#D07A3A", "#4C9A5A"
GREY = "#7A7A7A"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean_pct(runs, key="test_acc"):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return 100 * st.mean(vals), 100 * st.pstdev(vals)


def fig_calibration():
    """Single-panel deployment calibration: centering vs robust z."""
    cd = load(os.path.join(RUNS, "fair_deploy_center_only", "fair_deploy.json"))["curve"]
    rd = load(os.path.join(RUNS, "fair_deploy_robust_z", "fair_deploy.json"))["curve"]
    ks = ["1", "2", "3", "5", "10", "20", "all"]
    kx = np.arange(len(ks))
    co = [100 * cd[k][0] for k in ks]; ce = [100 * cd[k][1] for k in ks]
    ro = [100 * rd[k][0] for k in ks]; re = [100 * rd[k][1] for k in ks]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(kx, co, yerr=ce, fmt="-o", color=EMA_C, ms=3.5, lw=1.2,
                capsize=2, label="Centering")
    ax.errorbar(kx, ro, yerr=re, fmt="-s", color=ACO_C, ms=3.5, lw=1.2,
                capsize=2, label="Robust $z$")
    ax.set_xticks(kx); ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "All"])
    ax.set_xlabel("Calibration Utterances ($k$)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(20, 84)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_calibration.png")
    figcheck.save(fig, p, bbox_inches="tight"); print("wrote", p)
    plt.close(fig)


def fig_shapley():
    d = load(os.path.join(RUNS, "paper", "exp2", "shapley.json"))
    agg = d["aggregate"]["shapley_acc"]
    names = {"G1": "Position", "G2": "Velocity", "G3": "Acceleration",
             "G4": "Geometry", "G5": "Voicing", "G6": "Formant",
             "G7": "Spectral", "G8": "Energy"}
    kind = d["kind"]
    order = sorted(d["groups"], key=lambda g: agg[g]["mean"])
    means = [agg[g]["mean"] * 100 for g in order]
    stds = [agg[g]["std"] * 100 for g in order]
    cols = [EMA_C if kind[g] == "ema" else ACO_C for g in order]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    y = np.arange(len(order))
    ax.barh(y, means, xerr=stds, color=cols, height=0.68, capsize=2,
            error_kw={"lw": 0.8})
    for yi, m, e in zip(y, means, stds):
        ax.text(m + e + 1.2, yi, f"{m:+.1f}", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} {names[g]}" for g in order], fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Shapley Value (Accuracy Points)")
    ax.set_xlim(-4, 58)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=EMA_C, label="EMA"),
                       Line2D([], [], marker="s", ls="", color=ACO_C, label="Acoustic")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_shapley.png"); figcheck.save(fig, p, bbox_inches="tight"); print("wrote", p)
    plt.close(fig)


def fig_crosscorpus():
    bg = load(os.path.join(RUNS, "paper", "exp2", "shapley.json"))["aggregate"]["shapley_acc"]
    sg = load(os.path.join(SAIT, "shapley_full.json"))["shapley"]
    gs = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    b = [bg[g]["mean"] * 100 for g in gs]
    s = [sg[g]["mean"] * 100 for g in gs]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    ax = axes[0]
    mx = max(b + s) * 1.15
    ax.plot([0, mx], [0, mx], ls=":", color="0.6", lw=1)
    ax.scatter(b, s, s=34, color=EMA_C, zorder=3)
    # Alternate the label above/below the marker: G5 and G8 land almost on top of each
    # other, so a single fixed offset prints them into the same pixels.
    for i, (g, bi, si) in enumerate(zip(gs, b, s)):
        ax.annotate(g, (bi, si), textcoords="offset points",
                    xytext=(4, 3) if i % 2 == 0 else (4, -9), fontsize=7.5)
    ax.set_xlabel("Cantonese Shapley (Points)"); ax.set_ylabel("Mandarin Shapley (Points)")
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_title("(a) Group attribution ($\\rho=0.74$)", fontsize=9.5)

    ax = axes[1]
    lab = ["offset removal", "medial-lateral\ndropped"]
    bruce = [9.5, 1.5]; mand = [20.5, 2.9]
    xx = np.arange(len(lab)); w = 0.36
    ax.bar(xx - w / 2, bruce, w, color=EMA_C, label="Cantonese")
    ax.bar(xx + w / 2, mand, w, color=ACO_C, label="Mandarin")
    for xi, v in zip(xx - w / 2, bruce):
        ax.text(xi, v + 0.4, f"+{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(xx + w / 2, mand):
        ax.text(xi, v + 0.4, f"+{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("Accuracy Gain (Points)"); ax.set_ylim(0, 24)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Replicated levers", fontsize=9.5)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_crosscorpus.png"); figcheck.save(fig, p, bbox_inches="tight"); print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_calibration()
    fig_shapley()
    fig_crosscorpus()
