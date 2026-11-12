"""Paper figures for the TASLP EMA-observability manuscript.

Generates clean, presentation-ready figures for a two-column IEEE layout.
Conventions: no descriptive titles (they live in the LaTeX captions), capitalised
axis labels and legends, single hyphens only, and compact readable annotations.

Run with:

    python -m src.paper_figures --out <dir>

Only reads saved artifacts / the aligned dataset; no model training.

Figures this module owns in the submitted manuscript:

- ``fig_montage_daai_sequence.png`` : the Fig. 1 panels, one /daai/ token
- ``fig_trajectories.png``          : mean height / lip-aperture trajectories (README)
- ``fig_mechanistic.png``           : Fig. 5, mechanistic taxonomy of recoverability

The other submitted figures are built elsewhere: Fig. 2 by
``scripts/make_scaling_figure.py``, Figs. 3-4 by ``scripts/make_taslp_figures.py``,
Fig. 7 by ``scripts/make_montage_figure.py``. Figs. 1 and 6 are assembled by hand
from the panels those scripts emit.
"""
import argparse
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from src.config import RESULTS_DIR, DATA_DIR, REPO_ROOT
from src.figcheck import save as _save_fig

from src import figstyle  # noqa: E402

figstyle.apply()

TONGUE_C = "#2F6DB5"
LIP_C = "#C36B4A"
REF_C = "#6E6E6E"


def _load(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Fig: corpus-averaged normalized height trajectories
# --------------------------------------------------------------------------- #
def montage_and_trajectories(out_path):
    """Mean +/- SD tongue height (TDz) and lip aperture (UL-LL distance) for
    /baa/, /daai/, /gu/, in the per-utterance size-normalized frame,
    interpolated to normalized time per utterance."""
    man = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))
    pos = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                      usecols=["source", "time", "TDz",
                               "ULx", "ULy", "ULz", "LLx", "LLy", "LLz"])
    pos["Lip aperture"] = np.sqrt((pos["ULx"] - pos["LLx"]) ** 2 +
                                  (pos["ULy"] - pos["LLy"]) ** 2 +
                                  (pos["ULz"] - pos["LLz"]) ** 2)
    pos = pos.merge(man[["source", "syllable"]], on="source", how="inner")
    n_grid = 50
    xi = np.linspace(0.0, 1.0, n_grid)
    picks = [("baa", "Bilabial"), ("daai", "Alveolar"), ("gu", "Velar")]
    colors = ["#2E5C8A", "#C77B30", "#4F8A5B"]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True)
    for ax, (col, ylab) in zip(axes, (("TDz", "Tongue Height"),
                                      ("Lip aperture", "Lip Aperture"))):
        for (syl, lab), c in zip(picks, colors):
            curves = []
            for _, g in pos[pos["syllable"] == syl].groupby("source"):
                g = g.sort_values("time")
                t = g["time"].to_numpy(dtype=float)
                if len(t) < 2 or t[-1] <= t[0]:
                    continue
                tn = (t - t[0]) / (t[-1] - t[0])
                curves.append(np.interp(xi, tn, g[col].to_numpy(dtype=float)))
            if not curves:
                continue
            curves = np.array(curves)
            m, sd = curves.mean(0), curves.std(0)
            ax.plot(xi, m, color=c, lw=1.6, label=f"/{syl}/ ({lab})")
            ax.fill_between(xi, m - sd, m + sd, color=c, alpha=0.18, lw=0)
        ax.set_ylabel(ylab)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].set_xlabel("Normalized Time")
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   fontsize=7.5)
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _style_3d_axes(ax):
    ax.set_proj_type("ortho")
    ax.computed_zorder = False
    ax.grid(True, linestyle=":", color="0.82", linewidth=0.55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor("0.78")
        axis.pane.set_linewidth(0.6)
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.22
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.9)
        axis._axinfo["grid"]["linewidth"] = 0.55
        axis._axinfo["grid"]["linestyle"] = ":"
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.tick_params(pad=2, labelsize=8)
    ax.set_xlabel("Front–Back", labelpad=10)
    ax.set_ylabel("Left–Right", labelpad=10)
    ax.set_zlabel("")


def _sensor_colors():
    """Tongue coils blue, lip coils orange (shared with the Fig. 1 scripts)."""
    color = {n: TONGUE_C for n in ("TD", "TL", "TT")}
    color.update({n: LIP_C for n in ("UL", "LL", "LC")})
    return color


# --------------------------------------------------------------------------- #
# Fig: single-token /daai/ 3D sequence + height traces
# --------------------------------------------------------------------------- #
def _load_script(filename):
    path = os.path.join(REPO_ROOT, "scripts", filename)
    name = os.path.splitext(filename)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def daai_sequence(out_path):
    """One /daai/ token: 3D coil snapshots (constriction, vowel target,
    offglide) over tongue- and lip-height traces, no in-image title."""
    baa = _load_script("plot_montage_3d_baa.py")
    seq, vow = baa.seq, baa.vow
    man = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))
    feat = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                       usecols=["source", "time"] + seq.POS_COLS)
    speakers = seq._collect_speakers(man, feat, vow._subj_folders())
    subj, picked = seq._choose(speakers, np.random.RandomState(seq.DEFAULT_SEED))
    fig, _ = baa._build_figure(subj, picked["daai"], "daai", title=False)
    _save_fig(fig, out_path, facecolor="white")
    plt.close(fig)
    return out_path


def mechanistic(out_path):
    d = _load(os.path.join(RESULTS_DIR, "exp4_correlation.json"))
    table = d["table"]
    xs = [r["single_ridge_r2_in"] for r in table]
    ys = [r["joint_ridge_r2_in"] - r["single_ridge_r2_in"] for r in table]
    names = [r["acoustic_feature"].replace("MFCC_", "MFCC") for r in table]
    mech = {
        "F0": ("Weakly Linearly Observable", "#C0504D"),
        "F3": ("Weakly Linearly Observable", "#C0504D"),
        "MFCC2": ("Jointly Recoverable", "#30638E"),
        "MFCC5": ("Jointly Recoverable", "#30638E"),
        "MFCC6": ("Jointly Recoverable", "#30638E"),
        "F2": ("Captured (1-D)", "#2E8B57"),
    }
    inter = ("Intermediate", "#9AA0A6")
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    seen = {}
    label_pos = {
        "MFCC3": (-9, -2, "right", "center"),
        "MFCC4": (9, -2, "left", "center"),
        "MFCC6": (0, 10, "center", "bottom"),
    }
    for x, y, name in zip(xs, ys, names):
        label, color = mech.get(name, inter)
        seen[label] = color
        ax.scatter(x, y, s=70, color=color, zorder=3, edgecolor="white",
                   linewidth=0.8)
        dx, dy, ha, va = label_pos.get(name, (7, 5, "left", "baseline"))
        disp = (f"MFCC$_{name[4:]}$" if name.startswith("MFCC") else name)
        ax.annotate(disp, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, ha=ha, va=va)
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    ax.set_xlabel("Single-Channel Ridge $R^2$")
    ax.set_ylabel("Joint Gain over Single Channel ($R^2$)")
    order = ["Weakly Linearly Observable", "Intermediate",
             "Jointly Recoverable", "Captured (1-D)"]
    ax.legend(handles=[Patch(color=seen[l], label=l) for l in order if l in seen],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=8.0, columnspacing=1.0, handlelength=1.1)
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


FIGURES = {
    "fig_montage_daai_sequence.png": daai_sequence,
    "fig_trajectories.png": montage_and_trajectories,
    "fig_mechanistic.png": mechanistic,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional subset of figure filenames to write")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.only) if args.only else None
    for name, fn in FIGURES.items():
        if wanted and name not in wanted:
            continue
        fn(os.path.join(args.out, name))
        print("wrote", os.path.join(args.out, name))


if __name__ == "__main__":
    main()
