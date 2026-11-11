"""Paper figures for the TASLP EMA-observability manuscript.

Generates clean, presentation-ready figures for a two-column IEEE layout.
Conventions: no descriptive titles (they live in the LaTeX captions), capitalised
axis labels and legends, single hyphens only, and compact readable annotations.

Run with:

    conda run -n audio-env python -m src.paper_figures --out <dir>

Only reads saved artifacts / the aligned dataset; no model training.
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
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FormatStrFormatter

from src.config import RESULTS_DIR, DATA_DIR, RUNS_DIR, REPO_ROOT
from src.figcheck import save as _save_fig

from src import figstyle  # noqa: E402

figstyle.apply()

EMA_C = figstyle.NAVY
ACOUSTIC_C = figstyle.AMBER
SSL_C = "#5B8C5A"
FULL_C = "#8E6C8A"
GRID_C = "#BBBBBB"
TONGUE_C = "#2F6DB5"
LIP_C = "#C36B4A"
REF_C = "#6E6E6E"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _panel(ax, s, loc="upper left", color="black"):
    x, y = (0.02, 0.97) if loc == "upper left" else (0.98, 0.97)
    ha = "left" if loc == "upper left" else "right"
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top",
            fontsize=10, fontweight="bold", color=color)


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


# --------------------------------------------------------------------------- #
# Fig: observability taxonomy (conceptual)
# --------------------------------------------------------------------------- #
def observability_taxonomy(out_path):
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.0, 11.2)
    ax.axis("off")
    specs = [
        ("#E9F0F7", EMA_C, "Directly observed",
         "Position, velocity,\nacceleration, geometry"),
        ("#DCE8E0", "#5F8F69", "Redundantly encoded",
         "Position versus derivatives\nand geometry"),
        ("#F7E8D8", ACOUSTIC_C, "Indirectly recoverable",
         "F2, mid-band MFCCs"),
        ("#F4DCDC", "#C0504D", "Weakly observable",
         "F0/voicing, F3/rounding,\naspiration detail"),
    ]
    n, gap, x_left, x_right = 4, 1.8, 0.4, 99.6
    width = ((x_right - x_left) - gap * (n - 1)) / n
    boxes = []
    for i, (fc, ec, title, body) in enumerate(specs):
        x0 = x_left + i * (width + gap)
        x1 = x0 + width
        boxes.append((x0, x1))
        ax.add_patch(FancyBboxPatch((x0, 2.35), width, 6.15,
                                    boxstyle="round,pad=0.12,rounding_size=0.45",
                                    linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text((x0 + x1) / 2, 7.75, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=ec)
        ax.text((x0 + x1) / 2, 4.65, body, ha="center", va="center", fontsize=7.5)
    split = (boxes[1][1] + boxes[2][0]) / 2
    ax.plot([split, split], [1.45, 9.55], color="0.3", lw=1.2, ls="--")
    left_mid = (boxes[0][0] + boxes[1][1]) / 2
    right_mid = (boxes[2][0] + boxes[3][1]) / 2
    ax.text(left_mid, 9.55, "Measured or derived from the coils", ha="center",
            va="bottom", fontsize=8, color="0.35", style="italic")
    ax.text(right_mid, 9.55, "Not directly measured", ha="center",
            va="bottom", fontsize=8, color="0.35", style="italic")
    ax.annotate("", xy=(99.2, 1.05), xytext=(0.8, 1.05),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2))
    ax.text(50, -0.35, "Increasing difficulty of recovery from articulatory motion",
            ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Fig: grouped confusion by place of articulation
# --------------------------------------------------------------------------- #
def grouped_confusion(out_path):
    cg = _load(os.path.join(RESULTS_DIR, "confusion_grouped.json"))
    places = cg["places"]
    abbr = [p.capitalize() for p in places]
    models = [("ema_only", "EMA-only"), ("full", "Full"), ("hubert", "HuBERT")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    for letter, (ax, (key, title)) in zip("abc", zip(axes, models)):
        m = np.array(cg["models"][key]["grouped_norm_pct"])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(places)))
        ax.set_yticks(range(len(places)))
        # 90 deg: at 45 deg the longer place names (Bilabial, Labiodental) run into
        # each other once the panel is narrowed to a third of the figure width.
        ax.set_xticklabels(abbr, rotation=90, ha="center", va="top", fontsize=7.0)
        ax.set_yticklabels(abbr if key == "ema_only" else [""] * len(places),
                           fontsize=7.0)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                        color="white" if v > 55 else "0.2")
        ax.set_xlabel("Predicted Place")
        if key == "ema_only":
            ax.set_ylabel("True Place")
        ax.text(0.5, 1.03, f"({letter})", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="Row-normalized (%)")
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Fig: tone probe confound (naive vs syllable-controlled)
# --------------------------------------------------------------------------- #
def tone_control(out_path):
    """Syllable-controlled tone decoding (robust_z).

    (a) On all utterances the probes sit *below* the syllable-to-tone majority
        ceiling (90.2%). (b) Restricted to syllables that occur with >=2 tones,
        the EMA probe and the full model are both *above* the per-syllable
        majority baseline.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    labels = ["EMA-only", "Full"]
    colors = [EMA_C, FULL_C]

    def panel(ax, vals, base, base_label, letter):
        x = np.arange(2)
        ax.bar(x, vals, color=colors, width=0.6, alpha=0.92)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
        ax.axhline(base, color="#C0504D", ls="--", lw=1.3)
        ax.text(1.45, base + 1.5, base_label, ha="right", fontsize=7,
                color="#C0504D")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 102)
        ax.set_ylabel("Tone Accuracy (%)")
        _panel(ax, letter)

    tw = _load(os.path.join(RUNS_DIR, "tone", "tone_within_syllable.json"))
    all_vals = [tw["naive_ema"]["probe_mean"] * 100, tw["naive_full"]["probe_mean"] * 100]
    all_base = tw["naive_full"]["baseline_mean"] * 100
    multi_vals = [tw["within_ema"]["probe_mean"] * 100, tw["within_full"]["probe_mean"] * 100]
    multi_base = tw["within_full"]["baseline_mean"] * 100
    panel(ax1, all_vals, all_base,
          "Syllable-to-tone ceiling", "(a) All utterances")
    panel(ax2, multi_vals, multi_base,
          "Per-syllable majority", "(b) Multi-tone syllables")
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Fig: Shapley stability across runs (per-run distribution)
# --------------------------------------------------------------------------- #
def shapley_stability(out_path):
    d = _load(os.path.join(RUNS_DIR, "exp2", "shapley.json"))
    agg = d["aggregate"]["shapley_acc"]
    names = d["names"]
    kind = d["kind"]
    short = {"G1": "Position", "G2": "Velocity", "G3": "Acceleration",
             "G4": "Geometry", "G5": "Voicing", "G6": "Formant",
             "G7": "Spectral", "G8": "Energy"}
    groups = sorted(d["groups"], key=lambda g: agg[g]["mean"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    rng = np.random.RandomState(0)
    for i, g in enumerate(groups):
        vals = np.array(agg[g]["values"]) * 100
        c = EMA_C if kind[g] == "ema" else ACOUSTIC_C
        ax.scatter(vals, i + rng.uniform(-0.14, 0.14, len(vals)), s=14,
                   color=c, alpha=0.7, edgecolor="white", linewidth=0.3, zorder=2)
        ax.plot([agg[g]["mean"] * 100] * 2, [i - 0.26, i + 0.26], color="k",
                lw=1.6, zorder=3)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g} {short[g]}" for g in groups], fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Shapley Value (Accuracy Points)")
    ax.legend(handles=[Patch(color=EMA_C, label="EMA"),
                       Patch(color=ACOUSTIC_C, label="Acoustic")],
              frameon=False, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Regenerated result figures (clean titles/labels)
# --------------------------------------------------------------------------- #
def _probing(out_path, src, letter):
    d = _load(os.path.join(RUNS_DIR, "exp3", src))
    stages = d["stages"]
    props = d["props"]
    stage_labels = ["Proj", "Layer 0", "Layer 1", "Pooled"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for p in props:
        if p == "tone":  # tone is confounded with syllable; handled separately
            continue
        ys = [d["aggregate"][f"{p}__{s}"]["acc_mean"] * 100 for s in stages]
        ax.plot(range(len(stages)), ys, marker="o", label=p.capitalize())
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stage_labels, fontsize=7.5)
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    _panel(ax, letter)
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def probing_full(out_path):
    return _probing(out_path, "probing_full.json", "(a)")


def probing_ema(out_path):
    return _probing(out_path, "probing_ema.json", "(b)")


def faithfulness(out_path):
    d = _load(os.path.join(RUNS_DIR, "exp3", "faithfulness.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    n = len(d["shapley_order"])
    xs = list(range(n + 1))
    styles = {"shapley": "-o", "reverse": "-s", "random": "-^"}
    for name, style in styles.items():
        ax1.plot(xs, np.array(d["deletion"][name]["mean"]) * 100, style,
                 label=name.capitalize(), markersize=4)
    ax1.set_xlabel("Groups Removed")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, loc="center", fontsize=8,
               handlelength=1.4, borderaxespad=0.2)
    _panel(ax1, "(a)")

    ins = d["insertion"]
    xi = list(range(len(ins["mean"])))
    ax2.plot(xi, np.array(ins["mean"]) * 100, "-o", color=SSL_C, markersize=4)
    labels = ["EMA"] + [g.capitalize() for g in d["insertion_groups_added"]]
    ax2.set_xticks(xi)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax2.set_xlabel("Groups Added (Shapley Order)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    _panel(ax2, "(b)")
    fig.tight_layout()
    _save_fig(fig, out_path, bbox_inches="tight")
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
    "fig_taxonomy.png": observability_taxonomy,
    "fig_grouped_confusion.png": grouped_confusion,
    "fig_tone_control.png": tone_control,
    "fig_shapley_stability.png": shapley_stability,
    "fig_probing_full.png": probing_full,
    "fig_probing_ema.png": probing_ema,
    "fig_faithfulness.png": faithfulness,
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
