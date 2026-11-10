"""Figures for the paper (Exp 2-4)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import FEATURE_GROUPS, GROUP_IDS

# Exp-4 mechanistic taxonomy: each top acoustic feature's recovery mechanism.
MECHANISM = {
    "F0": ("structural gap", "#D1495B"),
    "F3": ("structural gap", "#D1495B"),
    "MFCC_2": ("jointly-recoverable", "#30638E"),
    "MFCC_5": ("jointly-recoverable", "#30638E"),
    "MFCC_6": ("jointly-recoverable", "#30638E"),
    "F2": ("captured (1-D)", "#2E8B57"),
}
MECH_INTERMEDIATE = ("intermediate", "#9AA0A6")


def plot_shapley(summary, out_path):
    agg = summary["aggregate"]["shapley_acc"]
    groups = sorted(GROUP_IDS, key=lambda g: agg[g]["mean"])
    names = [f"{g} {FEATURE_GROUPS[g]['name']}" for g in groups]
    means = [agg[g]["mean"] * 100 for g in groups]
    stds = [agg[g]["std"] * 100 for g in groups]
    colors = ["#4C72B0" if FEATURE_GROUPS[g]["kind"] == "ema" else "#DD8452"
              for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, means, xerr=stds, color=colors, capsize=3, alpha=0.9)
    ax.set_xlabel("Shapley value (accuracy points)")
    ax.set_title("SAGE feature-group Shapley decomposition")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="EMA"),
                       Patch(color="#DD8452", label="Acoustic")])
    ax.axvline(0, color="k", lw=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_probing(summary, out_path):
    stages = summary["stages"]
    props = summary["props"]
    stage_x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in props:
        ys = [summary["aggregate"][f"{p}__{s}"]["acc_mean"] * 100 for s in stages]
        ax.plot(stage_x, ys, marker="o", label=p)
    ax.set_xticks(stage_x)
    ax.set_xticklabels(stages)
    ax.set_xlabel("layer depth")
    ax.set_ylabel("probe accuracy (%)")
    ax.set_title("Linear probes across model depth")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_faithfulness(summary, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    n = len(summary["shapley_order"])
    xs = list(range(n + 1))
    for name, style in (("shapley", "-o"), ("reverse", "-s"), ("random", "-^")):
        d = summary["deletion"][name]
        ax1.plot(xs, np.array(d["mean"]) * 100, style, label=name)
    ax1.set_xlabel("groups removed")
    ax1.set_ylabel("accuracy (%)")
    ax1.set_title("Deletion curves")
    ax1.legend()
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 100)

    ins = summary["insertion"]
    xi = list(range(len(ins["mean"])))
    ax2.plot(xi, np.array(ins["mean"]) * 100, "-o", color="g")
    labels = ["EMA"] + summary["insertion_groups_added"]
    ax2.set_xticks(xi)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    ax2.set_xlabel("groups added (Shapley order)")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Insertion curve (from EMA-only)")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_mechanistic_scatter(corr_summary, out_path):
    """Exp-4 mechanistic taxonomy scatter.

    x = single-sensor ridge R^2 (global, within-utterance standardised).
    y = joint-minus-single ridge R^2 gain (the extra variance recovered by the
        other 49 EMA features jointly).

    Three clusters emerge: structural gap (F0, F3: low single, low gain),
    architecture-limited (MFCC_2/5/6: low single, high gain), and captured/1-D
    (F2: high single, ~zero gain).
    """
    table = corr_summary["table"]
    xs = [r["single_ridge_r2_in"] for r in table]
    ys = [r["joint_ridge_r2_in"] - r["single_ridge_r2_in"] for r in table]
    names = [r["acoustic_feature"] for r in table]

    fig, ax = plt.subplots(figsize=(8, 6))
    from matplotlib.patches import Patch
    seen = set()
    for x, y, name in zip(xs, ys, names):
        label, color = MECHANISM.get(name, MECH_INTERMEDIATE)
        ax.scatter(x, y, s=90, color=color, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=9)
        seen.add((label, color))

    ax.axhline(0, color="k", lw=0.8, zorder=1)
    ax.set_xlabel("single-sensor ridge R$^2$ (best EMA sensor)")
    ax.set_ylabel("joint $-$ single ridge R$^2$ gain")
    ax.set_title("How acoustic features map onto articulatory motion")
    handles = [Patch(color=c, label=l) for l, c in sorted(seen)]
    ax.legend(handles=handles, loc="best", framealpha=0.9)
    fig.text(0.02, 0.01,
             "MFCC_2/5/6 are jointly recoverable from EMA, but Experiment 5 shows "
             "this recoverability does not translate to a classification gain.",
             fontsize=8, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
