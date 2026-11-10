"""Export coil snapshots for external (Illustrator) annotation.

Two modes, both using the same speaker, /daai/ token, timestamps and 3D axis
limits as the manuscript's Fig. 1 top row:

  --kind 3d          the three 3D montage panels (as in Fig. 1)
  --kind midsagittal a 2D mid-sagittal projection (front-back vs height) of the
                     four mid-sagittal coils (TD, TT, UL, LL), with the lateral
                     TL and LC coils omitted, on a transparent background

Writes coil_constriction.png / coil_vowel_target.png / coil_offglide.png plus a
combined coil_sequence.png (3d), or midsagittal_*.png (midsagittal).
"""
import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from src.config import DATA_DIR
from src.paths import get as path

DEFAULT_OUT = os.path.join(str(path("figures")), "coil_panels")
STAGE_NAMES = ["constriction", "vowel_target", "offglide"]
MIDSAG_COILS = ("TD", "TT", "UL", "LL")   # medial coils only
FIG1_ASPECT = 1.0723                      # aspect of Fig. 1's montage subplots


def _load(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    name = os.path.splitext(filename)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stages_and_limits(dpi):
    """Reproduce Fig. 1's top row: speaker, /daai/ token, stages, 3D limits."""
    baa = _load("plot_montage_3d_baa.py")
    seq, vow = baa.seq, baa.vow
    man = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))
    feat = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                       usecols=["source", "time"] + seq.POS_COLS)
    speakers = seq._collect_speakers(man, feat, vow._subj_folders())
    subj, picked = seq._choose(speakers, np.random.RandomState(seq.DEFAULT_SEED))
    rec = picked["daai"]
    g = rec["g"]
    stages = seq._pick_stages(g, "daai", rec["span"], rec["onset"])
    stages = baa._consonant_at_segment_mid(g, rec, stages)
    all_xyz = {st: stages[st]["xyz"] for st in seq.STAGES}
    limits = seq._tight_limits(all_xyz)
    return baa, subj, stages, limits


# --------------------------------------------------------------------------- #
# 3D montage panels (Fig. 1 style)
# --------------------------------------------------------------------------- #
def _save_3d(baa, stages, limits, out_dir, dpi):
    seq = baa.seq
    ims220 = [baa._render_3d(stages[st]["xyz"], limits, show_labels=True, dpi=220)
              for st in seq.STAGES]
    ims220 = baa._crop_equal(ims220, pad=18)
    ims220 = baa._white_border(ims220, px=28)
    h0, w0 = ims220[0].shape[:2]
    aspect = w0 / h0

    imgs = [baa._render_3d(stages[st]["xyz"], limits, show_labels=True, dpi=dpi)
            for st in seq.STAGES]
    r = dpi / 220.0
    imgs = baa._crop_equal(imgs, pad=round(18 * r))
    hc, wc = imgs[0].shape[:2]
    border = max(0, round((wc - aspect * hc) / (2 * (aspect - 1))))
    imgs = baa._white_border(imgs, px=border)
    for im, nm in zip(imgs, STAGE_NAMES):
        h, w = im.shape[:2]
        fig = plt.figure(figsize=(w / 300.0, h / 300.0), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(im, interpolation="hanning")
        ax.set_axis_off()
        path = os.path.join(out_dir, f"coil_{nm}.png")
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print(f"wrote {path}  ({w}x{h}px)")
    fig = plt.figure(figsize=(8.4, 2.75))
    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.88, bottom=0.02,
                          wspace=0.06)
    for i, st in enumerate(seq.STAGES):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(imgs[i], interpolation="hanning")
        h, w = imgs[i].shape[:2]
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(baa.STAGE_C[st])
            spine.set_linewidth(1.15)
        ax.set_title(baa.STAGE_TITLE["daai"][st], color=baa.STAGE_C[st],
                     fontsize=10, fontweight="bold", pad=6)
    path = os.path.join(out_dir, "coil_sequence.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# 2D mid-sagittal projection (medial coils only, transparent)
# --------------------------------------------------------------------------- #
def _medial_limits(baa, stages):
    """Union of the four medial coils over the stages, padded for the labels."""
    xs, zs = [], []
    for st in baa.seq.STAGES:
        for nm in MIDSAG_COILS:
            p = stages[st]["xyz"][nm]
            xs.append(float(p[0]))
            zs.append(float(p[2]))
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    dx, dz = (x1 - x0) or 1.0, (z1 - z0) or 1.0
    return (x0 - 0.26 * dx, x1 + 0.26 * dx), (z0 - 0.30 * dz, z1 + 0.30 * dz)


def _fit_aspect(xlim, zlim, aspect):
    """Expand x or z so the frame matches `aspect` (data stays equal-aspect)."""
    xr, zr = xlim[1] - xlim[0], zlim[1] - zlim[0]
    if xr / zr > aspect:
        zr2 = xr / aspect
        zc = 0.5 * (zlim[0] + zlim[1])
        zlim = (zc - zr2 / 2, zc + zr2 / 2)
    else:
        xr2 = zr * aspect
        xc = 0.5 * (xlim[0] + xlim[1])
        xlim = (xc - xr2 / 2, xc + xr2 / 2)
    return xlim, zlim


def _draw_midsagittal(ax, xyz, tongue_c, lip_c):
    ax.plot([xyz["TD"][0], xyz["TT"][0]], [xyz["TD"][2], xyz["TT"][2]],
            color=tongue_c, lw=2.4, solid_capstyle="round", zorder=3)
    ax.plot([xyz["UL"][0], xyz["LL"][0]], [xyz["UL"][2], xyz["LL"][2]],
            color=lip_c, lw=2.4, solid_capstyle="round", zorder=3)
    offs = {
        "TD": (-16, 12, "right", "bottom"),
        "TT": (16, 9, "left", "center"),
        "UL": (16, 7, "left", "center"),
        "LL": (16, -14, "left", "top"),
    }
    for name in MIDSAG_COILS:
        p = xyz[name]
        c = tongue_c if name in ("TD", "TT") else lip_c
        ax.scatter([p[0]], [p[2]], s=130, color=c, edgecolor="white",
                   linewidth=1.2, zorder=5)
        dx, dy, ha, va = offs[name]
        ax.annotate(name, (p[0], p[2]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=20, fontweight="bold",
                    color="0.08", ha=ha, va=va, zorder=6)


def _midsag_axes(fig, spec):
    ax = fig.add_subplot(spec)
    ax.patch.set_alpha(0.0)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.axis("off")
    return ax


def _save_midsagittal(baa, stages, out_dir, dpi, aspect=FIG1_ASPECT):
    tongue_c, lip_c = baa.TONGUE_C, baa.LIP_C
    xlim, zlim = _fit_aspect(*_medial_limits(baa, stages), aspect)
    xr, zr = xlim[1] - xlim[0], zlim[1] - zlim[0]
    print(f"midsagittal frame: x=[{xlim[0]:.3f}, {xlim[1]:.3f}] "
          f"z=[{zlim[0]:.3f}, {zlim[1]:.3f}]  aspect={xr / zr:.3f}")
    h_in = 3.2
    w_in = h_in * (xr / zr)
    for st, nm in zip(baa.seq.STAGES, STAGE_NAMES):
        fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = _midsag_axes(fig, 111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        _draw_midsagittal(ax, stages[st]["xyz"], tongue_c, lip_c)
        path = os.path.join(out_dir, f"midsagittal_{nm}.png")
        fig.savefig(path, transparent=True, dpi=dpi)
        plt.close(fig)
        print("wrote", path)

    fig = plt.figure(figsize=(3 * w_in, h_in), dpi=dpi)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(1, 3, left=0.003, right=0.997, top=0.997,
                          bottom=0.003, wspace=0.02)
    for i, st in enumerate(baa.seq.STAGES):
        ax = _midsag_axes(fig, gs[0, i])
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        _draw_midsagittal(ax, stages[st]["xyz"], tongue_c, lip_c)
    path = os.path.join(out_dir, "midsagittal_sequence.png")
    fig.savefig(path, transparent=True, dpi=dpi)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["3d", "midsagittal", "both"],
                    default="midsagittal")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    baa, subj, stages, limits = _stages_and_limits(args.dpi)
    if args.kind in ("3d", "both"):
        _save_3d(baa, stages, limits, args.out, args.dpi)
    if args.kind in ("midsagittal", "both"):
        _save_midsagittal(baa, stages, args.out, args.dpi)
    print(f"speaker sub_{subj:02d}  (kind={args.kind})")


if __name__ == "__main__":
    main()
