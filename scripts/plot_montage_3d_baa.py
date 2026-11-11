"""Packed single-syllable figure: three 3D snapshots over height traces.

Same layout for /baa/, /daai/, or /gu/. Default remains /baa/.

    python scripts/plot_montage_3d_baa.py
    python scripts/plot_montage_3d_baa.py --syl daai

Does not touch ``plot_montage_3d_vowels.py`` or ``fig_montage_3d_sequence.png``.

Writes ``paper/illustrations/3d/fig_montage_3d_{syl}_sequence.png``.
"""
import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


def _load_mod(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    name = os.path.splitext(filename)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seq = _load_mod("plot_montage_3d_sequence.py")
vow = seq.base

from src.config import DATA_DIR
from src.paper_figures import TONGUE_C, LIP_C, REF_C, _style_3d_axes
import pandas as pd

STAGES = seq.STAGES
STAGE_C = seq.STAGE_C
OUT_DIR = seq.OUT_DIR
POS_COLS = seq.POS_COLS
PLACE = seq.PLACE
PHONE_ONSET = "#C45C4A"
PHONE_VOWEL = "#6A8BB8"
# Packed-figure display names. Internal stage keys stay consonant/vowel/end.
STAGE_TITLE = {
    "baa": {"consonant": "Consonant", "vowel": "Vowel", "end": "End"},
    "gu": {"consonant": "Consonant", "vowel": "Vowel", "end": "End"},
    "daai": {
        "consonant": "Constriction",
        "vowel": "Vowel target",
        "end": "Offglide",
    },
}
STAGE_MARK = {
    "baa": {"consonant": "C", "vowel": "V", "end": "E"},
    "gu": {"consonant": "C", "vowel": "V", "end": "E"},
    "daai": {"consonant": "C", "vowel": "V", "end": "O"},
}


def _content_bbox(img, thresh=250):
    rgb = img[..., :3]
    ink = np.any(rgb < thresh, axis=-1)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, img.shape[0], img.shape[1]
    return int(rows[0]), int(cols[0]), int(rows[-1] + 1), int(cols[-1] + 1)


def _crop_equal(imgs, pad=10):
    """Tight-crop each RGBA image, then pad to a shared canvas."""
    crops = []
    for im in imgs:
        r0, c0, r1, c1 = _content_bbox(im)
        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
        r1 = min(im.shape[0], r1 + pad)
        c1 = min(im.shape[1], c1 + pad)
        crops.append(im[r0:r1, c0:c1])
    mh = max(c.shape[0] for c in crops)
    mw = max(c.shape[1] for c in crops)
    out = []
    for c in crops:
        canvas = np.full((mh, mw, c.shape[2]), 255, dtype=c.dtype)
        if c.shape[2] == 4:
            canvas[..., 3] = 255
        y = (mh - c.shape[0]) // 2
        x = (mw - c.shape[1]) // 2
        canvas[y:y + c.shape[0], x:x + c.shape[1]] = c
        out.append(canvas)
    return out


def _white_border(imgs, px=36):
    """Keep colored panel frames off the 3D drawing."""
    out = []
    for im in imgs:
        h, w, c = im.shape
        canvas = np.full((h + 2 * px, w + 2 * px, c), 255, dtype=im.dtype)
        if c == 4:
            canvas[..., 3] = 255
        canvas[px:px + h, px:px + w] = im
        out.append(canvas)
    return out


def _render_3d(xyz, limits, show_labels, dpi=220):
    """Standalone 3D panel, same geometry as ``plot_montage_3d_vowels._draw``."""
    fig = plt.figure(figsize=(4.6, 3.7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax)
    vow._draw(ax, xyz, limits)
    ax.view_init(elev=22, azim=-38)
    x_lo, x_hi = limits["x"]
    y_lo, y_hi = limits["y"]
    z_lo, z_hi = limits["z"]
    ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo), zoom=1.05)
    ax.tick_params(pad=1, labelsize=7.5)
    ax.set_xlabel("Front–Back", labelpad=2, fontsize=8)
    ax.set_ylabel("Left–Right", labelpad=2, fontsize=8)
    ax.set_zlabel("Height", labelpad=3, fontsize=8)
    if not show_labels:
        for txt in list(ax.texts):
            lab = txt.get_text()
            if lab in ("TD", "TL", "TT", "UL", "LL", "LC", "Midsagittal"):
                txt.remove()
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.94)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def _t0(g):
    return float(g["time"].to_numpy(dtype=float)[0])


def _rel_time(g, tg_t, rec):
    ema = seq._ema_time(g, rec["span"], tg_t)
    return float(ema - _t0(g))


def _phone_windows(g, rec):
    """Utterance-relative times (s) of TextGrid onset and vowel."""
    os_, oe = rec["onset"]
    _, _, va, vb = rec["span"]
    return {
        "onset": (_rel_time(g, os_, rec), _rel_time(g, oe, rec)),
        "vowel": (_rel_time(g, va, rec), _rel_time(g, vb, rec)),
    }


def _stage_time(g, rec_st):
    return float(rec_st["row"]["time"] - _t0(g))


def _consonant_at_segment_mid(g, rec, stages):
    """Put C at the midpoint of the labelled onset so it is not drawn on the y-axis."""
    os_, oe = rec["onset"]
    tgt = 0.5 * (seq._ema_time(g, rec["span"], os_) + seq._ema_time(g, rec["span"], oe))
    row, idx = seq._nearest_row(g, tgt)
    stages["consonant"] = {
        "row": row,
        "index": int(idx),
        "frac": seq._frac(g, row),
        "xyz": vow._xyz_from_row(row),
    }
    return stages


def _traj_ylim(g, cols, pad=0.12):
    lo = min(float(g[c].min()) for c in cols)
    hi = max(float(g[c].max()) for c in cols)
    span = hi - lo if hi > lo else 1.0
    return lo - pad * span, hi + pad * span


def _shade_phones(ax, phones):
    o0, o1 = phones["onset"]
    v0, v1 = phones["vowel"]
    ax.axvspan(o0, o1, facecolor=PHONE_ONSET, alpha=0.28, lw=0, zorder=0)
    ax.axvspan(v0, v1, facecolor=PHONE_VOWEL, alpha=0.11, lw=0, zorder=0)
    ax.axvline(o1, color=PHONE_ONSET, ls=":", lw=1.3, zorder=1)
    ax.axvline(v1, color=PHONE_VOWEL, ls=":", lw=1.15, zorder=1)


def _plot_traj(ax, g, stages, col, ylim, xlabel, phones):
    t = g["time"].to_numpy(dtype=float)
    x = t - t[0]
    y = g[col].to_numpy(dtype=float)
    _shade_phones(ax, phones)
    ax.plot(x, y, color="0.12", lw=1.85, zorder=3, solid_capstyle="round")
    for stage in STAGES:
        ts = _stage_time(g, stages[stage])
        ax.axvline(ts, color=STAGE_C[stage], lw=1.85, zorder=5, solid_capstyle="butt")
        yi = float(np.interp(ts, x, y))
        ax.scatter([ts], [yi], s=38, color=STAGE_C[stage], edgecolor="white",
                   linewidth=0.85, zorder=6)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    if xlabel:
        ax.set_xlabel("Time (s)", labelpad=6)
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelsize=8, pad=2)
    ax.grid(True, axis="y", color="0.9", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_figure(subj, rec, syl, title=True):
    g = rec["g"]
    stages = seq._pick_stages(g, syl, rec["span"], rec["onset"])
    stages = _consonant_at_segment_mid(g, rec, stages)
    rec["stages"] = stages
    all_xyz = {st: stages[st]["xyz"] for st in STAGES}
    limits = seq._tight_limits(all_xyz)

    imgs = _white_border(_crop_equal([
        _render_3d(stages[st]["xyz"], limits, show_labels=(i == 0))
        for i, st in enumerate(STAGES)
    ], pad=18), px=28)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(8.4, 6.15))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.22, 1.00],
        left=0.09, right=0.985,
        top=0.90 if title else 0.965, bottom=0.14,
        hspace=0.12,
    )
    gs_top = gs[0].subgridspec(1, 3, wspace=0.06)
    gs_bot = gs[1].subgridspec(2, 1, hspace=0.32)

    if title:
        fig.text(0.09, 0.965, f"/{syl}/  ·  {PLACE[syl]}  ·  sub {subj}",
                 ha="left", va="top", fontsize=12, fontweight="bold")

    for i, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(imgs[i], interpolation="hanning")
        h, w = imgs[i].shape[:2]
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_clip_on(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(STAGE_C[stage])
            spine.set_linewidth(1.15)
            spine.set_clip_on(False)
        ax.set_title(STAGE_TITLE[syl][stage], color=STAGE_C[stage],
                     fontsize=10, fontweight="bold", pad=6)

    phones = _phone_windows(g, rec)
    g = g.copy()
    g["Lip aperture"] = np.sqrt((g["ULx"] - g["LLx"]) ** 2 +
                                (g["ULy"] - g["LLy"]) ** 2 +
                                (g["ULz"] - g["LLz"]) ** 2)
    td_lim = _traj_ylim(g, ["TDz"])
    ap_lim = _traj_ylim(g, ["Lip aperture"])
    ax_td = fig.add_subplot(gs_bot[0, 0])
    ax_ul = fig.add_subplot(gs_bot[1, 0], sharex=ax_td)
    _plot_traj(ax_td, g, stages, "TDz", td_lim, False, phones)
    _plot_traj(ax_ul, g, stages, "Lip aperture", ap_lim, True, phones)
    ax_td.set_ylabel("Tongue Height")
    ax_ul.set_ylabel("Lip Aperture")
    t = g["time"].to_numpy(dtype=float)
    t_lo, t_hi = 0.0, float(t[-1] - t[0])
    for stage in STAGES:
        ts = _stage_time(g, stages[stage])
        ha = "center"
        if ts < t_lo + 0.02 * (t_hi - t_lo):
            ha = "left"
        elif ts > t_hi - 0.02 * (t_hi - t_lo):
            ha = "right"
        ax_td.annotate(
            STAGE_MARK[syl][stage], xy=(ts, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3), textcoords="offset points",
            ha=ha, va="bottom", fontsize=9, fontweight="bold",
            color=STAGE_C[stage], annotation_clip=False)

    titles = STAGE_TITLE[syl]
    handles = [
        Line2D([0], [0], color=STAGE_C[s], lw=1.8, label=titles[s])
        for s in STAGES
    ] + [
        Patch(color=TONGUE_C, label="Tongue (TD, TL, TT)"),
        Patch(color=LIP_C, label="Lips (UL, LL, LC)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.54, 0.052), ncol=5, frameon=False,
               fontsize=8, columnspacing=1.2, handlelength=1.35)
    return fig, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=seq.DEFAULT_SEED)
    ap.add_argument("--subj", type=int, default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--syl", choices=seq.SYLLABLES, default="baa")
    ap.add_argument("--no-title", action="store_true",
                    help="Omit the in-image title (for the manuscript figure)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))
    feat = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                       usecols=["source", "time"] + POS_COLS)
    folders = vow._subj_folders()
    speakers = seq._collect_speakers(man, feat, folders)
    rng = np.random.RandomState(args.seed)
    subj, picked = seq._choose(
        speakers, rng, args.subj, random_subj=args.random)
    rec = picked[args.syl]

    fig, stages = _build_figure(subj, rec, args.syl, title=not args.no_title)
    out = args.out or os.path.join(OUT_DIR, f"fig_montage_3d_{args.syl}_sequence.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)

    tok = rec["tok"]
    phones = _phone_windows(rec["g"], rec)
    print(f"speaker sub_{subj:02d}  {tok['source']}  {tok['n_frames']} frames")
    print(
        f"  phones  /{tok['onset']}/={phones['onset'][0]:.3f}-{phones['onset'][1]:.3f}s  "
        f"/{tok['vowel']}/={phones['vowel'][0]:.3f}-{phones['vowel'][1]:.3f}s"
    )
    for st, rec_st in stages.items():
        row = rec_st["row"]
        print(
            f"  {st[0].upper()}  t={_stage_time(rec['g'], rec_st):.3f}s  "
            f"TDz={row['TDz']:+.3f}  ULz={row['ULz']:+.3f}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
