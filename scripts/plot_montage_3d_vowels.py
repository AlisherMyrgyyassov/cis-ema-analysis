"""Single-speaker 3D EMA montages for /baa/, /daai/, /gu/.

These are the same three syllables as the trajectory panel in
``fig_montage_trajectories.png`` (bilabial / alveolar / velar). One speaker,
one token each, one frame at 62% of the TextGrid vowel interval.

Writes ``paper/illustrations/3d/fig_montage_3d_{baa,daai,gu}.png``.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.prepare_dataset import parse_textgrid
from src.config import DATA_DIR, RAW_DIR, REPO_ROOT
from src.paper_figures import TONGUE_C, LIP_C, REF_C, _style_3d_axes, _sensor_colors

OUT_DIR = os.path.join(REPO_ROOT, "paper", "illustrations", "3d")
SYLLABLES = ("baa", "daai", "gu")
POS_COLS = [f"{s}{a}" for s in ("TD", "TL", "TT", "UL", "LL", "LC") for a in "xyz"]
Y_SIGN = -1.0
VOWEL_FRAC = 0.62


def _subj_folders():
    out = {}
    for name in os.listdir(RAW_DIR):
        p = os.path.join(RAW_DIR, name)
        if not os.path.isdir(p) or not name.startswith("sub_"):
            continue
        try:
            sid = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        out[sid] = p
    return out


def _tg_path(folder, file_num):
    num = f"{int(file_num):04d}"
    for sub in ("TG", "TG_edgeTrimmed"):
        p = os.path.join(folder, sub, f"{num}.TextGrid")
        if os.path.isfile(p):
            return p
    return None


def _word_vowel_span(tg_path, vowel):
    """Return (word_start, word_end, vowel_start, vowel_end) in TextGrid time."""
    tiers = parse_textgrid(tg_path)
    words = [(a, b, t.strip()) for a, b, t in tiers.get("word", []) if t.strip()]
    if not words:
        for key in ("syllable", "sound", "sonud"):
            words = [(a, b, t.strip()) for a, b, t in tiers.get(key, []) if t.strip()]
            if words:
                break
    if not words:
        return None
    ws, we, _ = words[0]
    segs = [(a, b, t.strip()) for a, b, t in tiers.get("segment", []) if t.strip()]
    vowel_hits = [(a, b) for a, b, t in segs
                  if t == vowel and a >= ws - 1e-4 and b <= we + 1e-4]
    if not vowel_hits and vowel == "aai":
        vowel_hits = [(a, b) for a, b, t in segs
                      if t in ("aai", "aa", "i") and a >= ws - 1e-4 and b <= we + 1e-4]
        if vowel_hits:
            va, vb = vowel_hits[0][0], vowel_hits[-1][1]
            if vb - va >= 0.04:
                return ws, we, va, vb
    if not vowel_hits:
        vowel_hits = [(a, b) for a, b, t in segs if t == vowel]
    if not vowel_hits:
        # Last labelled segment inside the word is the vowel of a CV syllable.
        inside = [(a, b) for a, b, t in segs if a >= ws - 1e-4 and b <= we + 1e-4]
        if len(inside) < 1:
            return None
        vowel_hits = [inside[-1]]
    va, vb = max(vowel_hits, key=lambda ab: ab[1] - ab[0])
    if vb - va < 0.04:
        return None
    return ws, we, va, vb


def _vowel_frames(g, span):
    """Map TextGrid vowel bounds onto this utterance's feature rows."""
    ws, we, va, vb = span
    t = g["time"].to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    dur = we - ws
    if dur <= 1e-6:
        return g.iloc[0:0]
    ema_va = t0 + (va - ws) / dur * (t1 - t0)
    ema_vb = t0 + (vb - ws) / dur * (t1 - t0)
    m = (t >= ema_va - 1e-6) & (t <= ema_vb + 1e-6)
    return g.loc[m]


def _vowel_target_row(vowel_g):
    """Single frame at VOWEL_FRAC of the vowel interval."""
    n = len(vowel_g)
    if n == 0:
        return None
    i = int(round(VOWEL_FRAC * (n - 1)))
    i = min(max(i, 0), n - 1)
    return vowel_g.iloc[i], i, n


def _pick_speaker(man, feat, folders):
    """Speaker with /baa/, /daai/ and /gu/ and the longest shortest-vowel."""
    ranked = []
    for subj, folder in folders.items():
        sub = man[man["subj"] == subj]
        if not set(SYLLABLES).issubset(set(sub["syllable"])):
            continue
        cand = {}
        ok = True
        min_vowel_n = 10**9
        for syl in SYLLABLES:
            pool = sub[sub["syllable"] == syl].sort_values(
                "n_frames", ascending=False)
            chosen = None
            for _, tok in pool.iterrows():
                tg = _tg_path(folder, tok["file_num"])
                if tg is None:
                    continue
                span = _word_vowel_span(tg, tok["vowel"])
                if span is None:
                    continue
                g = feat[feat["source"] == tok["source"]]
                vg = _vowel_frames(g, span)
                got = _vowel_target_row(vg)
                if got is None:
                    continue
                row, idx, n_vowel = got
                if not _canonical_side(row):
                    continue
                chosen = (tok, span, row, idx, n_vowel)
                break
            if chosen is None:
                ok = False
                break
            cand[syl] = chosen
            min_vowel_n = min(min_vowel_n, chosen[4])
        if ok:
            ranked.append((min_vowel_n, subj, cand))
    if not ranked:
        raise SystemExit("no speaker has canonical-side /baa/, /daai/ and /gu/")
    ranked.sort(reverse=True)
    min_n, subj, cand = ranked[0]
    return subj, cand, min_n


def _canonical_side(row):
    return float(row["TLy"]) > 0.12 and float(row["LCy"]) > 0.35


def _xyz_from_row(row):
    xyz = {}
    for s in ("TD", "TL", "TT", "UL", "LL", "LC"):
        xyz[s] = np.array([row[f"{s}x"], Y_SIGN * row[f"{s}y"], row[f"{s}z"]],
                          dtype=float)
    return xyz


def _shared_limits(all_xyz):
    pts = np.stack([p for xyz in all_xyz.values() for p in xyz.values()])
    lo = pts.min(0) - 0.18
    hi = pts.max(0) + 0.18
    lo[1] = min(lo[1], -0.20)
    hi[1] = max(hi[1], 0.20)
    lo[2] = min(lo[2], -0.12)
    return {
        "x": (float(lo[0]), float(hi[0])),
        "y": (float(lo[1]), float(hi[1])),
        "z": (float(lo[2]), float(hi[2])),
    }


def _draw(ax, xyz, limits):
    color = _sensor_colors()
    x_lo, x_hi = limits["x"]
    y_lo, y_hi = limits["y"]
    z_lo, z_hi = limits["z"]
    plane = [[(x_lo, 0.0, z_lo), (x_hi, 0.0, z_lo),
              (x_hi, 0.0, z_hi), (x_lo, 0.0, z_hi)]]
    ax.add_collection3d(Poly3DCollection(
        plane, facecolor=(0.45, 0.58, 0.72, 0.12),
        edgecolor=(0.45, 0.50, 0.58, 0.40), linewidth=0.7, zorder=1))
    ax.text(x_hi - 0.12, 0.0, z_hi - 0.10, "Midsagittal",
            fontsize=7.5, color="0.42", style="italic", zorder=2, ha="right")

    def _seg(a, b, c, lw=1.35, ls="-"):
        ax.plot([xyz[a][0], xyz[b][0]],
                [xyz[a][1], xyz[b][1]],
                [xyz[a][2], xyz[b][2]],
                color=c, lw=lw, ls=ls, zorder=3, solid_capstyle="round")

    _seg("TD", "TT", TONGUE_C, lw=1.6)
    _seg("TD", "TL", TONGUE_C, lw=1.2, ls=":")
    _seg("TL", "TT", TONGUE_C, lw=1.2, ls=":")
    _seg("UL", "LL", LIP_C, lw=1.6)
    _seg("UL", "LC", LIP_C, lw=1.2, ls=":")
    _seg("LL", "LC", LIP_C, lw=1.2, ls=":")

    for name in ("TL", "LC"):
        p = xyz[name]
        ax.plot([p[0], p[0]], [p[1], 0.0], [p[2], p[2]],
                color=color[name], lw=0.9, ls="--", alpha=0.6, zorder=2)
        ax.scatter([p[0]], [0.0], [p[2]], s=18, facecolors="white",
                   edgecolors=color[name], linewidths=0.8, depthshade=False,
                   zorder=4)

    for name, p in xyz.items():
        ax.scatter([p[0]], [p[1]], [p[2]],
                   s=95,
                   color=color[name], edgecolor="white", linewidth=0.8,
                   depthshade=False, zorder=6)

    off = {
        "TD": (-0.20, 0.04, 0.12),
        "TL": (0.02, -0.16, 0.10),
        "TT": (0.16, 0.08, -0.05),
        "UL": (0.18, 0.04, 0.10),
        "LL": (0.20, 0.04, -0.14),
        "LC": (0.04, -0.18, 0.08),
    }
    for name, p in xyz.items():
        d = off[name]
        ax.text(p[0] + d[0], p[1] + d[1], p[2] + d[2], name,
                fontsize=9.5, fontweight="bold", color="0.12",
                zorder=7, ha="center", va="center")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0])
    ax.set_yticks([-1.0, -0.5, 0.0])
    ax.set_zticks([0.0, 0.5, 1.0])


def _save_one(xyz, token, subj, limits, out_path):
    fig = plt.figure(figsize=(5.15, 4.15))
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax)
    _draw(ax, xyz, limits)
    ax.view_init(elev=22, azim=-38)
    ax.text2D(1.16, 0.52, "Height", transform=ax.transAxes,
              rotation=90, ha="center", va="center", fontsize=9)
    ax.text2D(0.0, 1.02, f"/{token['syllable']}/  ·  sub {subj}",
              transform=ax.transAxes, ha="left", va="bottom",
              fontsize=10, fontweight="bold")
    handles = [Patch(color=TONGUE_C, label="Tongue (TD, TL, TT)"),
               Patch(color=LIP_C, label="Lips (UL, LL, LC)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=2, frameon=False, fontsize=8, columnspacing=1.3,
              handlelength=1.2)
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.16, top=0.94)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))
    feat = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                       usecols=["source", "time"] + POS_COLS)
    folders = _subj_folders()
    subj, cand, min_n = _pick_speaker(man, feat, folders)

    xyzs = {syl: _xyz_from_row(cand[syl][2]) for syl in SYLLABLES}
    limits = _shared_limits(xyzs)
    print(f"speaker sub_{subj:02d}  (shortest vowel interval {min_n} frames)")
    for syl in SYLLABLES:
        tok, span, row, idx, n_vowel = cand[syl]
        xyz = xyzs[syl]
        out = os.path.join(OUT_DIR, f"fig_montage_3d_{syl}.png")
        _save_one(xyz, tok, subj, limits, out)
        ws, we, va, vb = span
        print(
            f"wrote {out}\n"
            f"  {tok['source']}  /{tok['syllable']}/  onset={tok['onset']}  "
            f"vowel {vb-va:.3f}s  frame {idx}/{n_vowel} "
            f"({VOWEL_FRAC:.0%} of vowel)\n"
            f"  TDx={row['TDx']:+.3f} TDz={row['TDz']:+.3f}  "
            f"TTx={row['TTx']:+.3f} TTz={row['TTz']:+.3f}"
        )
    for stale in ("aa", "u", "i"):
        p = os.path.join(OUT_DIR, f"fig_montage_3d_{stale}.png")
        if os.path.isfile(p):
            os.remove(p)
            print("removed", p)


if __name__ == "__main__":
    main()
