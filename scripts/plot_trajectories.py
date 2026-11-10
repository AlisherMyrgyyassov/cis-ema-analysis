"""Paper illustration: normalized articulatory trajectories.

Randomly selects ``--n-syllables`` syllables with a fixed seed, then plots the
within-utterance-normalized trajectories of
  (1) the selected full syllables,
  (2) every vowel occurring in those syllables,
  (3) every onset  occurring in those syllables.

Normalization ("normalized pronunciation"): each utterance is
  - time-warped onto a common 0..1 grid (interpolated to ``--n-frames``),
  - z-scored per channel within the utterance (speaker scale/level removed),
so the overlaid mean (solid) and +/-1 std band (shaded) reflect the *shape* of
the articulatory gesture across speakers, not speaker level or duration.

Feature computation reproduces ``preprocessing/build_features.py`` (LL-reference
coordinate normalization, Savitzky-Golay + 20 Hz Butterworth smoothing, and the
UL-LL lip aperture).

Output: ``{out_dir}/paper/illustrations/trajectories`` with one comparison
figure per group (syllables/vowels/onsets) plus an isolated figure per class.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402
from preprocessing.build_features import (
    EMA_COLS, KEEP_SENSORS, AXES,
    fix_outliers, normalize_coords, lowpass,
)

# Panels shown in every figure: (title, channel key, y-label).
PANELS = [
    ("TTx", "TTx", "tongue tip front-back"),
    ("TTz", "TTz", "tongue tip height"),
    ("TDx", "TDx", "tongue dorsum front-back"),
    ("TDz", "TDz", "tongue dorsum height"),
    ("UL-LL distance", "UL_LL_distance", "lip aperture"),
    ("F2", "F2", "formant"),
]


def utterance_features(npz_path):
    """Reproduce the build_features.py per-utterance feature frame (T, n_features)."""
    d = np.load(npz_path)
    t = d["time"].astype(np.float64)
    ema_raw = d["ema"]
    ema_clean = normalize_coords(fix_outliers(ema_raw))
    w = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
    if w >= 3:
        ema_sm = np.stack([savgol_filter(ema_clean[:, j], w, 2)
                           for j in range(ema_clean.shape[1])], axis=1)
    else:
        ema_sm = ema_clean
    ema_sm = lowpass(ema_sm)

    def P(s):
        return ema_sm[:, [EMA_COLS.index(f"{s}{a}") for a in AXES]]

    TT, TD, UL, LL = P("TT"), P("TD"), P("UL"), P("LL")
    ul_ll = np.sqrt(((UL - LL) ** 2).sum(1))

    f2 = d["f2"].astype(np.float64)
    f2[f2 <= 0] = np.nan
    good = np.where(~np.isnan(f2))[0]
    if len(good) >= 2:
        f2 = np.interp(np.arange(len(f2)), good, f2[good])
    else:
        f2 = np.nan_to_num(f2, nan=0.0)

    feat = {}
    for k, col in enumerate([f"{s}{a}" for s in KEEP_SENSORS for a in AXES]):
        feat[col] = ema_sm[:, EMA_COLS.index(col)]
    feat["UL_LL_distance"] = ul_ll
    feat["F2"] = f2
    return t, feat


def resample_zscore(x, n_frames):
    """Time-warp to [0,1] and z-score within the utterance."""
    t = np.linspace(0.0, 1.0, len(x))
    tn = np.linspace(0.0, 1.0, n_frames)
    xr = np.interp(tn, t, x)
    mu, sd = xr.mean(), xr.std()
    if sd < 1e-9:
        return np.zeros_like(xr)
    return (xr - mu) / sd


def collect_trajectories(rows, data_dir, n_frames, max_per_class):
    """Return (channels, n) where channels[c] is a (n, T) array of z-scored,
    time-warped trajectories for the class's utterances (speaker-balanced)."""
    # group utterances by subject, then round-robin to cap class size
    # without dropping any one speaker's tokens entirely.
    per_subj = {}
    for _, r in rows.iterrows():
        per_subj.setdefault(int(r["subj"]), []).append(r)
    chosen = []
    idx = {s: 0 for s in per_subj}
    while len(chosen) < max_per_class:
        added = False
        for s in sorted(per_subj):
            if len(chosen) >= max_per_class:
                break
            if idx[s] < len(per_subj[s]):
                chosen.append(per_subj[s][idx[s]])
                idx[s] += 1
                added = True
        if not added:
            break

    chan_keys = [p[1] for p in PANELS]
    chan = {c: [] for c in chan_keys}
    for r in chosen:
        npz = os.path.join(data_dir, r["path"])
        _, feat = utterance_features(npz)
        for c in chan_keys:
            chan[c].append(resample_zscore(feat[c], n_frames))
    for c in chan_keys:
        chan[c] = np.stack(chan[c])  # (n, T)
    return chan, len(chosen)


def plot_comparison(groups, data_dir, n_frames, max_per_class, title, out_path):
    """One figure with the mean (+/- 1 std band) trajectory of each class."""
    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.2 * n_panels), sharex=True)
    cmap = plt.get_cmap("tab10")

    for gi, (label, rows) in enumerate(groups.items()):
        chan, n = collect_trajectories(rows, data_dir, n_frames, max_per_class)
        tn = np.linspace(0.0, 1.0, n_frames)
        color = cmap(gi % 10)
        for ai, (_ptitle, pkey, ylab) in enumerate(PANELS):
            ax = axes[ai]
            y = chan[pkey]
            mean = y.mean(0)
            sd = y.std(0)
            ax.plot(tn, mean, color=color, lw=1.8,
                    label=f"{label} (n={n})")
            ax.fill_between(tn, mean - sd, mean + sd, color=color, alpha=0.20)
            ax.set_ylabel(ylab, fontsize=8)
        # keep one label per class, not per panel
    for ai, (ptitle, _pkey, _ylab) in enumerate(PANELS):
        axes[ai].set_title(ptitle, fontsize=9, loc="left")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("normalized time (utterance duration)")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_single(label, rows, data_dir, n_frames, max_per_class, title, out_path):
    """Isolated mean +/- 1 std band figure for one class."""
    chan, n = collect_trajectories(rows, data_dir, n_frames, max_per_class)
    tn = np.linspace(0.0, 1.0, n_frames)
    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.2 * n_panels), sharex=True)
    for ai, (ptitle, pkey, ylab) in enumerate(PANELS):
        ax = axes[ai]
        y = chan[pkey]
        mean, sd = y.mean(0), y.std(0)
        ax.plot(tn, mean, color="#C0392B", lw=2.0)
        ax.fill_between(tn, mean - sd, mean + sd, color="#C0392B", alpha=0.25)
        ax.set_title(f"{ptitle}", fontsize=9, loc="left")
        ax.set_ylabel(ylab, fontsize=8)
    axes[-1].set_xlabel("normalized time (utterance duration)")
    fig.suptitle(f"{title} — {label} (n={n})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(path("cis_dataset")))
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "paper", "illustrations", "trajectories"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-syllables", type=int, default=5)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--max-per-class", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))

    rng = np.random.default_rng(args.seed)
    syllables = sorted(manifest["syllable"].unique())
    sel_syllables = rng.choice(syllables, size=args.n_syllables, replace=False)
    sel_syllables = sorted(sel_syllables.tolist())

    vowels = sorted(manifest[manifest["syllable"].isin(sel_syllables)]["vowel"].unique())
    onsets = sorted(manifest[manifest["syllable"].isin(sel_syllables)]["onset"].unique())
    print("seed:", args.seed)
    print("selected syllables:", sel_syllables)
    print("vowels from those:", vowels)
    print("onsets from those:", onsets)

    groups = {s: manifest[manifest["syllable"] == s] for s in sel_syllables}
    plot_comparison(groups, args.data_dir, args.n_frames, args.max_per_class,
                    f"Normalized trajectories — {len(sel_syllables)} random syllables (seed {args.seed})",
                    os.path.join(args.out_dir, "syllables.png"))

    vgroups = {v: manifest[manifest["vowel"] == v] for v in vowels}
    plot_comparison(vgroups, args.data_dir, args.n_frames, args.max_per_class,
                    f"Normalized trajectories — vowels in the selected syllables (seed {args.seed})",
                    os.path.join(args.out_dir, "vowels.png"))

    ogroups = {o: manifest[manifest["onset"] == o] for o in onsets}
    plot_comparison(ogroups, args.data_dir, args.n_frames, args.max_per_class,
                    f"Normalized trajectories — onsets in the selected syllables (seed {args.seed})",
                    os.path.join(args.out_dir, "onsets.png"))

    for s in sel_syllables:
        plot_single(s, groups[s], args.data_dir, args.n_frames, args.max_per_class,
                    "Full syllable", os.path.join(args.out_dir, f"syllable_{s}.png"))
    for v in vowels:
        plot_single(v, vgroups[v], args.data_dir, args.n_frames, args.max_per_class,
                    "Vowel", os.path.join(args.out_dir, f"vowel_{v}.png"))
    for o in onsets:
        plot_single(o, ogroups[o], args.data_dir, args.n_frames, args.max_per_class,
                    "Onset", os.path.join(args.out_dir, f"onset_{o}.png"))

    print("\nwrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
