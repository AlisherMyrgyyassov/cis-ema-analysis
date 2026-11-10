"""Stage 2 (SAIT-EMA): build the feature table from aligned per-utterance npz.

Applies the paper's corrected articulatory representation: per-speaker robust-z
of the head-corrected coordinates, (x - median_s)/IQR_s per channel, computed
from the speaker's own frames (label-free, transductive). Then Savitzky-Golay +
20 Hz Butterworth smoothing, velocity/acceleration, and geometry. Acoustic
features are carried over from the npz.

Only the monosyllable sections are written by default (mono_t1 + mono_tones:
575 items/speaker) because those are the classification/tone analyses.

Output:
  features.csv        long-format (one row per frame)
  meta.json           column groups + per-speaker normalization stats
"""
from __future__ import annotations

import argparse
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

from .reader import FS, SENSORS, AXES

EMA_COLS = [f"{s}{a}" for s in SENSORS for a in AXES]
META = ["source", "speaker", "group", "sex", "l1", "item", "section",
        "syllable", "tone", "time"]
MONO_SECTIONS = ["mono_t1", "mono_tones"]
GEOM_DIST = [("TT", "TD"), ("UL", "LL"), ("TT", "LL")]
GEOM_ANG = [("TT", "TD", "LL"), ("UL", "LL", "LJ")]


def fix_outliers(arr, z=3.0):
    out = arr.astype(np.float64).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        sd = col.std()
        if sd < 1e-9:
            continue
        bad = np.abs((col - col.mean()) / sd) > z
        if bad.any():
            good = np.where(~bad)[0]
            if len(good) >= 2:
                tmp = out[:, j].copy(); tmp[bad] = np.nan
                out[:, j] = np.interp(np.arange(len(col)), good, tmp[good])
    return out.astype(np.float32)


def speaker_stats(npz_list, chunk_frames=200000):
    """Median/IQR per channel over all of a speaker's frames (streamed)."""
    med, q1, q3 = [], [], []
    buf = []
    for path in npz_list:
        buf.append(np.load(path)["ema"])
    allf = np.concatenate(buf, axis=0).astype(np.float64)
    med = np.median(allf, axis=0)
    q1 = np.percentile(allf, 25, axis=0)
    q3 = np.percentile(allf, 75, axis=0)
    iqr = np.where((q3 - q1) < 1e-6, 1.0, q3 - q1)
    return med.astype(np.float32), iqr.astype(np.float32)


def apply_norm(ema, med, iqr, mode):
    if mode == "robust_z":
        return (ema - med) / iqr
    if mode == "ps_z":
        return (ema - med)  # caller can supply mean/std instead
    if mode == "center_only":
        return ema - med
    return ema


def lowpass(x, cutoff=20.0, order=4):
    b, a = butter(order, cutoff / (0.5 * FS), btype="low")
    return filtfilt(b, a, x, axis=0)


def winsorize(x, k=6.0):
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-9
    return np.clip(x, mu - k * sd, mu + k * sd)


def build(out_dir, sections, norm="robust_z"):
    man = pd.read_csv(os.path.join(out_dir, "manifest.csv"))
    man = man[man["section"].isin(sections)].copy()
    n0 = len(man)
    man = man[man["n_frames"] >= 5].copy()
    if len(man) < n0:
        print(f"dropped {n0 - len(man)} utterances with <5 frames")
    man = man.reset_index(drop=True)
    print(f"utterances: {len(man)}  speakers: {man['speaker'].nunique()}")

    # pass 1: per-speaker normalization stats
    stats = {}
    for spk, grp in man.groupby("speaker"):
        paths = [os.path.join(out_dir, p) for p in grp["npz"]]
        med, iqr = speaker_stats(paths)
        stats[spk] = {"median": med.tolist(), "iqr": iqr.tolist()}
        print(f"  stats {spk}: {len(paths)} utt")

    rows = []
    for _, r in man.iterrows():
        d = np.load(os.path.join(out_dir, r["npz"]))
        t = d["time"].astype(np.float64)
        ema_raw = d["ema"]  # (T,18)
        med = np.array(stats[r["speaker"]]["median"], dtype=np.float32)
        iqr = np.array(stats[r["speaker"]]["iqr"], dtype=np.float32)
        ema = fix_outliers(ema_raw)
        ema = apply_norm(ema, med, iqr, norm)
        w = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
        ema = np.stack([savgol_filter(ema[:, j], w, 2) for j in range(ema.shape[1])], axis=1) if w >= 3 else ema
        if len(t) >= 20:
            ema = lowpass(ema)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.004
        pos = ema
        vel = winsorize(np.gradient(pos, dt, axis=0))
        acc = winsorize(np.gradient(vel, dt, axis=0))

        def P(s):
            return ema[:, [EMA_COLS.index(f"{s}{a}") for a in AXES]]
        def dist(a, b):
            return np.sqrt(((a - b) ** 2).sum(1))
        def angle(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            n1 = np.linalg.norm(v1, axis=1) + 1e-9
            n2 = np.linalg.norm(v2, axis=1) + 1e-9
            return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)))

        base = {c: r[c] for c in ["speaker", "group", "sex", "l1", "item", "section", "syllable", "tone"]}
        base["time"] = t
        for k, col in enumerate(EMA_COLS):
            base[col] = pos[:, k]
        for k, col in enumerate([c + "_velocity" for c in EMA_COLS]):
            base[col] = vel[:, k]
        for k, col in enumerate([c + "_acceleration" for c in EMA_COLS]):
            base[col] = acc[:, k]
        for a, b in GEOM_DIST:
            base[f"{a}_{b}_distance"] = dist(P(a), P(b))
        for a, b, c in GEOM_ANG:
            base[f"{a}_{b}_{c}_angle"] = angle(P(a), P(b), P(c))
        # acoustic (clean formants)
        f1 = d["f1"].astype(np.float64); f2 = d["f2"].astype(np.float64); f3 = d["f3"].astype(np.float64)
        for f in (f1, f2, f3):
            f[f <= 0] = np.nan
        def interp(x):
            n = len(x); good = np.where(~np.isnan(x))[0]
            if np.isnan(x).all():
                return np.zeros(n)
            if len(good) < 2:
                return np.nan_to_num(x, nan=0.0)
            return np.interp(np.arange(n), good, x[good])
        base["F1"], base["F2"], base["F3"] = interp(f1), interp(f2), interp(f3)
        base["F0"] = d["f0"]; base["voiced"] = d["voiced"]
        for k in range(13):
            base[f"MFCC_{k+1}"] = d["mfcc"][:, k]
            base[f"MFCC_{k+1}_d"] = d["mfcc_d"][:, k]
            base[f"MFCC_{k+1}_dd"] = d["mfcc_dd"][:, k]
        base["energy"] = d["energy"]; base["energy_d"] = d["energy_d"]; base["energy_dd"] = d["energy_dd"]
        df = pd.DataFrame(base)
        df.insert(0, "source", r["source"])
        rows.append(df)

    full = pd.concat(rows, ignore_index=True)
    full = full[META + [c for c in full.columns if c not in META]]
    suffix = "" if norm == "robust_z" else f"_{norm}"
    out_csv = os.path.join(out_dir, f"features_mono{suffix}.csv")
    full.to_csv(out_csv, index=False)

    feat_cols = [c for c in full.columns if c not in META]
    acoustic = [c for c in feat_cols if c in ("F1", "F2", "F3", "F0", "voiced")
                or c.startswith("MFCC") or c.startswith("energy")]
    ema = [c for c in feat_cols if c not in acoustic]
    meta = {"norm": norm, "sections": sections, "n_utterances": int(len(man)),
            "n_frames": int(len(full)), "speakers": sorted(man["speaker"].unique().tolist()),
            "ema_columns": ema, "acoustic_columns": acoustic,
            "feature_columns": feat_cols, "speaker_stats": stats,
            "syllables": sorted({str(x) for x in full["syllable"].dropna().unique()}),
            "tones": sorted({int(x) for x in full["tone"].dropna().unique()})}
    with open(os.path.join(out_dir, f"meta_mono{suffix}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out_csv}: {len(full)} frames, {len(feat_cols)} features ({len(ema)} EMA, {len(acoustic)} acoustic)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(path("sait_dataset")))
    ap.add_argument("--norm", default="robust_z")
    ap.add_argument("--sections", default="mono_t1,mono_tones")
    args = ap.parse_args()
    build(args.out, args.sections.split(","), args.norm)


if __name__ == "__main__":
    main()
