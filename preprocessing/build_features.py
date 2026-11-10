"""Stage 2: build the feature table from the aligned per-utterance npz files.

Applies the same articulatory feature engineering as the original pipeline
(LL-reference normalization, per-utterance outlier fix, Savitzky-Golay
smoothing, velocity/acceleration, distances, angles) and appends the richer
acoustic features (F1-F3, F0, MFCC+delta+delta-delta, energy).

Output:
  features.csv   long-format table (one row per frame)
  meta.json      column groups + stats (feature->group mapping)
"""
import argparse
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

FS = 250.0  # EMA sampling rate (Hz)

SENSORS = ["TD", "TL", "TT", "UL", "LL", "LC"]
AXES = ["x", "y", "z"]
EMA_COLS = [f"{s}{a}" for s in SENSORS for a in AXES]
KEEP_SENSORS = ["TD", "TL", "TT", "UL", "LL", "LC"]  # all 6 articulatory coils
META = ["source", "subj", "sex", "time", "syllable", "vowel", "onset"]


def fix_outliers(arr, z_thresh=3.0):
    """Per-column (feature) z-score outlier fix + linear interpolation."""
    arr = arr.astype(np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mu, sd = col.mean(), col.std()
        if sd < 1e-9:
            continue
        z = np.abs((col - mu) / sd)
        bad = z > z_thresh
        if bad.any():
            good_idx = np.where(~bad)[0]
            arr[bad, j] = np.nan
            if len(good_idx) >= 2:
                arr[:, j] = np.interp(np.arange(len(col)), good_idx, arr[good_idx, j])
            else:
                arr[:, j] = np.nan_to_num(arr[:, j], nan=mu)
    return arr.astype(np.float32)


def normalize_coords(ema):
    """Subtract LL (translation) and divide by a per-utterance constant scale.

    Scale = median UL-LL distance (speaker/jaw size), constant over the utterance
    so that the time-varying lip aperture is preserved (and no near-zero blowup).
    """
    ll = {a: ema[:, EMA_COLS.index(f"LL{a}")] for a in AXES}
    ul = {a: ema[:, EMA_COLS.index(f"UL{a}")] for a in AXES}
    inst_scale = np.sqrt(sum((ul[a] - ll[a]) ** 2 for a in AXES))
    scale = float(np.median(inst_scale))
    if scale < 1e-6:
        scale = 1.0
    out = ema.copy()
    for i, c in enumerate(EMA_COLS):
        a = c[2]
        out[:, i] = (ema[:, i] - ll[a]) / scale
    return out


def compute_vel_accel(pos, dt):
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def lowpass(x, cutoff=20.0, order=4):
    """4th-order Butterworth zero-phase low-pass (articulatory bandwidth ~20 Hz)."""
    b, a = butter(order, cutoff / (0.5 * FS), btype="low")
    return filtfilt(b, a, x, axis=0)


def winsorize(x, k=6.0):
    """Clip per-column to mean +- k*std (removes residual derivative spikes)."""
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-9
    return np.clip(x, mu - k * sd, mu + k * sd)


def build_features(manifest_path, out_dir):
    manifest = pd.read_csv(manifest_path)
    meta_cols = [c for c in META if c != "source"]
    rows = []

    for _, r in manifest.iterrows():
        npz = os.path.join(out_dir, r["path"])
        d = np.load(npz)
        t = d["time"].astype(np.float64)
        ema_raw = d["ema"]  # (T, 18)

        # 1. outlier fix (raw coords) + 2. LL normalization
        ema_clean = normalize_coords(fix_outliers(ema_raw))

        # 3. smooth: Savitzky-Golay (light) then Butterworth low-pass (20 Hz)
        w = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
        if w >= 3:
            ema_sm = np.stack([savgol_filter(ema_clean[:, j], w, 2) for j in range(ema_clean.shape[1])], axis=1)
        else:
            ema_sm = ema_clean
        ema_sm = lowpass(ema_sm)

        dt = float(np.median(np.diff(t)))
        # keep only non-LL sensors (LL is reference)
        keep_idx = [EMA_COLS.index(f"{s}{a}") for s in KEEP_SENSORS for a in AXES]
        pos = ema_sm[:, keep_idx]            # (T, 15)
        vel, acc = compute_vel_accel(pos, dt)
        vel = winsorize(vel)
        acc = winsorize(acc)

        # distances + angles on normalized coords
        def P(s):
            return ema_sm[:, [EMA_COLS.index(f"{s}{a}") for a in AXES]]

        TT, TD, UL, LL, LC = P("TT"), P("TD"), P("UL"), P("LL"), P("LC")
        def dist(a, b):
            return np.sqrt(((a - b) ** 2).sum(1))
        tt_td = dist(TT, TD)
        ul_ll = dist(UL, LL)
        tt_ll = dist(TT, LL)

        def angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            n1 = np.linalg.norm(v1, axis=1) + 1e-9
            n2 = np.linalg.norm(v2, axis=1) + 1e-9
            cos = (v1 * v2).sum(1) / (n1 * n2)
            return np.degrees(np.arccos(np.clip(cos, -1, 1)))
        a1 = angle(TT, TD, LL)
        a2 = angle(UL, LL, LC)

        # acoustic: formants (0 = undefined -> interp), f0/voiced, mfcc, energy
        f1 = d["f1"].astype(np.float64)
        f2 = d["f2"].astype(np.float64)
        f3 = d["f3"].astype(np.float64)
        for f in (f1, f2, f3):
            f[f <= 0] = np.nan
        def interp(x):
            if np.isnan(x).all():
                return np.zeros_like(x)
            n = len(x)
            good = np.where(~np.isnan(x))[0]
            if len(good) < 2:
                return np.nan_to_num(x, nan=0.0)
            xp = good
            fp = x[good]
            return np.interp(np.arange(n), xp, fp)
        f1, f2, f3 = interp(f1), interp(f2), interp(f3)
        f0 = d["f0"]
        voiced = d["voiced"]
        mfcc = d["mfcc"]; mfcc_d = d["mfcc_d"]; mfcc_dd = d["mfcc_dd"]
        energy = d["energy"]; energy_d = d["energy_d"]; energy_dd = d["energy_dd"]

        # assemble frame dict
        base = {
            "subj": r["subj"], "sex": r["sex"], "time": t,
            "syllable": r["syllable"], "vowel": r["vowel"], "onset": r["onset"],
        }
        for k, col in enumerate([f"{s}{a}" for s in KEEP_SENSORS for a in AXES]):
            base[col] = pos[:, k]
        for k, col in enumerate([f"{s}{a}_velocity" for s in KEEP_SENSORS for a in AXES]):
            base[col] = vel[:, k]
        for k, col in enumerate([f"{s}{a}_acceleration" for s in KEEP_SENSORS for a in AXES]):
            base[col] = acc[:, k]
        base["TT_TD_distance"] = tt_td
        base["UL_LL_distance"] = ul_ll
        base["TT_LL_distance"] = tt_ll
        base["TT_TD_LL_angle"] = a1
        base["UL_LL_LC_angle"] = a2
        base["F1"] = f1; base["F2"] = f2; base["F3"] = f3
        base["F0"] = f0; base["voiced"] = voiced
        for k in range(13):
            base[f"MFCC_{k+1}"] = mfcc[:, k]
            base[f"MFCC_{k+1}_d"] = mfcc_d[:, k]
            base[f"MFCC_{k+1}_dd"] = mfcc_dd[:, k]
        base["energy"] = energy; base["energy_d"] = energy_d; base["energy_dd"] = energy_dd

        df = pd.DataFrame(base)
        df.insert(0, "source", r["source"])
        rows.append(df)

    full = pd.concat(rows, ignore_index=True)
    # move meta columns first
    full = full[["source", "subj", "sex", "time", "syllable", "vowel", "onset"]
                + [c for c in full.columns if c not in ["source", "subj", "sex", "time", "syllable", "vowel", "onset"]]]
    full.to_csv(os.path.join(out_dir, "features.csv"), index=False)

    # meta.json: column groups
    feat_cols = [c for c in full.columns if c not in META]
    acoustic = [c for c in feat_cols if c in ("F1", "F2", "F3", "F0", "voiced") or c.startswith("MFCC") or c.startswith("energy")]
    ema = [c for c in feat_cols if c not in acoustic]
    meta = {
        "n_utterances": int(len(manifest)),
        "n_frames": int(len(full)),
        "n_subjects": int(full["subj"].nunique()),
        "meta_columns": META,
        "ema_columns": ema,
        "acoustic_columns": acoustic,
        "all_feature_columns": feat_cols,
        "syllables": sorted(full["syllable"].unique().tolist()),
        "vowels": sorted(full["vowel"].unique().tolist()),
        "onsets": sorted(full["onset"].unique().tolist()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"features.csv: {len(full)} frames, {len(feat_cols)} features "
          f"({len(ema)} EMA, {len(acoustic)} acoustic)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(path("cis_dataset")))
    args = ap.parse_args()
    build_features(os.path.join(args.out_dir, "manifest.csv"), args.out_dir)


if __name__ == "__main__":
    main()
