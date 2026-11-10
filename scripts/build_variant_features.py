"""Build EMA feature tables under alternative normalization schemes.

Reads the aligned per-utterance npz files (which contain the raw head-frame
`ema` (T,18) plus acoustic targets) and writes a full feature table for a chosen
normalization, so recoverability/classification can be compared against the
locked `features.csv` (which uses the LL-reference scheme).

Modes:
  ll            (V0, current): (coil - LL) / scale        [scale=median UL-LL]
  scaled        (V2):          coil / scale
  raw           (V1):          coil
  centroid      (V5):          (coil - centroid) / scale  [centroid over 6 sensors]

Distances/angles (G4) are rigid invariants and are computed from the chosen frame.
Acoustic columns are identical to the locked build.
"""
import argparse
import importlib.util
import os
import sys

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "build_features", os.path.join(HERE, "..", "preprocessing", "build_features.py"))
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

SENSORS = bf.SENSORS
AXES = bf.AXES
EMA_COLS = bf.EMA_COLS
KEEP = bf.KEEP_SENSORS
META = bf.META


def transform(ema_raw, mode):
    """Return (T,18) normalized coordinates."""
    xidx = [EMA_COLS.index(f"{s}x") for s in SENSORS]
    yidx = [EMA_COLS.index(f"{s}y") for s in SENSORS]
    zidx = [EMA_COLS.index(f"{s}z") for s in SENSORS]
    ll = {a: ema_raw[:, EMA_COLS.index(f"LL{a}")] for a in AXES}
    ul = {a: ema_raw[:, EMA_COLS.index(f"UL{a}")] for a in AXES}
    inst_scale = np.sqrt(sum((ul[a] - ll[a]) ** 2 for a in AXES))
    scale = float(np.median(inst_scale))
    if scale < 1e-6:
        scale = 1.0
    out = ema_raw.copy().astype(np.float64)
    if mode == "ll":
        for i, c in enumerate(EMA_COLS):
            out[:, i] = (ema_raw[:, i] - ll[c[2]]) / scale
    elif mode == "scaled":
        out = ema_raw / scale
    elif mode == "raw":
        out = ema_raw.astype(np.float64)
    elif mode == "centroid":
        cen = {"x": ema_raw[:, xidx].mean(1), "y": ema_raw[:, yidx].mean(1),
               "z": ema_raw[:, zidx].mean(1)}
        for i, c in enumerate(EMA_COLS):
            out[:, i] = (ema_raw[:, i] - cen[c[2]]) / scale
    else:
        raise ValueError(mode)
    return out


def build(manifest_path, npz_dir, out_dir, mode):
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, r in manifest.iterrows():
        d = np.load(os.path.join(npz_dir, r["path"]))
        t = d["time"].astype(np.float64)
        ema_raw = d["ema"].astype(np.float64)
        ema_clean = transform(bf.fix_outliers(ema_raw), mode)
        w = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
        if w >= 3:
            ema_sm = np.stack([bf.savgol_filter(ema_clean[:, j], w, 2)
                               for j in range(ema_clean.shape[1])], axis=1)
        else:
            ema_sm = ema_clean
        ema_sm = bf.lowpass(ema_sm)
        dt = float(np.median(np.diff(t)))
        keep_idx = [EMA_COLS.index(f"{s}{a}") for s in KEEP for a in AXES]
        pos = ema_sm[:, keep_idx]
        vel, acc = bf.compute_vel_accel(pos, dt)
        vel = bf.winsorize(vel); acc = bf.winsorize(acc)

        def P(s):
            return ema_sm[:, [EMA_COLS.index(f"{s}{a}") for a in AXES]]
        TT, TD, UL, LL, LC = P("TT"), P("TD"), P("UL"), P("LL"), P("LC")
        dist = lambda a, b: np.sqrt(((a - b) ** 2).sum(1))
        tt_td, ul_ll, tt_ll = dist(TT, TD), dist(UL, LL), dist(TT, LL)

        def angle(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            n1 = np.linalg.norm(v1, axis=1) + 1e-9
            n2 = np.linalg.norm(v2, axis=1) + 1e-9
            return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)))
        a1, a2 = angle(TT, TD, LL), angle(UL, LL, LC)

        f1, f2, f3 = d["f1"].astype(float), d["f2"].astype(float), d["f3"].astype(float)
        for f in (f1, f2, f3):
            f[f <= 0] = np.nan

        def interp(x):
            if np.isnan(x).all():
                return np.zeros_like(x)
            good = np.where(~np.isnan(x))[0]
            if len(good) < 2:
                return np.nan_to_num(x, nan=0.0)
            return np.interp(np.arange(len(x)), good, x[good])
        f1, f2, f3 = interp(f1), interp(f2), interp(f3)

        base = {"source": r["source"], "subj": r["subj"], "sex": r["sex"], "time": t,
                "syllable": r["syllable"], "vowel": r["vowel"], "onset": r["onset"]}
        for k, col in enumerate([f"{s}{a}" for s in KEEP for a in AXES]):
            base[col] = pos[:, k]
        for k, col in enumerate([f"{s}{a}_velocity" for s in KEEP for a in AXES]):
            base[col] = vel[:, k]
        for k, col in enumerate([f"{s}{a}_acceleration" for s in KEEP for a in AXES]):
            base[col] = acc[:, k]
        base["TT_TD_distance"] = tt_td
        base["UL_LL_distance"] = ul_ll
        base["TT_LL_distance"] = tt_ll
        base["TT_TD_LL_angle"] = a1
        base["UL_LL_LC_angle"] = a2
        base["F1"], base["F2"], base["F3"] = f1, f2, f3
        base["F0"] = d["f0"]; base["voiced"] = d["voiced"]
        mfcc, mfcc_d, mfcc_dd = d["mfcc"], d["mfcc_d"], d["mfcc_dd"]
        for k in range(13):
            base[f"MFCC_{k+1}"] = mfcc[:, k]
            base[f"MFCC_{k+1}_d"] = mfcc_d[:, k]
            base[f"MFCC_{k+1}_dd"] = mfcc_dd[:, k]
        base["energy"] = d["energy"]; base["energy_d"] = d["energy_d"]
        base["energy_dd"] = d["energy_dd"]
        rows.append(pd.DataFrame(base))

    full = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "features.csv")
    full.to_csv(out_csv, index=False)
    print(f"[{mode}] wrote {out_csv}: {len(full)} frames, {full.shape[1]} cols")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(path("cis_dataset")))
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.dataset, "variants", args.mode)
    build(os.path.join(args.dataset, "manifest.csv"), args.dataset, out, args.mode)


if __name__ == "__main__":
    main()
