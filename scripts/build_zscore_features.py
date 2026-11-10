"""Build EMA feature tables on the V2 base (absolute head-frame coords, no LL)
with a statistical normalization stage.

Modes:
  raw         head-corrected coordinates, no per-speaker normalization
  ps_z        per-speaker z-score           (x-mean_s)/std_s
  pu_z        per-utterance z-score         (x-mean_u)/std_u
  center_only per-speaker mean removal      x-mean_s
  robust_z    per-speaker robust z-score    (x-median_s)/IQR_s
  proc        per-speaker Procrustes        loc/rot/scale to grand-mean template
  proc_ps_z   Procrustes then per-speaker z-score

Positions are normalized, then velocity/acceleration are the derivatives of the
normalized positions and geometry is computed from them. Acoustic columns are
identical to the locked build.
"""
import argparse
import collections
import importlib.util
import os

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

SENSORS, AXES, EMA_COLS, KEEP = bf.SENSORS, bf.AXES, bf.EMA_COLS, bf.KEEP_SENSORS
EPS = 1e-6
MODES = ["raw", "ps_z", "pu_z", "center_only", "robust_z", "proc", "proc_ps_z"]


def procrustes_transform(template, ref):
    mu_t, mu_r = template.mean(0), ref.mean(0)
    X, Y = template - mu_t, ref - mu_r
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = (S * np.array([1, 1, d])).sum() / (X ** 2).sum()
    t = mu_r - scale * (R @ mu_t)
    return scale, R, t


def apply_sim(pts, scale, R, t):
    return scale * (pts @ R.T) + t


def smooth(ema, t):
    w = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
    if w >= 3:
        ema = np.stack([bf.savgol_filter(ema[:, j], w, 2) for j in range(ema.shape[1])], axis=1)
    return bf.lowpass(ema)


def compute_targets(d):
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
    return interp(f1), interp(f2), interp(f3)


def build(manifest_path, npz_dir, out_dir, mode, ref_subj=1, calib_k=0):
    assert mode in MODES, mode
    manifest = pd.read_csv(manifest_path)

    subj_coords = collections.defaultdict(list)
    templates_acc = collections.defaultdict(list)
    need_proc = mode in ("proc", "proc_ps_z")
    for _, r in manifest.iterrows():
        d = np.load(os.path.join(npz_dir, r["path"]))
        e = bf.fix_outliers(d["ema"].astype(np.float64))
        subj_coords[int(r["subj"])].append(e)
        if need_proc:
            templates_acc[int(r["subj"])].append(e.reshape(-1, 6, 3).mean(0))

    proc = {}
    if need_proc:
        templates = {s: np.mean(v, 0) for s, v in templates_acc.items()}
        ref = templates[ref_subj] if ref_subj in templates else np.mean(list(templates.values()), 0)
        proc = {s: procrustes_transform(templates[s], ref) for s in templates}

    stats = {}
    for s, lst in subj_coords.items():
        if calib_k and len(lst) > calib_k:
            rng = np.random.RandomState(42 + int(s))
            idx = rng.choice(len(lst), calib_k, replace=False)
            sel = [lst[i] for i in idx]
        else:
            sel = lst
        e = np.concatenate(sel, 0)
        if need_proc:
            sc, R, tt = proc[s]
            e = apply_sim(e.reshape(-1, 6, 3), sc, R, tt).reshape(e.shape[0], 18)
        mean = e.mean(0)
        std = np.sqrt(((e - mean) ** 2).mean(0)) + EPS
        if mode == "robust_z":
            med = np.median(e, 0)
            q1, q3 = np.percentile(e, [25, 75], axis=0)
            stats[s] = {"med": med, "iqr": (q3 - q1) + EPS}
        else:
            stats[s] = {"mean": mean, "std": std}

    rows = []
    for _, r in manifest.iterrows():
        d = np.load(os.path.join(npz_dir, r["path"]))
        t = d["time"].astype(np.float64)
        ema = bf.fix_outliers(d["ema"].astype(np.float64))
        ema = smooth(ema, t)
        s = int(r["subj"])
        if need_proc:
            sc, R, tt = proc[s]
            Tn = ema.shape[0]
            ema = apply_sim(ema.reshape(Tn, 6, 3), sc, R, tt).reshape(Tn, 18)
        if mode in ("ps_z", "proc_ps_z"):
            ema = (ema - stats[s]["mean"]) / stats[s]["std"]
        elif mode == "center_only":
            ema = ema - stats[s]["mean"]
        elif mode == "robust_z":
            ema = (ema - stats[s]["med"]) / stats[s]["iqr"]
        elif mode == "pu_z":
            ema = (ema - ema.mean(0)) / (ema.std(0) + EPS)

        dt = float(np.median(np.diff(t)))
        keep_idx = [EMA_COLS.index(f"{x}{a}") for x in KEEP for a in AXES]
        pos = ema[:, keep_idx]
        vel, acc = bf.compute_vel_accel(pos, dt)
        vel, acc = bf.winsorize(vel), bf.winsorize(acc)

        def P(x):
            return ema[:, [EMA_COLS.index(f"{x}{a}") for a in AXES]]
        TT, TD, UL, LL, LC = P("TT"), P("TD"), P("UL"), P("LL"), P("LC")
        dist = lambda a, b: np.sqrt(((a - b) ** 2).sum(1))
        tt_td, ul_ll, tt_ll = dist(TT, TD), dist(UL, LL), dist(TT, LL)

        def angle(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            n1 = np.linalg.norm(v1, axis=1) + 1e-9
            n2 = np.linalg.norm(v2, axis=1) + 1e-9
            return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)))
        a1, a2 = angle(TT, TD, LL), angle(UL, LL, LC)
        f1, f2, f3 = compute_targets(d)

        base = {"source": r["source"], "subj": r["subj"], "sex": r["sex"], "time": t,
                "syllable": r["syllable"], "vowel": r["vowel"], "onset": r["onset"]}
        for k, col in enumerate([f"{x}{a}" for x in KEEP for a in AXES]):
            base[col] = pos[:, k]
        for k, col in enumerate([f"{x}{a}_velocity" for x in KEEP for a in AXES]):
            base[col] = vel[:, k]
        for k, col in enumerate([f"{x}{a}_acceleration" for x in KEEP for a in AXES]):
            base[col] = acc[:, k]
        base["TT_TD_distance"] = tt_td
        base["UL_LL_distance"] = ul_ll
        base["TT_LL_distance"] = tt_ll
        base["TT_TD_LL_angle"] = a1
        base["UL_LL_LC_angle"] = a2
        base["F1"], base["F2"], base["F3"] = f1, f2, f3
        base["F0"] = d["f0"]; base["voiced"] = d["voiced"]
        for k in range(13):
            base[f"MFCC_{k+1}"] = d["mfcc"][:, k]
            base[f"MFCC_{k+1}_d"] = d["mfcc_d"][:, k]
            base[f"MFCC_{k+1}_dd"] = d["mfcc_dd"][:, k]
        base["energy"] = d["energy"]; base["energy_d"] = d["energy_d"]
        base["energy_dd"] = d["energy_dd"]
        rows.append(pd.DataFrame(base))

    full = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    full.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    print(f"[{mode}] wrote {os.path.join(out_dir,'features.csv')}: {len(full)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(path("cis_dataset")))
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--ref-subj", type=int, default=1)
    ap.add_argument("--calib-k", type=int, default=0, help="stats from k utterances/speaker (0=all)")
    args = ap.parse_args()
    suffix = f"_calib{args.calib_k}" if args.calib_k else ""
    out = os.path.join(args.dataset, "variants", args.mode + suffix)
    build(os.path.join(args.dataset, "manifest.csv"), args.dataset, out, args.mode,
          args.ref_subj, args.calib_k)


if __name__ == "__main__":
    main()
