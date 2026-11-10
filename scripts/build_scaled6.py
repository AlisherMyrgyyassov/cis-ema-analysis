"""Build a 6-sensor (incl. LL) size-normalized GEOMETRIC table for the corpus
figures (frame B). Same convention as the `scaled` variant -- head-frame
coordinates divided by the per-utterance lip-aperture scale -- but LL is kept so
the montage can show all six coils at their real positions.

Writes: variants/scaled6/features.csv  (source, time, TDx..LCz incl. LL)
"""
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

DATA = str(path("cis_dataset"))
SENSORS = bf.SENSORS          # TD, TL, TT, UL, LL, LC
AXES = bf.AXES
EMA_COLS = bf.EMA_COLS


def main():
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    rows = []
    for _, r in man.iterrows():
        d = np.load(os.path.join(DATA, r["path"]))
        t = d["time"].astype(np.float64)
        e = bf.fix_outliers(d["ema"].astype(np.float64))
        ll = {a: e[:, EMA_COLS.index(f"LL{a}")] for a in AXES}
        ul = {a: e[:, EMA_COLS.index(f"UL{a}")] for a in AXES}
        scale = float(np.median(np.sqrt(sum((ul[a] - ll[a]) ** 2 for a in AXES))))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        e = e / scale
        w = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
        if w >= 3:
            e = np.stack([bf.savgol_filter(e[:, j], w, 2) for j in range(e.shape[1])], axis=1)
        e = bf.lowpass(e)
        base = {"source": r["source"], "time": t}
        for i, c in enumerate(EMA_COLS):
            base[c] = e[:, i]
        rows.append(pd.DataFrame(base))
    full = pd.concat(rows, ignore_index=True)
    out_dir = os.path.join(DATA, "variants", "scaled6")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "features.csv")
    full.to_csv(out, index=False)
    print(f"wrote {out}: {full.shape}")


if __name__ == "__main__":
    main()
