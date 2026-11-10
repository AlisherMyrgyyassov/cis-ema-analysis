"""Build center_only calibration variants (stats from k random utterances/speaker),
mirroring the robust_z pessimistic k-curve so the two are directly comparable."""
import importlib.util
import os
import sys

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = str(path("cis_dataset"))
KS = [1, 2, 3, 5, 10, 20]

spec = importlib.util.spec_from_file_location("bz", os.path.join(ROOT, "scripts", "build_zscore_features.py"))
bz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bz)

man = os.path.join(DATA, "manifest.csv")
for k in KS:
    out = os.path.join(DATA, "variants", f"center_only_calib{k}")
    print(f"[center_only calib_k={k}]")
    bz.build(man, DATA, out, "center_only", 1, k)
