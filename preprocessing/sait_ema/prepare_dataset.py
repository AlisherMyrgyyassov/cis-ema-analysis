"""Stage 1 (SAIT-EMA): align EMA + acoustics at the 250 Hz frame grid.

For every (speaker, item) in manifest.csv, select the EMA frames inside the
TextGrid word interval and extract acoustic features at those exact frame times
(same recipe as the Bruce pipeline): F1-F3 (Praat Burg), F0 + voiced, 13 MFCC +
delta + delta-delta (librosa), energy + delta + delta-delta. EMA frame 0 and wav
sample 0 share t=0 (verified: frames/duration = 250.0).

Output: utterances/<speaker>/<item:04d>.npz with keys
  time, ema(18: TT,TB,TD,LJ,UL,LL x xyz), f1,f2,f3, f0, voiced,
  mfcc(13), mfcc_d(13), mfcc_dd(13), energy, energy_d, energy_dd

Usage:
  python -m preprocessing.sait_ema.prepare_dataset --workers 8
"""
from __future__ import annotations

import argparse
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SR = 48000
HOP = 0.004
WIN = 0.025
N_MFCC = 13


def extract_acoustic(wav_path, ema_times, ceiling):
    import parselmouth
    import librosa

    sound = parselmouth.Sound(wav_path)
    formant = sound.to_formant_burg(time_step=HOP, max_number_of_formants=5,
                                    maximum_formant=ceiling, window_length=WIN,
                                    pre_emphasis_from=50.0)
    pitch = sound.to_pitch(time_step=HOP, pitch_floor=75.0, pitch_ceiling=600.0)

    f1 = np.array([formant.get_value_at_time(1, t) for t in ema_times], dtype=np.float32)
    f2 = np.array([formant.get_value_at_time(2, t) for t in ema_times], dtype=np.float32)
    f3 = np.array([formant.get_value_at_time(3, t) for t in ema_times], dtype=np.float32)
    f0 = np.array([pitch.get_value_at_time(t) for t in ema_times], dtype=np.float32)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0 = np.where(f0 < 0, 0.0, f0)
    voiced = (f0 > 0.0).astype(np.float32)

    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    hop = int(SR * HOP)
    n_fft = int(SR * WIN)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=n_fft,
                                hop_length=hop, center=True).astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop,
                              center=True).astype(np.float32)[0]
    frame_times = (np.arange(mfcc.shape[1]) * hop + n_fft / 2) / SR

    mfcc_al = np.stack([np.interp(ema_times, frame_times, mfcc[i])
                        for i in range(mfcc.shape[0])], axis=1).astype(np.float32)
    energy = np.interp(ema_times, frame_times, rms).astype(np.float32)
    return f1, f2, f3, f0, voiced, mfcc_al, energy


def process_one(task):
    from .reader import read_pos, parse_textgrid
    root, out_dir, row = task
    pos_p = os.path.join(root, row["pos"])
    wav_p = os.path.join(root, row["wav"])
    tg_p = os.path.join(root, row["textgrid"])
    if not (os.path.isfile(pos_p) and os.path.isfile(wav_p) and os.path.isfile(tg_p)):
        return None
    s, e = row["start"], row["end"]
    time, ema = read_pos(pos_p, row["speaker"])
    mask = (time >= s - 1e-6) & (time <= e + 1e-6)
    if mask.sum() < 2:
        return None
    ema_times = time[mask]
    ema_sel = ema[mask]
    ceiling = 5000 if row["sex"] == "M" else 5500
    try:
        f1, f2, f3, f0, voiced, mfcc, energy = extract_acoustic(wav_p, ema_times, ceiling)
    except Exception as ex:  # noqa
        print(f"  acoustic fail {row['source']}: {ex}")
        return None
    mfcc_d = np.gradient(mfcc, axis=0) / HOP
    mfcc_dd = np.gradient(mfcc_d, axis=0) / HOP
    energy_d = np.gradient(energy) / HOP
    energy_dd = np.gradient(energy_d) / HOP

    out_sub = os.path.join(out_dir, "utterances", row["speaker"])
    os.makedirs(out_sub, exist_ok=True)
    npz_path = os.path.join(out_sub, f"{int(row['item']):04d}.npz")
    np.savez_compressed(
        npz_path, time=ema_times.astype(np.float64), ema=ema_sel,
        f1=f1, f2=f2, f3=f3, f0=f0, voiced=voiced,
        mfcc=mfcc, mfcc_d=mfcc_d.astype(np.float32), mfcc_dd=mfcc_dd.astype(np.float32),
        energy=energy, energy_d=energy_d.astype(np.float32), energy_dd=energy_dd.astype(np.float32))
    return os.path.relpath(npz_path, out_dir).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(path("sait_raw")))
    ap.add_argument("--out", default=str(path("sait_dataset")))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="limit rows (debug)")
    args = ap.parse_args()

    manifest = pd.read_csv(os.path.join(args.out, "manifest.csv"))
    if args.limit:
        manifest = manifest.head(args.limit)
    os.makedirs(os.path.join(args.out, "utterances"), exist_ok=True)

    tasks = [(args.root, args.out, r) for _, r in manifest.iterrows()]
    paths = [None] * len(tasks)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                paths[i] = fut.result()
            except Exception as ex:  # noqa
                print("worker error:", ex)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)}")
    manifest["npz"] = paths
    if not args.limit:
        manifest.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    ok = sum(p is not None for p in paths)
    print(f"done: {ok}/{len(tasks)} utterances written")


if __name__ == "__main__":
    main()
