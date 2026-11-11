"""Stage 1: extract and align the CIS-EMA corpus.

Reads raw EMA (posR/*.txt, 250 Hz head-rotated), per-utterance audio (wav/*.wav,
48 kHz) and TextGrid labels (TG/*.TextGrid), and produces a clean, fully-aligned
per-utterance archive on a single 250 Hz (0.004 s) grid.

Acoustic features are extracted at the exact EMA frame times:
  - F1-F3 (Praat Burg via parselmouth, window 0.025 s, pre-emph 50, ceiling 5000M/5500F)
  - F0  (Praat autocorrelation, floor 75 / ceiling 600 Hz) + voiced flag
  - 13 MFCCs + 13 deltas + 13 delta-deltas (librosa, 0.025 s window / 0.004 s hop)
  - energy (RMS) + delta + delta-delta

Output (--out-dir):
  utterances/sub_XX/YYYY.npz   (keys: time, ema(18), f1,f2,f3, f0, voiced,
                                mfcc(13), mfcc_d(13), mfcc_dd(13), energy, energy_d, energy_dd)
  manifest.csv                 (source, subj, sex, file_num, syllable, vowel, onset,
                                n_frames, duration, path)
"""
import argparse
import os
import re

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import numpy as np
import pandas as pd
import parselmouth
import librosa

SENSORS = ["TD", "TL", "TT", "UL", "LL", "LC"]
AXES = ["x", "y", "z"]
EMA_COLS = [f"{s}{a}" for s in SENSORS for a in AXES]
SR = 48000
HOP = 0.004
N_MFCC = 13
WIN = 0.025


def parse_textgrid(path):
    """Parse a Praat long-format TextGrid -> dict tier_name -> list[(xmin, xmax, text)]."""
    tiers = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    # split into tier blocks
    tier_blocks = re.split(r'item \[\d+\]:', text)
    for block in tier_blocks:
        m = re.search(r'name = "([^"]+)"', block)
        if not m:
            continue
        name = m.group(1)
        intervals = re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', block)
        tiers[name] = [(float(a), float(b), t) for a, b, t in intervals]
    return tiers


def read_posr(path):
    df = pd.read_csv(path, sep="\t")
    t = df["time"].to_numpy(dtype=np.float64)
    ema = df[EMA_COLS].to_numpy(dtype=np.float32)
    return t, ema


def extract_acoustic(wav_path, ema_times, ceiling, offset=0.0):
    """Extract acoustics at the EMA frame times.

    offset: seconds to subtract from ema_times to get times in the wav file's
    own clock (0.004 for edge-trimmed files, 0 for untrimmed files).
    """
    q_times = ema_times - offset
    sound = parselmouth.Sound(wav_path)

    formant = sound.to_formant_burg(
        time_step=HOP, max_number_of_formants=5, maximum_formant=ceiling,
        window_length=WIN, pre_emphasis_from=50.0)
    pitch = sound.to_pitch(time_step=HOP, pitch_floor=75.0, pitch_ceiling=600.0)

    f1 = np.array([formant.get_value_at_time(1, t) for t in q_times], dtype=np.float32)
    f2 = np.array([formant.get_value_at_time(2, t) for t in q_times], dtype=np.float32)
    f3 = np.array([formant.get_value_at_time(3, t) for t in q_times], dtype=np.float32)
    f0 = np.array([pitch.get_value_at_time(t) for t in q_times], dtype=np.float32)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0 = np.where(f0 < 0, 0.0, f0)
    voiced = (f0 > 0.0).astype(np.float32)

    # MFCC + energy from raw samples (48 kHz), hop 0.004 s
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    hop = int(SR * HOP)          # 192
    n_fft = int(SR * WIN)        # 1200
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop,
                                center=True).astype(np.float32)  # (13, F)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True).astype(np.float32)[0]  # (F,)
    frame_times = (np.arange(mfcc.shape[1]) * hop + n_fft / 2) / SR

    def interp1d_cols(mat, from_t, to_t):
        # mat: (n_features, n_frames) -> (n_to, n_features)
        out = np.stack([np.interp(to_t, from_t, mat[i]) for i in range(mat.shape[0])], axis=1)
        return out.astype(np.float32)

    mfcc_al = interp1d_cols(mfcc, frame_times, q_times)              # (T, 13)
    energy_al = np.interp(q_times, frame_times, rms).astype(np.float32)  # (T,)
    return f1, f2, f3, f0, voiced, mfcc_al, energy_al


def process_subject(raw_dir, subj_folder, out_dir):
    subject_id = int(re.match(r'sub_(\d+)_', subj_folder).group(1))
    sex = subj_folder.split("_")[-1]
    ceiling = 5000 if sex == "M" else 5500

    base = os.path.join(raw_dir, subj_folder)

    def file_map(folder, ext):
        p = os.path.join(base, folder)
        if not os.path.isdir(p):
            return {}
        return {os.path.splitext(f)[0]: f for f in os.listdir(p) if f.endswith(ext)}

    posr_files = file_map("posR", ".txt")
    wav_untrim = file_map("wav", ".wav")
    tg_untrim = file_map("TG", ".TextGrid")
    wav_trim = file_map("wav_edgeTrimmed", ".wav")
    tg_trim = file_map("TG_edgeTrimmed", ".TextGrid")

    out_sub = os.path.join(out_dir, "utterances", f"sub_{subject_id:02d}")
    os.makedirs(out_sub, exist_ok=True)

    rows = []
    for num in sorted(posr_files.keys(), key=int):
        posr_path = os.path.join(base, "posR", posr_files[num])
        # choose audio + textgrid independently, with per-file clock offset
        if num in wav_untrim:
            wav_path, wav_offset = os.path.join(base, "wav", wav_untrim[num]), 0.0
        elif num in wav_trim:
            wav_path, wav_offset = os.path.join(base, "wav_edgeTrimmed", wav_trim[num]), 0.004
        else:
            continue
        if num in tg_untrim:
            tg_path, tg_offset = os.path.join(base, "TG", tg_untrim[num]), 0.0
        elif num in tg_trim:
            tg_path, tg_offset = os.path.join(base, "TG_edgeTrimmed", tg_trim[num]), 0.004
        else:
            continue

        tiers = parse_textgrid(tg_path)
        seg_tier = tiers.get("segment", [])
        word_tier = None
        for name in ("word", "syllable", "sound", "sonud"):
            if name in tiers:
                word_tier = tiers[name]
                break
        if word_tier is None:
            for name, ints in tiers.items():
                if name != "segment":
                    word_tier = ints
                    break
        if not word_tier:
            continue

        word_ints = [(a, b, t) for a, b, t in word_tier if t.strip()]
        if not word_ints:
            continue
        s, e, syllable = word_ints[0]
        segs = [t for a, b, t in seg_tier if t.strip() and a >= s - 1e-6 and b <= e + 1e-6]
        onset = segs[0] if segs else ""
        vowel = "".join(segs[1:]) if len(segs) > 1 else ""
        if not vowel and onset and syllable.startswith(onset):
            vowel = syllable[len(onset):]

        # EMA times = textgrid times + tg_offset (EMA clock == untrimmed clock)
        t_ema, ema = read_posr(posr_path)
        mask = (t_ema >= s + tg_offset - 1e-6) & (t_ema <= e + tg_offset + 1e-6)
        if mask.sum() < 2:
            continue
        ema_times = t_ema[mask]
        ema_sel = ema[mask]

        f1, f2, f3, f0, voiced, mfcc, energy = extract_acoustic(wav_path, ema_times, ceiling, offset=wav_offset)
        mfcc_d = np.gradient(mfcc, axis=0) / HOP
        mfcc_dd = np.gradient(mfcc_d, axis=0) / HOP
        energy_d = np.gradient(energy) / HOP
        energy_dd = np.gradient(energy_d) / HOP

        source = f"sub_{subject_id:02d}_{num}"
        npz_path = os.path.join(out_sub, f"{num}.npz")
        np.savez_compressed(
            npz_path, time=ema_times, ema=ema_sel.astype(np.float32),
            f1=f1, f2=f2, f3=f3, f0=f0, voiced=voiced,
            mfcc=mfcc.astype(np.float32), mfcc_d=mfcc_d.astype(np.float32), mfcc_dd=mfcc_dd.astype(np.float32),
            energy=energy, energy_d=energy_d, energy_dd=energy_dd)

        rows.append({
            "source": source, "subj": subject_id, "sex": sex, "file_num": num,
            "syllable": syllable, "vowel": vowel, "onset": onset,
            "n_frames": int(mask.sum()), "duration": round(float(e - s), 5),
            "path": os.path.relpath(npz_path, out_dir),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(path("cis_raw")))
    ap.add_argument("--out-dir", default=str(path("cis_dataset")))
    ap.add_argument("--subjects", default=None, help="comma-separated subject folders (default: all)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "utterances"), exist_ok=True)

    folders = sorted([d for d in os.listdir(args.raw_dir)
                      if os.path.isdir(os.path.join(args.raw_dir, d)) and d.startswith("sub_")])
    if args.subjects:
        sel = set(args.subjects.split(","))
        folders = [d for d in folders if d in sel]

    all_rows = []
    for fld in folders:
        print(f"processing {fld} ...")
        rows = process_subject(args.raw_dir, fld, args.out_dir)
        all_rows.extend(rows)
        print(f"  {len(rows)} utterances")

    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"\ntotal utterances: {len(manifest)}")
    print(f"syllables: {manifest['syllable'].nunique()}, vowels: {manifest['vowel'].nunique()}, "
          f"onsets: {manifest['onset'].nunique()}")
    print(f"frames/utterance: median {manifest['n_frames'].median():.0f}, "
          f"mean {manifest['n_frames'].mean():.1f}, min {manifest['n_frames'].min()}, max {manifest['n_frames'].max()}")


if __name__ == "__main__":
    main()
