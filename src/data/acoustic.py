"""Raw-audio resolution and 16 kHz caching for the SSL audio ceiling (Exp 1.5).

Audio is not stored in the processed npz archive. Crops are taken from the raw
48 kHz wavs (same untrim / wav_edgeTrimmed + 0.004 s clock-offset rule used by
``preprocessing/prepare_dataset.py``), resampled to 16 kHz, and cached under
``{data_dir}/audio_16k/{source}.npy``.
"""
import os
import re

import numpy as np
import pandas as pd

TARGET_SR = 16000
TRIM_WAV_DIRS = ("wav_edgeTrimmed", "wav_edge_trimmed", "wav_edgeTrimmed")


def index_subject_folders(raw_dir):
    """Map subject id (int) -> folder name under raw_dir (sub_XX_M / sub_XX_F)."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw-dir not found: {raw_dir}")
    mapping = {}
    for name in os.listdir(raw_dir):
        m = re.match(r"sub_(\d+)_", name)
        path = os.path.join(raw_dir, name)
        if m and os.path.isdir(path):
            mapping[int(m.group(1))] = name
    return mapping


def file_map(folder, ext):
    if not os.path.isdir(folder):
        return {}
    return {os.path.splitext(f)[0]: f for f in os.listdir(folder) if f.endswith(ext)}


def lookup_stem(fmap, stem):
    if stem in fmap:
        return fmap[stem]
    try:
        n = int(stem)
    except ValueError:
        return None
    for k, v in fmap.items():
        try:
            if int(k) == n:
                return v
        except ValueError:
            continue
    return None


def resolve_wav(raw_dir, subj_folder, stem):
    """Return (wav_path, offset_s) using the prepare_dataset.py clock rule."""
    base = os.path.join(raw_dir, subj_folder)
    wav_untrim = file_map(os.path.join(base, "wav"), ".wav")
    name = lookup_stem(wav_untrim, stem)
    if name is not None:
        return os.path.join(base, "wav", name), 0.0
    for trim_name in TRIM_WAV_DIRS:
        wav_trim = file_map(os.path.join(base, trim_name), ".wav")
        name = lookup_stem(wav_trim, stem)
        if name is not None:
            return os.path.join(base, trim_name, name), 0.004
    return None, None


def stem_from_source(source):
    return str(source).split("_")[-1]


def crop_and_resample(wav_path, t0, t1, offset, target_sr=TARGET_SR):
    import librosa
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    t0w = max(0.0, float(t0) - offset)
    t1w = max(t0w, float(t1) - offset)
    start = int(round(t0w * sr))
    end = int(round(t1w * sr))
    start = max(0, min(start, len(y)))
    end = max(start, min(end, len(y)))
    if end - start < int(0.02 * sr):
        end = min(len(y), start + max(int(0.05 * sr), 1))
    crop = y[start:end].astype(np.float32)
    if sr != target_sr:
        crop = librosa.resample(crop, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    min_samples = int(0.25 * target_sr)  # >10 encoder frames after /320 downsample
    if crop.shape[0] < min_samples:
        crop = np.pad(crop, (0, min_samples - crop.shape[0]))
    return crop


def cache_audio(records, data_dir, raw_dir):
    """Return {source: 16 kHz float32 ndarray}; writes missing npy files."""
    cache_dir = os.path.join(data_dir, "audio_16k")
    os.makedirs(cache_dir, exist_ok=True)

    manifest_path = os.path.join(data_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found in {data_dir}")
    manifest = pd.read_csv(manifest_path)
    man_by_source = {str(r["source"]): r for _, r in manifest.iterrows()}

    subj_folders = index_subject_folders(raw_dir)
    audio = {}
    n_cached, n_built, n_fail = 0, 0, 0
    for rec in records:
        source = str(rec["source"])
        npy_path = os.path.join(cache_dir, f"{source}.npy")
        if os.path.isfile(npy_path):
            audio[source] = np.load(npy_path)
            n_cached += 1
            continue

        row = man_by_source.get(source)
        if row is None:
            n_fail += 1
            continue
        subj = int(row["subj"])
        folder = subj_folders.get(subj)
        if folder is None:
            n_fail += 1
            continue
        stem = stem_from_source(source)
        wav_path, offset = resolve_wav(raw_dir, folder, stem)
        if wav_path is None:
            n_fail += 1
            continue

        npz_path = os.path.join(data_dir, row["path"])
        if not os.path.isfile(npz_path):
            n_fail += 1
            continue
        z = np.load(npz_path)
        time_key = next((k for k in z.files if k.lower() == "time"), None)
        if time_key is None:
            n_fail += 1
            continue
        t = z[time_key]
        wav = crop_and_resample(wav_path, float(t[0]), float(t[-1]), offset)
        np.save(npy_path, wav)
        audio[source] = wav
        n_built += 1

    print(f"audio cache: {n_cached} loaded, {n_built} built, {n_fail} missing "
          f"({len(audio)}/{len(records)} utterances)")
    if not audio:
        raise RuntimeError("no audio crops could be built; check raw_dir")
    return audio
