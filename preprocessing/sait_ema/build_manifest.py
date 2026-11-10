"""Build manifest.csv for SAIT-EMA from extracted speakers + central TextGrids.

One row per (speaker, item): metadata, section, TextGrid word/phone labels,
interval bounds and frame counts. Run after the archives are extracted.

Usage:
  python -m preprocessing.sait_ema.build_manifest \
      --root data/SAIT-EMA \
      --out  data/SAIT-EMA/Dataset

Both default to the paths resolved by ``src.paths`` (see paths.example.toml).
"""
from __future__ import annotations

import argparse
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import get as path  # noqa: E402

import pandas as pd

from .reader import (FS, NATIVE, L2, SPEAKERS, load_speech_materials,
                     parse_textgrid, parse_label, read_pos, section_of)


def _first_word(tiers):
    """First non-empty interval of the word/syllable tier."""
    for name in ("words", "word", "syllable"):
        if name in tiers:
            for a, b, t in tiers[name]:
                if t.strip():
                    return a, b, t.strip()
    return None


def _phones(tiers, s, e):
    segs = []
    for name in ("phones", "phone", "segment"):
        if name in tiers:
            segs = [t for a, b, t in tiers[name]
                    if t.strip() and a >= s - 1e-6 and b <= e + 1e-6]
            break
    return segs


def build(root: str, out_dir: str) -> pd.DataFrame:
    mats = load_speech_materials(os.path.join(root, "speech_materials", "speech_materials.txt"))
    rows = []
    for spk in SPEAKERS:
        group, sex, l1 = SPEAKERS[spk]
        pos_dir = os.path.join(root, spk, "pos")
        wav_dir = os.path.join(root, spk, "wav")
        tg_dir = os.path.join(root, "TextGrids", spk)
        if not os.path.isdir(pos_dir):
            print(f"  !! missing {spk}/pos")
            continue
        n_ok = 0
        for item in range(1, 963):
            i4 = f"{item:04d}"
            pos_p = os.path.join(pos_dir, i4 + ".pos")
            wav_p = os.path.join(wav_dir, i4 + ".wav")
            tg_p = os.path.join(tg_dir, i4 + ".TextGrid")
            if not (os.path.isfile(pos_p) and os.path.isfile(tg_p)):
                continue
            tiers = parse_textgrid(tg_p)
            fw = _first_word(tiers)
            if fw is None:
                continue
            s, e, label = fw
            syl, tone = parse_label(label)
            mat = mats.get(item)
            # prefer TextGrid label for tone; fall back to materials
            if tone == 0 and mat is not None and mat.tone:
                tone = mat.tone
            if not syl and mat is not None:
                syl = mat.syllable
            phones = _phones(tiers, s, e)
            # frame count on the 250 Hz EMA grid
            n_frames = int(round((e - s) * FS))
            try:
                time, ema = read_pos(pos_p, spk)
                n_frames = int(((time >= s - 1e-6) & (time <= e + 1e-6)).sum())
            except Exception as ex:  # noqa
                ema = None
            rows.append({
                "source": f"{spk}_{i4}",
                "speaker": spk, "group": group, "sex": sex, "l1": l1,
                "item": item, "section": section_of(item),
                "label": label, "syllable": syl, "tone": tone,
                "pinyin": mat.pinyin if mat else "",
                "phones": "|".join(phones),
                "start": round(s, 5), "end": round(e, 5),
                "duration": round(e - s, 5), "n_frames": n_frames,
                "pos": os.path.relpath(pos_p, root).replace("\\", "/"),
                "wav": os.path.relpath(wav_p, root).replace("\\", "/"),
                "textgrid": os.path.relpath(tg_p, root).replace("\\", "/"),
            })
            n_ok += 1
        print(f"  {spk}: {n_ok} utterances")

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(path("sait_raw")))
    ap.add_argument("--out", default=str(path("sait_dataset")))
    args = ap.parse_args()
    df = build(args.root, args.out)
    print(f"\ntotal rows: {len(df)}")
    print(df.groupby(["section"]).size())
    print("\nspeakers:", df["speaker"].nunique(), "| native rows:",
          (df["group"] == "native").sum(), "| l2 rows:", (df["group"] == "l2").sum())
    if len(df):
        print("frames/utt: median %.0f min %d max %d" %
              (df["n_frames"].median(), df["n_frames"].min(), df["n_frames"].max()))


if __name__ == "__main__":
    main()
