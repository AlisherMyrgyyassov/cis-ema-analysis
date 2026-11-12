"""Ground-truth Cantonese tone labels from ``stimuli.xlsx``.

The corpus transcript (TextGrid word tiers) and the rebuilt feature table store
tone-less Jyutping syllable strings (e.g. ``saa``). The stimulus list
``stimuli.xlsx`` contains the *tone-marked* target syllable of each item
(``Transcription1``, e.g. ``saa1``) together with a ``Tone`` column (1-6).

Each utterance in the dataset is identified by its ``source`` / ``file_num``
(e.g. ``sub_10_0002`` -> item 2 -> ``saa1``, tone 1), so the item number is the
join key that recovers the lexical tone of every utterance.

Item 114 ("陶域道", transcribed ``jiu4``) is excluded because the character 陶
has an alternate tonal reading across speakers (``tou4`` vs ``jiu4``), making
its tone label unreliable.
"""
import re

import pandas as pd

from src.config import STIMULI_PATH, EXCLUDED_TONE_ITEMS, CANTONESE_TONES


def load_stimuli(path=STIMULI_PATH):
    """Load the stimulus sheet -> DataFrame (155 items, Item No. 2..156)."""
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["Item No."].notna()].copy()
    df["Item No."] = df["Item No."].astype(int)
    df["tone"] = df["Tone"].astype(int)
    df["trans1"] = df["Transcription1"].astype(str).str.strip()
    df["syllable"] = df["trans1"].str.replace(r"[0-9]", "", regex=True)
    return df


def item_to_tone(path=STIMULI_PATH):
    """Map item number -> lexical tone (1-6)."""
    df = load_stimuli(path)
    return {int(r["Item No."]): int(r["tone"]) for _, r in df.iterrows()}


def item_to_syllable(path=STIMULI_PATH):
    """Map item number -> tone-marked target syllable (e.g. 'saa1')."""
    df = load_stimuli(path)
    return {int(r["Item No."]): str(r["trans1"]) for _, r in df.iterrows()}


def file_num_from_source(source):
    """Extract the item number from a source id (e.g. 'sub_10_0002' -> 2)."""
    return int(str(source).split("_")[-1])


def tone_for_record(record, tone_map=None, exclude=EXCLUDED_TONE_ITEMS):
    """Return the lexical tone (int 1-6) for a record, or None if excluded."""
    fn = file_num_from_source(record["source"])
    if fn in exclude:
        return None
    if tone_map is None:
        tone_map = item_to_tone()
    return tone_map.get(fn)


def add_tone_to_records(records, tone_map=None, exclude=EXCLUDED_TONE_ITEMS):
    """Annotate records in place with a ``tone`` field (None for excluded)."""
    if tone_map is None:
        tone_map = item_to_tone()
    for r in records:
        r["tone"] = tone_for_record(r, tone_map, exclude)
    return records
