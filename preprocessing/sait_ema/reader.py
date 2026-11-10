"""SAIT-EMA corpus reader: head-corrected .pos files, TextGrids, and metadata.

The `.pos` files are Carstens AG501 binary: a latin-1 text header whose SECOND
line gives the header byte length, followed by float32 data reshaped to
(n_frames, 16 channels x 7 values) with value order x, y, z, phi, theta, rms,
extra. Articulatory sensors are TT, TB, TD, LJ (lower jaw/incisor), UL, LL;
references are NOSE, LE, RE. Per-speaker channel maps are required because some
speakers have irregular coil wiring (see `CHANNEL_MAPS`).

Frame rate is 250 Hz (confirmed from the header: SamplingFrequencyHz=250); audio
is 48 kHz mono. Axes follow the standard AG501 frame: x = front-back,
z = height (superior-inferior), y = medial-lateral (see docs/findings/sait-ema.md).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

FS = 250.0  # EMA frame rate (Hz)
SENSORS = ["TT", "TB", "TD", "LJ", "UL", "LL"]  # articulatory coils modelled
AXES = ["x", "y", "z"]
VALUES = ["x", "y", "z", "phi", "theta", "rms", "extra"]
NCHAN = 16
NVAL = 7

# 1-based channel index of each sensor, per speaker. "normal" wiring otherwise.
NORMAL_MAP = {"NOSE": 1, "LE": 2, "RE": 3, "TD": 4, "TB": 5, "TT": 6,
              "LJ": 7, "LL": 8, "UL": 9}
CHANNEL_MAPS = {
    "L_M002": {"NOSE": 11, "LE": 12, "RE": 13, "TD": 14, "TB": 5, "TT": 6, "LJ": 7, "LL": 8, "UL": 9},
    "L_M003": {"NOSE": 11, "LE": 12, "RE": 13, "TD": 14, "TB": 5, "TT": 6, "LJ": 7, "LL": 8, "UL": 9},
    "A_F001": {"NOSE": 11, "LE": 12, "RE": 13, "TD": 14, "TB": 10, "TT": 16, "LJ": 7, "LL": 8, "UL": 9},
    "L_F003": {"NOSE": 1, "LE": 5, "RE": 3, "TD": 11, "TB": 10, "TT": 12, "LJ": 7, "LL": 8, "UL": 9},
}

# speaker -> (group, sex, L1)
SPEAKERS = {
    "F001": ("native", "F", "Chinese"), "F002": ("native", "F", "Chinese"),
    "F003": ("native", "F", "Chinese"), "F004": ("native", "F", "Chinese"),
    "F005": ("native", "F", "Chinese"), "M001": ("native", "M", "Chinese"),
    "M002": ("native", "M", "Chinese"), "M003": ("native", "M", "Chinese"),
    "M004": ("native", "M", "Chinese"), "M005": ("native", "M", "Chinese"),
    "A_F001": ("native", "F", "Chinese"), "A_M001": ("native", "M", "Chinese"),
    "L_F001": ("l2", "F", "Vietnamese"), "L_F002": ("l2", "F", "Peru"),
    "L_F003": ("l2", "F", "Russian"), "L_M001": ("l2", "M", "Vietnamese"),
    "L_M002": ("l2", "M", "Russian"), "L_M003": ("l2", "M", "Mexico"),
}
NATIVE = [s for s, v in SPEAKERS.items() if v[0] == "native"]
L2 = [s for s, v in SPEAKERS.items() if v[0] == "l2"]

# stimulus sections by item number (from speech_materials.txt)
SECTIONS = [(1, 407, "mono_t1"), (408, 691, "disyllable"),
            (692, 794, "sentence"), (795, 962, "mono_tones")]


def section_of(item: int) -> str:
    for lo, hi, name in SECTIONS:
        if lo <= item <= hi:
            return name
    return "unknown"


def channel_map(speaker: str) -> dict:
    return CHANNEL_MAPS.get(speaker, NORMAL_MAP)


def read_pos(path: str, speaker: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_s (F,), ema (F, 18)) with columns sensor-major [x,y,z]."""
    cmap = channel_map(speaker)
    with open(path, "rb") as f:
        f.readline()
        header_size = int(f.readline().strip())
    raw = np.fromfile(path, np.float32, offset=header_size)
    data = raw.reshape(-1, NCHAN * NVAL)
    cols = []
    for sensor in SENSORS:
        base = (cmap[sensor] - 1) * NVAL
        for v in AXES:
            cols.append(data[:, base + VALUES.index(v)])
    ema = np.stack(cols, axis=1).astype(np.float32)
    time = (np.arange(data.shape[0], dtype=np.float64) / FS)
    return time, ema


def parse_textgrid(path: str) -> dict:
    """Parse a Praat long TextGrid -> {tier_name: [(xmin, xmax, text), ...]}."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    tiers = {}
    for block in re.split(r"item \[\d+\]:", text):
        m = re.search(r'name = "([^"]+)"', block)
        if not m:
            continue
        intervals = re.findall(
            r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', block)
        tiers[m.group(1)] = [(float(a), float(b), t) for a, b, t in intervals]
    return tiers


# --- pinyin tone parsing (for speech_materials; TextGrid labels already carry tone) ---
_TONE_MARKS = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("ü", 1), "ǘ": ("ü", 2), "ǚ": ("ü", 3), "ǜ": ("ü", 4), "ü": ("ü", 0),
}


def parse_pinyin(py: str) -> tuple[str, int]:
    """Return (tone-less syllable, tone). tone 0 = unmarked/neutral."""
    py = py.strip().lower()
    tone, out = 0, []
    for ch in py:
        if ch in _TONE_MARKS:
            base, t = _TONE_MARKS[ch]
            out.append(base)
            if t:
                tone = t
        else:
            out.append(ch)
    s = "".join(out)
    m = re.search(r"([1-5])$", s)
    if m:
        tone = int(m.group(1))
        s = s[:m.start()]
    return s, tone


def parse_label(label: str) -> tuple[str, int]:
    """Parse a TextGrid word label such as 'che1' -> ('che', 1)."""
    return parse_pinyin(label)


# Standard Mandarin pinyin initials (longest first for greedy match)
_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
             "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]


def split_syllable(syl: str) -> tuple[str, str]:
    """Split a tone-less pinyin syllable into (initial, final)."""
    if not isinstance(syl, str):
        return "", ""
    for ini in _INITIALS:
        if syl.startswith(ini):
            return ini, syl[len(ini):]
    return "", syl


@dataclass
class Material:
    item: int
    section: str
    pinyin: str
    syllable: str
    tone: int


def load_speech_materials(path: str) -> dict[int, Material]:
    mats = {}
    with open(path, encoding="utf-16") as f:
        for line in f:
            m = re.match(r"^(\d+)\s+(.*)$", line.rstrip())
            if not m:
                continue
            item = int(m.group(1))
            toks = m.group(2).strip().split()
            py = toks[-1] if toks else ""
            syl, tone = parse_pinyin(py)
            mats[item] = Material(item, section_of(item), py, syl, tone)
    return mats
