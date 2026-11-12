"""Dataset loading and feature-group slicing for the EMA-SSI paper pipeline.

Loads the rebuilt, aligned ``features.csv`` (Experiment 0.1) and exposes
per-utterance frames sliced by the G1-G8 feature-group taxonomy defined in
``src.config``. All feature selection is done *by column name*, so the group
definitions are the single source of truth and are reused identically by every
experiment.
"""
import os

import numpy as np
import pandas as pd

from src.config import ALL_COLUMNS, columns_for

META_COLS = ["source", "subj", "sex", "time", "syllable", "vowel", "onset"]


def load_dataset(dataset_dir):
    """Load features.csv and group into per-utterance records.

    Returns:
        records: list of dicts {source, subj, sex, syllable, vowel, onset, frame}
                 where ``frame`` is a (T, n_features) float32 array in the CSV's
                 own feature-column order.
        col_index: dict mapping feature name -> position within ``frame``.
    """
    df = pd.read_csv(os.path.join(dataset_dir, "features.csv"))
    feat_cols = [c for c in df.columns if c not in META_COLS]
    col_index = {c: i for i, c in enumerate(feat_cols)}

    missing = [c for c in ALL_COLUMNS if c not in col_index]
    if missing:
        raise KeyError(f"features.csv missing expected columns: {missing}")

    records = []
    for _, g in df.groupby("source", sort=False):
        records.append({
            "source": g["source"].iloc[0],
            "subj": int(g["subj"].iloc[0]),
            "sex": g["sex"].iloc[0],
            "syllable": g["syllable"].iloc[0],
            "vowel": g["vowel"].iloc[0],
            "onset": g["onset"].iloc[0],
            "frame": g[feat_cols].to_numpy(np.float32),
        })
    return records, col_index


def select(record, col_index, columns):
    """Return record['frame'] restricted to ``columns`` (in the given order)."""
    idx = [col_index[c] for c in columns]
    return record["frame"][:, idx]


def view(records, col_index, groups):
    """Per-utterance arrays for a subset of feature groups (config order)."""
    cols = columns_for(groups)
    return [select(r, col_index, cols) for r in records]


def records_for(records, subjects):
    subj_set = set(subjects)
    return [r for r in records if r["subj"] in subj_set]


def fit_stats(seqs):
    a = np.concatenate(seqs, axis=0)
    return a.mean(0, keepdims=True), a.std(0, keepdims=True) + 1e-6


def standardize(seqs, mean, std):
    return [(s - mean) / std for s in seqs]


def pooled(seqs):
    """Per-utterance mean/std/min/max pooling -> (n, 4 * n_features)."""
    feats = []
    for s in seqs:
        feats.append(np.concatenate([s.mean(0), s.std(0), s.min(0), s.max(0)]))
    return np.stack(feats)
