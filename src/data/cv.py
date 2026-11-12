"""Subject-wise cross-validation for the EMA syllable dataset.

Produces the sex-stratified, subject-disjoint folds used identically across all
experiments (Experiment 0.2).

**The folds are frozen data, not code.** ``folds/cis_folds.json`` holds the
subject assignment used for every number in the paper, and :func:`make_folds`
returns it verbatim when it matches the subject set it is asked about. The
generator below is kept so the file can be re-derived and audited, but the
published results never depend on it: a numpy or scikit-learn upgrade must not be
able to reshuffle the test speakers silently.
"""
import json
import os
import warnings

import numpy as np

from src.config import FOLDS_PATH


def stratify_subjects(subjects, sex_map, k, seed):
    """Split subjects into k groups balanced by sex (round-robin within sex)."""
    rng = np.random.RandomState(seed)
    groups = [[] for _ in range(k)]
    for sex in ("F", "M"):
        grp = sorted([s for s in subjects if sex_map.get(s) == sex])
        rng.shuffle(grp)
        for i, s in enumerate(grp):
            groups[i % k].append(s)
    return [sorted(g) for g in groups if g]


def compute_folds(subjects, sex_map, n_folds=6, n_val=4, seed=0):
    """Derive the folds from the seed. Used to build ``folds/cis_folds.json``."""
    test_groups = stratify_subjects(subjects, sex_map, n_folds, seed)
    folds = []
    for gi in range(n_folds):
        test = test_groups[gi]
        remaining = [s for s in subjects if s not in test]
        n_groups = max(1, (len(remaining) + n_val - 1) // n_val)
        val_groups = stratify_subjects(remaining, sex_map, n_groups, seed + 1000)
        val = []
        for g in val_groups:
            for s in g:
                if len(val) < n_val:
                    val.append(s)
        train = [s for s in remaining if s not in val]
        folds.append({"train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return folds


def load_folds(path=None):
    """Read the frozen fold assignment, or ``None`` when the file is absent."""
    path = path or FOLDS_PATH
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        payload = json.load(fh)
    folds = payload["folds"] if isinstance(payload, dict) else payload
    return [
        {"train": list(f["train"]), "val": list(f["val"]), "test": list(f["test"])}
        for f in folds
    ]


def make_folds(subjects, sex_map, n_folds=6, n_val=4, seed=0, frozen_path=None):
    """Return n_folds dicts {train, val, test} of subject id lists.

    Prefers the frozen assignment in ``folds/cis_folds.json``. Falls back to
    deriving them from ``seed`` (with a warning) only when the frozen file is
    missing or describes a different subject set, e.g. a subsampled run.
    """
    subjects = sorted(subjects)
    frozen = load_folds(frozen_path)
    if frozen is not None:
        covered = sorted({s for f in frozen for s in f["test"]})
        if covered == subjects and len(frozen) == n_folds:
            return frozen
        warnings.warn(
            "folds/cis_folds.json does not match the requested subject set "
            f"({len(covered)} frozen vs {len(subjects)} requested); deriving the "
            "folds from the seed instead. Published numbers use the frozen file.",
            RuntimeWarning,
            stacklevel=2,
        )
    return compute_folds(subjects, sex_map, n_folds=n_folds, n_val=n_val, seed=seed)


def legacy_subject_split():
    """Reproduce the original train-test-split-27-30 subject assignment."""
    train = list(range(1, 27)) + [31, 32, 33]
    return {"train": train, "val": [27, 28], "test": [29, 30]}
