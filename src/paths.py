"""Single source of truth for every location this code reads or writes.

Nothing in this repository hard-codes an absolute path. Resolution order, per key:

1. an environment variable (see the table below),
2. an entry in ``paths.toml`` at the repository root (copy ``paths.example.toml``),
3. a sensible default relative to the repository.

===========================  ==========================  ==============================
key                          environment variable        default
===========================  ==========================  ==============================
``cis_dataset``              ``CIS_EMA_DATASET``         ``data/CIS-EMA``
``cis_raw``                  ``CIS_EMA_RAW``             ``data/raw/CIS-EMA``
``cis_stimuli``              ``CIS_EMA_STIMULI``         ``data/raw/CIS-EMA/stimuli.xlsx``
``sait_dataset``             ``SAIT_EMA_DATASET``        ``data/SAIT-EMA/Dataset``
``sait_raw``                 ``SAIT_EMA_RAW``            ``data/SAIT-EMA``
``runs``                     ``EMA_RUNS``                ``runs/paper``
``results``                  ``EMA_RESULTS``             ``paper/results``
``figures``                  ``EMA_FIGURES``             ``paper/figures``
===========================  ==========================  ==============================

Example::

    export CIS_EMA_DATASET="/scratch/ema/CIS-EMA"
    python scripts/run_exp1.py --seeds 0,1

or, equivalently, in ``paths.toml``::

    [paths]
    cis_dataset = "/scratch/ema/CIS-EMA"
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Environment variables that override each key.
ENV = {
    "cis_dataset": "CIS_EMA_DATASET",
    "cis_raw": "CIS_EMA_RAW",
    "cis_stimuli": "CIS_EMA_STIMULI",
    "sait_dataset": "SAIT_EMA_DATASET",
    "sait_raw": "SAIT_EMA_RAW",
    "runs": "EMA_RUNS",
    "results": "EMA_RESULTS",
    "figures": "EMA_FIGURES",
}

DEFAULTS = {
    "cis_dataset": "data/CIS-EMA",
    "cis_raw": "data/raw/CIS-EMA",
    "cis_stimuli": "data/raw/CIS-EMA/stimuli.xlsx",
    "sait_dataset": "data/SAIT-EMA/Dataset",
    "sait_raw": "data/SAIT-EMA",
    "runs": "runs/paper",
    "results": "paper/results",
    "figures": "paper/figures",
}

#: Human-readable description of each key, used in error messages.
DESCRIPTION = {
    "cis_dataset": "the derived CIS-EMA corpus (features.csv, manifest.csv, utterances/)",
    "cis_raw": "the raw CIS-EMA recordings (posR/, wav/, TG/)",
    "cis_stimuli": "the CIS-EMA stimulus list (stimuli.xlsx)",
    "sait_dataset": "the derived SAIT-EMA corpus (Dataset/ with features_mono.csv)",
    "sait_raw": "the raw SAIT-EMA download",
    "runs": "where experiment artifacts are written",
    "results": "where summary tables are written",
    "figures": "where generated figures are written",
}

_cache: dict[str, Path] | None = None


def _config_file() -> Path:
    return Path(os.environ.get("EMA_PATHS", REPO_ROOT / "paths.toml"))


def _load() -> dict[str, Path]:
    global _cache
    if _cache is not None:
        return _cache

    table: dict[str, str] = {}
    cfg = _config_file()
    if cfg.is_file():
        with cfg.open("rb") as fh:
            table = tomllib.load(fh).get("paths", {})

    resolved: dict[str, Path] = {}
    for key, env in ENV.items():
        raw = os.environ.get(env) or table.get(key) or DEFAULTS[key]
        p = Path(raw).expanduser()
        resolved[key] = p if p.is_absolute() else (REPO_ROOT / p)
    _cache = resolved
    return resolved


def get(key: str) -> Path:
    """Resolve one path key. Raises ``KeyError`` for an unknown key."""
    if key not in DEFAULTS:
        raise KeyError(f"unknown path key {key!r}; known keys: {sorted(DEFAULTS)}")
    return _load()[key]


def require(key: str, *, extra: str = "") -> Path:
    """Resolve one path key and fail with instructions if it does not exist."""
    p = get(key)
    if not p.exists():
        raise FileNotFoundError(
            f"{key} not found at {p}\n"
            f"  This is {DESCRIPTION[key]}.\n"
            f"  Point at it with {ENV[key]}=/abs/path or with [paths] {key} = \"...\" "
            f"in paths.toml (see paths.example.toml)."
            + (f"\n  {extra}" if extra else "")
        )
    return p


def describe() -> str:
    """One-line-per-key summary, printed by ``verify_results.py --paths``."""
    out = []
    for key in DEFAULTS:
        p = get(key)
        out.append(f"{key:14s} {'ok  ' if p.exists() else 'MISS'} {p}")
    return "\n".join(out)
