"""Central configuration for the EMA-SSI paper pipeline.

This file freezes the protocol of Experiment 0 (see paper/experimentation_plan_*):
  - dataset location and raw-audio location
  - the 8-group feature taxonomy (G1-G8)
  - subject-disjoint sex-stratified CV constants (folds / seeds / val size)
  - onset -> place-of-articulation and onset -> manner (sonorant) maps

Everything in Experiments 1-4 imports these definitions so that the CV splits
and feature groups are identical across experiments (the Exp-0 requirement).
"""
import os

from src.paths import get as path

# --------------------------------------------------------------------------- #
# Paths
#
# Resolved by ``src.paths`` (environment variable > paths.toml > default).
# Nothing here is an absolute path; see paths.example.toml.
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = str(path("cis_dataset"))
RAW_DIR = str(path("cis_raw"))
STIMULI_PATH = str(path("cis_stimuli"))
RUNS_DIR = str(path("runs"))
RESULTS_DIR = str(path("results"))
FIGURES_DIR = str(path("figures"))
FOLDS_PATH = os.path.join(REPO_ROOT, "folds", "cis_folds.json")

# Item numbers excluded from tone analysis (ambiguous tonal pronunciation).
# 114 = "陶域道" (transcribed jiu4); the character 陶 has an alternate tonal
# reading across speakers, so its tone label is unreliable.
EXCLUDED_TONE_ITEMS = (114,)

# --------------------------------------------------------------------------- #
# Evaluation protocol (Experiment 0.2)
# --------------------------------------------------------------------------- #
N_FOLDS = 6
N_VAL_SUBJECTS = 4
SEEDS = (0, 1)            # two random seeds per fold -> 12 runs per config
FOLD_SEED = 0             # seed used only to build the subject folds

# --------------------------------------------------------------------------- #
# Feature-group taxonomy (Experiment 0.3)
#
#   G1 EMA-position      18   raw xyz of 6 sensors
#   G2 EMA-velocity      18   1st derivative
#   G3 EMA-acceleration  18   2nd derivative
#   G4 EMA-geometry       5   3 distances + 2 angles
#   G5 Acoustic-voicing   2   F0 + voiced flag
#   G6 Acoustic-formant   3   F1, F2, F3
#   G7 Acoustic-spectral 39   13 MFCC + delta + delta-delta
#   G8 Acoustic-energy    3   energy + delta + delta-delta
#
# (Note: the plan text lists G8 as "energy | 1", but the rebuilt feature table
# stores three energy features -- energy, energy_d, energy_dd -- so G8 has 3.
# This is what makes the totals add up to 59 EMA + 47 acoustic = 106.)
# --------------------------------------------------------------------------- #
SENSORS = ["TD", "TL", "TT", "UL", "LL", "LC"]   # 6 articulatory coils (LL included)
AXES = ["x", "y", "z"]

POSITION = [f"{s}{a}" for s in SENSORS for a in AXES]
VELOCITY = [f"{s}{a}_velocity" for s in SENSORS for a in AXES]
ACCELERATION = [f"{s}{a}_acceleration" for s in SENSORS for a in AXES]
GEOMETRY = [
    "TT_TD_distance", "UL_LL_distance", "TT_LL_distance",
    "TT_TD_LL_angle", "UL_LL_LC_angle",
]
VOICING = ["F0", "voiced"]
FORMANTS = ["F1", "F2", "F3"]
SPECTRAL = [f"MFCC_{k}{s}" for k in range(1, 14) for s in ("", "_d", "_dd")]
ENERGY = ["energy", "energy_d", "energy_dd"]

FEATURE_GROUPS = {
    "G1": {"name": "EMA-position", "kind": "ema", "columns": POSITION},
    "G2": {"name": "EMA-velocity", "kind": "ema", "columns": VELOCITY},
    "G3": {"name": "EMA-acceleration", "kind": "ema", "columns": ACCELERATION},
    "G4": {"name": "EMA-geometry", "kind": "ema", "columns": GEOMETRY},
    "G5": {"name": "Acoustic-voicing", "kind": "acoustic", "columns": VOICING},
    "G6": {"name": "Acoustic-formant", "kind": "acoustic", "columns": FORMANTS},
    "G7": {"name": "Acoustic-spectral", "kind": "acoustic", "columns": SPECTRAL},
    "G8": {"name": "Acoustic-energy", "kind": "acoustic", "columns": ENERGY},
}
GROUP_IDS = list(FEATURE_GROUPS.keys())

EMA_GROUPS = ["G1", "G2", "G3", "G4"]
ACOUSTIC_GROUPS = ["G5", "G6", "G7", "G8"]

EMA_COLUMNS = [c for g in EMA_GROUPS for c in FEATURE_GROUPS[g]["columns"]]
ACOUSTIC_COLUMNS = [c for g in ACOUSTIC_GROUPS for c in FEATURE_GROUPS[g]["columns"]]
ALL_COLUMNS = EMA_COLUMNS + ACOUSTIC_COLUMNS
assert len(EMA_COLUMNS) == 59 and len(ACOUSTIC_COLUMNS) == 47, \
    (len(EMA_COLUMNS), len(ACOUSTIC_COLUMNS))


def columns_for(groups):
    """Ordered feature columns for a subset of groups (e.g. ['G1','G2','G3','G4'])."""
    cols = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g]["columns"])
    return cols


# --------------------------------------------------------------------------- #
# Phonetic label maps (Experiment 3.1)
#
# The corpus stores onset (15 classes) and vowel (7 classes). Place-of-
# articulation and manner (sonorant vs obstruent) are derived from the onset
# phoneme (Jyutping conventions). Cantonese has no phonemic voicing contrast in
# obstruents, so this "manner" bit is derived as sonorant vs obstruent
# (sonorants are phonetically voiced; obstruents are voiceless).
# --------------------------------------------------------------------------- #
ONSET_PLACE = {
    "b": "bilabial", "p": "bilabial",
    "f": "labiodental",
    "d": "alveolar", "t": "alveolar", "z": "alveolar", "c": "alveolar",
    "s": "alveolar", "l": "alveolar",
    "g": "velar", "k": "velar", "N": "velar",
    "h": "glottal",
    "j": "glide", "w": "glide",
}

ONSET_SONORANT = {
    "l": True, "N": True, "j": True, "w": True,
    "b": False, "c": False, "d": False, "f": False,
    "g": False, "h": False, "k": False, "p": False,
    "s": False, "t": False, "z": False,
}

N_SYLLABLES = 43
N_VOWELS = 7
N_ONSETS = 15
N_TONES = 6             # Cantonese lexical tones 1-6
CANTONESE_TONES = (1, 2, 3, 4, 5, 6)
