"""House style for every figure in this paper and this repository.

One place sets the typography, palette and line weights, so the figures cannot
drift apart from each other or from the manuscript body text. Import and call
``apply()`` at the top of a generator::

    from src import figstyle
    figstyle.apply()

Why a serif: the manuscript is typeset by IEEEtran in Times, and figures drawn
in matplotlib's default DejaVu Sans read as objects pasted onto the page rather
than drawn as part of it. ``apply()`` sets a Times stack with STIX maths, so a
label like *R*² sits in the same face as the caption.

Colour carries meaning and is used the same way everywhere:

``NAVY``   articulatory data — EMA channels, tongue coils, the model under study
``AMBER``  the acoustic or audio reference — hand-crafted acoustics, HuBERT, lip
           coils, the ceiling a claim is measured against
``SLATE``  a second, neutral comparison (e.g. an audio-only baseline)
``GREY``   secondary annotation: step changes, notes, axis hints
``INK``    values a reader should read first
"""
from __future__ import annotations

import matplotlib
from matplotlib import pyplot as plt

# ---- palette -------------------------------------------------------------
NAVY = "#1F4E79"
AMBER = "#B45309"
SLATE = "#4A5568"
GREY = "#6B7280"
LIGHT = "#9AA3B2"
INK = "#111111"

#: Legacy names used by the existing generators.
EMA_C = NAVY
ACO_C = AMBER
ACOUSTIC_C = AMBER
SSL_C = SLATE
TONGUE = NAVY
LIP = AMBER

FONT_STACK = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]


def apply(scale: float = 1.0) -> None:
    """Set the house rcParams. ``scale`` nudges every size together."""
    s = scale
    matplotlib.rcParams.update({
        # typography
        "font.family": "serif",
        "font.serif": FONT_STACK,
        "mathtext.fontset": "stix",
        "font.size": 8 * s,
        "axes.labelsize": 8 * s,
        "axes.titlesize": 8 * s,
        "xtick.labelsize": 7.5 * s,
        "ytick.labelsize": 7.5 * s,
        "legend.fontsize": 7.4 * s,
        # frame: open, hairlines only
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": 1.0,
        # output: raster for the repository, Type 42 vector for the manuscript
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def chip(alpha: float = 0.58):
    """Translucent rounded plate to put behind a value that sits on data.

    Opaque enough to separate a bold number from the marks underneath, light
    enough that those marks stay visible — and edged, so it reads as a design
    element rather than a rendering smudge.
    """
    return dict(boxstyle="round,pad=0.22,rounding_size=0.45",
                facecolor=(1, 1, 1, alpha), edgecolor="#C9CFD8", linewidth=0.4)


def save(fig, path, **kwargs):
    """Save through the QA gate (see ``src.figcheck``)."""
    from src import figcheck
    return figcheck.save(fig, path, **kwargs)
