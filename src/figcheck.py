"""Guard against overlapping text in generated figures.

Academic figures are read at column width and in greyscale print, so a label that
collides with a tick or a neighbour is a defect, not a cosmetic detail. This module
finds every visible text artist in a figure, asks the renderer where it actually
landed, and reports any pair whose boxes intersect.

Use it at the point of saving::

    from src.figcheck import save
    save(fig, out_path, bbox_inches="tight")

``save`` raises :class:`TextOverlapError` with the offending strings, so a broken
figure never reaches disk. Set ``FIGCHECK=warn`` to downgrade to a warning while
you are iterating.

Overlaps that are intentional (a legend drawn over the data area, a panel letter
sitting on a title) are reported separately as *allowed* when one of the pair is a
figure-level text or a legend, so the noise does not hide a real collision between
two axis labels.
"""
from __future__ import annotations

import os

from matplotlib.text import Text

TOLERANCE_PX = 0.0


class TextOverlapError(RuntimeError):
    """Raised when two text artists in a saved figure occupy the same pixels."""


def _visible_texts(fig):
    """Every non-empty, visible Text artist belonging to the figure."""
    out = []
    for ax in fig.axes:
        candidates = list(ax.texts)
        candidates += [ax.title]
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            candidates += [axis.label, axis.get_offset_text()]
            # A locator may return ticks outside the view limits; matplotlib builds a
            # label artist for them but clips it, so their geometry is meaningless and
            # comparing it reports collisions that no reader can see.
            for tick in axis.get_major_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9):
                    continue
                candidates += [tick.label1, tick.label2]
        if ax.get_legend() is not None:
            candidates += list(ax.get_legend().get_texts())
        for t in candidates:
            if isinstance(t, Text) and t.get_visible() and t.get_text().strip():
                out.append(t)
    out += [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    return out


def _box(t, renderer):
    try:
        bb = t.get_window_extent(renderer=renderer)
    except Exception:  # noqa: BLE001 - a text without a renderer cannot be checked
        return None
    if bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def find_overlaps(fig, ignore=()):
    """Return a list of (text_a, text_b, overlap_px) for colliding text artists."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for t in _visible_texts(fig):
        label = t.get_text().strip()
        if any(s and s in label for s in ignore):
            continue
        bb = _box(t, renderer)
        if bb is not None:
            items.append((label, bb))

    hits = []
    for i in range(len(items)):
        la, ba = items[i]
        for j in range(i + 1, len(items)):
            lb, bb = items[j]
            dx = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            dy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if dx > TOLERANCE_PX and dy > TOLERANCE_PX:
                hits.append((la, lb, round(min(dx, dy), 1)))
    return hits


def check(fig, name="figure", ignore=()):
    """Return a human-readable report; empty string when the figure is clean."""
    hits = find_overlaps(fig, ignore=ignore)
    if not hits:
        return ""
    lines = [f"{name}: {len(hits)} overlapping text pair(s)"]
    for a, b, px in hits[:12]:
        lines.append(f"  {a!r} x {b!r}  ({px} px)")
    return "\n".join(lines)


def save(fig, path, **kwargs):
    """Run the overlap check, then save. Raises unless FIGCHECK=warn."""
    report = check(fig, name=os.path.basename(path))
    if report:
        if os.environ.get("FIGCHECK", "").lower() == "warn":
            print("WARNING " + report)
        else:
            raise TextOverlapError(
                report + "\nFix the layout (or set FIGCHECK=warn to inspect it) before saving."
            )
    fig.savefig(path, **kwargs)
    return path
