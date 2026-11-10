"""One-speaker 3D sequence + height trajectories for /baa/, /daai/, /gu/.

Leaves ``plot_montage_3d_vowels.py`` untouched (single-frame 3D still works).
Does not touch ``fig_montage_trajectories.png``.

Layout (3 columns x 5 rows)
    columns:  /baa/  /daai/  /gu/
    rows 1-3: 3D snapshots at consonant, vowel, and end
    rows 4-5: tongue-dorsum height and upper-lip height for that token,
              with vertical lines at the three snapshot times

Time-point choice (phonetic, not equal fractions of the recording)
    Consonant  constriction in the onset window: min UL height for /baa/,
               max TT height for /daai/, max TD height for /gu/.
    Vowel      steady vowel target: midpoint of the labelled vowel for
               /baa/ and /gu/; 40% into /aai/ so the snapshot is the /aa/
               nucleus, not the offglide.
    End        late vowel, before rest: 75% of /aa/ and /u/; for /daai/,
               most anterior TT frame in the late vowel (the /i/ offglide).

Default speaker is the valid talker whose three tokens most clearly show
those contrasts; pass ``--random`` (and optionally ``--subj``) to override.

Writes ``paper/illustrations/3d/fig_montage_3d_sequence.png``.
"""
import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from preprocessing.prepare_dataset import parse_textgrid
from src.config import DATA_DIR
from src.paper_figures import TONGUE_C, LIP_C, REF_C, _sensor_colors


def _load_base():
    path = os.path.join(SCRIPT_DIR, "plot_montage_3d_vowels.py")
    spec = importlib.util.spec_from_file_location("plot_montage_3d_vowels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base()
OUT_DIR = base.OUT_DIR
SYLLABLES = base.SYLLABLES
POS_COLS = base.POS_COLS
PLACE = {"baa": "bilabial", "daai": "alveolar", "gu": "velar"}
STAGES = ("consonant", "vowel", "end")
STAGE_LABEL = {"consonant": "Consonant", "vowel": "Vowel", "end": "End"}
STAGE_C = {"consonant": "#B4473C", "vowel": "#2F6DB5", "end": "#3D8B6E"}
MIN_FRAMES = 48
DEFAULT_SEED = 21


def _onset_span(tg_path, word, onset):
    """Onset interval inside the word; fall back to [word_start, vowel_start)."""
    ws, we, va, vb = word
    tiers = parse_textgrid(tg_path)
    segs = [(a, b, t.strip()) for a, b, t in tiers.get("segment", []) if t.strip()]
    hits = [(a, b) for a, b, t in segs
            if t == onset and a >= ws - 1e-4 and b <= we + 1e-4]
    if hits:
        return hits[0]
    return ws, va


def _ema_time(g, word, tg_t):
    ws, we = word[0], word[1]
    t = g["time"].to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    dur = we - ws
    if dur <= 1e-6:
        return t0
    return t0 + (tg_t - ws) / dur * (t1 - t0)


def _nearest_row(g, t_tgt):
    t = g["time"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(t - t_tgt)))
    return g.iloc[i], i


def _window(g, t_lo, t_hi):
    t = g["time"].to_numpy(dtype=float)
    m = (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    return g.loc[m]


def _extremum_row(sub, col, mode):
    v = sub[col].to_numpy(dtype=float)
    j = int(np.argmin(v) if mode == "min" else np.argmax(v))
    return sub.iloc[j], int(sub.index[j])


def _frac(g, row):
    t = g["time"].to_numpy(dtype=float)
    return float((row["time"] - t[0]) / (t[-1] - t[0]))


def _pick_stages(g, syl, span, _onset=None):
    """Three phonetically motivated frames inside one token."""
    ws, we, va, vb = span
    t = g["time"].to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    ema_va = _ema_time(g, span, va)
    ema_vb = _ema_time(g, span, vb)

    # Closure lives at/before the acoustic burst, which is often only a few
    # labelled milliseconds. Search from the word start to 50 ms into the vowel.
    c_hi = min(ema_va + 0.050, t0 + 0.30 * (t1 - t0))
    c_sub = _window(g, t0, c_hi)
    if len(c_sub) == 0:
        c_sub = g.iloc[: max(3, len(g) // 8)]
    if syl == "baa":
        c_row, _ = _extremum_row(c_sub, "ULz", "min")
    elif syl == "daai":
        c_row, _ = _extremum_row(c_sub, "TTz", "max")
    else:
        c_row, _ = _extremum_row(c_sub, "TDz", "max")
    c_row, c_i = _nearest_row(g, float(c_row["time"]))

    if syl == "daai":
        v_tgt = ema_va + 0.40 * (ema_vb - ema_va)
    else:
        v_tgt = ema_va + 0.50 * (ema_vb - ema_va)
    v_row, v_i = _nearest_row(g, v_tgt)

    # Late vowel, not the return-to-rest tail. For /daai/ take the most
    # anterior tongue-tip frame in that window (the /i/ offglide).
    e_lo = ema_va + 0.58 * (ema_vb - ema_va)
    e_hi = ema_va + 0.82 * (ema_vb - ema_va)
    if syl == "daai":
        e_sub = _window(g, e_lo, e_hi)
        if len(e_sub) == 0:
            e_row, e_i = _nearest_row(g, ema_va + 0.75 * (ema_vb - ema_va))
        else:
            e_hit, _ = _extremum_row(e_sub, "TTx", "max")
            e_row, e_i = _nearest_row(g, float(e_hit["time"]))
    else:
        e_row, e_i = _nearest_row(g, ema_va + 0.75 * (ema_vb - ema_va))

    # Keep the three frames strictly ordered in time.
    idx = np.array([c_i, v_i, e_i], dtype=int)
    n = len(g)
    if idx[1] <= idx[0]:
        idx[1] = min(n - 2, idx[0] + max(3, n // 10))
    if idx[2] <= idx[1]:
        idx[2] = min(n - 1, idx[1] + max(3, n // 10))
    rows = [g.iloc[int(i)] for i in idx]
    return {
        stage: {
            "row": rows[k],
            "index": int(idx[k]),
            "frac": _frac(g, rows[k]),
            "xyz": base._xyz_from_row(rows[k]),
        }
        for k, stage in enumerate(STAGES)
    }


def _token_ok(feat, folder, tok):
    tg = base._tg_path(folder, tok["file_num"])
    if tg is None:
        return None
    span = base._word_vowel_span(tg, tok["vowel"])
    if span is None:
        return None
    g = feat[feat["source"] == tok["source"]].sort_values("time")
    if len(g) < MIN_FRAMES:
        return None
    vg = base._vowel_frames(g, span)
    got = base._vowel_target_row(vg)
    if got is None:
        return None
    row, _, _ = got
    if not base._canonical_side(row):
        return None
    onset = _onset_span(tg, span, tok["onset"])
    return {"tok": tok, "span": span, "onset": onset, "g": g, "tg": tg}


def _collect_speakers(man, feat, folders):
    out = {}
    for subj, folder in folders.items():
        sub = man[man["subj"] == subj]
        if not set(SYLLABLES).issubset(set(sub["syllable"])):
            continue
        cand = {}
        ok = True
        for syl in SYLLABLES:
            pool = sub[sub["syllable"] == syl].sort_values(
                "n_frames", ascending=False)
            kept = []
            for _, tok in pool.iterrows():
                got = _token_ok(feat, folder, tok)
                if got is not None:
                    kept.append(got)
            if not kept:
                ok = False
                break
            cand[syl] = kept
        if ok:
            out[subj] = cand
    if not out:
        raise SystemExit("no speaker has usable /baa/, /daai/ and /gu/")
    return out


def _score_triple(picked):
    """How clearly this speaker's tokens show place, vowel, and the /i/ offglide."""
    st = {
        syl: _pick_stages(picked[syl]["g"], syl, picked[syl]["span"], picked[syl]["onset"])
        for syl in SYLLABLES
    }
    baa_open = float(st["baa"]["vowel"]["row"]["ULz"] - st["baa"]["consonant"]["row"]["ULz"])
    daai_front = float(st["daai"]["end"]["row"]["TTx"] - st["daai"]["vowel"]["row"]["TTx"])
    daai_raise = float(st["daai"]["end"]["row"]["TTz"] - st["daai"]["vowel"]["row"]["TTz"])
    daai_c_tt = float(st["daai"]["consonant"]["row"]["TTz"] - st["daai"]["vowel"]["row"]["TTz"])
    gu_high = float(st["gu"]["consonant"]["row"]["TDz"] - st["baa"]["vowel"]["row"]["TDz"])
    return baa_open + 1.6 * daai_front + 1.3 * daai_raise + 0.7 * daai_c_tt + 0.5 * gu_high


def _choose(speakers, rng, subj=None, random_subj=False):
    ids = sorted(speakers)
    picked_of = lambda s: {syl: speakers[s][syl][0] for syl in SYLLABLES}
    if subj is not None:
        if subj not in speakers:
            raise SystemExit(f"sub_{subj:02d} has no usable tokens; try one of {ids}")
        return subj, picked_of(subj)
    if random_subj:
        subj = int(rng.choice(ids))
        return subj, picked_of(subj)
    ranked = sorted(
        ((_score_triple(picked_of(s)), s) for s in ids), reverse=True)
    return ranked[0][1], picked_of(ranked[0][1])


def _axis_ticks(lo, hi):
    span = hi - lo
    step = 1.0 if span > 2.2 else 0.5
    start = np.ceil(lo / step) * step
    ticks = np.arange(start, hi + 1e-9, step)
    return ticks if len(ticks) >= 2 else np.array([lo, hi])


def _style_3d_compact(ax):
    ax.set_proj_type("ortho")
    ax.computed_zorder = False
    ax.grid(True, linestyle=":", color="0.82", linewidth=0.45)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor("0.78")
        axis.pane.set_linewidth(0.5)
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.18
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.9)
        axis._axinfo["grid"]["linewidth"] = 0.45
        axis._axinfo["grid"]["linestyle"] = ":"
    ax.tick_params(pad=1, labelsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def _draw_compact(ax, xyz, limits, show_labels=False, show_midsagittal=False):
    """Same sensor geometry as ``plot_montage_3d_vowels._draw``, fewer labels."""
    color = _sensor_colors()
    x_lo, x_hi = limits["x"]
    y_lo, y_hi = limits["y"]
    z_lo, z_hi = limits["z"]
    plane = [[(x_lo, 0.0, z_lo), (x_hi, 0.0, z_lo),
              (x_hi, 0.0, z_hi), (x_lo, 0.0, z_hi)]]
    ax.add_collection3d(Poly3DCollection(
        plane, facecolor=(0.45, 0.58, 0.72, 0.12),
        edgecolor=(0.45, 0.50, 0.58, 0.35), linewidth=0.55, zorder=1))
    if show_midsagittal:
        ax.text(x_hi - 0.10, 0.0, z_hi - 0.10, "Midsagittal",
                fontsize=6.5, color="0.42", style="italic", zorder=2, ha="right")

    def _seg(a, b, c, lw=1.2, ls="-"):
        ax.plot([xyz[a][0], xyz[b][0]],
                [xyz[a][1], xyz[b][1]],
                [xyz[a][2], xyz[b][2]],
                color=c, lw=lw, ls=ls, zorder=3, solid_capstyle="round")

    _seg("TD", "TT", TONGUE_C, lw=1.45)
    _seg("TD", "TL", TONGUE_C, lw=1.05, ls=":")
    _seg("TL", "TT", TONGUE_C, lw=1.05, ls=":")
    _seg("UL", "LL", LIP_C, lw=1.45)
    _seg("UL", "LC", LIP_C, lw=1.05, ls=":")
    _seg("LL", "LC", LIP_C, lw=1.05, ls=":")

    for name in ("TL", "LC"):
        p = xyz[name]
        ax.plot([p[0], p[0]], [p[1], 0.0], [p[2], p[2]],
                color=color[name], lw=0.8, ls="--", alpha=0.55, zorder=2)
        ax.scatter([p[0]], [0.0], [p[2]], s=12, facecolors="white",
                   edgecolors=color[name], linewidths=0.7, depthshade=False,
                   zorder=4)

    for name, p in xyz.items():
        ax.scatter([p[0]], [p[1]], [p[2]],
                   s=62,
                   color=color[name], edgecolor="white", linewidth=0.65,
                   depthshade=False, zorder=6)

    if show_labels:
        off = {
            "TD": (-0.20, 0.04, 0.12),
            "TL": (0.02, -0.16, 0.10),
            "TT": (0.16, 0.08, -0.05),
            "UL": (0.18, 0.04, 0.10),
            "LL": (0.20, 0.04, -0.14),
            "LC": (0.04, -0.18, 0.08),
        }
        for name, p in xyz.items():
            d = off[name]
            ax.text(p[0] + d[0], p[1] + d[1], p[2] + d[2], name,
                    fontsize=7.0, fontweight="bold", color="0.12",
                    zorder=7, ha="center", va="center")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo), zoom=1.32)
    ax.set_xticks(_axis_ticks(x_lo, x_hi))
    ax.set_yticks(_axis_ticks(y_lo, y_hi))
    ax.set_zticks(_axis_ticks(z_lo, z_hi))


def _tight_limits(all_xyz):
    """Tighter box than the single-frame helper so coils fill a small panel."""
    pts = np.stack([p for xyz in all_xyz.values() for p in xyz.values()])
    lo = pts.min(0) - 0.08
    hi = pts.max(0) + 0.08
    lo[1] = min(lo[1], -0.12)
    hi[1] = max(hi[1], 0.12)
    lo[2] = min(lo[2], -0.04)
    return {
        "x": (float(lo[0]), float(hi[0])),
        "y": (float(lo[1]), float(hi[1])),
        "z": (float(lo[2]), float(hi[2])),
    }


def _traj_limits(picked, col):
    lo, hi = np.inf, -np.inf
    for syl in SYLLABLES:
        v = picked[syl]["g"][col].to_numpy(dtype=float)
        lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
    pad = 0.10 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _plot_traj(ax, g, stages, col, ylim, show_xlabel, ylabel):
    t = g["time"].to_numpy(dtype=float)
    x = (t - t[0]) / (t[-1] - t[0])
    y = g[col].to_numpy(dtype=float)
    ax.plot(x, y, color="0.18", lw=1.35, zorder=2, solid_capstyle="round")
    for stage in STAGES:
        f = stages[stage]["frac"]
        ax.axvline(f, color=STAGE_C[stage], lw=1.15, zorder=3)
        yi = float(np.interp(f, x, y))
        ax.scatter([f], [yi], s=22, color=STAGE_C[stage], edgecolor="white",
                   linewidth=0.5, zorder=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Normalized time", fontsize=8)
        ax.set_xticks([0.0, 0.5, 1.0])
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=7, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_figure(subj, picked):
    all_xyz = {}
    for syl in SYLLABLES:
        stages = _pick_stages(
            picked[syl]["g"], syl, picked[syl]["span"], picked[syl]["onset"])
        picked[syl]["stages"] = stages
        for st, rec in stages.items():
            all_xyz[f"{syl}_{st}"] = rec["xyz"]
    limits = _tight_limits(all_xyz)
    td_lim = _traj_limits(picked, "TDz")
    ul_lim = _traj_limits(picked, "ULz")

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(11.0, 12.2))
    left, right = 0.078, 0.985
    gs_3d = fig.add_gridspec(
        3, 3, left=left, right=right, top=0.945, bottom=0.385,
        hspace=0.04, wspace=0.03)
    gs_tr = fig.add_gridspec(
        2, 3, left=left, right=right, top=0.348, bottom=0.078,
        hspace=0.28, wspace=0.03)

    for c, syl in enumerate(SYLLABLES):
        x = left + (c + 0.5) * (right - left) / 3
        fig.text(x, 0.975, f"/{syl}/  ·  {PLACE[syl]}",
                 ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(right, 0.975, f"sub {subj}", ha="right", va="top",
             fontsize=8.5, color="0.4")

    for r, stage in enumerate(STAGES):
        for c, syl in enumerate(SYLLABLES):
            ax = fig.add_subplot(gs_3d[r, c], projection="3d")
            _style_3d_compact(ax)
            rec = picked[syl]["stages"][stage]
            _draw_compact(
                ax, rec["xyz"], limits,
                show_labels=(r == 0 and c == 0),
                show_midsagittal=(r == 0 and c == 0),
            )
            ax.view_init(elev=22, azim=-38)
            if r != 2:
                ax.set_xticklabels([])
            ax.set_yticklabels([])
            if c != 0:
                ax.set_zticklabels([])
            ax.text2D(0.03, 0.97, STAGE_LABEL[stage][0],
                      transform=ax.transAxes, ha="left", va="top",
                      fontsize=10, fontweight="bold", color=STAGE_C[stage])
            if c == 0:
                ax.text2D(-0.12, 0.50, STAGE_LABEL[stage],
                          transform=ax.transAxes, rotation=90,
                          ha="center", va="center", fontsize=10,
                          fontweight="bold", color=STAGE_C[stage])

    for c, syl in enumerate(SYLLABLES):
        g = picked[syl]["g"]
        stages = picked[syl]["stages"]
        ax_td = fig.add_subplot(gs_tr[0, c])
        ax_ul = fig.add_subplot(gs_tr[1, c])
        _plot_traj(ax_td, g, stages, "TDz", td_lim, False,
                   "Tongue height" if c == 0 else "")
        _plot_traj(ax_ul, g, stages, "ULz", ul_lim, True,
                   "Lip height" if c == 0 else "")
        if c != 0:
            ax_td.set_yticklabels([])
            ax_ul.set_yticklabels([])
        for stage in STAGES:
            f = stages[stage]["frac"]
            ha = "left" if f < 0.06 else ("right" if f > 0.94 else "center")
            ax_td.annotate(
                STAGE_LABEL[stage][0], xy=(f, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points",
                ha=ha, va="bottom", fontsize=8, fontweight="bold",
                color=STAGE_C[stage], annotation_clip=False)

    sensor_handles = [
        Patch(color=TONGUE_C, label="Tongue (TD, TL, TT)"),
        Patch(color=LIP_C, label="Lips (UL, LL, LC)"),
    ]
    time_handles = [
        Line2D([0], [0], color=STAGE_C[s], lw=1.6, label=STAGE_LABEL[s])
        for s in STAGES
    ]
    fig.legend(handles=sensor_handles + time_handles,
               loc="upper center", bbox_to_anchor=(0.53, 0.032),
               ncol=6, frameon=False, fontsize=8,
               columnspacing=1.15, handlelength=1.25)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--subj", type=int, default=None,
                    help="Force a speaker id")
    ap.add_argument("--random", action="store_true",
                    help="Pick a random valid speaker (uses --seed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))
    feat = pd.read_csv(os.path.join(DATA_DIR, "features.csv"),
                       usecols=["source", "time"] + POS_COLS)
    folders = base._subj_folders()
    speakers = _collect_speakers(man, feat, folders)
    rng = np.random.RandomState(args.seed)
    subj, picked = _choose(speakers, rng, args.subj, random_subj=args.random)

    fig = _build_figure(subj, picked)
    out = args.out or os.path.join(OUT_DIR, "fig_montage_3d_sequence.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)

    print(f"speaker sub_{subj:02d}  seed={args.seed}  "
          f"({len(speakers)} valid speakers)")
    for syl in SYLLABLES:
        tok = picked[syl]["tok"]
        span = picked[syl]["span"]
        stages = picked[syl]["stages"]
        ws, we, va, vb = span
        bits = "  ".join(
            f"{st[0].upper()} {rec['index']}/{len(picked[syl]['g'])} "
            f"t={rec['frac']:.2f}"
            for st, rec in stages.items()
        )
        print(
            f"/{syl}/  {tok['source']}  {tok['n_frames']} frames  "
            f"vowel {vb-va:.3f}s\n  {bits}"
        )
        for st, rec in stages.items():
            row = rec["row"]
            print(
                f"    {st[0].upper()}  TDx={row['TDx']:+.3f} TDz={row['TDz']:+.3f}  "
                f"TTx={row['TTx']:+.3f} TTz={row['TTz']:+.3f}  "
                f"ULz={row['ULz']:+.3f}"
            )
    print("wrote", out)


if __name__ == "__main__":
    main()
