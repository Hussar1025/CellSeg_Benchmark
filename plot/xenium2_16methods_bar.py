# -*- coding: utf-8 -*-
"""
Fig. 3b (plot) -- bar chart of transcript assignment consistency for 16 methods on xenium_2
================================================================
Reads metrics_transcript_xenium_2.csv produced by fig3d_metrics_xenium2.py,
and draws a two-sided bar chart relative to the baseline method (Cellpose-3) as the zero point, one PDF per metric.
Methods are in alphabetical order. Overlong bars get a diagonal break at the end to indicate clipping.

Changes from the previous version:
  · METHOD_TYPE extended to 16 methods; baseline is Cellpose-3; path changed to yourpath/result
  · bars thicker and row spacing tightened (BAR_HEIGHT 0.64 -> 0.86, ROW_H_IN 0.46 -> 0.36), smaller gaps
  · all methods now evaluated on the same set of qv>=20 transcripts, so image and transcript methods are directly comparable
Plotting logic unchanged. Read assign_rate together with the first four panels --
the low ARI of methods with low assignment rate comes mainly from coverage, not from assignment errors.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
SPASEG = "yourpath/result"
IN_CSV = f"{SPASEG}/metrics_transcript_xenium_2.csv"
OUT_DIR = f"{SPASEG}/figs_xenium_2"
FILE_FORMAT = "pdf"

REF_MODE = "baseline"
BASELINE = "Cellpose-3"
METRICS = ["ARI", "NMI", "HOM", "COM", "assign_rate"]
DATASET = None

METHOD_TYPE = {
    "Cellpose-SAM": "image", "Cellpose-3": "image", "CellSAM": "image", "Mesmer": "image",
    "StarDist": "image", "CelloType": "image", "UNSEG": "image",
    "ProSeg": "rna", "UCS": "rna", "ComSeg": "rna", "BOMS": "rna", "Cellist": "rna",
    "GeneSegNet": "rna", "Baysor": "rna", "Bering": "rna", "DISSECT": "rna",
}
SHOW_TYPE_MARK = False
TYPE_MARK = " †"

SORT_MODE = "alpha"
KEEP_BASELINE_ROW = True
SHOW_MISSING = True            # methods without results still occupy a row marked "n/a", so all 16 methods appear

# Bar-length compression: shrink the gap between long and short bars. Bar length is drawn as
# sign(Δ)·|Δ|^GAMMA while ticks still show the raw difference Δ, so readings stay the same and only
# long bars shrink and short bars grow. GAMMA=1 is linear (none), 0.5 is sqrt. Clip defaults off
# (CLIP=None); use GAMMA=1.0, CLIP="max_ratio" for linear + clip. After compression, the caption must state "x-axis is square-root scale".
GAMMA = 1.0                 # 1.0 = original linear scale (current); <1 = compressed scale (deprecated, switch kept)
SYMMETRIC_X = True
PAD_FRAC = 0.10
CLIP = "max_ratio"          # clip overlong bars and add a break mark
CLIP_RATIO = 0.80
MIN_SPAN = 0.02
BRK_GAP_FRAC = 0.022
BRK_SLANT_FRAC = 0.016
BRK_POS_FRAC = 0.055

POS_COLOR, NEG_COLOR = "#ED7D31", "#4490A9"
FRAME_COLOR, FRAME_WIDTH = "#1A1A1A", 0.8
BAR_EDGE_WIDTH = 0.4
BAR_HEIGHT = 0.74            # fraction of row height taken by the bar; larger = smaller gaps (1.0 = no gap)

PLOT_W_IN = 2.0
ROW_H_IN = 0.14             # row height (inch): thinner bars, tighter rows, shorter overall figure
LEFT_PAD_IN = 0.08
RIGHT_IN, TOP_IN, BOTTOM_IN = 0.08, 0.06, 0.34
TICK_FS, ROW_FS, LABEL_FS = 7, 8, 8
N_XTICKS = 5

METRIC_LABEL = {"assign_rate": "Assignment rate"}


def text_width_in(text, fs):
    if not hasattr(text_width_in, "_fig"):
        text_width_in._fig = plt.figure(figsize=(1, 1), dpi=100)
    f = text_width_in._fig
    t = f.text(0, 0, str(text), fontsize=fs)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    t.remove()
    return w


def label_of(m):
    return m + TYPE_MARK if (SHOW_TYPE_MARK and METHOD_TYPE.get(m) == "image") else m


def load(path):
    df = pd.read_csv(path)
    if DATASET is not None:
        df = df[df["dataset"].astype(str) == DATASET]
    ds = df["dataset"].astype(str).unique()
    if len(ds) != 1:
        raise SystemExit(f"csv has {len(ds)} datasets {list(ds)}, please set DATASET")
    have = [m for m in METRICS if m in df.columns]
    if set(METRICS) - set(have):
        print(f"[warn] csv missing these metric columns: {sorted(set(METRICS) - set(have))}")
    if not have:
        raise SystemExit("no metric columns at all")
    df = df.drop_duplicates("method").set_index("method")
    unk = [m for m in df.index if m not in METHOD_TYPE]
    if unk:
        print(f"[warn] these methods have no registered category: {unk}")
    miss = [m for m in METHOD_TYPE if m not in df.index]
    if miss:
        print(f"[warn] csv missing these methods: {miss}" +
              (" (kept as blank rows marked n/a)" if SHOW_MISSING else " (omitted from the figure)"))
    return df, have, ds[0]


def reference_of(df, metric):
    col = df[metric].astype(float)
    if REF_MODE == "median":
        return float(col.median()), "vs median"
    if REF_MODE == "mean":
        return float(col.mean()), "vs mean"
    if REF_MODE == "baseline":
        return float(col.loc[BASELINE]), f"vs {BASELINE}"
    if REF_MODE == "absolute":
        return 0.0, ""
    raise ValueError(REF_MODE)


def clip_threshold(v):
    a = np.abs(np.asarray(v, dtype=float))
    a = a[np.isfinite(a) & (a > 0)]
    if CLIP is None or len(a) == 0:
        return None
    if CLIP == "max_ratio":
        t = float(a.max()) * CLIP_RATIO
        return t if t > 0 else None
    return float(CLIP)


def xlim_of(v, clip_at):
    if REF_MODE == "absolute":
        return 0.0, float(np.nanmax(v)) * (1 + PAD_FRAC)
    m = max(clip_at if clip_at else float(np.nanmax(np.abs(v))), MIN_SPAN / 2)
    if not SYMMETRIC_X:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        span = max(hi - lo, MIN_SPAN)
        return min(lo, 0) - span * PAD_FRAC, max(hi, 0) + span * PAD_FRAC
    return -m * (1 + PAD_FRAC), m * (1 + PAD_FRAC)


def draw_break_gap(ax, x_tip, y, bar_h, span):
    from matplotlib.patches import Polygon
    sign = 1.0 if x_tip > 0 else -1.0
    gap, slant = span * BRK_GAP_FRAC, span * BRK_SLANT_FRAC
    xc = x_tip - sign * span * BRK_POS_FRAC
    h = bar_h / 2 * 1.10
    x1, x2 = xc - sign * gap / 2, xc + sign * gap / 2
    ax.add_patch(Polygon([(x1 - slant / 2, y - h), (x1 + slant / 2, y + h),
                          (x2 + slant / 2, y + h), (x2 - slant / 2, y - h)],
                         closed=True, facecolor="white", edgecolor="none", zorder=5))
    for x in (x1, x2):
        ax.plot([x - slant / 2, x + slant / 2], [y - h, y + h], color=FRAME_COLOR,
                linewidth=BAR_EDGE_WIDTH, solid_capstyle="butt", zorder=6)


def fwd(d):
    return np.sign(d) * np.abs(d) ** GAMMA


def nice_ticks(vmax):
    """Ticks in original units (symmetric about zero): 2 per side -- the largest round tick not
    exceeding the maximum, and a tick about halfway to it in plot scale, to avoid crowding near 0."""
    if vmax <= 0:
        return [0.0]
    cand = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0]
    pos = [c for c in cand if c <= vmax * 1.0001] or [vmax]
    big = pos[-1]
    half = min(pos[:-1] or [big / 2], key=lambda c: abs(fwd(c) - fwd(big) / 2))
    ticks = sorted({half, big})
    return [-t for t in reversed(ticks)] + [0.0] + ticks


def draw_one(df, metric, ds, out_path):
    base, ref_txt = reference_of(df, metric)
    rows = list(df.index) + ([m for m in METHOD_TYPE if m not in df.index] if SHOW_MISSING else [])
    rows = [m for m in rows if KEEP_BASELINE_ROW or REF_MODE != "baseline" or m != BASELINE]
    delta = df[metric].astype(float).reindex(rows) - base
    rows = (sorted(rows, key=str.lower) if SORT_MODE == "alpha"
            else list(delta.sort_values(ascending=False).index))
    v = delta.loc[rows].to_numpy()
    missing = ~np.isfinite(v)
    labels = [label_of(m) for m in rows]
    n_row = len(rows)

    left_in = max(text_width_in(s, ROW_FS) for s in labels) + LEFT_PAD_IN
    fig_w = left_in + PLOT_W_IN + RIGHT_IN
    fig_h = BOTTOM_IN + n_row * ROW_H_IN + TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, BOTTOM_IN / fig_h,
                       PLOT_W_IN / fig_w, n_row * ROW_H_IN / fig_h])

    ypos = np.arange(n_row)[::-1]
    clip_at = clip_threshold(v)
    v0 = np.where(missing, 0.0, v)
    v_lin = np.clip(v0, -clip_at, clip_at) if clip_at else v0.copy()
    v_draw = fwd(v_lin)
    clipped = (np.abs(v0) >= (clip_at - 1e-12)) & ~missing if clip_at else np.zeros(n_row, bool)

    ax.barh(ypos[~missing], v_draw[~missing], height=BAR_HEIGHT,
            color=[POS_COLOR if t > 0 else NEG_COLOR for t in v[~missing]],
            edgecolor=FRAME_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    if REF_MODE != "absolute":
        ax.axvline(0, color=FRAME_COLOR, linewidth=FRAME_WIDTH, zorder=4)

    m_ = max(float(np.nanmax(np.abs(v_lin))) if len(v_lin) else 0.0, MIN_SPAN / 2)
    lo, hi = -fwd(m_) * (1 + PAD_FRAC), fwd(m_) * (1 + PAD_FRAC)
    for i in np.nonzero(missing)[0]:
        ax.text(0, ypos[i], "  n/a", ha="left", va="center", fontsize=TICK_FS - 1,
                color="#8C8C8C", zorder=5)
    ax.set_xlim(lo, hi)
    if clipped.any():
        for i in np.nonzero(clipped)[0]:
            draw_break_gap(ax, v_draw[i], ypos[i], BAR_HEIGHT, hi - lo)
        print(f"    {metric}: clip threshold {clip_at:.4f}, clipped "
              f"{', '.join(rows[i] for i in np.nonzero(clipped)[0])}")

    ax.set_ylim(-0.6, n_row - 0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=ROW_FS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=TICK_FS, length=3.5,
                   width=FRAME_WIDTH, direction="out", color=FRAME_COLOR)
    ticks = nice_ticks(m_)
    ax.set_xticks([fwd(t) for t in ticks])
    ax.set_xticklabels([("0" if t == 0 else f"{t:+g}") for t in ticks])
    ax.grid(True, axis="x", linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(FRAME_COLOR)
        ax.spines[side].set_linewidth(FRAME_WIDTH)

    lab = METRIC_LABEL.get(metric, metric)
    scale_txt = "" if GAMMA == 1 else ("  [sqrt scale]" if abs(GAMMA - 0.5) < 1e-9 else f"  [|Δ|^{GAMMA:g} scale]")
    ax.set_xlabel((f"Δ{lab} ({ref_txt})" if ref_txt else lab) + scale_txt, fontsize=LABEL_FS)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    src, outdir = IN_CSV, OUT_DIR
    if len(sys.argv) == 3:
        src, outdir = sys.argv[1:3]
    os.makedirs(outdir, exist_ok=True)
    df, metrics, ds = load(src)
    if REF_MODE == "baseline" and BASELINE not in df.index:
        raise SystemExit(f"baseline '{BASELINE}' is not in the data. Methods: {sorted(df.index)}")
    n_img = sum(1 for m in df.index if METHOD_TYPE.get(m) == "image")
    print(f"dataset {ds} | {len(df)} methods (image {n_img}, RNA {len(df)-n_img})"
          f" | reference {REF_MODE} | sort {SORT_MODE} | output {FILE_FORMAT}")
    show = pd.DataFrame({m: df[m].astype(float) - reference_of(df, m)[0] for m in metrics})
    print("Zero point: " + ", ".join(f"{m}={reference_of(df,m)[0]:.4f}" for m in metrics))
    print(show.round(4).to_string())
    for met in metrics:
        draw_one(df, met, ds, os.path.join(outdir, f"bar_{ds}_{met}.{FILE_FORMAT}"))


if __name__ == "__main__":
    main()