# -*- coding: utf-8 -*-
"""
Composite robustness bar chart
==============================
Y-axis = 16 methods; each method has four bars for four metrics, colored to match
the bubble matrix class palette. X-axis ticks on top, legend at the bottom.

Uses only the two summary tables:
  benchmark_result_omics.xlsx  "Summary" sheet                   -> 9 RNA methods
  benchmark_result_image.xlsx  "Image Segmentation Summary" sheet -> 7 image methods
The only metrics truly shared by both tables are cell_ratio / precision / recall / f1
(the image table has no loc / count / vec columns), so exactly four bars.

[Adapt to latest excel] the two tables name the same datasets differently
(e.g. xenium_breast / xenium_2); DATASET_ALIAS maps them to the omics-table names
so RNA and image methods rank together on the same dataset.
Rows under xenium_liver with GT~10,654 are actually CosMx human-brain data, mapped to cosmx_3
(this rule matches nothing once the local data is fixed).

Same pipeline as the bubble matrix:
  1. rank within each dataset per platform (direction-corrected, rank 1 = best)
  2. normalize ranks to 0~1 to remove bias from differing method counts per dataset
  3. log-weighted average within platform by GT cell count
  4. equal-weight average across platforms
Then min-max normalize each metric across methods (worst = 0, best = 1);
bar length = 0.05 + normalized value x 0.9, so the worst bar stays barely visible
and the longest does not touch the right frame.

This figure answers whether a method is uniformly stable on the four key metrics,
or clearly imbalanced. Uneven bars = clear strengths/weaknesses; four even bars = balanced.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42       # TrueType embedding, easier to edit in Illustrator
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
# Paths (replace with your own)
# ==========================================================================
PATH_OMICS = "yourpath/benchmark_result_omics.xlsx"
PATH_IMAGE = "yourpath/benchmark_result_image.xlsx"
OUT_DIR = "yourpath"
OUT_NAME = "robustness_bars"
FILE_FORMAT = "pdf"

# Four shared metrics: (unified column, legend label, direction)
# direction: high=higher is better, low=lower is better, target1=closer to 1 is better
METRICS = [
    ("cell_ratio", "Cell count ratio", "target1"),
    ("precision",  "Precision",        "high"),
    ("recall",     "Recall",           "high"),
    ("f1",         "F1",               "high"),
]

# Same four class colors as the bubble matrix
BAR_COLORS = ["#4490A9", "#ED7D31", "#5B9BD5", "#FFC000"]

# Rank computation
RANK_MODE = "percentile"        # "percentile" / "raw"
WEIGHT_MODE = "log"             # "log" / "linear" / "sqrt" / "equal"

# Bar length = BAR_FLOOR + normalized value x BAR_SPAN
# 0.05 + v*0.9 keeps the shortest bar visible and the longest off the right frame
BAR_FLOOR, BAR_SPAN = 0.05, 0.90

SORT_MODE = "alpha"             # "alpha" alphabetical order / "mean" by four-metric mean descending

# Appearance
# Bar stroke: width 0 = no frame; increase to thicken
BAR_EDGE_COLOR, BAR_EDGE_WIDTH = "#1A1A1A", 0.0
FRAME_COLOR, FRAME_WIDTH = "#1A1A1A", 0.8
# Axes frame: empty tuple = no frame; list sides to keep, e.g. ("left", "top")
FRAME_SIDES = ()
GROUP_H = 1.0                   # vertical height per method (data units)
BAR_FRAC = 0.82                 # fraction of GROUP_H taken by the four bars; rest is inter-group gap
BAR_PACK = 1.0                  # fraction of its slot each bar fills; 1.0 = four bars packed tight
ROW_BG_A, ROW_BG_B = "#F2F2F2", "#FFFFFF"      # alternating gray/white row background

PLOT_W_IN = 3.2                 # plot area width; smaller = narrower figure
GROUP_H_IN = 0.62               # height per method (inches)
LEFT_PAD_IN = 0.26
RIGHT_IN = 0.30
TOP_IN = 0.62                   # top margin for x ticks and axis title
BOTTOM_IN = 0.86                # bottom margin for legend

FS = 12                         # uniform font size
LEG_FS = 11
N_XTICKS = 5
LEGEND_COLSPACING = 3.2         # spacing between legend swatches; too large overflows figure width
LEGEND_HANDLE_LEN = 1.4

DISPLAY_NAMES = {
    "proseg": "ProSeg", "ucs": "UCS", "boms": "BOMS", "baysor": "Baysor",
    "bering": "Bering", "cellist": "Cellist", "comseg": "ComSeg",
    "genesegnet": "GeneSegNet", "dissect": "DISSECT",            # Adapt: added DISSECT
    "cellpose": "Cellpose-SAM", "stardist": "StarDist", "cellsam": "CellSAM",   # Adapt: align with main text
    "mesmer": "Mesmer", "cellotype": "CelloType",
    "cellpose3": "Cellpose-3", "unseg": "UNSEG",                  # Adapt: added two image methods
}

# Adapt: map differing dataset names across the two tables to the omics-table names (keys lowercase).
# Rationale: the two tables' GT cell counts correspond one-to-one (image GT ~1%-3% higher than omics).
DATASET_ALIAS = {
    "cosmx_2_lymph_node":             "cosmx_2",
    "cosmx_lymph_node":               "cosmx_2",    # [New table] Cellpose-3 naming
    "cosmx_human_brain":              "cosmx_3",    # [New table] GT 10,654 vs omics 10,359
    "merfish_mouse_brain":            "merfish_1",  # [New table] GT 8,740, same as merfish_1
    "merfish_liver1":                 "merfish_2",  # [New table] GT 6,566, same as merfish_2
    "merfish_liver2":                 "merfish_3",  # [New table] GT 8,180, same as merfish_3
    "xenium7":                        "xenium_7",   # [New table] missing underscore
    "xenium9":                        "xenium_9",
    "xenium10":                       "xenium_10",
    "merfish_breast_cancer":          "merfish_breast",
    "merfish_lung_cancer":            "merfish_4",  # MERFISH_Lung_cancer also matches here after lowercasing
    "merfish_uterine":                "merfish_uterian",
    "stereo_mouse_embryo_center3000": "stereo_4",
    "xenium_breast":                  "xenium_2",
    "xenium_roi_crop":                "xenium_3",
    "xenium_liver":                   "xenium_4",
    "xenium_mouse_brain":             "xenium_5",
    "xenium_colon":                   "xenium_6",
    "xenium_pancreas_cancer":         "xenium_1",
}
# Adapt: rows under xenium_liver with GT~10,654 are actually CosMx human-brain data
LIVER_FIX_GT, LIVER_FIX_TOL, LIVER_FIX_TO = 10654, 50, "cosmx_human_brain"


# ==========================================================================
# Load and merge the two tables
# ==========================================================================
def normalize_method(name):
    """Strip a trailing "(supervised)"-style suffix; case-insensitive canonical lookup.
    The image table stores both 'Cellpose' and 'cellpose'; without merging they become two methods."""
    base = re.sub(r"\s*[(].*?[)]\s*$", "", str(name)).strip()
    return DISPLAY_NAMES.get(base.lower(), base)


def normalize_dataset(name):
    """Adapt: strip the new/original prefix, take the part before the space, lowercase
    (merfish_Melanoma -> merfish_melanoma), then look up the alias table."""
    s = re.sub(r"^(new|original)[_\-]?", "", str(name).strip())
    s = s.split()[0].lower()
    return DATASET_ALIAS.get(s, s)


def fix_liver(df, gt_col):
    """Adapt: rows under xenium_liver with GT~10,654 belong to CosMx human brain."""
    raw = df["dataset"].astype(str).str.strip().str.split().str[0].str.lower()
    gtv = pd.to_numeric(df[gt_col], errors="coerce")
    m = (raw == "xenium_liver") & ((gtv - LIVER_FIX_GT).abs() <= LIVER_FIX_TOL)
    if m.any():
        print(f"  [Fixed] {int(m.sum())} row(s) in xenium_liver with GT~{LIVER_FIX_GT:,} -> {LIVER_FIX_TO}")
        df.loc[m, "dataset"] = LIVER_FIX_TO
    return df


def load_both(path_omics, path_image):
    xl = pd.ExcelFile(path_omics)
    a = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
    a = a[a["method"].notna() & a["dataset"].notna()].copy()
    a = a.rename(columns={"gt_cells": "_gt"})
    a["_type"] = "rna"

    # The image table needs an explicit sheet; its first sheet holds RNA methods
    b = pd.read_excel(path_image, sheet_name="Image Segmentation Summary")
    b = b[b["method"].notna() & b["dataset"].notna()].copy()
    if "Dice" in b.columns:                       # the table has many placeholder blank rows
        b = b[pd.to_numeric(b["Dice"], errors="coerce").notna()]
    b = b.rename(columns={"Precision": "precision", "Recall": "recall",
                          "F1": "f1", "GT cells": "_gt"})
    # image table has no ready cell_ratio; compute it as predicted cells / GT cells
    b["cell_ratio"] = (pd.to_numeric(b["predicted cells"], errors="coerce")
                       / pd.to_numeric(b["_gt"], errors="coerce").replace(0, np.nan))
    b["_type"] = "image"
    b = fix_liver(b, "_gt")                        # Adapt

    cols = [m[0] for m in METRICS] + ["_gt"]
    keep = ["method", "dataset", "_type"] + cols
    d = pd.concat([a.reindex(columns=keep), b.reindex(columns=keep)],
                  ignore_index=True)
    d["method"] = d["method"].map(normalize_method)
    # Adapt: unify dataset names so the same dataset under two namings merges to one
    d["ds"] = d["dataset"].map(normalize_dataset)
    d["platform"] = d["ds"].str.split("_").str[0]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    agg = d.groupby(["platform", "ds", "method", "_type"], as_index=False)[cols].mean()
    gt = agg.groupby("ds")["_gt"].median()
    n_img = agg.loc[agg["_type"] == "image", "method"].nunique()
    print(f"  After merging: {agg['method'].nunique()} methods"
          f" ({n_img} image, {agg['method'].nunique() - n_img} RNA) | "
          f"{agg['ds'].nunique()} datasets | platforms {sorted(agg['platform'].unique())}")
    # Adapt: check how many datasets have both RNA and image methods ranked together
    both = agg.groupby("ds")["_type"].nunique()
    print(f"  Datasets with both RNA and image methods ranked together: {int((both == 2).sum())}; "
          f"RNA only {sorted(both[both == 1].index[agg.groupby('ds')['_type'].first()[both == 1] == 'rna'])}; "
          f"image only {sorted(both[both == 1].index[agg.groupby('ds')['_type'].first()[both == 1] == 'image'])}")
    return agg, gt


def to_score(s, direction):
    if direction == "high":
        return s
    if direction == "low":
        return -s
    if direction == "target1":          # cell_ratio: closer to 1 is better
        return -(s - 1.0).abs()
    raise ValueError(direction)


def dataset_weight(gt):
    gt = float(gt) if gt is not None else np.nan
    if not np.isfinite(gt) or gt <= 0:
        return None
    if WEIGHT_MODE == "log":
        return float(np.log10(max(gt, 10.0)))   # floor 10 avoids negative log
    if WEIGHT_MODE == "sqrt":
        return float(np.sqrt(gt))
    if WEIGHT_MODE == "equal":
        return 1.0
    return gt


def rank_matrix(agg, gt_cells):
    """Weighted within platform -> equal weight across platforms. RNA and image methods rank together per dataset."""
    methods = sorted(agg["method"].unique())
    out = pd.DataFrame(index=methods, columns=[m[0] for m in METRICS], dtype=float)
    for col, _disp, direction in METRICS:
        plat = {}
        for platform, pdf in agg.groupby("platform"):
            num, den = {}, {}
            for ds, ddf in pdf.groupby("ds"):
                vals = to_score(ddf.set_index("method")[col], direction).dropna()
                n = len(vals)
                if n < 2:                # single method, nothing to compare
                    continue
                rank = vals.rank(ascending=False, method="average")   # rank 1 = best
                norm = ((n - rank + 0.5) / n if RANK_MODE == "percentile"
                        else (n - rank) / max(n - 1, 1))
                w = dataset_weight(gt_cells.get(ds, np.nan))
                if w is None:
                    continue
                for m, v in norm.items():
                    num[m] = num.get(m, 0.0) + w * v
                    den[m] = den.get(m, 0.0) + w
            if den:
                plat[platform] = {m: num[m] / den[m] for m in num}
        for m in methods:
            vs = [p[m] for p in plat.values() if m in p]
            out.loc[m, col] = float(np.mean(vs)) if vs else np.nan
    return out


def normalize_columns(mat):
    """Per-metric min-max normalization across methods: worst = 0, best = 1."""
    out = mat.copy().astype(float)
    for c in out.columns:
        lo, hi = out[c].min(skipna=True), out[c].max(skipna=True)
        out[c] = 0.5 if not np.isfinite(hi - lo) or hi - lo < 1e-12 \
            else (out[c] - lo) / (hi - lo)
    return out


# ==========================================================================
# Plotting
# ==========================================================================
def text_width_in(text, fs):
    if not hasattr(text_width_in, "_fig"):
        text_width_in._fig = plt.figure(figsize=(1, 1), dpi=100)
    f = text_width_in._fig
    t = f.text(0, 0, str(text), fontsize=fs)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    t.remove()
    return w


def draw(norm_mat, out_path):
    rows = (sorted(norm_mat.index, key=str.lower) if SORT_MODE == "alpha"
            else list(norm_mat.mean(axis=1).sort_values(ascending=False).index))
    n_row, n_met = len(rows), len(METRICS)

    left_in = max(text_width_in(s, FS) for s in rows) + LEFT_PAD_IN
    fig_w = left_in + PLOT_W_IN + RIGHT_IN
    fig_h = TOP_IN + n_row * GROUP_H_IN + BOTTOM_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, BOTTOM_IN / fig_h,
                       PLOT_W_IN / fig_w, n_row * GROUP_H_IN / fig_h])

    bar_h = GROUP_H * BAR_FRAC / n_met
    # first method at top; within a group the first metric is also on top
    centers = np.arange(n_row)[::-1] * GROUP_H

    # alternating gray/white row background across the whole plot area
    for i in range(n_row):
        y0 = centers[i] - GROUP_H / 2
        ax.add_patch(plt.Rectangle((0, y0), 1.0, GROUP_H, zorder=0, edgecolor="none",
                                   facecolor=ROW_BG_A if i % 2 == 0 else ROW_BG_B))

    edge_kw = (dict(edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
               if BAR_EDGE_WIDTH > 0 else dict(edgecolor="none", linewidth=0))
    for j, (col, _disp, _d) in enumerate(METRICS):
        # within a group, top-to-bottom: j=0 at top
        offs = GROUP_H * BAR_FRAC / 2 - bar_h * (j + 0.5)
        vals = norm_mat.loc[rows, col].to_numpy(dtype=float)
        lens = BAR_FLOOR + np.nan_to_num(vals, nan=0.0) * BAR_SPAN
        ax.barh(centers + offs, lens, height=bar_h * BAR_PACK,
                color=BAR_COLORS[j % len(BAR_COLORS)], zorder=3, **edge_kw)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(centers.min() - GROUP_H / 2, centers.max() + GROUP_H / 2)
    ax.set_yticks(centers)
    ax.set_yticklabels(rows, fontsize=FS)
    ax.tick_params(axis="y", length=0)

    # put x ticks on top, evenly spaced 0 to 1
    ticks = np.linspace(0, 1, N_XTICKS)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=FS)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labelsize=FS, length=3.5, width=FRAME_WIDTH,
                   direction="out", color=FRAME_COLOR)
    ax.set_xlabel("Normalized rank within each metric   "
                  "(0 = worst method, 1 = best)", fontsize=FS, labelpad=6)
    # note: bar length = 0.05 + normalized x 0.9, so the worst bar stays visible and the longest off the right frame

    ax.grid(True, axis="x", linestyle="--", alpha=0.28, zorder=1)
    ax.set_axisbelow(False)
    for side in ("top", "right", "bottom", "left"):
        keep = side in FRAME_SIDES
        ax.spines[side].set_visible(keep)
        if keep:
            ax.spines[side].set_color(FRAME_COLOR)
            ax.spines[side].set_linewidth(FRAME_WIDTH)

    # legend below the plot; swatch spacing widened but kept within figure width
    handles = [Patch(facecolor=BAR_COLORS[j % len(BAR_COLORS)],
                     label=METRICS[j][1], **edge_kw) for j in range(n_met)]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(left_in / fig_w + PLOT_W_IN / fig_w / 2, 0.012),
               ncol=n_met, frameon=False, fontsize=LEG_FS,
               columnspacing=LEGEND_COLSPACING, handlelength=LEGEND_HANDLE_LEN,
               handletextpad=0.6)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    p_o, p_i, outdir = PATH_OMICS, PATH_IMAGE, OUT_DIR
    if len(sys.argv) == 4:      # only for switching paths while debugging
        p_o, p_i, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)

    agg, gt = load_both(p_o, p_i)
    mat = rank_matrix(agg, gt)
    print("\nRank matrix (0~1, higher is better):")
    print(mat.round(3).to_string())

    nm = normalize_columns(mat)
    print("\nPer-metric normalized (0 = worst method, 1 = best):")
    print(nm.round(3).to_string())
    print("\nFour-metric mean (higher = more evenly good) and range (smaller = more stable):")
    summary = pd.DataFrame({"mean": nm.mean(axis=1), "range": nm.max(axis=1) - nm.min(axis=1)})
    print(summary.round(3).sort_values("mean", ascending=False).to_string())

    draw(nm, os.path.join(outdir, f"{OUT_NAME}.{FILE_FORMAT}"))


if __name__ == "__main__":
    main()
