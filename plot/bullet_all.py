# -*- coding: utf-8 -*-
"""
Method x metric rank bubble matrix (two metric systems)
=======================================================

Figure 1  RNA methods   <- benchmark_result_omics.xlsx / sheet 'Summary'
Figure 2  image segmentation methods <- benchmark_result_image.xlsx / sheet 'Image Segmentation Summary'

Image segmentation methods (Cellpose-SAM / Cellpose-3 / StarDist / CellSAM / Mesmer / CelloType / UNSEG)
live in the "Image Segmentation Summary" sheet, which Figure 2 reads.

The two metric systems differ (RNA methods have no pixel masks, so Dice/IoU/PQ cannot be
computed), but both have 17 metrics in four categories with 4/4/5/4 per category, so the
two figures stack and align vertically.

[Adapt to latest Excel]
  . The same dataset is spelled several ways in the sheet (xenium7 / xenium_7,
    merfish_liver1 / merfish_2, MERFISH_Lung_cancer / merfish_lung_cancer, etc.);
    DATASET_ALIAS unifies them, otherwise one dataset is split and Cellpose-3 / UNSEG get
    skipped on many datasets because only one method remains; platform names are lowercased
    too (MERFISH -> merfish).
  . Rows under xenium_liver with GT = 10,654 are actually CosMx human brain data and are
    reassigned to cosmx_3 (matches nothing if the local data is already fixed).
  . Added display names for DISSECT / Cellpose-3 / UNSEG; Cellpose is shown as Cellpose-SAM.

Computation chain for each cell value:
  1. Within a platform, rank the methods per dataset on that metric (corrected by metric
     direction, rank 1 = best)
  2. Normalize ranks to 0~1 to remove bias from unequal method counts per dataset
  3. Log-weighted average within the platform, weighted by each dataset's GT cell count
  4. Equal-weight average across platforms

Layout: metric names are slanted above the category names, with short leader lines back to
      their columns; cell size is derived from the largest shape actually present in the
      data, leaving only the physical minimum gap so edges never touch; a half-cell gap
      separates the four categories; legend rows use looser spacing; four hues = four
      categories; each column is color-normalized independently, the worst cell taking the
      raw color and the best approaching white (COLOR_SCOPE="global" switches back to
      absolute-score coloring); the round/square exponent is a high power so only the top
      slice near full score looks square, the rest stay round; rows alternate gray/white
      background; every shape has a black outline; a separate legend column on the right.
      All text uses Arial 12.

Run directly with python; set the yourpath placeholders below first.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

# Use Arial throughout. Liberation Sans is metrically identical to Arial as a fallback for
# Linux / no-Arial environments; the layout is identical either way.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42       # embed fonts as TrueType so Illustrator can edit them
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
# Paths (yourpath placeholders)
# ==========================================================================
PATH_OMICS = "yourpath/benchmark_result_omics.xlsx"
PATH_IMAGE = "yourpath/benchmark_result_image.xlsx"
OUT_DIR = "yourpath/supplement"

FILE_FORMAT = "pdf"     # "pdf" / "png" / "svg"


# ==========================================================================
# Key methodology switches
# ==========================================================================
# Datasets have unequal numbers of methods (cosmx_1 has 8 methods in the RNA sheet, some
# datasets only 2). Averaging raw ranks is wrong: rank 2 of 2 (last) and rank 2 of 8 (near
# best) are both the number 2, reversing the conclusion. So ranks are normalized to 0~1
# before averaging by default.
#   "percentile" : normalized rank = (n - rank + 0.5) / n   -- recommended, default
#   "raw"        : raw rank linearly reversed, strictly "rank" as written
RANK_MODE = "percentile"

# Weight within a platform by GT cell count. xenium_1 has 190,965 cells, 78x xenium_3
# (2,442); linear weighting gives it 73% of the xenium platform, while the log scale drops
# it to ~21% (equal weight is 16.7%).
#   "log" / "linear" / "sqrt" / "equal"
WEIGHT_MODE = "log"

SORT_ROWS_BY_SCORE = True   # sort methods by total score descending; False = alphabetical
SHOW_VALUES = False         # print the value in the center of each shape

# Whether to force the four categories of both sheets to share one name set. The two
# figures are aligned by the metric count and color position of each category, not by
# semantics -- the first RNA category is Detection accuracy while the image one is
# Boundary quality, so they do not correspond. This switch must stay False.
UNIFIED_CATEGORY_NAMES = False
UNIFIED_NAMES = ["Detection", "Spatial agreement", "Per-cell fidelity", "Overall quality"]


# ==========================================================================
# Appearance
# ==========================================================================
# Hues for the four categories. The original spec's second value (237,12,549) had 549 out
# of RGB range, so it is treated as the matching Office orange (237,125,49).
CAT_COLORS = ["#4490A9", "#ED7D31", "#5B9BD5", "#FFC000"]
ROW_BG_A, ROW_BG_B = "#EDEDED", "#FFFFFF"                   # alternating gray/white row background
SHAPE_EDGE_COLOR = "#1A1A1A"    # shape outline
SHAPE_EDGE_WIDTH = 1.4

# Color gradient by value: the four CAT_COLORS are the darkest end, used raw (no darkening);
# with COLOR_SCOPE="column", the worst cell in each column takes the category's color, and
# higher scores transition toward white.
GRADIENT = True
# Color value range:
#   "column" normalizes each metric column independently -- the worst cell takes the raw
#            color, the best takes the lightest. Maximum contrast, but colors become
#            relative within a column and are not comparable across columns: one column
#            spanning 0.4~0.6 and another 0.1~0.9 both show the same darkest shade. Size
#            and shape are still absolute, so a "darker but bigger" cell can occur.
#   "global" uses absolute scores 0~1, comparable across columns, but if the lowest score is
#            0.14 the darkest level is never used and the color span is wasted.
COLOR_SCOPE = "column"
# LIGHT_MAX: how close to white the top score gets. 1.0 = pure white; 0.90 keeps about 10%
# of the color, i.e. "very near white with a hint of a pale color".
LIGHT_MAX = 0.90
# LIGHT_GAMMA: gradient curve. 1.0 linear; below 1 the mid-range lightens faster, so high
# scores cluster in a narrow near-white band and low-score differences stand out more.
LIGHT_GAMMA = 0.80

# Shape radius range (inches). Uniform scaling does not make it look tighter -- gaps and
# shapes shrink together. Density is set by the gap / shape-diameter ratio, and the gap
# floor is fixed by the outline width, so enlarging shapes while keeping the outline
# constant shrinks the relative gap and looks more compact.
HALF_MIN_IN, HALF_MAX_IN = 0.077, 0.300

# Gap at the physical floor: the outline is drawn centered on the path and extends half a
# line width outward, so two adjacent shapes need at least "half line width x 2" to not touch.
GAP_EPS_IN = 0.0                                    # extra margin, 0 = outlines just touch
EDGE_HALF_IN = SHAPE_EDGE_WIDTH / 72.0 / 2.0        # half the outline's outward extent
MIN_INDENT_IN = EDGE_HALF_IN + GAP_EPS_IN

# Cell size is derived from the largest value actually present in the data, not the
# theoretical maximum. Full-score cells are rare; reserving for 1.0 wastes a ring of space.
FIT_TO_DATA = True
CELL_IN = 2 * (HALF_MAX_IN + MIN_INDENT_IN)    # fallback used when FIT_TO_DATA=False
ROW_H_IN = CELL_IN

# Superellipse exponent: 2 = circle, larger = closer to a square. A higher SQUARE_POWER
# makes squares appear only in the small band near full score; the rest stay round.
SQUARE_RANGE = (2.0, 8.0)
SQUARE_POWER = 6.0

# Horizontal gap between the four categories, in units of cell size
CAT_GAP_FRAC = 0.5

# Metric names slanted above the category names, short leader lines pointing to each column,
# so column spacing can be minimal
LABEL_ROT = 45                  # metric label slant angle
LEADER_IN = 0.13                # leader line length
LEADER_COLOR, LEADER_WIDTH = "#6B7280", 0.7

# Arial 12 throughout the figure
FS = 12
LABEL_FS = ROW_FS = CAT_FS = TITLE_FS = LEG_FS = FS

RIGHT_IN, BOTTOM_IN = 0.30, 0.35
LEGEND_GAP_IN = 0.34            # whitespace between the metric area and the legend column
LEGEND_TITLE = "Rank"
LEGEND_TICKS = [1.00, 0.75, 0.50, 0.25, 0.00]   # reference values shown in the legend
LEGEND_SPACING = 1.5            # spacing between legend rows, in units of cell size
LEGEND_COLOR = "#595959"   # neutral dark gray for the legend, also fading to white


# ==========================================================================
# Metric systems
# ==========================================================================
# (column name, display abbreviation, category, direction, derived?)
# direction: high=higher is better, low=lower is better, target1=closer to 1 is better
# derived metrics get a * after their abbreviation on the plot

METRICS_RNA = [
    ("cell_ratio",       "Ratio",     "Detection accuracy", "target1", False),
    ("precision",        "Prec",      "Detection accuracy", "high",    False),
    ("recall",           "Rec",       "Detection accuracy", "high",    False),
    ("f1",               "F1",        "Detection accuracy", "high",    False),

    ("loc_mean_um",      "Loc.mean",  "Spatial localization", "low",   False),
    ("loc_median_um",    "Loc.med",   "Spatial localization", "low",   False),
    ("loc_p95_um",       "Loc.p95",   "Spatial localization", "low",   False),
    ("loc_tail_ratio",   "Loc.tail",  "Spatial localization", "low",   True),

    ("count_pearson",    "Cnt.r",     "Count fidelity", "high",  False),
    ("count_spearman",   "Cnt.rho",   "Count fidelity", "high",  False),
    ("count_mae",        "Cnt.MAE",   "Count fidelity", "low",   False),
    ("count_rmse",       "Cnt.RMSE",  "Count fidelity", "low",   False),
    ("count_tail_ratio", "Cnt.tail",  "Count fidelity", "low",   True),

    ("vec_cosine",       "Exp.cos",   "Expression & assignment", "high", False),
    ("vec_js_dist",      "Exp.JS",    "Expression & assignment", "low",  False),
    ("vec_pearson",      "Exp.r",     "Expression & assignment", "high", False),
    ("assign_accuracy",  "Assign",    "Expression & assignment", "high", False),
]

# Order: Boundary quality is moved first and Instance detection third, so each category has
# 4/4/5/4 metrics, exactly matching the RNA sheet and letting the two figures stack/align.
# Category colors are taken from CAT_COLORS in order of appearance, so moving positions moves
# colors too, and positions 1~4 match the RNA sheet's colors.
# Note: alignment is by metric count and color position per category, not semantics -- the
# first RNA category is Detection accuracy while the image one is Boundary quality, so
# UNIFIED_CATEGORY_NAMES must stay False or they get mismatched.
METRICS_IMG = [
    ("Boundary F1",        "B.F1",    "Boundary quality", "high", False),
    ("Boundary Precision", "B.Prec",  "Boundary quality", "high", False),
    ("Boundary Recall",    "B.Rec",   "Boundary quality", "high", False),
    ("SQ",                 "SQ",      "Boundary quality", "high", False),

    ("Dice",               "Dice",    "Region overlap", "high", False),
    ("IoU",                "IoU",     "Region overlap", "high", False),
    ("Pixel Accuracy",     "PixAcc",  "Region overlap", "high", False),
    ("AJI",                "AJI",     "Region overlap", "high", False),

    ("F1",                 "F1",      "Instance detection", "high",    False),
    ("Precision",          "Prec",    "Instance detection", "high",    False),
    ("Recall",             "Rec",     "Instance detection", "high",    False),
    ("DQ",                 "DQ",      "Instance detection", "high",    False),
    ("cell_ratio",         "Ratio",   "Instance detection", "target1", True),

    ("PQ",                 "PQ",      "Overall quality", "high", False),
    ("mDA(AP0.5:0.95)",    "mDA",     "Overall quality", "high", False),
    ("over_seg_rate",      "Over",    "Overall quality", "low",  True),
    ("under_seg_rate",     "Under",   "Overall quality", "low",  True),
]


# ==========================================================================
# Data sources
# ==========================================================================
SOURCES = [
    dict(
        label="RNA-based methods",
        path=PATH_OMICS,
        sheet=None,                 # None = first sheet
        gt_col="gt_cells",
        metrics=METRICS_RNA,
        out="bubble_full_omics",
        # Treat exact 0 as missing: 0 um localization error is physically impossible
        # (ComSeg on xenium_3 has loc_mean/median/p95 all 0); without cleaning it gets
        # ranked as the most accurate localization method
        zero_is_missing=["loc_mean_um", "loc_median_um", "loc_p95_um"],
        check_f1=True,
    ),
    dict(
        label="Image-based segmentation methods",
        path=PATH_IMAGE,
        sheet="Image Segmentation Summary",
        gt_col="GT cells",
        metrics=METRICS_IMG,
        out="bubble_result_image",
        zero_is_missing=[],
        check_f1=False,
    ),
]

# Unify method names: case-insensitive dedup + canonical display names. In the image sheet
# 'Cellpose' and 'cellpose' are the same method stored twice and would become two rows if
# not merged; likewise 'ProSeg(supervised)' and 'ProSeg' in the RNA sheet.
DISPLAY_NAMES = {
    "proseg": "ProSeg", "ucs": "UCS", "boms": "BOMS", "baysor": "Baysor",
    "bering": "Bering", "cellist": "Cellist", "comseg": "ComSeg",
    "genesegnet": "GeneSegNet", "dissect": "DISSECT",            # Adapt: added DISSECT
    "cellpose": "Cellpose-SAM", "stardist": "StarDist", "cellsam": "CellSAM",   # Adapt: match the paper text
    "mesmer": "Mesmer", "cellotype": "CelloType",
    "cellpose3": "Cellpose-3", "unseg": "UNSEG",                  # Adapt: added two image methods
}

# Adapt: map different spellings of the same dataset to one name (keys lowercase; unified to
# the omics sheet naming so all three figures share the same scope)
DATASET_ALIAS = {
    "cosmx_2_lymph_node":             "cosmx_2",
    "cosmx_lymph_node":               "cosmx_2",
    "cosmx_human_brain":              "cosmx_3",
    "merfish_mouse_brain":            "merfish_1",
    "merfish_liver1":                 "merfish_2",
    "merfish_liver2":                 "merfish_3",
    "merfish_lung_cancer":            "merfish_4",
    "merfish_breast_cancer":          "merfish_breast",
    "merfish_uterine":                "merfish_uterian",
    "stereo_mouse_embryo_center3000": "stereo_4",
    "xenium7":                        "xenium_7",
    "xenium9":                        "xenium_9",
    "xenium10":                       "xenium_10",
    "xenium_pancreas_cancer":         "xenium_1",
    "xenium_breast":                  "xenium_2",
    "xenium_roi_crop":                "xenium_3",
    "xenium_liver":                   "xenium_4",
    "xenium_mouse_brain":             "xenium_5",
    "xenium_colon":                   "xenium_6",
}
# Adapt: rows under xenium_liver with GT~10,654 are actually CosMx human brain data
LIVER_FIX_GT, LIVER_FIX_TOL, LIVER_FIX_TO = 10654, 50, "cosmx_human_brain"


# ==========================================================================
# Text width measurement (basis of the header layout)
# ==========================================================================
_SCRATCH = None


def text_width_in(text, fs):
    """Measure the rendered width of this text in inches."""
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = plt.figure(figsize=(1, 1), dpi=100)
    t = _SCRATCH.text(0, 0, str(text), fontsize=fs)
    _SCRATCH.canvas.draw()
    bb = t.get_window_extent(renderer=_SCRATCH.canvas.get_renderer())
    t.remove()
    return bb.width / _SCRATCH.dpi


# ==========================================================================
# Data preparation
# ==========================================================================
def normalize_method(name):
    """Strip bracketed suffixes like (supervised), then look up the canonical name by lowercase."""
    base = re.sub(r"\s*[(].*?[)]\s*$", "", str(name)).strip()
    return DISPLAY_NAMES.get(base.lower(), base)


def normalize_dataset(name):
    """Adapt: strip the new/original prefix, take the part before the space, lowercase, then look up the alias table."""
    s = re.sub(r"^(new|original)[_\-]?", "", str(name).strip())
    s = s.split()[0].lower()
    return DATASET_ALIAS.get(s, s)


def fix_liver(df, gt_col):
    """Adapt: reassign xenium_liver rows with GT~10,654 to the CosMx human brain."""
    raw = df["dataset"].astype(str).str.strip().str.split().str[0].str.lower()
    gtv = pd.to_numeric(df[gt_col], errors="coerce")
    m = (raw == "xenium_liver") & ((gtv - LIVER_FIX_GT).abs() <= LIVER_FIX_TOL)
    if m.any():
        print(f"  [Fixed] xenium_liver rows with GT~{LIVER_FIX_GT:,}: {int(m.sum())} -> {LIVER_FIX_TO}")
        df.loc[m, "dataset"] = LIVER_FIX_TO
    return df


def load_source(src):
    xl = pd.ExcelFile(src["path"])
    sheet = src["sheet"] or xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet)

    metrics = src["metrics"]
    native = [m[0] for m in metrics if not m[4] and m[0] in df.columns]
    # the image sheet has many placeholder empty rows; require at least one native metric value
    df = df[df["method"].notna() & df["dataset"].notna()].copy()
    if native:
        df = df[df[native].notna().any(axis=1)]

    df["method"] = df["method"].map(normalize_method)
    # Adapt: unify dataset and platform names
    df = fix_liver(df, src["gt_col"])
    df["ds"] = df["dataset"].map(normalize_dataset)
    df["platform"] = df["ds"].str.split("_").str[0]

    for col in src["zero_is_missing"]:
        if col in df.columns:
            n = int((df[col] == 0).sum())
            if n:
                who = df.loc[df[col] == 0, "method"].value_counts().to_dict()
                print(f"  [Clean] {col}: {n} zeros treated as missing -> {who}")
                df.loc[df[col] == 0, col] = np.nan

    # f1=0 while precision/recall are both >0 is arithmetically impossible: blank, not truly 0
    if src["check_f1"] and {"f1", "precision", "recall"} <= set(df.columns):
        bad = (df["f1"] == 0) & (df["precision"] > 0) & (df["recall"] > 0)
        if bad.any():
            print(f"  [Clean] f1: {int(bad.sum())} zeros inconsistent with precision/recall treated as missing")
            df.loc[bad, "f1"] = np.nan

    # auxiliary columns needed for derived metrics
    extra = [c for c in ["predicted cells", "TP", "FP", "FN", "loc_p95_um", "loc_median_um",
                         "count_rmse", "count_mae"] if c in df.columns]
    keep = sorted(set(native + extra + [src["gt_col"]]))
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # columns like 'SQ' come in as object dtype

    agg = df.groupby(["platform", "ds", "method"], as_index=False)[keep].mean()
    agg = derive_metrics(agg, src)
    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg = agg.rename(columns={src["gt_col"]: "_gt"})

    gt = agg.groupby("ds")["_gt"].median()
    return agg, gt, sheet


def derive_metrics(agg, src):
    """Compute derived metrics on demand, using only columns already present."""
    gt = src["gt_col"]
    cols = set(agg.columns)

    # RNA: tail robustness
    if {"loc_p95_um", "loc_median_um"} <= cols:
        agg["loc_tail_ratio"] = agg["loc_p95_um"] / agg["loc_median_um"].replace(0, np.nan)
    if {"count_rmse", "count_mae"} <= cols:
        agg["count_tail_ratio"] = agg["count_rmse"] / agg["count_mae"].replace(0, np.nan)

    # Image: predicted/GT cell ratio, plus over- and under-segmentation rates. The sheet's
    # "over-segmentation (FP)" / "under-segmentation (FN)" columns are raw FP/FN counts that
    # span orders of magnitude across datasets; use the clean TP/FP/FN integer columns to
    # compute rates with a clear meaning.
    if {"predicted cells", gt} <= cols:
        agg["cell_ratio"] = agg["predicted cells"] / agg[gt].replace(0, np.nan)
    if {"FP", gt} <= cols:
        agg["over_seg_rate"] = agg["FP"] / agg[gt].replace(0, np.nan)
    if {"FN", gt} <= cols:
        agg["under_seg_rate"] = agg["FN"] / agg[gt].replace(0, np.nan)

    return agg


def to_score(series, direction):
    """Convert to a higher-is-better comparable quantity for ranking."""
    if direction == "high":
        return series
    if direction == "low":
        return -series
    if direction == "target1":       # cell_ratio: closer to 1 is better
        return -(series - 1.0).abs()
    raise ValueError(direction)


def dataset_weight(gt):
    """Convert GT cell count to a weight per WEIGHT_MODE; return None for invalid values."""
    gt = float(gt) if gt is not None else np.nan
    if not np.isfinite(gt) or gt <= 0:
        return None
    if WEIGHT_MODE == "log":
        return float(np.log10(max(gt, 10.0)))   # floor of 10 avoids a negative log
    if WEIGHT_MODE == "sqrt":
        return float(np.sqrt(gt))
    if WEIGHT_MODE == "equal":
        return 1.0
    return gt


def report_weights(agg, gt_cells):
    """Print each dataset's weight within its platform to confirm none dominates."""
    print(f"\n  Weight distribution (WEIGHT_MODE={WEIGHT_MODE}):")
    for platform, pdf in agg.groupby("platform"):
        items = []
        for ds in sorted(pdf["ds"].unique()):
            w = dataset_weight(gt_cells.get(ds, np.nan))
            if w is not None:
                items.append((ds, w, float(gt_cells.get(ds))))
        total = sum(w for _, w, _ in items)
        if not total:
            continue
        txt = "  ".join(f"{ds}({g:,.0f}) {w / total:5.1%}" for ds, w, g in items)
        print(f"    {platform:<8}{txt}")


def rank_matrix(agg, gt_cells, metrics):
    """Weight within platform, then equal-weight across platforms, giving the method x metric matrix."""
    methods = sorted(agg["method"].unique())
    out = pd.DataFrame(index=methods, columns=[m[0] for m in metrics], dtype=float)

    for col, _disp, _cat, direction, _der in metrics:
        if col not in agg.columns:
            continue
        plat_scores = {}

        for platform, pdf in agg.groupby("platform"):
            num, den = {}, {}
            for ds, ddf in pdf.groupby("ds"):
                vals = to_score(ddf.set_index("method")[col], direction).dropna()
                n = len(vals)
                if n < 2:            # only one method, nothing to compare
                    continue
                rank = vals.rank(ascending=False, method="average")   # rank 1 = best
                if RANK_MODE == "percentile":
                    norm = (n - rank + 0.5) / n
                else:
                    norm = (n - rank) / max(n - 1, 1)
                w = dataset_weight(gt_cells.get(ds, np.nan))
                if w is None:
                    continue
                for m, v in norm.items():
                    num[m] = num.get(m, 0.0) + w * v
                    den[m] = den.get(m, 0.0) + w
            if den:
                plat_scores[platform] = {m: num[m] / den[m] for m in num}

        for m in methods:
            vals = [ps[m] for ps in plat_scores.values() if m in ps]
            out.loc[m, col] = float(np.mean(vals)) if vals else np.nan

    return out


# ==========================================================================
# Plotting
# ==========================================================================
def superellipse(cx, cy, half, n_exp, k=180):
    """n_exp=2 is a circle; larger values approach a square."""
    t = np.linspace(0, 2 * np.pi, k)
    ct, st = np.cos(t), np.sin(t)
    x = np.sign(ct) * np.abs(ct) ** (2.0 / n_exp)
    y = np.sign(st) * np.abs(st) ** (2.0 / n_exp)
    return np.column_stack([cx + half * x, cy + half * y])


def shade(base_hex, v):
    """v=0 gives the CAT_COLORS hex itself (no extra darkening); v=1 fades toward white.
    With COLOR_SCOPE="column", the worst cell in each column takes exactly the color you set."""
    from matplotlib.colors import to_rgb
    rgb = np.asarray(to_rgb(base_hex), dtype=float)
    if not GRADIENT:
        return tuple(rgb)
    t = float(np.clip(v, 0, 1)) ** LIGHT_GAMMA * LIGHT_MAX
    return tuple(rgb + (1.0 - rgb) * t)


def shape_of(v):
    """Score -> (radius in inches, superellipse exponent). The high power makes squares appear only near full score."""
    v = float(np.clip(v, 0, 1))
    s_lo, s_hi = SQUARE_RANGE
    return (HALF_MIN_IN + v * (HALF_MAX_IN - HALF_MIN_IN),
            s_lo + (v ** SQUARE_POWER) * (s_hi - s_lo))


def draw(matrix, metrics, title, out_path):
    """Axes are in inches (aspect=equal); the header is drawn in the same coordinates for precise placement."""
    cats = list(dict.fromkeys(m[2] for m in metrics))
    cat_color = {c: CAT_COLORS[i % len(CAT_COLORS)] for i, c in enumerate(cats)}
    rows = list(matrix.index)
    n_col, n_row = len(metrics), len(rows)

    # --- cell size: set by the largest shape actually in the data; gap only the outline's physical floor ---
    if FIT_TO_DATA:
        vmax = float(np.nanmax(matrix.to_numpy(dtype=float))) if matrix.notna().any().any() else 1.0
        vmax = float(np.clip(vmax, 0.0, 1.0))
        half_top = HALF_MIN_IN + vmax * (HALF_MAX_IN - HALF_MIN_IN)
        cell = 2 * (half_top + MIN_INDENT_IN)
    else:
        cell = CELL_IN
    col_w = row_h = cell

    # compute each column's left edge; insert CAT_GAP_FRAC cells at category boundaries
    col_x, x_cur, prev_cat = [], 0.0, None
    for m in metrics:
        if prev_cat is not None and m[2] != prev_cat:
            x_cur += cell * CAT_GAP_FRAC
        col_x.append(x_cur)
        x_cur += cell
        prev_cat = m[2]
    grid_w = x_cur
    grid_h = n_row * row_h
    print(f"  cell {cell:.4f} in | max shape diameter {2*(cell/2-MIN_INDENT_IN):.4f} "
          f"| relative gap {2*MIN_INDENT_IN/(cell-2*MIN_INDENT_IN):.1%}")

    # --- header layer heights (bottom-up: color bar -> category name -> leader -> slanted metric name) ---
    rot = np.deg2rad(LABEL_ROT)
    bar_gap, bar_h = 0.06, 0.14
    cat_h = CAT_FS * 1.35 / 72.0
    y_bar0 = grid_h + bar_gap
    y_cat = y_bar0 + bar_h + 0.05                     # category name baseline
    y_lead0 = y_cat + cat_h + 0.06                    # leader line origin
    lab_len = max(text_width_in(m[1] + (" *" if m[4] else ""), LABEL_FS)
                  for m in metrics)
    head_h = (y_lead0 + LEADER_IN * np.sin(rot)
              + lab_len * np.sin(rot) + 0.10) - grid_h

    # --- legend column ---
    leg_num_w = max(text_width_in(f"{t:.2f}", LEG_FS) for t in LEGEND_TICKS)
    leg_w = 2 * HALF_MAX_IN + 0.10 + leg_num_w + 0.06
    # the rightmost slanted metric label extends right; the legend must clear that
    leg_x0 = grid_w + max(LEGEND_GAP_IN, lab_len * np.cos(rot) * 0.55)

    left_in = max(text_width_in(m, ROW_FS) for m in rows) + 0.22
    title_h = TITLE_FS * 1.6 / 72.0
    plot_w = leg_x0 + leg_w
    fig_w = left_in + plot_w + RIGHT_IN
    fig_h = BOTTOM_IN + grid_h + head_h + title_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, BOTTOM_IN / fig_h,
                       plot_w / fig_w, (grid_h + head_h) / fig_h])
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, grid_h + head_h)
    ax.set_aspect("equal")          # 1 data unit = 1 inch, so shapes are not stretched
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # --- color value range: record each column's min/max when normalizing per column ---
    if COLOR_SCOPE == "column":
        cmin = matrix.min(axis=0, skipna=True)
        cmax = matrix.max(axis=0, skipna=True)

    def color_v(col, v):
        """Return the 0~1 value used for coloring. 0 -> raw color, 1 -> near white."""
        if COLOR_SCOPE != "column" or col not in matrix.columns:
            return v
        lo, hi = float(cmin[col]), float(cmax[col])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return 0.5
        return (v - lo) / (hi - lo)

    def row_y(i):
        return (n_row - 1 - i) * row_h + row_h / 2

    # --- alternating gray/white row background, only over the metric area ---
    for i in range(n_row):
        ax.add_patch(Rectangle((0, (n_row - 1 - i) * row_h), grid_w, row_h,
                               zorder=0, edgecolor="none",
                               facecolor=ROW_BG_A if i % 2 == 0 else ROW_BG_B))

    # --- each cell's shape, centered in its column; the largest fills the cell, leaving only the outline floor ---
    for i, method in enumerate(rows):
        y = row_y(i)
        for j, (col, _disp, cat, _d, _der) in enumerate(metrics):
            x = col_x[j] + col_w / 2
            v = matrix.loc[method, col] if col in matrix.columns else np.nan
            if not np.isfinite(v):
                ax.plot([x - 0.07, x + 0.07], [y, y], color="#9E9E9E",
                        linewidth=1.1, zorder=2)
                continue
            half, n_exp = shape_of(v)          # size and roundness always use the absolute score
            ax.add_patch(Polygon(superellipse(x, y, half, n_exp), closed=True,
                                 facecolor=shade(cat_color[cat], color_v(col, v)),
                                 edgecolor=SHAPE_EDGE_COLOR,
                                 linewidth=SHAPE_EDGE_WIDTH, zorder=2))
            if SHOW_VALUES:
                ax.text(x, y, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.2, color="#333333", zorder=3)

    # --- method names ---
    for i, method in enumerate(rows):
        ax.text(-0.10, row_y(i), method, ha="right", va="center",
                fontsize=ROW_FS, color="#1A1A1A")

    # --- category color bars + names, just above the grid ---
    j = 0
    while j < n_col:
        cat = metrics[j][2]
        k = j
        while k + 1 < n_col and metrics[k + 1][2] == cat:
            k += 1
        x0, x1 = col_x[j], col_x[k] + col_w
        ax.add_patch(Rectangle((x0 + 0.006, y_bar0), (x1 - x0) - 0.012, bar_h,
                               facecolor=cat_color[cat], edgecolor="none", zorder=2))
        ax.text((x0 + x1) / 2, y_cat, cat, ha="center", va="bottom",
                fontsize=CAT_FS, color="#1A1A1A")
        j = k + 1

    # --- metric names: slanted above the category names, short leader lines back to each column ---
    for j, (_c, disp, _cat, _d, derived) in enumerate(metrics):
        xc = col_x[j] + col_w / 2
        x_end = xc + LEADER_IN * np.cos(rot)
        y_end = y_lead0 + LEADER_IN * np.sin(rot)
        ax.plot([xc, x_end], [y_lead0, y_end], color=LEADER_COLOR,
                linewidth=LEADER_WIDTH, solid_capstyle="round", zorder=2)
        ax.text(x_end + 0.015 * np.cos(rot), y_end + 0.015 * np.sin(rot),
                disp + (" *" if derived else ""), rotation=LABEL_ROT,
                rotation_mode="anchor", ha="left", va="center",
                fontsize=LABEL_FS, color="#1A1A1A")

    # --- legend column ---
    ax.text(leg_x0 + leg_w / 2, y_bar0, LEGEND_TITLE, ha="center", va="bottom",
            fontsize=LABEL_FS, color="#1A1A1A")
    # legend rows use looser spacing and are vertically centered within the grid height
    n_leg = len(LEGEND_TICKS)
    room = (grid_h - row_h) / max(n_leg - 1, 1)      # keep first/last inside the grid
    leg_h = min(row_h * LEGEND_SPACING, room)
    y_leg_top = grid_h / 2 + (n_leg - 1) * leg_h / 2
    for k2, tv in enumerate(LEGEND_TICKS):
        y = y_leg_top - k2 * leg_h
        half, n_exp = shape_of(tv)
        ax.add_patch(Polygon(superellipse(leg_x0 + HALF_MAX_IN, y, half, n_exp),
                             closed=True, facecolor=shade(LEGEND_COLOR, tv),
                             edgecolor=SHAPE_EDGE_COLOR,
                             linewidth=SHAPE_EDGE_WIDTH, zorder=2))
        ax.text(leg_x0 + 2 * HALF_MAX_IN + 0.07, y, f"{tv:.2f}",
                ha="left", va="center", fontsize=LEG_FS, color="#4B5563")

    fig.text(left_in / fig_w - 0.004, 1 - 0.06 / fig_h, title,
             ha="left", va="top", fontsize=TITLE_FS, color="#1A1A1A")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==========================================================================
def apply_unified_names(metrics):
    """Replace each sheet's category names with UNIFIED_NAMES in order of appearance."""
    if not UNIFIED_CATEGORY_NAMES:
        return metrics
    order = list(dict.fromkeys(m[2] for m in metrics))
    mapping = {c: UNIFIED_NAMES[i] for i, c in enumerate(order) if i < len(UNIFIED_NAMES)}
    return [(c, d, mapping.get(cat, cat), dr, dv) for c, d, cat, dr, dv in metrics]


def run(src, outdir):
    print(f"\n===== {src['label']} =====")
    agg, gt, sheet = load_source(src)
    print(f"  sheet: {sheet} | {agg['method'].nunique()} methods | "
          f"{agg['ds'].nunique()} datasets | platforms {sorted(agg['platform'].unique())}")
    print(f"  Methods: {sorted(agg['method'].unique())}")

    metrics = apply_unified_names(src["metrics"])
    report_weights(agg, gt)
    mat = rank_matrix(agg, gt, metrics)
    if SORT_ROWS_BY_SCORE:
        mat = mat.loc[mat.mean(axis=1, skipna=True).sort_values(ascending=False).index]
    else:
        mat = mat.loc[sorted(mat.index, key=str.lower)]

    print("\n  Rank matrix (0~1, higher is better):")
    print(mat.round(3).to_string())
    n_na = int(mat.isna().sum().sum())
    if n_na:
        print(f"  {n_na} missing cells, drawn as short gray dashes")

    cov = agg.groupby("method")["ds"].nunique()
    print("  Datasets covered per method:", cov.to_dict())

    draw(mat, metrics,
         f"{src['label']}  -  composite rank across platforms"
         f"   (larger & squarer = better rank)",
         os.path.join(outdir, f"{src['out']}.{FILE_FORMAT}"))
    return mat


def main():
    outdir = OUT_DIR
    if len(sys.argv) == 4:      # only for overriding paths when debugging; normally run without args
        SOURCES[0]["path"], SOURCES[1]["path"], outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    for src in SOURCES:
        run(src, outdir)


if __name__ == "__main__":
    main()
