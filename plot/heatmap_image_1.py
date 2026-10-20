#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image segmentation benchmark normalized heatmap
==========================================================
Data source: the "Image Segmentation Summary" sheet in the workbook
Rows  = method x dataset (one row per method-dataset combination)
Cols  = metrics (six core metrics: Dice / IoU / PQ / AJI / F1 / Boundary F1)
Color = the metric value normalized per column (light red = better, light blue = worse)
In-cell number = raw metric value (only the color is the normalized result)
No borders, no legend. Missing cells are light gray and unannotated.

Usage (no args needed; reads/writes the yourpath files, outputs next to the input):
    python plot_seg_heatmap.py
    python plot_seg_heatmap.py --norm minmax
    python plot_seg_heatmap.py --row-order dataset
    python plot_seg_heatmap.py --metrics Dice,IoU,PQ,AJI,F1,Boundary F1
    python plot_seg_heatmap.py --add-counts          # also plot count columns (TP/FP/FN/over-seg...)
    python plot_seg_heatmap.py --annot-normalized    # show normalized values in cells
    python plot_seg_heatmap.py --colorbar            # add when a colorbar is needed
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 0. Input/output paths (yourpath placeholders) + sheet
# ============================================================
DEFAULT_XLSX = "yourpath/benchmark_full_all.xlsx"
DEFAULT_OUTDIR = os.path.dirname(DEFAULT_XLSX)     # output next to the input
DEFAULT_SHEET = "Image Segmentation Summary"

# Metric display order (plotted only if present and non-null). Six core, all higher-is-better quality metrics.
QUALITY_METRICS = [
    "Dice", "IoU", "PQ", "AJI", "F1", "Boundary F1",
]
# Count columns (scale with dataset size; excluded by default, added with --add-counts).
# Among these, FP/FN/over/under are lower-is-better.
COUNT_METRICS = ["over-segmentation (FP)", "under-segmentation (FN)", "TP", "FP", "FN"]

# lower-is-better -> flip after normalizing; append ↓ to the column label
LOWER_IS_BETTER = {"over-segmentation (FP)", "under-segmentation (FN)", "FP", "FN"}
# no absolute direction (cell_ratio is closer-to-1 better; here colored by raw value, not flipped)
NEUTRAL = {"cell_ratio"}

# non-metric columns (never treated as plottable metrics)
NON_METRIC_COLS = {"method", "dataset", "reference", "mDA_note",
                   "GT cells", "predicted cells", "gt_cells", "pred_cells"}

# method name canonicalization: unify case, keep suffixes like (+15px)
METHOD_CANON = {
    "cellpose": "Cellpose", "stardist": "StarDist", "cellsam": "CellSAM",
    "mesmer": "Mesmer", "cellotype": "CelloType", "celltype": "CelloType",
}

PLATFORM_ORDER = ["xenium", "merfish", "cosmx", "stereo"]


# ============================================================
# 1. Font setup
# ============================================================
def setup_font(user_font: str | None = None) -> str:
    prefer = [user_font] if user_font else []
    prefer += ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
               "Source Han Sans SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
               "Arial Unicode MS", "DejaVu Sans"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in prefer if f and f in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen] + [f for f in prefer if f and f in installed]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ============================================================
# 2. Read data / canonicalize names
# ============================================================
def load_sheet(xlsx: str, sheet: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx)
    if sheet not in xls.sheet_names:
        raise SystemExit(f"[Error] sheet {sheet!r} not found. Available: {xls.sheet_names}")
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def canon_method(m: str) -> str:
    s = str(m).strip()
    mm = re.match(r"^([A-Za-z]+)(.*)$", s)
    if not mm:
        return s
    stem, rest = mm.group(1), mm.group(2)
    return (METHOD_CANON.get(stem.lower(), stem) + rest).strip()


def dataset_base(name: str) -> str:
    m = re.match(r"^([a-zA-Z]+_\d+)", str(name).strip())
    return m.group(1) if m else str(name).strip()


def platform_of(base: str) -> str:
    return base.split("_")[0].lower()


def ds_sort_key(base: str):
    plat = platform_of(base)
    pi = PLATFORM_ORDER.index(plat) if plat in PLATFORM_ORDER else len(PLATFORM_ORDER)
    m = re.search(r"_(\d+)", base)
    return (pi, plat, int(m.group(1)) if m else 99, base)


# ============================================================
# 3. Cleaning / deduplication
# ============================================================
def tidy(df: pd.DataFrame, dedup: str) -> pd.DataFrame:
    df = df.copy()
    for need in ("method", "dataset"):
        if need not in df.columns:
            raise SystemExit(f"[Error] missing column {need!r}. Actual: {list(df.columns)}")
    df["method"] = df["method"].map(canon_method)
    df["ds_base"] = df["dataset"].map(dataset_base)
    df["platform"] = df["ds_base"].map(platform_of)

    # labels: for each base, keep the longest (most informative) dataset spelling
    var_label: dict[str, str] = {}
    for raw in df["dataset"].astype(str):
        b = dataset_base(raw)
        if len(raw.strip()) > len(var_label.get(b, "")):
            var_label[b] = raw.strip()
    df["ds_label"] = df["ds_base"].map(lambda b: var_label.get(b, b))

    # cell_ratio = predicted/GT
    gt = pd.to_numeric(df.get("GT cells"), errors="coerce") if "GT cells" in df else pd.Series(np.nan, index=df.index)
    pred = pd.to_numeric(df.get("predicted cells"), errors="coerce") if "predicted cells" in df else pd.Series(np.nan, index=df.index)
    df["cell_ratio"] = pred / gt.replace(0, np.nan)

    # convert all candidate metrics to numeric
    for c in QUALITY_METRICS + COUNT_METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if dedup != "none":
        present = [c for c in (QUALITY_METRICS + COUNT_METRICS + ["cell_ratio"]) if c in df.columns]
        df["__nc__"] = df[present].notna().sum(axis=1)
        df = (df.sort_values("__nc__", ascending=False)
                .drop_duplicates(["method", "ds_base"], keep="first")
                .drop(columns="__nc__"))
    return df


# ============================================================
# 4. Normalization (per column)
# ============================================================
def normalize_col(s: pd.Series, how: str, lower_better: bool) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if len(valid) == 0:
        return out
    if len(valid.unique()) == 1:
        out.loc[valid.index] = 0.5
    elif how == "rank":
        out.loc[valid.index] = valid.rank(pct=True)          # percentile rank (0,1]
    elif how == "minmax":
        lo, hi = valid.min(), valid.max()
        out.loc[valid.index] = (valid - lo) / (hi - lo)
    elif how == "robust":
        lo, hi = valid.quantile(0.02), valid.quantile(0.98)
        if hi == lo:
            lo, hi = valid.min(), valid.max()
        out.loc[valid.index] = ((valid - lo) / (hi - lo)).clip(0, 1)
    elif how == "zscore":
        mu, sd = valid.mean(), valid.std(ddof=0)
        z = (valid - mu) / (sd if sd else 1.0)
        out.loc[valid.index] = (z.clip(-2, 2) + 2) / 4
    else:
        raise SystemExit(f"[Error] unknown normalization method {how!r}")
    if lower_better:
        out = 1.0 - out
    return out


# ============================================================
# 5. Draw heatmap
# ============================================================
def make_cmap(name: str | None):
    if name:
        return plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        "light_rb",
        ["#a7c7e0", "#cfe0ee", "#f2f1ee", "#f6ded9", "#e8a6a6"])


def fmt_val(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def draw(df: pd.DataFrame, metrics: list[str], args) -> plt.Figure:
    if args.row_order == "dataset":
        df = df.sort_values(by=["ds_base", "method"],
                            key=lambda col: col.map(ds_sort_key) if col.name == "ds_base" else col)
    else:
        df = df.sort_values(by=["method", "ds_base"],
                            key=lambda col: col.map(ds_sort_key) if col.name == "ds_base" else col)
    df = df.reset_index(drop=True)

    row_labels = [f"{m} × {d}" for m, d in zip(df["method"], df["ds_label"])]
    col_labels = [c + (" ↓" if c in LOWER_IS_BETTER else "") for c in metrics]

    norm_mat = np.full((len(df), len(metrics)), np.nan)
    raw_mat = np.full((len(df), len(metrics)), np.nan)
    for j, c in enumerate(metrics):
        raw = pd.to_numeric(df[c], errors="coerce")
        nrm = normalize_col(raw, args.norm, c in LOWER_IS_BETTER and c not in NEUTRAL)
        raw_mat[:, j] = raw.values
        norm_mat[:, j] = nrm.values

    masked = np.ma.masked_invalid(norm_mat)
    cmap = make_cmap(args.cmap).copy()
    cmap.set_bad(color="#e6e6e6")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig_w = max(args.fig_w, 3.4 + args.cell_w * len(metrics))
    fig_h = max(args.fig_h, 1.6 + args.cell_h * len(df))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)

    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest", origin="upper")

    ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(col_labels, rotation=42, ha="left", fontsize=args.col_fs)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for t, c in zip(ax.get_xticklabels(), metrics):
        if c in LOWER_IS_BETTER:
            t.set_color("#7a4a4a")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(row_labels, fontsize=args.row_fs)

    for s in ax.spines.values():
        s.set_visible(False)

    if args.annot:
        for i in range(len(df)):
            for j in range(len(metrics)):
                v_raw, v_nrm = raw_mat[i, j], norm_mat[i, j]
                if np.isnan(v_nrm):
                    continue
                txt = f"{v_nrm:.2f}" if args.annot_normalized else fmt_val(v_raw)
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=args.annot_fs, color="#242424")

    if args.colorbar:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.018, pad=0.01)
        cb.set_label("normalized per column (light red = better / light blue = worse)", fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8)

    fig.tight_layout()
    return fig


# ============================================================
# 6. main
# ============================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Image segmentation benchmark normalized heatmap: rows=method x dataset, cols=six core metrics, light red/light blue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--xlsx", default=DEFAULT_XLSX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--prefix", default="heatmap_seg")
    p.add_argument("--metrics", default=None,
                   help="metrics to show, comma-separated; default is the six core metrics")
    p.add_argument("--add-counts", action="store_true",
                   help="also plot count columns (TP/FP/FN/over-seg/under-seg)")
    p.add_argument("--norm", default="rank",
                   choices=["rank", "minmax", "robust", "zscore"],
                   help="per-column normalization method (default rank, robust to extremes)")
    p.add_argument("--dedup", default="complete",
                   choices=["complete", "none"], help="when a combination has multiple rows, keep only the most complete one")
    p.add_argument("--methods", default=None, help="only plot these methods, comma-separated")
    p.add_argument("--datasets", default=None, help="only plot these datasets (base id), comma-separated")
    p.add_argument("--row-order", default="method", choices=["method", "dataset"],
                   help="row order: method=group by method, dataset=group by dataset")
    p.add_argument("--cmap", default=None, help="override the default light red/light blue colormap (e.g. RdBu_r)")
    p.add_argument("--no-annot", dest="annot", action="store_false", help="do not write in-cell numbers")
    p.add_argument("--annot-normalized", action="store_true",
                   help="show normalized values in cells instead of raw values")
    p.add_argument("--colorbar", action="store_true", help="show the colorbar (hidden by default)")
    p.add_argument("--cell-w", type=float, default=0.72, help="column width (inches)")
    p.add_argument("--cell-h", type=float, default=0.32, help="row height (inches)")
    p.add_argument("--annot-fs", type=float, default=6.8, help="in-cell font size")
    p.add_argument("--row-fs", type=float, default=8.0, help="row label font size")
    p.add_argument("--col-fs", type=float, default=9.5, help="column label font size")
    p.add_argument("--fig-w", type=float, default=11.0)
    p.add_argument("--fig-h", type=float, default=7.0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--font", default=None)
    p.add_argument("--no-pdf", dest="save_pdf", action="store_false")
    p.set_defaults(annot=True, save_pdf=True)
    args = p.parse_args(argv)

    if not os.path.exists(args.xlsx):
        raise SystemExit(f"[Error] file not found: {args.xlsx}")
    os.makedirs(args.outdir, exist_ok=True)

    font_used = setup_font(args.font)
    df = tidy(load_sheet(args.xlsx, args.sheet), args.dedup)

    if args.methods:
        df = df[df["method"].isin({canon_method(m.strip()) for m in args.methods.split(",")})]
    if args.datasets:
        df = df[df["ds_base"].isin({d.strip() for d in args.datasets.split(",")})]
    if df.empty:
        raise SystemExit("[Error] no data after filtering; check --methods / --datasets.")

    if args.metrics:
        want = [m.strip() for m in args.metrics.split(",")]
        metrics = [m for m in want if m in df.columns]
        missing = [m for m in want if m not in df.columns]
        if missing:
            print(f"[Hint] ignoring metrics not present in data: {missing}")
    else:
        metrics = [m for m in QUALITY_METRICS if m in df.columns]
        if args.add_counts:
            metrics += [m for m in COUNT_METRICS if m in df.columns]
    metrics = [m for m in metrics if pd.to_numeric(df[m], errors="coerce").notna().any()]
    if not metrics:
        raise SystemExit("[Error] no usable metric columns.")

    fig = draw(df, metrics, args)

    base = os.path.join(args.outdir, args.prefix)
    png = base + ".png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    outs = [png]
    if args.save_pdf:
        pdf = base + ".pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        outs.append(pdf)
    plt.close(fig)

    long = df.copy()
    long["row"] = long["method"] + " × " + long["ds_label"]
    keep = ["row", "method", "ds_label"] + metrics
    long_csv = base + "_data.csv"
    long[keep].to_csv(long_csv, index=False, encoding="utf-8-sig")
    outs.append(long_csv)

    print(f"Font:{font_used}")
    print(f"Rows (method x dataset):{len(df)}   Cols (metrics):{len(metrics)}")
    print(f"Normalization:{args.norm}    in-cell shows:{'normalized value' if args.annot_normalized else 'raw value'}")
    print("Output:")
    for o in outs:
        print("  " + o)


if __name__ == "__main__":
    sys.exit(main())
