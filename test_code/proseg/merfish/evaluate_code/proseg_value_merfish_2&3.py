#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_merfish_proseg.py

Proseg on the MERFISH liver ROIs.

Proseg has no label raster of its own, so its cell polygons are rasterised onto
the shared ROI grid.  Doing it this way rather than reading the centroids out of
cell-metadata means Proseg goes through exactly the same centroid, area and
transcript-lookup code as the other three methods.

  python eval_merfish_proseg.py --datasets liver1,liver2

This file is self-contained: the shared evaluation core is included
below, so it can be copied anywhere and run on its own.

----- shared core -----
evaluation core   v2026.08.13

Shared evaluation core for the two MERFISH liver datasets.

Every method is evaluated through the SAME path:

    label array (any grid size) -> ROI-local mosaic pixels -> transcript lookup
                                -> pred centroids / areas / counts / gene vectors
                                -> sparse Hungarian match against platform cells
                                -> the identical metric columns used for the
                                   35 rows already in the benchmark workbook

The only thing a per-method script has to supply is a label array plus the
mosaic-pixel size of one of its grid pixels.  Everything downstream is common,
so UCS / Proseg / Cellist / GeneSegNet are measured with one ruler.

Geometry, once and for all
-------------------------
  orientation           normal (no flip) for both livers
  ROI                   10000 x 10000 mosaic px centred on the mosaic
                          liver1  y0=36396  x0=49272
                          liver2  y0=46028  x0=53436
  mosaic px -> micron   um = (px - offset) / scale   from the dataset's own
                        micron_to_mosaic_pixel_transform.csv
  ROI width             10000 px / 9.2593 px per um = 1080.0 um

That last line is the reason the four methods are comparable at all: the UCS
1081-px grid, the Cellist 2160-bin grid and the GeneSegNet 10000-px mask all
span the same 1080 um square.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.13-idbridge"

# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

DATASETS = {
    "liver1": dict(
        data_root="/data/qiuyijia/dataset/merfish_liver1",
        roi_y0=36396,
        roi_x0=49272,
        ucs_dir="/data/qiuyijia/ucs_merfish_liver1",
        proseg_dir="/data/qiuyijia/proseg/output/liver1",
        cellist_dir="/data/qiuyijia/cellist/merfish_liver1",
        comseg_dir="/data/qiuyijia/comseg/merfish_liver1",
        genesegnet_dir="/data/qiuyijia/genesegnet_merfish_liver/liver1",
    ),
    "liver2": dict(
        data_root="/data/qiuyijia/dataset/merfish_liver2",
        roi_y0=46028,
        roi_x0=53436,
        ucs_dir="/data/qiuyijia/ucs_merfish_liver2",
        proseg_dir="/data/qiuyijia/proseg/output/liver2",
        cellist_dir="/data/qiuyijia/cellist/merfish_liver2",
        comseg_dir="/data/qiuyijia/comseg/merfish_liver2",
        genesegnet_dir="/data/qiuyijia/genesegnet_merfish_liver/liver2",
    ),
}

ROI_SIZE = 10000

# exact column order of the rows already in the workbook
REQUESTED_METRICS = [
    "method",
    "Pred cell count",
    "GT cell count",
    "Cell count ratio",
    "Matched cell count",
    "Detection precision",
    "Detection recall",
    "Detection F1",
    "Mean centroid shift",
    "Median centroid shift",
    "p95 centroid shift",
    "matched_pair_n_transcripts_vs_total_counts_pearson",
    "matched_pair_n_transcripts_vs_total_counts_spearman",
    "matched_pair_n_transcripts_vs_total_counts_mae",
    "matched_pair_n_transcripts_vs_total_counts_rmse",
    "Pred transcript rows",
    "GT transcript rows",
    "Pred transcript assignment rate",
    "Transcript matched cell pairs",
    "matched_cell_transcript_count_pearson",
    "matched_cell_transcript_count_spearman",
    "matched_cell_transcript_count_mae",
    "matched_cell_transcript_count_rmse",
    "matched_cell_gene_vector_mean_cosine",
    "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson",
    "matched_cell_gene_vector_valid_pairs",
    "transcript_id_overlap_n",
    "transcript_assignment_accuracy_via_matched_cells",
    "dataset",
    "Overall correct fraction",
]

SELECTED = [
    "method", "dataset",
    "Pred cell count", "GT cell count", "Cell count ratio",
    "Detection precision", "Detection recall", "Detection F1",
    "Median centroid shift",
    "Pred transcript assignment rate",
    "transcript_assignment_accuracy_via_matched_cells",
    "Overall correct fraction",
    "matched_cell_gene_vector_mean_cosine",
    "matched_cell_transcript_count_pearson",
    "median_area_um2", "median_diameter_um", "coverage_fraction",
    "uses_platform_prior", "gt_counts_source",
]

log = logging.getLogger("merfish_eval")


def setup_log(out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(out_dir / f"{tag}.log", mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)


def rule(msg: str = "") -> None:
    log.info("=" * 78)
    if msg:
        log.info(msg)
        log.info("=" * 78)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@dataclass
class Geometry:
    """Everything needed to move between micron, mosaic px and ROI-local px."""

    sx: float
    sy: float
    ox: float
    oy: float
    y0: int
    x0: int
    size: int = ROI_SIZE

    @property
    def um_per_px_x(self) -> float:
        return 1.0 / self.sx

    @property
    def um_per_px_y(self) -> float:
        return 1.0 / self.sy

    def um_bounds(self):
        return (
            (self.x0 - self.ox) / self.sx,
            (self.x0 + self.size - self.ox) / self.sx,
            (self.y0 - self.oy) / self.sy,
            (self.y0 + self.size - self.oy) / self.sy,
        )

    def um_to_local(self, ux, uy):
        """micron -> ROI-local mosaic pixels (float)."""
        return (
            np.asarray(ux, float) * self.sx + self.ox - self.x0,
            np.asarray(uy, float) * self.sy + self.oy - self.y0,
        )

    def local_to_um(self, px, py):
        return (
            (np.asarray(px, float) + self.x0 - self.ox) / self.sx,
            (np.asarray(py, float) + self.y0 - self.oy) / self.sy,
        )

    def inside(self, ux, uy):
        px, py = self.um_to_local(ux, uy)
        return (px >= 0) & (px < self.size) & (py >= 0) & (py < self.size)


def find_one(root: Path, pattern: str):
    hits = sorted(root.rglob(pattern))
    return hits[0] if hits else None


def load_geometry(dataset: str) -> tuple[Geometry, dict]:
    cfg = DATASETS[dataset]
    root = Path(cfg["data_root"])
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    tf = find_one(root, "*micron_to_mosaic_pixel_transform.csv")
    if tf is None:
        raise FileNotFoundError(
            f"no *micron_to_mosaic_pixel_transform.csv under {root}; without it "
            "micron and mosaic pixels cannot be related"
        )
    m = np.loadtxt(tf)
    geo = Geometry(
        sx=float(m[0, 0]), sy=float(m[1, 1]),
        ox=float(m[0, 2]), oy=float(m[1, 2]),
        y0=int(cfg["roi_y0"]), x0=int(cfg["roi_x0"]),
    )
    x0u, x1u, y0u, y1u = geo.um_bounds()
    log.info(f"transform   {tf.name}")
    log.info(f"  {geo.sx:.4f} px/um  ->  {geo.um_per_px_x:.4f} um/px")
    log.info(f"ROI         y[{geo.y0},{geo.y0+geo.size}) x[{geo.x0},{geo.x0+geo.size}) mosaic px")
    log.info(f"            x {x0u:.1f}..{x1u:.1f} um   y {y0u:.1f}..{y1u:.1f} um"
             f"   ({x1u-x0u:.1f} x {y1u-y0u:.1f} um)")
    return geo, cfg


# --------------------------------------------------------------------------
# transcripts (shared across every method)
# --------------------------------------------------------------------------


@dataclass
class Transcripts:
    ux: np.ndarray            # micron
    uy: np.ndarray
    gene: np.ndarray          # gene index
    genes: list               # gene names
    gt_label: np.ndarray      # platform cell id per transcript, 0 = unassigned
    source: str
    has_gt_labels: bool


def _open_text(p: Path):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p, "r")


def load_transcripts(dataset: str, geo: Geometry, cfg: dict,
                     override: str | None = None) -> Transcripts:
    """
    Preferred source is the Proseg ROI subset: it is already cropped to exactly
    this ROI, its coordinates are micron, and its cell_id column carries the
    platform assignment, which is the only per-transcript ground truth MERFISH
    offers (detected_transcripts.csv has no cell column).

    Falling back to detected_transcripts.csv works but streams ~270 M rows and
    yields no per-transcript ground truth.
    """
    rule("TRANSCRIPTS")

    cand = []
    if override:
        cand.append(Path(override))
    cand.append(Path(cfg["proseg_dir"]) / "transcripts_roi_with_prior.csv.gz")
    cand.append(Path(cfg["proseg_dir"]) / "transcripts_roi_with_prior.csv")

    src = next((p for p in cand if p.exists()), None)
    if src is not None:
        log.info(f"source      {src}")
        df = pd.read_csv(src)
        log.info(f"  {len(df):,} rows   columns {list(df.columns)}")
        xcol = _pick(df, ["global_x", "x_um", "x"])
        ycol = _pick(df, ["global_y", "y_um", "y"])
        gcol = _pick(df, ["gene", "target", "gene_name"])
        ccol = _pick(df, ["cell_id", "cell", "prior_cell"], required=False)
        ux = df[xcol].to_numpy(float)
        uy = df[ycol].to_numpy(float)
        keep = geo.inside(ux, uy)
        if keep.sum() < len(df):
            log.info(f"  {int(keep.sum()):,} of {len(df):,} fall inside the ROI "
                     f"({keep.mean()*100:.2f}%)")
        df = df.loc[keep]
        ux, uy = ux[keep], uy[keep]
        gname = df[gcol].astype(str).to_numpy()
        genes = sorted(set(gname))
        gidx = {g: i for i, g in enumerate(genes)}
        gene = np.array([gidx[g] for g in gname], np.int32)
        if ccol is not None:
            raw = pd.to_numeric(df[ccol], errors="coerce").fillna(0)
            gt = np.array(raw.to_numpy(np.int64), copy=True)
            gt[gt < 0] = 0
            frac = float((gt > 0).mean())
            log.info(f"  per-transcript platform labels from '{ccol}': "
                     f"{frac*100:.1f}% assigned, {len(np.unique(gt[gt>0])):,} distinct cells")
            has_gt = frac > 0.05
            if not has_gt:
                log.info("  too few labelled rows to be a usable ground truth; "
                         "transcript-level accuracy will be reported as missing")
        else:
            gt = np.zeros(len(df), np.int64)
            has_gt = False
            log.info("  no cell column, so transcript-level accuracy is not available")
        log.info(f"  genes {len(genes):,}")
        return Transcripts(ux, uy, gene, genes, gt, str(src), has_gt)

    # fallback: stream the raw table
    raw = find_one(Path(cfg["data_root"]), "detected_transcripts.csv*")
    if raw is None:
        raise FileNotFoundError("no transcript table found")
    log.info(f"source      {raw}  (streaming; no per-transcript ground truth)")
    x0u, x1u, y0u, y1u = geo.um_bounds()
    parts = []
    scanned = 0
    for chunk in pd.read_csv(raw, chunksize=2_000_000):
        scanned += len(chunk)
        xcol = _pick(chunk, ["global_x", "x"])
        ycol = _pick(chunk, ["global_y", "y"])
        gcol = _pick(chunk, ["gene", "target"])
        sel = chunk[(chunk[xcol] >= x0u) & (chunk[xcol] < x1u)
                    & (chunk[ycol] >= y0u) & (chunk[ycol] < y1u)]
        if len(sel):
            parts.append(sel[[xcol, ycol, gcol]].rename(
                columns={xcol: "gx", ycol: "gy", gcol: "gene"}))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["gx", "gy", "gene"])
    log.info(f"  scanned {scanned:,}, kept {len(df):,}")
    gname = df["gene"].astype(str).to_numpy()
    genes = sorted(set(gname))
    gidx = {g: i for i, g in enumerate(genes)}
    return Transcripts(df["gx"].to_numpy(float), df["gy"].to_numpy(float),
                       np.array([gidx[g] for g in gname], np.int32), genes,
                       np.zeros(len(df), np.int64), str(raw), False)


def _pick(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if required:
        raise KeyError(f"none of {names} in {list(df.columns)}")
    return None


# --------------------------------------------------------------------------
# ground truth cells
# --------------------------------------------------------------------------


@dataclass
class GroundTruth:
    ids: np.ndarray                   # id used by cell_metadata / cell_by_gene
    tx_ids: np.ndarray                # SAME cell, but in the transcript table's
                                      # id space; -1 when it could not be bridged
    ux: np.ndarray
    uy: np.ndarray
    total_counts: np.ndarray
    gene_matrix: np.ndarray | None    # cells x genes, aligned to Transcripts.genes
    source: str
    counts_source: str = "unknown"


def load_ground_truth(cfg: dict, geo: Geometry, tx: Transcripts,
                      counts_source: str = "auto",
                      bridge_radius: float = 5.0) -> GroundTruth:
    rule("GROUND TRUTH (platform cells)")
    root = Path(cfg["data_root"])
    meta = find_one(root, "*cell_metadata.csv")
    if meta is None:
        raise FileNotFoundError(f"no *cell_metadata.csv under {root}")
    log.info(f"metadata    {meta}")
    md = pd.read_csv(meta)
    idcol = md.columns[0] if md.columns[0].lower() in ("", "unnamed: 0", "cell",
                                                      "cell_id", "entityid") \
        else _pick(md, ["cell", "cell_id", "EntityID"], required=False)
    if idcol is None:
        md = md.reset_index().rename(columns={"index": "_cell"})
        idcol = "_cell"
    xcol = _pick(md, ["center_x", "centroid_x", "x"])
    ycol = _pick(md, ["center_y", "centroid_y", "y"])
    log.info(f"  {len(md):,} cells on the whole section; id column '{idcol}'")

    ux = md[xcol].to_numpy(float)
    uy = md[ycol].to_numpy(float)
    keep = geo.inside(ux, uy)
    md = md.loc[keep]
    ux, uy = ux[keep], uy[keep]
    ids = pd.to_numeric(md[idcol], errors="coerce").fillna(-1).to_numpy(np.int64)
    log.info(f"  {len(md):,} have their centroid inside the ROI  <- GT cell count")

    # ------------------------------------------------------------------
    # bridge the two id spaces
    #
    # cell_metadata numbers the cells across the whole section, while the
    # transcript table's prior column numbers only the cells of this ROI, from
    # 1.  Comparing the two directly makes every transcript look misassigned,
    # which is exactly how this used to report an accuracy of 0.0000.  The two
    # are reconciled through the one thing they share: the centroid.
    # ------------------------------------------------------------------
    tx_ids = np.full(len(md), -1, np.int64)
    if tx.has_gt_labels:
        lab = tx.gt_label
        sel = lab > 0
        uniq, inv = np.unique(lab[sel], return_inverse=True)
        cxs = np.bincount(inv, weights=tx.ux[sel]) / np.bincount(inv)
        cys = np.bincount(inv, weights=tx.uy[sel]) / np.bincount(inv)
        direct = float(np.isin(ids, uniq).mean())
        log.info(f"id spaces   {direct*100:.1f}% of ROI metadata ids also occur "
                 f"in the transcript prior column")
        if direct > 0.8:
            tx_ids = ids.copy()
            log.info("            the two id spaces agree, using them directly")
        else:
            from scipy.spatial import cKDTree
            d, j = cKDTree(np.column_stack([cxs, cys])).query(
                np.column_stack([ux, uy]), k=1)
            ok = d <= bridge_radius
            tx_ids[ok] = uniq[j[ok]]
            log.info(f"            bridged by centroid: {int(ok.sum()):,} of "
                     f"{len(ids):,} GT cells matched within {bridge_radius} um "
                     f"(median distance {np.median(d[ok]):.2f} um)")
            if ok.mean() < 0.9:
                log.info("            WARNING: less than 90% bridged; the "
                         "transcript-level accuracy will only cover that subset")

    # counts and gene vectors from the official cell x gene table
    gm = None
    tot = np.full(len(md), np.nan)
    csource = "none"
    cbg = None
    if counts_source in ("auto", "official"):
        for pat in ("*cell_by_gene.csv", "*cell_by_gene.csv.gz",
                    "*cell_by_gene*.csv", "*by_gene*.csv"):
            cbg = find_one(root, pat)
            if cbg is not None:
                break
    if cbg is not None:
        log.info(f"cell_by_gene {cbg}")
        cg = pd.read_csv(cbg, index_col=0)
        cg.index = pd.to_numeric(cg.index, errors="coerce").fillna(-1).astype(np.int64)
        cg = cg.loc[~cg.index.duplicated()]
        common = cg.index.intersection(pd.Index(ids))
        log.info(f"  {cg.shape[0]:,} x {cg.shape[1]:,};"
                 f" {len(common):,} of {len(ids):,} ROI cells found")
        sub = cg.reindex(ids)
        tot = sub.sum(axis=1).to_numpy(float)
        cols = [c for c in cg.columns]
        pos = {c: i for i, c in enumerate(cols)}
        gm = np.zeros((len(ids), len(tx.genes)), np.float32)
        hit = 0
        for j, g in enumerate(tx.genes):
            if g in pos:
                gm[:, j] = np.nan_to_num(sub.iloc[:, pos[g]].to_numpy(float))
                hit += 1
        log.info(f"  {hit}/{len(tx.genes)} transcript genes present in the table")
        log.info(f"  GT counts/cell median {np.nanmedian(tot):.0f}")
        csource = f"official {Path(cbg).name}"

    if gm is None and tx.has_gt_labels and counts_source in ("auto", "prior"):
        # Derive them from the transcript table's own platform labels.  Same ROI,
        # same filtering, and available for every dataset -- but note that for
        # UCS / Proseg / ComSeg this column is also their input, so agreement
        # with it is partly circular.
        log.info("counts      derived from the transcript table's platform "
                 "labels (no usable cell_by_gene table)")
        pos = {int(v): i for i, v in enumerate(tx_ids) if v > 0}
        gm = np.zeros((len(ids), len(tx.genes)), np.float32)
        sel = tx.gt_label > 0
        rowi = np.array([pos.get(int(v), -1) for v in tx.gt_label[sel]])
        ok = rowi >= 0
        np.add.at(gm, (rowi[ok], tx.gene[sel][ok]), 1.0)
        tot = gm.sum(axis=1)
        tot[tx_ids < 0] = np.nan
        log.info(f"  {int((tx_ids>0).sum()):,} GT cells receive counts; "
                 f"median {np.nanmedian(tot):.0f}")
        csource = "transcript prior column (partly circular for prior-based methods)"

    if gm is None:
        log.info("no count source at all, so count and gene-vector agreement "
                 "will be reported as missing")
        log.info(f"  files under {root}: "
                 + ", ".join(sorted(p.name for p in root.iterdir())[:12]))

    log.info(f"counts source  {csource}")
    return GroundTruth(ids, tx_ids, ux, uy, tot, gm, str(meta), csource)


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------


@dataclass
class Prediction:
    """A label array covering the ROI plus the mosaic-px size of one of its pixels."""

    labels: np.ndarray
    px_per_cell: float                     # mosaic px per label-array pixel
    method: str
    uses_platform_prior: bool
    note: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class PredCells:
    ids: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    area_um2: np.ndarray
    n_tx: np.ndarray
    gene_matrix: np.ndarray
    coverage: float
    tx_label: np.ndarray


def summarise_prediction(pred: Prediction, geo: Geometry, tx: Transcripts) -> PredCells:
    rule(f"PREDICTION  {pred.method}")
    lab = pred.labels
    g = lab.shape[0]
    s = pred.px_per_cell
    um_x = s / geo.sx
    um_y = s / geo.sy
    log.info(f"grid        {lab.shape}  dtype {lab.dtype}")
    log.info(f"            1 grid px = {s:.4f} mosaic px = {um_x:.4f} um")
    log.info(f"            grid spans {g*um_x:.1f} x {lab.shape[0]*um_y:.1f} um")
    if pred.note:
        log.info(f"note        {pred.note}")

    flat = lab.reshape(-1)
    if flat.min() < 0:
        flat = np.where(flat < 0, 0, flat)
    counts = np.bincount(flat.astype(np.int64))
    ids = np.nonzero(counts)[0]
    ids = ids[ids > 0]
    if len(ids) == 0:
        raise RuntimeError("the label array is empty")
    area_px = counts[ids].astype(float)
    coverage = float((lab > 0).mean())
    log.info(f"cells       {len(ids):,}   coverage {coverage*100:.2f}%")

    # geometric centroids
    rows, cols = np.nonzero(lab)
    lv = lab[rows, cols].astype(np.int64)
    order = np.argsort(lv, kind="stable")
    lv, rows, cols = lv[order], rows[order], cols[order]
    bounds = np.searchsorted(lv, ids)
    bounds = np.append(bounds, len(lv))
    cy = np.array([rows[bounds[i]:bounds[i + 1]].mean() for i in range(len(ids))])
    cx = np.array([cols[bounds[i]:bounds[i + 1]].mean() for i in range(len(ids))])
    pux, puy = geo.local_to_um((cx + 0.5) * s, (cy + 0.5) * s)

    area_um2 = area_px * um_x * um_y
    med_a = float(np.median(area_um2))
    log.info(f"area        median {med_a:.1f} um^2   equivalent diameter "
             f"{2*np.sqrt(med_a/np.pi):.2f} um")

    # transcript assignment through the same array
    lpx, lpy = geo.um_to_local(tx.ux, tx.uy)
    r = np.clip((lpy / s).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip((lpx / s).astype(np.int64), 0, lab.shape[1] - 1)
    tx_lab = lab[r, c].astype(np.int64)
    tx_lab[tx_lab < 0] = 0
    n_assigned = int((tx_lab > 0).sum())
    log.info(f"transcripts {len(tx.ux):,} in ROI, {n_assigned:,} assigned "
             f"({n_assigned/max(len(tx.ux),1)*100:.1f}%)")

    pos = {int(v): i for i, v in enumerate(ids)}
    n_tx = np.zeros(len(ids), np.int64)
    gmat = np.zeros((len(ids), len(tx.genes)), np.float32)
    sel = tx_lab > 0
    if sel.any():
        rowi = np.array([pos.get(int(v), -1) for v in tx_lab[sel]])
        ok = rowi >= 0
        rowi = rowi[ok]
        gi = tx.gene[sel][ok]
        np.add.at(n_tx, rowi, 1)
        np.add.at(gmat, (rowi, gi), 1.0)
    log.info(f"counts/cell median {np.median(n_tx):.0f}")

    return PredCells(ids, pux, puy, area_um2, n_tx, gmat, coverage, tx_lab)


# --------------------------------------------------------------------------
# matching  (same recipe as the rows already in the workbook)
# --------------------------------------------------------------------------


def infer_match_radius(gx, gy, quantile=0.95, factor=1.25, floor=2.0):
    from scipy.spatial import cKDTree
    pts = np.column_stack([gx, gy])
    if len(pts) < 2:
        return floor
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        return floor
    return float(max(np.quantile(nn, quantile) * factor, floor))


def build_candidate_edges(px, py, gx, gy, radius):
    from scipy.spatial import cKDTree
    tp = cKDTree(np.column_stack([px, py]))
    tg = cKDTree(np.column_stack([gx, gy]))
    pairs = tp.query_ball_tree(tg, r=radius)
    pi, gi = [], []
    for i, js in enumerate(pairs):
        for j in js:
            pi.append(i)
            gi.append(j)
    pi = np.asarray(pi, np.int64)
    gi = np.asarray(gi, np.int64)
    if len(pi) == 0:
        return pi, gi, np.zeros(0)
    d = np.hypot(px[pi] - gx[gi], py[pi] - gy[gi])
    return pi, gi, d


def _cc_bipartite(pi, gi, n_pred, n_gt):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = n_pred + n_gt
    a = coo_matrix((np.ones(len(pi)), (pi, n_pred + gi)), shape=(n, n))
    a = a + a.T
    ncc, lab = connected_components(a, directed=False)
    return ncc, lab


def match_sparse_hungarian(px, py, gx, gy, radius, greedy_above=400):
    """Optimal within each small connected component, greedy in the huge ones."""
    from scipy.optimize import linear_sum_assignment
    pi, gi, d = build_candidate_edges(px, py, gx, gy, radius)
    if len(pi) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    ncc, lab = _cc_bipartite(pi, gi, len(px), len(gx))
    epred, egt, edist = [], [], []
    comp_of_edge = lab[pi]
    order = np.argsort(comp_of_edge, kind="stable")
    pi, gi, d, comp_of_edge = pi[order], gi[order], d[order], comp_of_edge[order]
    starts = np.searchsorted(comp_of_edge, np.unique(comp_of_edge))
    starts = np.append(starts, len(pi))
    n_greedy = 0
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        sp, sg, sd = pi[a:b], gi[a:b], d[a:b]
        up = np.unique(sp)
        ug = np.unique(sg)
        if len(up) * len(ug) > greedy_above * greedy_above:
            n_greedy += 1
            o = np.argsort(sd)
            usedp, usedg = set(), set()
            for t in o:
                if sp[t] in usedp or sg[t] in usedg:
                    continue
                usedp.add(sp[t]); usedg.add(sg[t])
                epred.append(sp[t]); egt.append(sg[t]); edist.append(sd[t])
            continue
        pmap = {v: i for i, v in enumerate(up)}
        gmap = {v: i for i, v in enumerate(ug)}
        big = sd.max() * 10 + 1
        cost = np.full((len(up), len(ug)), big)
        cost[[pmap[v] for v in sp], [gmap[v] for v in sg]] = sd
        r, c = linear_sum_assignment(cost)
        for rr, cc in zip(r, c):
            if cost[rr, cc] >= big:
                continue
            epred.append(up[rr]); egt.append(ug[cc]); edist.append(cost[rr, cc])
    if n_greedy:
        log.info(f"  {n_greedy} very large component(s) matched greedily")
    return (np.asarray(epred, np.int64), np.asarray(egt, np.int64),
            np.asarray(edist, float))


def numeric_pair_metrics(a, b, prefix):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = {f"{prefix}_pearson": np.nan, f"{prefix}_spearman": np.nan,
           f"{prefix}_mae": np.nan, f"{prefix}_rmse": np.nan}
    if ok.sum() < 3:
        return out
    a, b = a[ok], b[ok]
    if a.std() > 0 and b.std() > 0:
        out[f"{prefix}_pearson"] = float(pearsonr(a, b)[0])
        out[f"{prefix}_spearman"] = float(spearmanr(a, b)[0])
    out[f"{prefix}_mae"] = float(np.mean(np.abs(a - b)))
    out[f"{prefix}_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
    return out


def gene_vector_metrics(pm, gm):
    """cosine / Jensen-Shannon distance / pearson, averaged over valid pairs."""
    from scipy.spatial.distance import jensenshannon
    cos, js, pr = [], [], []
    for i in range(pm.shape[0]):
        a = pm[i].astype(float)
        b = gm[i].astype(float)
        if a.sum() <= 0 or b.sum() <= 0:
            continue
        na = a / np.linalg.norm(a)
        nb = b / np.linalg.norm(b)
        cos.append(float(np.dot(na, nb)))
        d = jensenshannon(a / a.sum(), b / b.sum(), base=2)
        if np.isfinite(d):
            js.append(float(d))
        if a.std() > 0 and b.std() > 0:
            pr.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "matched_cell_gene_vector_mean_cosine": float(np.mean(cos)) if cos else np.nan,
        "matched_cell_gene_vector_mean_js_distance": float(np.mean(js)) if js else np.nan,
        "matched_cell_gene_vector_mean_pearson": float(np.mean(pr)) if pr else np.nan,
        "matched_cell_gene_vector_valid_pairs": len(cos),
    }


# --------------------------------------------------------------------------
# the evaluation itself
# --------------------------------------------------------------------------


def evaluate(pred: Prediction, dataset: str, out_dir: Path,
             transcripts_override=None, counts_source: str = "auto",
             bridge_radius: float = 5.0) -> dict:
    geo, cfg = load_geometry(dataset)
    tx = load_transcripts(dataset, geo, cfg, transcripts_override)
    gt = load_ground_truth(cfg, geo, tx, counts_source, bridge_radius)
    pc = summarise_prediction(pred, geo, tx)

    rule("MATCHING")
    radius = infer_match_radius(gt.ux, gt.uy)
    log.info(f"match radius {radius:.2f} um  (GT nearest-neighbour p95 x 1.25)")
    mp, mg, md = match_sparse_hungarian(pc.ux, pc.uy, gt.ux, gt.uy, radius)
    n_match = len(mp)
    log.info(f"matched     {n_match:,} pairs of "
             f"{len(pc.ids):,} predicted and {len(gt.ux):,} GT")

    prec = n_match / max(len(pc.ids), 1)
    rec = n_match / max(len(gt.ux), 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
    log.info(f"precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    if n_match:
        log.info(f"centroid shift mean {md.mean():.2f}  median {np.median(md):.2f}"
                 f"  p95 {np.percentile(md,95):.2f} um")

    row = {
        "gt_counts_source": gt.counts_source,
        "method": pred.method,
        "dataset": dataset,
        "Pred cell count": len(pc.ids),
        "GT cell count": len(gt.ux),
        "Cell count ratio": len(pc.ids) / max(len(gt.ux), 1),
        "Matched cell count": n_match,
        "Detection precision": prec,
        "Detection recall": rec,
        "Detection F1": f1,
        "Mean centroid shift": float(md.mean()) if n_match else np.nan,
        "Median centroid shift": float(np.median(md)) if n_match else np.nan,
        "p95 centroid shift": float(np.percentile(md, 95)) if n_match else np.nan,
        "Pred transcript rows": int((pc.tx_label > 0).sum()),
        "GT transcript rows": int(len(tx.ux)),
        "Pred transcript assignment rate": float((pc.tx_label > 0).mean()),
        "Transcript matched cell pairs": n_match,
        "median_area_um2": float(np.median(pc.area_um2)),
        "median_diameter_um": float(2 * np.sqrt(np.median(pc.area_um2) / np.pi)),
        "coverage_fraction": pc.coverage,
        "uses_platform_prior": pred.uses_platform_prior,
    }

    # count agreement -------------------------------------------------------
    if n_match:
        pn = pc.n_tx[mp].astype(float)
        gtot = gt.total_counts[mg].astype(float)
        row.update(numeric_pair_metrics(
            pn, gtot, "matched_pair_n_transcripts_vs_total_counts"))
        row.update(numeric_pair_metrics(
            pn, gtot, "matched_cell_transcript_count"))
    else:
        for p in ("matched_pair_n_transcripts_vs_total_counts",
                  "matched_cell_transcript_count"):
            for s in ("pearson", "spearman", "mae", "rmse"):
                row[f"{p}_{s}"] = np.nan

    # gene vectors ----------------------------------------------------------
    if n_match and gt.gene_matrix is not None:
        row.update(gene_vector_metrics(pc.gene_matrix[mp], gt.gene_matrix[mg]))
        log.info(f"gene vector cosine {row['matched_cell_gene_vector_mean_cosine']:.4f}"
                 f"  over {row['matched_cell_gene_vector_valid_pairs']:,} pairs")
    else:
        row.update({
            "matched_cell_gene_vector_mean_cosine": np.nan,
            "matched_cell_gene_vector_mean_js_distance": np.nan,
            "matched_cell_gene_vector_mean_pearson": np.nan,
            "matched_cell_gene_vector_valid_pairs": 0,
        })

    # per-transcript accuracy ----------------------------------------------
    if tx.has_gt_labels and n_match:
        gt_of_pred = {}
        for a, b in zip(mp, mg):
            t = int(gt.tx_ids[b])
            if t > 0:
                gt_of_pred[int(pc.ids[a])] = t
        pl = pc.tx_label
        gl = tx.gt_label
        both = (pl > 0) & (gl > 0)
        mapped = np.array([gt_of_pred.get(int(v), -1) for v in pl[both]])
        valid = mapped >= 0
        overlap = int(valid.sum())
        acc = float((mapped[valid] == gl[both][valid]).mean()) if overlap else np.nan
        row["transcript_id_overlap_n"] = overlap
        row["transcript_assignment_accuracy_via_matched_cells"] = acc
        row["Overall correct fraction"] = (
            row["Pred transcript assignment rate"] * acc
            if np.isfinite(acc) else np.nan)
        log.info(f"transcript accuracy {acc:.4f} over {overlap:,} rows"
                 f"  -> overall correct {row['Overall correct fraction']:.4f}")
    else:
        row["transcript_id_overlap_n"] = 0
        row["transcript_assignment_accuracy_via_matched_cells"] = np.nan
        row["Overall correct fraction"] = np.nan
        if not tx.has_gt_labels:
            log.info("transcript-level accuracy unavailable: this dataset has no "
                     "per-transcript platform labels")

    # write ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{pred.method}_{dataset}"
    full = pd.DataFrame([{k: row.get(k, np.nan) for k in REQUESTED_METRICS}])
    full.to_csv(out_dir / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    sel = pd.DataFrame([{k: row.get(k, np.nan) for k in SELECTED}])
    sel.to_csv(out_dir / f"{tag}_sparse_hungarian_qc_metrics_selected.csv", index=False)
    with open(out_dir / f"{tag}_sparse_hungarian_qc_metrics_full.json", "w") as f:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in row.items()}, f, indent=2, default=str)

    if n_match:
        pd.DataFrame({
            "pred_id": pc.ids[mp], "gt_id": gt.ids[mg],
            "centroid_shift_um": md,
            "pred_n_transcripts": pc.n_tx[mp],
            "gt_total_counts": gt.total_counts[mg],
            "pred_area_um2": pc.area_um2[mp],
        }).to_csv(out_dir / f"{tag}_matched_pairs.csv", index=False)

    pd.DataFrame({
        "pred_id": pc.ids, "centroid_x_um": pc.ux, "centroid_y_um": pc.uy,
        "area_um2": pc.area_um2, "n_transcripts": pc.n_tx,
    }).to_csv(out_dir / f"{tag}_pred_cells.csv", index=False)

    rule("RESULT")
    for k in SELECTED:
        v = row.get(k)
        if isinstance(v, float):
            log.info(f"  {k:52s} {v:.4f}")
        else:
            log.info(f"  {k:52s} {v}")
    log.info(f"-> {out_dir}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    return row


def common_parser(method: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Evaluate {method} on the MERFISH liver ROIs")
    p.add_argument("--datasets", default="liver1,liver2",
                   help="comma separated: liver1,liver2")
    p.add_argument("--output-dir", default="/data/qiuyijia/eval_merfish")
    p.add_argument("--transcripts", default=None,
                   help="override the transcript table")
    p.add_argument("--gt-counts", choices=["auto", "official", "prior"],
                   default="auto",
                   help="where GT counts and gene vectors come from. 'prior' "
                        "forces the transcript table for every dataset, which "
                        "keeps liver1 and liver2 on the same footing")
    p.add_argument("--bridge-radius", type=float, default=5.0,
                   help="radius for reconciling the two cell id spaces")
    p.add_argument("--version", action="store_true")
    return p


def main_loop(method: str, build, argv=None):
    """build(dataset, cfg) -> Prediction | None"""
    ap = common_parser(method)
    build_args = getattr(build, "add_arguments", None)
    if build_args:
        build_args(ap)
    args = ap.parse_args(argv)
    if args.version:
        print(VERSION)
        return
    out = Path(args.output_dir)
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        setup_log(out, f"eval_{method}_{ds}")
        rule(f"{method.upper()}  x  MERFISH {ds}    core v{VERSION}")
        try:
            pred = build(ds, DATASETS[ds], args)
        except FileNotFoundError as e:
            log.info(f"[SKIP] {e}")
            continue
        if pred is None:
            log.info("[SKIP] no output for this dataset")
            continue
        rows.append(evaluate(pred, ds, out, args.transcripts,
                             args.gt_counts, args.bridge_radius))
    if rows:
        df = pd.DataFrame([{k: r.get(k, np.nan) for k in REQUESTED_METRICS}
                           for r in rows])
        path = out / f"{method}_all_datasets.csv"
        df.to_csv(path, index=False)
        print(f"\n-> {path}")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(pd.DataFrame([{k: r.get(k) for k in SELECTED} for r in rows])
                  .to_string(index=False))


# ==========================================================================
# proseg specific part
# ==========================================================================


import cv2



def add_arguments(ap):
    ap.add_argument("--polygons", default=None)
    ap.add_argument("--raster-grid", type=int, default=4000,
                    help="grid used to rasterise the polygons (default 4000, "
                         "i.e. 0.27 um per pixel)")


def _iter_polygons(geom):
    t = geom.get("type")
    if t == "Polygon":
        yield geom["coordinates"][0]
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly[0]
    elif t == "GeometryCollection":
        for g in geom.get("geometries", []):
            yield from _iter_polygons(g)


def build(dataset, cfg, args):
    root = Path(cfg["proseg_dir"])
    path = Path(args.polygons) if args.polygons else None
    if path is None:
        for name in ("cell-polygons.geojson.gz", "cell-polygons.geojson",
                     "cell-polygons-layers.geojson.gz"):
            if (root / name).exists():
                path = root / name
                break
    if path is None or not path.exists():
        raise FileNotFoundError(f"no cell polygons under {root}")
    log.info(f"polygons    {path}")

    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        gj = json.load(f)
    feats = gj.get("features", gj if isinstance(gj, list) else [])
    log.info(f"  {len(feats):,} features")

    geo, _ = load_geometry(dataset)
    g = int(args.raster_grid)
    s = ROI_SIZE / g
    lab = np.zeros((g, g), np.int32)

    # draw the biggest cells first so that, where polygons overlap, the smaller
    # cell survives rather than being erased entirely
    prepared = []
    for ft in feats:
        props = ft.get("properties") or {}
        cid = None
        for k in ("cell", "cell_id", "id", "Cell", "label"):
            if k in props and props[k] is not None:
                cid = props[k]
                break
        if cid is None:
            continue
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            continue
        if cid <= 0:
            cid += 1                       # proseg numbers from 0
        for ring in _iter_polygons(ft.get("geometry") or {}):
            a = np.asarray(ring, float)
            if a.ndim != 2 or len(a) < 3:
                continue
            px, py = geo.um_to_local(a[:, 0], a[:, 1])
            pts = np.column_stack([px / s, py / s])
            x, y = pts[:, 0], pts[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            prepared.append((area, cid, np.rint(pts).astype(np.int32)))

    if not prepared:
        raise RuntimeError("no usable polygon rings; check the property names "
                           "reported above")
    prepared.sort(key=lambda t: -t[0])
    for _, cid, pts in prepared:
        cv2.fillPoly(lab, [pts], int(cid))

    drawn = int(np.unique(lab[lab > 0]).size)
    ncell = len({c for _, c, _ in prepared})
    log.info(f"  rasterised {len(prepared):,} rings for {ncell:,} cells at "
             f"{s/geo.sx:.3f} um per pixel")
    log.info(f"  {drawn:,} cells survive in the raster; coverage "
             f"{(lab>0).mean()*100:.2f}%")
    note = ""
    if drawn < ncell * 0.95:
        note = (f"{ncell-drawn:,} cells were completely covered by others during "
                "rasterisation; raise --raster-grid if this matters")
        log.info(f"  {note}")

    return Prediction(labels=lab.astype(np.int64), px_per_cell=s, method="Proseg",
                      uses_platform_prior=True, note=note)


build.add_arguments = add_arguments

if __name__ == "__main__":
    main_loop("Proseg", build)