#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lung_proseg.py

Evaluate Proseg on the MERFISH lung ROI.

Self-contained.  The metric code is the same block that produced the
MERFISH liver rows, so the numbers are directly comparable; only the
ground truth differs, and it differs in the right direction: the lung
dataset ships cell_boundaries, so the per-transcript ground truth comes
from the platform outlines instead of being borrowed from one method's
prior column, which removes the circularity the liver rows carry.
"""


from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.13"

DATA_ROOT = "/data/qiuyijia/dataset/MERFISH_Lung_cancer"
ROI_SIZE = 10000
DATASET = "lung"

log = logging.getLogger("lung_eval")


def setup_log(d: Path, tag: str):
    d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    f = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(f); log.addHandler(sh)
    fh = logging.FileHandler(d / f"{tag}.log", mode="w"); fh.setFormatter(f)
    log.addHandler(fh)


def rule(msg=""):
    log.info("=" * 78)
    if msg:
        log.info(msg)
        log.info("=" * 78)


def find(root: Path, *pats):
    for p in pats:
        h = sorted(root.rglob(p))
        if h:
            return h[0]
    return None


def read_tab(path: Path):
    for p in (path, path.with_suffix(".csv.gz"), path.with_suffix(".csv")):
        if p.exists():
            return pd.read_csv(p) if str(p).endswith((".csv", ".csv.gz")) \
                else pd.read_parquet(p)
    return None


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    sx: float
    sy: float
    ox: float
    oy: float
    y0: int
    x0: int
    size: int

    @property
    def um_per_px(self):
        return 1.0 / self.sx

    def um_to_local(self, ux, uy):
        return (np.asarray(ux, float) * self.sx + self.ox - self.x0,
                np.asarray(uy, float) * self.sy + self.oy - self.y0)

    def local_to_um(self, px, py):
        return ((np.asarray(px, float) + self.x0 - self.ox) / self.sx,
                (np.asarray(py, float) + self.y0 - self.oy) / self.sy)

    def inside(self, ux, uy):
        px, py = self.um_to_local(ux, uy)
        return (px >= 0) & (px < self.size) & (py >= 0) & (py < self.size)


def load_geometry(root: Path, roi_size=ROI_SIZE) -> Geometry:
    rule("GEOMETRY")
    tf = find(root, "*micron_to_mosaic*transform*.csv")
    if tf is None:
        raise SystemExit(f"no transform under {root}")
    m = np.loadtxt(tf)
    import tifffile
    dapi = next((p for p in sorted(root.rglob("*.tif"))
                 if "dapi" in p.name.lower()), None)
    if dapi is None:
        raise SystemExit("no *dapi*.tif; needed only for the mosaic size")
    with tifffile.TiffFile(dapi) as t:
        H, W = t.series[0].shape[-2:]
    geo = Geometry(float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]),
                   (H - roi_size) // 2, (W - roi_size) // 2, roi_size)
    a, b = geo.local_to_um(0, 0)
    c, d = geo.local_to_um(roi_size, roi_size)
    log.info(f"{geo.sx:.4f} px/um -> {geo.um_per_px:.5f} um/px   mosaic {H} x {W}")
    log.info(f"ROI y[{geo.y0},{geo.y0+roi_size}) x[{geo.x0},{geo.x0+roi_size})"
             f"   x {a:.1f}..{c:.1f} um   y {b:.1f}..{d:.1f} um")
    return geo


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------


@dataclass
class Transcripts:
    ux: np.ndarray
    uy: np.ndarray
    gene: np.ndarray
    genes: list
    gt_label: np.ndarray
    source: str
    has_gt_labels: bool


def load_transcripts(root: Path, geo: Geometry, gt_mask: np.ndarray) -> Transcripts:
    rule("TRANSCRIPTS")
    cache = find(root / "_roi_cache", f"tx_roi_{geo.y0}_{geo.x0}_{geo.size}.parquet",
                 f"tx_roi_{geo.y0}_{geo.x0}_{geo.size}.csv.gz")
    if cache is None:
        raise SystemExit(
            f"no ROI transcript cache under {root/'_roi_cache'}. The pipelines "
            "write it during --stage prep; run one of them first so that every "
            "method is scored against exactly the same transcript set.")
    df = read_tab(cache)
    log.info(f"source      {cache.name}   {len(df):,} rows")
    ux = df.global_x.to_numpy(float)
    uy = df.global_y.to_numpy(float)
    gname = df.gene.astype(str).to_numpy()
    genes = sorted(set(gname))
    gi = {g: i for i, g in enumerate(genes)}
    gene = np.array([gi[g] for g in gname], np.int32)
    log.info(f"            {len(genes):,} targets")

    lx, ly = geo.um_to_local(ux, uy)
    r = np.clip(ly.astype(np.int64), 0, gt_mask.shape[0] - 1)
    c = np.clip(lx.astype(np.int64), 0, gt_mask.shape[1] - 1)
    gt = gt_mask[r, c].astype(np.int64)
    gt[gt < 0] = 0
    log.info(f"platform labels from the rasterised cell_boundaries: "
             f"{(gt>0).mean()*100:.1f}% of transcripts assigned, "
             f"{len(np.unique(gt[gt>0])):,} distinct cells")
    log.info("            this is a genuine per-transcript ground truth, not a "
             "prior column borrowed from one of the methods")
    return Transcripts(ux, uy, gene, genes, gt, str(cache), (gt > 0).mean() > 0.05)


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    ids: np.ndarray
    tx_ids: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    total_counts: np.ndarray
    gene_matrix: np.ndarray
    counts_source: str


def load_gt_mask(root: Path, geo: Geometry, explicit=None) -> np.ndarray:
    """The platform label raster the pipelines already built during prep."""
    rule("GROUND TRUTH RASTER")
    import tifffile
    cands = [Path(explicit)] if explicit else []
    cands += [Path(p) / "platform_labels.tif" for p in
              ("/data/qiuyijia/ucs_merfish_lung_nuc",
               "/data/qiuyijia/proseg_merfish_lung_nuc",
               "/data/qiuyijia/comseg_merfish_lung",
               "/data/qiuyijia/ucs_merfish_lung",
               "/data/qiuyijia/proseg_merfish_lung")]
    p = next((q for q in cands if q.exists()), None)
    if p is None:
        raise SystemExit(
            "no platform_labels.tif found. It is written by any pipeline's "
            "--stage prep with --prior polygon; pass --gt-mask to point at one.")
    lab = tifffile.imread(p)
    if lab.shape != (geo.size, geo.size):
        raise SystemExit(f"{p} is {lab.shape}, expected "
                         f"({geo.size}, {geo.size})")
    log.info(f"{p}")
    log.info(f"  {int(np.unique(lab[lab>0]).size):,} cells, coverage "
             f"{(lab>0).mean()*100:.2f}%")
    if (lab > 0).mean() < 0.3:
        log.info("  !! low coverage for a cell-boundary raster; if this file was "
                 "built with --prior centroid it holds nucleus disks, not cells, "
                 "and the transcript-level ground truth would be truncated. "
                 "Point --gt-mask at a polygon-prior directory.")
    return lab


def load_ground_truth(root: Path, geo: Geometry, tx: Transcripts) -> GroundTruth:
    rule("PLATFORM CELLS")
    md = find(root, "cell_metadata.csv*")
    d = pd.read_csv(md)
    d = d.rename(columns={d.columns[0]: "cell"})
    keep = geo.inside(d.center_x.to_numpy(float), d.center_y.to_numpy(float))
    d = d.loc[keep].reset_index(drop=True)
    ids = pd.to_numeric(d.cell, errors="coerce").fillna(-1).to_numpy(np.int64)
    ux = d.center_x.to_numpy(float)
    uy = d.center_y.to_numpy(float)
    log.info(f"{md.name}: {len(d):,} cells with their centroid inside the ROI"
             f"   <- GT cell count")

    # the raster stores cell_id % 2147483647, and the ids here are small, so the
    # two spaces coincide; verify rather than assume
    lab_ids = np.unique(tx.gt_label[tx.gt_label > 0])
    hit = float(np.isin(ids, lab_ids).mean())
    log.info(f"id spaces   {hit*100:.1f}% of the metadata ids occur in the raster")
    tx_ids = ids.copy()
    if hit < 0.8:
        from scipy.spatial import cKDTree
        log.info("            bridging by centroid instead")
        sel = tx.gt_label > 0
        u, inv = np.unique(tx.gt_label[sel], return_inverse=True)
        cx = np.bincount(inv, weights=tx.ux[sel]) / np.bincount(inv)
        cy = np.bincount(inv, weights=tx.uy[sel]) / np.bincount(inv)
        dd, jj = cKDTree(np.column_stack([cx, cy])).query(
            np.column_stack([ux, uy]), k=1)
        tx_ids = np.where(dd <= 5.0, u[jj], -1)
        log.info(f"            {int((tx_ids>0).sum()):,} of {len(ids):,} bridged")

    pos = {int(v): i for i, v in enumerate(tx_ids) if v > 0}
    gm = np.zeros((len(ids), len(tx.genes)), np.float32)
    sel = tx.gt_label > 0
    rows = np.array([pos.get(int(v), -1) for v in tx.gt_label[sel]])
    ok = rows >= 0
    np.add.at(gm, (rows[ok], tx.gene[sel][ok]), 1.0)
    tot = gm.sum(axis=1)
    tot[tx_ids < 0] = np.nan
    log.info(f"GT counts/cell median {np.nanmedian(tot):.0f}  (from the platform "
             "boundaries, independent of every method's own assignment)")
    return GroundTruth(ids, tx_ids, ux, uy, tot, gm,
                       "rasterised cell_boundaries (genuine platform GT)")


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    labels: np.ndarray
    px_per_cell: float
    method: str
    uses_platform_prior: bool
    note: str = ""


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


def summarise_prediction(pred: Prediction, geo: Geometry,
                         tx: Transcripts) -> PredCells:
    rule(f"PREDICTION  {pred.method}")
    lab = pred.labels
    s = pred.px_per_cell
    um = s * geo.um_per_px
    log.info(f"grid        {lab.shape}   1 px = {s:.4f} mosaic px = {um:.4f} um"
             f"   span {lab.shape[0]*um:.1f} um")
    if pred.note:
        log.info(f"note        {pred.note}")

    flat = np.where(lab.reshape(-1) < 0, 0, lab.reshape(-1)).astype(np.int64)
    cnt = np.bincount(flat)
    ids = np.nonzero(cnt)[0]
    ids = ids[ids > 0]
    if len(ids) == 0:
        raise SystemExit("the label image is empty")
    area = cnt[ids].astype(float) * um * um
    cov = float((lab > 0).mean())
    log.info(f"cells       {len(ids):,}   coverage {cov*100:.2f}%")

    rows, cols = np.nonzero(lab)
    lv = lab[rows, cols].astype(np.int64)
    o = np.argsort(lv, kind="stable")
    lv, rows, cols = lv[o], rows[o], cols[o]
    b = np.append(np.searchsorted(lv, ids), len(lv))
    cy = np.array([rows[b[i]:b[i+1]].mean() for i in range(len(ids))])
    cx = np.array([cols[b[i]:b[i+1]].mean() for i in range(len(ids))])
    pux, puy = geo.local_to_um((cx + 0.5) * s, (cy + 0.5) * s)
    med = float(np.median(area))
    log.info(f"area        median {med:.1f} um^2   equivalent diameter "
             f"{2*np.sqrt(med/np.pi):.2f} um")

    lx, ly = geo.um_to_local(tx.ux, tx.uy)
    r = np.clip((ly / s).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip((lx / s).astype(np.int64), 0, lab.shape[1] - 1)
    tl = lab[r, c].astype(np.int64)
    tl[tl < 0] = 0
    log.info(f"transcripts {len(tx.ux):,} in ROI, {int((tl>0).sum()):,} assigned "
             f"({(tl>0).mean()*100:.1f}%)")

    pos = {int(v): i for i, v in enumerate(ids)}
    ntx = np.zeros(len(ids), np.int64)
    gm = np.zeros((len(ids), len(tx.genes)), np.float32)
    sel = tl > 0
    ri = np.array([pos.get(int(v), -1) for v in tl[sel]])
    ok = ri >= 0
    np.add.at(ntx, ri[ok], 1)
    np.add.at(gm, (ri[ok], tx.gene[sel][ok]), 1.0)
    log.info(f"counts/cell median {np.median(ntx):.0f}")
    return PredCells(ids, pux, puy, area, ntx, gm, cov, tl)

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



METHOD = "Proseg"
# ==========================================================================
# Proseg
# ==========================================================================

def add_args(ap):
    ap.add_argument("--run-dir", default="/data/qiuyijia/proseg_merfish_lung_nuc")
    ap.add_argument("--mask", default=None)


def build(args, geo):
    import tifffile
    d = Path(args.run_dir)
    p = Path(args.mask) if args.mask else d / "proseg_segmentation.tif"
    if not p.exists():
        raise SystemExit(f"{p} missing; run --stage post")
    lab = np.squeeze(tifffile.imread(p)).astype(np.int64)
    log.info(f"mask        {p}   {lab.shape}")
    note = ""
    f = d / "prep_meta.json"
    if f.exists():
        m = json.loads(f.read_text())
        note = f"prior={m.get('prior')}"
        log.info(f"prep_meta   {note}")
    s = args.roi_size / lab.shape[0]
    return Prediction(lab, s, "Proseg", uses_platform_prior=True, note=note)


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------


def evaluate(pred: Prediction, root: Path, out_dir: Path, geo, tx, gt) -> dict:
    pc = summarise_prediction(pred, geo, tx)

    rule("MATCHING")
    radius = infer_match_radius(gt.ux, gt.uy)
    log.info(f"match radius {radius:.2f} um  (GT nearest-neighbour p95 x 1.25)")
    mp, mg, md = match_sparse_hungarian(pc.ux, pc.uy, gt.ux, gt.uy, radius)
    n = len(mp)
    log.info(f"matched     {n:,} pairs of {len(pc.ids):,} predicted and "
             f"{len(gt.ux):,} GT")
    prec = n / max(len(pc.ids), 1)
    rec = n / max(len(gt.ux), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    log.info(f"precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    if n:
        log.info(f"centroid shift mean {md.mean():.2f}  median {np.median(md):.2f}"
                 f"  p95 {np.percentile(md,95):.2f} um")

    row = {
        "method": pred.method, "dataset": DATASET,
        "Pred cell count": len(pc.ids), "GT cell count": len(gt.ux),
        "Cell count ratio": len(pc.ids) / max(len(gt.ux), 1),
        "Matched cell count": n,
        "Detection precision": prec, "Detection recall": rec, "Detection F1": f1,
        "Mean centroid shift": float(md.mean()) if n else np.nan,
        "Median centroid shift": float(np.median(md)) if n else np.nan,
        "p95 centroid shift": float(np.percentile(md, 95)) if n else np.nan,
        "Pred transcript rows": int((pc.tx_label > 0).sum()),
        "GT transcript rows": int(len(tx.ux)),
        "Pred transcript assignment rate": float((pc.tx_label > 0).mean()),
        "Transcript matched cell pairs": n,
        "median_area_um2": float(np.median(pc.area_um2)),
        "median_diameter_um": float(2 * np.sqrt(np.median(pc.area_um2) / np.pi)),
        "coverage_fraction": pc.coverage,
        "uses_platform_prior": pred.uses_platform_prior,
        "gt_counts_source": gt.counts_source,
    }

    if n:
        pn = pc.n_tx[mp].astype(float)
        gtot = gt.total_counts[mg].astype(float)
        row.update(numeric_pair_metrics(
            pn, gtot, "matched_pair_n_transcripts_vs_total_counts"))
        row.update(numeric_pair_metrics(pn, gtot, "matched_cell_transcript_count"))
        row.update(gene_vector_metrics(pc.gene_matrix[mp], gt.gene_matrix[mg]))
        log.info(f"gene vector cosine "
                 f"{row['matched_cell_gene_vector_mean_cosine']:.4f} over "
                 f"{row['matched_cell_gene_vector_valid_pairs']:,} pairs")
    else:
        for p in ("matched_pair_n_transcripts_vs_total_counts",
                  "matched_cell_transcript_count"):
            for k in ("pearson", "spearman", "mae", "rmse"):
                row[f"{p}_{k}"] = np.nan
        row.update({"matched_cell_gene_vector_mean_cosine": np.nan,
                    "matched_cell_gene_vector_mean_js_distance": np.nan,
                    "matched_cell_gene_vector_mean_pearson": np.nan,
                    "matched_cell_gene_vector_valid_pairs": 0})

    if tx.has_gt_labels and n:
        m = {int(pc.ids[a]): int(gt.tx_ids[b])
             for a, b in zip(mp, mg) if int(gt.tx_ids[b]) > 0}
        both = (pc.tx_label > 0) & (tx.gt_label > 0)
        mapped = np.array([m.get(int(v), -1) for v in pc.tx_label[both]])
        ok = mapped >= 0
        ov = int(ok.sum())
        acc = float((mapped[ok] == tx.gt_label[both][ok]).mean()) if ov else np.nan
        row["transcript_id_overlap_n"] = ov
        row["transcript_assignment_accuracy_via_matched_cells"] = acc
        row["Overall correct fraction"] = (
            row["Pred transcript assignment rate"] * acc
            if np.isfinite(acc) else np.nan)
        log.info(f"transcript accuracy {acc:.4f} over {ov:,} rows -> overall "
                 f"correct {row['Overall correct fraction']:.4f}")
    else:
        row["transcript_id_overlap_n"] = 0
        row["transcript_assignment_accuracy_via_matched_cells"] = np.nan
        row["Overall correct fraction"] = np.nan

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{pred.method}_{DATASET}"
    pd.DataFrame([{k: row.get(k, np.nan) for k in REQUESTED_METRICS}]).to_csv(
        out_dir / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame([{k: row.get(k, np.nan) for k in SELECTED}]).to_csv(
        out_dir / f"{tag}_sparse_hungarian_qc_metrics_selected.csv", index=False)
    json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
               for k, v in row.items()},
              open(out_dir / f"{tag}_sparse_hungarian_qc_metrics_full.json", "w"),
              indent=2, default=str)
    if n:
        pd.DataFrame({"pred_id": pc.ids[mp], "gt_id": gt.ids[mg],
                      "centroid_shift_um": md,
                      "pred_n_transcripts": pc.n_tx[mp],
                      "gt_total_counts": gt.total_counts[mg],
                      "pred_area_um2": pc.area_um2[mp]}).to_csv(
            out_dir / f"{tag}_matched_pairs.csv", index=False)

    rule("RESULT")
    for k in SELECTED:
        v = row.get(k)
        log.info(f"  {k:52s} {v:.4f}" if isinstance(v, float) else
                 f"  {k:52s} {v}")
    log.info(f"-> {out_dir}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    return row


def main_loop(method: str, build):
    ap = argparse.ArgumentParser(
        description=f"Evaluate {method} on the MERFISH lung ROI")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--out-dir", default="/data/qiuyijia/eval_merfish_lung")
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--gt-mask", default=None,
                    help="platform_labels.tif built with --prior polygon")
    ap.add_argument("--version", action="store_true")
    add_args(ap)
    args = ap.parse_args()
    if args.version:
        print(VERSION)
        return
    root, out = Path(args.root), Path(args.out_dir)
    setup_log(out, f"eval_{method}_lung")
    rule(f"{method.upper()}  x  MERFISH lung   core v{VERSION}")
    geo = load_geometry(root, args.roi_size)
    gt_mask = load_gt_mask(root, geo, args.gt_mask)
    tx = load_transcripts(root, geo, gt_mask)
    gt = load_ground_truth(root, geo, tx)
    pred = build(args, geo)
    evaluate(pred, root, out, geo, tx, gt)


if __name__ == "__main__":
    main_loop(METHOD, build)