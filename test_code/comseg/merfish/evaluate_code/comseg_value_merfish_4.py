#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lung_comseg.py   v2026.08.16

ComSeg on MERFISH lung cancer (merfish_4), scored with the same 19 metrics as
every other row in the benchmark workbook.

Ground truth here is genuinely independent, which is not true of the two liver
sections.  MERFISH_Lung_cancer ships cell_boundaries/*.hdf5, so the per-cell
polygons come from the platform, get rasterised once, and are used both for the
cell list and for the per-transcript labels.  The liver rows had to borrow
Proseg's prior column for that, which makes their assignment metrics partly
circular; this dataset has no such problem.

    python eval_lung_comseg.py
    python eval_lung_comseg.py --pred /path/to/comseg_labels_native.tif

Two things worth knowing about the numbers this prints.

  * ComSeg labels transcripts, not pixels, so its raster is a point cloud until
    it is dilated.  The runner writes the filled version; median area is
    therefore sensitive to that fill radius.  Both the raw occupancy and the
    filled one appear in the runner's post log.

  * ComSeg is seeded from a nucleus prior derived from the platform cells, so
    its detection columns are closer to circular than a reference-free method's.
    The transcript-level columns are the informative ones.
"""

from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = "/data/qiuyijia/dataset/MERFISH_Lung_cancer"
RUN = "/data/qiuyijia/comseg_merfish_lung"
GT_MASK = "/data/qiuyijia/ucs_merfish_lung/platform_labels.tif"
OUT = "/data/qiuyijia/eval_merfish_lung"
ROI = (31434, 36213, 10000)          # y0, x0, size in mosaic px
UM_PER_PX = 0.108                    # confirmed from the dataset transform
METHOD, DATASET = "ComSeg", "merfish_4"
# ---- metric core (identical in every evaluator; kept inline so each file runs
# ---- on its own) --------------------------------------------------------- #

def log(m=""):
    print(f"{time.strftime('%H:%M:%S')}  {m}", flush=True)


def banner(m):
    log("=" * 78); log(m); log("=" * 78)


def props(lab):
    """centroid (x, y) in raster px and pixel area, per label."""
    a = np.asarray(lab)
    flat = a.reshape(-1).astype(np.int64)
    n = int(flat.max())
    if n == 0:
        return np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros(0)
    area = np.bincount(flat, minlength=n + 1)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    sy = np.bincount(flat, weights=yy.reshape(-1), minlength=n + 1)
    sx = np.bincount(flat, weights=xx.reshape(-1), minlength=n + 1)
    ids = np.nonzero(area)[0]
    ids = ids[ids > 0]
    cen = np.stack([sx[ids] / area[ids], sy[ids] / area[ids]], 1)
    return ids, cen, area[ids].astype(float)


def match_radius(gt_xy):
    from scipy.spatial import cKDTree
    if len(gt_xy) < 2:
        return 20.0
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    return float(np.percentile(d[:, 1], 95) * 1.25)


def sparse_hungarian(pred_xy, gt_xy, radius):
    """Pairs within `radius`, solved exactly inside each connected component of
    the candidate graph; components above `BIG` fall back to greedy."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import linear_sum_assignment
    BIG = 3000
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    pairs = cKDTree(pred_xy).query_ball_tree(cKDTree(gt_xy), radius)
    pi, gi = [], []
    for p, gs in enumerate(pairs):
        for g in gs:
            pi.append(p); gi.append(g)
    if not pi:
        return np.zeros((0, 2), int), np.zeros(0)
    pi = np.asarray(pi); gi = np.asarray(gi)
    d = np.linalg.norm(pred_xy[pi] - gt_xy[gi], axis=1)
    npred, ngt = len(pred_xy), len(gt_xy)
    A = coo_matrix((np.ones(len(pi)), (pi, gi + npred)),
                   shape=(npred + ngt, npred + ngt))
    ncc, lbl = connected_components(A + A.T, directed=False)
    out_p, out_g, out_d = [], [], []
    big = 0
    for c in range(ncc):
        m = lbl[pi] == c
        if not m.any():
            continue
        sp = np.unique(pi[m]); sg = np.unique(gi[m])
        if len(sp) * len(sg) > BIG * BIG:
            big += 1
            order = np.argsort(d[m])
            up, ug = set(), set()
            for k in np.nonzero(m)[0][order]:
                if pi[k] in up or gi[k] in ug:
                    continue
                up.add(pi[k]); ug.add(gi[k])
                out_p.append(pi[k]); out_g.append(gi[k]); out_d.append(d[k])
            continue
        rp = {v: i for i, v in enumerate(sp)}
        rg = {v: i for i, v in enumerate(sg)}
        C = np.full((len(sp), len(sg)), 1e6)
        for k in np.nonzero(m)[0]:
            C[rp[pi[k]], rg[gi[k]]] = d[k]
        r, cidx = linear_sum_assignment(C)
        for a, b in zip(r, cidx):
            if C[a, b] < 1e5:
                out_p.append(sp[a]); out_g.append(sg[b]); out_d.append(C[a, b])
    if big:
        log(f"    {big} very large component(s) matched greedily")
    return np.stack([np.asarray(out_p), np.asarray(out_g)], 1), np.asarray(out_d)


def pair_stats(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return dict(pearson=np.nan, spearman=np.nan, mae=np.nan, rmse=np.nan)
    from scipy.stats import pearsonr, spearmanr
    return dict(pearson=float(pearsonr(a, b)[0]),
                spearman=float(spearmanr(a, b)[0]),
                mae=float(np.mean(np.abs(a - b))),
                rmse=float(np.sqrt(np.mean((a - b) ** 2))))


def vector_stats(P, G):
    """cosine / Jensen-Shannon / pearson between paired gene vectors."""
    out = dict(cosine=[], js=[], pearson=[])
    for p, g in zip(P, G):
        p = np.asarray(p, float); g = np.asarray(g, float)
        if p.sum() <= 0 or g.sum() <= 0:
            continue
        out["cosine"].append(float(p @ g / (np.linalg.norm(p) * np.linalg.norm(g))))
        pp, gg = p / p.sum(), g / g.sum()
        mm = 0.5 * (pp + gg)
        def kl(u, v):
            m = u > 0
            return float(np.sum(u[m] * np.log2(u[m] / v[m])))
        out["js"].append(float(np.sqrt(max(0.5 * kl(pp, mm) + 0.5 * kl(gg, mm), 0))))
        if p.std() > 0 and g.std() > 0:
            out["pearson"].append(float(np.corrcoef(p, g)[0, 1]))
    return (float(np.mean(out["cosine"])) if out["cosine"] else np.nan,
            float(np.mean(out["js"])) if out["js"] else np.nan,
            float(np.mean(out["pearson"])) if out["pearson"] else np.nan,
            len(out["cosine"]))


def cell_by_gene(labels, gene_idx, n_genes, n_cells_max):
    """rows are label-1, columns are gene index; labels of 0 are dropped."""
    from scipy.sparse import coo_matrix
    m = labels > 0
    if n_cells_max < 1:
        return coo_matrix((1, n_genes)).tocsr()
    return coo_matrix((np.ones(int(m.sum()), np.float32),
                       (labels[m] - 1, gene_idx[m])),
                      shape=(n_cells_max, n_genes)).tocsr()


# --------------------------------------------------------------------------- #

def transform_um_per_px(root: Path) -> float:
    for p in root.rglob("*micron_to_mosaic_pixel_transform.csv"):
        M = np.loadtxt(p)
        return float(1.0 / M[0, 0])
    return UM_PER_PX


def find_pred(explicit, run: Path) -> Path:
    if explicit:
        return Path(explicit)
    for n in ("comseg_labels_native.tif", "comseg_segmentation.tif",
              "comseg_labels.tif"):
        p = run / n
        if p.exists():
            return p
    c = sorted(run.glob("*.tif"))
    if c:
        return c[0]
    sys.exit(f"no ComSeg raster under {run}; run --stage post first "
             f"(and comseg_salvage.py before it if the tiles predate the fix)")


def load_raster(path: Path, roi_px: int, um_per_px: float, what: str):
    import tifffile
    a = tifffile.imread(path)
    a = np.squeeze(a)
    if a.ndim != 2:
        sys.exit(f"{path} is not a 2-D label image ({a.shape})")
    scale = roi_px / a.shape[0]                 # mosaic px per raster px
    um = scale * um_per_px
    ids, cen, area = props(a)
    log(f"  {what:11s} {path.name}  {a.shape}  1 px = {scale:.3f} mosaic px "
        f"= {um:.4f} um")
    log(f"              {len(ids):,} labels   coverage {(a>0).mean()*100:.2f}%"
        f"   median area {np.median(area)*um*um:.1f} um^2"
        f" -> diameter {2*np.sqrt(np.median(area)*um*um/np.pi):.2f} um")
    return a, ids, cen, area, scale, um


def read_transcripts(root: Path, roi, um_per_px, cache: Path) -> pd.DataFrame:
    y0, x0, s = roi
    cp = cache / f"tx_roi_{y0}_{x0}_{s}.parquet"
    if cp.exists():
        log(f"  cache       {cp.name}")
        return pd.read_parquet(cp)
    src = None
    for n in ("detected_transcripts.csv", "detected_transcripts.csv.gz"):
        if (root / n).exists():
            src = root / n
            break
    if src is None:
        sys.exit("detected_transcripts.csv not found")
    M = None
    for p in root.rglob("*micron_to_mosaic_pixel_transform.csv"):
        M = np.loadtxt(p)
        break
    parts, tot = [], 0
    for d in pd.read_csv(src, chunksize=2_000_000):
        tot += len(d)
        px = d.global_x.to_numpy() * M[0, 0] + M[0, 2]
        py = d.global_y.to_numpy() * M[1, 1] + M[1, 2]
        m = (px >= x0) & (px < x0 + s) & (py >= y0) & (py < y0 + s)
        if m.any():
            parts.append(pd.DataFrame({"px": px[m] - x0, "py": py[m] - y0,
                                       "gene": d.gene.to_numpy()[m]}))
    t = pd.concat(parts, ignore_index=True)
    log(f"  scanned {tot:,} rows, {len(t):,} inside the ROI")
    cache.mkdir(parents=True, exist_ok=True)
    t.to_parquet(cp)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None)
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--gt-mask", default=GT_MASK)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--min-gt-coverage", type=float, default=0.30,
                    help="reject a GT raster covering less of the field than "
                         "this; a nucleus-disk raster slipped through once and "
                         "silently truncated every assignment metric")
    args = ap.parse_args()

    root, run = Path(ROOT), Path(args.run_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    banner(f"{args.method} x MERFISH lung ({DATASET})   v2026.08.16")
    um_per_px = transform_um_per_px(root)
    y0, x0, S = ROI
    log(f"  ROI         y[{y0},{y0+S}) x[{x0},{x0+S}) mosaic px "
        f"= {S*um_per_px:.0f} x {S*um_per_px:.0f} um at {um_per_px:.5f} um/px")

    banner("PREDICTION")
    pred_p = find_pred(args.pred, run)
    pred, pid, pxy, parea, pscale, pum = load_raster(pred_p, S, um_per_px, "prediction")

    banner("GROUND TRUTH")
    gt_p = Path(args.gt_mask)
    if not gt_p.exists():
        sys.exit(f"{gt_p} missing; any runner's --stage prep --prior polygon "
                 f"writes it")
    gt, gid, gxy, garea, gscale, gum = load_raster(gt_p, S, um_per_px, "platform")
    cov = (gt > 0).mean()
    if cov < args.min_gt_coverage:
        sys.exit(f"  the GT raster covers only {cov*100:.1f}% of the field.  "
                 f"That is the signature of a nucleus-disk prior rather than the "
                 f"cell outlines, and it truncates every assignment metric.  "
                 f"Pass --gt-mask pointing at a polygon-prior platform_labels.tif.")
    gdiam = 2 * np.sqrt(np.median(garea) * gum * gum / np.pi)
    log(f"  platform cells {len(gid):,}, median diameter {gdiam:.2f} um "
        f"<- this is the reference, rasterised from cell_boundaries")

    banner("TRANSCRIPTS")
    tx = read_transcripts(root, ROI, um_per_px, run / "cache")
    py_p = np.clip((tx.py.to_numpy() / pscale).astype(int), 0, pred.shape[0] - 1)
    px_p = np.clip((tx.px.to_numpy() / pscale).astype(int), 0, pred.shape[1] - 1)
    py_g = np.clip((tx.py.to_numpy() / gscale).astype(int), 0, gt.shape[0] - 1)
    px_g = np.clip((tx.px.to_numpy() / gscale).astype(int), 0, gt.shape[1] - 1)
    lp = pred[py_p, px_p].astype(np.int64)
    lg = gt[py_g, px_g].astype(np.int64)
    rate_p = float((lp > 0).mean()); rate_g = float((lg > 0).mean())
    log(f"  {len(tx):,} transcripts;  prediction assigns {rate_p*100:.1f}%,  "
        f"the platform outlines cover {rate_g*100:.1f}%")

    genes = tx.gene.astype(str).to_numpy()
    uniq = np.unique(genes)
    gindex = {g: i for i, g in enumerate(uniq)}
    gi = np.array([gindex[g] for g in genes])
    Cp = cell_by_gene(lp, gi, len(uniq), int(pred.max()))
    Cg = cell_by_gene(lg, gi, len(uniq), int(gt.max()))
    log(f"  {len(uniq)} genes; counts/cell median pred "
        f"{np.median(np.asarray(Cp.sum(1)).ravel()[pid-1]):.0f}, GT "
        f"{np.median(np.asarray(Cg.sum(1)).ravel()[gid-1]):.0f}")

    banner("MATCHING")
    pxy_um = pxy * pum
    gxy_um = gxy * gum
    R = match_radius(gxy_um)
    log(f"  match radius {R:.2f} um  (GT nearest-neighbour p95 x 1.25)")
    pairs, dist = sparse_hungarian(pxy_um, gxy_um, R)
    nm = len(pairs)
    prec = nm / max(len(pid), 1); rec = nm / max(len(gid), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    log(f"  matched {nm:,} of {len(pid):,} predicted and {len(gid):,} GT")
    log(f"  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    log(f"  centroid shift mean {dist.mean():.2f} median "
        f"{np.median(dist):.2f} p95 {np.percentile(dist,95):.2f} um")

    plab = pid[pairs[:, 0]]; glab = gid[pairs[:, 1]]
    cnt = pair_stats(np.asarray(Cp.sum(1)).ravel()[plab - 1],
                     np.asarray(Cg.sum(1)).ravel()[glab - 1])
    cos, js, vpe, nvec = vector_stats(Cp[plab - 1].toarray(), Cg[glab - 1].toarray())

    p2g = np.zeros(int(pred.max()) + 1, np.int64)
    p2g[plab] = glab
    both = (lp > 0) & (lg > 0)
    acc = float((p2g[lp[both]] == lg[both]).mean()) if both.any() else np.nan
    log(f"  transcript accuracy {acc:.4f} over {int(both.sum()):,} rows "
        f"-> overall correct {rate_p*acc:.4f}")

    banner("RESULT")
    row = {
        "method": args.method, "dataset": DATASET,
        "GT cell count": len(gid), "Pred cell count": len(pid),
        "Cell count ratio": len(pid) / max(len(gid), 1),
        "Matched cell count": nm,
        "Detection precision": prec, "Detection recall": rec, "Detection F1": f1,
        "Mean centroid shift": float(dist.mean()),
        "Median centroid shift": float(np.median(dist)),
        "p95 centroid shift": float(np.percentile(dist, 95)),
        "matched_cell_transcript_count_pearson": cnt["pearson"],
        "matched_cell_transcript_count_spearman": cnt["spearman"],
        "matched_cell_transcript_count_mae": cnt["mae"],
        "matched_cell_transcript_count_rmse": cnt["rmse"],
        "matched_cell_gene_vector_mean_cosine": cos,
        "matched_cell_gene_vector_mean_js_distance": js,
        "matched_cell_gene_vector_mean_pearson": vpe,
        "matched_cell_gene_vector_valid_pairs": nvec,
        "transcript_id_overlap_n": int(both.sum()),
        "transcript_assignment_accuracy_via_matched_cells": acc,
        "Pred transcript rows": int((lp > 0).sum()),
        "GT transcript rows": len(tx),
        "Pred transcript assignment rate": rate_p,
        "Overall correct fraction": rate_p * acc,
        "median_area_um2": float(np.median(parea) * pum * pum),
        "median_diameter_um": float(2 * np.sqrt(np.median(parea) * pum * pum / np.pi)),
        "coverage_fraction": float((pred > 0).mean()),
        "uses_platform_prior": True,
        "gt_counts_source": "rasterised cell_boundaries (genuine platform GT)",
    }
    for k, v in row.items():
        log(f"    {k:52s} {v}")
    tag = f"{args.method.lower()}_{DATASET}"
    pd.DataFrame([row]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame({"pred_id": plab, "gt_id": glab, "dist_um": dist}).to_csv(
        out / f"{tag}_matched_pairs.csv", index=False)
    log(f"  -> {out}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    log("note: ComSeg is seeded from a nucleus prior derived from the platform "
        "cells, so the detection columns are close to circular; the "
        "transcript-level columns carry the signal for this row")


if __name__ == "__main__":
    main()