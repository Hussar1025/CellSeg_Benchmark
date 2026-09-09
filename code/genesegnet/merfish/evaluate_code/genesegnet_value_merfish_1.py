#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_value_merfish_1_fullmetrics.py

Full independent evaluation for GeneSegNet on MERFISH1 center 20000x20000 ROI.

Prediction:
  /data/qiuyijia/genesegnet_merfish_1_infer/genesegnet_merfish1_roi20000_mask.tif

Ground truth:
  official MERFISH cell metadata + official cell-boundary polygons in
  /data/qiuyijia/dataset/merfish_mouse_brain/cell_boundaries/*.hdf5

This evaluator reconstructs true GT transcript->cell assignment from official
cell polygons. It therefore computes detection, localization, count fidelity,
assignment accuracy, overall correctness, and matched-cell expression-vector
metrics without nearest-centroid pseudo-GT.

Official boundaries are used ONLY for evaluation; GeneSegNet inference itself
remains prior-free.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/data/qiuyijia/dataset/merfish_mouse_brain")

MASK_DEFAULT = Path(
    "/data/qiuyijia/genesegnet_merfish_1_infer/"
    "genesegnet_merfish1_roi20000_mask.tif"
)
META_DEFAULT = ROOT / (
    "datasets_mouse_brain_map_BrainReceptorShowcase_"
    "Slice1_Replicate1_cell_metadata_S1R1.csv"
)
TX_DEFAULT = ROOT / (
    "datasets_mouse_brain_map_BrainReceptorShowcase_"
    "Slice1_Replicate1_detected_transcripts_S1R1.csv"
)
BOUNDARY_DIR_DEFAULT = ROOT / "cell_boundaries"
TRANSFORM_DEFAULT = ROOT / "images" / (
    "datasets_mouse_brain_map_BrainReceptorShowcase_"
    "Slice1_Replicate1_images_micron_to_mosaic_pixel_transform.csv"
)
OUT_DEFAULT = Path("/data/qiuyijia/eval_results_qv20/merfish_1")

H, W = 61310, 89085
ROI_SIZE = 20000
ROI_Y0 = (H - ROI_SIZE) // 2
ROI_X0 = (W - ROI_SIZE) // 2
ROI_Y1 = ROI_Y0 + ROI_SIZE
ROI_X1 = ROI_X0 + ROI_SIZE
ZINDEX = 3


def safe_corr(a, b, kind="pearson"):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(a, b)[0])


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else np.nan


def js_distance(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    p = a / a.sum()
    q = b / b.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        z = x > 0
        return np.sum(x[z] * np.log(x[z] / np.maximum(y[z], 1e-300)))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def load_transform(path):
    M = np.loadtxt(path)
    if M.shape != (3, 3):
        raise RuntimeError(f"unexpected transform shape {M.shape}")
    return M


def micron_to_px(x, y, M):
    q = np.c_[x, y, np.ones(len(x), float)]
    o = q @ M.T
    return o[:, 0], o[:, 1]


def mask_properties(mask, px_um):
    fg = mask > 0
    labs, _ = np.unique(mask[fg], return_counts=True)
    yy, xx = np.nonzero(fg)
    vals = mask[yy, xx].astype(np.int64)

    lab2row = {int(v): i for i, v in enumerate(labs)}
    row = np.fromiter(
        (lab2row[int(v)] for v in vals),
        dtype=np.int64,
        count=len(vals),
    )

    n = len(labs)
    area_px = np.bincount(row, minlength=n).astype(float)
    cx = np.bincount(row, weights=xx, minlength=n) / np.maximum(area_px, 1)
    cy = np.bincount(row, weights=yy, minlength=n) / np.maximum(area_px, 1)

    area_um2 = area_px * px_um * px_um
    diam_um = 2 * np.sqrt(area_um2 / np.pi)

    return labs.astype(np.int64), area_px, np.c_[cx, cy], area_um2, diam_um


def sparse_hungarian(pred, gt, radius):
    if len(pred) == 0 or len(gt) == 0:
        return np.array([], int), np.array([], int), np.array([], float)

    tp = cKDTree(pred)
    tg = cKDTree(gt)
    neigh = tp.query_ball_tree(tg, r=radius)

    p2g = {i: js for i, js in enumerate(neigh) if js}
    g2p = defaultdict(list)
    for i, js in p2g.items():
        for j in js:
            g2p[j].append(i)

    seen = set()
    mp, mg, md = [], [], []

    for start in p2g:
        if start in seen:
            continue

        ps, gs = set(), set()
        stack = [start]

        while stack:
            i = stack.pop()
            if i in ps:
                continue
            ps.add(i)
            seen.add(i)

            for j in p2g.get(i, []):
                if j not in gs:
                    gs.add(j)
                    stack.extend(ii for ii in g2p[j] if ii not in ps)

        pl = sorted(ps)
        gl = sorted(gs)
        gj = {g: k for k, g in enumerate(gl)}
        C = np.full((len(pl), len(gl)), radius * 1000.0)

        for r, i in enumerate(pl):
            for j in p2g[i]:
                C[r, gj[j]] = np.linalg.norm(pred[i] - gt[j])

        rr, cc = linear_sum_assignment(C)

        for r, c in zip(rr, cc):
            if C[r, c] <= radius:
                mp.append(pl[r])
                mg.append(gl[c])
                md.append(C[r, c])

    return np.asarray(mp, int), np.asarray(mg, int), np.asarray(md, float)


def load_gt_metadata(meta_path, M):
    m = pd.read_csv(meta_path)

    if "Unnamed: 0" not in m.columns:
        raise KeyError("metadata does not contain 'Unnamed: 0' cell ID")

    m["cell_id"] = m["Unnamed: 0"].astype(str)

    x_um = pd.to_numeric(m["center_x"], errors="coerce").to_numpy(float)
    y_um = pd.to_numeric(m["center_y"], errors="coerce").to_numpy(float)

    px, py = micron_to_px(x_um, y_um, M)

    inside = (
        np.isfinite(px) & np.isfinite(py) &
        (px >= ROI_X0) & (px < ROI_X1) &
        (py >= ROI_Y0) & (py < ROI_Y1)
    )

    q = m.loc[inside].copy().reset_index(drop=True)
    q["mosaic_x_px"] = px[inside]
    q["mosaic_y_px"] = py[inside]
    q["roi_x_px"] = px[inside] - ROI_X0
    q["roi_y_px"] = py[inside] - ROI_Y0
    return q


def get_boundary_files(boundary_dir):
    fs = sorted(boundary_dir.glob("*.hdf5"))
    if not fs:
        fs = sorted(boundary_dir.rglob("*.hdf5"))
    if not fs:
        raise FileNotFoundError(f"no HDF5 boundary files under {boundary_dir}")
    return fs


def normalize_polygon_array(a):
    a = np.asarray(a)
    a = np.squeeze(a)

    if a.ndim != 2:
        return None

    if a.shape[1] != 2 and a.shape[0] == 2:
        a = a.T

    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
        return None

    return a.astype(float)


def read_roi_polygons(boundary_dir, gt):
    """
    Scan boundary HDF5 files once; keep only GT cells whose centroids are in ROI.
    """
    wanted = {str(cid): i for i, cid in enumerate(gt["cell_id"])}
    out = defaultdict(list)

    for hp in get_boundary_files(boundary_dir):
        with h5py.File(hp, "r") as h:
            if "featuredata" not in h:
                continue

            fd = h["featuredata"]
            ids = set(fd.keys()).intersection(wanted.keys())
            if not ids:
                continue

            for cid in ids:
                zname = f"zIndex_{ZINDEX}"
                if zname not in fd[cid]:
                    continue

                zg = fd[cid][zname]

                for pname in zg.keys():
                    if not str(pname).startswith("p_"):
                        continue
                    pg = zg[pname]
                    if "coordinates" not in pg:
                        continue

                    poly = normalize_polygon_array(pg["coordinates"])
                    if poly is not None:
                        out[wanted[cid]].append(poly)

    return out


def calibrate_fov_offsets(polys, gt):
    """
    Boundary coordinates observed in this dataset are micron-like and differ
    from metadata by a small per-FOV translation. Estimate that translation
    robustly using median polygon-centroid residual.
    """
    residuals = defaultdict(list)

    for gi, ps in polys.items():
        if not ps:
            continue

        vertices = np.concatenate(ps, axis=0)
        pc = np.mean(vertices, axis=0)

        gx = float(gt.iloc[gi]["center_x"])
        gy = float(gt.iloc[gi]["center_y"])
        fov = int(gt.iloc[gi]["fov"])

        residuals[fov].append((gx - pc[0], gy - pc[1]))

    offsets = {}

    for fov, vals in residuals.items():
        a = np.asarray(vals, float)
        offsets[fov] = (
            float(np.median(a[:, 0])),
            float(np.median(a[:, 1])),
        )

    return offsets


def prepare_polygon_objects(polys, gt, offsets, M):
    objects = []

    for gi, ps in polys.items():
        fov = int(gt.iloc[gi]["fov"])
        dx, dy = offsets.get(fov, (0.0, 0.0))

        for p in ps:
            pu = p.copy()
            pu[:, 0] += dx
            pu[:, 1] += dy

            px, py = micron_to_px(pu[:, 0], pu[:, 1], M)
            poly = np.c_[px - ROI_X0, py - ROI_Y0]

            xmin = float(np.min(poly[:, 0]))
            xmax = float(np.max(poly[:, 0]))
            ymin = float(np.min(poly[:, 1]))
            ymax = float(np.max(poly[:, 1]))

            if xmax < 0 or xmin >= ROI_SIZE or ymax < 0 or ymin >= ROI_SIZE:
                continue

            objects.append({
                "gt_idx": int(gi),
                "path": MplPath(poly, closed=True),
                "bbox": (xmin, xmax, ymin, ymax),
            })

    return objects


def assign_points_to_gt(x, y, objects, bin_px=128):
    """
    Exact point-in-polygon assignment with a spatial bin index.
    """
    poly_bins = defaultdict(list)

    for oi, o in enumerate(objects):
        xmin, xmax, ymin, ymax = o["bbox"]

        bx0 = max(0, int(math.floor(xmin / bin_px)))
        bx1 = min((ROI_SIZE - 1) // bin_px, int(math.floor(xmax / bin_px)))
        by0 = max(0, int(math.floor(ymin / bin_px)))
        by1 = min((ROI_SIZE - 1) // bin_px, int(math.floor(ymax / bin_px)))

        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                poly_bins[(bx, by)].append(oi)

    valid = (
        (x >= 0) & (x < ROI_SIZE) &
        (y >= 0) & (y < ROI_SIZE)
    )

    bx = np.floor_divide(x[valid].astype(np.int64), bin_px)
    by = np.floor_divide(y[valid].astype(np.int64), bin_px)

    valid_idx = np.flatnonzero(valid)
    keys = bx + by * (ROI_SIZE // bin_px + 1)

    order = np.argsort(keys)
    keys_sorted = keys[order]
    idx_sorted = valid_idx[order]

    out = np.full(len(x), -1, dtype=np.int32)

    if len(keys_sorted) == 0:
        return out

    split = np.r_[0, np.flatnonzero(np.diff(keys_sorted)) + 1, len(keys_sorted)]

    for a, b in zip(split[:-1], split[1:]):
        ids = idx_sorted[a:b]

        sample = ids[0]
        key = (
            int(x[sample] // bin_px),
            int(y[sample] // bin_px),
        )

        cand = poly_bins.get(key, [])
        if not cand:
            continue

        pts = np.c_[x[ids], y[ids]]
        unresolved = np.ones(len(ids), dtype=bool)

        for oi in cand:
            if not unresolved.any():
                break

            loc = np.flatnonzero(unresolved)
            inside = objects[oi]["path"].contains_points(
                pts[loc],
                radius=1e-9,
            )

            hit = loc[inside]

            if len(hit):
                out[ids[hit]] = objects[oi]["gt_idx"]
                unresolved[hit] = False

    return out


def build_tx_cache(tx_path, M, cache, chunksize):
    if cache.exists():
        print("reuse transcript cache:", cache, flush=True)
        return pd.read_parquet(cache)

    pieces = []
    scanned = 0
    kept = 0

    for d in pd.read_csv(tx_path, chunksize=chunksize):
        scanned += len(d)

        gx = pd.to_numeric(d["global_x"], errors="coerce").to_numpy(float)
        gy = pd.to_numeric(d["global_y"], errors="coerce").to_numpy(float)

        px, py = micron_to_px(gx, gy, M)

        inside = (
            np.isfinite(px) & np.isfinite(py) &
            (px >= ROI_X0) & (px < ROI_X1) &
            (py >= ROI_Y0) & (py < ROI_Y1)
        )

        if np.any(inside):
            q = d.loc[inside, ["gene"]].copy()
            q["x"] = px[inside] - ROI_X0
            q["y"] = py[inside] - ROI_Y0
            pieces.append(q)
            kept += len(q)

        print(
            f"transcripts scanned={scanned:,}; ROI kept={kept:,}",
            flush=True,
        )

    if not pieces:
        raise RuntimeError("zero transcripts in ROI")

    tx = pd.concat(pieces, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(cache, index=False)
    return tx


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--mask", type=Path, default=MASK_DEFAULT)
    ap.add_argument("--meta", type=Path, default=META_DEFAULT)
    ap.add_argument("--tx", type=Path, default=TX_DEFAULT)
    ap.add_argument("--boundaries", type=Path, default=BOUNDARY_DIR_DEFAULT)
    ap.add_argument("--transform", type=Path, default=TRANSFORM_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)

    ap.add_argument("--match-radius-um", type=float, default=10.0)
    ap.add_argument("--tx-chunksize", type=int, default=2_000_000)
    ap.add_argument("--poly-bin-px", type=int, default=128)

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for p in [args.mask, args.meta, args.tx, args.boundaries, args.transform]:
        if not p.exists():
            raise FileNotFoundError(p)

    M = load_transform(args.transform)
    px_per_um = math.sqrt(abs(float(M[0, 0]) * float(M[1, 1])))
    px_um = 1.0 / px_per_um

    print("=" * 96)
    print("GeneSegNet MERFISH1 FULL-METRIC EVALUATION")
    print("=" * 96)

    mask = tifffile.imread(args.mask)

    if mask.shape != (ROI_SIZE, ROI_SIZE):
        raise RuntimeError(
            f"prediction mask shape={mask.shape}; expected {(ROI_SIZE,ROI_SIZE)}"
        )

    pred_labs, area_px, pred_cent, area_um2, diam_um = mask_properties(
        mask, px_um
    )

    gt = load_gt_metadata(args.meta, M)

    gt_cent = gt[["roi_x_px", "roi_y_px"]].to_numpy(float)

    mp, mg, dist = sparse_hungarian(
        pred_cent * px_um,
        gt_cent * px_um,
        args.match_radius_um,
    )

    print(
        f"GT cells={len(gt):,}; pred={len(pred_labs):,}; matched={len(mp):,}",
        flush=True,
    )

    print("reading official GT cell polygons ...", flush=True)
    polys = read_roi_polygons(args.boundaries, gt)

    n_poly_cells = sum(bool(v) for v in polys.values())

    print(
        f"GT cells with zIndex_{ZINDEX} polygons="
        f"{n_poly_cells:,}/{len(gt):,}",
        flush=True,
    )

    offsets = calibrate_fov_offsets(polys, gt)
    objects = prepare_polygon_objects(polys, gt, offsets, M)

    print(
        f"calibrated FOVs={len(offsets):,}; polygon objects={len(objects):,}",
        flush=True,
    )

    tx = build_tx_cache(
        args.tx,
        M,
        args.out / "cache/roi_transcripts.parquet",
        args.tx_chunksize,
    )

    tx_x = tx["x"].to_numpy(float)
    tx_y = tx["y"].to_numpy(float)

    # Prediction assignment.
    xi = np.rint(tx_x).astype(np.int64)
    yi = np.rint(tx_y).astype(np.int64)

    in_mask = (
        (xi >= 0) & (xi < ROI_SIZE) &
        (yi >= 0) & (yi < ROI_SIZE)
    )

    pred_label = np.zeros(len(tx), dtype=np.int64)
    pred_label[in_mask] = mask[yi[in_mask], xi[in_mask]].astype(np.int64)

    pl2row = {int(v): i for i, v in enumerate(pred_labs)}
    pred_row = np.array(
        [pl2row.get(int(v), -1) for v in pred_label],
        dtype=np.int32,
    )

    print("assigning transcripts to official GT polygons ...", flush=True)
    gt_row = assign_points_to_gt(
        tx_x,
        tx_y,
        objects,
        args.poly_bin_px,
    )

    pred_assigned = pred_row >= 0
    gt_assigned = gt_row >= 0
    both = pred_assigned & gt_assigned

    match_map = np.full(len(pred_labs), -1, dtype=np.int32)
    match_map[mp] = mg

    safe_pred = np.clip(pred_row, 0, max(len(pred_labs) - 1, 0))
    correct = both & (match_map[safe_pred] == gt_row)

    assign_accuracy = (
        float(correct.sum() / both.sum())
        if both.sum() else np.nan
    )
    overall_correct = float(correct.sum() / len(tx))

    genes = pd.Index(sorted(tx["gene"].astype(str).unique()))
    gmap = {g: i for i, g in enumerate(genes)}
    gene_idx = tx["gene"].astype(str).map(gmap).to_numpy(np.int32)

    pred_mat = coo_matrix(
        (
            np.ones(int(pred_assigned.sum()), dtype=np.int32),
            (pred_row[pred_assigned], gene_idx[pred_assigned]),
        ),
        shape=(len(pred_labs), len(genes)),
    ).tocsr()

    gt_mat = coo_matrix(
        (
            np.ones(int(gt_assigned.sum()), dtype=np.int32),
            (gt_row[gt_assigned], gene_idx[gt_assigned]),
        ),
        shape=(len(gt), len(genes)),
    ).tocsr()

    pred_counts = np.asarray(pred_mat.sum(axis=1)).ravel()
    gt_counts = np.asarray(gt_mat.sum(axis=1)).ravel()

    matched_pred_counts = pred_counts[mp].astype(float)
    matched_gt_counts = gt_counts[mg].astype(float)

    vcos, vjs, vpear = [], [], []

    for pi, gi in zip(mp, mg):
        a = pred_mat.getrow(pi).toarray().ravel().astype(float)
        b = gt_mat.getrow(gi).toarray().ravel().astype(float)

        vcos.append(cosine(a, b))
        vjs.append(js_distance(a, b))
        vpear.append(safe_corr(a, b, "pearson"))

    precision = len(mp) / len(pred_labs)
    recall = len(mp) / len(gt)
    f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "method": "GeneSegNet",
        "dataset": "merfish_1",
        "roi": f"y[{ROI_Y0},{ROI_Y1}) x[{ROI_X0},{ROI_X1})",

        "gt_cells": int(len(gt)),
        "pred_cells": int(len(pred_labs)),
        "matched": int(len(mp)),
        "cell_ratio": float(len(pred_labs) / len(gt)),

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "loc_mean_um": float(np.mean(dist)),
        "loc_median_um": float(np.median(dist)),
        "loc_p95_um": float(np.percentile(dist, 95)),
        "match_radius_um": float(args.match_radius_um),

        "foreground_fraction": float((mask > 0).mean()),
        "median_area_px": float(np.median(area_px)),
        "median_area_um2": float(np.median(area_um2)),
        "median_diameter_um": float(np.median(diam_um)),

        "n_transcripts": int(len(tx)),
        "assign_rate": float(pred_assigned.mean()),
        "coverage_fraction": float(pred_assigned.mean()),
        "gt_assign_rate": float(gt_assigned.mean()),

        "tx_per_cell_median": float(np.median(pred_counts)),
        "tx_per_cell_mean": float(np.mean(pred_counts)),
        "tx_per_cell_p5": float(np.percentile(pred_counts, 5)),
        "tx_per_cell_p95": float(np.percentile(pred_counts, 95)),
        "frag_lt10": float(np.mean(pred_counts < 10)),

        "count_pearson": safe_corr(
            matched_pred_counts, matched_gt_counts, "pearson"
        ),
        "count_spearman": safe_corr(
            matched_pred_counts, matched_gt_counts, "spearman"
        ),
        "count_mae": float(
            np.mean(np.abs(matched_pred_counts - matched_gt_counts))
        ),
        "count_rmse": float(
            np.sqrt(
                np.mean(
                    (matched_pred_counts - matched_gt_counts) ** 2
                )
            )
        ),

        "assign_accuracy": assign_accuracy,
        "overall_correct": overall_correct,
        "assignment_n": int(both.sum()),

        "vec_cosine": float(np.nanmean(vcos)),
        "vec_js_dist": float(np.nanmean(vjs)),
        "vec_pearson": float(np.nanmean(vpear)),
        "vec_n": int(len(mp)),
        "vec_genes": int(len(genes)),

        "gt_polygon_cells": int(n_poly_cells),
        "gt_polygon_fraction": float(n_poly_cells / len(gt)),

        "uses_platform_prior": False,
        "detection_comparable": True,
        "assignment_independent": True,
        "vector_independent": True,
        "gt_used_only_for_evaluation": True,

        "mask": str(args.mask),
    }

    pairs = pd.DataFrame({
        "pred_idx": mp,
        "pred_label": pred_labs[mp],
        "gt_idx": mg,
        "gt_cell_id": gt.iloc[mg]["cell_id"].to_numpy(),
        "distance_um": dist,
        "pred_tx": matched_pred_counts,
        "gt_tx": matched_gt_counts,
        "vec_cosine": vcos,
        "vec_js_dist": vjs,
        "vec_pearson": vpear,
    })

    metrics_path = args.out / "genesegnet_merfish_1_metrics.csv"
    pairs_path = args.out / "genesegnet_merfish_1_pairs.csv"
    offsets_path = args.out / "genesegnet_merfish_1_boundary_offsets.tsv"
    meta_path = args.out / "genesegnet_merfish_1_eval_meta.json"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pairs.to_csv(pairs_path, index=False)

    pd.DataFrame([
        {
            "fov": fov,
            "offset_x_um": dx,
            "offset_y_um": dy,
        }
        for fov, (dx, dy) in sorted(offsets.items())
    ]).to_csv(offsets_path, sep="\t", index=False)

    meta_path.write_text(
        json.dumps({
            "roi_y0": ROI_Y0,
            "roi_y1": ROI_Y1,
            "roi_x0": ROI_X0,
            "roi_x1": ROI_X1,
            "zindex": ZINDEX,
            "pixel_size_um": px_um,
            "boundary_dir": str(args.boundaries),
            "transform": str(args.transform),
            "note": (
                "Official MERFISH boundaries are used only as GT during "
                "evaluation, never during GeneSegNet inference."
            ),
        }, indent=2, ensure_ascii=False)
    )

    print("=" * 96)
    for k, v in metrics.items():
        print(f"{k:32s} {v}")
    print("=" * 96)
    print("metrics ->", metrics_path)
    print("pairs   ->", pairs_path)
    print("offsets ->", offsets_path)
    print("meta    ->", meta_path)


if __name__ == "__main__":
    main()
