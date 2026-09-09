#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_value_cosmx_2.py

Formal evaluation of GeneSegNet on CosMx2 lymph-node held-out FOV48/FOV261.

Inputs
------
Prediction masks:
  /data/qiuyijia/genesegnet_cosmx_2_infer/fovs/fov48/genesegnet_fov48_mask.tif
  /data/qiuyijia/genesegnet_cosmx_2_infer/fovs/fov261/genesegnet_fov261_mask.tif

GT metadata:
  /data/qiuyijia/dataset/cosmx_lymph_node/flat_files/S0_metadata_file.csv.gz

Transcript cache:
  /data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/tx_fov_48_261.parquet

Coordinate convention
---------------------
Masks and transcript local coordinates are 4256 x 4256 native CosMx pixels,
0.12 um / pixel.

Metrics
-------
- GT / prediction counts and cell ratio
- one-to-one centroid matching, precision / recall / F1
- localization mean / median / p95
- predicted area / diameter
- transcript assignment rate and tx/cell distribution
- matched-cell transcript-count Pearson / Spearman / MAE / RMSE
- transcript assignment accuracy against CosMx cell_ID
- matched expression-vector cosine / JS distance / Pearson

No platform segmentation prior is used by GeneSegNet inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import coo_matrix


PX_UM = 0.12
FOV_PX = 4256

META_DEFAULT = (
    "/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/"
    "S0_metadata_file.csv.gz"
)

TX_DEFAULT = (
    "/data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/"
    "tx_fov_48_261.parquet"
)

PRED_ROOT_DEFAULT = "/data/qiuyijia/genesegnet_cosmx_2_infer/fovs"
OUT_DEFAULT = "/data/qiuyijia/eval_results_qv20/cosmx_2"


def safe_corr(a, b, kind="pearson"):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r = pearsonr(a, b) if kind == "pearson" else spearmanr(a, b)
    return float(r[0])


def choose(cols, names, required=True):
    lut = {str(c).lower(): c for c in cols}
    for n in names:
        if n.lower() in lut:
            return lut[n.lower()]
    if required:
        raise KeyError(f"cannot find any of {names}; columns={list(cols)}")
    return None


def mask_properties(mask):
    lab, cnt = np.unique(mask, return_counts=True)
    lab = lab[lab > 0]
    cnt = cnt[np.unique(mask) > 0] if False else None

    # More direct and safe:
    labs, counts = np.unique(mask[mask > 0], return_counts=True)
    if len(labs) == 0:
        return (
            np.empty(0, np.int64),
            np.empty(0, float),
            np.empty((0, 2), float),
        )

    yy, xx = np.nonzero(mask > 0)
    vals = mask[yy, xx].astype(np.int64)

    # labels may not be exactly 1..N, make explicit map.
    lut = {int(v): i for i, v in enumerate(labs)}
    idx = np.fromiter((lut[int(v)] for v in vals),
                      dtype=np.int64, count=len(vals))

    n = len(labs)
    npix = np.bincount(idx, minlength=n).astype(float)
    sx = np.bincount(idx, weights=xx, minlength=n)
    sy = np.bincount(idx, weights=yy, minlength=n)
    cx = sx / np.maximum(npix, 1)
    cy = sy / np.maximum(npix, 1)

    return labs.astype(np.int64), npix, np.c_[cx, cy]


def sparse_hungarian(pred, gt, radius):
    if len(pred) == 0 or len(gt) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, float)

    tp = cKDTree(pred)
    tg = cKDTree(gt)
    neigh = tp.query_ball_tree(tg, r=radius)

    p2g = {i: js for i, js in enumerate(neigh) if js}
    g2p = {}
    for i, js in p2g.items():
        for j in js:
            g2p.setdefault(j, []).append(i)

    seen = set()
    mp, mg, md = [], [], []

    for start in p2g:
        if start in seen:
            continue

        ps = {start}
        gs = set()
        stack = [start]

        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            for j in p2g.get(i, []):
                if j not in gs:
                    gs.add(j)
                    for ii in g2p.get(j, []):
                        if ii not in ps:
                            ps.add(ii)
                            stack.append(ii)

        pl = sorted(ps)
        gl = sorted(gs)
        gj = {g: k for k, g in enumerate(gl)}

        C = np.full((len(pl), len(gl)), radius * 1000.0, dtype=float)
        for r, i in enumerate(pl):
            for j in p2g.get(i, []):
                C[r, gj[j]] = np.linalg.norm(pred[i] - gt[j])

        rr, cc = linear_sum_assignment(C)
        for r, c in zip(rr, cc):
            if C[r, c] <= radius:
                mp.append(pl[r])
                mg.append(gl[c])
                md.append(C[r, c])

    return np.asarray(mp, int), np.asarray(mg, int), np.asarray(md, float)


def load_metadata(meta_path, fov):
    m = pd.read_csv(meta_path)
    fc = choose(m.columns, ("fov", "FOV"))
    q = m[pd.to_numeric(m[fc], errors="coerce") == fov].copy().reset_index(drop=True)

    xc = choose(q.columns, ("CenterX_local_px", "x_local_px"))
    yc = choose(q.columns, ("CenterY_local_px", "y_local_px"))
    idc = choose(q.columns, ("cell_ID", "cell_id", "CellID"), required=False)
    area = choose(q.columns, ("Area", "area", "NucArea"), required=False)
    count = choose(q.columns, ("nCount_RNA", "totalcounts", "Total_counts", "nCount"),
                   required=False)

    out = pd.DataFrame({
        "x_px": pd.to_numeric(q[xc], errors="coerce"),
        "y_px": pd.to_numeric(q[yc], errors="coerce"),
    })

    if idc:
        out["cell_id"] = q[idc].astype(str)
    else:
        out["cell_id"] = q.index.astype(str)

    if area:
        out["area_px"] = pd.to_numeric(q[area], errors="coerce")
    if count:
        out["gt_count_meta"] = pd.to_numeric(q[count], errors="coerce")

    return out, q


def assign_transcripts(mask, tx, pred_labs):
    xc = choose(tx.columns, ("x_local_px", "x"))
    yc = choose(tx.columns, ("y_local_px", "y"))
    gc = choose(tx.columns, ("target", "gene", "feature_name"), required=False)
    cc = choose(tx.columns, ("cell_ID", "cell_id", "cell"), required=False)

    x = np.rint(pd.to_numeric(tx[xc], errors="coerce").to_numpy()).astype(np.int64)
    y = np.rint(pd.to_numeric(tx[yc], errors="coerce").to_numpy()).astype(np.int64)

    inside = (
        (x >= 0) & (x < mask.shape[1]) &
        (y >= 0) & (y < mask.shape[0])
    )

    pred_label = np.zeros(len(tx), dtype=np.int64)
    pred_label[inside] = mask[y[inside], x[inside]].astype(np.int64)

    lab2row = {int(v): i for i, v in enumerate(pred_labs)}
    pred_row = np.array([lab2row.get(int(v), -1) for v in pred_label],
                        dtype=np.int64)

    return pred_row, gc, cc, inside


def js_dist_rows(A, B):
    sa = A.sum(1)
    sb = B.sum(1)
    keep = (sa > 0) & (sb > 0)
    if not np.any(keep):
        return np.nan, keep

    A = A[keep] / sa[keep, None]
    B = B[keep] / sb[keep, None]
    M = (A + B) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        kl1 = np.nansum(np.where(A > 0, A * np.log2(A / M), 0), axis=1)
        kl2 = np.nansum(np.where(B > 0, B * np.log2(B / M), 0), axis=1)

    return float(np.mean(np.sqrt((kl1 + kl2) / 2))), keep


def eval_one(fov, pred_root, meta_path, tx_path, match_radius, vec_limit):
    mask_path = Path(pred_root) / f"fov{fov}" / f"genesegnet_fov{fov}_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    mask = tifffile.imread(mask_path)
    if mask.shape != (FOV_PX, FOV_PX):
        raise RuntimeError(f"FOV{fov}: mask shape={mask.shape}, expected {(FOV_PX,FOV_PX)}")

    labs, area_px, loc_px = mask_properties(mask)
    pred_xy_um = loc_px * PX_UM

    meta, meta_raw = load_metadata(meta_path, fov)
    gt_xy_um = meta[["x_px", "y_px"]].to_numpy(float) * PX_UM

    mp, mg, dist = sparse_hungarian(pred_xy_um, gt_xy_um, match_radius)

    pred_n = len(labs)
    gt_n = len(meta)
    matched = len(mp)

    precision = matched / pred_n if pred_n else np.nan
    recall = matched / gt_n if gt_n else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan

    area_um2 = area_px * PX_UM * PX_UM
    diam_um = 2 * np.sqrt(area_um2 / np.pi)

    result = {
        "method": "GeneSegNet",
        "dataset": "cosmx_2",
        "fov": fov,
        "gt_cells": gt_n,
        "pred_cells": pred_n,
        "matched": matched,
        "cell_ratio": pred_n / gt_n if gt_n else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loc_mean_um": float(np.mean(dist)) if len(dist) else np.nan,
        "loc_median_um": float(np.median(dist)) if len(dist) else np.nan,
        "loc_p95_um": float(np.percentile(dist, 95)) if len(dist) else np.nan,
        "match_radius_um": match_radius,
        "foreground_fraction": float((mask > 0).mean()),
        "median_area_px": float(np.median(area_px)) if len(area_px) else np.nan,
        "median_area_um2": float(np.median(area_um2)) if len(area_um2) else np.nan,
        "median_diameter_um": float(np.median(diam_um)) if len(diam_um) else np.nan,
        "uses_platform_prior": False,
        "detection_comparable": True,
        "assignment_independent": True,
        "vector_independent": True,
        "mask": str(mask_path),
    }

    # GT area reference.
    if "area_px" in meta:
        result["gt_median_area_px"] = float(np.nanmedian(meta["area_px"]))
        result["gt_median_diameter_um"] = float(
            2 * np.sqrt(np.nanmedian(meta["area_px"]) / np.pi) * PX_UM
        )

    # Transcript based metrics.
    tx_all = pd.read_parquet(tx_path)
    fc = choose(tx_all.columns, ("fov", "FOV"))
    tx = tx_all[pd.to_numeric(tx_all[fc], errors="coerce") == fov].copy().reset_index(drop=True)

    pred_row, gc, cc, inside = assign_transcripts(mask, tx, labs)
    assigned = pred_row >= 0

    pred_counts = np.bincount(
        pred_row[assigned],
        minlength=pred_n
    ).astype(float)

    result.update({
        "n_transcripts": int(len(tx)),
        "assign_rate": float(assigned.mean()),
        "coverage_fraction": float(assigned.mean()),
        "tx_per_cell_median": float(np.median(pred_counts)) if pred_n else np.nan,
        "tx_per_cell_mean": float(np.mean(pred_counts)) if pred_n else np.nan,
        "tx_per_cell_p5": float(np.percentile(pred_counts, 5)) if pred_n else np.nan,
        "tx_per_cell_p95": float(np.percentile(pred_counts, 95)) if pred_n else np.nan,
        "frag_lt10": float(np.mean(pred_counts < 10)) if pred_n else np.nan,
    })

    # GT transcript counts from tx cell ID whenever possible, otherwise metadata count.
    gt_counts = None
    tx_gt_row = None

    if cc:
        id2row = {str(v): i for i, v in enumerate(meta["cell_id"].astype(str))}
        tx_gt_row = tx[cc].astype(str).map(id2row).fillna(-1).astype(np.int64).to_numpy()
        valid_gt = tx_gt_row >= 0
        gt_counts = np.bincount(
            tx_gt_row[valid_gt],
            minlength=gt_n
        ).astype(float)
        result["gt_tx_id_match_rate"] = float(valid_gt.mean())
    elif "gt_count_meta" in meta:
        gt_counts = meta["gt_count_meta"].to_numpy(float)

    if gt_counts is not None and matched:
        A = pred_counts[mp]
        B = gt_counts[mg]
        result.update({
            "count_pearson": safe_corr(A, B, "pearson"),
            "count_spearman": safe_corr(A, B, "spearman"),
            "count_mae": float(np.nanmean(np.abs(A-B))),
            "count_rmse": float(np.sqrt(np.nanmean((A-B)**2))),
        })

    # Assignment accuracy: among transcripts assigned to a predicted cell and
    # having a platform GT cell ID, ask whether that GT cell is the centroid-matched one.
    if tx_gt_row is not None and matched:
        p2g = {int(p): int(g) for p, g in zip(mp, mg)}
        valid = assigned & (tx_gt_row >= 0)
        if np.any(valid):
            want = np.array([p2g.get(int(p), -2) for p in pred_row[valid]], dtype=np.int64)
            truth = tx_gt_row[valid]
            acc = float(np.mean(want == truth))
            result["assign_accuracy"] = acc
            result["overall_correct"] = acc * float(assigned.mean())
            result["assignment_n"] = int(valid.sum())

    # Expression vectors.
    if gc and tx_gt_row is not None and matched:
        genes = pd.Index(sorted(tx[gc].astype(str).unique()))
        gmap = {g: i for i, g in enumerate(genes)}
        gi = tx[gc].astype(str).map(gmap).to_numpy(np.int64)

        vp = pred_row >= 0
        PV = coo_matrix(
            (np.ones(int(vp.sum()), dtype=np.float32),
             (pred_row[vp], gi[vp])),
            shape=(pred_n, len(genes))
        ).tocsr()

        vg = tx_gt_row >= 0
        GV = coo_matrix(
            (np.ones(int(vg.sum()), dtype=np.float32),
             (tx_gt_row[vg], gi[vg])),
            shape=(gt_n, len(genes))
        ).tocsr()

        take = np.arange(min(matched, vec_limit))
        A = np.asarray(PV[mp[take]].todense(), float)
        B = np.asarray(GV[mg[take]].todense(), float)

        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(B, axis=1)
        k = (na > 0) & (nb > 0)

        if np.any(k):
            result["vec_cosine"] = float(np.mean(
                (A[k] * B[k]).sum(1) / (na[k] * nb[k])
            ))

            js, _ = js_dist_rows(A[k], B[k])
            result["vec_js_dist"] = js

            vals = []
            for aa, bb in zip(A[k], B[k]):
                r = safe_corr(aa, bb, "pearson")
                if np.isfinite(r):
                    vals.append(r)
            result["vec_pearson"] = float(np.mean(vals)) if vals else np.nan
            result["vec_n"] = int(k.sum())
            result["vec_genes"] = int(len(genes))

    pairs = pd.DataFrame({
        "pred_idx": mp,
        "gt_idx": mg,
        "distance_um": dist,
        "pred_label": labs[mp] if len(mp) else np.array([], dtype=int),
        "gt_cell_id": meta.iloc[mg]["cell_id"].to_numpy() if len(mg) else np.array([], dtype=str),
        "pred_tx": pred_counts[mp] if len(mp) else np.array([], dtype=float),
        "gt_tx": gt_counts[mg] if gt_counts is not None and len(mg) else np.array([], dtype=float),
    })

    return result, pairs


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--pred-root", default=PRED_ROOT_DEFAULT)
    ap.add_argument("--meta", default=META_DEFAULT)
    ap.add_argument("--tx", default=TX_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--fovs", default="48,261")
    ap.add_argument("--match-radius", type=float, default=10.0)
    ap.add_argument("--vec-limit", type=int, default=3000)
    args = ap.parse_args()

    fovs = [int(x) for x in args.fovs.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    all_pairs = []

    for fov in fovs:
        print("=" * 92)
        print(f"GeneSegNet @ CosMx2 FOV {fov}")
        print("=" * 92)

        r, p = eval_one(
            fov=fov,
            pred_root=args.pred_root,
            meta_path=args.meta,
            tx_path=args.tx,
            match_radius=args.match_radius,
            vec_limit=args.vec_limit,
        )

        rows.append(r)
        p["fov"] = fov
        all_pairs.append(p)

        for k, v in r.items():
            print(f"{k:30s} {v}")

        pd.DataFrame([r]).to_csv(
            out / f"genesegnet_cosmx_2_fov{fov}_metrics.csv",
            index=False
        )
        p.to_csv(
            out / f"genesegnet_cosmx_2_fov{fov}_pairs.csv",
            index=False
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "genesegnet_cosmx_2_metrics_by_fov.csv", index=False)

    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(
            out / "genesegnet_cosmx_2_pairs_all.csv",
            index=False
        )

    # Combined summary: additive count/detection metrics, weighted/macro for the rest.
    summary = {
        "method": "GeneSegNet",
        "dataset": "cosmx_2",
        "fovs": ",".join(map(str, fovs)),
        "gt_cells": int(df.gt_cells.sum()),
        "pred_cells": int(df.pred_cells.sum()),
        "matched": int(df.matched.sum()),
    }

    summary["cell_ratio"] = summary["pred_cells"] / summary["gt_cells"]
    summary["precision"] = summary["matched"] / summary["pred_cells"]
    summary["recall"] = summary["matched"] / summary["gt_cells"]
    summary["f1"] = (
        2 * summary["precision"] * summary["recall"] /
        (summary["precision"] + summary["recall"])
    )

    for col in (
        "loc_mean_um", "loc_median_um", "loc_p95_um",
        "foreground_fraction", "median_diameter_um",
        "assign_rate", "tx_per_cell_median", "tx_per_cell_mean",
        "frag_lt10", "count_pearson", "count_spearman",
        "assign_accuracy", "overall_correct",
        "vec_cosine", "vec_js_dist", "vec_pearson",
    ):
        if col in df:
            summary[col] = float(np.nanmean(pd.to_numeric(df[col], errors="coerce")))

    summary.update({
        "uses_platform_prior": False,
        "detection_comparable": True,
        "assignment_independent": True,
        "vector_independent": True,
    })

    pd.DataFrame([summary]).to_csv(
        out / "genesegnet_cosmx_2_metrics.csv",
        index=False
    )

    (out / "genesegnet_cosmx_2_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n" + "=" * 92)
    print("COMBINED FOV48 + FOV261")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k:30s} {v}")
    print("=" * 92)
    print("metrics ->", out / "genesegnet_cosmx_2_metrics.csv")


if __name__ == "__main__":
    main()
