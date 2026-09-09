#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_value_stereo_3.py
Evaluate UCS on Stereo-seq stereo_3 (E16.5_E2S7).

Current finalized run:
    nucleus_erode = 2
    BIN_UM        = 0.5 um / bin
    GT cells      = 7,960
    final mask    = ucs_log/pred/segmentation_mask.tif

Important interpretation:
1. nuclei_mask is rasterized from the GEM CellBin `cell` column and GT is defined
   from the same column. Therefore the prior and GT are not independent.
2. precision / recall / F1 mainly test whether UCS preserved the supplied prior.
3. assignment / coverage / expression-vector metrics are circularly optimistic.
4. loc_* / median_diameter_um / count_* are retained for consistency with the
   existing benchmark, but count_* must also be interpreted cautiously because
   cell identity originates from the same CellBin prior.
5. stereo_2 and stereo_3 both use nucleus_erode=2 as a parameter. In stereo_3,
   however, the repaired erosion code keeps a nucleus unchanged when erosion
   would erase it; 7,950 / 7,960 nuclei took this fallback in the recorded run.
   Thus the nominal erosion parameter is unified, while effective erosion is
   weaker in stereo_3.

Usage:
    python3 ucs_value_stereo_3.py --match-radius 10
    python3 ucs_value_stereo_3.py --mask segmentation_mask.tif --match-radius 10
    python3 ucs_value_stereo_3.py --mask mask_shift_0.tif --match-radius 10
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = "/data/qiuyijia/ucs_stereoseq_E16.5_E2S7"
GEM = (
    "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
    "E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
)
OUT = "/data/qiuyijia/eval_results_qv20/stereo_3"

METHOD = "UCS"
DATASET = "stereo_3"
SLICE = "E16.5_E2S7"

BIN_UM = 0.5
GT_N = 7960
ERODE = 2

# Recorded behavior of the repaired erosion implementation for this run.
ERODE_SHRUNK = 10
ERODE_FALLBACK_KEEP = 7950


def log(msg=""):
    print(msg, flush=True)


def choose_col(cols, candidates, required=True):
    low = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise KeyError(
            f"cannot find any of {candidates}; columns={list(cols)[:30]}"
        )
    return None


def resolve_mask(root: str, mask_arg: str) -> str:
    p = Path(mask_arg)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)

    candidates = [
        Path(root) / "ucs_log" / "pred" / mask_arg,
        Path(root) / mask_arg,
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"cannot find mask={mask_arg}; checked:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def read_gem(path: str) -> pd.DataFrame:
    """Read CellBin GEM, skipping leading '#' metadata lines."""
    skip = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break

    d = pd.read_csv(path, sep="\t", skiprows=skip)
    if d.empty:
        raise RuntimeError(f"GEM is empty: {path}")

    # Preserve original names but standardize the columns used below.
    xc = choose_col(d.columns, ("x", "x_location"))
    yc = choose_col(d.columns, ("y", "y_location"))
    cc = choose_col(d.columns, ("cell", "cellid", "cell_id", "label"))
    gc = choose_col(d.columns, ("gene", "genename", "gene_name"), required=False)
    nc = choose_col(
        d.columns,
        ("count", "midcounts", "mid_count", "umi", "umi_count"),
        required=False,
    )

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(d[xc], errors="coerce"),
            "y": pd.to_numeric(d[yc], errors="coerce"),
            "cell": pd.to_numeric(d[cc], errors="coerce"),
        }
    )
    if gc is not None:
        out["gene"] = d[gc].astype(str)
    if nc is not None:
        out["count"] = pd.to_numeric(d[nc], errors="coerce").fillna(0)
    else:
        out["count"] = 1.0

    out = out.dropna(subset=["x", "y", "cell"])
    out = out[out["cell"] > 0].copy()
    out["cell"] = out["cell"].astype(np.int64)
    out["count"] = out["count"].astype(np.float64)

    if len(out) == 0:
        raise RuntimeError("GEM contains no cell>0 rows")
    return out


def mask_stats(mask: np.ndarray):
    lab, cnt = np.unique(mask, return_counts=True)
    pos = lab > 0
    ids = lab[pos].astype(np.int64)
    areas = cnt[pos].astype(np.int64)
    if len(ids) == 0:
        raise RuntimeError("mask has no positive labels")
    return ids, areas


def mask_centroids(mask: np.ndarray, ids: np.ndarray, x0: int, y0: int):
    from scipy import ndimage

    # center_of_mass returns (row=y, col=x)
    centers = ndimage.center_of_mass(mask > 0, mask, ids.tolist())
    centers = np.asarray(centers, dtype=np.float64)

    good = np.isfinite(centers).all(axis=1)
    ids = ids[good]
    centers = centers[good]

    xy_um = np.column_stack(
        [
            (centers[:, 1] + x0) * BIN_UM,
            (centers[:, 0] + y0) * BIN_UM,
        ]
    )
    return ids, xy_um


def gt_cell_table(gem: pd.DataFrame) -> pd.DataFrame:
    """
    GT centroid is count-weighted when MIDCounts/count exists.
    GT total count is also MIDCounts-weighted.
    """
    w = gem["count"].to_numpy(np.float64)
    xw = gem["x"].to_numpy(np.float64) * w
    yw = gem["y"].to_numpy(np.float64) * w

    tmp = pd.DataFrame(
        {
            "cell": gem["cell"].to_numpy(np.int64),
            "w": w,
            "xw": xw,
            "yw": yw,
        }
    )
    g = tmp.groupby("cell", sort=False).sum()
    g["x"] = g["xw"] / np.maximum(g["w"], 1e-12)
    g["y"] = g["yw"] / np.maximum(g["w"], 1e-12)
    g["n"] = g["w"]
    return g[["x", "y", "n"]]


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius_um: float):
    """
    Existing benchmark convention:
    each prediction queries its nearest GT; candidate pairs are processed from
    shortest distance to longest, and each GT can be used once.
    """
    from scipy.spatial import cKDTree

    dist, gi = cKDTree(gt_xy).query(
        pred_xy, k=1, distance_upper_bound=radius_um
    )
    ok = np.isfinite(dist) & (dist < radius_um)

    order = np.argsort(np.where(ok, dist, np.inf))
    used_gt = set()
    pairs = []

    for pi in order:
        if not ok[pi]:
            break
        g = int(gi[pi])
        if g in used_gt:
            continue
        used_gt.add(g)
        pairs.append((int(pi), g, float(dist[pi])))

    return pairs


def safe_corr(a, b, method="pearson"):
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    if method == "spearman":
        r = spearmanr(a, b)
        # SciPy compatibility:
        # older versions expose .correlation, newer ones may expose .statistic
        val = getattr(r, "statistic", getattr(r, "correlation", np.nan))
        return float(val)
    r = pearsonr(a, b)
    # older scipy.stats.pearsonr returns a tuple; newer versions expose .statistic
    val = getattr(r, "statistic", r[0] if hasattr(r, "__getitem__") else np.nan)
    return float(val)


def expression_metrics(
    gem: pd.DataFrame,
    pred_label_for_row: np.ndarray,
    pred_to_gt_cell: dict[int, int],
):
    """
    Compute assignment and vector metrics from GEM rows.

    These values are reported only for continuity with the benchmark.
    They are explicitly marked non-independent because the UCS prior and GT
    originate from the same CellBin `cell` column.
    """
    result = {
        "assign_rate": np.nan,
        "coverage_fraction": np.nan,
        "assign_accuracy": np.nan,
        "overall_correct": np.nan,
        "vec_cosine": np.nan,
        "vec_js_dist": np.nan,
        "vec_pearson": np.nan,
        "vec_n": 0,
        "vec_genes": 0,
    }

    n = len(gem)
    if n == 0:
        return result

    pred_lab = np.asarray(pred_label_for_row, dtype=np.int64)
    gt_cell = gem["cell"].to_numpy(np.int64)
    weights = gem["count"].to_numpy(np.float64)

    assigned = pred_lab > 0
    result["assign_rate"] = float(
        weights[assigned].sum() / max(weights.sum(), 1e-12)
    )
    result["coverage_fraction"] = result["assign_rate"]

    mapped_gt = np.full(n, -1, dtype=np.int64)
    if pred_to_gt_cell:
        # Number of predicted labels is only ~8k; dictionary lookup is cheap.
        for lab, cell in pred_to_gt_cell.items():
            mapped_gt[pred_lab == lab] = int(cell)

    comparable = mapped_gt > 0
    correct = comparable & (mapped_gt == gt_cell)

    denom_comp = weights[comparable].sum()
    denom_all = weights.sum()

    result["assign_accuracy"] = (
        float(weights[correct].sum() / denom_comp)
        if denom_comp > 0
        else np.nan
    )
    result["overall_correct"] = float(weights[correct].sum() / max(denom_all, 1e-12))

    if "gene" not in gem.columns or not pred_to_gt_cell:
        return result

    # Vector metrics only on matched predicted cells.
    cell_ids = sorted(set(pred_to_gt_cell.values()))
    if not cell_ids:
        return result

    cell_to_row = {c: i for i, c in enumerate(cell_ids)}
    genes, gene_code = np.unique(gem["gene"].to_numpy(str), return_inverse=True)
    n_gene = len(genes)

    from scipy.sparse import coo_matrix

    # Ground-truth expression matrix.
    gt_rows = np.fromiter(
        (cell_to_row.get(int(c), -1) for c in gt_cell),
        dtype=np.int64,
        count=n,
    )
    gt_ok = gt_rows >= 0
    GT = coo_matrix(
        (
            weights[gt_ok],
            (gt_rows[gt_ok], gene_code[gt_ok]),
        ),
        shape=(len(cell_ids), n_gene),
    ).tocsr()

    # Predicted expression matrix, but row identity is the GT cell matched to
    # that predicted mask instance.
    pr_rows = np.fromiter(
        (cell_to_row.get(int(c), -1) for c in mapped_gt),
        dtype=np.int64,
        count=n,
    )
    pr_ok = pr_rows >= 0
    PR = coo_matrix(
        (
            weights[pr_ok],
            (pr_rows[pr_ok], gene_code[pr_ok]),
        ),
        shape=(len(cell_ids), n_gene),
    ).tocsr()

    row_sums_gt = np.asarray(GT.sum(axis=1)).ravel()
    row_sums_pr = np.asarray(PR.sum(axis=1)).ravel()
    valid = (row_sums_gt > 0) & (row_sums_pr > 0)
    idx = np.where(valid)[0]
    result["vec_n"] = int(len(idx))
    result["vec_genes"] = int(n_gene)

    if len(idx) == 0:
        return result

    # Cosine similarity.
    dot = np.asarray(PR[idx].multiply(GT[idx]).sum(axis=1)).ravel()
    nr_pr = np.sqrt(np.asarray(PR[idx].multiply(PR[idx]).sum(axis=1)).ravel())
    nr_gt = np.sqrt(np.asarray(GT[idx].multiply(GT[idx]).sum(axis=1)).ravel())
    cos = dot / np.maximum(nr_pr * nr_gt, 1e-12)
    result["vec_cosine"] = float(np.mean(cos))

    # Jensen-Shannon distance, computed densely in manageable chunks.
    js_vals = []
    pear_vals = []
    max_pear = min(2000, len(idx))

    for start in range(0, len(idx), 128):
        ii = idx[start : start + 128]
        A = PR[ii].toarray().astype(np.float64)
        B = GT[ii].toarray().astype(np.float64)

        Ap = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        Bp = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-12)
        M = 0.5 * (Ap + Bp)

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(Ap > 0, Ap * np.log2(Ap / np.maximum(M, 1e-300)), 0)
            t2 = np.where(Bp > 0, Bp * np.log2(Bp / np.maximum(M, 1e-300)), 0)
        js = np.sqrt(np.maximum(0, 0.5 * (t1.sum(axis=1) + t2.sum(axis=1))))
        js_vals.extend(js.tolist())

    # Pearson per cell on at most 2000 matched cells, matching prior benchmark.
    for i in idx[:max_pear]:
        a = PR[i].toarray().ravel()
        b = GT[i].toarray().ravel()
        r = safe_corr(a, b, "pearson")
        if np.isfinite(r):
            pear_vals.append(r)

    result["vec_js_dist"] = float(np.mean(js_vals)) if js_vals else np.nan
    result["vec_pearson"] = float(np.mean(pear_vals)) if pear_vals else np.nan
    return result


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--gem", default=GEM)
    ap.add_argument("--mask", default="segmentation_mask.tif")
    ap.add_argument("--out", default=OUT)
    ap.add_argument(
        "--match-radius",
        type=float,
        default=10.0,
        help="one-to-one centroid matching radius in um",
    )
    a = ap.parse_args()

    log("=" * 78)
    log(f"{METHOD} @ {DATASET} ({SLICE})")
    log("=" * 78)
    log(f"nominal nucleus_erode = {ERODE}")
    log(
        f"stereo_3 erosion behavior: shrunk={ERODE_SHRUNK:,}, "
        f"fallback-kept={ERODE_FALLBACK_KEEP:,}"
    )
    log(
        "NOTE: stereo_2 and stereo_3 use the same nominal erosion radius (2), "
        "but stereo_3 effective erosion is weaker because erased nuclei are kept."
    )

    mask_path = resolve_mask(a.root, a.mask)
    if not os.path.exists(a.gem):
        sys.exit(f"GEM not found: {a.gem}")

    import tifffile

    log(f"\nmask: {mask_path}")
    mask = tifffile.imread(mask_path)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        sys.exit(f"mask must be 2-D after squeeze, got {mask.shape}")

    pred_ids, areas = mask_stats(mask)
    area_med = float(np.median(areas))
    diam_med = float(2 * np.sqrt(area_med / np.pi) * BIN_UM)

    log(
        f"mask shape {mask.shape}; labels {len(pred_ids):,}; "
        f"GT reference {GT_N:,}; ratio {len(pred_ids)/GT_N:.4f}"
    )
    log(
        f"foreground {(mask > 0).mean()*100:.3f}%  "
        f"area median {area_med:.1f} px = {diam_med:.2f} um"
    )

    if area_med <= 2:
        sys.exit(
            "INVALID: median mask area <=2 px, indicating the historical "
            "single-pixel erosion failure"
        )

    log(f"\nreading GEM: {a.gem}")
    gem = read_gem(a.gem)
    x0 = int(gem["x"].min())
    y0 = int(gem["y"].min())
    log(
        f"GEM rows(cell>0) {len(gem):,}; origin=({x0},{y0}); "
        f"GT unique cells={gem['cell'].nunique():,}"
    )
    if gem["cell"].nunique() != GT_N:
        log(
            f"WARNING: GEM unique cells {gem['cell'].nunique():,} != "
            f"configured GT_N {GT_N:,}; metrics use the GEM itself."
        )

    gt = gt_cell_table(gem)
    gt_ids = gt.index.to_numpy(np.int64)
    gt_xy = gt[["x", "y"]].to_numpy(np.float64) * BIN_UM

    pred_ids, pred_xy = mask_centroids(mask, pred_ids, x0, y0)

    pairs = greedy_match(pred_xy, gt_xy, a.match_radius)
    matched = len(pairs)

    precision = matched / max(len(pred_xy), 1)
    recall = matched / max(len(gt_xy), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    dist = np.asarray([q[2] for q in pairs], dtype=float)
    if len(dist):
        loc_mean = float(np.mean(dist))
        loc_median = float(np.median(dist))
        loc_p95 = float(np.percentile(dist, 95))
    else:
        loc_mean = loc_median = loc_p95 = np.nan

    log("\n[detection / localization]")
    log(
        f"pred {len(pred_xy):,}  GT {len(gt_xy):,}  matched {matched:,}  "
        f"radius {a.match_radius:.2f} um"
    )
    log(
        f"precision {precision:.4f}  recall {recall:.4f}  F1 {f1:.4f}"
    )
    log(
        f"loc mean/median/p95 = "
        f"{loc_mean:.3f} / {loc_median:.3f} / {loc_p95:.3f} um"
    )

    # Pair maps.
    pred_to_gt_cell = {}
    pair_rows = []
    for pi, gi, d in pairs:
        pred_label = int(pred_ids[pi])
        gt_cell = int(gt_ids[gi])
        pred_to_gt_cell[pred_label] = gt_cell
        pair_rows.append(
            {
                "pred_index": pi,
                "pred_label": pred_label,
                "gt_index": gi,
                "gt_cell": gt_cell,
                "dist_um": d,
            }
        )

    # Count metrics on matched cell pairs.
    pred_count_by_label = {}
    # Every GEM row is mapped into the output mask below; use MIDCounts weights.
    yy = gem["y"].to_numpy(np.int64) - y0
    xx = gem["x"].to_numpy(np.int64) - x0
    inside = (
        (yy >= 0)
        & (yy < mask.shape[0])
        & (xx >= 0)
        & (xx < mask.shape[1])
    )
    pred_for_row = np.zeros(len(gem), dtype=np.int64)
    pred_for_row[inside] = mask[yy[inside], xx[inside]].astype(np.int64)

    tmp = pd.DataFrame(
        {
            "pred": pred_for_row,
            "count": gem["count"].to_numpy(np.float64),
        }
    )
    pc = tmp[tmp["pred"] > 0].groupby("pred")["count"].sum()
    pred_count_by_label = pc.to_dict()

    gt_count_by_cell = gt["n"].to_dict()

    pair_pred_counts = np.asarray(
        [pred_count_by_label.get(int(r["pred_label"]), 0.0) for r in pair_rows]
    )
    pair_gt_counts = np.asarray(
        [gt_count_by_cell.get(int(r["gt_cell"]), 0.0) for r in pair_rows]
    )

    count_pearson = safe_corr(pair_pred_counts, pair_gt_counts, "pearson")
    count_spearman = safe_corr(pair_pred_counts, pair_gt_counts, "spearman")
    count_mae = (
        float(np.mean(np.abs(pair_pred_counts - pair_gt_counts)))
        if matched
        else np.nan
    )
    count_rmse = (
        float(np.sqrt(np.mean((pair_pred_counts - pair_gt_counts) ** 2)))
        if matched
        else np.nan
    )

    log("\n[counts on matched cells]")
    log(
        f"pearson {count_pearson:.4f}  spearman {count_spearman:.4f}  "
        f"MAE {count_mae:.3f}  RMSE {count_rmse:.3f}"
    )

    extra = expression_metrics(gem, pred_for_row, pred_to_gt_cell)

    log("\n[assignment / vectors -- NON-INDEPENDENT, prior and GT share CellBin cell]")
    log(
        f"assign_rate {extra['assign_rate']:.4f}  "
        f"coverage {extra['coverage_fraction']:.4f}  "
        f"assign_accuracy {extra['assign_accuracy']:.4f}  "
        f"overall_correct {extra['overall_correct']:.4f}"
    )
    if extra["vec_n"]:
        log(
            f"vec_cosine {extra['vec_cosine']:.4f}  "
            f"vec_js_dist {extra['vec_js_dist']:.4f}  "
            f"vec_pearson {extra['vec_pearson']:.4f}  "
            f"(n={extra['vec_n']:,}, genes={extra['vec_genes']:,})"
        )

    result = {
        "method": METHOD,
        "dataset": DATASET,
        "slice": SLICE,
        "gt_cells": int(len(gt_xy)),
        "pred_cells": int(len(pred_xy)),
        "matched": int(matched),
        "cell_ratio": float(len(pred_xy) / max(len(gt_xy), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "loc_mean_um": loc_mean,
        "loc_median_um": loc_median,
        "loc_p95_um": loc_p95,
        "median_area_px": area_med,
        "median_diameter_um": diam_med,
        "count_pearson": count_pearson,
        "count_spearman": count_spearman,
        "count_mae": count_mae,
        "count_rmse": count_rmse,
        **extra,
        "match_radius_um": float(a.match_radius),
        "nucleus_erode": ERODE,
        "erosion_shrunk": ERODE_SHRUNK,
        "erosion_fallback_keep": ERODE_FALLBACK_KEEP,
        "uses_platform_prior": True,
        "detection_comparable": True,
        "assignment_independent": False,
        "vector_independent": False,
        "mask": os.path.basename(mask_path),
        "note": (
            "nuclei_mask and GT both originate from the CellBin GEM cell column; "
            "precision/recall/F1 mainly test preservation of the supplied prior, "
            "and assignment/coverage/vector metrics are circularly optimistic. "
            "stereo_2 and stereo_3 both use nominal nucleus_erode=2. In stereo_3, "
            "7,950/7,960 nuclei would disappear after erosion and are therefore "
            "kept at their original shape by the repaired fallback, so effective "
            "erosion is weaker than stereo_2."
        ),
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics_csv = out / f"ucs_{DATASET}_metrics.csv"
    pairs_csv = out / f"ucs_{DATASET}_pairs.csv"
    meta_json = out / f"ucs_{DATASET}_eval_meta.json"

    pd.DataFrame([result]).to_csv(metrics_csv, index=False)
    pd.DataFrame(pair_rows).to_csv(pairs_csv, index=False)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    log("\n" + "=" * 78)
    for k, v in result.items():
        if k == "note":
            continue
        log(f"{k:<26} {v}")
    log("=" * 78)
    log(f"metrics -> {metrics_csv}")
    log(f"pairs   -> {pairs_csv}")
    log(f"meta    -> {meta_json}")
    log(
        "\nINTERPRETATION: nominal erosion radius is 2. "
        "Do not treat assignment/coverage/vector metrics as independent UCS quality."
    )


if __name__ == "__main__":
    main()
