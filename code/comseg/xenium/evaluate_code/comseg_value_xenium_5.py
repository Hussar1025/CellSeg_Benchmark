#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_value_xenium_5.py
Formal evaluation for ComSeg @ xenium_5 (Xenium mouse brain, centered 10000x10000 ROI).

Why this evaluator is different from naive h5ad evaluation
-----------------------------------------------------------
Do NOT use comseg_cells.h5ad obs centroid directly for detection/localization.
For ComSeg those centroids can inherit nucleus-landmark coordinates. Instead,
this evaluator recomputes every predicted cell centroid directly from the
deduplicated assigned transcript coordinates in spot_assignment*.parquet.

Evaluation convention
---------------------
1. Merge all tile spot_assignment.parquet files.
2. Remove overlap duplicates by (global_x, global_y, gene).
3. Recompute predicted centroids from assigned transcript coordinates.
4. Load platform cells.parquet and restrict GT to exactly the prep ROI.
5. KD-tree nearest-neighbour + radius threshold + one-to-one greedy matching.
   There is NO cell-ID shortcut for detection.
6. Count metrics use matched predicted/GT cell pairs.
7. Expression-vector metrics compare predicted gene vectors with platform
   transcript vectors for the spatially matched GT cell.
8. Assignment accuracy is calculated only when transcript-coordinate recovery
   succeeds at high enough rate. It is never silently fabricated.
9. Median diameter is read from a merged label raster if one can be identified;
   otherwise it is reported NaN rather than guessed.

Expected finished ComSeg run
----------------------------
root:
  yourpath/comseg_xenium_mouse_brain_roi10000

known run summary:
  merged cells = 12,045
  platform GT  = 11,961
  ratio        = 1.007

Usage
-----
python3 comseg_value_xenium_5.py
python3 comseg_value_xenium_5.py --skip-vectors
python3 comseg_value_xenium_5.py --skip-assignment
python3 comseg_value_xenium_5.py --match-radius 20

Outputs
-------
yourpath/eval_results_qv20/mouse_brain/
    comseg_xenium_5_metrics.csv
    comseg_xenium_5_pairs.csv
    comseg_xenium_5_pred_cells.csv
    comseg_xenium_5_gt_cells.csv
    comseg_xenium_5_eval_meta.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = "yourpath/comseg_xenium_mouse_brain_roi10000"
GT_ROOT = "yourpath/dataset/xenium_mouse_brain"
OUT = "yourpath/eval_results_qv20/mouse_brain"

METHOD = "ComSeg"
DATASET = "xenium_5"
TISSUE = "mouse_brain"

CONTROL_PREFIXES = (
    "NegControl",
    "Blank",
    "Unassigned",
    "DeprecatedCodeword",
    "Intergenic",
    "SystemControl",
)


def log(msg=""):
    print(msg, flush=True)


def pick(root: Path, rels):
    for rel in rels:
        p = root / rel
        if p.exists():
            return p
    return None


def choose_col(cols, candidates, required=True):
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(
            f"Cannot find any of {candidates}; columns={list(cols)[:40]}"
        )
    return None


def safe_corr(a, b, kind="pearson"):
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    if kind == "spearman":
        r = spearmanr(a, b)
        return float(getattr(r, "statistic", getattr(r, "correlation", r[0])))

    r = pearsonr(a, b)
    return float(getattr(r, "statistic", r[0]))


def load_meta(root):
    p = Path(root) / "input" / "prep_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"missing prep meta: {p}")
    with open(p) as f:
        m = json.load(f)

    required = ("x0", "y0", "size_x", "size_y", "pixel_size")
    miss = [k for k in required if k not in m]
    if miss:
        raise RuntimeError(f"prep_meta missing keys: {miss}")
    return m


def load_prediction(root, dedup=True, save_merged=True):
    root = Path(root)
    merged = root / "spot_assignment_merged.parquet"

    if merged.exists() and dedup:
        log(f"  reuse merged prediction: {merged}")
        d = pd.read_parquet(merged)
        return d

    fs = sorted(
        glob.glob(str(root / "comseg_out" / "tile_*" / "spot_assignment.parquet"))
    )
    if not fs:
        raise FileNotFoundError(
            f"no spot_assignment.parquet under {root}/comseg_out/tile_*"
        )

    log(f"  tile assignments: {len(fs)}")
    chunks = []
    for i, f in enumerate(fs, 1):
        q = pd.read_parquet(f)
        chunks.append(q)
        if i == 1:
            log(f"  assignment columns: {list(q.columns)}")
        if i % 5 == 0 or i == len(fs):
            log(f"    read {i}/{len(fs)} tiles")

    d = pd.concat(chunks, ignore_index=True)
    log(f"  concatenated rows: {len(d):,}")

    required = {"global_x", "global_y", "gene", "cell"}
    miss = required.difference(d.columns)
    if miss:
        raise RuntimeError(f"spot_assignment missing columns {sorted(miss)}")

    if dedup:
        n0 = len(d)
        d = d.drop_duplicates(
            subset=["global_x", "global_y", "gene"],
            keep="first",
        ).reset_index(drop=True)
        log(
            f"  overlap dedup: {n0:,} -> {len(d):,} "
            f"(removed {(n0-len(d))/max(n0,1)*100:.2f}%)"
        )

        if save_merged:
            d.to_parquet(merged, index=False)
            log(f"  -> {merged}")

    return d


def build_pred_cells(d, meta):
    px = float(meta["pixel_size"])
    x0 = float(meta["x0"])
    y0 = float(meta["y0"])

    cell = pd.to_numeric(d["cell"], errors="coerce")
    use = cell > 0

    q = d.loc[use, ["cell", "global_x", "global_y", "gene"]].copy()
    q["cell"] = pd.to_numeric(q["cell"], errors="coerce").astype(np.int64)

    # global_x/y in this engine are ROI-local native-pixel coordinates.
    q["x_um"] = (pd.to_numeric(q["global_x"], errors="coerce") + x0) * px
    q["y_um"] = (pd.to_numeric(q["global_y"], errors="coerce") + y0) * px

    g = (
        q.groupby("cell", sort=False)
        .agg(
            n_tx=("gene", "size"),
            x_um=("x_um", "mean"),
            y_um=("y_um", "mean"),
        )
        .reset_index()
    )

    return q, g


def load_gt_cells(gt_root, meta):
    root = Path(gt_root)
    p = pick(root, ["outs/cells.parquet", "cells.parquet"])
    if p is None:
        raise FileNotFoundError(f"cannot find cells.parquet under {root}")

    d = pd.read_parquet(p)

    xc = choose_col(d.columns, ("x_centroid", "x_location", "x"))
    yc = choose_col(d.columns, ("y_centroid", "y_location", "y"))
    ic = choose_col(d.columns, ("cell_id", "cellid", "id"))
    tc = choose_col(
        d.columns,
        (
            "transcript_counts",
            "transcript_count",
            "total_counts",
            "count",
        ),
        required=False,
    )

    px = float(meta["pixel_size"])
    x0_um = float(meta["x0"]) * px
    y0_um = float(meta["y0"]) * px
    x1_um = (float(meta["x0"]) + float(meta["size_x"])) * px
    y1_um = (float(meta["y0"]) + float(meta["size_y"])) * px

    x = pd.to_numeric(d[xc], errors="coerce")
    y = pd.to_numeric(d[yc], errors="coerce")

    hit = (
        x.ge(x0_um)
        & x.lt(x1_um)
        & y.ge(y0_um)
        & y.lt(y1_um)
    )

    out = pd.DataFrame(
        {
            "gt_cell": d.loc[hit, ic].astype(str).to_numpy(),
            "x_um": x.loc[hit].to_numpy(float),
            "y_um": y.loc[hit].to_numpy(float),
        }
    )

    if tc is not None:
        out["n_tx_platform"] = pd.to_numeric(
            d.loc[hit, tc], errors="coerce"
        ).to_numpy(float)
    else:
        out["n_tx_platform"] = np.nan

    return out.reset_index(drop=True), p


def greedy_match(pred_xy, gt_xy, radius_um):
    from scipy.spatial import cKDTree

    dist, gi = cKDTree(gt_xy).query(
        pred_xy,
        k=1,
        distance_upper_bound=radius_um,
    )
    ok = np.isfinite(dist) & (gi < len(gt_xy)) & (dist <= radius_um)

    # Globally process the shortest candidate edges first.
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


def find_label_raster(root, expected_cells, roi_um):
    """
    Try to identify the final merged label raster automatically.
    Returns (path, median_area_um2, median_diameter_um) or all None.
    """
    import tifffile

    pats = [
        "**/*cell*mask*.tif",
        "**/*segmentation*.tif",
        "**/*raster*.tif",
        "**/*label*.tif",
        "**/*mask*.tif",
    ]
    seen = set()
    candidates = []

    for pat in pats:
        for p in Path(root).glob(pat):
            if p in seen:
                continue
            seen.add(p)
            # Skip prepared nucleus/tile masks.
            s = str(p).lower()
            if "/input/mask/" in s or "/comseg_out/tile_" in s:
                continue
            candidates.append(p)

    best = None

    for p in candidates:
        try:
            a = tifffile.imread(p)
            a = np.squeeze(a)
            if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
                continue
            lab, cnt = np.unique(a, return_counts=True)
            areas = cnt[lab > 0]
            n = len(areas)
            if n == 0:
                continue

            rel = abs(n - expected_cells) / max(expected_cells, 1)
            if rel > 0.15:
                continue

            pix_um_x = roi_um[0] / a.shape[1]
            pix_um_y = roi_um[1] / a.shape[0]
            pix_area = pix_um_x * pix_um_y
            med_area = float(np.median(areas) * pix_area)
            med_diam = float(2 * np.sqrt(med_area / np.pi))

            score = rel
            if best is None or score < best[0]:
                best = (score, p, n, med_area, med_diam, a.shape)
        except Exception:
            continue

    if best is None:
        return None, np.nan, np.nan

    _, p, n, area, diam, shape = best
    log(
        f"  morphology raster: {p}  shape={shape} labels={n:,}  "
        f"median area={area:.2f} um2 diameter={diam:.2f} um"
    )
    return str(p), area, diam


def load_gt_tx(gt_root, meta, qv_min):
    """
    Load raw Xenium transcripts in the evaluation ROI.
    Used only for expression vectors and optional transcript assignment metrics.
    """
    root = Path(gt_root)
    p = pick(root, ["outs/transcripts.parquet", "transcripts.parquet"])
    if p is None:
        raise FileNotFoundError(f"cannot find transcripts.parquet under {root}")

    import pyarrow.parquet as pq

    schema = pq.read_schema(p)
    cols = list(schema.names)

    xc = choose_col(cols, ("x_location", "x"))
    yc = choose_col(cols, ("y_location", "y"))
    gc = choose_col(cols, ("feature_name", "gene", "gene_name"))
    cc = choose_col(cols, ("cell_id", "cellid", "cell"), required=False)
    qc = choose_col(cols, ("qv", "quality_value"), required=False)
    ig = choose_col(cols, ("is_gene",), required=False)

    want = [xc, yc, gc]
    for c in (cc, qc, ig):
        if c is not None and c not in want:
            want.append(c)

    px = float(meta["pixel_size"])
    x0_um = float(meta["x0"]) * px
    y0_um = float(meta["y0"]) * px
    x1_um = (float(meta["x0"]) + float(meta["size_x"])) * px
    y1_um = (float(meta["y0"]) + float(meta["size_y"])) * px

    filters = [
        (xc, ">=", x0_um),
        (xc, "<", x1_um),
        (yc, ">=", y0_um),
        (yc, "<", y1_um),
    ]
    if qc is not None:
        filters.append((qc, ">=", float(qv_min)))

    try:
        d = pd.read_parquet(p, columns=want, filters=filters)
    except Exception:
        d = pd.read_parquet(p, columns=want)
        d = d[
            pd.to_numeric(d[xc], errors="coerce").ge(x0_um)
            & pd.to_numeric(d[xc], errors="coerce").lt(x1_um)
            & pd.to_numeric(d[yc], errors="coerce").ge(y0_um)
            & pd.to_numeric(d[yc], errors="coerce").lt(y1_um)
        ]
        if qc is not None:
            d = d[pd.to_numeric(d[qc], errors="coerce") >= qv_min]

    if ig is not None:
        d = d[d[ig] == True]

    gene = d[gc].astype(str)
    bad = np.zeros(len(d), dtype=bool)
    for prefix in CONTROL_PREFIXES:
        bad |= gene.str.startswith(prefix, na=False).to_numpy()
    d = d.loc[~bad].copy()

    out = pd.DataFrame(
        {
            "gene": d[gc].astype(str).to_numpy(),
            "x_um": pd.to_numeric(d[xc], errors="coerce").to_numpy(float),
            "y_um": pd.to_numeric(d[yc], errors="coerce").to_numpy(float),
        }
    )

    if cc is not None:
        out["gt_cell"] = d[cc].astype(str).to_numpy()
    else:
        out["gt_cell"] = ""

    return out, p


def vector_metrics(pred_spots, gt_tx, pairs_df, max_cells=3000):
    """
    Compare gene vectors for spatially matched cell pairs.

    Pred vectors are reconstructed from deduplicated ComSeg spot assignments.
    GT vectors come from platform transcript cell_id.
    """
    if len(pairs_df) == 0:
        return dict(
            vec_cosine=np.nan,
            vec_js_dist=np.nan,
            vec_pearson=np.nan,
            vec_n=0,
            vec_genes=0,
        )

    selected = pairs_df.head(max_cells).copy()
    pred_ids = set(selected["pred_cell"].astype(np.int64))
    gt_ids = set(selected["gt_cell"].astype(str))

    p = pred_spots[pred_spots["cell"].isin(pred_ids)].copy()
    g = gt_tx[gt_tx["gt_cell"].isin(gt_ids)].copy()

    genes = np.union1d(
        p["gene"].astype(str).unique(),
        g["gene"].astype(str).unique(),
    )
    if len(genes) == 0:
        return dict(
            vec_cosine=np.nan,
            vec_js_dist=np.nan,
            vec_pearson=np.nan,
            vec_n=0,
            vec_genes=0,
        )

    gene_to_col = {x: i for i, x in enumerate(genes)}
    pair_to_row = {
        (int(r.pred_cell), str(r.gt_cell)): i
        for i, r in selected.reset_index(drop=True).iterrows()
    }
    pred_to_row = {
        int(r.pred_cell): i for i, r in selected.reset_index(drop=True).iterrows()
    }
    gt_to_row = {
        str(r.gt_cell): i for i, r in selected.reset_index(drop=True).iterrows()
    }

    from scipy.sparse import coo_matrix

    pr_row = p["cell"].map(pred_to_row)
    pr_ok = pr_row.notna()
    PR = coo_matrix(
        (
            np.ones(int(pr_ok.sum()), dtype=np.float32),
            (
                pr_row[pr_ok].astype(np.int32),
                p.loc[pr_ok, "gene"].map(gene_to_col).astype(np.int32),
            ),
        ),
        shape=(len(selected), len(genes)),
    ).tocsr()

    gt_row = g["gt_cell"].map(gt_to_row)
    gt_ok = gt_row.notna()
    GT = coo_matrix(
        (
            np.ones(int(gt_ok.sum()), dtype=np.float32),
            (
                gt_row[gt_ok].astype(np.int32),
                g.loc[gt_ok, "gene"].map(gene_to_col).astype(np.int32),
            ),
        ),
        shape=(len(selected), len(genes)),
    ).tocsr()

    s1 = np.asarray(PR.sum(axis=1)).ravel()
    s2 = np.asarray(GT.sum(axis=1)).ravel()
    valid = (s1 > 0) & (s2 > 0)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return dict(
            vec_cosine=np.nan,
            vec_js_dist=np.nan,
            vec_pearson=np.nan,
            vec_n=0,
            vec_genes=len(genes),
        )

    dot = np.asarray(PR[idx].multiply(GT[idx]).sum(axis=1)).ravel()
    n1 = np.sqrt(np.asarray(PR[idx].multiply(PR[idx]).sum(axis=1)).ravel())
    n2 = np.sqrt(np.asarray(GT[idx].multiply(GT[idx]).sum(axis=1)).ravel())
    cosine = dot / np.maximum(n1 * n2, 1e-12)

    js = []
    prs = []

    for i in idx:
        a = PR[i].toarray().ravel().astype(float)
        b = GT[i].toarray().ravel().astype(float)

        pa = a / max(a.sum(), 1e-12)
        pb = b / max(b.sum(), 1e-12)
        m = 0.5 * (pa + pb)

        with np.errstate(divide="ignore", invalid="ignore"):
            ka = np.where(pa > 0, pa * np.log2(pa / np.maximum(m, 1e-300)), 0)
            kb = np.where(pb > 0, pb * np.log2(pb / np.maximum(m, 1e-300)), 0)
        js.append(np.sqrt(max(0.0, 0.5 * (ka.sum() + kb.sum()))))

        r = safe_corr(a, b, "pearson")
        if np.isfinite(r):
            prs.append(r)

    return dict(
        vec_cosine=float(np.mean(cosine)),
        vec_js_dist=float(np.mean(js)),
        vec_pearson=float(np.mean(prs)) if prs else np.nan,
        vec_n=int(len(idx)),
        vec_genes=int(len(genes)),
    )


def assignment_metrics(pred_spots, gt_tx, pairs_df, meta):
    """
    Recover platform transcript identity by quantized (x,y,gene) keys.

    This is deliberately conservative. If <90% of ComSeg spot rows recover a
    platform transcript, assignment_accuracy is reported NaN rather than using
    a misleading partial subset.
    """
    if "gt_cell" not in gt_tx.columns or len(pairs_df) == 0:
        return {
            "assign_accuracy": np.nan,
            "overall_correct": np.nan,
            "assignment_recovery": 0.0,
        }

    px = float(meta["pixel_size"])
    # 0.01 native pixel resolution (~0.0021 um), still tolerant to float I/O.
    scale = 100.0

    p = pred_spots[["cell", "gene", "x_um", "y_um"]].copy()
    g = gt_tx[["gene", "x_um", "y_um", "gt_cell"]].copy()

    p["qx"] = np.rint(p["x_um"] / px * scale).astype(np.int64)
    p["qy"] = np.rint(p["y_um"] / px * scale).astype(np.int64)
    g["qx"] = np.rint(g["x_um"] / px * scale).astype(np.int64)
    g["qy"] = np.rint(g["y_um"] / px * scale).astype(np.int64)

    # Avoid many-to-many explosions at pathological duplicate keys.
    g = g.drop_duplicates(["qx", "qy", "gene"], keep="first")

    m = p.merge(
        g[["qx", "qy", "gene", "gt_cell"]],
        on=["qx", "qy", "gene"],
        how="left",
    )

    recovery = float(m["gt_cell"].notna().mean())
    if recovery < 0.90:
        log(
            f"  WARNING: transcript assignment recovery only {recovery*100:.1f}%; "
            "assignment accuracy is suppressed"
        )
        return {
            "assign_accuracy": np.nan,
            "overall_correct": np.nan,
            "assignment_recovery": recovery,
        }

    pred_to_gt = dict(
        zip(
            pairs_df["pred_cell"].astype(np.int64),
            pairs_df["gt_cell"].astype(str),
        )
    )

    m["matched_gt"] = m["cell"].map(pred_to_gt)
    comparable = m["matched_gt"].notna() & m["gt_cell"].notna()
    correct = comparable & (m["matched_gt"].astype(str) == m["gt_cell"].astype(str))

    acc = float(correct.sum() / max(comparable.sum(), 1))
    overall = float(correct.sum() / max(len(m), 1))

    return {
        "assign_accuracy": acc,
        "overall_correct": overall,
        "assignment_recovery": recovery,
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--gt-root", default=GT_ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=0.0,
                    help="0 = sqrt(ROI area / GT cell count)")
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--skip-vectors", action="store_true")
    ap.add_argument("--skip-assignment", action="store_true")
    ap.add_argument("--vector-max-cells", type=int, default=3000)
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("=" * 88)
    log(f"{METHOD} @ {DATASET} ({TISSUE})")
    log("=" * 88)

    meta = load_meta(root)

    px = float(meta["pixel_size"])
    roi_w_um = float(meta["size_x"]) * px
    roi_h_um = float(meta["size_y"]) * px

    log(
        f"ROI native px: x[{meta['x0']},{meta['x0']+meta['size_x']}) "
        f"y[{meta['y0']},{meta['y0']+meta['size_y']})"
    )
    log(f"ROI physical: {roi_w_um:.1f} x {roi_h_um:.1f} um")
    log(
        f"prep: {meta.get('n_transcripts',0):,} tx  "
        f"GT reference={meta.get('gt_expected_cells')}  qv={meta.get('qv_min')}"
    )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    log("\n[PREDICTION]")
    d = load_prediction(
        root,
        dedup=not a.no_dedup,
        save_merged=not a.no_dedup,
    )
    pred_spots, pred = build_pred_cells(d, meta)

    assigned_rows = int((pd.to_numeric(d["cell"], errors="coerce") > 0).sum())
    assign_rate = assigned_rows / max(len(d), 1)
    coverage_fraction = assign_rate

    ncell = len(pred)
    tx_counts = pred["n_tx"].to_numpy()

    log(
        f"  assigned rows {assigned_rows:,}/{len(d):,} "
        f"({assign_rate*100:.2f}%)"
    )
    log(f"  predicted cells {ncell:,}")
    log(
        f"  tx/cell median={np.median(tx_counts):.0f} "
        f"mean={np.mean(tx_counts):.1f} "
        f"p5/p95={np.percentile(tx_counts,5):.0f}/{np.percentile(tx_counts,95):.0f} "
        f"<10={(tx_counts<10).mean()*100:.3f}%"
    )

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    log("\n[GROUND TRUTH]")
    gt, gt_cells_path = load_gt_cells(a.gt_root, meta)
    log(f"  GT cells file: {gt_cells_path}")
    log(f"  ROI GT cells: {len(gt):,}")

    if meta.get("gt_expected_cells") is not None:
        diff = len(gt) - int(meta["gt_expected_cells"])
        log(f"  vs prep GT reference: {diff:+,}")

    if a.match_radius > 0:
        radius = float(a.match_radius)
    else:
        radius = math.sqrt((roi_w_um * roi_h_um) / max(len(gt), 1))

    log(f"  match radius = {radius:.3f} um")

    # ------------------------------------------------------------------
    # Detection/localization
    # ------------------------------------------------------------------
    pairs = greedy_match(
        pred[["x_um", "y_um"]].to_numpy(float),
        gt[["x_um", "y_um"]].to_numpy(float),
        radius,
    )

    pair_rows = []
    for pi, gi, dist in pairs:
        pair_rows.append(
            {
                "pred_index": pi,
                "pred_cell": int(pred.iloc[pi]["cell"]),
                "gt_index": gi,
                "gt_cell": str(gt.iloc[gi]["gt_cell"]),
                "dist_um": dist,
                "pred_n_tx": float(pred.iloc[pi]["n_tx"]),
                "gt_n_tx": float(gt.iloc[gi]["n_tx_platform"])
                if np.isfinite(gt.iloc[gi]["n_tx_platform"])
                else np.nan,
            }
        )
    pairs_df = pd.DataFrame(pair_rows)

    matched = len(pairs_df)
    precision = matched / max(len(pred), 1)
    recall = matched / max(len(gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    cell_ratio = len(pred) / max(len(gt), 1)

    distances = (
        pairs_df["dist_um"].to_numpy(float)
        if matched
        else np.asarray([], dtype=float)
    )
    loc_mean = float(np.mean(distances)) if matched else np.nan
    loc_median = float(np.median(distances)) if matched else np.nan
    loc_p95 = float(np.percentile(distances, 95)) if matched else np.nan

    log("\n[DETECTION / LOCALIZATION]")
    log(
        f"  pred={len(pred):,} GT={len(gt):,} ratio={cell_ratio:.4f} "
        f"matched={matched:,}"
    )
    log(
        f"  precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}"
    )
    log(
        f"  loc mean/median/p95="
        f"{loc_mean:.3f}/{loc_median:.3f}/{loc_p95:.3f} um"
    )

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------
    if matched and pairs_df["gt_n_tx"].notna().sum() >= 3:
        x = pairs_df["pred_n_tx"].to_numpy(float)
        y = pairs_df["gt_n_tx"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        count_pearson = safe_corr(x[ok], y[ok], "pearson")
        count_spearman = safe_corr(x[ok], y[ok], "spearman")
        count_mae = float(np.mean(np.abs(x[ok] - y[ok])))
        count_rmse = float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))
    else:
        count_pearson = count_spearman = count_mae = count_rmse = np.nan

    log("\n[COUNTS]")
    log(
        f"  pearson={count_pearson:.4f} spearman={count_spearman:.4f} "
        f"MAE={count_mae:.2f} RMSE={count_rmse:.2f}"
    )

    # ------------------------------------------------------------------
    # Morphology raster
    # ------------------------------------------------------------------
    log("\n[MORPHOLOGY]")
    raster_path, median_area_um2, median_diameter_um = find_label_raster(
        root,
        len(pred),
        (roi_w_um, roi_h_um),
    )
    if raster_path is None:
        log("  no trustworthy merged label raster found; diameter = NaN")

    # ------------------------------------------------------------------
    # GT transcript data for vector / assignment metrics
    # ------------------------------------------------------------------
    vec = {
        "vec_cosine": np.nan,
        "vec_js_dist": np.nan,
        "vec_pearson": np.nan,
        "vec_n": 0,
        "vec_genes": 0,
    }
    assignment = {
        "assign_accuracy": np.nan,
        "overall_correct": np.nan,
        "assignment_recovery": np.nan,
    }

    if (not a.skip_vectors) or (not a.skip_assignment):
        log("\n[RAW TRANSCRIPTS]")
        gt_tx, gt_tx_path = load_gt_tx(a.gt_root, meta, a.qv_min)
        log(f"  GT transcript file: {gt_tx_path}")
        log(f"  ROI gene transcripts after controls/QV: {len(gt_tx):,}")

        if not a.skip_vectors:
            log("\n[EXPRESSION VECTORS]")
            vec = vector_metrics(
                pred_spots,
                gt_tx,
                pairs_df,
                max_cells=a.vector_max_cells,
            )
            log(
                f"  cosine={vec['vec_cosine']:.4f} "
                f"JS={vec['vec_js_dist']:.4f} "
                f"pearson={vec['vec_pearson']:.4f} "
                f"n={vec['vec_n']:,} genes={vec['vec_genes']:,}"
            )

        if not a.skip_assignment:
            log("\n[TRANSCRIPT ASSIGNMENT]")
            assignment = assignment_metrics(
                pred_spots,
                gt_tx,
                pairs_df,
                meta,
            )
            log(
                f"  coordinate recovery={assignment['assignment_recovery']:.4f} "
                f"accuracy={assignment['assign_accuracy']:.4f} "
                f"overall_correct={assignment['overall_correct']:.4f}"
            )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    result = {
        "method": METHOD,
        "dataset": DATASET,
        "tissue": TISSUE,
        "gt_cells": int(len(gt)),
        "pred_cells": int(len(pred)),
        "matched": int(matched),
        "cell_ratio": float(cell_ratio),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "loc_mean_um": loc_mean,
        "loc_median_um": loc_median,
        "loc_p95_um": loc_p95,
        "match_radius_um": float(radius),
        "assign_rate": float(assign_rate),
        "coverage_fraction": float(coverage_fraction),
        "tx_per_cell_median": float(np.median(tx_counts)),
        "tx_per_cell_mean": float(np.mean(tx_counts)),
        "tx_per_cell_p5": float(np.percentile(tx_counts, 5)),
        "tx_per_cell_p95": float(np.percentile(tx_counts, 95)),
        "frag_lt10": float((tx_counts < 10).mean()),
        "median_area_um2": median_area_um2,
        "median_diameter_um": median_diameter_um,
        "count_pearson": count_pearson,
        "count_spearman": count_spearman,
        "count_mae": count_mae,
        "count_rmse": count_rmse,
        **assignment,
        **vec,
        "qv_min": float(a.qv_min),
        "uses_platform_prior": False,
        "detection_comparable": True,
        "centroid_source": "mean coordinates of deduplicated assigned transcripts",
        "deduplicated_tile_overlap": bool(not a.no_dedup),
        "raster_used": raster_path,
        "prep_gt_reference": meta.get("gt_expected_cells"),
        "prep_n_transcripts": meta.get("n_transcripts"),
        "note": (
            "Detection/localization uses centroids recomputed from assigned "
            "transcripts and never h5ad landmark centroids. Tile-overlap rows "
            "are deduplicated by (global_x,global_y,gene)."
        ),
    }

    metrics_path = out / "comseg_xenium_5_metrics.csv"
    pairs_path = out / "comseg_xenium_5_pairs.csv"
    pred_path = out / "comseg_xenium_5_pred_cells.csv"
    gt_path = out / "comseg_xenium_5_gt_cells.csv"
    meta_path = out / "comseg_xenium_5_eval_meta.json"

    pd.DataFrame([result]).to_csv(metrics_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    pred.to_csv(pred_path, index=False)
    gt.to_csv(gt_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    log("\n" + "=" * 88)
    for k, v in result.items():
        if k in {"note", "raster_used"}:
            continue
        log(f"{k:<28} {v}")
    log("=" * 88)
    log(f"metrics -> {metrics_path}")
    log(f"pairs   -> {pairs_path}")
    log(f"pred    -> {pred_path}")
    log(f"GT      -> {gt_path}")
    log(f"meta    -> {meta_path}")


if __name__ == "__main__":
    main()
