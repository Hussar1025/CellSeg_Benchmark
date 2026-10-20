#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCS evaluation -- 19-metric unified framework
=========================================================================
Adapted for the Xenium ROI (human lymph node):
  GT cells         : cells.parquet   (x_centroid, y_centroid, cell_area)
  GT per-transcript: transcripts.parquet cell_id column (94.5% assigned)
  Prediction       : UCS segmentation_mask.tif (label map in bin space)

19 metrics in five groups:
  Basic(3)        : GT/pred cell counts, ratio
  Detection(4)    : matched count, Precision, Recall, F1
  Localization(3) : mean / median / p95 centroid shift
  Count(4)        : transcript count Pearson / Spearman / MAE / RMSE
  Vector(3)       : gene vector cosine / JS distance / Pearson
  Assignment(2)   : transcript ID overlap, assignment accuracy
(3+4+3+4+3+2 = 19)

Alignment pitfalls (lessons learned):
  * Coordinate system: pred is in bin space (origin = ROI bottom-left,
    1 bin = bin_um um); GT is in um global coordinates. Convert GT to bin
    space before comparing.
  * valid_pairs: accuracy/cosine are computed only on transcripts assigned
    to a cell by both GT and pred; otherwise coordinate misalignment inflates
    the metrics (survivorship bias). The valid_pairs count is reported too.
  * Matching: Hungarian matching degrades on large images; above 2000 cells
    use centroid-based greedy nearest-neighbour matching.
  * Circular inflation: when UCS uses a nucleus prior and GT derives from the
    same nuclei, transcript accuracy is inflated. Xenium GT (cell_id) is
    independent of UCS's nucleus input, so accuracy here is valid -- but a
    note is still printed to distinguish it from MERFISH (where it is
    meaningless).

Usage
-----
python eval_ucs_xenium.py \
    --label-path yourpath/ucs_xenium_roi/ucs_k5_i2_t5/pred/segmentation_mask.tif \
    --data-dir   yourpath/dataset/xenium_roi_crop \
    --out        yourpath/eval_results/ucs/xenium_lymph.csv
"""

import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment

SEP = "=" * 72


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-path", required=True, help="UCS segmentation_mask.tif")
    p.add_argument("--data-dir",   required=True, help="xenium_roi_crop directory")
    p.add_argument("--out",        default=None,  help="result CSV path")
    p.add_argument("--method",     default="UCS")
    p.add_argument("--dataset",    default="xenium_lymph")
    p.add_argument("--bin-um",     type=float, default=1.0)
    p.add_argument("--qv-min",     type=float, default=20.0)
    p.add_argument("--match-dist", type=float, default=None,
                   help="max centroid distance for matching (um), default = cell diameter")
    return p.parse_args()


def find_one(base, patterns, label, required=True):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(base, "**", pat), recursive=True))
        if hits:
            hits.sort(key=len)
            return hits[0]
    if required:
        raise FileNotFoundError(f"not found {label}: {patterns}")
    return None


def read_table(path, columns=None):
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


# ══════════════════════════════════════════════════════════════════════════════
def load_all(cfg):
    base = cfg.data_dir
    f_tx    = find_one(base, ["transcripts.parquet", "transcripts.csv.gz"], "transcripts")
    f_cells = find_one(base, ["cells.parquet", "cells.csv.gz", "cells.csv"], "GT cells")
    f_roi   = find_one(base, ["experiment_roi.json"], "ROI metadata", required=False)

    # ROI origin + pixel size
    x0 = y0 = None
    pixel = 0.2125
    if f_roi:
        meta = json.load(open(f_roi))
        roi = meta.get("roi", {})
        x0 = roi.get("x_min_um"); y0 = roi.get("y_min_um")
        pixel = meta.get("pixels", {}).get("pixel_size_um", 0.2125)

    # predicted mask
    mask = tifffile.imread(cfg.label_path)
    H, W = mask.shape
    print(f"  predicted mask: {mask.shape}  labels {len(np.unique(mask))-1:,}")

    # transcripts
    cols = ["x_location", "y_location", "feature_name"]
    opt  = [c for c in ["qv", "is_gene", "cell_id"] if c in
            (pd.read_parquet(f_tx).head(1) if f_tx.endswith("parquet")
             else pd.read_csv(f_tx, nrows=1)).columns]
    tx = read_table(f_tx, columns=cols + opt)
    if "is_gene" in tx.columns:
        tx = tx[tx["is_gene"] == True]
    if "qv" in tx.columns:
        tx = tx[tx["qv"] >= cfg.qv_min]
    tx = tx.reset_index(drop=True)

    # ROI origin fallback: use the transcript min when no json
    if x0 is None:
        x0 = float(np.floor(tx.x_location.min()))
        y0 = float(np.floor(tx.y_location.min()))
    print(f"  ROI origin ({x0:.1f}, {y0:.1f}) um   bin {cfg.bin_um} um")

    # transcripts -> bin coordinates (aligned with mask)
    tx["bx"] = ((tx.x_location - x0) / cfg.bin_um).astype(int)
    tx["by"] = ((tx.y_location - y0) / cfg.bin_um).astype(int)
    inb = (tx.bx >= 0) & (tx.bx < W) & (tx.by >= 0) & (tx.by < H)
    tx = tx[inb].reset_index(drop=True)

    # predicted assignment: which pred label each transcript falls in
    tx["pred_cell"] = mask[tx.by.values, tx.bx.values]

    # GT assignment: cell_id column (UNASSIGNED -> 0)
    if "cell_id" in tx.columns:
        cid = tx["cell_id"].astype(str)
        tx["gt_cell"] = cid.where(cid != "UNASSIGNED", "0")
    else:
        tx["gt_cell"] = "0"

    # GT cell centroids (converted to bin space)
    cells = read_table(f_cells)
    cells = cells.copy()
    cells["gx"] = (cells.x_centroid - x0) / cfg.bin_um
    cells["gy"] = (cells.y_centroid - y0) / cfg.bin_um
    cells["cell_id"] = cells["cell_id"].astype(str)

    return mask, tx, cells, pixel


# ══════════════════════════════════════════════════════════════════════════════
def pred_centroids(mask):
    """Centroid of each pred label (bin coordinates)."""
    ys, xs = np.nonzero(mask)
    labels = mask[ys, xs]
    df = pd.DataFrame({"label": labels, "x": xs, "y": ys})
    c = df.groupby("label")[["x", "y"]].mean()
    return c  # index = label


def match_cells(gt_xy, pred_xy, max_dist):
    """
    Centroid matching. Hungarian for small graphs, greedy nearest-neighbour
    for large (>2000). Returns the matched-pair list [(gt_idx, pred_idx, dist), ...]
    """
    n_gt, n_pred = len(gt_xy), len(pred_xy)
    if n_gt == 0 or n_pred == 0:
        return []

    # greedy nearest-neighbour: for each GT find the nearest pred, one-to-one
    if max(n_gt, n_pred) > 2000:
        tree = cKDTree(pred_xy)
        dist, idx = tree.query(gt_xy, k=1)
        order = np.argsort(dist)
        used_pred = set()
        pairs = []
        for gi in order:
            if dist[gi] > max_dist:
                continue
            pi = idx[gi]
            if pi in used_pred:
                # fall back to the next nearest unused pred
                dd, ii = tree.query(gt_xy[gi], k=min(5, n_pred))
                found = False
                for d2, i2 in zip(np.atleast_1d(dd), np.atleast_1d(ii)):
                    if i2 not in used_pred and d2 <= max_dist:
                        pi, found = i2, True
                        dist[gi] = d2
                        break
                if not found:
                    continue
            used_pred.add(pi)
            pairs.append((gi, pi, float(dist[gi])))
        return pairs

    # Hungarian (exact for small graphs)
    from scipy.spatial.distance import cdist
    D = cdist(gt_xy, pred_xy)
    D[D > max_dist] = 1e6
    ri, ci = linear_sum_assignment(D)
    return [(int(r), int(c), float(D[r, c]))
            for r, c in zip(ri, ci) if D[r, c] < 1e6]


# ══════════════════════════════════════════════════════════════════════════════
def evaluate(cfg):
    print(SEP); print(f"UCS evaluation  {cfg.dataset}"); print(SEP)
    mask, tx, cells, pixel = load_all(cfg)

    diam = 2 * np.sqrt(cells["cell_area"].median() / np.pi) if \
        "cell_area" in cells.columns else 7.0
    max_dist_um = cfg.match_dist if cfg.match_dist else diam
    max_dist_bin = max_dist_um / cfg.bin_um
    print(f"  GT cell diameter {diam:.1f} um  ->  match threshold {max_dist_um:.1f} um "
          f"({max_dist_bin:.1f} bin)")

    pc = pred_centroids(mask)                       # index=label, cols x,y
    gt_xy   = cells[["gx", "gy"]].values
    pred_xy = pc[["x", "y"]].values
    print(f"\n  GT {len(gt_xy):,} cells   pred {len(pred_xy):,} cells")

    pairs = match_cells(gt_xy, pred_xy, max_dist_bin)
    n_match = len(pairs)
    gt_ids   = cells["cell_id"].values
    pred_lbl = pc.index.values

    # ---- detection metrics ----
    precision = n_match / max(len(pred_xy), 1)
    recall    = n_match / max(len(gt_xy), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # ---- localization metrics ----
    if n_match:
        dists_um = np.array([d for _, _, d in pairs]) * cfg.bin_um
        loc_mean, loc_med, loc_p95 = (float(np.mean(dists_um)),
                                      float(np.median(dists_um)),
                                      float(np.percentile(dists_um, 95)))
    else:
        loc_mean = loc_med = loc_p95 = np.nan

    # ---- per matched cell transcript counts & gene vectors ----
    genes = np.sort(tx.feature_name.unique())
    g2i = {g: i for i, g in enumerate(genes)}
    tx["gi"] = tx.feature_name.map(g2i).values

    # pred cell -> gene count vector
    pred_grp = tx[tx.pred_cell > 0].groupby(["pred_cell", "gi"]).size()
    # GT cell -> gene count vector
    gt_grp   = tx[tx.gt_cell != "0"].groupby(["gt_cell", "gi"]).size()

    def vec(grp, key, n_gene):
        v = np.zeros(n_gene)
        if key in grp.index.get_level_values(0):
            sub = grp.loc[key]
            v[sub.index.values] = sub.values
        return v

    gt_counts, pred_counts = [], []
    cos_list, js_list = [], []
    valid_vec = 0
    for gi, pi, _ in pairs:
        gid = gt_ids[gi]
        plb = pred_lbl[pi]
        vg = vec(gt_grp,   gid, len(genes))
        vp = vec(pred_grp, plb, len(genes))
        cg, cp = vg.sum(), vp.sum()
        gt_counts.append(cg)
        pred_counts.append(cp)
        if cg > 0 and cp > 0:
            valid_vec += 1
            ng, npv = vg / cg, vp / cp
            denom = np.linalg.norm(vg) * np.linalg.norm(vp)
            cos_list.append(float(vg @ vp / denom) if denom > 0 else 0.0)
            js_list.append(float(jensenshannon(ng, npv, base=2)))

    gt_counts  = np.array(gt_counts)
    pred_counts = np.array(pred_counts)

    # ---- count metrics ----
    valid_c = (gt_counts > 0) & (pred_counts > 0)
    if valid_c.sum() >= 2:
        cnt_pear = float(pearsonr(gt_counts[valid_c], pred_counts[valid_c])[0])
        cnt_spear = float(spearmanr(gt_counts[valid_c], pred_counts[valid_c])[0])
        cnt_mae  = float(np.mean(np.abs(gt_counts[valid_c] - pred_counts[valid_c])))
        cnt_rmse = float(np.sqrt(np.mean((gt_counts[valid_c] - pred_counts[valid_c])**2)))
    else:
        cnt_pear = cnt_spear = cnt_mae = cnt_rmse = np.nan

    # ---- vector metrics ----
    vec_cos = float(np.mean(cos_list)) if cos_list else np.nan
    vec_js  = float(np.mean(js_list))  if js_list  else np.nan
    if valid_vec >= 2:
        # per-cell cosine already computed; now overall gene abundance Pearson
        gt_tot   = np.zeros(len(genes))
        pred_tot = np.zeros(len(genes))
        for gi, pi, _ in pairs:
            gt_tot   += vec(gt_grp,   gt_ids[gi],  len(genes))
            pred_tot += vec(pred_grp, pred_lbl[pi], len(genes))
        m = (gt_tot > 0) | (pred_tot > 0)
        vec_pear = float(pearsonr(gt_tot[m], pred_tot[m])[0]) if m.sum() >= 2 else np.nan
    else:
        vec_pear = np.nan

    # ---- assignment metrics: per-transcript GT vs pred ----
    #   build a majority-vote map pred_label -> gt_cell, then check each RNA
    gt2pred = {gt_ids[gi]: pred_lbl[pi] for gi, pi, _ in pairs}
    pred2gt = {pred_lbl[pi]: gt_ids[gi] for gi, pi, _ in pairs}

    # compute only on transcripts assigned by both GT and pred (valid_pairs)
    both = tx[(tx.gt_cell != "0") & (tx.pred_cell > 0)].copy()
    both["pred_maps_to_gt"] = both.pred_cell.map(pred2gt)
    valid_pairs = len(both)
    if valid_pairs:
        assign_acc = float((both["pred_maps_to_gt"].astype(str)
                            == both["gt_cell"].astype(str)).mean())
        # ID overlap: fraction of transcripts where pred and gt map to the same pair
        overlap = float(both["pred_maps_to_gt"].notna().mean())
    else:
        assign_acc = overlap = np.nan

    total_asg = ((tx.gt_cell != "0") | (tx.pred_cell > 0)).sum()
    print(f"\n  valid_pairs (assigned by both GT & pred) = {valid_pairs:,} / "
          f"{total_asg:,} assigned transcripts")
    print(f"  per-cell vector valid = {valid_vec:,} / {n_match:,} matched pairs")

    # ---- summary ----
    R = {
        "method":  cfg.method,
        "dataset": cfg.dataset,
        # basic
        "gt_cells":   len(gt_xy),
        "pred_cells": len(pred_xy),
        "cell_ratio": len(pred_xy) / max(len(gt_xy), 1),
        # detection
        "matched":   n_match,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        # localization
        "loc_mean_um":   loc_mean,
        "loc_median_um": loc_med,
        "loc_p95_um":    loc_p95,
        # count
        "count_pearson":  cnt_pear,
        "count_spearman": cnt_spear,
        "count_mae":      cnt_mae,
        "count_rmse":     cnt_rmse,
        # vector
        "vec_cosine":  vec_cos,
        "vec_js_dist": vec_js,
        "vec_pearson": vec_pear,
        # assignment
        "assign_overlap":  overlap,
        "assign_accuracy": assign_acc,
        # diagnostics (not counted in the 19, kept for the record)
        "_valid_pairs":     valid_pairs,
        "_valid_vec_pairs": valid_vec,
    }

    # ---- print ----
    print(f"\n{SEP}\nResults\n{SEP}")
    groups = [
        ("Basic", ["gt_cells", "pred_cells", "cell_ratio"]),
        ("Detection", ["matched", "precision", "recall", "f1"]),
        ("Localization", ["loc_mean_um", "loc_median_um", "loc_p95_um"]),
        ("Count", ["count_pearson", "count_spearman", "count_mae", "count_rmse"]),
        ("Vector", ["vec_cosine", "vec_js_dist", "vec_pearson"]),
        ("Assignment", ["assign_overlap", "assign_accuracy"]),
    ]
    for gname, keys in groups:
        print(f"\n  [{gname}]")
        for k in keys:
            v = R[k]
            s = f"{v:,.4f}" if isinstance(v, float) else f"{v:,}"
            print(f"    {k:<18s} {s}")

    print(f"\n  Note: GT (transcripts.cell_id) is independent of UCS's nucleus prior,"
          f"so assign_accuracy={assign_acc:.3f} is valid")
    print(f"    (cf. MERFISH: there GT shares the nucleus source, so assignment is circularly inflated and meaningless)")

    # ---- save ----
    out = cfg.out or f"yourpath/eval_results/{cfg.method.lower()}/" \
                     f"{cfg.dataset}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([R]).to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return R


if __name__ == "__main__":
    evaluate(parse_args())