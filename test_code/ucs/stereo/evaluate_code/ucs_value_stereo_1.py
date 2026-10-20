#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_ucs_stereoseq.py  —— Stereo-seq x UCS evaluation (19 metrics)
=========================================================================
GT comes from the GEM cell column (CellBin), unlike Xenium where GT is built from cells.parquet.

⚠ assignment accuracy is circularly inflated: nuclei_mask and GT both come from the same cell column,
  the two share the same source. Only detection/localization/count metrics are valid; vector/assignment metrics are reference only.

Usage
-----
python eval_ucs_stereoseq.py
python eval_ucs_stereoseq.py \
    --mask  yourpath/ucs_stereoseq/segmentation_mask.tif \
    --gem   yourpath/dataset/stereoseq/stereo-seq_data_all/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz \
    --out   yourpath/eval_results/ucs/stereoseq.csv
"""

import os, gzip, argparse, warnings
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
SEP = "=" * 68
BIN_UM = 0.5    # Stereo-seq bin1 = 0.5 um/px


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", default="yourpath/ucs_stereoseq/segmentation_mask.tif")
    p.add_argument("--gem",  default="yourpath/dataset/stereoseq/"
        "stereo-seq_data_all/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz")
    p.add_argument("--out",  default="yourpath/eval_results/ucs/stereoseq.csv")
    p.add_argument("--match-radius-um", type=float, default=15.0,
                   help="match radius (um), cell diameter ~12.9um, use 1.2x")
    return p.parse_args()


# ── read GEM ────────────────────────────────────────────────────────────────────
def load_gem(path):
    print(f"  reading GEM: {os.path.basename(path)} ...")
    opener = gzip.open if path.endswith(".gz") else open
    skip = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"): skip += 1
            else: break
    df = pd.read_csv(path, sep="\t", skiprows=skip)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("geneid","gene"):      ren[c] = "gene"
        elif cl == "x":                  ren[c] = "x"
        elif cl == "y":                  ren[c] = "y"
        elif cl in ("midcounts","midcount","count","counts"): ren[c] = "count"
        elif cl in ("cell","cellid","label"): ren[c] = "cell"
    df = df.rename(columns=ren)
    if "count" not in df.columns: df["count"] = 1
    print(f"    {len(df):,} rows  cell>0: {(df.cell>0).sum():,}  "
          f"GT cells {df.loc[df.cell>0,'cell'].nunique():,}")
    return df


# ── build pred cell centroids + per-cell counts from mask ──────────────────────
def pred_from_mask(mask, x0_bin, y0_bin):
    """
    Return pred_df: label, cx_um, cy_um, n_px
    x0_bin / y0_bin = gem.x.min() / gem.y.min()
    ucs_stereoseq.py shifts by (x-x0)//bin_px when building gene_map;
    that offset must be added back here to align with the GT GEM bin coordinates.
    """
    print("  parsing pred mask ...")
    print(f"    GEM origin: x0={x0_bin}, y0={y0_bin} bin  "
          f"= {x0_bin*BIN_UM:.0f}, {y0_bin*BIN_UM:.0f} um")
    ids, cnt = np.unique(mask[mask > 0], return_counts=True)
    rows, cols = np.where(mask > 0)
    labels = mask[rows, cols]
    df = pd.DataFrame({"label": labels, "row": rows, "col": cols})
    cent = df.groupby("label")[["row","col"]].mean().reset_index()
    # ★ add back the GEM origin offset to align with GT coordinates
    cent["cx_um"] = (cent["col"] + x0_bin) * BIN_UM
    cent["cy_um"] = (cent["row"] + y0_bin) * BIN_UM
    # pixel count
    npx = pd.Series(cnt, index=ids, name="n_px")
    cent = cent.merge(npx.reset_index().rename(columns={"index":"label",0:"n_px"}),
                      on="label", how="left")
    if "n_px" not in cent.columns:
        # add directly from cnt
        cnt_df = pd.DataFrame({"label": ids, "n_px": cnt})
        cent = cent.merge(cnt_df, on="label", how="left")
    print(f"    pred cells {len(cent):,}  median pixels {cent.n_px.median():.0f}  "
          f"equivalent diameter {2*np.sqrt(cent.n_px.median()*BIN_UM**2/np.pi):.1f}um")
    return cent


# ── GT centroids + per-cell counts ──────────────────────────────────────────────
def gt_from_gem(gem):
    """GT centroid = MIDCounts-weighted (x,y) mean; GT count = total MIDCounts"""
    print("  building GT centroids (CellBin cell column)...")
    g = gem[gem.cell > 0].copy()
    # weighted centroid (by MIDCounts)
    g["wx"] = g.x * g["count"]
    g["wy"] = g.y * g["count"]
    agg = g.groupby("cell").agg(
        wx_sum=("wx", "sum"), wy_sum=("wy", "sum"),
        count_sum=("count", "sum")).reset_index()
    agg["cx_um"] = agg.wx_sum / agg.count_sum * BIN_UM
    agg["cy_um"] = agg.wy_sum / agg.count_sum * BIN_UM
    agg = agg.rename(columns={"count_sum": "gt_count"})
    print(f"    GT cells {len(agg):,}  median count {agg.gt_count.median():.0f}")
    return agg[["cell","cx_um","cy_um","gt_count"]]


# ── sparse Hungarian matching ──────────────────────────────────────────────────
def match_cells(pred_cent, gt_cent, radius_um):
    print(f"  matching cells (radius={radius_um}um)...")
    pred_xy = pred_cent[["cx_um","cy_um"]].values
    gt_xy   = gt_cent[["cx_um","cy_um"]].values
    tree = cKDTree(gt_xy)
    dists, gt_idx = tree.query(pred_xy, k=1, distance_upper_bound=radius_um)
    matched_mask = dists < radius_um
    matched_pred = np.where(matched_mask)[0]
    matched_gt   = gt_idx[matched_mask]
    # resolve one-to-many: keep only the nearest pred per GT
    best = {}
    for pi, gi, di in zip(matched_pred, matched_gt, dists[matched_mask]):
        if gi not in best or di < best[gi][1]:
            best[gi] = (pi, di)
    pairs = [(pi, gi, di) for gi, (pi, di) in best.items()]
    print(f"    pred {len(pred_cent):,}  GT {len(gt_cent):,}  matched {len(pairs):,}")
    return pairs


# ── per-cell pred counts (look up GEM points on the mask) ──────────────────────
def pred_counts_from_gem(mask, gem, x0_bin, y0_bin):
    """Map each GEM row coordinate onto the mask and sum MIDCounts per pred cell"""
    print("  computing pred per-cell MIDCounts (look up GEM on the mask)...")
    H, W = mask.shape
    xi = (gem.x.values - x0_bin).astype(np.int32)
    yi = (gem.y.values - y0_bin).astype(np.int32)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi, yi = xi[ok], yi[ok]
    counts = gem["count"].values[ok]
    pred_labels = mask[yi, xi]
    df = pd.DataFrame({"label": pred_labels, "count": counts})
    pred_cnt = df[df.label > 0].groupby("label")["count"].sum().reset_index()
    pred_cnt.columns = ["label", "pred_count"]
    print(f"    {(df.label > 0).sum():,} / {len(df):,} GEM rows fall inside pred cells")
    return pred_cnt


# ── main evaluation ────────────────────────────────────────────────────────────
def main():
    cfg = parse_args()
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    print(SEP); print("Stereo-seq x UCS evaluation"); print(SEP)

    # --- read data ---
    mask = tifffile.imread(cfg.mask)
    print(f"  mask shape {mask.shape}  dtype {mask.dtype}")
    gem = load_gem(cfg.gem)

    # GEM origin (offset of the mask coordinate system)
    x0_bin = int(gem.x.min())   # 2925
    y0_bin = int(gem.y.min())   # 4875

    # --- pred & GT ---
    pred_cent = pred_from_mask(mask, x0_bin, y0_bin)
    gt_cent   = gt_from_gem(gem)

    # --- pred counts ---
    pred_cnt = pred_counts_from_gem(mask, gem, x0_bin, y0_bin)
    pred_cent = pred_cent.merge(pred_cnt, on="label", how="left")
    pred_cent["pred_count"] = pred_cent.get("pred_count",
                                             pred_cent.get("n_px", 0)).fillna(0)

    # --- matching ---
    pairs = match_cells(pred_cent, gt_cent, cfg.match_radius_um)
    n_match = len(pairs)
    n_pred  = len(pred_cent)
    n_gt    = len(gt_cent)

    # --- metrics ---
    metrics = {}

    # basics
    metrics["gt_cells"]   = n_gt
    metrics["pred_cells"] = n_pred
    metrics["cell_ratio"] = n_pred / n_gt

    # detection
    prec   = n_match / n_pred if n_pred else 0
    recall = n_match / n_gt   if n_gt   else 0
    f1     = 2*prec*recall/(prec+recall) if prec+recall else 0
    metrics["matched"]   = n_match
    metrics["precision"] = prec
    metrics["recall"]    = recall
    metrics["f1"]        = f1

    # localization (centroid shift um)
    shifts = [pairs[i][2] for i in range(len(pairs))]
    if shifts:
        metrics["loc_mean_um"]   = np.mean(shifts)
        metrics["loc_median_um"] = np.median(shifts)
        metrics["loc_p95_um"]    = np.percentile(shifts, 95)
    else:
        metrics["loc_mean_um"] = metrics["loc_median_um"] = metrics["loc_p95_um"] = np.nan

    # count correlation
    if pairs:
        pi_arr = [p[0] for p in pairs]
        gi_arr = [p[1] for p in pairs]
        p_cnt = pred_cent.iloc[pi_arr]["pred_count"].values.astype(float)
        g_cnt = gt_cent.iloc[gi_arr]["gt_count"].values.astype(float)
        if len(p_cnt) >= 3 and p_cnt.std() > 0 and g_cnt.std() > 0:
            metrics["count_pearson"]  = pearsonr(p_cnt, g_cnt)[0]
            metrics["count_spearman"] = spearmanr(p_cnt, g_cnt)[0]
        else:
            metrics["count_pearson"] = metrics["count_spearman"] = np.nan
        metrics["count_mae"]  = float(np.mean(np.abs(p_cnt - g_cnt)))
        metrics["count_rmse"] = float(np.sqrt(np.mean((p_cnt - g_cnt)**2)))
    else:
        for k in ["count_pearson","count_spearman","count_mae","count_rmse"]:
            metrics[k] = np.nan

    # gene vectors (top-gene cosine)
    print("  computing gene-vector metrics (pred vs GT, on matched pairs)...")
    try:
        # build pred gene vectors
        gene_names = np.sort(gem.gene.unique())
        g2i = {g: i for i, g in enumerate(gene_names)}
        ng = len(gene_names)
        H, W = mask.shape
        x0 = int(gem.x.min()); y0 = int(gem.y.min())
        xi = np.clip((gem.x.values - x0).astype(np.int32), 0, W-1)
        yi = np.clip((gem.y.values - y0).astype(np.int32), 0, H-1)
        ok = (xi < W) & (yi < H)
        plabels = mask[yi[ok], xi[ok]]
        gi_vals = gem.gene.map(g2i).values[ok]
        counts  = gem["count"].values[ok]

        # GT gene vectors (for matched GT cells)
        gi_arr = [p[1] for p in pairs]
        pi_arr = [p[0] for p in pairs]
        gt_cells_matched = gt_cent.iloc[gi_arr]["cell"].values
        pred_labels_matched = pred_cent.iloc[pi_arr]["label"].values

        # subsample (up to 2000 pairs)
        np.random.seed(42)
        idx = np.random.permutation(len(pairs))[:2000]
        cosines = []
        for i in idx:
            pi, gi = pi_arr[i], gi_arr[i]
            pl = pred_cent.iloc[pi]["label"]
            gc = gt_cent.iloc[gi]["cell"]
            # pred vector
            pm = plabels == pl
            pv = np.zeros(ng)
            if pm.any():
                np.add.at(pv, gi_vals[pm], counts[pm])
            # GT vector
            gm = (gem.cell == gc)
            gv = np.zeros(ng)
            gene_idx = gem.gene.map(g2i).values
            np.add.at(gv, gene_idx[gm], gem["count"].values[gm])
            # cosine
            nn = np.linalg.norm(pv) * np.linalg.norm(gv)
            if nn > 0:
                cosines.append(np.dot(pv, gv) / nn)

        if cosines:
            metrics["vec_cosine"] = float(np.mean(cosines))
        else:
            metrics["vec_cosine"] = np.nan
    except Exception as e:
        print(f"    gene-vector computation failed: {e}")
        metrics["vec_cosine"] = np.nan

    metrics["vec_js_dist"]  = np.nan   # too expensive, skip
    metrics["vec_pearson"]  = np.nan

    # assignment (same source, circularly inflated, record only)
    metrics["assign_accuracy"] = np.nan  # same source, skip
    metrics["assign_overlap"]  = np.nan

    # --- print ---
    print(f"\n{SEP}")
    print(f"results (Stereo-seq x UCS)")
    print(f"{SEP}")
    print(f"  {'GT cells':16} {metrics['gt_cells']:,}")
    print(f"  {'pred cells':16} {metrics['pred_cells']:,}")
    print(f"  {'cell_ratio':16} {metrics['cell_ratio']:.4f}")
    print(f"  {'matched':16} {metrics['matched']:,}")
    print(f"  {'precision':16} {metrics['precision']:.4f}")
    print(f"  {'recall':16} {metrics['recall']:.4f}")
    print(f"  {'F1':16} {metrics['f1']:.4f}")
    print(f"  {'loc_median um':16} {metrics['loc_median_um']:.2f}")
    print(f"  {'loc_p95 um':16} {metrics['loc_p95_um']:.2f}")
    print(f"  {'count_pearson':16} {metrics['count_pearson']:.4f}")
    print(f"  {'count_spearman':16} {metrics['count_spearman']:.4f}")
    print(f"  {'count_mae':16} {metrics['count_mae']:.1f}")
    print(f"  {'count_rmse':16} {metrics['count_rmse']:.1f}")
    print(f"  {'vec_cosine':16} {metrics['vec_cosine']:.4f}")
    print(f"\n  ⚠ assign_accuracy and vec_js/pearson skipped (same source / too expensive)")

    # --- write out ---
    pd.DataFrame([metrics]).to_csv(cfg.out, index=False)
    print(f"\n  saved -> {cfg.out}")
    print(SEP)


if __name__ == "__main__":
    main()
