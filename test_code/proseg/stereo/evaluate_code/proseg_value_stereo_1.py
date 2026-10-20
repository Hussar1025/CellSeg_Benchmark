#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_value_stereo_multi.py -- ProSeg × Stereo-seq evaluation (19 metrics, multi-dataset)
=========================================================================
Rewritten from proseg_value_stereo_1.py (E16.5_E2S6-specific); only the hardcoded
paths were switched to --dataset. The evaluation logic (GT weighted-centroid
construction, matching, count / gene-vector computation) is unchanged.

Two important warnings carried over from the original:
  1. assign_accuracy is circularly inflated: ProSeg prior = CellBin cell column,
     GT is also CellBin; only detection / localization / count / vector metrics
     are valid, assign metrics are set to NaN.
  2. ProSeg does not officially support Stereo-seq (proseg_stereo_run.py already
     did an experimental adaptation by "expanding UMI into fake single molecules"),
     so these results are for internal-consistency reference only and do not
     represent ProSeg's true official performance on this platform.

Usage:
  conda activate proseg_eval
  cd yourpath/proseg
  python proseg_value_stereo_multi.py --dataset E14.5_E1S3
  python proseg_value_stereo_multi.py --dataset E16.5_E2S7
"""

import os, gzip, warnings, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

DATA_DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
BIN_UM = 0.5          # 1 bin = 0.5 µm
MATCH_RADIUS = 15.0   # um; consistent with E16.5_E2S6 evaluation (carried over from the original)

# keep consistent with the DATASETS dict in proseg_stereo_run.py
DATASETS = {
    "E14.5_E1S3": f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
    "E16.5_E2S6": f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
    "E16.5_E2S7": f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proseg_dir = f"yourpath/proseg/output/{a.dataset}"
    gem_path = DATASETS[a.dataset]
    out_path = a.out or f"yourpath/proseg/value/{a.dataset}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(SEP); print(f"ProSeg × Stereo-seq evaluation -- {a.dataset}"); print(SEP)
    print("warning: ProSeg does not officially support Stereo-seq (expanded-UMI adaptation); results are for internal reference only")
    print("warning: assign metrics are circular (same source) and set to NaN; detection/localization/count/vector metrics are valid.")

    # ========================================================================
    # 1. read ProSeg cell-metadata
    # ========================================================================
    print("\n[1] read ProSeg cell-metadata ...")
    pred = pd.read_csv(os.path.join(proseg_dir, "cell-metadata.csv.gz"))
    print(f"  cell count: {len(pred):,}")
    print(f"  centroid_x: [{pred.centroid_x.min():.1f},{pred.centroid_x.max():.1f}] µm")
    print(f"  centroid_y: [{pred.centroid_y.min():.1f},{pred.centroid_y.max():.1f}] µm")
    pred_xy = pred[["centroid_x", "centroid_y"]].values.astype(float)
    n_pred = len(pred)

    # ========================================================================
    # 2. build GT from GEM (MIDCounts weighted centroid, bin->um)
    # ========================================================================
    print("\n[2] read GEM to build GT centroids ...")
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x_bin"
        elif cl == "y":
            ren[c] = "y_bin"
        elif cl in ("midcounts", "midcount", "count", "counts"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1

    g_ = gem[gem.cell > 0].copy()
    g_["wx"] = g_.x_bin * g_["count"]
    g_["wy"] = g_.y_bin * g_["count"]
    agg = g_.groupby("cell").agg(
        wx_sum=("wx", "sum"), wy_sum=("wy", "sum"),
        count_sum=("count", "sum")).reset_index()
    agg["cx_um"] = agg.wx_sum / agg.count_sum * BIN_UM
    agg["cy_um"] = agg.wy_sum / agg.count_sum * BIN_UM
    agg = agg.rename(columns={"count_sum": "gt_count"})
    n_gt = len(agg)
    gt_xy = agg[["cx_um", "cy_um"]].values.astype(float)
    print(f"  GT cells: {n_gt:,}")
    print(f"  GT cx_um: [{gt_xy[:,0].min():.1f},{gt_xy[:,0].max():.1f}] µm")
    print(f"  GT cy_um: [{gt_xy[:,1].min():.1f},{gt_xy[:,1].max():.1f}] µm")

    px_span = pred_xy[:, 0].max() - pred_xy[:, 0].min()
    gx_span = gt_xy[:, 0].max() - gt_xy[:, 0].min()
    print(f"  x span: pred={px_span:.0f}µm  GT={gx_span:.0f}µm"
          f"  {'✓' if abs(px_span-gx_span)/max(px_span,gx_span)<0.3 else '⚠ large discrepancy'}")

    gt_gene_agg = gem[gem.cell > 0].groupby(["cell", "gene"])["count"].sum().unstack(fill_value=0)

    # ========================================================================
    # 3. read ProSeg transcript-metadata
    # ========================================================================
    print("\n[3] read ProSeg transcript-metadata ...")
    tx_pred = pd.read_csv(os.path.join(proseg_dir, "transcript-metadata.csv.gz"))
    print(f"  transcripts: {len(tx_pred):,}")
    tx_asgn = tx_pred[tx_pred["background"] == False].copy()
    tx_asgn["cell_int"] = tx_asgn["assignment"].astype(int)
    pred_cnt_ser = tx_asgn.groupby("cell_int").size()
    pred_gene_grp = tx_asgn.groupby(["cell_int", "gene"]).size().unstack(fill_value=0)
    print(f"  non-background: {len(tx_asgn):,}  assignment rate: {len(tx_asgn)/len(tx_pred)*100:.1f}%")

    # ========================================================================
    # 4. match
    # ========================================================================
    print(f"\n[4] match (radius={MATCH_RADIUS} um)...")
    tree = cKDTree(gt_xy)
    dists, gi = tree.query(pred_xy, k=1, distance_upper_bound=MATCH_RADIUS)
    best = {}
    for pi in range(n_pred):
        if dists[pi] >= MATCH_RADIUS:
            continue
        g = int(gi[pi])
        if g not in best or dists[pi] < best[g][1]:
            best[g] = (pi, dists[pi])
    pairs = [(pi, g, d_) for g, (pi, d_) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred if n_pred else 0
    recall = n_match / n_gt if n_gt else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0
    shifts = [p[2] for p in pairs]
    print(f"  pred {n_pred:,}  GT {n_gt:,}  matched {n_match:,}")
    print(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={recall:.4f}")

    pi_arr = [p[0] for p in pairs]
    gi_arr = [p[1] for p in pairs]

    # ========================================================================
    # 5. count metrics
    # ========================================================================
    pred_cell_ids = pred.iloc[pi_arr]["cell"].values
    p_cnt = np.array([pred_cnt_ser.get(int(cid), 0) for cid in pred_cell_ids], dtype=float)
    g_cnt = agg.iloc[gi_arr]["gt_count"].values.astype(float)

    if len(p_cnt) >= 3 and p_cnt.std() > 0 and g_cnt.std() > 0:
        count_pearson, _ = pearsonr(p_cnt, g_cnt)
        count_spearman, _ = spearmanr(p_cnt, g_cnt)
        count_mae = float(np.mean(np.abs(p_cnt - g_cnt)))
        count_rmse = float(np.sqrt(np.mean((p_cnt - g_cnt) ** 2)))
    else:
        count_pearson = count_spearman = count_mae = count_rmse = np.nan
    print(f"\n[5] counts: pearson={count_pearson:.4f}  spearman={count_spearman:.4f}")

    # ========================================================================
    # 6. gene vector (sample 800 pairs)
    # ========================================================================
    print("\n[6] gene vector ...")
    gene_names_all = np.sort(tx_asgn["gene"].dropna().unique())
    g2i = {g_n: i for i, g_n in enumerate(gene_names_all)}
    ng = len(gene_names_all)

    np.random.seed(42)
    samp = np.random.choice(len(pairs), min(800, len(pairs)), replace=False)
    cosines, jsd, prs = [], [], []
    for idx in samp:
        pi, gi_val, _ = pairs[idx]
        pcid = int(pred.iloc[pi]["cell"])
        gcid = int(agg.iloc[gi_val]["cell"])
        pv = np.zeros(ng)
        if pcid in pred_gene_grp.index:
            for gname, cnt in pred_gene_grp.loc[pcid].items():
                if gname in g2i and cnt > 0:
                    pv[g2i[gname]] = cnt
        gv = np.zeros(ng)
        if gcid in gt_gene_agg.index:
            for gname, cnt in gt_gene_agg.loc[gcid].items():
                if gname in g2i and cnt > 0:
                    gv[g2i[gname]] = cnt
        nn = np.linalg.norm(pv) * np.linalg.norm(gv)
        if nn > 0:
            cosines.append(float(np.dot(pv, gv) / nn))
            pn = pv / pv.sum() if pv.sum() > 0 else pv
            gn = gv / gv.sum() if gv.sum() > 0 else gv
            jsd.append(float(jensenshannon(pn, gn)))
            if pv.std() > 0 and gv.std() > 0:
                prs.append(float(pearsonr(pv, gv)[0]))

    vec_cosine = float(np.mean(cosines)) if cosines else np.nan
    vec_js = float(np.mean(jsd)) if jsd else np.nan
    vec_pearson = float(np.mean(prs)) if prs else np.nan
    print(f"  cosine={vec_cosine:.4f}  JS={vec_js:.4f}  pearson={vec_pearson:.4f}")

    # ========================================================================
    # summary
    # ========================================================================
    metrics = dict(
        method="ProSeg", dataset=a.dataset,
        gt_cells=n_gt,
        pred_cells=n_pred,
        cell_ratio=round(n_pred / n_gt, 4),
        matched=n_match,
        precision=round(prec, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        loc_mean_um=round(float(np.mean(shifts)), 4) if shifts else np.nan,
        loc_median_um=round(float(np.median(shifts)), 4) if shifts else np.nan,
        loc_p95_um=round(float(np.percentile(shifts, 95)), 4) if shifts else np.nan,
        count_pearson=round(count_pearson, 4),
        count_spearman=round(count_spearman, 4),
        count_mae=round(count_mae, 2),
        count_rmse=round(count_rmse, 2),
        vec_cosine=round(vec_cosine, 4),
        vec_js_dist=round(vec_js, 4),
        vec_pearson=round(vec_pearson, 4),
        assign_overlap=np.nan,
        assign_accuracy=np.nan,
    )
    print(f"\n{SEP}\nResult (ProSeg × {a.dataset}, experimental)\n{SEP}")
    for k, v in metrics.items():
        if k not in ("method", "dataset"):
            print(f"  {k:22} {v}")
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    print(f"\n  written -> {out_path}")


if __name__ == "__main__":
    main()