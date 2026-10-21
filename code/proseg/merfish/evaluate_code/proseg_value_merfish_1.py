#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_proseg_merfish1.py  --  ProSeg × MERFISH mouse brain evaluation (19 metrics)
=========================================================================
ProSeg v3 output columns (same as xenium_3/stereo_1):
  cell-metadata:       cell(ID,int), centroid_x, centroid_y (um)
  transcript-metadata: gene, assignment(float->int), background(bool), x, y (um)

GT: cell_metadata_S1R1.csv (center_x, center_y, um)
    cells inside the ROI [3589,5761] x [2327,4499] um

ROI filter: ProSeg processes transcripts inside the ROI, but output centroids may
          slightly exceed the boundary; both pred and GT are filtered to the ROI
          for a fair comparison.

Count / gene-vector GT source:
  transcripts_roi_with_prior.csv.gz (nearest-neighbor pre-assignment from the run script)
  - not a truly independent GT, it is the NN baseline
  - but the best approximation this MERFISH dataset can provide
  - the report notes "GT counts from NN assignment baseline"

assign metric:
  ProSeg prior = NN assignment (built by us), GT is also the NN assignment
  -> partly circular, so the assign metrics are set to NaN

conda activate proseg_eval
python eval_proseg_merfish1.py
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

# ── paths ─────────────────────────────────────────────────────────────────────
PROSEG_DIR   = "yourpath/proseg/output/merfish_1"
CELL_META    = ("yourpath/dataset/merfish_mouse_brain/"
                "datasets_mouse_brain_map_BrainReceptorShowcase"
                "_Slice1_Replicate1_cell_metadata_S1R1.csv")
PRIOR_TX     = "yourpath/proseg/output/merfish_1/transcripts_roi_with_prior.csv.gz"
META_ROI     = "yourpath/proseg/output/merfish_1/cell_metadata_roi.csv"
OUT_PATH     = "yourpath/proseg/value/merfish_1.csv"

ROI_X0, ROI_X1 = 3589.0, 5761.0
ROI_Y0, ROI_Y1 = 2327.0, 4499.0
MATCH_RADIUS   = 10.0   # um; brain cells ~10 um diameter

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print(SEP); print("ProSeg × MERFISH mouse brain evaluation"); print(SEP)
print(f"ROI: x[{ROI_X0},{ROI_X1}]  y[{ROI_Y0},{ROI_Y1}] µm")
print("note: count/vec GT uses the NN pre-assignment baseline (not independent GT; noted in evaluation)")

# ════════════════════════════════════════════════════════════════════════
# 1. read ProSeg cell-metadata, filter to ROI
# ════════════════════════════════════════════════════════════════════════
print("\n[1] read ProSeg cell-metadata ...")
pred_all = pd.read_csv(os.path.join(PROSEG_DIR, "cell-metadata.csv.gz"))
print(f"  all output: {len(pred_all):,} cells")
print(f"  centroid_x: [{pred_all.centroid_x.min():.0f},{pred_all.centroid_x.max():.0f}]")
# ROI filter
pred = pred_all[
    (pred_all.centroid_x >= ROI_X0) & (pred_all.centroid_x < ROI_X1) &
    (pred_all.centroid_y >= ROI_Y0) & (pred_all.centroid_y < ROI_Y1)
].reset_index(drop=True).copy()
print(f"  pred in ROI: {len(pred):,} cells")
pred_xy = pred[["centroid_x","centroid_y"]].values.astype(float)
n_pred  = len(pred)

# ════════════════════════════════════════════════════════════════════════
# 2. read GT (cell_metadata_S1R1.csv, independent source)
# ════════════════════════════════════════════════════════════════════════
print("\n[2] read GT (cell_metadata_S1R1.csv) ...")
meta = pd.read_csv(CELL_META).rename(columns={"Unnamed: 0": "cell_id_orig"})
gt = meta[
    (meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
    (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
].reset_index(drop=True).copy()
gt["gt_idx"] = np.arange(len(gt))   # 0-based index
n_gt    = len(gt)
gt_xy   = gt[["center_x","center_y"]].values.astype(float)
print(f"  GT in ROI: {n_gt:,} cells")
print(f"  GT center_x: [{gt_xy[:,0].min():.0f},{gt_xy[:,0].max():.0f}] µm")

# coordinate range check
px_span = pred_xy[:,0].max() - pred_xy[:,0].min()
gx_span = gt_xy[:,0].max()   - gt_xy[:,0].min()
print(f"  x span: pred={px_span:.0f}µm  GT={gx_span:.0f}µm"
      f"  {'✓' if abs(px_span-gx_span)/max(px_span,gx_span)<0.4 else '⚠ large discrepancy'}")

# ════════════════════════════════════════════════════════════════════════
# 3. read ProSeg transcript-metadata (pred counts + gene vector)
# ════════════════════════════════════════════════════════════════════════
print("\n[3] read ProSeg transcript-metadata ...")
tx_pred = pd.read_csv(os.path.join(PROSEG_DIR, "transcript-metadata.csv.gz"))
print(f"  total transcripts: {len(tx_pred):,}  cols: {list(tx_pred.columns)}")
tx_asgn = tx_pred[tx_pred["background"] == False].copy()
tx_asgn["cell_int"] = tx_asgn["assignment"].astype(int)
pred_cnt_ser  = tx_asgn.groupby("cell_int").size()
pred_gene_grp = tx_asgn.groupby(["cell_int","gene"]).size().unstack(fill_value=0)
print(f"  non-background transcripts: {len(tx_asgn):,}  assignment rate: {len(tx_asgn)/len(tx_pred)*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# 4. read GT count baseline (NN pre-assigned transcripts_roi_with_prior)
# ════════════════════════════════════════════════════════════════════════
print("\n[4] read GT transcript baseline (NN pre-assignment)...")
gt_cnt_ser   = pd.Series(dtype=int)
gt_gene_grp  = None
if os.path.exists(PRIOR_TX):
    tx_prior = pd.read_csv(PRIOR_TX)
    print(f"  prior transcripts: {len(tx_prior):,}  cols: {list(tx_prior.columns)}")
    # cell_metadata_roi.csv has cell_id (1-based sequential) matching GT cells
    if os.path.exists(META_ROI):
        meta_roi = pd.read_csv(META_ROI)
        # meta_roi cell_id is 1-based, matching tx_prior cell_id
        # meta_roi row order matches gt (both are the same cells filtered by ROI)
        # but gt uses cell_id_orig while meta_roi uses sequential cell_id
        # so align via center_x/y
        gt_cnt_ser = tx_prior[tx_prior.cell_id > 0].groupby("cell_id").size()
        gt_gene_grp = (tx_prior[tx_prior.cell_id > 0]
                       .groupby(["cell_id","gene"]).size().unstack(fill_value=0))
        print(f"  GT counts (NN baseline): {len(gt_cnt_ser):,} cells with counts")
    else:
        print("  cell_metadata_roi.csv not found; skip count/vec metrics")
else:
    print(f"  {PRIOR_TX} not found; skip count/vec metrics")

# ════════════════════════════════════════════════════════════════════════
# 5. match
# ════════════════════════════════════════════════════════════════════════
print(f"\n[5] match (radius={MATCH_RADIUS} um)...")
tree = cKDTree(gt_xy)
dists, gi = tree.query(pred_xy, k=1, distance_upper_bound=MATCH_RADIUS)
best = {}
for pi in range(n_pred):
    if dists[pi] >= MATCH_RADIUS: continue
    g = int(gi[pi])
    if g not in best or dists[pi] < best[g][1]:
        best[g] = (pi, dists[pi])
pairs   = [(pi, g, d_) for g, (pi, d_) in best.items()]
n_match = len(pairs)
prec   = n_match / n_pred if n_pred else 0
recall = n_match / n_gt   if n_gt   else 0
f1     = 2*prec*recall/(prec+recall) if prec+recall else 0
shifts = [p[2] for p in pairs]
print(f"  pred {n_pred:,}  GT {n_gt:,}  matched {n_match:,}")
print(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={recall:.4f}")

pi_arr = [p[0] for p in pairs]
gi_arr = [p[1] for p in pairs]

# ════════════════════════════════════════════════════════════════════════
# 6. count metrics (needs GT baseline)
# ════════════════════════════════════════════════════════════════════════
count_pearson = count_spearman = count_mae = count_rmse = np.nan
if len(gt_cnt_ser) > 0 and os.path.exists(META_ROI):
    # align meta_roi cell_id with GT rows (via centroid matching)
    meta_roi = pd.read_csv(META_ROI)
    # meta_roi columns: cell_id, center_x, center_y
    tree_meta = cKDTree(meta_roi[["center_x","center_y"]].values)
    d_gt, idx_gt = tree_meta.query(gt_xy, k=1, distance_upper_bound=2.0)
    gt["meta_cell_id"] = np.where(
        d_gt < 2.0,
        meta_roi.iloc[idx_gt.clip(0,len(meta_roi)-1)]["cell_id"].values,
        -1)

    pred_cell_ids = pred.iloc[pi_arr]["cell"].values
    p_cnt = np.array([pred_cnt_ser.get(int(c), 0) for c in pred_cell_ids], dtype=float)
    gt_meta_ids = gt.iloc[gi_arr]["meta_cell_id"].values
    g_cnt = np.array([gt_cnt_ser.get(int(mid), 0) if mid > 0 else 0
                      for mid in gt_meta_ids], dtype=float)

    valid = (p_cnt > 0) & (g_cnt > 0)
    if valid.sum() >= 3 and p_cnt[valid].std() > 0 and g_cnt[valid].std() > 0:
        count_pearson,  _ = pearsonr(p_cnt[valid],  g_cnt[valid])
        count_spearman, _ = spearmanr(p_cnt[valid], g_cnt[valid])
        count_mae   = float(np.mean(np.abs(p_cnt - g_cnt)))
        count_rmse  = float(np.sqrt(np.mean((p_cnt - g_cnt)**2)))
        print(f"\n[6] counts: pearson={count_pearson:.4f}  "
              f"spearman={count_spearman:.4f}  (valid pairs {valid.sum():,})")
    else:
        print(f"\n[6] counts: too few valid pairs ({valid.sum()}); skip")

# ════════════════════════════════════════════════════════════════════════
# 7. gene vector
# ════════════════════════════════════════════════════════════════════════
vec_cosine = vec_js = vec_pearson = np.nan
if gt_gene_grp is not None and os.path.exists(META_ROI):
    print("\n[7] gene vector ...")
    gene_names_all = np.sort(tx_asgn["gene"].dropna().unique())
    g2i = {g_: i for i, g_ in enumerate(gene_names_all)}
    ng  = len(gene_names_all)
    meta_roi = pd.read_csv(META_ROI)
    tree_meta = cKDTree(meta_roi[["center_x","center_y"]].values)
    d_gt2, idx_gt2 = tree_meta.query(gt_xy, k=1, distance_upper_bound=2.0)
    gt_meta_ids_all = np.where(
        d_gt2 < 2.0,
        meta_roi.iloc[idx_gt2.clip(0,len(meta_roi)-1)]["cell_id"].values, -1)

    np.random.seed(42)
    samp = np.random.choice(len(pairs), min(800, len(pairs)), replace=False)
    cosines, jsd, prs = [], [], []
    for idx in samp:
        pi, gi_val, _ = pairs[idx]
        pcid = int(pred.iloc[pi]["cell"])
        gmid = int(gt_meta_ids_all[gi_val])
        if gmid < 0: continue
        pv = np.zeros(ng)
        if pcid in pred_gene_grp.index:
            for gname, cnt in pred_gene_grp.loc[pcid].items():
                if gname in g2i and cnt > 0: pv[g2i[gname]] = cnt
        gv = np.zeros(ng)
        if gmid in gt_gene_grp.index:
            for gname, cnt in gt_gene_grp.loc[gmid].items():
                if gname in g2i and cnt > 0: gv[g2i[gname]] = cnt
        nn = np.linalg.norm(pv) * np.linalg.norm(gv)
        if nn > 0:
            cosines.append(float(np.dot(pv, gv) / nn))
            pn = pv/pv.sum() if pv.sum() > 0 else pv
            gn = gv/gv.sum() if gv.sum() > 0 else gv
            jsd.append(float(jensenshannon(pn, gn)))
            if pv.std() > 0 and gv.std() > 0:
                prs.append(float(pearsonr(pv, gv)[0]))
    vec_cosine  = float(np.mean(cosines)) if cosines else np.nan
    vec_js      = float(np.mean(jsd))     if jsd     else np.nan
    vec_pearson = float(np.mean(prs))     if prs     else np.nan
    print(f"  cosine={vec_cosine:.4f}  JS={vec_js:.4f}  pearson={vec_pearson:.4f}")

# ════════════════════════════════════════════════════════════════════════
# summary
# ════════════════════════════════════════════════════════════════════════
metrics = dict(
    gt_cells        = n_gt,
    pred_cells      = n_pred,
    cell_ratio      = round(n_pred/n_gt, 4),
    matched         = n_match,
    precision       = round(prec, 4),
    recall          = round(recall, 4),
    f1              = round(f1, 4),
    loc_mean_um     = round(float(np.mean(shifts)),           4) if shifts else np.nan,
    loc_median_um   = round(float(np.median(shifts)),         4) if shifts else np.nan,
    loc_p95_um      = round(float(np.percentile(shifts, 95)), 4) if shifts else np.nan,
    count_pearson   = round(count_pearson,  4) if not np.isnan(count_pearson)  else np.nan,
    count_spearman  = round(count_spearman, 4) if not np.isnan(count_spearman) else np.nan,
    count_mae       = round(count_mae,  2)     if not np.isnan(count_mae)      else np.nan,
    count_rmse      = round(count_rmse, 2)     if not np.isnan(count_rmse)     else np.nan,
    vec_cosine      = round(vec_cosine,  4) if not np.isnan(vec_cosine)  else np.nan,
    vec_js_dist     = round(vec_js,      4) if not np.isnan(vec_js)      else np.nan,
    vec_pearson     = round(vec_pearson, 4) if not np.isnan(vec_pearson) else np.nan,
    assign_overlap  = np.nan,   # prior = GT NN baseline (same source); skip
    assign_accuracy = np.nan,   # same as above
)

print(f"\n{SEP}\nResult (ProSeg × MERFISH mouse brain)\n{SEP}")
for k, v in metrics.items():
    print(f"  {k:22} {v}")
pd.DataFrame([metrics]).to_csv(OUT_PATH, index=False)
print(f"\n  written -> {OUT_PATH}")
print("  note: count/vec GT is the NN assignment baseline; assign skipped because prior = GT; detection/localization metrics are valid")
print(SEP)