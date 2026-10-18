#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_proseg_xenium3.py  —— ProSeg × Xenium 淋巴结 ROI 评估（19 指标）
=========================================================================
ProSeg v3 输出列名（已确认）：
  cell-metadata:       cell(ID), centroid_x, centroid_y (µm)
  transcript-metadata: gene, assignment(float→int=cell_id), background(bool)

坐标：pred centroid µm = GT x_centroid/y_centroid µm，范围完全一致，无需转换
      已验证：ProSeg x[3983.4,4319.0] vs GT x[3982.7,4317.5] ✓

match_radius = 6.9 µm（= 细胞直径，与 UCS/BOMS 淋巴结评估一致）

conda activate proseg_eval
python eval_proseg_xenium3.py
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

PROSEG_DIR   = "/data/qiuyijia/proseg/output/xenium_3"
DATA_DIR     = "/data/qiuyijia/dataset/xenium_roi_crop"
OUT_PATH     = "/data/qiuyijia/proseg/value/xenium_3.csv"
MATCH_RADIUS = 6.9   # µm
QV_MIN       = 20.0

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print(SEP); print("ProSeg × Xenium 淋巴结 ROI 评估"); print(SEP)

# ════════════════════════════════════════════════════════════════════════
# 1. 读 ProSeg cell-metadata
# ════════════════════════════════════════════════════════════════════════
print("\n[1] 读 ProSeg cell-metadata ...")
pred = pd.read_csv(os.path.join(PROSEG_DIR, "cell-metadata.csv.gz"))
print(f"  细胞数: {len(pred):,}")
print(f"  centroid_x: [{pred.centroid_x.min():.1f},{pred.centroid_x.max():.1f}] µm")
print(f"  centroid_y: [{pred.centroid_y.min():.1f},{pred.centroid_y.max():.1f}] µm")
pred_xy = pred[["centroid_x","centroid_y"]].values.astype(float)
n_pred  = len(pred)

# ════════════════════════════════════════════════════════════════════════
# 2. 读 GT
# ════════════════════════════════════════════════════════════════════════
print("\n[2] 读 GT (cells.parquet) ...")
gt = pd.read_parquet(os.path.join(DATA_DIR, "cells.parquet"),
                     columns=["cell_id","x_centroid","y_centroid"])
print(f"  GT 细胞: {len(gt):,}")
print(f"  GT x: [{gt.x_centroid.min():.1f},{gt.x_centroid.max():.1f}] µm")
gt_xy = gt[["x_centroid","y_centroid"]].values.astype(float)

# ════════════════════════════════════════════════════════════════════════
# 3. 读 GT 转录本（计数 + 基因向量）
# ════════════════════════════════════════════════════════════════════════
print("\n[3] 读 GT 转录本 ...")
tx_gt = pd.read_parquet(os.path.join(DATA_DIR, "transcripts.parquet"),
                        columns=["cell_id","feature_name","qv","is_gene"])
tx_gt = tx_gt[(tx_gt.is_gene == True) & (tx_gt.qv >= QV_MIN)]
gt_cnt_ser = tx_gt.groupby("cell_id").size()
gt = gt.copy()
gt["gt_count"] = gt["cell_id"].map(gt_cnt_ser).fillna(0).astype(int)
gt_gene_grp = tx_gt[tx_gt.cell_id.isin(gt.cell_id)].groupby(
    ["cell_id","feature_name"]).size().unstack(fill_value=0)
print(f"  GT 转录本: {len(tx_gt):,}  有计数 GT 细胞: {(gt.gt_count>0).sum():,}")

# ════════════════════════════════════════════════════════════════════════
# 4. 读 ProSeg transcript-metadata（pred 计数 + 基因向量）
# ════════════════════════════════════════════════════════════════════════
print("\n[4] 读 ProSeg transcript-metadata ...")
tx_pred = pd.read_csv(os.path.join(PROSEG_DIR, "transcript-metadata.csv.gz"))
print(f"  转录本: {len(tx_pred):,}  列: {list(tx_pred.columns)}")
# assignment 是 float，转 int 与 pred["cell"] 对齐
tx_asgn = tx_pred[tx_pred["background"] == False].copy()
tx_asgn["cell_int"] = tx_asgn["assignment"].astype(int)
pred_cnt_ser  = tx_asgn.groupby("cell_int").size()
pred_gene_grp = tx_asgn.groupby(["cell_int","gene"]).size().unstack(fill_value=0)
print(f"  非背景转录本: {len(tx_asgn):,}  分配率: {len(tx_asgn)/len(tx_pred)*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# 5. 匹配
# ════════════════════════════════════════════════════════════════════════
print(f"\n[5] 匹配（radius={MATCH_RADIUS}µm）...")
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
n_gt    = len(gt)
prec   = n_match / n_pred if n_pred else 0
recall = n_match / n_gt   if n_gt   else 0
f1     = 2*prec*recall/(prec+recall) if prec+recall else 0
shifts = [p[2] for p in pairs]
print(f"  pred {n_pred:,}  GT {n_gt:,}  matched {n_match:,}")
print(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={recall:.4f}")

pi_arr = [p[0] for p in pairs]
gi_arr = [p[1] for p in pairs]

# ════════════════════════════════════════════════════════════════════════
# 6. 计数指标
# ════════════════════════════════════════════════════════════════════════
pred_cell_ids = pred.iloc[pi_arr]["cell"].values  # int
p_cnt = np.array([pred_cnt_ser.get(int(cid), 0) for cid in pred_cell_ids], dtype=float)
g_cnt = gt.iloc[gi_arr]["gt_count"].values.astype(float)

if len(p_cnt) >= 3 and p_cnt.std() > 0 and g_cnt.std() > 0:
    count_pearson,  _ = pearsonr(p_cnt, g_cnt)
    count_spearman, _ = spearmanr(p_cnt, g_cnt)
    count_mae   = float(np.mean(np.abs(p_cnt - g_cnt)))
    count_rmse  = float(np.sqrt(np.mean((p_cnt - g_cnt)**2)))
else:
    count_pearson = count_spearman = count_mae = count_rmse = np.nan
print(f"\n[6] 计数: pearson={count_pearson:.4f}  spearman={count_spearman:.4f}")

# ════════════════════════════════════════════════════════════════════════
# 7. 基因向量（抽样 800 对）
# ════════════════════════════════════════════════════════════════════════
print("\n[7] 基因向量 ...")
gene_names_all = np.sort(tx_asgn["gene"].dropna().unique())
g2i = {g: i for i, g in enumerate(gene_names_all)}
ng  = len(gene_names_all)

np.random.seed(42)
samp = np.random.choice(len(pairs), min(800, len(pairs)), replace=False)
cosines, jsd, prs = [], [], []
for idx in samp:
    pi, gi_val, _ = pairs[idx]
    pcid = int(pred.iloc[pi]["cell"])
    gcid = gt.iloc[gi_val]["cell_id"]
    pv = np.zeros(ng)
    if pcid in pred_gene_grp.index:
        for gname, cnt in pred_gene_grp.loc[pcid].items():
            if gname in g2i and cnt > 0: pv[g2i[gname]] = cnt
    gv = np.zeros(ng)
    if gcid in gt_gene_grp.index:
        for gname, cnt in gt_gene_grp.loc[gcid].items():
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
# 8. 分配指标
# ════════════════════════════════════════════════════════════════════════
print("\n[8] 分配指标 ...")
# ProSeg 转录本坐标（revised position：x,y 列）
tx_xy = tx_pred[["x","y"]].values.astype(float)
tree_gt = cKDTree(gt_xy)
d_tx, gi_tx = tree_gt.query(tx_xy, k=1, distance_upper_bound=MATCH_RADIUS)
gt_cell_per_tx = np.where(
    d_tx < MATCH_RADIUS,
    gt.iloc[gi_tx.clip(0, len(gt)-1)]["cell_id"].values, -1)

pred_assign_all = tx_pred["assignment"].fillna(-1).astype(int).values

ov_list, ac_list = [], []
for pi, gi_val, _ in pairs:
    pcid = int(pred.iloc[pi]["cell"])
    gcid = gt.iloc[gi_val]["cell_id"]
    pm   = (pred_assign_all == pcid) & (~tx_pred["background"].values)
    if pm.sum() == 0: continue
    gta = gt_cell_per_tx[pm]
    ov_list.append(float((gta == gcid).mean()))
    if len(gta) > 0:
        maj = pd.Series(gta).value_counts().index[0]
        ac_list.append(float((gta == maj).mean()))

assign_overlap  = float(np.mean(ov_list)) if ov_list else np.nan
assign_accuracy = float(np.mean(ac_list))  if ac_list  else np.nan
print(f"  overlap={assign_overlap:.4f}  accuracy={assign_accuracy:.4f}")

# ════════════════════════════════════════════════════════════════════════
# 汇总
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
    count_pearson   = round(count_pearson,  4),
    count_spearman  = round(count_spearman, 4),
    count_mae       = round(count_mae,  2),
    count_rmse      = round(count_rmse, 2),
    vec_cosine      = round(vec_cosine,  4),
    vec_js_dist     = round(vec_js,      4),
    vec_pearson     = round(vec_pearson, 4),
    assign_overlap  = round(assign_overlap,  4),
    assign_accuracy = round(assign_accuracy, 4),
)
print(f"\n{SEP}\n结果 (ProSeg × Xenium 淋巴结)\n{SEP}")
for k, v in metrics.items():
    print(f"  {k:22} {v}")
pd.DataFrame([metrics]).to_csv(OUT_PATH, index=False)
print(f"\n  已写 → {OUT_PATH}")