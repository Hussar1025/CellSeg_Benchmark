#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_boms_xenium_breast.py  —— BOMS × Xenium 乳腺评估（19 指标）
=========================================================================
GT：cells.parquet，ROI 过滤后实际 8,809 细胞
ROI 与 ucs_xenium_breast.py 完全一致（从图像中心计算，非硬编码）
pred：boms_xenium_breast.npz，280 基因，全量（无 HVG 截断）
坐标：pred(µm) 和 GT(µm) 直接对齐，无需转换

conda activate ucs
python eval_boms_xenium_breast.py
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

# ── 路径 ─────────────────────────────────────────────────────────────────────
NPZ_PATH  = "/data/qiuyijia/boms_xenium_breast/boms_xenium_breast.npz"
DATA_DIR  = "/data/qiuyijia/dataset/xenium_breast"
OUT_PATH  = "/data/qiuyijia/eval_results/boms/xenium_breast.csv"

# ── ROI（与 ucs_xenium_breast.py 完全一致，从图像尺寸计算）──────────────────
PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE   # 5736.86 µm
CY_UM = 27420 / 2 * PIXEL_SIZE   # 2913.38 µm
HALF  = 5000  * PIXEL_SIZE / 2   # 531.25 µm
ROI_X0, ROI_X1 = CX_UM - HALF, CX_UM + HALF
ROI_Y0, ROI_Y1 = CY_UM - HALF, CY_UM + HALF

MATCH_RADIUS = 20.0  # µm；乳腺细胞直径约 10-30µm
QV_MIN = 20.0

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print(SEP); print("BOMS × Xenium 乳腺 评估"); print(SEP)
print(f"ROI: x[{ROI_X0:.1f},{ROI_X1:.1f}]  y[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm")

# ════════════════════════════════════════════════════════════════════════
# 1. 读 BOMS npz
# ════════════════════════════════════════════════════════════════════════
print("\n[1] 读 BOMS npz ...")
d          = np.load(NPZ_PATH, allow_pickle=True)
seg        = d["seg"].astype(np.int64)      # per-tx cell 分配（1-based，0=背景）
tx_x       = d["x"].astype(np.float64)     # 转录本 x µm
tx_y       = d["y"].astype(np.float64)     # 转录本 y µm
gene_idx   = d["gene_idx"].astype(np.int64)
gene_names = d["gene_names"]               # shape=(n_gene,)
cell_loc   = d["cell_loc"]                 # shape=(n_pred,2)，µm，[x,y]
n_pred = cell_loc.shape[0]
n_gene = len(gene_names)
print(f"  pred 细胞: {n_pred:,}  转录本: {len(seg):,}  基因: {n_gene}")
print(f"  cell_loc x: [{cell_loc[:,0].min():.1f},{cell_loc[:,0].max():.1f}]  "
      f"y: [{cell_loc[:,1].min():.1f},{cell_loc[:,1].max():.1f}] µm")
pred_xy = cell_loc  # (n_pred,2)，第 0 列=x，第 1 列=y

# pred 每细胞转录本计数（1-based cell_id）
pred_counts = np.bincount(seg[seg > 0])  # pred_counts[i] = cell_id=i 的计数

# ════════════════════════════════════════════════════════════════════════
# 2. 读 GT（cells.parquet + ROI 过滤）
# ════════════════════════════════════════════════════════════════════════
print("\n[2] 读 GT ...")
gt = pd.read_parquet(os.path.join(DATA_DIR, "cells.parquet"),
                     columns=["cell_id","x_centroid","y_centroid"])
gt_all = len(gt)
gt = gt[(gt.x_centroid >= ROI_X0) & (gt.x_centroid < ROI_X1) &
        (gt.y_centroid >= ROI_Y0) & (gt.y_centroid < ROI_Y1)].reset_index(drop=True)
print(f"  全图 {gt_all:,}  ROI 内 {len(gt):,} 细胞")
gt_xy = gt[["x_centroid","y_centroid"]].values.astype(float)

# ════════════════════════════════════════════════════════════════════════
# 3. 读 GT 转录本（计数 + 基因向量）
# ════════════════════════════════════════════════════════════════════════
print("\n[3] 读 GT 转录本 ...")
tx_gt = pd.read_parquet(
    os.path.join(DATA_DIR, "transcripts.parquet"),
    columns=["cell_id","feature_name","qv","is_gene","x_location","y_location"])
tx_gt = tx_gt[
    (tx_gt.is_gene == True) & (tx_gt.qv >= QV_MIN) &
    (tx_gt.x_location >= ROI_X0) & (tx_gt.x_location < ROI_X1) &
    (tx_gt.y_location >= ROI_Y0) & (tx_gt.y_location < ROI_Y1)]
print(f"  ROI 内 GT 转录本: {len(tx_gt):,}")
gt_cnt_ser = tx_gt.groupby("cell_id").size()
gt = gt.copy()
gt["gt_count"] = gt["cell_id"].map(gt_cnt_ser).fillna(0).astype(int)
gt_gene_grp = (tx_gt[tx_gt.cell_id.isin(gt.cell_id)]
               .groupby(["cell_id","feature_name"]).size().unstack(fill_value=0))

# ════════════════════════════════════════════════════════════════════════
# 4. 匹配
# ════════════════════════════════════════════════════════════════════════
print(f"\n[4] 匹配（radius={MATCH_RADIUS}µm）...")
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
# 5. 计数指标
# ════════════════════════════════════════════════════════════════════════
# pred cell_id = pi+1 (1-based)
p_cnt = np.array([pred_counts[pi+1] if pi+1 < len(pred_counts) else 0
                  for pi in pi_arr], dtype=float)
g_cnt = gt.iloc[gi_arr]["gt_count"].values.astype(float)

if len(p_cnt) >= 3 and p_cnt.std() > 0 and g_cnt.std() > 0:
    count_pearson,  _ = pearsonr(p_cnt, g_cnt)
    count_spearman, _ = spearmanr(p_cnt, g_cnt)
    count_mae   = float(np.mean(np.abs(p_cnt - g_cnt)))
    count_rmse  = float(np.sqrt(np.mean((p_cnt - g_cnt)**2)))
else:
    count_pearson = count_spearman = count_mae = count_rmse = np.nan
print(f"\n[5] 计数: pearson={count_pearson:.4f}  spearman={count_spearman:.4f}")

# ════════════════════════════════════════════════════════════════════════
# 6. 基因向量（pred vs GT，280 基因全集）
# ════════════════════════════════════════════════════════════════════════
print("\n[6] 基因向量（抽样 800 对）...")
gene2i = {g: i for i, g in enumerate(gene_names)}
np.random.seed(42)
samp = np.random.choice(len(pairs), min(800, len(pairs)), replace=False)
cosines, jsd, prs = [], [], []
for idx in samp:
    pi, gi_val, _ = pairs[idx]
    # pred 向量
    mask = (seg == pi + 1)
    pv = np.zeros(n_gene)
    np.add.at(pv, gene_idx[mask], 1)
    # GT 向量
    gc = gt.iloc[gi_val]["cell_id"]
    gv = np.zeros(n_gene)
    if gc in gt_gene_grp.index:
        row = gt_gene_grp.loc[gc]
        for gname in row.index:
            if gname in gene2i and row[gname] > 0:
                gv[gene2i[gname]] = row[gname]
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
# 7. 分配指标
# ════════════════════════════════════════════════════════════════════════
print("\n[7] 分配指标 ...")
tree_gt = cKDTree(gt_xy)
d_tx, gi_tx = tree_gt.query(
    np.stack([tx_x, tx_y], 1), k=1, distance_upper_bound=MATCH_RADIUS)
gt_cell_per_tx = np.where(
    d_tx < MATCH_RADIUS,
    gt.iloc[gi_tx.clip(0, len(gt)-1)]["cell_id"].values, -1)

ov_list, ac_list = [], []
for pi, gi_val, _ in pairs:
    pm = (seg == pi + 1)
    if pm.sum() == 0: continue
    gc = gt.iloc[gi_val]["cell_id"]
    gta = gt_cell_per_tx[pm]
    ov_list.append(float((gta == gc).mean()))
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
print(f"\n{SEP}\n结果 (BOMS × Xenium 乳腺)\n{SEP}")
for k, v in metrics.items():
    print(f"  {k:22} {v}")
pd.DataFrame([metrics]).to_csv(OUT_PATH, index=False)
print(f"\n  已写 → {OUT_PATH}")