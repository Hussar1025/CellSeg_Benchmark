#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_proseg_merfish1.py  —— ProSeg × MERFISH 小鼠脑评估（19 指标）
=========================================================================
ProSeg v3 输出列名（与 xenium_3/stereo_1 相同）：
  cell-metadata:       cell(ID,int), centroid_x, centroid_y（µm）
  transcript-metadata: gene, assignment(float→int), background(bool), x, y（µm）

GT：cell_metadata_S1R1.csv（center_x, center_y，µm）
    取 ROI 内 [3589,5761] × [2327,4499] µm 的细胞

ROI 过滤：ProSeg 处理的是 ROI 内转录本，但输出质心可能略超边界，
          pred 和 GT 均需过滤到 ROI 内确保公平比较。

计数/基因向量 GT 来源：
  transcripts_roi_with_prior.csv.gz（run 脚本生成的最近邻预分配）
  - 不是真正独立的 GT，是 NN 基线
  - 但这是 MERFISH 数据集能提供的最好近似
  - 评估报告中标注"GT counts from NN assignment baseline"

assign 指标：
  ProSeg 先验 = NN assignment（我们自建的），GT 也是 NN assignment
  → 部分循环，assign 指标标记 NaN

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

# ── 路径 ─────────────────────────────────────────────────────────────────────
PROSEG_DIR   = "/data/qiuyijia/proseg/output/merfish_1"
CELL_META    = ("/data/qiuyijia/dataset/merfish_mouse_brain/"
                "datasets_mouse_brain_map_BrainReceptorShowcase"
                "_Slice1_Replicate1_cell_metadata_S1R1.csv")
PRIOR_TX     = "/data/qiuyijia/proseg/output/merfish_1/transcripts_roi_with_prior.csv.gz"
META_ROI     = "/data/qiuyijia/proseg/output/merfish_1/cell_metadata_roi.csv"
OUT_PATH     = "/data/qiuyijia/proseg/value/merfish_1.csv"

ROI_X0, ROI_X1 = 3589.0, 5761.0
ROI_Y0, ROI_Y1 = 2327.0, 4499.0
MATCH_RADIUS   = 10.0   # µm；脑细胞 ~10µm 直径

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print(SEP); print("ProSeg × MERFISH 小鼠脑 评估"); print(SEP)
print(f"ROI: x[{ROI_X0},{ROI_X1}]  y[{ROI_Y0},{ROI_Y1}] µm")
print("注：count/vec GT 使用 NN 预分配基线（非独立 GT，评估时备注）")

# ════════════════════════════════════════════════════════════════════════
# 1. 读 ProSeg cell-metadata，过滤到 ROI
# ════════════════════════════════════════════════════════════════════════
print("\n[1] 读 ProSeg cell-metadata ...")
pred_all = pd.read_csv(os.path.join(PROSEG_DIR, "cell-metadata.csv.gz"))
print(f"  全部输出: {len(pred_all):,} 细胞")
print(f"  centroid_x: [{pred_all.centroid_x.min():.0f},{pred_all.centroid_x.max():.0f}]")
# ROI 过滤
pred = pred_all[
    (pred_all.centroid_x >= ROI_X0) & (pred_all.centroid_x < ROI_X1) &
    (pred_all.centroid_y >= ROI_Y0) & (pred_all.centroid_y < ROI_Y1)
].reset_index(drop=True).copy()
print(f"  ROI 内 pred: {len(pred):,} 细胞")
pred_xy = pred[["centroid_x","centroid_y"]].values.astype(float)
n_pred  = len(pred)

# ════════════════════════════════════════════════════════════════════════
# 2. 读 GT（cell_metadata_S1R1.csv，独立来源）
# ════════════════════════════════════════════════════════════════════════
print("\n[2] 读 GT (cell_metadata_S1R1.csv) ...")
meta = pd.read_csv(CELL_META).rename(columns={"Unnamed: 0": "cell_id_orig"})
gt = meta[
    (meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
    (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
].reset_index(drop=True).copy()
gt["gt_idx"] = np.arange(len(gt))   # 0-based 索引
n_gt    = len(gt)
gt_xy   = gt[["center_x","center_y"]].values.astype(float)
print(f"  ROI 内 GT: {n_gt:,} 细胞")
print(f"  GT center_x: [{gt_xy[:,0].min():.0f},{gt_xy[:,0].max():.0f}] µm")

# 坐标范围检查
px_span = pred_xy[:,0].max() - pred_xy[:,0].min()
gx_span = gt_xy[:,0].max()   - gt_xy[:,0].min()
print(f"  x span: pred={px_span:.0f}µm  GT={gx_span:.0f}µm"
      f"  {'✓' if abs(px_span-gx_span)/max(px_span,gx_span)<0.4 else '⚠ 差异较大'}")

# ════════════════════════════════════════════════════════════════════════
# 3. 读 ProSeg transcript-metadata（pred 计数 + 基因向量）
# ════════════════════════════════════════════════════════════════════════
print("\n[3] 读 ProSeg transcript-metadata ...")
tx_pred = pd.read_csv(os.path.join(PROSEG_DIR, "transcript-metadata.csv.gz"))
print(f"  总转录本: {len(tx_pred):,}  列: {list(tx_pred.columns)}")
tx_asgn = tx_pred[tx_pred["background"] == False].copy()
tx_asgn["cell_int"] = tx_asgn["assignment"].astype(int)
pred_cnt_ser  = tx_asgn.groupby("cell_int").size()
pred_gene_grp = tx_asgn.groupby(["cell_int","gene"]).size().unstack(fill_value=0)
print(f"  非背景转录本: {len(tx_asgn):,}  分配率: {len(tx_asgn)/len(tx_pred)*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════
# 4. 读 GT 计数基线（NN 预分配的 transcripts_roi_with_prior）
# ════════════════════════════════════════════════════════════════════════
print("\n[4] 读 GT 转录本基线（NN 预分配）...")
gt_cnt_ser   = pd.Series(dtype=int)
gt_gene_grp  = None
if os.path.exists(PRIOR_TX):
    tx_prior = pd.read_csv(PRIOR_TX)
    print(f"  prior 转录本: {len(tx_prior):,}  列: {list(tx_prior.columns)}")
    # cell_metadata_roi.csv 有 cell_id（1-based sequential）对应 GT 细胞
    if os.path.exists(META_ROI):
        meta_roi = pd.read_csv(META_ROI)
        # meta_roi 的 cell_id 是 1-based，对应 tx_prior 的 cell_id
        # meta_roi 的行顺序对应 gt（两者都按 ROI 过滤后的相同细胞）
        # 但 gt 用的是 cell_id_orig，meta_roi 用的是 sequential cell_id
        # 需要通过 center_x/y 做对齐
        gt_cnt_ser = tx_prior[tx_prior.cell_id > 0].groupby("cell_id").size()
        gt_gene_grp = (tx_prior[tx_prior.cell_id > 0]
                       .groupby(["cell_id","gene"]).size().unstack(fill_value=0))
        print(f"  GT 计数（NN 基线）: {len(gt_cnt_ser):,} 细胞有计数")
    else:
        print("  cell_metadata_roi.csv 不存在，count/vec 指标跳过")
else:
    print(f"  {PRIOR_TX} 不存在，count/vec 指标跳过")

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
prec   = n_match / n_pred if n_pred else 0
recall = n_match / n_gt   if n_gt   else 0
f1     = 2*prec*recall/(prec+recall) if prec+recall else 0
shifts = [p[2] for p in pairs]
print(f"  pred {n_pred:,}  GT {n_gt:,}  matched {n_match:,}")
print(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={recall:.4f}")

pi_arr = [p[0] for p in pairs]
gi_arr = [p[1] for p in pairs]

# ════════════════════════════════════════════════════════════════════════
# 6. 计数指标（需要 GT 基线）
# ════════════════════════════════════════════════════════════════════════
count_pearson = count_spearman = count_mae = count_rmse = np.nan
if len(gt_cnt_ser) > 0 and os.path.exists(META_ROI):
    # 将 meta_roi 的 cell_id 与 GT 的行对齐（通过质心匹配）
    meta_roi = pd.read_csv(META_ROI)
    # meta_roi 列: cell_id, center_x, center_y
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
        print(f"\n[6] 计数: pearson={count_pearson:.4f}  "
              f"spearman={count_spearman:.4f}  (有效对 {valid.sum():,})")
    else:
        print(f"\n[6] 计数: 有效对太少（{valid.sum()}），跳过")

# ════════════════════════════════════════════════════════════════════════
# 7. 基因向量
# ════════════════════════════════════════════════════════════════════════
vec_cosine = vec_js = vec_pearson = np.nan
if gt_gene_grp is not None and os.path.exists(META_ROI):
    print("\n[7] 基因向量 ...")
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
    count_pearson   = round(count_pearson,  4) if not np.isnan(count_pearson)  else np.nan,
    count_spearman  = round(count_spearman, 4) if not np.isnan(count_spearman) else np.nan,
    count_mae       = round(count_mae,  2)     if not np.isnan(count_mae)      else np.nan,
    count_rmse      = round(count_rmse, 2)     if not np.isnan(count_rmse)     else np.nan,
    vec_cosine      = round(vec_cosine,  4) if not np.isnan(vec_cosine)  else np.nan,
    vec_js_dist     = round(vec_js,      4) if not np.isnan(vec_js)      else np.nan,
    vec_pearson     = round(vec_pearson, 4) if not np.isnan(vec_pearson) else np.nan,
    assign_overlap  = np.nan,   # 先验=GT NN基线，同源，跳过
    assign_accuracy = np.nan,   # 同上
)

print(f"\n{SEP}\n结果 (ProSeg × MERFISH 小鼠脑)\n{SEP}")
for k, v in metrics.items():
    print(f"  {k:22} {v}")
pd.DataFrame([metrics]).to_csv(OUT_PATH, index=False)
print(f"\n  已写 → {OUT_PATH}")
print("  注：count/vec GT 为 NN 分配基线；assign 因先验=GT 跳过；检测/定位指标有效")
print(SEP)