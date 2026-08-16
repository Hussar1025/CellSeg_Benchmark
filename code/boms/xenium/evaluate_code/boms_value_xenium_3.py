#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOMS 评估 —— 19 指标（与 UCS 同框架，适配 BOMS 的输出格式）
=========================================================================
BOMS 输出和 UCS 不同：
  UCS  → segmentation_mask.tif（像素 label 图）
  BOMS → seg（每条转录本的细胞标签）+ cell_loc（质心）+ count_mat（细胞×基因）

所以 pred 归属直接来自 seg，不需要"转录本落在哪个像素"这一步。
其余 19 指标定义与 eval_ucs_xenium.py 完全一致，便于横向比较。

坑位对齐：
  * BOMS 用了 top-N HVG 子集 → GT 基因向量也只在这 N 个基因上比，否则维度不一致
  * valid_pairs 仍单独打印（防坐标/匹配错位虚高）
  * BOMS 坐标就是 µm（和 GT 同系），无需 bin 换算
  * assign_accuracy 有效（GT cell_id 独立于 BOMS）

用法
-----
python eval_boms_xenium.py \
    --npz      /data/qiuyijia/boms_xenium_roi/boms_xenium_roi.npz \
    --data-dir /data/qiuyijia/dataset/xenium_roi_crop
"""

import os, json, glob, argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon, cdist
from scipy.optimize import linear_sum_assignment

SEP = "=" * 72


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--method", default="BOMS")
    p.add_argument("--dataset", default="xenium_lymph")
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--match-dist", type=float, default=None)
    return p.parse_args()


def find_one(base, pats, label, required=True):
    for pat in pats:
        h = sorted(glob.glob(os.path.join(base, "**", pat), recursive=True))
        if h:
            h.sort(key=len); return h[0]
    if required:
        raise FileNotFoundError(f"{label}: {pats}")
    return None


def read_table(path, columns=None):
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def match_cells(gt_xy, pred_xy, max_dist):
    n_gt, n_pred = len(gt_xy), len(pred_xy)
    if n_gt == 0 or n_pred == 0:
        return []
    if max(n_gt, n_pred) > 2000:
        tree = cKDTree(pred_xy)
        dist, idx = tree.query(gt_xy, k=1)
        order = np.argsort(dist)
        used, pairs = set(), []
        for gi in order:
            if dist[gi] > max_dist:
                continue
            pi = idx[gi]
            if pi in used:
                dd, ii = tree.query(gt_xy[gi], k=min(5, n_pred))
                ok = False
                for d2, i2 in zip(np.atleast_1d(dd), np.atleast_1d(ii)):
                    if i2 not in used and d2 <= max_dist:
                        pi, ok = i2, True; dist[gi] = d2; break
                if not ok:
                    continue
            used.add(pi)
            pairs.append((gi, pi, float(dist[gi])))
        return pairs
    D = cdist(gt_xy, pred_xy)
    D[D > max_dist] = 1e6
    ri, ci = linear_sum_assignment(D)
    return [(int(r), int(c), float(D[r, c]))
            for r, c in zip(ri, ci) if D[r, c] < 1e6]


# ══════════════════════════════════════════════════════════════════════════════
def evaluate(cfg):
    print(SEP); print(f"BOMS 评估  {cfg.dataset}"); print(SEP)

    d = np.load(cfg.npz, allow_pickle=True)
    seg      = d["seg"]                 # (N,) 每条转录本的细胞标签
    tx_x     = d["x"]; tx_y = d["y"]
    gene_idx = d["gene_idx"]
    gene_names = d["gene_names"]
    cell_loc = d["cell_loc"]            # (n_cell, 2) µm
    n_gene = len(gene_names)
    hvg_set = set(gene_names.tolist())
    print(f"  BOMS: {len(seg):,} 转录本  "
          f"{int(seg.max()) if len(cell_loc)==0 else len(cell_loc):,} 细胞  "
          f"{n_gene} 基因(HVG)  h_s={float(d['h_s'])}")

    base = cfg.data_dir
    f_tx    = find_one(base, ["transcripts.parquet"], "转录本")
    f_cells = find_one(base, ["cells.parquet", "cells.csv"], "GT 细胞")
    f_roi   = find_one(base, ["experiment_roi.json"], "ROI", required=False)

    # GT 转录本归属（cell_id）—— 只在 HVG 上，和 BOMS 对齐
    gt_tx = read_table(f_tx, columns=["x_location", "y_location",
                                      "feature_name", "qv", "is_gene", "cell_id"])
    gt_tx = gt_tx[(gt_tx.is_gene == True) & (gt_tx.qv >= cfg.qv_min)]
    gt_tx = gt_tx[gt_tx.feature_name.isin(hvg_set)].reset_index(drop=True)
    cid = gt_tx["cell_id"].astype(str)
    gt_tx["gt_cell"] = cid.where(cid != "UNASSIGNED", "0")
    g2i = {g: i for i, g in enumerate(gene_names)}
    gt_tx["gi"] = gt_tx.feature_name.map(g2i).values

    # GT 细胞质心
    cells = read_table(f_cells)
    cells["cell_id"] = cells["cell_id"].astype(str)
    diam = 2*np.sqrt(cells["cell_area"].median()/np.pi) if \
        "cell_area" in cells.columns else 7.0
    max_dist = cfg.match_dist if cfg.match_dist else diam
    print(f"  GT {len(cells):,} 细胞  直径 {diam:.1f}µm  匹配阈值 {max_dist:.1f}µm")

    # pred 细胞质心（BOMS 直接给）
    n_cell = len(cell_loc) if len(cell_loc) else int(seg.max())
    if len(cell_loc) == 0:
        # 从 seg 现算质心
        cell_loc = np.zeros((n_cell, 2))
        for c in range(1, n_cell+1):
            m = seg == c
            if m.any():
                cell_loc[c-1] = [tx_x[m].mean(), tx_y[m].mean()]

    gt_xy = cells[["x_centroid", "y_centroid"]].values
    print(f"\n  匹配 GT {len(gt_xy):,} × pred {len(cell_loc):,} ...")
    pairs = match_cells(gt_xy, cell_loc, max_dist)
    n_match = len(pairs)

    precision = n_match / max(len(cell_loc), 1)
    recall    = n_match / max(len(gt_xy), 1)
    f1 = 2*precision*recall / max(precision+recall, 1e-9)

    dists = np.array([p[2] for p in pairs]) if n_match else np.array([np.nan])
    loc_mean, loc_med, loc_p95 = (float(np.mean(dists)), float(np.median(dists)),
                                  float(np.percentile(dists, 95))) if n_match \
                                 else (np.nan, np.nan, np.nan)

    # ── 每个匹配细胞的基因向量 ──────────────────────────────────────────────
    # pred: 从 seg 聚合
    pred_df = pd.DataFrame({"cell": seg, "gi": gene_idx})
    pred_df = pred_df[pred_df.cell > 0]
    pred_grp = pred_df.groupby(["cell", "gi"]).size()
    # GT: 从 cell_id 聚合
    gt_df = gt_tx[gt_tx.gt_cell != "0"]
    gt_grp = gt_df.groupby(["gt_cell", "gi"]).size()

    def vec(grp, key):
        v = np.zeros(n_gene)
        if key in grp.index.get_level_values(0):
            s = grp.loc[key]; v[s.index.values] = s.values
        return v

    gt_ids = cells["cell_id"].values
    gt_counts, pred_counts, cos_l, js_l = [], [], [], []
    valid_vec = 0
    for gi, pi, _ in pairs:
        vg = vec(gt_grp, gt_ids[gi])
        vp = vec(pred_grp, pi+1)                 # seg 是 1-based
        cg, cp = vg.sum(), vp.sum()
        gt_counts.append(cg); pred_counts.append(cp)
        if cg > 0 and cp > 0:
            valid_vec += 1
            den = np.linalg.norm(vg)*np.linalg.norm(vp)
            cos_l.append(float(vg@vp/den) if den > 0 else 0.0)
            js_l.append(float(jensenshannon(vg/cg, vp/cp, base=2)))

    gt_counts, pred_counts = np.array(gt_counts), np.array(pred_counts)
    vc = (gt_counts > 0) & (pred_counts > 0)
    if vc.sum() >= 2:
        cnt_pear  = float(pearsonr(gt_counts[vc], pred_counts[vc])[0])
        cnt_spear = float(spearmanr(gt_counts[vc], pred_counts[vc])[0])
        cnt_mae   = float(np.mean(np.abs(gt_counts[vc]-pred_counts[vc])))
        cnt_rmse  = float(np.sqrt(np.mean((gt_counts[vc]-pred_counts[vc])**2)))
    else:
        cnt_pear = cnt_spear = cnt_mae = cnt_rmse = np.nan

    vec_cos = float(np.mean(cos_l)) if cos_l else np.nan
    vec_js  = float(np.mean(js_l)) if js_l else np.nan
    gt_tot, pred_tot = np.zeros(n_gene), np.zeros(n_gene)
    for gi, pi, _ in pairs:
        gt_tot += vec(gt_grp, gt_ids[gi]); pred_tot += vec(pred_grp, pi+1)
    mk = (gt_tot > 0) | (pred_tot > 0)
    vec_pear = float(pearsonr(gt_tot[mk], pred_tot[mk])[0]) if mk.sum() >= 2 \
               else np.nan

    # ── 分配指标（逐转录本，最近邻把 GT RNA 映射到 pred 细胞标签）──────────
    #   BOMS 的 seg 是按转录本的，但 GT 转录本集合和 BOMS 的可能顺序不同，
    #   用坐标最近邻把两套转录本对上（同一批 RNA，坐标唯一）
    tree = cKDTree(np.stack([tx_x, tx_y], 1))
    gx = gt_tx.x_location.values; gy = gt_tx.y_location.values
    dd, ii = tree.query(np.stack([gx, gy], 1), k=1)
    matched_rna = dd < 0.5                       # 0.5µm 内视为同一条
    gt_tx = gt_tx.copy()
    gt_tx["boms_cell"] = 0
    gt_tx.loc[matched_rna, "boms_cell"] = seg[ii[matched_rna]]

    pred2gt = {pi+1: gt_ids[gi] for gi, pi, _ in pairs}
    both = gt_tx[(gt_tx.gt_cell != "0") & (gt_tx.boms_cell > 0)].copy()
    both["maps"] = both.boms_cell.map(pred2gt)
    valid_pairs = len(both)
    assign_acc = float((both["maps"].astype(str) == both["gt_cell"].astype(str)
                        ).mean()) if valid_pairs else np.nan
    overlap = float(both["maps"].notna().mean()) if valid_pairs else np.nan

    tot_asg = ((gt_tx.gt_cell != "0") | (gt_tx.boms_cell > 0)).sum()
    print(f"  valid_pairs = {valid_pairs:,} / {tot_asg:,}   "
          f"RNA 坐标匹配率 {matched_rna.mean()*100:.1f}%")
    print(f"  逐细胞向量 valid = {valid_vec:,} / {n_match:,}")

    R = {
        "method": cfg.method, "dataset": cfg.dataset,
        "gt_cells": len(gt_xy), "pred_cells": len(cell_loc),
        "cell_ratio": len(cell_loc)/max(len(gt_xy), 1),
        "matched": n_match, "precision": precision, "recall": recall, "f1": f1,
        "loc_mean_um": loc_mean, "loc_median_um": loc_med, "loc_p95_um": loc_p95,
        "count_pearson": cnt_pear, "count_spearman": cnt_spear,
        "count_mae": cnt_mae, "count_rmse": cnt_rmse,
        "vec_cosine": vec_cos, "vec_js_dist": vec_js, "vec_pearson": vec_pear,
        "assign_overlap": overlap, "assign_accuracy": assign_acc,
        "_valid_pairs": valid_pairs, "_n_hvg": n_gene,
    }

    print(f"\n{SEP}\n结果 (top-{n_gene} HVG)\n{SEP}")
    for gname, keys in [
        ("基础", ["gt_cells", "pred_cells", "cell_ratio"]),
        ("检测", ["matched", "precision", "recall", "f1"]),
        ("定位", ["loc_mean_um", "loc_median_um", "loc_p95_um"]),
        ("计数", ["count_pearson", "count_spearman", "count_mae", "count_rmse"]),
        ("向量", ["vec_cosine", "vec_js_dist", "vec_pearson"]),
        ("分配", ["assign_overlap", "assign_accuracy"]),
    ]:
        print(f"\n  【{gname}】")
        for k in keys:
            v = R[k]
            print(f"    {k:<18s} "
                  f"{v:,.4f}" if isinstance(v, float) else f"    {k:<18s} {v:,}")

    print(f"\n  ⚠ BOMS 用 top-{n_gene} HVG 子集（内存所限，全 4495 基因会 OOM）")
    print(f"     基因向量/计数指标仅在这 {n_gene} 个基因上计算，"
          f"与其他方法比较时需注意口径")

    out = cfg.out or f"/data/qiuyijia/eval_results/{cfg.method.lower()}/" \
                     f"{cfg.dataset}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([R]).to_csv(out, index=False)
    print(f"\n  已写 → {out}")
    return R


if __name__ == "__main__":
    evaluate(parse_args())