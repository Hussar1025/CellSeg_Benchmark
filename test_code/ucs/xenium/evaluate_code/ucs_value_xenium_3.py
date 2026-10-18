#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCS 评估 —— 19 指标统一框架
=========================================================================
适配 Xenium ROI（人淋巴结）：
  GT 细胞     : cells.parquet   （x_centroid, y_centroid, cell_area）
  GT 逐转录本 : transcripts.parquet 的 cell_id 列（94.5% 已归属）
  预测        : UCS segmentation_mask.tif（bin 空间的 label 图）

19 指标分五组：
  基础(3)  : GT/pred 细胞数、比例
  检测(4)  : 匹配数、Precision、Recall、F1
  定位(3)  : mean / median / p95 质心偏移
  计数(4)  : 转录本计数 Pearson / Spearman / MAE / RMSE
  向量(3)  : 基因向量 cosine / JS距离 / Pearson
  分配(2)  : 转录本 ID overlap、assignment accuracy
（3+4+3+4+3+2 = 19）

坑位对齐（历史教训）：
  * 坐标系：pred 在 bin 空间（原点 = ROI 左下角，1 bin = bin_um µm），
    GT 在 µm 全局坐标。评估前把 GT 换算到 bin 空间统一比较。
  * valid_pairs：只在 GT 和 pred 都判给某个细胞的转录本上算 accuracy/cosine，
    否则坐标错位会让指标虚高（幸存者偏差）。会同时报 valid_pairs 数量。
  * �apiน配：匈牙利匹配对大图会退化，>2000 细胞时用基于质心的贪心近邻匹配。
  * 循环虚高：UCS 用核先验、GT 也源自同一核时，transcript accuracy 会虚高。
    Xenium 的 GT (cell_id) 独立于 UCS 的核输入，故此处 accuracy 有效——
    但仍打印一句提示，方便和 MERFISH（循环无意义）区分。

用法
-----
python eval_ucs_xenium.py \
    --label-path /data/qiuyijia/ucs_xenium_roi/ucs_k5_i2_t5/pred/segmentation_mask.tif \
    --data-dir   /data/qiuyijia/dataset/xenium_roi_crop \
    --out        /data/qiuyijia/eval_results/ucs/xenium_lymph.csv
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
    p.add_argument("--data-dir",   required=True, help="xenium_roi_crop 目录")
    p.add_argument("--out",        default=None,  help="结果 CSV 路径")
    p.add_argument("--method",     default="UCS")
    p.add_argument("--dataset",    default="xenium_lymph")
    p.add_argument("--bin-um",     type=float, default=1.0)
    p.add_argument("--qv-min",     type=float, default=20.0)
    p.add_argument("--match-dist", type=float, default=None,
                   help="匹配的最大质心距离(µm)，默认 = 细胞直径")
    return p.parse_args()


def find_one(base, patterns, label, required=True):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(base, "**", pat), recursive=True))
        if hits:
            hits.sort(key=len)
            return hits[0]
    if required:
        raise FileNotFoundError(f"找不到 {label}: {patterns}")
    return None


def read_table(path, columns=None):
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


# ══════════════════════════════════════════════════════════════════════════════
def load_all(cfg):
    base = cfg.data_dir
    f_tx    = find_one(base, ["transcripts.parquet", "transcripts.csv.gz"], "转录本")
    f_cells = find_one(base, ["cells.parquet", "cells.csv.gz", "cells.csv"], "GT 细胞")
    f_roi   = find_one(base, ["experiment_roi.json"], "ROI 元信息", required=False)

    # ROI 原点 + 像素尺寸
    x0 = y0 = None
    pixel = 0.2125
    if f_roi:
        meta = json.load(open(f_roi))
        roi = meta.get("roi", {})
        x0 = roi.get("x_min_um"); y0 = roi.get("y_min_um")
        pixel = meta.get("pixels", {}).get("pixel_size_um", 0.2125)

    # 预测 mask
    mask = tifffile.imread(cfg.label_path)
    H, W = mask.shape
    print(f"  预测 mask: {mask.shape}  label 数 {len(np.unique(mask))-1:,}")

    # 转录本
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

    # ROI 原点兜底：没有 json 就用转录本最小值
    if x0 is None:
        x0 = float(np.floor(tx.x_location.min()))
        y0 = float(np.floor(tx.y_location.min()))
    print(f"  ROI 原点 ({x0:.1f}, {y0:.1f}) µm   bin {cfg.bin_um} µm")

    # 转录本 → bin 坐标（和 mask 对齐）
    tx["bx"] = ((tx.x_location - x0) / cfg.bin_um).astype(int)
    tx["by"] = ((tx.y_location - y0) / cfg.bin_um).astype(int)
    inb = (tx.bx >= 0) & (tx.bx < W) & (tx.by >= 0) & (tx.by < H)
    tx = tx[inb].reset_index(drop=True)

    # 预测归属：每条转录本落在哪个 pred label
    tx["pred_cell"] = mask[tx.by.values, tx.bx.values]

    # GT 归属：cell_id 列（UNASSIGNED → 0）
    if "cell_id" in tx.columns:
        cid = tx["cell_id"].astype(str)
        tx["gt_cell"] = cid.where(cid != "UNASSIGNED", "0")
    else:
        tx["gt_cell"] = "0"

    # GT 细胞质心（换算到 bin 空间）
    cells = read_table(f_cells)
    cells = cells.copy()
    cells["gx"] = (cells.x_centroid - x0) / cfg.bin_um
    cells["gy"] = (cells.y_centroid - y0) / cfg.bin_um
    cells["cell_id"] = cells["cell_id"].astype(str)

    return mask, tx, cells, pixel


# ══════════════════════════════════════════════════════════════════════════════
def pred_centroids(mask):
    """每个 pred label 的质心（bin 坐标）"""
    ys, xs = np.nonzero(mask)
    labels = mask[ys, xs]
    df = pd.DataFrame({"label": labels, "x": xs, "y": ys})
    c = df.groupby("label")[["x", "y"]].mean()
    return c  # index = label


def match_cells(gt_xy, pred_xy, max_dist):
    """
    质心匹配。小图用匈牙利，大图（>2000）用贪心近邻。
    返回 matched 对列表 [(gt_idx, pred_idx, dist), ...]
    """
    n_gt, n_pred = len(gt_xy), len(pred_xy)
    if n_gt == 0 or n_pred == 0:
        return []

    # 贪心近邻：对每个 GT 找最近 pred，双向唯一
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
                # 退而找次近的未用 pred
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

    # 匈牙利（小图精确）
    from scipy.spatial.distance import cdist
    D = cdist(gt_xy, pred_xy)
    D[D > max_dist] = 1e6
    ri, ci = linear_sum_assignment(D)
    return [(int(r), int(c), float(D[r, c]))
            for r, c in zip(ri, ci) if D[r, c] < 1e6]


# ══════════════════════════════════════════════════════════════════════════════
def evaluate(cfg):
    print(SEP); print(f"UCS 评估  {cfg.dataset}"); print(SEP)
    mask, tx, cells, pixel = load_all(cfg)

    diam = 2 * np.sqrt(cells["cell_area"].median() / np.pi) if \
        "cell_area" in cells.columns else 7.0
    max_dist_um = cfg.match_dist if cfg.match_dist else diam
    max_dist_bin = max_dist_um / cfg.bin_um
    print(f"  GT 细胞直径 {diam:.1f} µm  →  匹配阈值 {max_dist_um:.1f} µm "
          f"({max_dist_bin:.1f} bin)")

    pc = pred_centroids(mask)                       # index=label, cols x,y
    gt_xy   = cells[["gx", "gy"]].values
    pred_xy = pc[["x", "y"]].values
    print(f"\n  GT {len(gt_xy):,} 细胞   pred {len(pred_xy):,} 细胞")

    pairs = match_cells(gt_xy, pred_xy, max_dist_bin)
    n_match = len(pairs)
    gt_ids   = cells["cell_id"].values
    pred_lbl = pc.index.values

    # ── 检测指标 ────────────────────────────────────────────────────────────
    precision = n_match / max(len(pred_xy), 1)
    recall    = n_match / max(len(gt_xy), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # ── 定位指标 ────────────────────────────────────────────────────────────
    if n_match:
        dists_um = np.array([d for _, _, d in pairs]) * cfg.bin_um
        loc_mean, loc_med, loc_p95 = (float(np.mean(dists_um)),
                                      float(np.median(dists_um)),
                                      float(np.percentile(dists_um, 95)))
    else:
        loc_mean = loc_med = loc_p95 = np.nan

    # ── 每个匹配细胞的转录本计数 & 基因向量 ──────────────────────────────────
    genes = np.sort(tx.feature_name.unique())
    g2i = {g: i for i, g in enumerate(genes)}
    tx["gi"] = tx.feature_name.map(g2i).values

    # pred 细胞 → 基因计数向量
    pred_grp = tx[tx.pred_cell > 0].groupby(["pred_cell", "gi"]).size()
    # GT 细胞 → 基因计数向量
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

    # ── 计数指标 ────────────────────────────────────────────────────────────
    valid_c = (gt_counts > 0) & (pred_counts > 0)
    if valid_c.sum() >= 2:
        cnt_pear = float(pearsonr(gt_counts[valid_c], pred_counts[valid_c])[0])
        cnt_spear = float(spearmanr(gt_counts[valid_c], pred_counts[valid_c])[0])
        cnt_mae  = float(np.mean(np.abs(gt_counts[valid_c] - pred_counts[valid_c])))
        cnt_rmse = float(np.sqrt(np.mean((gt_counts[valid_c] - pred_counts[valid_c])**2)))
    else:
        cnt_pear = cnt_spear = cnt_mae = cnt_rmse = np.nan

    # ── 向量指标 ────────────────────────────────────────────────────────────
    vec_cos = float(np.mean(cos_list)) if cos_list else np.nan
    vec_js  = float(np.mean(js_list))  if js_list  else np.nan
    if valid_vec >= 2:
        # 逐细胞 cosine 已有；再算整体基因丰度的 pearson
        gt_tot   = np.zeros(len(genes))
        pred_tot = np.zeros(len(genes))
        for gi, pi, _ in pairs:
            gt_tot   += vec(gt_grp,   gt_ids[gi],  len(genes))
            pred_tot += vec(pred_grp, pred_lbl[pi], len(genes))
        m = (gt_tot > 0) | (pred_tot > 0)
        vec_pear = float(pearsonr(gt_tot[m], pred_tot[m])[0]) if m.sum() >= 2 else np.nan
    else:
        vec_pear = np.nan

    # ── 分配指标：逐转录本 GT vs pred ───────────────────────────────────────
    #   建立 pred_label → gt_cell 的多数投票映射，再看每条 RNA 是否一致
    gt2pred = {gt_ids[gi]: pred_lbl[pi] for gi, pi, _ in pairs}
    pred2gt = {pred_lbl[pi]: gt_ids[gi] for gi, pi, _ in pairs}

    # 只在 GT 和 pred 都有归属的转录本上算（valid_pairs）
    both = tx[(tx.gt_cell != "0") & (tx.pred_cell > 0)].copy()
    both["pred_maps_to_gt"] = both.pred_cell.map(pred2gt)
    valid_pairs = len(both)
    if valid_pairs:
        assign_acc = float((both["pred_maps_to_gt"].astype(str)
                            == both["gt_cell"].astype(str)).mean())
        # ID overlap：pred 和 gt 归到"同一对"的转录本占比
        overlap = float(both["pred_maps_to_gt"].notna().mean())
    else:
        assign_acc = overlap = np.nan

    total_asg = ((tx.gt_cell != "0") | (tx.pred_cell > 0)).sum()
    print(f"\n  valid_pairs (GT&pred 都归属) = {valid_pairs:,} / "
          f"{total_asg:,} 有归属转录本")
    print(f"  逐细胞向量 valid = {valid_vec:,} / {n_match:,} 匹配对")

    # ── 汇总 ────────────────────────────────────────────────────────────────
    R = {
        "method":  cfg.method,
        "dataset": cfg.dataset,
        # 基础
        "gt_cells":   len(gt_xy),
        "pred_cells": len(pred_xy),
        "cell_ratio": len(pred_xy) / max(len(gt_xy), 1),
        # 检测
        "matched":   n_match,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        # 定位
        "loc_mean_um":   loc_mean,
        "loc_median_um": loc_med,
        "loc_p95_um":    loc_p95,
        # 计数
        "count_pearson":  cnt_pear,
        "count_spearman": cnt_spear,
        "count_mae":      cnt_mae,
        "count_rmse":     cnt_rmse,
        # 向量
        "vec_cosine":  vec_cos,
        "vec_js_dist": vec_js,
        "vec_pearson": vec_pear,
        # 分配
        "assign_overlap":  overlap,
        "assign_accuracy": assign_acc,
        # 诊断（不计入 19，但要留痕）
        "_valid_pairs":     valid_pairs,
        "_valid_vec_pairs": valid_vec,
    }

    # ── 打印 ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n结果\n{SEP}")
    groups = [
        ("基础", ["gt_cells", "pred_cells", "cell_ratio"]),
        ("检测", ["matched", "precision", "recall", "f1"]),
        ("定位", ["loc_mean_um", "loc_median_um", "loc_p95_um"]),
        ("计数", ["count_pearson", "count_spearman", "count_mae", "count_rmse"]),
        ("向量", ["vec_cosine", "vec_js_dist", "vec_pearson"]),
        ("分配", ["assign_overlap", "assign_accuracy"]),
    ]
    for gname, keys in groups:
        print(f"\n  【{gname}】")
        for k in keys:
            v = R[k]
            s = f"{v:,.4f}" if isinstance(v, float) else f"{v:,}"
            print(f"    {k:<18s} {s}")

    print(f"\n  ⚠ 说明：GT (transcripts.cell_id) 独立于 UCS 的核先验输入，"
          f"故 assign_accuracy={assign_acc:.3f} 有效")
    print(f"    （对比 MERFISH：那里 GT 与核同源，assignment 循环虚高、无意义）")

    # ── 落盘 ────────────────────────────────────────────────────────────────
    out = cfg.out or f"/data/qiuyijia/eval_results/{cfg.method.lower()}/" \
                     f"{cfg.dataset}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([R]).to_csv(out, index=False)
    print(f"\n  已写 → {out}")
    return R


if __name__ == "__main__":
    evaluate(parse_args())