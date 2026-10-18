#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_ucs_merfish.py — UCS 在 MERFISH 小鼠脑（Slice1 Replicate1）上的分割评估
                       【v2：修复大连通分量被静默丢弃的匹配 bug】

v1 → v2 唯一改动：match_bipartite()
  v1 对 len(gsub)*len(psub) > 36,000,000 的连通分量直接 continue，
  MERFISH 细胞密度低导致 8448×8390 的巨型分量被整块丢弃 → 只匹配到 18 个。
  v2 改为：小分量走匈牙利求最优，大分量走按距离升序的贪心，绝不丢弃任何分量。

其余逻辑（ROI 校验、GT 加载、转录本扫描、指标口径）与 v1 完全一致。

MERFISH 的两点特殊处理：
  1. detected_transcripts.csv 没有 cell_id 列（Vizgen 不提供逐转录本官方分配），
     GT 侧的转录本归属改为查 nuclei_mask.tif —— 它是从官方 cell_boundaries
     HDF5 栅格化来的，即 Vizgen 官方分割在 1 µm/bin 空间的表示。
  2. cell_metadata.csv 的 cell_id（39 位数字）与 nuclei_mask 的顺序 ID 无对应关系，
     通过质心最近邻建立 {nuclei_mask_id: gt_cell_id} 映射把两者打通。

注意：UCS 把 nuclei_mask 当作 prior 使用，因此转录本级指标对 UCS 略有偏向
      （与 CosMx 版本用 CellLabels 当 prior 是同一情况，口径一致）。

用法：
  python eval_ucs_merfish.py --output-dir /data/qiuyijia/eval_results/ucs_merfish
  python eval_ucs_merfish.py --match-radius 15        # 手动指定更严格的匹配半径
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import tifffile

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = "/data/qiuyijia/dataset/merfish_mouse_brain"
_PFX     = "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1"

DEF_META_CSV  = os.path.join(DATA_DIR, f"{_PFX}_cell_metadata_S1R1.csv")
DEF_TX_CSV    = os.path.join(DATA_DIR, f"{_PFX}_detected_transcripts_S1R1.csv")

DEF_PRED_MASK = "/data/qiuyijia/ucs_merfish_brain/ucs_log/pred/segmentation_mask.tif"
DEF_NUC_MASK  = "/data/qiuyijia/ucs_merfish_brain/nuclei_mask.tif"

SCALE_X   = 9.205861091613770       # px per µm
SCALE_Y   = 9.205849647521973
DEF_CX_UM  = 4675.0
DEF_CY_UM  = 3413.0
DEF_ROI_PX = 20000
# ══════════════════════════════════════════════════════════════════════════════


def log(msg):
    print(msg, flush=True)


def hr(title):
    log("\n" + "─" * 68)
    log(f"  {title}")
    log("─" * 68)


# ─────────────────────────────────────────────────────────────────────────────
#  ROI 参数（与分割脚本同一套公式 + 硬校验）
# ─────────────────────────────────────────────────────────────────────────────
def compute_roi(cx_um, cy_um, roi_px, mask_shape):
    half_x = roi_px / 2.0 / SCALE_X
    half_y = roi_px / 2.0 / SCALE_Y
    x0, x1 = cx_um - half_x, cx_um + half_x
    y0, y1 = cy_um - half_y, cy_um + half_y

    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    map_w = int(np.ceil(x1)) - ix0
    map_h = int(np.ceil(y1)) - iy0

    log(f"[ROI] 中心 ({cx_um:.1f}, {cy_um:.1f}) µm   边长 {roi_px} px")
    log(f"[ROI] µm 范围  x=[{x0:.2f},{x1:.2f}]  y=[{y0:.2f},{y1:.2f}]")
    log(f"[ROI] bin 原点 ({ix0}, {iy0})   推算 map = {map_h}×{map_w}")
    log(f"[ROI] 实际 mask shape        = {mask_shape[0]}×{mask_shape[1]}")

    if (map_h, map_w) != tuple(mask_shape):
        log("\n  ✘ ROI 推算尺寸与 mask 不符，评估会全错，已中止。")
        log("    请用 --cx-um / --cy-um / --roi-px 传入分割时实际参数，")
        log("    或用 --roi-ix0 / --roi-iy0 直接覆盖 bin 原点。")
        sys.exit(1)

    log("[ROI] ✓ 尺寸校验通过")
    return ix0, iy0, map_w, map_h, x0, x1, y0, y1


# ─────────────────────────────────────────────────────────────────────────────
#  GT 细胞（cell_metadata.csv）
# ─────────────────────────────────────────────────────────────────────────────
def load_gt_cells(meta_csv, x0, x1, y0, y1):
    log(f"[GT] 加载: {meta_csv}")
    df = pd.read_csv(meta_csv)
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str)

    m = ((df.center_x >= x0) & (df.center_x <= x1) &
         (df.center_y >= y0) & (df.center_y <= y1))
    roi = df[m].copy().reset_index(drop=True)
    roi = roi.rename(columns={"center_x": "x", "center_y": "y"})

    log(f"[GT] 全图 {len(df):,} 个 → ROI 内 {len(roi):,} 个")
    log(f"[GT] x=[{roi.x.min():.1f},{roi.x.max():.1f}]µm  "
        f"y=[{roi.y.min():.1f},{roi.y.max():.1f}]µm")
    return roi[["cell_id", "x", "y", "volume"]]


# ─────────────────────────────────────────────────────────────────────────────
#  mask → 细胞质心（bin → µm）
# ─────────────────────────────────────────────────────────────────────────────
def mask_to_cells(mask, ix0, iy0, tag):
    ids = np.unique(mask)
    ids = ids[ids > 0]
    log(f"[{tag}] mask shape={mask.shape} dtype={mask.dtype}  细胞 ID {len(ids):,} 个")

    t0    = time.time()
    ones  = np.ones_like(mask, dtype=np.uint8)
    coms  = np.asarray(ndi.center_of_mass(ones, mask, ids), dtype=np.float64)
    sizes = np.asarray(ndi.sum(ones, mask, ids), dtype=np.int64)

    df = pd.DataFrame({
        "mask_id" : ids.astype(np.int64),
        "x"       : coms[:, 1] + ix0,
        "y"       : coms[:, 0] + iy0,
        "n_pixels": sizes,
    })
    log(f"[{tag}] 质心完成 {len(df):,} 个  ({time.time()-t0:.1f}s)  "
        f"x=[{df.x.min():.1f},{df.x.max():.1f}] y=[{df.y.min():.1f},{df.y.max():.1f}]")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  {nuclei_mask_id → gt_cell_id}
# ─────────────────────────────────────────────────────────────────────────────
def build_nuclei_to_gt_map(nuc_cells, gt_cells, max_dist=5.0):
    tree   = cKDTree(gt_cells[["x", "y"]].values)
    d, idx = tree.query(nuc_cells[["x", "y"]].values, k=1)

    ok = d <= max_dist
    mapping = {int(mid): str(gid) for mid, gid in
               zip(nuc_cells.loc[ok, "mask_id"].values,
                   gt_cells["cell_id"].values[idx[ok]])}

    log(f"[映射] nuclei_mask {len(nuc_cells):,} 个 → 对上 GT {int(ok.sum()):,} 个 "
        f"({ok.sum()/max(len(nuc_cells),1)*100:.1f}%)")
    log(f"[映射] 质心距离  中位 {np.median(d):.2f}µm  "
        f"p95 {np.percentile(d,95):.2f}µm  max {d.max():.2f}µm")
    if ok.sum() / max(len(nuc_cells), 1) < 0.8:
        log("  ⚠ 对上比例偏低，转录本级 GT 归属可能不完整")
    return mapping


# ═════════════════════════════════════════════════════════════════════════════
#  【v2 修复】二分图匹配：小分量匈牙利求最优，大分量贪心，绝不丢弃分量
# ═════════════════════════════════════════════════════════════════════════════
def match_bipartite(gt, pred, radius, dense_side=2000):
    gt_xy   = gt[["x", "y"]].values
    pred_xy = pred[["x", "y"]].values
    n_g, n_p = len(gt_xy), len(pred_xy)

    log(f"[匹配] radius={radius:.4f} µm  GT={n_g:,}  Pred={n_p:,}")

    # ── 候选边（向量化构建）─────────────────────────────────────────────────
    t0 = time.time()
    tg, tp = cKDTree(gt_xy), cKDTree(pred_xy)
    pairs  = tg.query_ball_tree(tp, r=radius)
    lens   = np.fromiter((len(x) for x in pairs), dtype=np.int64, count=n_g)
    if lens.sum() == 0:
        log("[匹配] ✘ 没有任何候选边，坐标系可能不匹配")
        return pd.DataFrame(columns=["gt_idx", "pred_idx", "dist"])

    gi = np.repeat(np.arange(n_g, dtype=np.int64), lens)
    pi = np.concatenate([np.asarray(x, dtype=np.int64) for x in pairs if len(x)])
    dd = np.hypot(gt_xy[gi, 0] - pred_xy[pi, 0], gt_xy[gi, 1] - pred_xy[pi, 1])
    log(f"[匹配] 候选边 {len(gi):,} 条  ({time.time()-t0:.1f}s)")

    # ── 连通分量拆解 ────────────────────────────────────────────────────────
    adj = csr_matrix((np.ones(len(gi)), (gi, pi + n_g)),
                     shape=(n_g + n_p, n_g + n_p))
    adj = adj + adj.T
    n_comp, labels = connected_components(adj, directed=False)

    ec   = labels[gi]
    srt  = np.argsort(ec, kind="stable")
    ec_s = ec[srt]
    uniq = np.unique(ec_s)
    starts = np.searchsorted(ec_s, uniq, side="left")
    starts = np.append(starts, len(ec_s))
    log(f"[匹配] 连通分量 {len(uniq):,} 个（含边的）")

    rows        = []
    n_hung_comp = n_greedy_comp = 0
    n_hung_cell = n_greedy_cell = 0
    dense_cap   = dense_side * dense_side

    for i in range(len(uniq)):
        eidx = srt[starts[i]:starts[i + 1]]
        g_e, p_e, d_e = gi[eidx], pi[eidx], dd[eidx]
        gsub = np.unique(g_e)
        psub = np.unique(p_e)

        if len(gsub) * len(psub) <= dense_cap:
            # 小分量：稠密匈牙利，求全局最优
            gmap = {g: k for k, g in enumerate(gsub)}
            pmap = {p: k for k, p in enumerate(psub)}
            cost = np.full((len(gsub), len(psub)), radius * 10.0)
            for g, p, d in zip(g_e, p_e, d_e):
                cost[gmap[g], pmap[p]] = d
            r, c = linear_sum_assignment(cost)
            hit = 0
            for a, b in zip(r, c):
                if cost[a, b] <= radius:
                    rows.append((int(gsub[a]), int(psub[b]), float(cost[a, b])))
                    hit += 1
            n_hung_comp += 1
            n_hung_cell += hit
        else:
            # 大分量：按距离升序贪心（避免 O(n^3) 爆炸，绝不丢弃）
            order = np.argsort(d_e, kind="stable")
            g_used, p_used = set(), set()
            hit = 0
            for k in order:
                g, p = int(g_e[k]), int(p_e[k])
                if g in g_used or p in p_used:
                    continue
                g_used.add(g); p_used.add(p)
                rows.append((g, p, float(d_e[k])))
                hit += 1
            n_greedy_comp += 1
            n_greedy_cell += hit

    m = pd.DataFrame(rows, columns=["gt_idx", "pred_idx", "dist"])
    log(f"[匹配] 匈牙利分量 {n_hung_comp:,} 个 → {n_hung_cell:,} 对")
    log(f"[匹配] 贪心分量   {n_greedy_comp:,} 个 → {n_greedy_cell:,} 对")
    log(f"[匹配] 匹配细胞数: {len(m):,}")

    if len(m) < 0.5 * min(n_g, n_p):
        log("  ⚠ 匹配数不足细胞数的一半，请核对 ROI 原点与坐标换算")
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  逐转录本扫描
# ─────────────────────────────────────────────────────────────────────────────
def scan_transcripts(tx_csv, pred_mask, nuc_mask, ix0, iy0,
                     pred_cells, gt_cells, nuc2gt, matches,
                     chunksize=5_000_000, drop_blank=True):
    map_h, map_w = pred_mask.shape

    pred_ids = pred_cells["mask_id"].values.astype(np.int64)
    pred_pos = np.full(int(pred_mask.max()) + 1, -1, dtype=np.int64)
    pred_pos[pred_ids] = np.arange(len(pred_ids))

    gt_id_list = gt_cells["cell_id"].values
    gt_index   = {cid: i for i, cid in enumerate(gt_id_list)}

    nuc_max = int(nuc_mask.max())
    nuc_pos = np.full(nuc_max + 1, -1, dtype=np.int64)
    for nid, gid in nuc2gt.items():
        if 0 < nid <= nuc_max and gid in gt_index:
            nuc_pos[nid] = gt_index[gid]

    n_pred, n_gt = len(pred_ids), len(gt_id_list)

    pair_of_pred = np.full(n_pred, -1, dtype=np.int64)
    for _, r in matches.iterrows():
        pair_of_pred[int(r["pred_idx"])] = int(r["gt_idx"])

    # ── 第一遍：基因集合 ────────────────────────────────────────────────────
    log("[TX] 第 1 遍：扫描基因列表 ...")
    genes, total_rows = set(), 0
    for ch in pd.read_csv(tx_csv, usecols=["gene"], chunksize=chunksize):
        genes.update(ch["gene"].dropna().unique())
        total_rows += len(ch)
    if drop_blank:
        n_all = len(genes)
        genes = {g for g in genes if not str(g).lower().startswith("blank")}
        log(f"[TX] 基因 {n_all} 个 → 剔除 Blank 后 {len(genes)} 个")
    gene_list = sorted(genes)
    gene_idx  = {g: i for i, g in enumerate(gene_list)}
    n_genes   = len(gene_list)
    log(f"[TX] 全图转录本 {total_rows:,} 行   参与统计基因 {n_genes} 个")

    # ── 第二遍：查 mask 归属并累加 ──────────────────────────────────────────
    pred_gene = np.zeros((n_pred, n_genes), dtype=np.float32)
    gt_gene   = np.zeros((n_gt,   n_genes), dtype=np.float32)
    pred_ntx  = np.zeros(n_pred, dtype=np.int64)
    gt_ntx    = np.zeros(n_gt,   dtype=np.int64)

    n_roi = n_pred_assigned = n_gt_assigned = 0
    n_both = n_in_matched = n_correct = 0

    log("[TX] 第 2 遍：查 mask 归属并累加（约 5-10 分钟）...")
    t0, read_rows = time.time(), 0
    for ch in pd.read_csv(tx_csv,
                          usecols=["global_x", "global_y", "gene"],
                          chunksize=chunksize):
        read_rows += len(ch)

        bx = np.round(ch["global_x"].values).astype(np.int64) - ix0
        by = np.round(ch["global_y"].values).astype(np.int64) - iy0
        inside = (bx >= 0) & (bx < map_w) & (by >= 0) & (by < map_h)
        if not inside.any():
            log(f"[TX]   已读 {read_rows:,}  ROI 内 {n_roi:,}")
            continue

        bx, by = bx[inside], by[inside]
        gnames = ch["gene"].values[inside]

        gidx = np.array([gene_idx.get(g, -1) for g in gnames], dtype=np.int64)
        keep = gidx >= 0
        bx, by, gidx = bx[keep], by[keep], gidx[keep]
        n_roi += len(bx)

        p_raw = pred_mask[by, bx].astype(np.int64)
        n_raw = nuc_mask[by, bx].astype(np.int64)
        p_i = np.where(p_raw > 0, pred_pos[np.clip(p_raw, 0, len(pred_pos) - 1)], -1)
        g_i = np.where(n_raw > 0, nuc_pos[np.clip(n_raw, 0, nuc_max)], -1)

        has_p, has_g = p_i >= 0, g_i >= 0
        n_pred_assigned += int(has_p.sum())
        n_gt_assigned   += int(has_g.sum())
        n_both          += int((has_p & has_g).sum())

        if has_p.any():
            flat = p_i[has_p] * n_genes + gidx[has_p]
            pred_gene += np.bincount(flat, minlength=n_pred * n_genes
                                     ).reshape(n_pred, n_genes).astype(np.float32)
            pred_ntx += np.bincount(p_i[has_p], minlength=n_pred)
        if has_g.any():
            flat = g_i[has_g] * n_genes + gidx[has_g]
            gt_gene += np.bincount(flat, minlength=n_gt * n_genes
                                   ).reshape(n_gt, n_genes).astype(np.float32)
            gt_ntx += np.bincount(g_i[has_g], minlength=n_gt)

        sel = has_p & has_g
        if sel.any():
            expect = pair_of_pred[p_i[sel]]
            valid  = expect >= 0
            n_in_matched += int(valid.sum())
            n_correct    += int((expect[valid] == g_i[sel][valid]).sum())

        log(f"[TX]   已读 {read_rows:,}  ROI 内 {n_roi:,}")

    log(f"[TX] 扫描完成  耗时 {time.time()-t0:.1f}s")
    log(f"[TX] ROI 内转录本            : {n_roi:,}")
    log(f"[TX] Pred 已分配             : {n_pred_assigned:,} "
        f"({n_pred_assigned/max(n_roi,1)*100:.1f}%)")
    log(f"[TX] GT   已分配             : {n_gt_assigned:,} "
        f"({n_gt_assigned/max(n_roi,1)*100:.1f}%)")
    log(f"[TX] 双侧都有归属 (overlap)  : {n_both:,}")
    log(f"[TX] 落在 matched pair 内    : {n_in_matched:,}")
    log(f"[TX] 其中分配一致            : {n_correct:,} "
        f"({n_correct/max(n_in_matched,1)*100:.1f}%)")

    return {
        "gene_list": gene_list,
        "pred_gene": pred_gene, "gt_gene": gt_gene,
        "pred_ntx" : pred_ntx,  "gt_ntx" : gt_ntx,
        "n_roi": n_roi,
        "n_pred_assigned": n_pred_assigned,
        "n_gt_assigned"  : n_gt_assigned,
        "n_both": n_both,
        "n_in_matched": n_in_matched,
        "n_correct": n_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  基因向量相似度
# ─────────────────────────────────────────────────────────────────────────────
def gene_vector_metrics(matches, pred_gene, gt_gene):
    cos_l, js_l, pr_l = [], [], []
    gidx = matches["gt_idx"].values.astype(int)
    pidx = matches["pred_idx"].values.astype(int)

    for pi_, gi_ in zip(pidx, gidx):
        p, g = pred_gene[pi_], gt_gene[gi_]
        if p.sum() <= 0 or g.sum() <= 0:
            continue

        cos_l.append(float(np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g))))

        pp, gg = p / p.sum(), g / g.sum()
        mm = 0.5 * (pp + gg)

        def _kl(a, b):
            nz = a > 0
            return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

        js = 0.5 * _kl(pp, mm) + 0.5 * _kl(gg, mm)
        js_l.append(float(np.sqrt(max(js, 0.0))))

        if p.std() > 0 and g.std() > 0:
            pr_l.append(float(np.corrcoef(p, g)[0, 1]))

    return {
        "matched_cell_gene_vector_mean_cosine"     : float(np.mean(cos_l)) if cos_l else np.nan,
        "matched_cell_gene_vector_mean_js_distance": float(np.mean(js_l))  if js_l  else np.nan,
        "matched_cell_gene_vector_mean_pearson"    : float(np.mean(pr_l))  if pr_l  else np.nan,
        "matched_cell_gene_vector_valid_pairs"     : len(cos_l),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    hr("加载 mask")
    pred_mask = tifffile.imread(cfg.pred_mask)
    nuc_mask  = tifffile.imread(cfg.nuclei_mask)
    log(f"[Pred] {cfg.pred_mask}")
    log(f"[Nuc ] {cfg.nuclei_mask}")
    if nuc_mask.shape != pred_mask.shape:
        log(f"  ✘ nuclei_mask {nuc_mask.shape} 与 pred_mask {pred_mask.shape} 形状不符")
        sys.exit(1)
    if int(nuc_mask.max()) == 0:
        log("  ✘ nuclei_mask 为空，无法评估转录本级指标")
        sys.exit(1)

    hr("ROI 校验")
    if cfg.roi_ix0 is not None and cfg.roi_iy0 is not None:
        ix0, iy0 = cfg.roi_ix0, cfg.roi_iy0
        map_h, map_w = pred_mask.shape
        x0, x1 = ix0, ix0 + map_w
        y0, y1 = iy0, iy0 + map_h
        log(f"[ROI] 使用手动指定原点 ({ix0}, {iy0})")
    else:
        ix0, iy0, map_w, map_h, x0, x1, y0, y1 = compute_roi(
            cfg.cx_um, cfg.cy_um, cfg.roi_px, pred_mask.shape)

    hr("GT 细胞 / Pred 细胞")
    gt_cells   = load_gt_cells(cfg.meta_csv, x0, x1, y0, y1)
    pred_cells = mask_to_cells(pred_mask, ix0, iy0, "Pred")
    nuc_cells  = mask_to_cells(nuc_mask,  ix0, iy0, "Nuc ")

    hr("打通 nuclei_mask ↔ GT cell_id")
    nuc2gt = build_nuclei_to_gt_map(nuc_cells, gt_cells, cfg.map_max_dist)

    hr("细胞匹配")
    gt_xy = gt_cells[["x", "y"]].values
    nn = cKDTree(gt_xy).query(gt_xy, k=2)[0][:, 1]
    radius = (cfg.match_radius if cfg.match_radius is not None
              else float(np.quantile(nn, cfg.radius_quantile) * cfg.radius_multiplier))
    log(f"[匹配] GT 最近邻距离  中位 {np.median(nn):.2f}µm  "
        f"q{cfg.radius_quantile:.2f} {np.quantile(nn, cfg.radius_quantile):.2f}µm")
    matches = match_bipartite(gt_cells, pred_cells, radius, cfg.dense_side)

    hr("逐转录本扫描")
    tx = scan_transcripts(cfg.tx_csv, pred_mask, nuc_mask, ix0, iy0,
                          pred_cells, gt_cells, nuc2gt, matches,
                          chunksize=cfg.chunksize, drop_blank=not cfg.keep_blank)

    # ── 指标汇总 ────────────────────────────────────────────────────────────
    hr("指标汇总")
    n_gt, n_pred, n_match = len(gt_cells), len(pred_cells), len(matches)
    m = {"method": cfg.method_name}

    m["Pred cell count"]  = n_pred
    m["GT cell count"]    = n_gt
    m["Cell count ratio"] = n_pred / n_gt if n_gt else np.nan

    m["Matched cell count"]  = n_match
    m["Detection precision"] = n_match / n_pred if n_pred else np.nan
    m["Detection recall"]    = n_match / n_gt   if n_gt   else np.nan
    p, r = m["Detection precision"], m["Detection recall"]
    m["Detection F1"] = (2 * p * r / (p + r)) if (p and r and p + r > 0) else np.nan

    if n_match:
        d = matches["dist"].values
        m["Mean centroid shift"]   = float(d.mean())
        m["Median centroid shift"] = float(np.median(d))
        m["p95 centroid shift"]    = float(np.percentile(d, 95))
    else:
        for k in ["Mean centroid shift", "Median centroid shift", "p95 centroid shift"]:
            m[k] = np.nan

    if n_match:
        pv = tx["pred_ntx"][matches["pred_idx"].values.astype(int)].astype(float)
        gv = tx["gt_ntx"][matches["gt_idx"].values.astype(int)].astype(float)
        ok = np.isfinite(pv) & np.isfinite(gv)
        if ok.sum() >= 3:
            pear = float(pearsonr(pv[ok], gv[ok])[0])
            spea = float(spearmanr(pv[ok], gv[ok])[0])
            mae  = float(np.mean(np.abs(pv[ok] - gv[ok])))
            rmse = float(np.sqrt(np.mean((pv[ok] - gv[ok]) ** 2)))
        else:
            pear = spea = mae = rmse = np.nan
    else:
        pear = spea = mae = rmse = np.nan

    for prefix in ["matched_pair_n_transcripts_vs_total_counts",
                   "matched_cell_transcript_count"]:
        m[f"{prefix}_pearson"]  = pear
        m[f"{prefix}_spearman"] = spea
        m[f"{prefix}_mae"]      = mae
        m[f"{prefix}_rmse"]     = rmse

    m["Pred transcript rows"]            = int(tx["n_pred_assigned"])
    m["GT transcript rows"]              = int(tx["n_roi"])
    m["Pred transcript assignment rate"] = (tx["n_pred_assigned"] / tx["n_roi"]
                                            if tx["n_roi"] else np.nan)
    m["Transcript matched cell pairs"]   = n_match

    m.update(gene_vector_metrics(matches, tx["pred_gene"], tx["gt_gene"]))

    m["transcript_id_overlap_n"] = int(tx["n_both"])
    m["transcript_assignment_accuracy_via_matched_cells"] = (
        tx["n_correct"] / tx["n_in_matched"] if tx["n_in_matched"] else np.nan)

    m["_roi_ix0"]        = ix0
    m["_roi_iy0"]        = iy0
    m["_match_radius"]   = radius
    m["_gt_assigned_n"]  = int(tx["n_gt_assigned"])
    m["_n_in_matched"]   = int(tx["n_in_matched"])
    m["_n_genes_used"]   = len(tx["gene_list"])
    m["_nuc_mask_cells"] = int(len(nuc_cells))

    # ── 输出 ────────────────────────────────────────────────────────────────
    stem = os.path.join(cfg.output_dir,
                        f"{cfg.method_name.lower()}_merfish_sparse_hungarian_qc")

    with open(f"{stem}_metrics_full.json", "w") as f:
        json.dump({k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                   for k, v in m.items()}, f, indent=2, ensure_ascii=False)

    REQUESTED = [
        "method",
        "Pred cell count", "GT cell count", "Cell count ratio",
        "Matched cell count", "Detection precision", "Detection recall", "Detection F1",
        "Mean centroid shift", "Median centroid shift", "p95 centroid shift",
        "matched_pair_n_transcripts_vs_total_counts_pearson",
        "matched_pair_n_transcripts_vs_total_counts_spearman",
        "matched_pair_n_transcripts_vs_total_counts_mae",
        "matched_pair_n_transcripts_vs_total_counts_rmse",
        "Pred transcript rows", "GT transcript rows",
        "Pred transcript assignment rate", "Transcript matched cell pairs",
        "matched_cell_transcript_count_pearson",
        "matched_cell_transcript_count_spearman",
        "matched_cell_transcript_count_mae",
        "matched_cell_transcript_count_rmse",
        "matched_cell_gene_vector_mean_cosine",
        "matched_cell_gene_vector_mean_js_distance",
        "matched_cell_gene_vector_mean_pearson",
        "matched_cell_gene_vector_valid_pairs",
        "transcript_id_overlap_n",
        "transcript_assignment_accuracy_via_matched_cells",
    ]
    sel = pd.DataFrame([{k: m.get(k, np.nan) for k in REQUESTED}])
    sel.to_csv(f"{stem}_metrics_selected.csv", index=False)

    if not cfg.no_save_matched_table and n_match:
        mt = matches.copy()
        mt["gt_cell_id"]         = gt_cells["cell_id"].values[mt["gt_idx"].astype(int)]
        mt["pred_mask_id"]       = pred_cells["mask_id"].values[mt["pred_idx"].astype(int)]
        mt["pred_n_transcripts"] = tx["pred_ntx"][mt["pred_idx"].astype(int)]
        mt["gt_n_transcripts"]   = tx["gt_ntx"][mt["gt_idx"].astype(int)]
        mt.to_csv(f"{stem}_matched_pairs.csv", index=False)

    log(f"[DONE] → {stem}_metrics_selected.csv")
    print("\n=== SELECTED METRICS ===")
    print(sel.T.to_string(header=False))
    return m


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv",    default=DEF_META_CSV)
    ap.add_argument("--tx-csv",      default=DEF_TX_CSV)
    ap.add_argument("--pred-mask",   default=DEF_PRED_MASK)
    ap.add_argument("--nuclei-mask", default=DEF_NUC_MASK)
    ap.add_argument("--output-dir",  default="/data/qiuyijia/eval_results/ucs_merfish")
    ap.add_argument("--method-name", default="UCS")

    ap.add_argument("--cx-um",   type=float, default=DEF_CX_UM)
    ap.add_argument("--cy-um",   type=float, default=DEF_CY_UM)
    ap.add_argument("--roi-px",  type=int,   default=DEF_ROI_PX)
    ap.add_argument("--roi-ix0", type=int,   default=None)
    ap.add_argument("--roi-iy0", type=int,   default=None)

    ap.add_argument("--match-radius",      type=float, default=None)
    ap.add_argument("--radius-quantile",   type=float, default=0.95)
    ap.add_argument("--radius-multiplier", type=float, default=1.5)
    ap.add_argument("--dense-side",        type=int,   default=2000,
                    help="连通分量边长阈值：不超过则用匈牙利，超过则用贪心")
    ap.add_argument("--map-max-dist",      type=float, default=5.0)

    ap.add_argument("--chunksize", type=int, default=5_000_000)
    ap.add_argument("--keep-blank", action="store_true")
    ap.add_argument("--no-save-matched-table", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())