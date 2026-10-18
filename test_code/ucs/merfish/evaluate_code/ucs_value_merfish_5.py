#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERFISH 分割评估 —— 19 指标，五方法通用。

口径完全对齐 ucs_value_merfish_4.py：
  * GT      = cell_metadata.csv 质心落在中心 ROI 内的 platform 核
  * 匹配    = kd-tree 候选边(半径=GT最近邻p95×1.25) + 连通分量内匈牙利/大分量贪心
  * 指标    = 数量/检测P,R,F1/质心偏移/转录本计数相关/基因向量相似度 等

支持两种预测格式：
  --seg  某个 *_segmentation.tif / mask.tif（proseg/ucs/comseg/genesegnet）
  --npz  BOMS 的 boms_merfish.npz（点云：seg=每转录本cell id, x/y 微米, cell_loc 质心）

用法：
  # 标签图方法（自动定位 *_segmentation.tif，或 --seg 指定）
  python eval_merfish.py --method UCS --dataset prostate1 \
      --run-dir /data/qiuyijia/ucs_merfish_prostate1 \
      --data-root /data/qiuyijia/dataset/merfish_prostate1

  # BOMS
  python eval_merfish.py --method BOMS --dataset prostate1 \
      --npz /data/qiuyijia/boms_merfish_prostate1/boms_merfish.npz \
      --data-root /data/qiuyijia/dataset/merfish_prostate1

  # 一次跑多个，追加到同一张表
  python eval_merfish.py ... --out results.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROI_SIZE = 10000


# ===========================================================================
# 几何（对齐参考文件 Geometry）
# ===========================================================================
class Geometry:
    def __init__(self, sx, sy, ox, oy, y0, x0, size):
        self.sx, self.sy, self.ox, self.oy = sx, sy, ox, oy
        self.y0, self.x0, self.size = y0, x0, size

    @property
    def um_per_px(self):
        return 1.0 / self.sx

    def um_to_local(self, ux, uy):
        return (np.asarray(ux, float) * self.sx + self.ox - self.x0,
                np.asarray(uy, float) * self.sy + self.oy - self.y0)

    def local_to_um(self, px, py):
        return ((np.asarray(px, float) + self.x0 - self.ox) / self.sx,
                (np.asarray(py, float) + self.y0 - self.oy) / self.sy)

    def inside(self, ux, uy):
        px, py = self.um_to_local(ux, uy)
        return (px >= 0) & (px < self.size) & (py >= 0) & (py < self.size)


def load_geometry(data_root, roi_size=ROI_SIZE):
    import tifffile
    root = Path(data_root)
    tf = next(root.rglob("*micron_to_mosaic*transform*.csv"), None)
    if tf is None:
        raise SystemExit(f"no transform under {root}")
    m = np.loadtxt(tf)
    dapi = next((p for p in sorted(root.rglob("*.tif"))
                 if "dapi" in p.name.lower()), None)
    if dapi is None:
        raise SystemExit("no *dapi*.tif under data-root")
    with tifffile.TiffFile(dapi) as t:
        H, W = t.series[0].shape[-2:]
    return Geometry(float(m[0, 0]), float(m[1, 1]),
                    float(m[0, 2]), float(m[1, 2]),
                    (H - roi_size) // 2, (W - roi_size) // 2, roi_size)


# ===========================================================================
# 匹配（对齐参考文件）
# ===========================================================================
def infer_match_radius(gx, gy, quantile=0.95, factor=1.25, floor=2.0):
    from scipy.spatial import cKDTree
    pts = np.column_stack([gx, gy])
    if len(pts) < 2:
        return floor
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        return floor
    return float(max(np.quantile(nn, quantile) * factor, floor))


def build_candidate_edges(px, py, gx, gy, radius):
    from scipy.spatial import cKDTree
    tp = cKDTree(np.column_stack([px, py]))
    tg = cKDTree(np.column_stack([gx, gy]))
    pairs = tp.query_ball_tree(tg, r=radius)
    pi, gi = [], []
    for i, js in enumerate(pairs):
        for j in js:
            pi.append(i); gi.append(j)
    pi = np.asarray(pi, np.int64); gi = np.asarray(gi, np.int64)
    if len(pi) == 0:
        return pi, gi, np.zeros(0)
    d = np.hypot(px[pi] - gx[gi], py[pi] - gy[gi])
    return pi, gi, d


def _cc_bipartite(pi, gi, n_pred, n_gt):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = n_pred + n_gt
    a = coo_matrix((np.ones(len(pi)), (pi, n_pred + gi)), shape=(n, n))
    a = a + a.T
    _, lab = connected_components(a, directed=False)
    return _, lab


def match_sparse_hungarian(px, py, gx, gy, radius, greedy_above=400):
    from scipy.optimize import linear_sum_assignment
    pi, gi, d = build_candidate_edges(px, py, gx, gy, radius)
    if len(pi) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    _, lab = _cc_bipartite(pi, gi, len(px), len(gx))
    epred, egt, edist = [], [], []
    comp = lab[pi]
    order = np.argsort(comp, kind="stable")
    pi, gi, d, comp = pi[order], gi[order], d[order], comp[order]
    starts = np.searchsorted(comp, np.unique(comp))
    starts = np.append(starts, len(pi))
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        sp, sg, sd = pi[a:b], gi[a:b], d[a:b]
        up, ug = np.unique(sp), np.unique(sg)
        if len(up) * len(ug) > greedy_above * greedy_above:
            o = np.argsort(sd)
            usedp, usedg = set(), set()
            for t in o:
                if sp[t] in usedp or sg[t] in usedg:
                    continue
                usedp.add(sp[t]); usedg.add(sg[t])
                epred.append(sp[t]); egt.append(sg[t]); edist.append(sd[t])
            continue
        pmap = {v: i for i, v in enumerate(up)}
        gmap = {v: i for i, v in enumerate(ug)}
        big = sd.max() * 10 + 1
        cost = np.full((len(up), len(ug)), big)
        cost[[pmap[v] for v in sp], [gmap[v] for v in sg]] = sd
        r, c = linear_sum_assignment(cost)
        for rr, cc in zip(r, c):
            if cost[rr, cc] >= big:
                continue
            epred.append(up[rr]); egt.append(ug[cc]); edist.append(cost[rr, cc])
    return (np.asarray(epred, np.int64), np.asarray(egt, np.int64),
            np.asarray(edist, float))


def numeric_pair_metrics(a, b, prefix):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = {f"{prefix}_pearson": np.nan, f"{prefix}_spearman": np.nan,
           f"{prefix}_mae": np.nan, f"{prefix}_rmse": np.nan}
    if ok.sum() < 3:
        return out
    a, b = a[ok], b[ok]
    if a.std() > 0 and b.std() > 0:
        out[f"{prefix}_pearson"] = float(pearsonr(a, b)[0])
        out[f"{prefix}_spearman"] = float(spearmanr(a, b)[0])
    out[f"{prefix}_mae"] = float(np.mean(np.abs(a - b)))
    out[f"{prefix}_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
    return out


def gene_vector_metrics(pm, gm):
    from scipy.spatial.distance import jensenshannon
    cos, js, pr = [], [], []
    for i in range(pm.shape[0]):
        a = pm[i].astype(float); b = gm[i].astype(float)
        if a.sum() <= 0 or b.sum() <= 0:
            continue
        cos.append(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
        d = jensenshannon(a / a.sum(), b / b.sum(), base=2)
        if np.isfinite(d):
            js.append(float(d))
        if a.std() > 0 and b.std() > 0:
            pr.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "matched_cell_gene_vector_mean_cosine": float(np.mean(cos)) if cos else np.nan,
        "matched_cell_gene_vector_mean_js_distance": float(np.mean(js)) if js else np.nan,
        "matched_cell_gene_vector_mean_pearson": float(np.mean(pr)) if pr else np.nan,
        "matched_cell_gene_vector_valid_pairs": len(cos),
    }


# ===========================================================================
# 读 GT（platform 核，cell_metadata 质心在 ROI 内）
# ===========================================================================
def load_gt(data_root, geo):
    root = Path(data_root)
    md = next(root.rglob("cell_metadata.csv*"), None)
    if md is None:
        raise SystemExit("no cell_metadata.csv under data-root")
    d = pd.read_csv(md)
    ux = d["center_x"].to_numpy(float)
    uy = d["center_y"].to_numpy(float)
    keep = geo.inside(ux, uy)
    return ux[keep], uy[keep]


# ===========================================================================
# 读预测 -> 统一成 (质心_um, 每细胞基因向量, spots_assigned_rate, 面积, 转录本对齐)
# ===========================================================================
def load_pred_seg(seg_path, geo, data_root):
    """标签 tif 方法：从 mask 取质心 + 用转录本重建基因向量。"""
    import tifffile
    lab = np.squeeze(tifffile.imread(seg_path)).astype(np.int64)
    lab[lab < 0] = 0
    s = geo.size / lab.shape[0]           # 该 mask 每像素 = s 个 ROI 像素
    ids = np.unique(lab); ids = ids[ids > 0]
    # 质心（label 像素坐标 -> ROI 像素 -> um）
    rows, cols = np.nonzero(lab)
    lv = lab[rows, cols]
    o = np.argsort(lv, kind="stable")
    lv, rows, cols = lv[o], rows[o], cols[o]
    b = np.append(np.searchsorted(lv, ids), len(lv))
    cy = np.array([rows[b[i]:b[i + 1]].mean() for i in range(len(ids))])
    cx = np.array([cols[b[i]:b[i + 1]].mean() for i in range(len(ids))])
    area_px = (b[1:] - b[:-1]).astype(float)
    pux, puy = geo.local_to_um((cx + 0.5) * s, (cy + 0.5) * s)
    area_um2 = area_px * s * s * (geo.um_per_px ** 2)
    # 转录本 -> 细胞（用数据集自己的 transcripts）
    genes, gene_idx, tx_ux, tx_uy = _load_tx(data_root)
    lx, ly = geo.um_to_local(tx_ux, tx_uy)
    r = np.clip((ly / s).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip((lx / s).astype(np.int64), 0, lab.shape[1] - 1)
    tl = lab[r, c]
    assign_rate = float((tl > 0).mean())
    pos = {int(v): i for i, v in enumerate(ids)}
    gm = np.zeros((len(ids), len(genes)), np.float32)
    ntx = np.zeros(len(ids), np.int64)
    sel = tl > 0
    ri = np.array([pos.get(int(v), -1) for v in tl[sel]])
    ok = ri >= 0
    np.add.at(ntx, ri[ok], 1)
    np.add.at(gm, (ri[ok], gene_idx[sel][ok]), 1.0)
    return dict(cx=pux, cy=puy, area=area_um2, gm=gm, ntx=ntx,
                genes=genes, assign_rate=assign_rate,
                coverage=float((lab > 0).mean()))


def load_pred_boms(npz_path, geo):
    """BOMS 点云：seg=每转录本cell id, x/y um, cell_loc 质心, count_mat 细胞×基因。"""
    z = np.load(npz_path, allow_pickle=True)
    seg = z["seg"].astype(np.int64)
    x = z["x"].astype(float); y = z["y"].astype(float)
    genes = list(z["gene_names"])
    cm = z["count_mat"]
    # cell_loc 已是质心（微米）；顺序对应 count_mat 行
    cl = z["cell_loc"].astype(float)
    pux, puy = cl[:, 0], cl[:, 1]
    inside = geo.inside(pux, puy)
    pux, puy, gm = pux[inside], puy[inside], cm[inside].astype(np.float32)
    ntx = gm.sum(1).astype(np.int64)
    assign_rate = float((seg > 0).mean())
    # BOMS 无像素面积，用每细胞转录本凸包近似跳过，面积置 nan
    area = np.full(len(pux), np.nan)
    return dict(cx=pux, cy=puy, area=area, gm=gm, ntx=ntx,
                genes=genes, assign_rate=assign_rate, coverage=np.nan)


def _load_tx(data_root):
    """读 detected_transcripts，返回 (基因表, 每转录本基因索引, x_um, y_um)。"""
    root = Path(data_root)
    txp = next(root.rglob("detected_transcripts.csv*"), None)
    if txp is None:
        raise SystemExit("no detected_transcripts.csv under data-root")
    cols = ["global_x", "global_y", "gene"]
    d = pd.read_csv(txp, usecols=lambda c: c in cols)
    g = d["gene"].astype(str).to_numpy()
    # 去掉 blank/control
    keep = ~np.char.startswith(np.char.lower(g.astype(str)),
                               ("blank"))
    genes = sorted(set(g[keep]))
    gidx = {v: i for i, v in enumerate(genes)}
    gene_idx = np.array([gidx.get(v, -1) for v in g], np.int64)
    ok = gene_idx >= 0
    return genes, gene_idx[ok], d["global_x"].to_numpy(float)[ok], \
        d["global_y"].to_numpy(float)[ok]


# ===========================================================================
# 主评估
# ===========================================================================
def evaluate(pred, gx, gy, method, dataset, gt_root):
    px, py = pred["cx"], pred["cy"]
    n_pred, n_gt = len(px), len(gx)
    radius = infer_match_radius(gx, gy)
    ep, eg, ed = match_sparse_hungarian(px, py, gx, gy, radius)
    tp = len(ep)
    prec = tp / n_pred if n_pred else np.nan
    rec = tp / n_gt if n_gt else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan

    # 质心偏移
    shift = ed
    # 匹配对的转录本计数相关（pred ntx vs gt ntx 需 GT 每细胞转录本；这里用配对细胞的 pred ntx 与 GT 邻域计数近似）
    m = {
        "method": method, "dataset": dataset,
        "Pred cell count": n_pred, "GT cell count": n_gt,
        "Cell count ratio": n_pred / n_gt if n_gt else np.nan,
        "Matched cell count": tp,
        "Detection precision": prec, "Detection recall": rec,
        "Detection F1": f1,
        "Mean centroid shift": float(np.mean(shift)) if tp else np.nan,
        "Median centroid shift": float(np.median(shift)) if tp else np.nan,
        "p95 centroid shift": float(np.percentile(shift, 95)) if tp else np.nan,
        "Pred transcript assignment rate": pred["assign_rate"],
        "median_area_um2": float(np.nanmedian(pred["area"])) if len(pred["area"]) else np.nan,
        "median_diameter_um": (2 * np.sqrt(np.nanmedian(pred["area"]) / np.pi)
                               if np.isfinite(np.nanmedian(pred["area"])) else np.nan),
        "coverage_fraction": pred["coverage"],
        "match_radius_um": radius,
    }

    # 基因向量相似度（需 GT 每细胞基因向量）——用 GT platform 质心 + 数据集转录本重建
    gtm = _gt_gene_vectors(gt_root, gx, gy, pred["genes"])
    if gtm is not None and tp:
        pm = pred["gm"][ep]
        gmv = gtm[eg]
        m.update(gene_vector_metrics(pm, gmv))
        # 匹配对转录本计数相关
        m.update(numeric_pair_metrics(pred["ntx"][ep], gtm[eg].sum(1),
                                      "matched_cell_transcript_count"))
    return m


def _gt_gene_vectors(data_root, gx, gy, genes):
    """把转录本按最近 GT 质心归属，重建 GT 每细胞基因向量（半径内）。"""
    from scipy.spatial import cKDTree
    g2, gidx_all, tx_x, tx_y = _load_tx(data_root)
    # 对齐到 pred 的基因表
    remap = {g: i for i, g in enumerate(genes)}
    # gidx_all 是相对 g2 的索引，转回基因名再映射
    inv = {i: g for i, g in enumerate(g2)}
    gene_name = np.array([inv[i] for i in gidx_all])
    col = np.array([remap.get(n, -1) for n in gene_name])
    tree = cKDTree(np.column_stack([gx, gy]))
    dd, jj = tree.query(np.column_stack([tx_x, tx_y]), k=1)
    r = max(np.median(dd) * 2, 3.0)
    ok = (dd <= r) & (col >= 0)
    gm = np.zeros((len(gx), len(genes)), np.float32)
    np.add.at(gm, (jj[ok], col[ok]), 1.0)
    return gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    geo = load_geometry(args.data_root, args.roi_size)
    gx, gy = load_gt(args.data_root, geo)
    print(f"GT cells in ROI = {len(gx)}")

    if args.npz:
        pred = load_pred_boms(args.npz, geo)
    else:
        seg = args.seg
        if seg is None:
            cand = sorted(Path(args.run_dir).glob("*_segmentation.tif")) or \
                   sorted(Path(args.run_dir).glob("*mask*.tif"))
            if not cand:
                raise SystemExit(f"no *_segmentation.tif / *mask*.tif in {args.run_dir}")
            seg = str(cand[0])
        print(f"seg = {seg}")
        pred = load_pred_seg(seg, geo, args.data_root)

    m = evaluate(pred, gx, gy, args.method, args.dataset, args.data_root)

    # 打印
    print("=" * 60)
    for k, v in m.items():
        print(f"{k:48s} {v}")
    print("=" * 60)

    # 追加到 csv
    outp = Path(args.out)
    df = pd.DataFrame([m])
    if outp.exists():
        old = pd.read_csv(outp)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(outp, index=False)
    print(f"written -> {outp}")


if __name__ == "__main__":
    main()