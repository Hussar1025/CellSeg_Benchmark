#!/usr/bin/env python
"""
ProSeg × Xenium 乳腺 评估（ROI 口径）

★ 为什么必须过滤 ROI：
  ProSeg 跑的是全图 → 190,987 细胞 / 全图 GT 209,467 = 0.912（其实很好）
  但 UCS/BOMS/Cellist 都只跑 5000×5000px ROI（GT 8,809）
  不过滤会得到 ratio=21.68 的假象，误以为要调参

ROI 与其他方法完全一致：
  CX=53994/2×0.2125  CY=27420/2×0.2125  HALF=5000×0.2125/2
  → x[5205.6,6268.1]  y[2382.1,3444.6] µm

ProSeg 输出列名（已确认）：
  cell-metadata.csv.gz      : cell, centroid_x, centroid_y, volume, ...
  transcript-metadata.csv.gz: assignment(float), background(bool), gene, x, y

断点续跑：中间结果（GT/pred/转录本）缓存到 ckpt/，重跑秒级恢复
"""
import os, sys, time, pickle, argparse, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

PROSEG_DIR = "/data/qiuyijia/proseg/output/xenium_2"
DATA_DIR   = "/data/qiuyijia/dataset/xenium_breast"
OUT_CSV    = "/data/qiuyijia/proseg/value/xenium_2.csv"
CKPT_DIR   = "/data/qiuyijia/proseg/value/ckpt_xenium_2"

PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE
CY_UM = 27420 / 2 * PIXEL_SIZE
HALF  = 5000 * PIXEL_SIZE / 2
ROI_X0, ROI_X1 = CX_UM - HALF, CX_UM + HALF
ROI_Y0, ROI_Y1 = CY_UM - HALF, CY_UM + HALF
MATCH_RADIUS = 12.0        # 乳腺细胞直径 ~11-13 µm

os.makedirs(CKPT_DIR, exist_ok=True)
P = lambda *a: print(*a, flush=True)          # 实时输出


def ckpt(name, fn, force=False):
    """通用断点缓存"""
    f = f"{CKPT_DIR}/{name}.pkl"
    if os.path.exists(f) and not force:
        with open(f, "rb") as h:
            obj = pickle.load(h)
        P(f"    ⚡ 复用 {name}.pkl")
        return obj
    t0 = time.time()
    obj = fn()
    with open(f, "wb") as h:
        pickle.dump(obj, h, protocol=4)
    P(f"    ✓ {name} 完成 ({time.time()-t0:.0f}s) → 已缓存")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-image", action="store_true",
                    help="不过滤 ROI，对照全图 GT 209,467")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--vec-sample", type=int, default=3000)
    ap.add_argument("--force", action="store_true", help="忽略缓存重算")
    cfg = ap.parse_args()

    mode = "全图" if cfg.full_image else "ROI"
    P(SEP); P(f"ProSeg × Xenium 乳腺 评估（{mode} 口径）"); P(SEP)
    if not cfg.full_image:
        P(f"  ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm")
    P(f"  匹配半径 {MATCH_RADIUS} µm   断点目录 {CKPT_DIR}")
    sfx = "full" if cfg.full_image else "roi"

    # ── pred ────────────────────────────────────────────────────────────
    P(f"\n[1] ProSeg 细胞")
    def _pred():
        d = pd.read_csv(f"{PROSEG_DIR}/cell-metadata.csv.gz")
        n0 = len(d)
        if not cfg.full_image:
            d = d[(d.centroid_x >= ROI_X0) & (d.centroid_x < ROI_X1) &
                  (d.centroid_y >= ROI_Y0) & (d.centroid_y < ROI_Y1)
                  ].reset_index(drop=True)
        return d, n0
    pred, n_full = ckpt(f"pred_{sfx}", _pred, cfg.force)
    P(f"    全图 {n_full:,} → {mode} 内 {len(pred):,}")

    # ── GT ──────────────────────────────────────────────────────────────
    P(f"\n[2] GT 细胞")
    def _gt():
        g = pd.read_parquet(f"{DATA_DIR}/cells.parquet",
                            columns=["cell_id", "x_centroid", "y_centroid"])
        n0 = len(g)
        if not cfg.full_image:
            g = g[(g.x_centroid >= ROI_X0) & (g.x_centroid < ROI_X1) &
                  (g.y_centroid >= ROI_Y0) & (g.y_centroid < ROI_Y1)
                  ].reset_index(drop=True)
        return g, n0
    gt, n_gt_full = ckpt(f"gt_{sfx}", _gt, cfg.force)
    P(f"    全图 {n_gt_full:,} → {mode} 内 {len(gt):,}")

    # ── 转录本 ───────────────────────────────────────────────────────────
    P(f"\n[3] 转录本（首次约 2-3 分钟）")
    def _txp():
        t = pd.read_csv(f"{PROSEG_DIR}/transcript-metadata.csv.gz")
        t = t[t["background"] == False].copy()
        t["cell_int"] = t["assignment"].astype(int)
        if not cfg.full_image:
            t = t[(t.x >= ROI_X0) & (t.x < ROI_X1) &
                  (t.y >= ROI_Y0) & (t.y < ROI_Y1)]
        return t[["cell_int", "gene", "x", "y"]].reset_index(drop=True)
    tx_pred = ckpt(f"txpred_{sfx}", _txp, cfg.force)
    P(f"    ProSeg 已分配 {len(tx_pred):,}")

    def _txg():
        t = pd.read_parquet(f"{DATA_DIR}/transcripts.parquet",
                            columns=["cell_id", "feature_name",
                                     "x_location", "y_location", "qv"])
        t = t[t.qv >= 20]
        t = t[~t.feature_name.str.startswith(
            ("NegControl", "Unassigned", "BLANK", "antisense"), na=False)]
        if not cfg.full_image:
            t = t[(t.x_location >= ROI_X0) & (t.x_location < ROI_X1) &
                  (t.y_location >= ROI_Y0) & (t.y_location < ROI_Y1)]
        return t.reset_index(drop=True)
    tx_gt = ckpt(f"txgt_{sfx}", _txg, cfg.force)
    P(f"    GT 转录本 {len(tx_gt):,}")

    n_pred, n_gt = len(pred), len(gt)
    if n_pred == 0 or n_gt == 0:
        P("  ✘ 预测或 GT 为空"); return

    # ── 匹配 ─────────────────────────────────────────────────────────────
    P(f"\n[4] 一对一贪心匹配")
    pxy = pred[["centroid_x", "centroid_y"]].values.astype(float)
    gxy = gt[["x_centroid", "y_centroid"]].values.astype(float)
    tree = cKDTree(gxy)
    d, gi = tree.query(pxy, k=1, distance_upper_bound=MATCH_RADIUS)
    best = {}
    for pi in range(n_pred):
        if not np.isfinite(d[pi]) or d[pi] >= MATCH_RADIUS:
            continue
        g = int(gi[pi])
        if g not in best or d[pi] < best[g][1]:
            best[g] = (pi, d[pi])
    pairs = [(pi, g, x) for g, (pi, x) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred
    rec  = n_match / n_gt
    f1   = 2*prec*rec/(prec+rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    P(f"    匹配 {n_match:,}   P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    P(f"    偏移 中位 {np.nanmedian(shifts):.2f}µm  "
      f"p95 {np.nanpercentile(shifts,95):.2f}µm")

    # ── 计数一致性 ───────────────────────────────────────────────────────
    P(f"\n[5] 计数一致性")
    pred_cnt = tx_pred.groupby("cell_int").size()
    gt_cnt = tx_gt[tx_gt.cell_id.notna() &
                   (tx_gt.cell_id != "UNASSIGNED")].cell_id.value_counts()
    pc = np.array([pred_cnt.get(int(pred.cell.iloc[p[0]]), 0)
                   for p in pairs], float)
    gc = np.array([gt_cnt.get(gt.cell_id.iloc[p[1]], 0) for p in pairs], float)
    m = (pc > 0) & (gc > 0)
    if m.sum() > 10:
        cp = float(pearsonr(pc[m], gc[m])[0])
        cs = float(spearmanr(pc[m], gc[m])[0])
        cm = float(np.mean(np.abs(pc[m] - gc[m])))
        cr = float(np.sqrt(np.mean((pc[m] - gc[m])**2)))
    else:
        cp = cs = cm = cr = np.nan
    P(f"    Pearson {cp:.4f}  Spearman {cs:.4f}  有效对 {int(m.sum()):,}")

    # ── 基因向量（抽样）──────────────────────────────────────────────────
    P(f"\n[6] 基因向量（抽样 {cfg.vec_sample}）")
    rng = np.random.default_rng(0)
    samp = pairs if len(pairs) <= cfg.vec_sample else \
        [pairs[i] for i in rng.choice(len(pairs), cfg.vec_sample, replace=False)]
    sp = {int(pred.cell.iloc[p[0]]) for p in samp}
    sg = {gt.cell_id.iloc[p[1]] for p in samp}
    pv = (tx_pred[tx_pred.cell_int.isin(sp)]
            .groupby(["cell_int", "gene"]).size().unstack(fill_value=0))
    gv = (tx_gt[tx_gt.cell_id.isin(sg)]
            .groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0))
    common = sorted(set(pv.columns) & set(gv.columns))
    vc = vj = vp = np.nan
    n_valid = 0
    if len(common) >= 50:
        pv, gv = pv[common], gv[common]
        cos_l, js_l, pr_l = [], [], []
        for pi, gj, _ in samp:
            pid = int(pred.cell.iloc[pi]); gid = gt.cell_id.iloc[gj]
            if pid not in pv.index or gid not in gv.index:
                continue
            a = pv.loc[pid].values.astype(float)
            b = gv.loc[gid].values.astype(float)
            if a.sum() < 5 or b.sum() < 5:
                continue
            cos_l.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))))
            js_l.append(float(jensenshannon(a/a.sum(), b/b.sum())))
            if a.std() > 0 and b.std() > 0:
                pr_l.append(float(pearsonr(a, b)[0]))
        n_valid = len(cos_l)
        if n_valid > 10:
            vc = float(np.mean(cos_l))
            vj = float(np.nanmean(js_l))
            vp = float(np.mean(pr_l)) if pr_l else np.nan
    P(f"    共同基因 {len(common)}  有效对 {n_valid:,}  cosine {vc:.4f}")

    # ── 分配一致性（1µm 精度坐标对齐）────────────────────────────────────
    P(f"\n[7] 分配一致性")
    p2g = {int(pred.cell.iloc[p[0]]): gt.cell_id.iloc[p[1]] for p in pairs}
    kp = (tx_pred.x.round(0).astype(np.int64) * 100000 +
          tx_pred.y.round(0).astype(np.int64))
    kg = (tx_gt.x_location.round(0).astype(np.int64) * 100000 +
          tx_gt.y_location.round(0).astype(np.int64))
    gmap = dict(zip(kg.values, tx_gt.cell_id.values))
    both = correct = 0
    for k, pid in zip(kp.values, tx_pred.cell_int.values):
        g_ = gmap.get(k)
        if g_ is None or pd.isna(g_) or g_ == "UNASSIGNED":
            continue
        e = p2g.get(int(pid))
        if e is None:
            continue
        both += 1
        if e == g_:
            correct += 1
    overlap = both / max(len(tx_pred), 1)
    acc = correct / both if both else np.nan
    P(f"    双侧有归属 {both:,}  一致 {correct:,}  accuracy {acc:.4f}")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="ProSeg", dataset=f"xenium_breast_{mode}",
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred/n_gt), matched=n_match,
        precision=rnd(prec), recall=rnd(rec), f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(shifts)),
        loc_median_um=rnd(np.nanmedian(shifts)),
        loc_p95_um=rnd(np.nanpercentile(shifts, 95)),
        count_pearson=rnd(cp), count_spearman=rnd(cs),
        count_mae=rnd(cm, 2), count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc), vec_js_dist=rnd(vj), vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap), assign_accuracy=rnd(acc),
    )
    P(f"\n{SEP}\n结果 (ProSeg × Xenium 乳腺 {mode})\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            P(f"  {k:22} {v}")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    pd.DataFrame([res]).to_csv(cfg.out, index=False)
    P(f"\n  已写 → {cfg.out}")
    if not cfg.full_image:
        P(f"  全图口径对照: 加 --full-image（GT {n_gt_full:,}）")


if __name__ == "__main__":
    main()