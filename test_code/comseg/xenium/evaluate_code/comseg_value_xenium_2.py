#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_value_xenium_2.py — 评估 ComSeg @ xenium_2 (Xenium breast, ROI5000)

★ v2 修复坐标系错误
  spot_assignment 里的 global_x/global_y 是 ROI-local 像素 (x[0.1,5000.0])，
  而 cells.parquet 的 x_centroid/y_centroid 是全片微米。v1 直接拿前者的范围
  去框后者，框出了全片 74,308 个 GT (应为 8,393)，匹配半径被撑到 20 um，
  于是 count_pearson=-0.05、vec_cosine=0.31 等全部是伪结果。
  换算 (prep_meta.json):  um = (local_px + origin) * pixel_size
                          x0=24497  y0=11210  pixel_size=0.2125

评估口径与 comseg_value_xenium_4.py 一致:
  质心从 spot_assignment 按 cell 分组重算，不用 h5ad 的 obs.centroid_x/y ——
  后者继承自 nucleus prior 的 landmark(uns 里 _landmark_for_centroid 是证据)，
  会让 loc_* 恒为 0、precision 恒为 1。
  匹配走 KD-tree 最近邻 + 半径截断 + 一对一去重，无任何 ID 直连。
  半径 = sqrt(ROI面积 / ROI内GT细胞数)。

指标: 检测(P/R/F1, cell_ratio) 定位(loc mean/median/p95)
      计数(pearson/spearman/mae/rmse) 表达向量(cosine/JS/pearson)
      归属(assign_accuracy, overall_correct, coverage_fraction)
      形态(median_diameter_um) 分布(tx/cell 分位, <10 占比)

用法:
  python3 comseg_value_xenium_2.py
  python3 comseg_value_xenium_2.py --no-dedup
  python3 comseg_value_xenium_2.py --skip-tx     # 跳过表达向量(省内存)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/data/qiuyijia/comseg_xenium_breast_roi5000"
GT_CELLS = "/data/qiuyijia/dataset/xenium_breast/cells.parquet"
GT_TX = "/data/qiuyijia/dataset/xenium_breast/transcripts.parquet"
OUT = "/data/qiuyijia/eval_results_qv20/breast"
METHOD, DATASET = "ComSeg", "xenium_2"


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--gt-cells", default=GT_CELLS)
    ap.add_argument("--gt-tx", default=GT_TX)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=0.0)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--skip-tx", action="store_true")
    a = ap.parse_args()

    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr, spearmanr

    log("=" * 78)
    log(f"{METHOD} @ {DATASET}  (Xenium breast, ROI 5000 px)")
    log("=" * 78)

    # ── 坐标换算参数 ────────────────────────────────────────────────
    mp = os.path.join(a.root, "input", "prep_meta.json")
    if not os.path.exists(mp):
        sys.exit(f"✗ 缺 {mp}，无法确定 ROI 原点与像素尺寸")
    meta = json.load(open(mp))
    X0, Y0 = float(meta["x0"]), float(meta["y0"])
    PX = float(meta["pixel_size"])
    log(f"  ROI 原点 (x0={X0:.0f}, y0={Y0:.0f}) px   pixel_size {PX} um/px")
    log(f"  ROI {meta['size_x']} x {meta['size_y']} px = "
        f"{meta['size_x']*PX:.1f} x {meta['size_y']*PX:.1f} um")
    log(f"  prep: {meta['n_transcripts']:,} tx / GT {meta['gt_expected_cells']:,}"
        f" = {meta['n_transcripts']/meta['gt_expected_cells']:.1f} tx/cell"
        f"   qv_min={meta.get('qv_min')}")

    # ── 预测 ────────────────────────────────────────────────────────
    fs = sorted(glob.glob(os.path.join(
        a.root, "comseg_out", "tile_*", "spot_assignment.parquet")))
    if not fs:
        sys.exit(f"✗ 无 spot_assignment.parquet")
    log(f"\n【预测】{len(fs)} 个 tile")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    log(f"  合并 {len(d):,} 行   列: {list(d.columns)}")
    if not a.no_dedup:
        n0 = len(d)
        d = d.drop_duplicates(subset=["global_x", "global_y", "gene"],
                              keep="first")
        log(f"  tile overlap 去重: {n0:,} -> {len(d):,} "
            f"(删 {(n0-len(d))/n0*100:.1f}%)")

    # ROI-local px -> 全片 um
    d["_xum"] = (d["global_x"].astype(float) + X0) * PX
    d["_yum"] = (d["global_y"].astype(float) + Y0) * PX
    log(f"  local px  x[{d.global_x.min():.1f},{d.global_x.max():.1f}]")
    log(f"  全片 um   x[{d._xum.min():.1f},{d._xum.max():.1f}] "
        f"y[{d._yum.min():.1f},{d._yum.max():.1f}]")

    assigned = pd.to_numeric(d["cell"], errors="coerce") > 0
    log(f"  已分配 {int(assigned.sum()):,} ({assigned.mean()*100:.1f}%)")

    da = d[assigned].copy()
    g = da.groupby("cell").agg(x=("_xum", "mean"), y=("_yum", "mean"),
                               n=("_xum", "size"))
    q = g.n.quantile([.05, .25, .5, .75, .95])
    log(f"\n  预测细胞 {len(g):,}")
    log(f"  每细胞 tx  中位 {g.n.median():.0f}  均值 {g.n.mean():.0f}  "
        f"均/中 {g.n.mean()/max(g.n.median(),1):.2f}")
    log(f"             p5/p25/p75/p95 = {q.iloc[0]:.0f}/{q.iloc[1]:.0f}/"
        f"{q.iloc[3]:.0f}/{q.iloc[4]:.0f}   <10 的 {(g.n<10).mean()*100:.2f}%")

    sp = da.groupby("cell").agg(sx=("_xum", "std"), sy=("_yum", "std"))
    diam = 2 * np.sqrt(2) * np.sqrt((sp.sx ** 2 + sp.sy ** 2) / 2)
    log(f"  等效直径(2*sqrt2*RMS散布) 中位 {diam.median():.2f} um")

    # ── GT: 用 ROI 的真实边界裁，不用预测坐标的跨度 ─────────────────
    gt_all = pd.read_parquet(a.gt_cells)
    x0u, x1u = X0 * PX, (X0 + meta["size_x"]) * PX
    y0u, y1u = Y0 * PX, (Y0 + meta["size_y"]) * PX
    m = ((gt_all.x_centroid >= x0u) & (gt_all.x_centroid < x1u) &
         (gt_all.y_centroid >= y0u) & (gt_all.y_centroid < y1u))
    gt = gt_all[m].reset_index(drop=True)
    log(f"\n【GT】ROI x[{x0u:.1f},{x1u:.1f}] y[{y0u:.1f},{y1u:.1f}] um")
    log(f"  ROI 内 {len(gt):,}  (全片 {len(gt_all):,}；"
        f"prep 期望 {meta['gt_expected_cells']:,})")
    if abs(len(gt) - meta["gt_expected_cells"]) > .1 * meta["gt_expected_cells"]:
        log(f"  ⚠ 与 prep 期望差异 >10%，核对 ROI 换算")

    gtn = None
    for c in ("transcript_counts", "n_counts", "total_counts"):
        if c in gt.columns:
            gtn = gt[c].to_numpy(float)
            log(f"  GT 每细胞 tx 用列 {c}  中位 {np.median(gtn):.0f}")
            break

    # ── 匹配 ────────────────────────────────────────────────────────
    r = a.match_radius or float(np.sqrt((x1u - x0u) * (y1u - y0u) /
                                        max(len(gt), 1)))
    log(f"\n【匹配】KD-tree 一对一，半径 {r:.2f} um")
    P = np.c_[g.x.to_numpy(float), g.y.to_numpy(float)]
    G = gt[["x_centroid", "y_centroid"]].to_numpy(float)
    dist, gi = cKDTree(G).query(P, k=1, distance_upper_bound=r)
    ok = np.isfinite(dist) & (dist < r)
    order = np.argsort(np.where(ok, dist, np.inf))
    used, pairs = set(), []
    for pi in order:
        if not ok[pi]:
            break
        if gi[pi] in used:
            continue
        used.add(gi[pi])
        pairs.append((int(pi), int(gi[pi]), float(dist[pi])))
    matched = len(pairs)
    prec = matched / max(len(P), 1)
    rec = matched / max(len(G), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    dd = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    log(f"  pred {len(P):,}  gt {len(G):,}  matched {matched:,}")
    log(f"  precision {prec:.4f}  recall {rec:.4f}  f1 {f1:.4f}")
    log(f"  loc mean {np.nanmean(dd):.3f}  median {np.nanmedian(dd):.3f}  "
        f"p95 {np.nanpercentile(dd,95):.3f} um")

    det_ok = not (np.nanmedian(dd) < 1e-6)
    if not det_ok:
        log("  ★ loc_median = 0 -> 质心仍来自 prior landmark")

    res = dict(method=METHOD, dataset=DATASET,
               gt_cells=len(G), pred_cells=len(P), matched=matched,
               cell_ratio=len(P) / max(len(G), 1),
               precision=prec, recall=rec, f1=f1,
               loc_mean_um=float(np.nanmean(dd)),
               loc_median_um=float(np.nanmedian(dd)),
               loc_p95_um=float(np.nanpercentile(dd, 95)),
               match_radius_um=r,
               assign_rate=float(assigned.mean()),
               coverage_fraction=float(assigned.mean()),
               tx_per_cell_median=float(g.n.median()),
               tx_per_cell_mean=float(g.n.mean()),
               frag_lt10=float((g.n < 10).mean()),
               median_diameter_um=float(diam.median()),
               qv_min=meta.get("qv_min", 20.0),
               roi_um=f"x[{x0u:.1f},{x1u:.1f}] y[{y0u:.1f},{y1u:.1f}]",
               uses_platform_prior=False,
               detection_comparable=det_ok, data_error=False)

    # ── 计数 ────────────────────────────────────────────────────────
    if pairs and gtn is not None:
        pi = np.array([p[0] for p in pairs])
        gj = np.array([p[1] for p in pairs])
        A = g.n.to_numpy(float)[pi]
        B = gtn[gj]
        k = np.isfinite(A) & np.isfinite(B)
        if k.sum() >= 5:
            res.update(count_pearson=float(pearsonr(A[k], B[k])[0]),
                       count_spearman=float(spearmanr(A[k], B[k])[0]),
                       count_mae=float(np.abs(A[k] - B[k]).mean()),
                       count_rmse=float(np.sqrt(((A[k]-B[k])**2).mean())))
            log(f"\n【计数】n={int(k.sum()):,}  "
                f"pearson {res['count_pearson']:.4f}  "
                f"spearman {res['count_spearman']:.4f}  "
                f"mae {res['count_mae']:.1f}  rmse {res['count_rmse']:.1f}")

    # ── 归属 + 表达向量 ────────────────────────────────────────────
    if pairs and not a.skip_tx and os.path.exists(a.gt_tx):
        log(f"\n【归属 / 表达向量】读 GT 转录本表 ...")
        try:
            tx = pd.read_parquet(a.gt_tx, columns=[
                "x_location", "y_location", "feature_name", "cell_id", "qv"])
        except Exception:
            tx = pd.read_parquet(a.gt_tx)
        tx = tx[(tx.x_location >= x0u) & (tx.x_location < x1u) &
                (tx.y_location >= y0u) & (tx.y_location < y1u)].copy()
        if "qv" in tx.columns:
            tx = tx[tx.qv >= res["qv_min"]]
        log(f"  ROI 内 GT 转录本 {len(tx):,}  "
            f"(prep 记录 {meta['n_transcripts']:,})")

        id2row = {v: i for i, v in
                  enumerate(gt["cell_id"].astype(str).to_numpy())}
        tx["_gtrow"] = (tx["cell_id"].astype(str).map(id2row)
                        .fillna(-1).astype(np.int64))

        # 预测转录本 -> 最近的 GT 转录本，容差 0.5 um
        dq, iq = cKDTree(np.c_[tx.x_location.to_numpy(float),
                               tx.y_location.to_numpy(float)]).query(
            np.c_[da._xum.to_numpy(float), da._yum.to_numpy(float)],
            k=1, distance_upper_bound=0.5)
        hit = np.isfinite(dq)
        log(f"  预测转录本对上 GT 转录本 {int(hit.sum()):,} / {len(da):,} "
            f"({hit.mean()*100:.1f}%)")
        if hit.mean() < .5:
            log("  ⚠ 命中率 <50%，坐标换算可能仍有偏移")

        cell2row = {c: i for i, c in enumerate(g.index)}
        pred_row = np.array([cell2row.get(c, -1)
                             for c in da["cell"].to_numpy()])
        gt_row = np.full(len(da), -1, dtype=np.int64)
        gt_row[hit] = tx["_gtrow"].to_numpy()[iq[hit].astype(np.int64)]
        p2g = {p[0]: p[1] for p in pairs}
        valid = hit & (pred_row >= 0) & (gt_row >= 0)
        if valid.sum() >= 100:
            want = np.array([p2g.get(int(pr), -2) for pr in pred_row[valid]])
            acc = float((want == gt_row[valid]).mean())
            res["assign_accuracy"] = acc
            res["overall_correct"] = acc * float(assigned.mean())
            log(f"  assign_accuracy {acc:.4f}  "
                f"(在 {int(valid.sum()):,} 条可判定转录本上)")
            log(f"  overall_correct  {res['overall_correct']:.4f}")

        if "feature_name" in tx.columns:
            from scipy.sparse import coo_matrix
            genes = sorted(set(da["gene"].astype(str)) |
                           set(tx["feature_name"].astype(str)))
            gidx = {v: i for i, v in enumerate(genes)}
            vp = pred_row >= 0
            PV = coo_matrix(
                (np.ones(int(vp.sum())),
                 (pred_row[vp],
                  da["gene"].astype(str).map(gidx).to_numpy()[vp])),
                shape=(len(g), len(genes))).tocsr()
            vg = tx["_gtrow"].to_numpy() >= 0
            GV = coo_matrix(
                (np.ones(int(vg.sum())),
                 (tx["_gtrow"].to_numpy()[vg],
                  tx["feature_name"].astype(str).map(gidx).to_numpy()[vg])),
                shape=(len(gt), len(genes))).tocsr()
            pi = np.array([p[0] for p in pairs])
            gj = np.array([p[1] for p in pairs])
            A = np.asarray(PV[pi].todense(), float)
            B = np.asarray(GV[gj].todense(), float)
            na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
            k = (na > 0) & (nb > 0)
            if k.sum() >= 5:
                res["vec_cosine"] = float(np.mean(
                    (A[k] * B[k]).sum(1) / (na[k] * nb[k])))
                Ap = A[k] / A[k].sum(1, keepdims=True)
                Bp = B[k] / B[k].sum(1, keepdims=True)
                M = (Ap + Bp) / 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    kl1 = np.nansum(Ap * np.log2(Ap / M), axis=1)
                    kl2 = np.nansum(Bp * np.log2(Bp / M), axis=1)
                res["vec_js_dist"] = float(np.mean(np.sqrt((kl1+kl2)/2)))
                nn = min(2000, int(k.sum()))
                res["vec_pearson"] = float(np.mean(
                    [pearsonr(A[k][i], B[k][i])[0] for i in range(nn)]))
                log(f"  vec_cosine {res['vec_cosine']:.4f}  "
                    f"vec_js_dist {res['vec_js_dist']:.4f}  "
                    f"vec_pearson {res['vec_pearson']:.4f}  "
                    f"(n={int(k.sum()):,}, {len(genes):,} 基因)")

    res["note"] = ("质心由 spot_assignment 按 cell 分组重算(非 landmark)；"
                   "ROI-local px 已按 prep_meta 的 origin/pixel_size 换算为"
                   "全片 um；KD-tree 一对一匹配；tile overlap 已去重")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"comseg_{DATASET}_metrics.csv")
    pd.DataFrame([res]).to_csv(f, index=False)
    pd.DataFrame(pairs, columns=["pred_idx", "gt_idx", "dist_um"]).to_csv(
        os.path.join(a.out, f"comseg_{DATASET}_pairs.csv"), index=False)
    log("\n" + "=" * 78)
    for k, v in res.items():
        if k in ("note", "roi_um"):
            continue
        log(f"  {k:<24} {v}")
    log("=" * 78)
    log(f"-> {f}")


if __name__ == "__main__":
    main()