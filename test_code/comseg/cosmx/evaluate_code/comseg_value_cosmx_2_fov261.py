#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_value_cosmx_2.py — 评估 ComSeg @ cosmx_2 (CosMx lymph node, FOV 48)

数据状态（跑之前必须知道）:
  25 个 tile 里只有 14 个带 cell 列，tile_0014-0024 是 08-11 那次
  ComSegDataset API 报错留下的 6 列残骸（只有 index/x/y/z/gene/in_nucleus）。
  comseg_cosmx_3.py 的完成判定只看 spot_assignment 文件是否存在、不看内容，
  于是那 11 个残骸也被打了 DONE.flag，后续续跑一律跳过。

  但补跑不划算: 25 个 tile 并非空间互斥分块，坐标范围全是 x[0,532] y[0,532]，
  待补 tile 的坐标点有 97.5% 已被完成的 14 个覆盖。补跑要 82 小时、209 GB，
  只换来 2.5% 的新空间位置。因此本脚本只评估那 14 个 tile，
  并在结果里记 n_tiles_used / tile_coverage 供审稿时说明。

  14 个 tile 的质量本身是好的: 累计 4,675 细胞 / GT 4,898 = 0.955，
  平均已分配 97.5%，逐 tile 的 nuc 与 cell 数一对一（零合并零拆分）。

坐标: CosMx FOV-local，单位 um（0-531.5，即 4256 px x 0.12 um/px）。
  GT 用 S0_metadata_file 里 FOV 48 的 CenterX/Y_local_px x 0.12。

评估口径与 comseg_value_xenium_2.py 一致:
  质心从 spot_assignment 按 cell 分组重算，不用 h5ad 的 landmark 质心；
  KD-tree 最近邻 + 半径截断 + 一对一去重，无 ID 直连；
  半径 = sqrt(ROI面积 / ROI内GT细胞数)。

用法:
  python3 comseg_value_cosmx_2.py
  python3 comseg_value_cosmx_2.py --match-radius 8
  python3 comseg_value_cosmx_2.py --no-dedup
  python3 comseg_value_cosmx_2.py --skip-tx        # 跳过表达向量（省内存）
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/data/qiuyijia/comseg_cosmx_lymph_node/fov261"
META = ("/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/"
        "S0_metadata_file.csv.gz")
TX_CACHE = ("/data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/"
            "tx_fov_48_261.parquet")
OUT = "/data/qiuyijia/eval_results_qv20/cosmx_2"
METHOD, DATASET = "ComSeg", "cosmx_2"
FOV = 261
PX_UM = 0.12
FOV_PX = 4256
CELL_COLS = ("cell", "cell_index", "comseg", "cell_id", "label",
             "cell_label", "cell_index_pred")


def log(m):
    print(m, flush=True)


def load_pred(root, dedup=True):
    """只收带 cell 列的 tile；6 列的是 prep 残骸。"""
    fs = sorted(glob.glob(os.path.join(
        root, "comseg_out", "tile_*", "spot_assignment.parquet")))
    if not fs:
        sys.exit(f"✗ 无 spot_assignment.parquet: {root}/comseg_out/")
    good, bad, frames = [], [], []
    for f in fs:
        t = pd.read_parquet(f)
        name = os.path.basename(os.path.dirname(f))
        if any(c in t.columns for c in CELL_COLS):
            good.append(name)
            frames.append(t)
        else:
            bad.append(name)
    log(f"  {len(fs)} 个 tile: {len(good)} 个带 cell 列, "
        f"{len(bad)} 个只有 prep 输入")
    if bad:
        log(f"    残骸 tile（08-11 的 API 报错遗留）: {bad}")
    if not frames:
        sys.exit("✗ 没有任何 tile 带 cell 列")
    d = pd.concat(frames, ignore_index=True)
    log(f"  合并 {len(d):,} 行   列 {len(d.columns)}: "
        f"{list(d.columns)[:8]}...")
    if dedup:
        n0 = len(d)
        d = d.drop_duplicates(subset=["x", "y", "gene"], keep="first")
        log(f"  tile overlap 去重: {n0:,} -> {len(d):,} "
            f"(删 {(n0-len(d))/n0*100:.1f}%)")
    return d, good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--meta", default=META)
    ap.add_argument("--tx-cache", default=TX_CACHE)
    ap.add_argument("--fov", type=int, default=FOV)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=0.0)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--skip-tx", action="store_true")
    a = ap.parse_args()

    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr, spearmanr

    log("=" * 78)
    log(f"{METHOD} @ {DATASET}  (CosMx lymph node, FOV {a.fov})")
    log("=" * 78)

    log(f"\n【预测】{a.root}")
    d, good, bad = load_pred(a.root, not a.no_dedup)
    cc = next(c for c in CELL_COLS if c in d.columns)
    log(f"  用 cell 列: {cc}  （七个 cell 列数值完全相同，是同一结果的别名）")
    v = pd.to_numeric(d[cc], errors="coerce")
    assigned = v > 0
    log(f"  已分配 {int(assigned.sum()):,} ({assigned.mean()*100:.1f}%)")

    da = d[assigned].copy()
    g = da.groupby(cc).agg(x=("x", "mean"), y=("y", "mean"),
                           n=("x", "size"))
    q = g.n.quantile([.05, .25, .5, .75, .95])
    log(f"\n  预测细胞 {len(g):,}")
    log(f"  每细胞 tx  中位 {g.n.median():.0f}  均值 {g.n.mean():.0f}  "
        f"均/中 {g.n.mean()/max(g.n.median(),1):.2f}")
    log(f"             p5/p25/p75/p95 = {q.iloc[0]:.0f}/{q.iloc[1]:.0f}/"
        f"{q.iloc[3]:.0f}/{q.iloc[4]:.0f}   "
        f"<10 的 {(g.n<10).mean()*100:.2f}%")
    log(f"  质心 x[{g.x.min():.1f},{g.x.max():.1f}] "
        f"y[{g.y.min():.1f},{g.y.max():.1f}] um   "
        f"(FOV = {FOV_PX*PX_UM:.1f} um 见方)")

    sp = da.groupby(cc).agg(sx=("x", "std"), sy=("y", "std"))
    diam = 2 * np.sqrt(2) * np.sqrt((sp.sx ** 2 + sp.sy ** 2) / 2)
    log(f"  等效直径(2*sqrt2*RMS散布) 中位 {diam.median():.2f} um  "
        f"(metadata 实测淋巴细胞 7.86 um)")

    # 核先验一致性：in_nucleus 有多少核，预测出多少细胞
    if "in_nucleus" in d.columns:
        nuc = pd.to_numeric(d["in_nucleus"], errors="coerce")
        nn = int(nuc[nuc > 0].nunique())
        both = (nuc > 0) & assigned
        t2 = pd.DataFrame({"nuc": nuc[both].astype(int),
                           "cel": v[both].astype(int)})
        per = t2.groupby("cel")["nuc"].nunique()
        per2 = t2.groupby("nuc")["cel"].nunique()
        log(f"\n  核先验 {nn:,} 个（14 tile 内 {t2.nuc.nunique():,} 个参与）")
        log(f"  每细胞覆盖核数 均值 {per.mean():.2f}  "
            f">1 的 {int((per>1).sum()):,} ({(per>1).mean()*100:.1f}%)")
        log(f"  每核被拆到 >1 细胞的 {int((per2>1).sum()):,}")
        log(f"  -> 一对一说明既无核合并也无核拆分")

    log(f"\n【GT】{os.path.basename(a.meta)}")
    meta = pd.read_csv(a.meta)
    fc = next(c for c in meta.columns if c.lower() == "fov")
    sub = meta[pd.to_numeric(meta[fc], errors="coerce") == a.fov].copy()
    cx = next((c for c in ("CenterX_local_px", "x_local_px")
               if c in sub.columns), None)
    cy = next((c for c in ("CenterY_local_px", "y_local_px")
               if c in sub.columns), None)
    if cx is None:
        sys.exit(f"✗ metadata 无局部坐标列: {list(sub.columns)[:12]}")
    gx = sub[cx].to_numpy(float) * PX_UM
    gy = sub[cy].to_numpy(float) * PX_UM
    log(f"  FOV {a.fov} 内 GT {len(sub):,}   "
        f"x[{gx.min():.1f},{gx.max():.1f}] y[{gy.min():.1f},{gy.max():.1f}] um")
    gtn = None
    for c in ("nCount_RNA", "totalcounts", "Total_counts", "nCount"):
        if c in sub.columns:
            gtn = sub[c].to_numpy(float)
            log(f"  GT 每细胞 tx 用列 {c}  中位 {np.nanmedian(gtn):.0f}")
            break
    ga = None
    for c in ("Area", "area", "NucArea"):
        if c in sub.columns:
            ga = sub[c].to_numpy(float)
            log(f"  GT 面积用列 {c}  中位 {np.nanmedian(ga):.0f} px2 "
                f"= 直径 {2*np.sqrt(np.nanmedian(ga)/np.pi)*PX_UM:.2f} um")
            break

    r = a.match_radius or float(np.sqrt(
        (FOV_PX * PX_UM) ** 2 / max(len(sub), 1)))
    log(f"\n【匹配】KD-tree 一对一，半径 {r:.2f} um")
    P = np.c_[g.x.to_numpy(float), g.y.to_numpy(float)]
    G = np.c_[gx, gy]
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
        log("  ★ loc_median = 0 -> 质心继承自 prior landmark")

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
               n_tiles_used=len(good), n_tiles_total=len(good) + len(bad),
               uses_platform_prior=False,
               detection_comparable=det_ok, data_error=False)

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

    # 转录本级归属 + 表达向量：把预测转录本对到 GT 转录本
    if pairs and not a.skip_tx and os.path.exists(a.tx_cache):
        log(f"\n【归属 / 表达向量】{os.path.basename(a.tx_cache)}")
        tx = pd.read_parquet(a.tx_cache)
        tfc = "fov" if "fov" in tx.columns else "FOV"
        tx = tx[pd.to_numeric(tx[tfc], errors="coerce") == a.fov].copy()
        txc = next((c for c in ("x_local_px", "x") if c in tx.columns), None)
        tyc = next((c for c in ("y_local_px", "y") if c in tx.columns), None)
        tgc = next((c for c in ("target", "gene", "feature_name")
                    if c in tx.columns), None)
        tcc = next((c for c in ("cell_ID", "cell_id", "cell")
                    if c in tx.columns), None)
        log(f"  {len(tx):,} 转录本   列 x={txc} y={tyc} gene={tgc} "
            f"cell={tcc}")
        if txc and tyc and tcc:
            tx["_xum"] = tx[txc].astype(float) * PX_UM
            tx["_yum"] = tx[tyc].astype(float) * PX_UM
            gt_ids = sub["cell_ID"].astype(str).to_numpy() \
                if "cell_ID" in sub.columns \
                else sub.iloc[:, 0].astype(str).to_numpy()
            id2row = {s: i for i, s in enumerate(gt_ids)}
            tx["_gtrow"] = (tx[tcc].astype(str).map(id2row)
                            .fillna(-1).astype(np.int64))
            hit_gt = int((tx["_gtrow"] >= 0).sum())
            log(f"  GT 转录本能对上 metadata 的 {hit_gt:,} "
                f"({hit_gt/max(len(tx),1)*100:.1f}%)")

            dq, iq = cKDTree(np.c_[tx._xum.to_numpy(float),
                                   tx._yum.to_numpy(float)]).query(
                np.c_[da.x.to_numpy(float), da.y.to_numpy(float)],
                k=1, distance_upper_bound=0.3)
            hit = np.isfinite(dq)
            log(f"  预测转录本对上 GT 转录本 {int(hit.sum()):,} / "
                f"{len(da):,} ({hit.mean()*100:.1f}%)")
            if hit.mean() < .5:
                log("  ⚠ 命中率 <50%，坐标口径可能不一致")

            cell2row = {c: i for i, c in enumerate(g.index)}
            pred_row = np.array([cell2row.get(c, -1)
                                 for c in da[cc].to_numpy()])
            gt_row = np.full(len(da), -1, dtype=np.int64)
            gt_row[hit] = tx["_gtrow"].to_numpy()[iq[hit].astype(np.int64)]
            p2g = {p[0]: p[1] for p in pairs}
            valid = hit & (pred_row >= 0) & (gt_row >= 0)
            if valid.sum() >= 100:
                want = np.array([p2g.get(int(pr), -2)
                                 for pr in pred_row[valid]])
                acc = float((want == gt_row[valid]).mean())
                res["assign_accuracy"] = acc
                res["overall_correct"] = acc * float(assigned.mean())
                log(f"  assign_accuracy {acc:.4f}  "
                    f"(在 {int(valid.sum()):,} 条可判定转录本上)")
                log(f"  overall_correct  {res['overall_correct']:.4f}")

            if tgc:
                from scipy.sparse import coo_matrix
                genes = sorted(set(da["gene"].astype(str)) |
                               set(tx[tgc].astype(str)))
                gidx = {v_: i for i, v_ in enumerate(genes)}
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
                      tx[tgc].astype(str).map(gidx).to_numpy()[vg])),
                    shape=(len(sub), len(genes))).tocsr()
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
                    res["vec_js_dist"] = float(
                        np.mean(np.sqrt((kl1 + kl2) / 2)))
                    nn = min(2000, int(k.sum()))
                    res["vec_pearson"] = float(np.mean(
                        [pearsonr(A[k][i], B[k][i])[0] for i in range(nn)]))
                    log(f"  vec_cosine {res['vec_cosine']:.4f}  "
                        f"vec_js_dist {res['vec_js_dist']:.4f}  "
                        f"vec_pearson {res['vec_pearson']:.4f}  "
                        f"(n={int(k.sum()):,}, {len(genes):,} 基因)")

    res["note"] = (f"只用 {len(good)}/{len(good)+len(bad)} 个 tile —— "
                   f"tile_0014-0024 是 08-11 的 ComSegDataset API 报错遗留的 "
                   f"6 列 prep 残骸，且待补 tile 的坐标点 97.5% 已被完成的 "
                   f"tile 覆盖，补跑要 82 小时只换 2.5% 新空间；"
                   f"质心由 spot_assignment 按 cell 分组重算(非 landmark)；"
                   f"KD-tree 一对一匹配；tile overlap 已去重")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"comseg_{DATASET}_metrics.csv")
    pd.DataFrame([res]).to_csv(f, index=False)
    pd.DataFrame(pairs, columns=["pred_idx", "gt_idx", "dist_um"]).to_csv(
        os.path.join(a.out, f"comseg_{DATASET}_pairs.csv"), index=False)
    log("\n" + "=" * 78)
    for k, v_ in res.items():
        if k == "note":
            continue
        log(f"  {k:<24} {v_}")
    log("=" * 78)
    log(f"-> {f}")
    log(f"\n⚠ 入表时注明 n_tiles_used={len(good)}/{len(good)+len(bad)}")


if __name__ == "__main__":
    main()