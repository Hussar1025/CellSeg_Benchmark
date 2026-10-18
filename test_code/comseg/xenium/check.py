#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_recentroid_xenium_3.py
从 ComSeg h5ad 的 uns['df_spots'] 重算质心，摆脱 landmark 继承

v2 修复: anndata 保存 DataFrame 到 uns 时会拆成 dict of arrays，
        直接 pd.DataFrame(dict) 会因混入标量而报
        "If using all scalar values, you must pass an index"。
        现在先剥掉标量项、对齐长度，再构建 DataFrame。
        若 uns 不可用，自动回退到外部 tx.csv。

只改评估代码，不动 ComSeg 源码。
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

PRED = ("/data/qiuyijia/comseg_xenium/xenium_lymph/output_noprior/"
        "xenium_lymph_top300_noprior_mmc0p99.h5ad")
TX_CSV = ("/data/qiuyijia/comseg_xenium/xenium_lymph/cache_noprior/"
          "xenium_lymph_top300_noprior_tx.csv")
GT_CELLS = "/data/qiuyijia/dataset/xenium_roi_crop/cells.parquet"
OUT = "/data/qiuyijia/eval_results_qv20/lymph"
DATASET = "xenium_3"


def log(m):
    print(m, flush=True)


def uns_to_frame(obj):
    """把 anndata uns 里的 df_spots 还原成 DataFrame。
    可能是 DataFrame / dict of arrays / structured ndarray / recarray。"""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return pd.DataFrame({n: obj[n] for n in obj.dtype.names})
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        cols, lens = {}, []
        for k in obj.keys():
            v = obj[k]
            arr = np.asarray(v)
            if arr.ndim == 0:
                log(f"    [跳过标量项] {k} = {arr}")
                continue
            if arr.ndim > 1:
                log(f"    [跳过多维项] {k} shape={arr.shape}")
                continue
            cols[k] = arr
            lens.append(len(arr))
        if not cols:
            return None
        n = max(set(lens), key=lens.count)
        keep = {k: v for k, v in cols.items() if len(v) == n}
        drop = set(cols) - set(keep)
        if drop:
            log(f"    [长度不符，丢弃] {sorted(drop)}")
        log(f"    还原 DataFrame: {n:,} 行 x {len(keep)} 列")
        return pd.DataFrame(keep)
    return None


def pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--spots", default="", help="外部 tx.csv，留空则优先用 uns")
    ap.add_argument("--gt-cells", default=GT_CELLS)
    ap.add_argument("--match-radius", type=float, default=0.0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import anndata as ad
    from scipy.spatial import cKDTree

    log("=" * 74)
    log(f"ComSeg @ {DATASET} —— 重算质心")
    log("=" * 74)

    sp = None
    if not a.spots:
        A = ad.read_h5ad(a.pred)
        log(f"\n预测 h5ad: {A.n_obs:,} cells x {A.n_vars:,} genes")
        log(f"  uns 键: {list(A.uns.keys())}")
        if "df_spots" in A.uns:
            log("  解析 uns['df_spots'] ...")
            raw = A.uns["df_spots"]
            log(f"    类型: {type(raw).__name__}")
            if hasattr(raw, "keys"):
                log(f"    键: {list(raw.keys())}")
            sp = uns_to_frame(raw)

    if sp is None or len(sp) == 0:
        src = a.spots or TX_CSV
        log(f"\n  uns 不可用，改读外部文件: {src}")
        if not Path(src).exists():
            sys.exit(f"✗ 不存在: {src}")
        sp = pd.read_csv(src)
        log(f"  {len(sp):,} 行")

    log(f"\ndf_spots: {len(sp):,} 行")
    log(f"  列: {list(sp.columns)}")

    cx = pick(sp.columns, ["x", "global_x", "x_location", "centroid_x"])
    cy = pick(sp.columns, ["y", "global_y", "y_location", "centroid_y"])
    cc = pick(sp.columns, ["cell_index", "cell", "cell_id", "index_commu",
                           "cell_label", "label", "comseg"])
    if not (cx and cy and cc):
        sys.exit(f"✗ 缺坐标或 cell 列。实际: {list(sp.columns)}")
    log(f"  用列: x={cx}  y={cy}  cell={cc}")

    d = sp.copy()
    d[cc] = pd.to_numeric(d[cc], errors="coerce")
    d[cx] = pd.to_numeric(d[cx], errors="coerce")
    d[cy] = pd.to_numeric(d[cy], errors="coerce")
    d = d.dropna(subset=[cx, cy, cc])
    d = d[d[cc] > 0]
    log(f"  已分配 {len(d):,} / {len(sp):,} "
        f"({len(d)/max(len(sp),1)*100:.1f}%)")
    if len(d) == 0:
        sys.exit("✗ 没有已分配的转录本")

    grp = d.groupby(cc).agg(x=(cx, "mean"), y=(cy, "mean"), n=(cx, "size"))
    log(f"\n重算质心: {len(grp):,} cells")
    log(f"  每细胞转录本 中位 {grp.n.median():.0f}  均值 {grp.n.mean():.0f}")
    log(f"  x[{grp.x.min():.2f},{grp.x.max():.2f}] "
        f"y[{grp.y.min():.2f},{grp.y.max():.2f}]")

    if not a.spots:
        try:
            A2 = ad.read_h5ad(a.pred)
            if len(grp) == A2.n_obs:
                ox = np.sort(A2.obs["centroid_x"].to_numpy(float))
                oy = np.sort(A2.obs["centroid_y"].to_numpy(float))
                dd0 = np.hypot(np.sort(grp.x.to_numpy()) - ox,
                               np.sort(grp.y.to_numpy()) - oy)
                log(f"  与 h5ad landmark 质心的差 中位 {np.median(dd0):.4f} um")
                if np.median(dd0) < 1e-3:
                    log("  ★ 与 landmark 相同 -> df_spots 坐标也来自 prior")
        except Exception:
            pass

    gt = pd.read_parquet(a.gt_cells)
    log(f"\nGT: {len(gt):,} cells  "
        f"x[{gt.x_centroid.min():.1f},{gt.x_centroid.max():.1f}] "
        f"y[{gt.y_centroid.min():.1f},{gt.y_centroid.max():.1f}]")

    px, py = grp.x.to_numpy(float), grp.y.to_numpy(float)
    pad = 0.05 * max(px.max() - px.min(), py.max() - py.min())
    m = ((gt.x_centroid >= px.min() - pad) & (gt.x_centroid <= px.max() + pad) &
         (gt.y_centroid >= py.min() - pad) & (gt.y_centroid <= py.max() + pad))
    g = gt[m]
    area = (px.max() - px.min() + 2 * pad) * (py.max() - py.min() + 2 * pad)
    r = a.match_radius or float(np.sqrt(area / max(len(g), 1)))
    log(f"ROI 内 GT {len(g):,}   匹配半径 {r:.2f} um")

    gxy = g[["x_centroid", "y_centroid"]].to_numpy(float)
    pxy = np.c_[px, py]
    dist_, gi = cKDTree(gxy).query(pxy, k=1, distance_upper_bound=r)
    ok = np.isfinite(dist_) & (dist_ < r)
    order = np.argsort(np.where(ok, dist_, np.inf))
    used, pairs = set(), []
    for pi in order:
        if not ok[pi]:
            break
        if gi[pi] in used:
            continue
        used.add(gi[pi])
        pairs.append((int(pi), int(gi[pi]), float(dist_[pi])))

    matched = len(pairs)
    prec = matched / max(len(pxy), 1)
    rec = matched / max(len(gxy), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    dist = np.array([p[2] for p in pairs]) if pairs else np.array([0.0])

    log("\n" + "=" * 74)
    log(f"pred {len(pxy):,}   gt {len(gxy):,}   matched {matched:,}")
    log(f"precision {prec:.6f}   recall {rec:.6f}   f1 {f1:.6f}")
    log(f"loc_mean {dist.mean():.4f}   median {np.median(dist):.4f}   "
        f"p95 {np.percentile(dist, 95):.4f} um")
    log("=" * 74)

    if np.median(dist) < 1e-6:
        log("★ 重算后仍全为 0 -> 转录本坐标本身来自 prior landmark，")
        log("  ComSeg 在 noprior 配置下没有独立的空间信息。")
        log("  结论: ComSeg 的 detection/localization 全平台不可用，")
        log("        只报 assignment/count/vector 指标。")
    else:
        log("✓ 有了非零定位误差，这是 ComSeg 真实的 detection 表现。")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(
        method="ComSeg", dataset=DATASET,
        gt_cells=len(gxy), pred_cells=len(pxy), matched=matched,
        cell_ratio=len(pxy) / max(len(gxy), 1),
        precision=prec, recall=rec, f1=f1,
        loc_mean_um=float(dist.mean()),
        loc_median_um=float(np.median(dist)),
        loc_p95_um=float(np.percentile(dist, 95)),
        match_radius_um=r,
        tx_per_cell_median=float(grp.n.median()),
        note="质心由转录本按 cell 分组重算，未用 obs 的 landmark 质心"
    )]).to_csv(f"{a.out}/comseg_{DATASET}_recentroid.csv", index=False)
    log(f"\n-> {a.out}/comseg_{DATASET}_recentroid.csv")


if __name__ == "__main__":
    main()