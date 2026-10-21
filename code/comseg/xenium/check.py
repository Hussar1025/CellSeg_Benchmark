#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_recentroid_xenium_3.py
Recompute centroids from uns['df_spots'] in the ComSeg h5ad, avoiding landmark inheritance.

v2 fix: anndata splits a DataFrame saved to uns into a dict of arrays,
        so pd.DataFrame(dict) fails with mixed scalars:
        "If using all scalar values, you must pass an index".
        Now drop scalar entries and align lengths before building the DataFrame.
        If uns is unusable, fall back to the external tx.csv.

Only evaluation code changes; ComSeg source is untouched.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

PRED = ("yourpath/comseg_xenium/xenium_lymph/output_noprior/"
        "xenium_lymph_top300_noprior_mmc0p99.h5ad")
TX_CSV = ("yourpath/comseg_xenium/xenium_lymph/cache_noprior/"
          "xenium_lymph_top300_noprior_tx.csv")
GT_CELLS = "yourpath/dataset/xenium_roi_crop/cells.parquet"
OUT = "yourpath/eval_results_qv20/lymph"
DATASET = "xenium_3"


def log(m):
    print(m, flush=True)


def uns_to_frame(obj):
    """Rebuild df_spots from anndata uns as a DataFrame.
    May be a DataFrame / dict of arrays / structured ndarray / recarray."""
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
                log(f"    [Skip scalar] {k} = {arr}")
                continue
            if arr.ndim > 1:
                log(f"    [Skip ndim>1] {k} shape={arr.shape}")
                continue
            cols[k] = arr
            lens.append(len(arr))
        if not cols:
            return None
        n = max(set(lens), key=lens.count)
        keep = {k: v for k, v in cols.items() if len(v) == n}
        drop = set(cols) - set(keep)
        if drop:
            log(f"    [length mismatch, dropped] {sorted(drop)}")
        log(f"    rebuilt DataFrame: {n:,} rows x {len(keep)} columns")
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
    ap.add_argument("--spots", default="", help="external tx.csv; when empty, prefer uns")
    ap.add_argument("--gt-cells", default=GT_CELLS)
    ap.add_argument("--match-radius", type=float, default=0.0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import anndata as ad
    from scipy.spatial import cKDTree

    log("=" * 74)
    log(f"ComSeg @ {DATASET} - recompute centroids")
    log("=" * 74)

    sp = None
    if not a.spots:
        A = ad.read_h5ad(a.pred)
        log(f"\npredicted h5ad: {A.n_obs:,} cells x {A.n_vars:,} genes")
        log(f"  uns keys: {list(A.uns.keys())}")
        if "df_spots" in A.uns:
            log("  parsing uns['df_spots'] ...")
            raw = A.uns["df_spots"]
            log(f"    type: {type(raw).__name__}")
            if hasattr(raw, "keys"):
                log(f"    keys: {list(raw.keys())}")
            sp = uns_to_frame(raw)

    if sp is None or len(sp) == 0:
        src = a.spots or TX_CSV
        log(f"\n  uns unavailable; reading external file: {src}")
        if not Path(src).exists():
            sys.exit(f"not found: {src}")
        sp = pd.read_csv(src)
        log(f"  {len(sp):,} rows")

    log(f"\ndf_spots: {len(sp):,} rows")
    log(f"  columns: {list(sp.columns)}")

    cx = pick(sp.columns, ["x", "global_x", "x_location", "centroid_x"])
    cy = pick(sp.columns, ["y", "global_y", "y_location", "centroid_y"])
    cc = pick(sp.columns, ["cell_index", "cell", "cell_id", "index_commu",
                           "cell_label", "label", "comseg"])
    if not (cx and cy and cc):
        sys.exit(f"missing coordinate or cell column. Actual: {list(sp.columns)}")
    log(f"  using columns: x={cx}  y={cy}  cell={cc}")

    d = sp.copy()
    d[cc] = pd.to_numeric(d[cc], errors="coerce")
    d[cx] = pd.to_numeric(d[cx], errors="coerce")
    d[cy] = pd.to_numeric(d[cy], errors="coerce")
    d = d.dropna(subset=[cx, cy, cc])
    d = d[d[cc] > 0]
    log(f"  assigned {len(d):,} / {len(sp):,} "
        f"({len(d)/max(len(sp),1)*100:.1f}%)")
    if len(d) == 0:
        sys.exit("no assigned transcripts")

    grp = d.groupby(cc).agg(x=(cx, "mean"), y=(cy, "mean"), n=(cx, "size"))
    log(f"\nrecomputed centroids: {len(grp):,} cells")
    log(f"  transcripts per cell  median {grp.n.median():.0f}  mean {grp.n.mean():.0f}")
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
                log(f"  difference vs h5ad landmark centroid  median {np.median(dd0):.4f} um")
                if np.median(dd0) < 1e-3:
                    log("  same as landmark -> df_spots coordinates also come from the prior")
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
    log(f"GT in ROI {len(g):,}   match radius {r:.2f} um")

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
        log("still all zero after recomputation -> transcript coordinates themselves come from the prior landmark,")
        log("  so ComSeg has no independent spatial information in the noprior configuration.")
        log("  conclusion: ComSeg detection/localization is unusable on all platforms;")
        log("        report only assignment/count/vector metrics.")
    else:
        log("non-zero localization error -> this is ComSeg's real detection performance.")

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
        note="centroids recomputed from transcripts grouped by cell; obs landmark centroids not used"
    )]).to_csv(f"{a.out}/comseg_{DATASET}_recentroid.csv", index=False)
    log(f"\n-> {a.out}/comseg_{DATASET}_recentroid.csv")


if __name__ == "__main__":
    main()