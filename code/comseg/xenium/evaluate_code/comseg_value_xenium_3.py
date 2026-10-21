#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_value_xenium_3.py - rerun the ComSeg @ xenium_3 (lymph) evaluation

Same batch of anomalies as xenium_2: loc_* all zero, precision exactly 1.0, roi_um string
identical to the breast row. Lymph GT is xenium_roi_crop/cells.parquet
(2,442 rows, manually annotated whole-cell GT) - the 2,442 in the breast row was
very likely taken from it by mistake.

Force KD-tree spatial matching + one-to-one dedup; no direct ID connection.
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

PRED = ("yourpath/comseg_xenium/xenium_lymph/output_noprior/"
        "xenium_lymph_top300_noprior_mmc0p99.h5ad")
GT_CELLS = "yourpath/dataset/xenium_roi_crop/cells.parquet"
OUT = "yourpath/eval_results_qv20/lymph"
DATASET = "xenium_3"
TAG = "lymph"


def log(m):
    print(m, flush=True)


def md5head(p, n=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--gt-cells", default=GT_CELLS)
    ap.add_argument("--match-radius", type=float, default=0.0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import anndata as ad
    from scipy.spatial import cKDTree

    log("=" * 74)
    log(f"ComSeg @ {DATASET} ({TAG}) evaluation rerun")
    log("=" * 74)

    for name, p in (("pred", a.pred), ("gt_cells", a.gt_cells)):
        if not Path(p).exists():
            sys.exit(f"missing {name}: {p}")
        log(f"  {name:9s} {p}")
        log(f"            md5(head1M) = {md5head(p)}")

    A = ad.read_h5ad(a.pred)
    log(f"\nprediction: {A.n_obs:,} cells x {A.n_vars:,} genes")
    log(f"      obs columns: {list(A.obs.columns)}")
    cx = next((c for c in ("centroid_x", "x", "x_um") if c in A.obs), None)
    cy = next((c for c in ("centroid_y", "y", "y_um") if c in A.obs), None)
    if cx is None or cy is None:
        sys.exit(f"no centroid column in obs; actual columns: {list(A.obs.columns)}")
    px = A.obs[cx].to_numpy(float)
    py = A.obs[cy].to_numpy(float)
    log(f"      x[{px.min():.1f},{px.max():.1f}] y[{py.min():.1f},{py.max():.1f}]")

    gt = pd.read_parquet(a.gt_cells)
    log(f"\nGT full: {len(gt):,} cells  "
        f"x[{gt.x_centroid.min():.1f},{gt.x_centroid.max():.1f}] "
        f"y[{gt.y_centroid.min():.1f},{gt.y_centroid.max():.1f}]")

    pad = 0.05 * max(px.max() - px.min(), py.max() - py.min())
    x0, x1 = px.min() - pad, px.max() + pad
    y0, y1 = py.min() - pad, py.max() + pad
    m = ((gt.x_centroid >= x0) & (gt.x_centroid <= x1) &
         (gt.y_centroid >= y0) & (gt.y_centroid <= y1))
    g = gt[m]
    log(f"ROI x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]  ->  GT {len(g):,} cells")

    r = a.match_radius or float(np.sqrt((x1 - x0) * (y1 - y0) / max(len(g), 1)))
    log(f"\nmatch radius {r:.2f} um")

    gxy = g[["x_centroid", "y_centroid"]].to_numpy(float)
    pxy = np.c_[px, py]
    d, gi = cKDTree(gxy).query(pxy, k=1, distance_upper_bound=r)
    ok = np.isfinite(d) & (d < r)

    order = np.argsort(np.where(ok, d, np.inf))
    used, pairs = set(), []
    for pi in order:
        if not ok[pi]:
            break
        if gi[pi] in used:
            continue
        used.add(gi[pi])
        pairs.append((int(pi), int(gi[pi]), float(d[pi])))

    matched = len(pairs)
    prec = matched / max(len(pxy), 1)
    rec = matched / max(len(gxy), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    dd = np.array([p[2] for p in pairs]) if pairs else np.array([0.0])

    log("\n" + "=" * 74)
    log(f"pred {len(pxy):,}   gt {len(gxy):,}   matched {matched:,}")
    log(f"precision {prec:.6f}   recall {rec:.6f}   f1 {f1:.6f}")
    log(f"loc_mean {dd.mean():.4f}   median {np.median(dd):.4f}   "
        f"p95 {np.percentile(dd, 95):.4f} um")
    log("=" * 74)

    if np.median(dd) < 1e-6:
        log("centroid distances are still all 0 -> predicted centroids inherit the platform segmentation, not an independent result.")
    elif abs(prec - 1.0) < 1e-9:
        log("precision is still exactly 1.000000, suspicious.")
    else:
        log("metrics look normal; this result can replace that row in the summary table.")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(
        method="ComSeg", dataset=DATASET,
        gt_cells=len(gxy), pred_cells=len(pxy), matched=matched,
        cell_ratio=len(pxy) / max(len(gxy), 1),
        precision=prec, recall=rec, f1=f1,
        loc_mean_um=float(dd.mean()),
        loc_median_um=float(np.median(dd)),
        loc_p95_um=float(np.percentile(dd, 95)),
        match_radius_um=r,
        roi_um=f"x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]",
        note="forced KD-tree spatial matching + one-to-one dedup, no direct ID link; ROI inferred from predicted coordinates"
    )]).to_csv(f"{a.out}/comseg_{DATASET}_rematch.csv", index=False)
    pd.DataFrame(pairs, columns=["pred_idx", "gt_idx", "dist_um"]).to_csv(
        f"{a.out}/comseg_{DATASET}_rematch_pairs.csv", index=False)
    log(f"\n-> {a.out}/comseg_{DATASET}_rematch.csv")


if __name__ == "__main__":
    main()