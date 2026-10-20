#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_value_xenium_2.py - evaluate ComSeg @ xenium_2 (Xenium breast, ROI5000)

v2 fixes a coordinate-frame bug
  global_x/global_y in spot_assignment are ROI-local pixels (x[0.1,5000.0]),
  while x_centroid/y_centroid in cells.parquet are whole-slide um. v1 used the
  former range to bound the latter, selecting 74,308 GT cells over the whole
  slide (should be 8,393); the match radius was inflated to 20 um, so
  count_pearson=-0.05, vec_cosine=0.31 and the rest were spurious.
  Conversion (prep_meta.json):  um = (local_px + origin) * pixel_size
                           x0=24497  y0=11210  pixel_size=0.2125

Evaluation protocol matches comseg_value_xenium_4.py:
  centroids are recomputed from spot_assignment grouped by cell, not obs.centroid_x/y
  in the h5ad - the latter inherits the nucleus-prior landmark (uns has
  _landmark_for_centroid as evidence), which would make loc_* always 0 and precision always 1.
  Matching uses KD-tree nearest neighbor + radius cutoff + one-to-one dedup, never direct IDs.
  Radius = sqrt(ROI area / GT cells in ROI).

Metrics: detection (P/R/F1, cell_ratio), localization (loc mean/median/p95),
      count (pearson/spearman/mae/rmse), expression vector (cosine/JS/pearson),
      assignment (assign_accuracy, overall_correct, coverage_fraction),
      morphology (median_diameter_um), distribution (tx/cell quantiles, share <10)

Usage:
  python3 comseg_value_xenium_2.py
  python3 comseg_value_xenium_2.py --no-dedup
  python3 comseg_value_xenium_2.py --skip-tx     # skip expression vectors (saves memory)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "yourpath/comseg_xenium_breast_roi5000"
GT_CELLS = "yourpath/dataset/xenium_breast/cells.parquet"
GT_TX = "yourpath/dataset/xenium_breast/transcripts.parquet"
OUT = "yourpath/eval_results_qv20/breast"
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

    # --- coordinate conversion parameters -------------------------------------
    mp = os.path.join(a.root, "input", "prep_meta.json")
    if not os.path.exists(mp):
        sys.exit(f"missing {mp}; cannot determine the ROI origin and pixel size")
    meta = json.load(open(mp))
    X0, Y0 = float(meta["x0"]), float(meta["y0"])
    PX = float(meta["pixel_size"])
    log(f"  ROI origin (x0={X0:.0f}, y0={Y0:.0f}) px   pixel_size {PX} um/px")
    log(f"  ROI {meta['size_x']} x {meta['size_y']} px = "
        f"{meta['size_x']*PX:.1f} x {meta['size_y']*PX:.1f} um")
    log(f"  prep: {meta['n_transcripts']:,} tx / GT {meta['gt_expected_cells']:,}"
        f" = {meta['n_transcripts']/meta['gt_expected_cells']:.1f} tx/cell"
        f"   qv_min={meta.get('qv_min')}")

    # --- prediction -----------------------------------------------------------
    fs = sorted(glob.glob(os.path.join(
        a.root, "comseg_out", "tile_*", "spot_assignment.parquet")))
    if not fs:
        sys.exit(f"no spot_assignment.parquet")
    log(f"\nprediction: {len(fs)} tiles")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    log(f"  merged {len(d):,} rows   columns: {list(d.columns)}")
    if not a.no_dedup:
        n0 = len(d)
        d = d.drop_duplicates(subset=["global_x", "global_y", "gene"],
                              keep="first")
        log(f"  tile overlap dedup: {n0:,} -> {len(d):,} "
            f"(removed {(n0-len(d))/n0*100:.1f}%)")

    # ROI-local px -> whole-slide um
    d["_xum"] = (d["global_x"].astype(float) + X0) * PX
    d["_yum"] = (d["global_y"].astype(float) + Y0) * PX
    log(f"  local px  x[{d.global_x.min():.1f},{d.global_x.max():.1f}]")
    log(f"  whole-slide um   x[{d._xum.min():.1f},{d._xum.max():.1f}] "
        f"y[{d._yum.min():.1f},{d._yum.max():.1f}]")

    assigned = pd.to_numeric(d["cell"], errors="coerce") > 0
    log(f"  assigned {int(assigned.sum()):,} ({assigned.mean()*100:.1f}%)")

    da = d[assigned].copy()
    g = da.groupby("cell").agg(x=("_xum", "mean"), y=("_yum", "mean"),
                               n=("_xum", "size"))
    q = g.n.quantile([.05, .25, .5, .75, .95])
    log(f"\n  predicted cells {len(g):,}")
    log(f"  tx per cell  median {g.n.median():.0f}  mean {g.n.mean():.0f}  "
        f"mean/median {g.n.mean()/max(g.n.median(),1):.2f}")
    log(f"             p5/p25/p75/p95 = {q.iloc[0]:.0f}/{q.iloc[1]:.0f}/"
        f"{q.iloc[3]:.0f}/{q.iloc[4]:.0f}   <10: {(g.n<10).mean()*100:.2f}%")

    sp = da.groupby("cell").agg(sx=("_xum", "std"), sy=("_yum", "std"))
    diam = 2 * np.sqrt(2) * np.sqrt((sp.sx ** 2 + sp.sy ** 2) / 2)
    log(f"  equivalent diameter (2*sqrt2*RMS spread) median {diam.median():.2f} um")

    # --- GT: crop by the true ROI bounds, not by the predicted coordinate span ---
    gt_all = pd.read_parquet(a.gt_cells)
    x0u, x1u = X0 * PX, (X0 + meta["size_x"]) * PX
    y0u, y1u = Y0 * PX, (Y0 + meta["size_y"]) * PX
    m = ((gt_all.x_centroid >= x0u) & (gt_all.x_centroid < x1u) &
         (gt_all.y_centroid >= y0u) & (gt_all.y_centroid < y1u))
    gt = gt_all[m].reset_index(drop=True)
    log(f"\nGT: ROI x[{x0u:.1f},{x1u:.1f}] y[{y0u:.1f},{y1u:.1f}] um")
    log(f"  in ROI {len(gt):,}  (whole slide {len(gt_all):,}; "
        f"prep expected {meta['gt_expected_cells']:,})")
    if abs(len(gt) - meta["gt_expected_cells"]) > .1 * meta["gt_expected_cells"]:
        log(f"  [warn] differs from prep expectation by >10%; check the ROI conversion")

    gtn = None
    for c in ("transcript_counts", "n_counts", "total_counts"):
        if c in gt.columns:
            gtn = gt[c].to_numpy(float)
            log(f"  GT tx per cell from column {c}  median {np.median(gtn):.0f}")
            break

    # --- matching -------------------------------------------------------------
    r = a.match_radius or float(np.sqrt((x1u - x0u) * (y1u - y0u) /
                                        max(len(gt), 1)))
    log(f"\nmatching: KD-tree one-to-one, radius {r:.2f} um")
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
        log("  loc_median = 0 -> centroids still come from the prior landmark")

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

    # --- counts ---------------------------------------------------------------
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
            log(f"\ncounts: n={int(k.sum()):,}  "
                f"pearson {res['count_pearson']:.4f}  "
                f"spearman {res['count_spearman']:.4f}  "
                f"mae {res['count_mae']:.1f}  rmse {res['count_rmse']:.1f}")

    # --- assignment + expression vectors --------------------------------------
    if pairs and not a.skip_tx and os.path.exists(a.gt_tx):
        log(f"\nassignment / expression vectors: reading GT transcript table ...")
        try:
            tx = pd.read_parquet(a.gt_tx, columns=[
                "x_location", "y_location", "feature_name", "cell_id", "qv"])
        except Exception:
            tx = pd.read_parquet(a.gt_tx)
        tx = tx[(tx.x_location >= x0u) & (tx.x_location < x1u) &
                (tx.y_location >= y0u) & (tx.y_location < y1u)].copy()
        if "qv" in tx.columns:
            tx = tx[tx.qv >= res["qv_min"]]
        log(f"  GT transcripts in ROI {len(tx):,}  "
            f"(prep recorded {meta['n_transcripts']:,})")

        id2row = {v: i for i, v in
                  enumerate(gt["cell_id"].astype(str).to_numpy())}
        tx["_gtrow"] = (tx["cell_id"].astype(str).map(id2row)
                        .fillna(-1).astype(np.int64))

        # predicted transcript -> nearest GT transcript, tolerance 0.5 um
        dq, iq = cKDTree(np.c_[tx.x_location.to_numpy(float),
                               tx.y_location.to_numpy(float)]).query(
            np.c_[da._xum.to_numpy(float), da._yum.to_numpy(float)],
            k=1, distance_upper_bound=0.5)
        hit = np.isfinite(dq)
        log(f"  predicted transcripts matched to a GT transcript {int(hit.sum()):,} / {len(da):,} "
            f"({hit.mean()*100:.1f}%)")
        if hit.mean() < .5:
            log("  [warn] hit rate <50%; the coordinate conversion may still be off")

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
                f"(over {int(valid.sum()):,} decidable transcripts)")
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
                    f"(n={int(k.sum()):,}, {len(genes):,} genes)")

    res["note"] = ("centroids recomputed from spot_assignment grouped by cell (not landmark); "
                   "ROI-local px converted to whole-slide um with prep_meta origin/pixel_size; "
                   "KD-tree one-to-one matching; tile overlap deduplicated")

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