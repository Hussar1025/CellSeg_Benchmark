#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_value_stereo_2.py — evaluate UCS @ stereo_2 (Stereo-seq E14.5_E1S3)

Artifact: yourpath/ucs_stereoseq_E14.5/ucs_log/pred/segmentation_mask.tif
Health check passed:
  nuclei_mask      4,872 labels  diameter 8.3 um   (input OK)
  mask_shift_0     4,872         diameter 14.4 um
  segmentation     4,870 / 4872 = 1.000  diameter 14.4 um
  nucleus 8.3um -> cell 14.4um, reasonable expansion factor. The single-pixel
  degradation seen on E16.5_E2S7 (erode loop falls back to centroid when
  e.sum()==0) did not trigger on this dataset.

⚠ Circular bias: nuclei_mask is rasterized from the GEM cell column (official CellBin segmentation),
  and GT is the same cell column -> prior and GT share the same source.
  assignment / coverage metrics are circularly inflated; only detection/localization/count/morphology are valid.
  The result sets uses_platform_prior=True and notes it.

Coordinates: GEM bin 1 = 0.5 um; mask (0,0) maps to GEM (x.min, y.min),
      converted to um uniformly in evaluation.

Metrics: detection(P/R/F1, cell_ratio) localization(loc mean/median/p95)
         counts(pearson/spearman/mae/rmse) expression vectors(cosine/JS/pearson)
         assignment(assign_accuracy, coverage) morphology(area/equivalent-diameter quantiles)

Usage:
  python3 ucs_value_stereo_2.py
  python3 ucs_value_stereo_2.py --mask mask_shift_0.tif
  python3 ucs_value_stereo_2.py --match-radius 8
"""
import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd

ROOT = "yourpath/ucs_stereoseq_E14.5"
GEM = ("yourpath/dataset/stereoseq/stereo-seq_data_all/"
       "E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz")
OUT = "yourpath/eval_results_qv20/stereo_2"
METHOD, DATASET = "UCS", "stereo_2"
BIN_UM = 0.5
GT_N = 4872


def log(m):
    print(m, flush=True)


def read_gem(p):
    sk = 0
    with gzip.open(p, "rt") as f:
        for l in f:
            if l.startswith("#"):
                sk += 1
            else:
                break
    d = pd.read_csv(p, sep="\t", skiprows=sk)
    ren = {}
    for c in d.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcount", "midcounts", "count", "counts"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    d = d.rename(columns=ren)
    if "count" not in d.columns:
        d["count"] = 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--gem", default=GEM)
    ap.add_argument("--mask", default="segmentation_mask.tif")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=0.0)
    a = ap.parse_args()

    import tifffile
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr, spearmanr

    log("=" * 78)
    log(f"{METHOD} @ {DATASET}  (Stereo-seq E14.5_E1S3)")
    log("=" * 78)

    mp = os.path.join(a.root, "ucs_log", "pred", a.mask)
    if not os.path.exists(mp):
        sys.exit(f"✗ mask not found: {mp}")
    m = tifffile.imread(mp)
    lab, cnt = np.unique(m, return_counts=True)
    area = cnt[lab > 0]
    n_pred = int((lab > 0).sum())
    log(f"  {a.mask}  {m.shape} {m.dtype}")
    log(f"  labels {n_pred:,}   foreground {(m>0).mean()*100:.2f}%   "
        f"max label {lab[lab>0].max():,}")
    if np.median(area) <= 2:
        sys.exit("✗ area median <=2px, rasterization degraded, result invalid")

    q = np.percentile(area, [5, 25, 50, 75, 95])
    log(f"\n[morphology] area px  p5/p25/median/p75/p95 = "
        f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}/{q[4]:.0f}")
    log(f"        area um2 median {q[2]*BIN_UM**2:.1f}   "
        f"equivalent diameter median {2*np.sqrt(q[2]/np.pi)*BIN_UM:.2f} um  "
        f"(mouse embryo cells ~6-10 um)")
    log(f"        <10px: {int((area<10).sum()):,} "
        f"({(area<10).mean()*100:.2f}%)   "
        f"largest label / foreground {area.max()/area.sum()*100:.2f}%")

    # ── GEM / GT ────────────────────────────────────────────────────
    log(f"\n[GEM] {os.path.basename(a.gem)}")
    d = read_gem(a.gem)
    x0, y0 = int(d.x.min()), int(d.y.min())
    log(f"  {len(d):,} rows   genes {d.gene.nunique():,}   "
        f"origin ({x0},{y0})   span "
        f"{d.x.max()-x0+1} x {d.y.max()-y0+1} bin")
    gt = d[d.cell > 0].groupby("cell").agg(
        x=("x", "mean"), y=("y", "mean"), n=("count", "sum"))
    log(f"  GT cells {len(gt):,}  (expected {GT_N:,})   "
        f"median tx per cell {gt.n.median():.0f}")

    # ── predicted centroids ──────────────────────────────────────────
    idx = np.arange(1, int(m.max()) + 1)
    cen = ndimage.center_of_mass(m > 0, m, idx)
    keep = [i for i, c in enumerate(cen) if np.isfinite(c[0])]
    cen = np.asarray([cen[i] for i in keep])
    plab = idx[keep]
    px = (cen[:, 1] + x0) * BIN_UM
    py = (cen[:, 0] + y0) * BIN_UM
    log(f"\n[predicted] centroids {len(px):,}   "
        f"x[{px.min():.1f},{px.max():.1f}] y[{py.min():.1f},{py.max():.1f}] um")

    gx = gt.x.to_numpy(float) * BIN_UM
    gy = gt.y.to_numpy(float) * BIN_UM
    r = a.match_radius or float(np.sqrt(
        (gx.max() - gx.min()) * (gy.max() - gy.min()) / max(len(gt), 1)))
    log(f"\n[matching] KD-tree one-to-one, radius {r:.2f} um")
    dist, gi = cKDTree(np.c_[gx, gy]).query(np.c_[px, py], k=1,
                                            distance_upper_bound=r)
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
    prec = matched / max(len(px), 1)
    rec = matched / max(len(gt), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    dd = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    log(f"  pred {len(px):,}  gt {len(gt):,}  matched {matched:,}")
    log(f"  precision {prec:.4f}  recall {rec:.4f}  f1 {f1:.4f}")
    log(f"  loc mean {np.nanmean(dd):.3f}  median {np.nanmedian(dd):.3f}  "
        f"p95 {np.nanpercentile(dd,95):.3f} um")

    res = dict(method=METHOD, dataset=DATASET,
               gt_cells=len(gt), pred_cells=len(px), matched=matched,
               cell_ratio=len(px) / max(len(gt), 1),
               precision=prec, recall=rec, f1=f1,
               loc_mean_um=float(np.nanmean(dd)),
               loc_median_um=float(np.nanmedian(dd)),
               loc_p95_um=float(np.nanpercentile(dd, 95)),
               match_radius_um=r,
               median_diameter_um=float(2*np.sqrt(q[2]/np.pi)*BIN_UM),
               median_area_um2=float(q[2] * BIN_UM ** 2),
               frag_lt10=float((area < 10).mean()),
               uses_platform_prior=True, detection_comparable=True,
               data_error=False)

    # ── transcript-level assignment ──────────────────────────────────
    log(f"\n[assignment] project each GEM bin onto the mask")
    H, W = m.shape
    bx = (d.x.to_numpy(int) - x0)
    by = (d.y.to_numpy(int) - y0)
    inside = (bx >= 0) & (bx < W) & (by >= 0) & (by < H)
    plab_at = np.zeros(len(d), np.int64)
    plab_at[inside] = m[by[inside], bx[inside]]
    w = d["count"].to_numpy(float)
    assigned = plab_at > 0
    cov = float(w[assigned].sum() / w.sum())
    log(f"  bins inside mask {inside.mean()*100:.1f}%   "
        f"assigned transcript fraction {cov*100:.1f}%")
    res["coverage_fraction"] = cov
    res["assign_rate"] = cov

    lab2pi = {int(l): i for i, l in enumerate(plab)}
    p2g = {p[0]: p[1] for p in pairs}
    gt_cells_arr = gt.index.to_numpy()
    g2row = {int(c): i for i, c in enumerate(gt_cells_arr)}
    truth = d["cell"].to_numpy(int)
    ok2 = assigned & (truth > 0)
    if ok2.sum() > 100:
        pred_row = np.array([lab2pi.get(int(l), -1) for l in plab_at[ok2]])
        want = np.array([p2g.get(int(pr), -2) for pr in pred_row])
        real = np.array([g2row.get(int(t), -3) for t in truth[ok2]])
        acc = float(np.average(want == real, weights=w[ok2]))
        res["assign_accuracy"] = acc
        res["overall_correct"] = acc * cov
        log(f"  assign_accuracy {acc:.4f}  (MIDCount-weighted, "
            f"n={int(ok2.sum()):,} bins)")
        log(f"  overall_correct  {res['overall_correct']:.4f}")
        log(f"  ⚠ prior and GT share the same source; this metric is circularly inflated, not directly comparable with prior-free methods")

    # ── counts + expression vectors ─────────────────────────────────
    if pairs:
        pn = np.zeros(len(px))
        np.add.at(pn, [lab2pi.get(int(l), 0) for l in plab_at[assigned]],
                  w[assigned])
        pi = np.array([p[0] for p in pairs])
        gj = np.array([p[1] for p in pairs])
        A = pn[pi]
        B = gt.n.to_numpy(float)[gj]
        k = np.isfinite(A) & np.isfinite(B) & (A > 0) & (B > 0)
        if k.sum() >= 5:
            res.update(count_pearson=float(pearsonr(A[k], B[k])[0]),
                       count_spearman=float(spearmanr(A[k], B[k])[0]),
                       count_mae=float(np.abs(A[k]-B[k]).mean()),
                       count_rmse=float(np.sqrt(((A[k]-B[k])**2).mean())))
            log(f"\n[counts] n={int(k.sum()):,}  "
                f"pearson {res['count_pearson']:.4f}  "
                f"spearman {res['count_spearman']:.4f}  "
                f"mae {res['count_mae']:.1f}  rmse {res['count_rmse']:.1f}")

        log(f"\n[expression vectors] building pred / GT count matrices ...")
        genes = sorted(d.gene.astype(str).unique())
        gidx = {v: i for i, v in enumerate(genes)}
        gcode = d.gene.astype(str).map(gidx).to_numpy()
        from scipy.sparse import coo_matrix
        prow = np.array([lab2pi.get(int(l), -1) for l in plab_at])
        vp = assigned & (prow >= 0)
        PV = coo_matrix((w[vp], (prow[vp], gcode[vp])),
                        shape=(len(px), len(genes))).tocsr()
        grow = np.array([g2row.get(int(t), -1) for t in truth])
        vg = grow >= 0
        GV = coo_matrix((w[vg], (grow[vg], gcode[vg])),
                        shape=(len(gt), len(genes))).tocsr()
        Aa = np.asarray(PV[pi].todense(), float)
        Bb = np.asarray(GV[gj].todense(), float)
        na, nb = np.linalg.norm(Aa, 1 if False else None, axis=1), None
        na = np.linalg.norm(Aa, axis=1)
        nb = np.linalg.norm(Bb, axis=1)
        kk = (na > 0) & (nb > 0)
        if kk.sum() >= 5:
            res["vec_cosine"] = float(np.mean(
                (Aa[kk] * Bb[kk]).sum(1) / (na[kk] * nb[kk])))
            Ap = Aa[kk] / Aa[kk].sum(1, keepdims=True)
            Bp = Bb[kk] / Bb[kk].sum(1, keepdims=True)
            M = (Ap + Bp) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                kl1 = np.nansum(Ap * np.log2(Ap / M), axis=1)
                kl2 = np.nansum(Bp * np.log2(Bp / M), axis=1)
            res["vec_js_dist"] = float(np.mean(np.sqrt((kl1 + kl2) / 2)))
            nn = min(2000, int(kk.sum()))
            res["vec_pearson"] = float(np.mean(
                [pearsonr(Aa[kk][i], Bb[kk][i])[0] for i in range(nn)]))
            log(f"  vec_cosine {res['vec_cosine']:.4f}  "
                f"vec_js_dist {res['vec_js_dist']:.4f}  "
                f"vec_pearson {res['vec_pearson']:.4f}  "
                f"(n={int(kk.sum()):,}, {len(genes):,} genes)")

    res["note"] = ("nuclei_mask rasterized from the GEM cell column, same source as GT -> "
                   "assignment/coverage circularly inflated, only detection/localization/count/morphology valid; "
                   f"mask={a.mask}")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"ucs_{DATASET}_metrics.csv")
    pd.DataFrame([res]).to_csv(f, index=False)
    pd.DataFrame(pairs, columns=["pred_idx", "gt_idx", "dist_um"]).to_csv(
        os.path.join(a.out, f"ucs_{DATASET}_pairs.csv"), index=False)
    log("\n" + "=" * 78)
    for k, v in res.items():
        if k == "note":
            continue
        log(f"  {k:<24} {v}")
    log("=" * 78)
    log(f"-> {f}")


if __name__ == "__main__":
    main()
