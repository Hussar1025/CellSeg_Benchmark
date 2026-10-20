#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_genesegnet.py -- unified evaluation of GeneSegNet segmentation results

Input: stitched mask produced by genesegnet_xenium.py + prepared/prep_meta.json
Output: one CSV row whose column names exactly match the summary table, plus per-cell details

Metric definitions (consistent with benchmark_full_all.xlsx)
  detection  gt_cells / pred_cells / cell_ratio / matched / precision / recall / f1
          match = one-to-one greedy within KD-tree radius (occupy by ascending distance), no direct ID link
  location   loc_mean_um / loc_median_um / loc_p95_um   centroid distance of matched pairs
  count      count_pearson / count_spearman / count_mae / count_rmse
          transcripts per pred cell vs per GT cell on matched pairs
  vector     vec_cosine / vec_js_dist / vec_pearson
          mean pairwise similarity of gene expression vectors on matched pairs
  assignment assign_rate      fraction of transcripts assigned to a pred cell
         assign_accuracy  among transcripts "assigned to both GT and pred",
                          fraction where the pred cell is exactly the matched pair of the GT cell
         overall_correct  = assign_rate x assign_accuracy

Independence flags
  GeneSegNet inference does not use platform priors (nucleus prior / GT segmentation),
  so uses_platform_prior=FALSE and all three independent flags are TRUE.

Usage
  python eval_genesegnet.py --dataset xenium_2
  python eval_genesegnet.py --dataset xenium_5 --match-radius 12
  python eval_genesegnet.py --all
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

CTRL_PREFIX = ("NegControlProbe", "NegControlCodeword", "NegPrb", "BLANK",
               "Blank", "antisense", "UnassignedCodeword",
               "DeprecatedCodeword", "Intergenic", "Genomic")
UNASSIGNED = ("UNASSIGNED", "unassigned", "-1", "", "0")

COLS = ["method", "dataset", "gt_cells", "pred_cells", "cell_ratio", "matched",
        "precision", "recall", "f1", "loc_mean_um", "loc_median_um",
        "loc_p95_um", "count_pearson", "count_spearman", "count_mae",
        "count_rmse", "vec_cosine", "vec_js_dist", "vec_pearson",
        "assign_accuracy", "assign_rate", "overall_correct",
        "median_diameter_um", "uses_platform_prior", "assignment_independent",
        "vector_independent", "note"]


def log(m):
    print(m, flush=True)


# ══════════════════════════════════════════════════════════════════════════
def mask_stats(mask):
    """Tile-wise per-label area and centroid stats (avoids flattening the whole image into memory)."""
    H, W = mask.shape
    n = int(mask.max()) + 1
    area = np.zeros(n, np.int64)
    sy = np.zeros(n, np.float64)
    sx = np.zeros(n, np.float64)
    step = max(1, 40_000_000 // max(W, 1))
    cols = np.arange(W, dtype=np.float64)
    for r0 in range(0, H, step):
        blk = mask[r0:r0 + step]
        f = blk.ravel()
        area += np.bincount(f, minlength=n)
        rows = np.repeat(np.arange(r0, r0 + blk.shape[0], dtype=np.float64),
                         W)
        sy += np.bincount(f, weights=rows, minlength=n)
        sx += np.bincount(f, weights=np.tile(cols, blk.shape[0]),
                          minlength=n)
    lab = np.nonzero(area)[0]
    lab = lab[lab > 0]
    return lab, area[lab], sy[lab] / area[lab], sx[lab] / area[lab]


def greedy_match(pred_xy, gt_xy, radius):
    """One-to-one greedy matching within radius (occupy by ascending distance)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    tg = cKDTree(gt_xy)
    k = min(8, len(gt_xy))
    dist, idx = tg.query(pred_xy, k=k, distance_upper_bound=radius)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    pi = np.repeat(np.arange(len(pred_xy)), k)
    gi, dd = idx.ravel(), dist.ravel()
    ok = np.isfinite(dd) & (gi < len(gt_xy))
    pi, gi, dd = pi[ok], gi[ok], dd[ok]
    order = np.argsort(dd, kind="stable")
    up = np.zeros(len(pred_xy), bool)
    ug = np.zeros(len(gt_xy), bool)
    pairs, dists = [], []
    for j in order:
        a, b = pi[j], gi[j]
        if up[a] or ug[b]:
            continue
        up[a] = ug[b] = True
        pairs.append((a, b))
        dists.append(dd[j])
    return np.asarray(pairs, int).reshape(-1, 2), np.asarray(dists)


def profile(cell_idx, gene_idx, n_cell, n_gene):
    m = (cell_idx >= 0) & (gene_idx >= 0)
    return sparse.coo_matrix(
        (np.ones(int(m.sum()), np.float32), (cell_idx[m], gene_idx[m])),
        shape=(n_cell, n_gene)).tocsr()


def vector_metrics(A, B):
    """Compare two csr expression vectors row by row. Returns mean cosine / JS distance / pearson."""
    A = A.toarray().astype(np.float64)
    B = B.toarray().astype(np.float64)
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    ok = (na > 0) & (nb > 0)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    A, B = A[ok], B[ok]
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1)
                            * np.linalg.norm(B, axis=1))
    P = A / A.sum(1, keepdims=True)
    Q = B / B.sum(1, keepdims=True)
    M = 0.5 * (P + Q)

    def kl(x, y):
        m = x > 0
        out = np.zeros(len(x))
        xr, yr = np.where(m, x, 1.0), np.where(m, y, 1.0)
        return (np.where(m, xr * np.log2(xr / yr), 0.0)).sum(1)

    js = 0.5 * kl(P, M) + 0.5 * kl(Q, M)
    jsd = np.sqrt(np.clip(js, 0, 1))
    Ac = A - A.mean(1, keepdims=True)
    Bc = B - B.mean(1, keepdims=True)
    den = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    pr = np.where(den > 0, (Ac * Bc).sum(1) / np.where(den > 0, den, 1), np.nan)
    return (float(np.mean(cos)), float(np.mean(jsd)),
            float(np.nanmean(pr)))


# ══════════════════════════════════════════════════════════════════════════
def evaluate(ds, args):
    import tifffile
    out = Path(args.out_root) / f"genesegnet_{ds}"
    meta = json.loads((out / "prepared" / "prep_meta.json").read_text())
    mpath = out / f"genesegnet_{ds}_mask.tif"
    if not mpath.exists():
        raise SystemExit(f"✗ no mask: {mpath}")

    ps = float(meta["pixel_size_um"])
    y0, x0 = int(meta["roi_y0"]), int(meta["roi_x0"])
    rh, rw = int(meta["roi_h"]), int(meta["roi_w"])
    # benchmark GT / GeneSegNet mask are ROI-local.
    # full Xenium transcripts/cells are global coordinates.
    cmode = "local_gt_global_tx"
    ox = oy = 0.0
    gx0 = gy0 = 0

    log(f"\n{'='*74}\nGeneSegNet @ {ds}   ROI {rh}x{rw} px  {ps} um/px  "
        f"coord={cmode}")

    # ---------- 1. Pred cells ----------
    mask = tifffile.imread(str(mpath))
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask = mask.astype(np.int64, copy=False)
    mh, mw = mask.shape
    if (mh, mw) != (rh, rw):
        log(f"  ⚠ mask {mh}x{mw} differs from meta ROI {rh}x{rw}"
            f" (stitch_fov takes scalar size={min(rh, rw)}, output squeezed to a square),"
            f" using mask as reference")
        rh, rw = mh, mw
    lab, area_px, cy, cx = mask_stats(mask)
    pred = pd.DataFrame(dict(
        label=lab, area_px=area_px,
        x_um=cx * ps, y_um=cy * ps,
        area_um2=area_px * ps * ps))
    pred["diameter_um"] = 2 * np.sqrt(pred.area_um2 / np.pi)
    if args.min_area_um2 > 0:
        keep = pred.area_um2 >= args.min_area_um2
        log(f"  filter by min_area_um2={args.min_area_um2}, removed "
            f"{int((~keep).sum()):,} fragments")
        pred = pred[keep].reset_index(drop=True)
    log(f"  pred cells {len(pred):,}   median diameter "
        f"{pred.diameter_um.median():.2f} um")

    # ---------- 2. GT cells ----------
    g = pd.read_parquet(meta["gt"])
    gxc = next(c for c in ("x_centroid", "x", "center_x") if c in g.columns)
    gyc = next(c for c in ("y_centroid", "y", "center_y") if c in g.columns)
    gidc = next((c for c in ("cell_id", "cell", "id") if c in g.columns), None)
    # benchmark cells.parquet coordinates are already ROI-local um
    gpx = g[gxc].to_numpy(float) / ps
    gpy = g[gyc].to_numpy(float) / ps
    inroi = (gpx >= 0) & (gpx < rw) & (gpy >= 0) & (gpy < rh)
    g = g.loc[inroi].reset_index(drop=True)
    log(f"  GT cells (in ROI) {len(g):,}")
    if len(g) == 0:
        raise SystemExit("✗ GT in ROI is 0, coordinate mode is wrong")

    # ---------- 3. Match ----------
    r = args.match_radius if args.match_radius > 0 \
        else float(np.nanmedian(pred.diameter_um)) * 0.8
    pairs, dists = greedy_match(pred[["x_um", "y_um"]].to_numpy(),
                                g[[gxc, gyc]].to_numpy(), r)
    matched = len(pairs)
    prec = matched / max(len(pred), 1)
    rec = matched / max(len(g), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    log(f"  match radius {r:.2f} um   matched {matched:,}   "
        f"P {prec:.4f}  R {rec:.4f}  F1 {f1:.4f}")
    log(f"  centroid distance mean {dists.mean():.3f}  median "
        f"{np.median(dists):.3f}  p95 {np.percentile(dists,95):.3f} um")

    # ---------- 4. Transcripts ----------
    import pyarrow.dataset as pds
    d = pds.dataset(meta["tx"], format="parquet")
    names = list(d.schema.names)
    xc = "x_location" if "x_location" in names else "x"
    yc = "y_location" if "y_location" in names else "y"
    gc = next((c for c in ("feature_name", "gene", "target")
               if c in names), None)
    idc = next((c for c in ("cell_id", "cell") if c in names), None)
    qvc = "qv" if "qv" in names else None

    # full transcript parquet is global um; ROI origin is stored in full-image pixels
    xlo, xhi = x0 * ps, (x0 + rw) * ps
    ylo, yhi = y0 * ps, (y0 + rh) * ps
    f = ((pds.field(xc) >= xlo) & (pds.field(xc) < xhi) &
         (pds.field(yc) >= ylo) & (pds.field(yc) < yhi))
    if qvc:
        f = f & (pds.field(qvc) >= args.qv_min)
    cols = [c for c in (xc, yc, gc, idc) if c]
    t = d.to_table(columns=cols, filter=f).to_pandas()
    if gc and len(t):
        if isinstance(t[gc].iloc[0], bytes):
            t[gc] = t[gc].str.decode("utf-8")
        t[gc] = t[gc].astype(str)
        t = t[~t[gc].str.startswith(CTRL_PREFIX)].reset_index(drop=True)
    log(f"  transcripts {len(t):,}   genes {t[gc].nunique():,}")

    px = np.floor(t[xc].to_numpy(float) / ps).astype(np.int64) - x0
    py = np.floor(t[yc].to_numpy(float) / ps).astype(np.int64) - y0
    ok = (px >= 0) & (px < rw) & (py >= 0) & (py < rh)
    t, px, py = t.loc[ok].reset_index(drop=True), px[ok], py[ok]

    # pred assignment: mask value -> pred row index
    lab2row = np.full(int(mask.max()) + 1, -1, np.int64)
    lab2row[pred.label.to_numpy()] = np.arange(len(pred))
    pred_cell = lab2row[mask[py, px]]
    assign_rate = float((pred_cell >= 0).mean())
    log(f"  pred assign rate {assign_rate:.4f}")

    # GT assignment:
    # benchmark ROI cell_id was regenerated, so first recover the corresponding
    # full-section official cell_id by spatially matching centroids.
    full_cells = Path(meta["tx"]).parent / "cells.parquet"
    if idc is None or not full_cells.exists():
        log("  ⚠ no transcript cell_id or full cells.parquet, skip GT expression metrics")
        gt_cell = np.full(len(t), -1, np.int64)
    else:
        fg = pd.read_parquet(full_cells)
        fxc = next(c for c in ("x_centroid", "x", "center_x") if c in fg.columns)
        fyc = next(c for c in ("y_centroid", "y", "center_y") if c in fg.columns)
        fidc = next(c for c in ("cell_id", "cell", "id") if c in fg.columns)

        from scipy.spatial import cKDTree
        tree = cKDTree(fg[[fxc, fyc]].to_numpy(float))

        tgt = g[[gxc, gyc]].to_numpy(float).copy()
        tgt[:,0] += x0 * ps
        tgt[:,1] += y0 * ps

        dd, ii = tree.query(tgt, k=1)
        good = dd <= 3.0
        frac = float(good.mean()) if len(good) else 0.0
        log(f"  GT local->global cell mapping {good.sum():,}/{len(g):,} "
            f"({frac:.4f}), median_dist={np.median(dd):.3f} um")

        if frac < 0.80:
            raise SystemExit("✗ benchmark GT cannot be reliably mapped to full cells.parquet")

        official_to_gt = {
            str(fg.iloc[j][fidc]): i
            for i, j in enumerate(ii)
            if good[i]
        }

        ids = t[idc]
        if len(ids) and isinstance(ids.iloc[0], bytes):
            ids = ids.str.decode("utf-8")
        ids = ids.astype(str)

        gt_cell = ids.map(
            lambda v: -1 if v in UNASSIGNED else official_to_gt.get(v, -1)
        ).to_numpy(np.int64)

        log(f"  GT assign rate {float((gt_cell>=0).mean()):.4f}")

    gene_idx, gene_names = pd.factorize(t[gc])
    ng = len(gene_names)

    # ---------- 5. Count / vector / assignment ----------
    res = dict(count_pearson=np.nan, count_spearman=np.nan, count_mae=np.nan,
               count_rmse=np.nan, vec_cosine=np.nan, vec_js_dist=np.nan,
               vec_pearson=np.nan, assign_accuracy=np.nan,
               overall_correct=np.nan)
    if matched and (gt_cell >= 0).any():
        pi, gi = pairs[:, 0], pairs[:, 1]
        np_cnt = np.bincount(pred_cell[pred_cell >= 0],
                             minlength=len(pred))[pi]
        ng_cnt = np.bincount(gt_cell[gt_cell >= 0], minlength=len(g))[gi]
        m = (np_cnt + ng_cnt) > 0
        if m.sum() > 2:
            res["count_pearson"] = float(pearsonr(np_cnt[m], ng_cnt[m])[0])
            res["count_spearman"] = float(spearmanr(np_cnt[m], ng_cnt[m])[0])
        res["count_mae"] = float(np.mean(np.abs(np_cnt - ng_cnt)))
        res["count_rmse"] = float(np.sqrt(np.mean((np_cnt - ng_cnt) ** 2)))

        Ap = profile(pred_cell, gene_idx, len(pred), ng)[pi]
        Ag = profile(gt_cell, gene_idx, len(g), ng)[gi]
        res["vec_cosine"], res["vec_js_dist"], res["vec_pearson"] = \
            vector_metrics(Ap, Ag)

        # assignment accuracy: GT cell -> its matched pred cell
        g2p = np.full(len(g), -1, np.int64)
        g2p[gi] = pi
        both = (pred_cell >= 0) & (gt_cell >= 0)
        if both.sum():
            res["assign_accuracy"] = float(
                (pred_cell[both] == g2p[gt_cell[both]]).mean())
            res["overall_correct"] = res["assign_accuracy"] * assign_rate
        log(f"  count_pearson {res['count_pearson']:.4f}   "
            f"vec_cosine {res['vec_cosine']:.4f}   "
            f"assign_acc {res['assign_accuracy']:.4f}   "
            f"overall {res['overall_correct']:.4f}")

    # ---------- 6. Write to disk ----------
    row = dict(
        method="GeneSegNet", dataset=ds, gt_cells=len(g), pred_cells=len(pred),
        cell_ratio=len(pred) / max(len(g), 1), matched=matched,
        precision=prec, recall=rec, f1=f1,
        loc_mean_um=float(dists.mean()) if matched else np.nan,
        loc_median_um=float(np.median(dists)) if matched else np.nan,
        loc_p95_um=float(np.percentile(dists, 95)) if matched else np.nan,
        assign_rate=assign_rate,
        median_diameter_um=float(pred.diameter_um.median()),
        uses_platform_prior="FALSE", assignment_independent="TRUE",
        vector_independent="TRUE",
        note=(f"Xenium pancreas self-trained weights, held-out inference; "
              f"heat={meta.get('heatmap_mode')} sigma={meta.get('heatmap_sigma')}; "
              f"dapi={meta.get('dapi_strategy')}; "
              f"ROI {rh}x{rw}px @({y0},{x0}); match_radius={r:.2f}um"),
        **res)
    df = pd.DataFrame([row])[COLS]
    df.to_csv(out / f"eval_genesegnet_{ds}.csv", index=False)

    det = pred.copy()
    det["matched_gt"] = -1
    det["match_dist_um"] = np.nan
    if matched:
        det.loc[pairs[:, 0], "matched_gt"] = pairs[:, 1]
        det.loc[pairs[:, 0], "match_dist_um"] = dists
    det["n_transcripts"] = np.bincount(pred_cell[pred_cell >= 0],
                                       minlength=len(pred))
    det.to_csv(out / f"eval_genesegnet_{ds}_cells.csv", index=False)
    log(f"  -> {out}/eval_genesegnet_{ds}.csv")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out-root", default="yourpath")
    p.add_argument("--match-radius", type=float, default=0.0,
                   help="um; 0 = 0.8x median pred diameter")
    p.add_argument("--min-area-um2", type=float, default=0.0,
                   help="if >0, filter out fragments by area")
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--summary",
                   default="yourpath/genesegnet_eval_summary.csv")
    a = p.parse_args()

    dss = (["xenium_2", "xenium_3", "xenium_5", "xenium_6"] if a.all
           else [a.dataset])
    if not dss or dss == [None]:
        raise SystemExit("need --dataset or --all")

    rows = []
    for ds in dss:
        try:
            rows.append(evaluate(ds, a))
        except SystemExit as e:
            log(f"[{ds}] skip: {e}")
        except Exception as e:
            log(f"[{ds}] failed: {type(e).__name__}: {e}")
    if rows:
        df = pd.DataFrame(rows)[COLS]
        df.to_csv(a.summary, index=False)
        log(f"\n{'='*74}")
        print(df[["dataset", "gt_cells", "pred_cells", "cell_ratio", "f1",
                  "loc_median_um", "count_pearson", "vec_cosine",
                  "assign_rate", "median_diameter_um"]].to_string(index=False))
        log(f"\nsummary -> {a.summary}")


if __name__ == "__main__":
    main()