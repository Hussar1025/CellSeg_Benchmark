#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_comseg_xenium.py   v2026.08.16

ComSeg on the Xenium datasets that produced a raster: xenium_4 (liver) and
xenium_5 (mouse brain).  Same 19 metrics as every other row in the workbook.

    python eval_comseg_xenium.py --dataset liver
    python eval_comseg_xenium.py --dataset mouse_brain
    python eval_comseg_xenium.py --dataset liver --method UCS \
        --pred yourpath/ucs_xenium_liver_roi10000_prior/ucs_segmentation.tif

The ROI is the one every other method used on that section, written down here
and re-derived from the raster on the way in.

xenium_2 (breast) and xenium_3 (lymph) are deliberately absent.  ComSeg ran on
those two but wrote h5ad plus *_tx.csv under comseg_xenium/, not a raster, and
in a noprior configuration; their ROI is not recorded anywhere I can read.  A
loader for that format needs the column names and the ROI first -- see the note
this script prints when asked for them.

--fill-um matters here.  ComSeg labels transcripts, not pixels, so a raster
built straight from its output is a point cloud.  The runner writes the filled
version; if you point --pred at an unfilled one, set --fill-um to whatever the
runner used or the areas will be far too small.
"""

from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = {
    "liver": dict(
        root="yourpath/dataset/xenium_liver",
        run="yourpath/comseg_xenium_liver_roi10000",
        label="xenium_4", roi=(13704, 21974, 10000)),
    "mouse_brain": dict(
        root="yourpath/dataset/xenium_mouse_brain",
        run="yourpath/comseg_xenium_mouse_brain_roi10000",
        label="xenium_5", roi=(6956, 12077, 10000)),
}
# xenium_2 / xenium_3 were run in a different shape: no raster, a noprior
# configuration, and the output is an h5ad plus a per-transcript csv
#     tx.csv   x,y,gene,cell_id,_landmark_for_centroid
#     h5ad obs CellID, Name, centroid_z, centroid_y, centroid_x, image_name
# There is no recorded ROI, so it is taken as the bounding box of the
# transcripts that run was actually given, and the unit (um or native px) is
# decided by comparing that box against the platform centroids.
CSV_DATASETS = {
    "breast": dict(
        root="yourpath/dataset/xenium_breast",
        run="yourpath/comseg_xenium/xenium_breast",
        label="xenium_2", prior=False),
    "lymph": dict(
        root="yourpath/dataset/xenium_lymph",
        run="yourpath/comseg_xenium/xenium_lymph",
        label="xenium_3", prior=False),
}
OUT = "yourpath/eval_results_qv20"
PIXEL_SIZE = 0.2125
METHOD = "ComSeg"
# ---- metric core (identical in every evaluator; kept inline so each file runs
# ---- on its own) --------------------------------------------------------- #

def log(m=""):
    print(f"{time.strftime('%H:%M:%S')}  {m}", flush=True)


def banner(m):
    log("=" * 78); log(m); log("=" * 78)


def props(lab):
    """centroid (x, y) in raster px and pixel area, per label."""
    a = np.asarray(lab)
    flat = a.reshape(-1).astype(np.int64)
    n = int(flat.max())
    if n == 0:
        return np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros(0)
    area = np.bincount(flat, minlength=n + 1)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    sy = np.bincount(flat, weights=yy.reshape(-1), minlength=n + 1)
    sx = np.bincount(flat, weights=xx.reshape(-1), minlength=n + 1)
    ids = np.nonzero(area)[0]
    ids = ids[ids > 0]
    cen = np.stack([sx[ids] / area[ids], sy[ids] / area[ids]], 1)
    return ids, cen, area[ids].astype(float)


def match_radius(gt_xy):
    from scipy.spatial import cKDTree
    if len(gt_xy) < 2:
        return 20.0
    d, _ = cKDTree(gt_xy).query(gt_xy, k=2)
    return float(np.percentile(d[:, 1], 95) * 1.25)


def sparse_hungarian(pred_xy, gt_xy, radius):
    """Pairs within `radius`, solved exactly inside each connected component of
    the candidate graph; components above `BIG` fall back to greedy."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import linear_sum_assignment
    BIG = 3000
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    pairs = cKDTree(pred_xy).query_ball_tree(cKDTree(gt_xy), radius)
    pi, gi = [], []
    for p, gs in enumerate(pairs):
        for g in gs:
            pi.append(p); gi.append(g)
    if not pi:
        return np.zeros((0, 2), int), np.zeros(0)
    pi = np.asarray(pi); gi = np.asarray(gi)
    d = np.linalg.norm(pred_xy[pi] - gt_xy[gi], axis=1)
    npred, ngt = len(pred_xy), len(gt_xy)
    A = coo_matrix((np.ones(len(pi)), (pi, gi + npred)),
                   shape=(npred + ngt, npred + ngt))
    ncc, lbl = connected_components(A + A.T, directed=False)
    out_p, out_g, out_d = [], [], []
    big = 0
    for c in range(ncc):
        m = lbl[pi] == c
        if not m.any():
            continue
        sp = np.unique(pi[m]); sg = np.unique(gi[m])
        if len(sp) * len(sg) > BIG * BIG:
            big += 1
            order = np.argsort(d[m])
            up, ug = set(), set()
            for k in np.nonzero(m)[0][order]:
                if pi[k] in up or gi[k] in ug:
                    continue
                up.add(pi[k]); ug.add(gi[k])
                out_p.append(pi[k]); out_g.append(gi[k]); out_d.append(d[k])
            continue
        rp = {v: i for i, v in enumerate(sp)}
        rg = {v: i for i, v in enumerate(sg)}
        C = np.full((len(sp), len(sg)), 1e6)
        for k in np.nonzero(m)[0]:
            C[rp[pi[k]], rg[gi[k]]] = d[k]
        r, cidx = linear_sum_assignment(C)
        for a, b in zip(r, cidx):
            if C[a, b] < 1e5:
                out_p.append(sp[a]); out_g.append(sg[b]); out_d.append(C[a, b])
    if big:
        log(f"    {big} very large component(s) matched greedily")
    return np.stack([np.asarray(out_p), np.asarray(out_g)], 1), np.asarray(out_d)


def pair_stats(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return dict(pearson=np.nan, spearman=np.nan, mae=np.nan, rmse=np.nan)
    from scipy.stats import pearsonr, spearmanr
    return dict(pearson=float(pearsonr(a, b)[0]),
                spearman=float(spearmanr(a, b)[0]),
                mae=float(np.mean(np.abs(a - b))),
                rmse=float(np.sqrt(np.mean((a - b) ** 2))))


def vector_stats(P, G):
    """cosine / Jensen-Shannon / pearson between paired gene vectors."""
    out = dict(cosine=[], js=[], pearson=[])
    for p, g in zip(P, G):
        p = np.asarray(p, float); g = np.asarray(g, float)
        if p.sum() <= 0 or g.sum() <= 0:
            continue
        out["cosine"].append(float(p @ g / (np.linalg.norm(p) * np.linalg.norm(g))))
        pp, gg = p / p.sum(), g / g.sum()
        mm = 0.5 * (pp + gg)
        def kl(u, v):
            m = u > 0
            return float(np.sum(u[m] * np.log2(u[m] / v[m])))
        out["js"].append(float(np.sqrt(max(0.5 * kl(pp, mm) + 0.5 * kl(gg, mm), 0))))
        if p.std() > 0 and g.std() > 0:
            out["pearson"].append(float(np.corrcoef(p, g)[0, 1]))
    return (float(np.mean(out["cosine"])) if out["cosine"] else np.nan,
            float(np.mean(out["js"])) if out["js"] else np.nan,
            float(np.mean(out["pearson"])) if out["pearson"] else np.nan,
            len(out["cosine"]))


def cell_by_gene(labels, gene_idx, n_genes, n_cells_max):
    """rows are label-1, columns are gene index; labels of 0 are dropped."""
    from scipy.sparse import coo_matrix
    m = labels > 0
    if n_cells_max < 1:
        return coo_matrix((1, n_genes)).tocsr()
    return coo_matrix((np.ones(int(m.sum()), np.float32),
                       (labels[m] - 1, gene_idx[m])),
                      shape=(n_cells_max, n_genes)).tocsr()


# --------------------------------------------------------------------------- #

def locate(root: Path, names) -> Path | None:
    """Xenium bundles come either flat or with everything under outs/."""
    for d in (root, root / "outs"):
        for n in names:
            if (d / n).exists():
                return d / n
    for n in names:
        c = sorted(root.rglob(n))
        if c:
            return c[0]
    return None


def read_pixel_size(root: Path) -> float:
    for n in ("experiment.xenium", "experiment.json"):
        p = locate(root, [n])
        if p is None:
            continue
        try:
            d = json.load(open(p))
            for k in ("pixel_size", "pixel_size_um"):
                if k in d:
                    return float(d[k])
        except Exception:
            pass
    return PIXEL_SIZE


def find_pred(run: Path, explicit) -> Path:
    if explicit:
        return Path(explicit)
    for n in ("comseg_segmentation.tif", "comseg_labels_native.tif",
              "comseg_labels.tif"):
        if (run / n).exists():
            return run / n
    c = sorted(run.glob("*segmentation*.tif")) + sorted(run.glob("*labels*.tif"))
    if c:
        return c[0]
    sys.exit(f"no raster under {run}.  If the tiles finished but post skipped "
             f"them, run comseg_salvage.py first, then the runner's --stage post")


def read_transcripts(root: Path, roi, ps: float, qv: float,
                     cache: Path) -> pd.DataFrame:
    y0, x0, S = roi
    cache.mkdir(parents=True, exist_ok=True)
    cp = cache / f"tx_roi_{y0}_{x0}_{S}_qv{int(qv)}.parquet"
    if cp.exists():
        log(f"  cache       {cp.name}")
        return pd.read_parquet(cp)
    import pyarrow.parquet as pq
    src = locate(root, ["transcripts.parquet", "transcripts.csv.gz",
                        "transcripts.csv"])
    if src is None:
        sys.exit(f"no transcripts table under {root} or {root}/outs")
    log(f"  transcripts {src}")
    xm0, xm1 = x0 * ps, (x0 + S) * ps
    ym0, ym1 = y0 * ps, (y0 + S) * ps
    want = ["x_location", "y_location", "feature_name", "qv", "is_gene",
            "cell_id", "overlaps_nucleus"]
    parts, tot = [], 0
    if src.suffix == ".parquet":
        pf = pq.ParquetFile(src)
        cols = [c for c in pf.schema.names if c in want]
        it = (b.to_pandas() for b in pf.iter_batches(batch_size=2_000_000,
                                                     columns=cols))
    else:
        it = pd.read_csv(src, chunksize=2_000_000)
    for d in it:
        tot += len(d)
        d = d[(d.x_location >= xm0) & (d.x_location < xm1) &
              (d.y_location >= ym0) & (d.y_location < ym1)]
        if "qv" in d.columns:
            d = d[d.qv >= qv]
        if "is_gene" in d.columns:
            d = d[d.is_gene.astype(bool)]
        else:
            fn = d.feature_name.astype(str)
            d = d[~fn.str.startswith(("NegControl", "BLANK",
                                      "UnassignedCodeword", "antisense",
                                      "DeprecatedCodeword"))]
        if len(d):
            parts.append(d)
    log(f"  scanned {tot:,} transcript rows")
    t = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"px": t.x_location.to_numpy() / ps - x0,
                        "py": t.y_location.to_numpy() / ps - y0,
                        "gene": t.feature_name.astype(str).to_numpy()})
    if "cell_id" in t.columns:
        out["cell_id"] = t.cell_id.astype(str).to_numpy()
    out.to_parquet(cp)
    return out


def gt_cells(root: Path, roi, ps: float) -> pd.DataFrame:
    y0, x0, S = roi
    p = locate(root, ["cells.parquet", "cells.csv.gz", "cells.csv"])
    if p is None:
        sys.exit(f"no cells table under {root} or {root}/outs")
    log(f"  cells       {p}")
    d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    px = d.x_centroid.to_numpy() / ps - x0
    py = d.y_centroid.to_numpy() / ps - y0
    m = (px >= 0) & (px < S) & (py >= 0) & (py < S)
    out = pd.DataFrame({"cell_id": d.cell_id.astype(str).to_numpy()[m],
                        "px": px[m], "py": py[m]})
    for c in ("transcript_counts", "total_counts"):
        if c in d.columns:
            out["counts"] = d[c].to_numpy()[m]
            break
    return out


# --------------------------------------------------------------------------- #
# csv mode, for the runs that produced *_tx.csv instead of a raster
# --------------------------------------------------------------------------- #

def csv_mode(name: str, args) -> dict:
    cfg = CSV_DATASETS[name]
    root = Path(args.root or cfg["root"])
    run = Path(args.run_dir or cfg["run"])
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    banner(f"{args.method} x XENIUM {name} ({cfg['label']})  csv mode")
    txp = sorted(run.rglob("*_tx.csv"))
    if not txp:
        sys.exit(f"no *_tx.csv under {run}")
    # prefer the plain noprior table over the *_v2_* variants
    txp = sorted(txp, key=lambda p: ("_v2_" in p.name, len(p.name)))[0]
    log(f"  prediction  {txp}")
    t = pd.read_csv(txp)
    for c in ("x", "y", "gene", "cell_id"):
        if c not in t.columns:
            sys.exit(f"{txp} has no column {c}; found {list(t.columns)}")
    t["cell_id"] = pd.to_numeric(t.cell_id, errors="coerce").fillna(0).astype(np.int64)
    npred = int(t.loc[t.cell_id > 0, "cell_id"].nunique())
    rate_p = float((t.cell_id > 0).mean())
    log(f"              {len(t):,} transcripts, {npred:,} predicted cells, "
        f"{rate_p*100:.1f}% assigned")

    ps = read_pixel_size(root)
    gp = locate(root, ["cells.parquet", "cells.csv.gz", "cells.csv"])
    if gp is None:
        sys.exit(f"no cells table under {root} or {root}/outs.  If the breast "
                 f"and lymph bundles live somewhere else, pass --root.")
    gd = pd.read_parquet(gp) if gp.suffix == ".parquet" else pd.read_csv(gp)
    log(f"  cells       {gp}")

    # the run has no recorded ROI; take the transcripts it saw as the window,
    # and settle um versus native px by seeing which one the platform
    # centroids fall inside
    x0, x1 = float(t.x.min()), float(t.x.max())
    y0, y1 = float(t.y.min()), float(t.y.max())
    best, unit = None, None
    for u, (cx, cy) in (("um", (gd.x_centroid.to_numpy(), gd.y_centroid.to_numpy())),
                        ("px", (gd.x_centroid.to_numpy() / ps,
                                gd.y_centroid.to_numpy() / ps))):
        n = int(((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)).sum())
        log(f"  ROI as {u:2s}  x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}] "
            f"-> {n:,} platform cells inside")
        if best is None or n > best[0]:
            best, unit = (n, cx, cy), u
    ngt, cx, cy = best
    if ngt < 50:
        sys.exit("  neither interpretation puts a sensible number of platform "
                 "cells inside that window; the coordinates are in some third "
                 "frame and need to be resolved first")
    log(f"  reading the coordinates as {unit}: {ngt:,} platform cells in the ROI")
    to_um = 1.0 if unit == "um" else ps
    m = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    g = pd.DataFrame({"cell_id": gd.cell_id.astype(str).to_numpy()[m],
                      "x": cx[m] * to_um, "y": cy[m] * to_um})
    log(f"  ROI size    {(x1-x0)*to_um:.0f} x {(y1-y0)*to_um:.0f} um")

    # predicted centroids straight from the per-transcript table
    a = t[t.cell_id > 0]
    pc = a.groupby("cell_id")[["x", "y"]].mean()
    pxy_um = pc.to_numpy() * to_um
    plab_of_row = a.cell_id.to_numpy()
    dense = {c: i + 1 for i, c in enumerate(pc.index.to_numpy())}

    # per-transcript ground truth by matching coordinates back to the platform
    # table; the csv is a subset of it, so the coordinates should agree exactly
    tp = locate(root, ["transcripts.parquet", "transcripts.csv.gz",
                       "transcripts.csv"])
    glab_row = None
    if tp is not None:
        pt = pd.read_parquet(tp) if tp.suffix == ".parquet" else pd.read_csv(tp)
        kx = pt.x_location.to_numpy() / (1.0 if unit == "um" else ps)
        ky = pt.y_location.to_numpy() / (1.0 if unit == "um" else ps)
        key = pd.DataFrame({"k": np.round(kx, 3).astype(str) + "_" +
                                 np.round(ky, 3).astype(str),
                            "cid": pt.cell_id.astype(str)})
        key = key.drop_duplicates("k").set_index("k").cid
        tk = np.round(t.x.to_numpy(), 3).astype(str) + "_" + \
             np.round(t.y.to_numpy(), 3).astype(str)
        hit = pd.Series(tk).map(key)
        rate = float(hit.notna().mean())
        log(f"  matched {rate*100:.1f}% of the csv rows back to "
            f"transcripts.parquet by coordinate")
        if rate > 0.8:
            order = {c: i + 1 for i, c in enumerate(g.cell_id.to_numpy())}
            glab_row = hit.map(order).fillna(0).astype(np.int64).to_numpy()
        else:
            log("  too low to trust; the assignment columns will be reported "
                "as missing rather than guessed")

    genes = t.gene.astype(str).to_numpy()
    uniq = np.unique(genes)
    gi = pd.Series(genes).map({x: i for i, x in enumerate(uniq)}).to_numpy()
    lp = pd.Series(t.cell_id.to_numpy()).map(dense).fillna(0).astype(np.int64).to_numpy()
    Cp = cell_by_gene(lp, gi, len(uniq), max(len(pc), 1))
    Cg = (cell_by_gene(glab_row, gi, len(uniq), max(len(g), 1))
          if glab_row is not None else None)

    banner("MATCHING")
    gxy_um = np.stack([g.x.to_numpy(), g.y.to_numpy()], 1)
    R = match_radius(gxy_um)
    log(f"  match radius {R:.2f} um")
    pairs, dist = sparse_hungarian(pxy_um, gxy_um, R)
    nm = len(pairs)
    prec = nm / max(len(pc), 1); rec = nm / max(len(g), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    log(f"  matched {nm:,} of {len(pc):,} predicted and {len(g):,} GT")
    log(f"  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    log(f"  centroid shift mean {dist.mean():.2f} median {np.median(dist):.2f} "
        f"p95 {np.percentile(dist,95):.2f} um")

    prow = pairs[:, 0] + 1; grow = pairs[:, 1] + 1
    if Cg is not None:
        cnt = pair_stats(np.asarray(Cp.sum(1)).ravel()[prow - 1],
                         np.asarray(Cg.sum(1)).ravel()[grow - 1])
        cos, js, vpe, nvec = vector_stats(Cp[prow - 1].toarray(),
                                          Cg[grow - 1].toarray())
        p2g = np.zeros(len(pc) + 1, np.int64); p2g[prow] = grow
        both = (lp > 0) & (glab_row > 0)
        acc = float((p2g[lp[both]] == glab_row[both]).mean()) if both.any() else np.nan
        nover = int(both.sum())
        log(f"  transcript accuracy {acc:.4f} over {nover:,} rows "
            f"-> overall correct {rate_p*acc:.4f}")
    else:
        cnt = dict(pearson=np.nan, spearman=np.nan, mae=np.nan, rmse=np.nan)
        cos = js = vpe = acc = np.nan; nvec = nover = 0

    row = {
        "method": args.method, "dataset": cfg["label"],
        "GT cell count": len(g), "Pred cell count": len(pc),
        "Cell count ratio": len(pc) / max(len(g), 1),
        "Matched cell count": nm,
        "Detection precision": prec, "Detection recall": rec, "Detection F1": f1,
        "Mean centroid shift": float(dist.mean()),
        "Median centroid shift": float(np.median(dist)),
        "p95 centroid shift": float(np.percentile(dist, 95)),
        "matched_cell_transcript_count_pearson": cnt["pearson"],
        "matched_cell_transcript_count_spearman": cnt["spearman"],
        "matched_cell_transcript_count_mae": cnt["mae"],
        "matched_cell_transcript_count_rmse": cnt["rmse"],
        "matched_cell_gene_vector_mean_cosine": cos,
        "matched_cell_gene_vector_mean_js_distance": js,
        "matched_cell_gene_vector_mean_pearson": vpe,
        "matched_cell_gene_vector_valid_pairs": nvec,
        "transcript_id_overlap_n": nover,
        "transcript_assignment_accuracy_via_matched_cells": acc,
        "Pred transcript rows": int((lp > 0).sum()),
        "GT transcript rows": len(t),
        "Pred transcript assignment rate": rate_p,
        "Overall correct fraction": rate_p * acc if acc == acc else np.nan,
        "median_area_um2": np.nan,
        "median_diameter_um": np.nan,
        "coverage_fraction": np.nan,
        "uses_platform_prior": bool(cfg["prior"]),
        "gt_counts_source": "cells.parquet within the transcripts this run saw; "
                            "per-RNA GT joined back by coordinate",
    }
    banner("RESULT")
    for k, v in row.items():
        log(f"    {k:52s} {v}")
    tag = f"{args.method.lower()}_{name}"
    pd.DataFrame([row]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    log(f"  -> {out}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    log("note: this run used no platform prior, so unlike the raster rows its "
        "detection columns are a real measurement.  Area and coverage are blank "
        "because the output is a transcript table, not a mask.")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=sorted(DATASETS) + sorted(CSV_DATASETS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--fill-um", type=float, default=0.0,
                    help="dilate the raster before measuring area; ComSeg "
                         "labels transcripts, so an unfilled raster is a point "
                         "cloud.  The runner normally writes a filled one.")
    args = ap.parse_args()

    if args.dataset in CSV_DATASETS:
        csv_mode(args.dataset, args)
        return

    cfg = DATASETS[args.dataset]
    root = Path(args.root or cfg["root"])
    run = Path(args.run_dir or cfg["run"])
    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    y0, x0, S = cfg["roi"]
    ps = read_pixel_size(root)

    banner(f"{args.method} x XENIUM {args.dataset} ({cfg['label']})  v2026.08.16")
    log(f"  ROI         y[{y0},{y0+S}) x[{x0},{x0+S}) native px")
    log(f"              {S*ps:.1f} x {S*ps:.1f} um at {ps} um/px")

    banner("PREDICTION")
    import tifffile
    pred_p = find_pred(run, args.pred)
    pred = np.squeeze(tifffile.imread(pred_p))
    scale = S / pred.shape[0]
    pum = scale * ps
    log(f"  raster      {pred_p}")
    log(f"              {pred.shape}  1 px = {scale:.1f} native px = {pum:.4f} um")
    if args.fill_um > 0:
        from scipy import ndimage
        r = max(1, int(round(args.fill_um / pum)))
        before = (pred > 0).mean()
        pred = ndimage.grey_dilation(pred, size=2 * r + 1)
        log(f"  fill        {args.fill_um} um = {r} px: occupancy "
            f"{before*100:.2f}% -> {(pred>0).mean()*100:.2f}%")
    pid, pxy, parea = props(pred)
    log(f"  cells       {len(pid):,}   coverage {(pred>0).mean()*100:.2f}%")
    log(f"  area        median {np.median(parea)*pum*pum:.1f} um^2 -> diameter "
        f"{2*np.sqrt(np.median(parea)*pum*pum/np.pi):.2f} um")

    banner("GROUND TRUTH")
    g = gt_cells(root, cfg["roi"], ps)
    log(f"  cells.parquet: {len(g):,} cells with their centroid inside the ROI")
    if "counts" in g.columns:
        log(f"              platform counts/cell median {g.counts.median():.0f}")

    banner("TRANSCRIPTS")
    tx = read_transcripts(root, cfg["roi"], ps, args.qv_min, run / "cache")
    log(f"  {len(tx):,} transcripts, qv >= {args.qv_min}")
    yy = np.clip((tx.py.to_numpy() / scale).astype(int), 0, pred.shape[0] - 1)
    xx = np.clip((tx.px.to_numpy() / scale).astype(int), 0, pred.shape[1] - 1)
    lp = pred[yy, xx].astype(np.int64)
    rate_p = float((lp > 0).mean())
    log(f"  prediction assigns {rate_p*100:.1f}%")

    if "cell_id" not in tx.columns:
        sys.exit("the transcript table has no cell_id column, so there is no "
                 "per-transcript ground truth for this section")
    order = {c: i + 1 for i, c in enumerate(g.cell_id.to_numpy())}
    glab = tx.cell_id.map(order).fillna(0).astype(np.int64).to_numpy()
    rate_g = float((glab > 0).mean())
    log(f"  the platform cell_id covers {rate_g*100:.1f}%  <- per-RNA GT")

    genes = tx.gene.to_numpy()
    uniq = np.unique(genes)
    gi = pd.Series(genes).map({x: i for i, x in enumerate(uniq)}).to_numpy()
    Cp = cell_by_gene(lp, gi, len(uniq), max(int(pred.max()), 1))
    Cg = cell_by_gene(glab, gi, len(uniq), max(len(g), 1))
    log(f"  {len(uniq)} genes;  counts/cell median pred "
        f"{np.median(np.asarray(Cp.sum(1)).ravel()[pid-1]):.0f}, GT "
        f"{np.median(np.asarray(Cg.sum(1)).ravel()):.0f}")

    banner("MATCHING")
    pxy_um = pxy * pum
    gxy_um = np.stack([g.px.to_numpy(), g.py.to_numpy()], 1) * ps
    R = match_radius(gxy_um)
    log(f"  match radius {R:.2f} um  (GT nearest-neighbour p95 x 1.25)")
    pairs, dist = sparse_hungarian(pxy_um, gxy_um, R)
    nm = len(pairs)
    prec = nm / max(len(pid), 1)
    rec = nm / max(len(g), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    log(f"  matched {nm:,} of {len(pid):,} predicted and {len(g):,} GT")
    log(f"  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    log(f"  centroid shift mean {dist.mean():.2f} median {np.median(dist):.2f} "
        f"p95 {np.percentile(dist,95):.2f} um")

    plab = pid[pairs[:, 0]]
    grow = pairs[:, 1] + 1
    cnt = pair_stats(np.asarray(Cp.sum(1)).ravel()[plab - 1],
                     np.asarray(Cg.sum(1)).ravel()[grow - 1])
    cos, js, vpe, nvec = vector_stats(Cp[plab - 1].toarray(),
                                      Cg[grow - 1].toarray())
    p2g = np.zeros(int(pred.max()) + 1, np.int64)
    p2g[plab] = grow
    both = (lp > 0) & (glab > 0)
    acc = float((p2g[lp[both]] == glab[both]).mean()) if both.any() else np.nan
    log(f"  transcript accuracy {acc:.4f} over {int(both.sum()):,} rows "
        f"-> overall correct {rate_p*acc:.4f}")

    banner("RESULT")
    row = {
        "method": args.method, "dataset": cfg["label"],
        "GT cell count": len(g), "Pred cell count": len(pid),
        "Cell count ratio": len(pid) / max(len(g), 1),
        "Matched cell count": nm,
        "Detection precision": prec, "Detection recall": rec, "Detection F1": f1,
        "Mean centroid shift": float(dist.mean()),
        "Median centroid shift": float(np.median(dist)),
        "p95 centroid shift": float(np.percentile(dist, 95)),
        "matched_cell_transcript_count_pearson": cnt["pearson"],
        "matched_cell_transcript_count_spearman": cnt["spearman"],
        "matched_cell_transcript_count_mae": cnt["mae"],
        "matched_cell_transcript_count_rmse": cnt["rmse"],
        "matched_cell_gene_vector_mean_cosine": cos,
        "matched_cell_gene_vector_mean_js_distance": js,
        "matched_cell_gene_vector_mean_pearson": vpe,
        "matched_cell_gene_vector_valid_pairs": nvec,
        "transcript_id_overlap_n": int(both.sum()),
        "transcript_assignment_accuracy_via_matched_cells": acc,
        "Pred transcript rows": int((lp > 0).sum()),
        "GT transcript rows": len(tx),
        "Pred transcript assignment rate": rate_p,
        "Overall correct fraction": rate_p * acc,
        "median_area_um2": float(np.median(parea) * pum * pum),
        "median_diameter_um": float(2*np.sqrt(np.median(parea)*pum*pum/np.pi)),
        "coverage_fraction": float((pred > 0).mean()),
        "uses_platform_prior": True,
        "gt_counts_source": "cells.parquet + transcripts.parquet cell_id "
                            "(genuine per-RNA GT)",
    }
    for k, v in row.items():
        log(f"    {k:52s} {v}")
    tag = f"{args.method.lower()}_{args.dataset}"
    pd.DataFrame([row]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame({"pred_id": plab, "gt_row": grow, "dist_um": dist}).to_csv(
        out / f"{tag}_matched_pairs.csv", index=False)
    log(f"  -> {out}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    log("note: the prior is the platform nucleus segmentation, so the detection "
        "columns are close to circular; the transcript-level columns carry the "
        "signal for this row")


if __name__ == "__main__":
    main()