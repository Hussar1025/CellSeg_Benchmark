#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_proseg_cosmx.py   v2026.08.16

Proseg on the CosMx lymph node FOVs (cosmx_2), scored with the same 19 metrics
as every other row in the benchmark workbook.

Everything stays in FOV-local pixels.  The whole-slide napari image and the
flat-file tables never agreed on a frame -- offset tables, cell-id matching,
transcript-density correlation and whole-slide cross-correlation were all tried
and all failed -- so nothing here touches that image.  The polygons in
S0-polygons.csv.gz carry x_local_px / y_local_px, the metadata carries
CenterX_local_px, and the transcript table carries x_local_px plus a cell_ID.
All three are already in the same FOV frame.

That cell_ID column is a genuine per-transcript ground truth, which is better
than the MERFISH liver rows had.

    python eval_proseg_cosmx.py --fovs 48 261
    python eval_proseg_cosmx.py --fovs 48 --method UCS \
        --run-root /data/qiuyijia/ucs_cosmx_lymph_node

Per-FOV rows are written, plus a pooled cosmx_2 row that sums the counts and
weights the rate-like columns by cell count.
"""

from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

DATA = "/data/qiuyijia/dataset/cosmx_lymph_node"
RUN = "/data/qiuyijia/proseg_cosmx_lymph_node"
OUT = "/data/qiuyijia/eval_cosmx_lymph"
FOV_PX = 4256
UM_PER_PX = 0.12028      # from the FOV position table, cross-checked against
                         # metadata Area.um2 / Area; the two agree to 0.00%
METHOD, DATASET = "Proseg", "cosmx_2"
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
# Why there is no polygon raster in here
#
# S0-polygons.csv.gz was the obvious reference and it does not work.  For FOV 48
# the 4,896 rings have the right individual areas -- rasterised 3,435 px against
# the metadata's 3,347 px, so 101% -- yet together they cover only 26% of the
# field instead of the 93% those areas imply, and only 1,401 of them survive
# being painted over one another:
#
#     4,896 cells x 3,435 px / 4256^2  = 0.93   expected coverage
#     observed                          = 0.26          -> 3.6x overlap
#     density inside the covered part   = 71,900 cells/mm^2
#     density over the whole FOV        = 18,700 cells/mm^2   <- the real one
#
# Sizes are right, positions are packed about 2x too tightly per axis.  Until
# that frame is understood those rings cannot serve as ground truth.
#
# They are not needed.  The metadata carries CenterX_local_px / CenterY_local_px
# and Area for every cell, and the transcript table carries cell_ID per
# molecule, which is a genuine per-RNA ground truth.  Centroids, areas, counts,
# gene vectors and per-transcript labels all come from those two tables, in
# FOV-local pixels, with no rasterisation anywhere on the reference side.
# --------------------------------------------------------------------------- #

def flat(data: Path, kind: str) -> Path:
    pats = {"tx": "*tx_file*", "meta": "*metadata_file*",
            "poly": "*polygons*", "fov": "*fov_positions*"}
    for d in (data / "flat_files", data):
        c = sorted(d.glob(pats[kind])) if d.is_dir() else []
        if c:
            return c[0]
    sys.exit(f"no {kind} file under {data}")


def derive_um_per_px(data: Path) -> float:
    """Area.um2 / Area gives um^2 per px^2, independent of the position table."""
    try:
        m = pd.read_csv(flat(data, "meta"), usecols=["Area", "Area.um2"],
                        nrows=200000)
        r = float(np.sqrt(np.median(m["Area.um2"] / m["Area"])))
        if 0.05 < r < 0.5:
            return r
    except Exception:
        pass
    return UM_PER_PX


def load_meta(data: Path, fov: int) -> pd.DataFrame:
    m = pd.read_csv(flat(data, "meta"))
    m = m[m.fov == fov]
    idc = "cell_ID" if "cell_ID" in m.columns else "cell_id"
    cx, cy = "CenterX_local_px", "CenterY_local_px"
    for c in (idc, cx, cy):
        if c not in m.columns:
            sys.exit(f"metadata has no column {c}; found {list(m.columns)[:20]}")
    out = pd.DataFrame({"cid": pd.to_numeric(m[idc], errors="coerce"),
                        "x": m[cx].to_numpy(float),
                        "y": m[cy].to_numpy(float),
                        "area_px": (m["Area"].to_numpy(float)
                                    if "Area" in m.columns
                                    else np.full(len(m), np.nan))})
    return out.dropna(subset=["cid"]).astype({"cid": np.int64})


def load_tx(data: Path, fovs) -> pd.DataFrame:
    cache = data / "_inspect_cache"
    tag = "_".join(str(f) for f in sorted(fovs))
    cands = [cache / f"tx_fov_{tag}.parquet"] + sorted(cache.glob("tx_fov_*.parquet"))
    for p in cands:
        if p.exists():
            d = pd.read_parquet(p)
            if set(fovs).issubset(set(d.fov.unique())):
                log(f"  cache       {p.name}")
                return d[d.fov.isin(fovs)]
    log("  no usable cache; scanning the transcript table (the slow part)")
    keep = ["fov", "cell_ID", "x_local_px", "y_local_px", "target", "CellComp"]
    parts, tot = [], 0
    for d in pd.read_csv(flat(data, "tx"), chunksize=5_000_000):
        tot += len(d)
        d = d[d.fov.isin(fovs)]
        if len(d):
            parts.append(d[[c for c in keep if c in d.columns]])
    log(f"  scanned {tot:,} rows")
    t = pd.concat(parts, ignore_index=True)
    cache.mkdir(parents=True, exist_ok=True)
    t.to_parquet(cache / f"tx_fov_{tag}.parquet")
    return t


def find_pred(run_root: Path, fov: int, explicit) -> Path:
    if explicit:
        return Path(explicit)
    d = run_root / f"fov{fov}"
    for n in ("proseg_labels_native.tif", "proseg_segmentation.tif",
              "ucs_labels_native.tif", "ucs_segmentation.tif",
              "comseg_labels_native.tif", "comseg_segmentation.tif"):
        if (d / n).exists():
            return d / n
    c = sorted(d.glob("*labels*.tif")) + sorted(d.glob("*segmentation*.tif"))
    if c:
        return c[0]
    sys.exit(f"no raster under {d}")


def one_fov(fov: int, args, um_px: float) -> dict:
    import tifffile
    banner(f"FOV {fov}")

    pred_p = find_pred(Path(args.run_root), fov, args.pred)
    pred = np.squeeze(tifffile.imread(pred_p))
    pn = pred.shape[0]
    pum = FOV_PX / pn * um_px
    pid, pxy, parea = props(pred)
    log(f"  prediction  {pred_p.name}  {pred.shape}  1 px = {pum:.4f} um")
    log(f"              {len(pid):,} cells  coverage {(pred>0).mean()*100:.2f}%"
        f"  median diameter "
        f"{2*np.sqrt(np.median(parea)*pum*pum/np.pi):.2f} um")

    meta = load_meta(Path(args.data), fov)
    gxy = np.stack([meta.x.to_numpy(), meta.y.to_numpy()], 1)
    gcid = meta.cid.to_numpy()
    gdiam = 2 * np.sqrt(np.nanmedian(meta.area_px) * um_px * um_px / np.pi)
    log(f"  platform    {len(gcid):,} cells from the metadata, median diameter "
        f"{gdiam:.2f} um  <- centroids and areas, nothing rasterised")

    tx = load_tx(Path(args.data), [fov])
    tx = tx[(tx.x_local_px >= 0) & (tx.x_local_px < FOV_PX) &
            (tx.y_local_px >= 0) & (tx.y_local_px < FOV_PX)]
    if args.drop_controls and "target" in tx.columns:
        t = tx.target.astype(str)
        k = ~t.str.startswith(("NegPrb", "Negative", "SystemControl",
                               "Custom", "FalseCode"))
        log(f"  dropping {int((~k).sum()):,} control probes "
            f"({(~k).mean()*100:.2f}%)")
        tx = tx[k]

    s = pn / FOV_PX
    yy = np.clip((tx.y_local_px.to_numpy() * s).astype(int), 0, pn - 1)
    xx = np.clip((tx.x_local_px.to_numpy() * s).astype(int), 0, pn - 1)
    lp = pred[yy, xx].astype(np.int64)

    cid = pd.to_numeric(tx.cell_ID, errors="coerce").fillna(0).astype(np.int64).to_numpy()
    rate_p = float((lp > 0).mean())
    rate_g = float((cid > 0).mean())
    log(f"  transcripts {len(tx):,};  prediction assigns {rate_p*100:.1f}%,  "
        f"the platform cell_ID covers {rate_g*100:.1f}%  <- genuine per-RNA GT")

    order = {c: i + 1 for i, c in enumerate(gcid)}
    glab = pd.Series(cid).map(order).fillna(0).astype(np.int64).to_numpy()
    miss = int(((cid > 0) & (glab == 0)).sum())
    if miss:
        log(f"  {miss:,} transcripts carry a cell_ID that is absent from the "
            f"metadata; treated as unassigned")

    genes = tx.target.astype(str).to_numpy()
    uniq = np.unique(genes)
    gi = pd.Series(genes).map({g: i for i, g in enumerate(uniq)}).to_numpy()
    Cp = cell_by_gene(lp, gi, len(uniq), max(int(pred.max()), 1))
    Cg = cell_by_gene(glab, gi, len(uniq), max(len(gcid), 1))
    log(f"  {len(uniq)} genes;  counts/cell median pred "
        f"{np.median(np.asarray(Cp.sum(1)).ravel()[pid-1]):.0f}, GT "
        f"{np.median(np.asarray(Cg.sum(1)).ravel()):.0f}")

    banner(f"MATCHING  FOV {fov}")
    pxy_um = pxy * pum
    gxy_um = gxy * um_px
    R = match_radius(gxy_um)
    log(f"  match radius {R:.2f} um  (GT nearest-neighbour p95 x 1.25)")
    pairs, dist = sparse_hungarian(pxy_um, gxy_um, R)
    nm = len(pairs)
    prec = nm / max(len(pid), 1)
    rec = nm / max(len(gcid), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    log(f"  matched {nm:,} of {len(pid):,} predicted and {len(gcid):,} GT")
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

    return {
        "method": args.method, "dataset": DATASET, "fov": fov,
        "GT cell count": len(gcid), "Pred cell count": len(pid),
        "Cell count ratio": len(pid) / max(len(gcid), 1),
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
        "gt_counts_source": "metadata centroids + the transcript table's own "
                            "cell_ID (genuine per-RNA GT)",
    }


def pool(rows, method: str) -> dict:
    d = pd.DataFrame(rows)
    add = ["GT cell count", "Pred cell count", "Matched cell count",
           "transcript_id_overlap_n", "Pred transcript rows",
           "GT transcript rows", "matched_cell_gene_vector_valid_pairs"]
    w = d["Pred cell count"].to_numpy(float)
    out = {"method": method, "dataset": DATASET, "fov": "pooled"}
    for c in add:
        out[c] = int(d[c].sum())
    out["Cell count ratio"] = out["Pred cell count"] / max(out["GT cell count"], 1)
    out["Detection precision"] = out["Matched cell count"] / max(out["Pred cell count"], 1)
    out["Detection recall"] = out["Matched cell count"] / max(out["GT cell count"], 1)
    p, r = out["Detection precision"], out["Detection recall"]
    out["Detection F1"] = 2 * p * r / max(p + r, 1e-9)
    for c in d.columns:
        if c in out or c in ("method", "dataset", "fov", "uses_platform_prior",
                             "gt_counts_source"):
            continue
        v = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        out[c] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    out["uses_platform_prior"] = True
    out["gt_counts_source"] = rows[0]["gt_counts_source"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fovs", nargs="+", type=int, default=[48, 261])
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--run-root", default=RUN)
    ap.add_argument("--pred", default=None,
                    help="one explicit raster; only sensible with a single --fovs")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--drop-controls", action="store_true", default=True)
    ap.add_argument("--keep-controls", dest="drop_controls", action="store_false")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    banner(f"{args.method} x CosMx lymph node ({DATASET})   v2026.08.16b")
    um_px = derive_um_per_px(Path(args.data))
    log(f"  scale       {um_px:.5f} um/px   FOV {FOV_PX} px = {FOV_PX*um_px:.0f} um")
    log(f"  reference   metadata centroids + per-transcript cell_ID")
    log(f"              S0-polygons is deliberately not used: at these "
        f"coordinates its rings overlap 3.6x and only a third survive "
        f"rasterisation")

    rows = [one_fov(f, args, um_px) for f in args.fovs]
    if len(rows) > 1:
        rows.append(pool(rows, args.method))

    banner("RESULT")
    d = pd.DataFrame(rows)
    show = ["fov", "GT cell count", "Pred cell count", "Cell count ratio",
            "Detection F1", "Median centroid shift",
            "Pred transcript assignment rate",
            "transcript_assignment_accuracy_via_matched_cells",
            "Overall correct fraction", "median_diameter_um",
            "coverage_fraction"]
    log("\n" + d[show].to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))
    tag = f"{args.method.lower()}_{DATASET}"
    d.to_csv(out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    log(f"  -> {out}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    log("note: the prior is the platform's own segmentation, so the detection "
        "columns are close to circular; the transcript-level columns carry the "
        "signal for this row")


if __name__ == "__main__":
    main()