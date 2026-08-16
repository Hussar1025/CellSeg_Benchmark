#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
"""eval_xenium_proseg.py   v2026.08.14

Evaluate Proseg on the Xenium ROIs (xenium_4 liver / xenium_5 mouse brain /
xenium_6 colon).

Self-contained, and the metric code is the block that produced the other rows of
the workbook, so the numbers drop straight into the 19-metric sheet.

  ROI          the centred 10000 x 10000 native-pixel window every method shares
                 liver        y0=13704 x0=21974
                 mouse_brain  y0=6956  x0=12077
                 colon        y0=12052 x0=9241
  pixel size   0.2125 um/px
  GT           cells.parquet centroids inside the ROI
  per-RNA GT   transcripts.parquet carries cell_id, so transcript accuracy and
               gene-vector agreement need no borrowed prior column

  python eval_xenium_proseg.py --dataset colon --qv-min 20
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_comseg_xenium_liver.py   v2026.08.13

Evaluate the finished ComSeg run on the Xenium liver ROI (xenium_4).

The segmentation is already on disk:

    /data/qiuyijia/comseg_xenium_liver_roi10000/comseg_segmentation.tif
    25/25 tiles DONE, written 2026-08-11 19:45

so this only scores it.  Self-contained, and the metric code is the same block
that produced the other rows of the workbook, so the numbers drop straight into
the 19-metric sheet.

Geometry, which is where these evaluations usually go wrong
----------------------------------------------------------
    ROI            10000 x 10000 native px, centred, shared by every method
                     liver  y0=13704  x0=21974
    pixel size     0.2125 um/px  ->  the ROI is 2125 x 2125 um
    ComSeg raster  1000 x 1000, i.e. 10 native px per raster pixel
    GT             cells.parquet, centroid inside the ROI
    per-transcript GT   transcripts.parquet carries cell_id, so transcript
                   accuracy and gene-vector agreement are both computable
                   without borrowing any method's prior column

    python eval_comseg_xenium_liver.py --qv-min 20
"""


import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.13"

# ROI windows shared by every method on each dataset
DATASETS = {
    "liver": dict(data_root="/data/qiuyijia/dataset/xenium_liver",
                  y0=13704, x0=21974, label="xenium_4",
                  tissue="Xenium 肝"),
    "mouse_brain": dict(data_root="/data/qiuyijia/dataset/xenium_mouse_brain",
                        y0=6956, x0=12077, label="xenium_5",
                        tissue="Xenium 小鼠脑"),
    "colon": dict(data_root="/data/qiuyijia/dataset/xenium_colon",
                  y0=12052, x0=9241, label="xenium_6",
                  tissue="Xenium 小鼠结肠"),
}
ROI_SIZE = 10000
PIXEL_SIZE = 0.2125

REQUESTED_METRICS = [
    "method", "Pred cell count", "GT cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall",
    "Detection F1", "Mean centroid shift", "Median centroid shift",
    "p95 centroid shift",
    "matched_pair_n_transcripts_vs_total_counts_pearson",
    "matched_pair_n_transcripts_vs_total_counts_spearman",
    "matched_pair_n_transcripts_vs_total_counts_mae",
    "matched_pair_n_transcripts_vs_total_counts_rmse",
    "Pred transcript rows", "GT transcript rows",
    "Pred transcript assignment rate", "Transcript matched cell pairs",
    "matched_cell_transcript_count_pearson",
    "matched_cell_transcript_count_spearman",
    "matched_cell_transcript_count_mae", "matched_cell_transcript_count_rmse",
    "matched_cell_gene_vector_mean_cosine",
    "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson",
    "matched_cell_gene_vector_valid_pairs", "transcript_id_overlap_n",
    "transcript_assignment_accuracy_via_matched_cells", "dataset",
    "Overall correct fraction",
]

SELECTED = [
    "method", "dataset", "GT cell count", "Pred cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall",
    "Detection F1", "Mean centroid shift", "Median centroid shift",
    "p95 centroid shift", "matched_cell_transcript_count_pearson",
    "matched_cell_transcript_count_spearman", "matched_cell_transcript_count_mae",
    "matched_cell_transcript_count_rmse", "matched_cell_gene_vector_mean_cosine",
    "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson", "transcript_id_overlap_n",
    "transcript_assignment_accuracy_via_matched_cells",
    "Pred transcript rows", "GT transcript rows",
    "Pred transcript assignment rate", "Overall correct fraction",
    "median_area_um2", "median_diameter_um", "coverage_fraction",
    "uses_platform_prior",
]

log = logging.getLogger("eval_comseg")


def setup_log(d: Path, tag: str):
    d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    f = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(f); log.addHandler(sh)
    fh = logging.FileHandler(d / f"{tag}.log", mode="w"); fh.setFormatter(f)
    log.addHandler(fh)


def rule(m=""):
    log.info("=" * 78)
    if m:
        log.info(m)
        log.info("=" * 78)


def find(root: Path, *pats):
    for p in pats:
        h = sorted(root.rglob(p))
        if h:
            return h[0]
    return None


# ---------------------------------------------------------------------------
# matching and metrics, unchanged from the scripts that produced the other rows
# ---------------------------------------------------------------------------


def infer_match_radius(gx, gy, quantile=0.95, factor=1.25, floor=2.0):
    from scipy.spatial import cKDTree
    pts = np.column_stack([gx, gy])
    if len(pts) < 2:
        return floor
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(max(np.quantile(nn, quantile) * factor, floor)) if len(nn) else floor


def match_sparse_hungarian(px, py, gx, gy, radius, greedy_above=400):
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    tp, tg = cKDTree(np.column_stack([px, py])), cKDTree(np.column_stack([gx, gy]))
    pairs = tp.query_ball_tree(tg, r=radius)
    pi, gi = [], []
    for i, js in enumerate(pairs):
        for j in js:
            pi.append(i); gi.append(j)
    pi, gi = np.asarray(pi, np.int64), np.asarray(gi, np.int64)
    if len(pi) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    d = np.hypot(px[pi] - gx[gi], py[pi] - gy[gi])
    n = len(px) + len(gx)
    a = coo_matrix((np.ones(len(pi)), (pi, len(px) + gi)), shape=(n, n))
    a = a + a.T
    _, lab = connected_components(a, directed=False)
    comp = lab[pi]
    o = np.argsort(comp, kind="stable")
    pi, gi, d, comp = pi[o], gi[o], d[o], comp[o]
    st = np.append(np.searchsorted(comp, np.unique(comp)), len(pi))
    ep, eg, ed, n_greedy = [], [], [], 0
    for k in range(len(st) - 1):
        sp, sg, sd = pi[st[k]:st[k+1]], gi[st[k]:st[k+1]], d[st[k]:st[k+1]]
        up, ug = np.unique(sp), np.unique(sg)
        if len(up) * len(ug) > greedy_above * greedy_above:
            n_greedy += 1
            usedp, usedg = set(), set()
            for t in np.argsort(sd):
                if sp[t] in usedp or sg[t] in usedg:
                    continue
                usedp.add(sp[t]); usedg.add(sg[t])
                ep.append(sp[t]); eg.append(sg[t]); ed.append(sd[t])
            continue
        pm = {v: i for i, v in enumerate(up)}
        gm = {v: i for i, v in enumerate(ug)}
        big = sd.max() * 10 + 1
        cost = np.full((len(up), len(ug)), big)
        cost[[pm[v] for v in sp], [gm[v] for v in sg]] = sd
        r, c = linear_sum_assignment(cost)
        for rr, cc in zip(r, c):
            if cost[rr, cc] < big:
                ep.append(up[rr]); eg.append(ug[cc]); ed.append(cost[rr, cc])
    if n_greedy:
        log.info(f"  {n_greedy} very large component(s) matched greedily")
    return (np.asarray(ep, np.int64), np.asarray(eg, np.int64),
            np.asarray(ed, float))


def numeric_pair_metrics(a, b, prefix):
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = {f"{prefix}_{k}": np.nan for k in ("pearson", "spearman", "mae", "rmse")}
    if ok.sum() < 3:
        return out
    a, b = a[ok], b[ok]
    if a.std() > 0 and b.std() > 0:
        out[f"{prefix}_pearson"] = float(pearsonr(a, b)[0])
        out[f"{prefix}_spearman"] = float(spearmanr(a, b)[0])
    out[f"{prefix}_mae"] = float(np.mean(np.abs(a - b)))
    out[f"{prefix}_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
    return out


def gene_vector_metrics(pm, gm):
    from scipy.spatial.distance import jensenshannon
    cos, js, pr = [], [], []
    for i in range(pm.shape[0]):
        a, b = pm[i].astype(float), gm[i].astype(float)
        if a.sum() <= 0 or b.sum() <= 0:
            continue
        cos.append(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
        d = jensenshannon(a / a.sum(), b / b.sum(), base=2)
        if np.isfinite(d):
            js.append(float(d))
        if a.std() > 0 and b.std() > 0:
            pr.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "matched_cell_gene_vector_mean_cosine": float(np.mean(cos)) if cos else np.nan,
        "matched_cell_gene_vector_mean_js_distance": float(np.mean(js)) if js else np.nan,
        "matched_cell_gene_vector_mean_pearson": float(np.mean(pr)) if pr else np.nan,
        "matched_cell_gene_vector_valid_pairs": len(cos),
    }


# ---------------------------------------------------------------------------




def assign_transcripts(lab, ux, uy, x0u, y0u, ps, scale):
    r = np.clip(((uy - y0u) / ps / scale).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip(((ux - x0u) / ps / scale).astype(np.int64), 0, lab.shape[1] - 1)
    v = lab[r, c].astype(np.int64)
    v[v < 0] = 0
    return v


def raster_stats(lab, ps, scale):
    """cells, areas in um^2, centroids in ROI-local raster px, coverage."""
    flat = np.where(lab.reshape(-1) < 0, 0, lab.reshape(-1)).astype(np.int64)
    cnt = np.bincount(flat)
    ids = np.nonzero(cnt)[0]
    ids = ids[ids > 0]
    if len(ids) == 0:
        raise SystemExit("the raster is empty")
    um = scale * ps
    area = cnt[ids].astype(float) * um * um
    rows, cols = np.nonzero(lab)
    lv = lab[rows, cols].astype(np.int64)
    o = np.argsort(lv, kind="stable")
    lv, rows, cols = lv[o], rows[o], cols[o]
    b = np.append(np.searchsorted(lv, ids), len(lv))
    cy = np.array([rows[b[i]:b[i+1]].mean() for i in range(len(ids))])
    cx = np.array([cols[b[i]:b[i+1]].mean() for i in range(len(ids))])
    return ids, area, cx, cy, float((lab > 0).mean())


METHOD = "Proseg"
USES_PRIOR = True

# ==========================================================================
# Proseg
# ==========================================================================

DEFAULT_RUN = {"liver": "/data/qiuyijia/proseg_xenium_liver_roi10000",
               "mouse_brain": "/data/qiuyijia/proseg_xenium_mouse_brain_roi10000",
               "colon": "/data/qiuyijia/proseg_xenium_colon"}


def add_args(ap):
    ap.add_argument("--run-dir", default=None)


def load_prediction(args, cfg, ps, S):
    import tifffile
    d = Path(args.run_dir or DEFAULT_RUN[args.dataset])
    p = Path(args.mask) if args.mask else d / "proseg_segmentation.tif"
    if not p.exists():
        alt = find(d, "proseg_segmentation.tif", "proseg_labels_native.tif",
                   "*segmentation*.tif")
        if alt is None:
            raise SystemExit(f"no Proseg raster under {d}; run --stage post")
        p = alt
    lab = np.squeeze(tifffile.imread(p)).astype(np.int64)
    scale = S / lab.shape[0]
    log.info(f"raster      {p}")
    log.info(f"            {lab.shape}  1 px = {scale:.1f} native px = "
             f"{scale*ps:.4f} um")
    note = ""
    mp = d / "prep_meta.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        note = f"prior={m.get('prior')} prior_cells={m.get('prior_cells')}"
    ids, area, cx, cy, cov = raster_stats(lab, ps, scale)
    pux = (cx + 0.5) * scale * ps + cfg["x0"] * ps
    puy = (cy + 0.5) * scale * ps + cfg["y0"] * ps
    return ids, area, pux, puy, cov, lab, scale, note

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="liver", choices=list(DATASETS))
    ap.add_argument("--mask", default=None)
    ap.add_argument("--out-dir", default="/data/qiuyijia/eval_results_qv20")
    ap.add_argument("--qv-min", type=float, default=20.0,
                    help="0 disables; 20 matches what the pipelines were given")
    ap.add_argument("--pixel-size", type=float, default=PIXEL_SIZE)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--fill-um", type=float, default=0.0,
                    help="grow the raster before measuring area; ComSeg labels "
                         "spots, so 0 measures the scatter, not the footprint")
    ap.add_argument("--data-root", default=None)
    add_args(ap)
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()
    if args.version:
        print(VERSION)
        return

    cfg = DATASETS[args.dataset]
    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    setup_log(out, f"eval_{METHOD.lower()}_{args.dataset}")
    rule(f"{METHOD.upper()} x XENIUM {args.dataset}  "
         f"({cfg['label']} {cfg['tissue']})   v{VERSION}")

    # ---- geometry -------------------------------------------------------
    y0, x0, S = cfg["y0"], cfg["x0"], args.roi_size
    ps = args.pixel_size
    log.info(f"ROI         y[{y0},{y0+S}) x[{x0},{x0+S}) native px")
    log.info(f"            {S*ps:.1f} x {S*ps:.1f} um at {ps} um/px")
    x0u, y0u = x0 * ps, y0 * ps
    x1u, y1u = (x0 + S) * ps, (y0 + S) * ps
    log.info(f"            x {x0u:.1f}..{x1u:.1f} um   y {y0u:.1f}..{y1u:.1f} um")

    # ---- prediction -----------------------------------------------------
    rule("PREDICTION")
    ids, area, pux, puy, cov, lab, scale, note = load_prediction(args, cfg, ps, S)
    um = scale * ps
    med = float(np.median(area))
    log.info(f"cells       {len(ids):,}   coverage {cov*100:.2f}%")
    log.info(f"area        median {med:.1f} um^2 -> diameter "
             f"{2*np.sqrt(med/np.pi):.2f} um")
    if note:
        log.info(f"note        {note}")

    # ---- ground truth ---------------------------------------------------
    root = Path(args.data_root or cfg["data_root"])
    rule("GROUND TRUTH")
    cp = find(root, "cells.parquet", "cells.csv.gz")
    cells = pd.read_parquet(cp) if str(cp).endswith(".parquet") else pd.read_csv(cp)
    xc = next(c for c in cells.columns if c.lower() in ("x_centroid", "center_x"))
    yc = next(c for c in cells.columns if c.lower() in ("y_centroid", "center_y"))
    idc = next(c for c in cells.columns if c.lower() in ("cell_id", "cell"))
    inside = (cells[xc] >= x0u) & (cells[xc] < x1u) & \
             (cells[yc] >= y0u) & (cells[yc] < y1u)
    gt = cells.loc[inside].reset_index(drop=True)
    log.info(f"{cp.name}: {len(gt):,} cells with their centroid inside the ROI")
    gux = gt[xc].to_numpy(float)
    guy = gt[yc].to_numpy(float)
    gid = gt[idc].astype(str).to_numpy()
    tot_col = next((c for c in gt.columns
                    if c.lower() in ("total_counts", "transcript_counts")), None)
    gtot = gt[tot_col].to_numpy(float) if tot_col else np.full(len(gt), np.nan)
    if tot_col:
        log.info(f"            platform counts/cell median {np.nanmedian(gtot):.0f} "
                 f"(column '{tot_col}')")

    # ---- transcripts ----------------------------------------------------
    rule("TRANSCRIPTS")
    tp = find(root, "transcripts.parquet", "transcripts.csv.gz")
    log.info(f"{tp.name}")
    cols_need = None
    tx = pd.read_parquet(tp) if str(tp).endswith(".parquet") else pd.read_csv(tp)
    xt = next(c for c in tx.columns if c.lower() in ("x_location", "global_x"))
    yt = next(c for c in tx.columns if c.lower() in ("y_location", "global_y"))
    gtc = next(c for c in tx.columns if c.lower() in ("feature_name", "gene"))
    cidc = next((c for c in tx.columns if c.lower() in ("cell_id", "cell")), None)
    n_all = len(tx)
    tx = tx[(tx[xt] >= x0u) & (tx[xt] < x1u) & (tx[yt] >= y0u) & (tx[yt] < y1u)]
    log.info(f"  {n_all:,} rows, {len(tx):,} inside the ROI")
    if args.qv_min > 0 and "qv" in tx.columns:
        before = len(tx)
        tx = tx[tx.qv >= args.qv_min]
        log.info(f"  qv >= {args.qv_min}: {len(tx):,} of {before:,} kept "
                 f"({len(tx)/before*100:.1f}%)")
    if "is_gene" in tx.columns:
        tx = tx[tx.is_gene.astype(bool)]
    else:
        keep = ~tx[gtc].astype(str).str.startswith(
            ("NegControl", "BLANK", "Unassigned", "antisense", "DeprecatedCodeword"))
        tx = tx[keep]
    tx = tx.reset_index(drop=True)
    log.info(f"  {len(tx):,} coding transcripts, {tx[gtc].nunique():,} genes")

    genes = sorted(tx[gtc].astype(str).unique())
    gi = {g: i for i, g in enumerate(genes)}
    tgene = np.array([gi[g] for g in tx[gtc].astype(str)], np.int32)

    # predicted assignment through the raster
    tl = assign_transcripts(lab, tx[xt].to_numpy(float), tx[yt].to_numpy(float),
                            x0u, y0u, ps, scale)
    log.info(f"  predicted assignment {int((tl>0).sum()):,} "
             f"({(tl>0).mean()*100:.1f}%)")

    has_gt_tx = cidc is not None
    if has_gt_tx:
        raw = tx[cidc].astype(str).to_numpy()
        gt_lab = np.where(np.isin(raw, ("UNASSIGNED", "-1", "nan", "")), "", raw)
        log.info(f"  platform assignment from '{cidc}': "
                 f"{(gt_lab!='').mean()*100:.1f}%")
    else:
        gt_lab = None
        log.info("  no cell_id column, so transcript accuracy is unavailable")

    pos = {int(v): i for i, v in enumerate(ids)}
    n_tx = np.zeros(len(ids), np.int64)
    pmat = np.zeros((len(ids), len(genes)), np.float32)
    sel = tl > 0
    ri = np.array([pos.get(int(v), -1) for v in tl[sel]])
    ok = ri >= 0
    np.add.at(n_tx, ri[ok], 1)
    np.add.at(pmat, (ri[ok], tgene[sel][ok]), 1.0)
    log.info(f"  counts/cell median {np.median(n_tx):.0f}")

    gpos = {v: i for i, v in enumerate(gid)}
    gmat = np.zeros((len(gid), len(genes)), np.float32)
    if has_gt_tx:
        s2 = gt_lab != ""
        rj = np.array([gpos.get(v, -1) for v in gt_lab[s2]])
        ok2 = rj >= 0
        np.add.at(gmat, (rj[ok2], tgene[s2][ok2]), 1.0)

    # ---- matching -------------------------------------------------------
    rule("MATCHING")
    radius = infer_match_radius(gux, guy)
    log.info(f"match radius {radius:.2f} um  (GT nearest-neighbour p95 x 1.25)")
    mp, mg, md = match_sparse_hungarian(pux, puy, gux, guy, radius)
    n = len(mp)
    prec = n / max(len(ids), 1)
    rec = n / max(len(gux), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    log.info(f"matched {n:,} of {len(ids):,} predicted and {len(gux):,} GT")
    log.info(f"precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    if n:
        log.info(f"centroid shift mean {md.mean():.2f} median {np.median(md):.2f} "
                 f"p95 {np.percentile(md,95):.2f} um")

    row = {
        "method": METHOD, "dataset": cfg["label"],
        "Pred cell count": len(ids), "GT cell count": len(gux),
        "Cell count ratio": len(ids) / max(len(gux), 1),
        "Matched cell count": n,
        "Detection precision": prec, "Detection recall": rec, "Detection F1": f1,
        "Mean centroid shift": float(md.mean()) if n else np.nan,
        "Median centroid shift": float(np.median(md)) if n else np.nan,
        "p95 centroid shift": float(np.percentile(md, 95)) if n else np.nan,
        "Pred transcript rows": int((tl > 0).sum()),
        "GT transcript rows": int(len(tx)),
        "Pred transcript assignment rate": float((tl > 0).mean()),
        "Transcript matched cell pairs": n,
        "median_area_um2": med,
        "median_diameter_um": float(2 * np.sqrt(med / np.pi)),
        "coverage_fraction": cov,
        "uses_platform_prior": USES_PRIOR,
    }
    if n:
        pn = n_tx[mp].astype(float)
        row.update(numeric_pair_metrics(
            pn, gtot[mg], "matched_pair_n_transcripts_vs_total_counts"))
        row.update(numeric_pair_metrics(pn, gtot[mg],
                                        "matched_cell_transcript_count"))
        row.update(gene_vector_metrics(pmat[mp], gmat[mg]))
    else:
        for pfx in ("matched_pair_n_transcripts_vs_total_counts",
                    "matched_cell_transcript_count"):
            for k in ("pearson", "spearman", "mae", "rmse"):
                row[f"{pfx}_{k}"] = np.nan
        row.update({"matched_cell_gene_vector_mean_cosine": np.nan,
                    "matched_cell_gene_vector_mean_js_distance": np.nan,
                    "matched_cell_gene_vector_mean_pearson": np.nan,
                    "matched_cell_gene_vector_valid_pairs": 0})

    if has_gt_tx and n:
        m = {int(ids[a]): gid[b] for a, b in zip(mp, mg)}
        both = (tl > 0) & (gt_lab != "")
        mapped = np.array([m.get(int(v), "") for v in tl[both]])
        ok3 = mapped != ""
        ov = int(ok3.sum())
        acc = float((mapped[ok3] == gt_lab[both][ok3]).mean()) if ov else np.nan
        row["transcript_id_overlap_n"] = ov
        row["transcript_assignment_accuracy_via_matched_cells"] = acc
        row["Overall correct fraction"] = (
            row["Pred transcript assignment rate"] * acc
            if np.isfinite(acc) else np.nan)
        log.info(f"transcript accuracy {acc:.4f} over {ov:,} rows -> overall "
                 f"correct {row['Overall correct fraction']:.4f}")
    else:
        row["transcript_id_overlap_n"] = 0
        row["transcript_assignment_accuracy_via_matched_cells"] = np.nan
        row["Overall correct fraction"] = np.nan

    tag = f"{METHOD.lower()}_{args.dataset}"
    pd.DataFrame([{k: row.get(k, np.nan) for k in REQUESTED_METRICS}]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame([{k: row.get(k, np.nan) for k in SELECTED}]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_selected.csv", index=False)
    if n:
        pd.DataFrame({"pred_id": ids[mp], "gt_id": gid[mg],
                      "centroid_shift_um": md, "pred_n_transcripts": n_tx[mp],
                      "gt_total_counts": gtot[mg],
                      "pred_area_um2": area[mp]}).to_csv(
            out / f"{tag}_matched_pairs.csv", index=False)

    rule("RESULT")
    for k in SELECTED:
        v = row.get(k)
        log.info(f"  {k:52s} {v:.4f}" if isinstance(v, float) else
                 f"  {k:52s} {v}")
    log.info(f"-> {out}/{tag}_sparse_hungarian_qc_metrics_full.csv")
    if USES_PRIOR:
        log.info("note: the prior came from the platform segmentation, so the "
                 "detection metrics are close to circular; the transcript-level "
                 "columns are the informative ones for this row")
    else:
        log.info("note: this method uses no platform prior at inference, so its "
                 "detection metrics are a real measurement, unlike the "
                 "prior-based rows")


if __name__ == "__main__":
    main()