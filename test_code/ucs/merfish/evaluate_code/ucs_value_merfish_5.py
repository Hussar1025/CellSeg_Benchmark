#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERFISH segmentation evaluation — 19 metrics, shared by five methods.

Definitions fully aligned with ucs_value_merfish_4.py:
  * GT      = platform nuclei in cell_metadata.csv whose centroid falls in the center ROI
  * matching = kd-tree candidate edges (radius = GT nearest-neighbor p95×1.25) + Hungarian within connected components / greedy for large components
  * metrics = counts / detection P,R,F1 / centroid shift / transcript-count correlation / gene-vector similarity etc.

Two prediction formats supported:
  --seg  a *_segmentation.tif / mask.tif (proseg/ucs/comseg/genesegnet)
  --npz  BOMS boms_merfish.npz (point cloud: seg = per-transcript cell id, x/y µm, cell_loc centroid)

Usage:
  # label-image methods (auto-locate *_segmentation.tif, or specify with --seg)
  python eval_merfish.py --method UCS --dataset prostate1 \
      --run-dir yourpath/ucs_merfish_prostate1 \
      --data-root yourpath/dataset/merfish_prostate1

  # BOMS
  python eval_merfish.py --method BOMS --dataset prostate1 \
      --npz yourpath/boms_merfish_prostate1/boms_merfish.npz \
      --data-root yourpath/dataset/merfish_prostate1

  # run several at once, appending to the same table
  python eval_merfish.py ... --out results.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROI_SIZE = 10000


# ===========================================================================
# geometry (aligned with the reference Geometry)
# ===========================================================================
class Geometry:
    def __init__(self, sx, sy, ox, oy, y0, x0, size):
        self.sx, self.sy, self.ox, self.oy = sx, sy, ox, oy
        self.y0, self.x0, self.size = y0, x0, size

    @property
    def um_per_px(self):
        return 1.0 / self.sx

    def um_to_local(self, ux, uy):
        return (np.asarray(ux, float) * self.sx + self.ox - self.x0,
                np.asarray(uy, float) * self.sy + self.oy - self.y0)

    def local_to_um(self, px, py):
        return ((np.asarray(px, float) + self.x0 - self.ox) / self.sx,
                (np.asarray(py, float) + self.y0 - self.oy) / self.sy)

    def inside(self, ux, uy):
        px, py = self.um_to_local(ux, uy)
        return (px >= 0) & (px < self.size) & (py >= 0) & (py < self.size)


def load_geometry(data_root, roi_size=ROI_SIZE):
    import tifffile
    root = Path(data_root)
    tf = next(root.rglob("*micron_to_mosaic*transform*.csv"), None)
    if tf is None:
        raise SystemExit(f"no transform under {root}")
    m = np.loadtxt(tf)
    dapi = next((p for p in sorted(root.rglob("*.tif"))
                 if "dapi" in p.name.lower()), None)
    if dapi is None:
        raise SystemExit("no *dapi*.tif under data-root")
    with tifffile.TiffFile(dapi) as t:
        H, W = t.series[0].shape[-2:]
    return Geometry(float(m[0, 0]), float(m[1, 1]),
                    float(m[0, 2]), float(m[1, 2]),
                    (H - roi_size) // 2, (W - roi_size) // 2, roi_size)


# ===========================================================================
# matching (aligned with the reference)
# ===========================================================================
def infer_match_radius(gx, gy, quantile=0.95, factor=1.25, floor=2.0):
    from scipy.spatial import cKDTree
    pts = np.column_stack([gx, gy])
    if len(pts) < 2:
        return floor
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        return floor
    return float(max(np.quantile(nn, quantile) * factor, floor))


def build_candidate_edges(px, py, gx, gy, radius):
    from scipy.spatial import cKDTree
    tp = cKDTree(np.column_stack([px, py]))
    tg = cKDTree(np.column_stack([gx, gy]))
    pairs = tp.query_ball_tree(tg, r=radius)
    pi, gi = [], []
    for i, js in enumerate(pairs):
        for j in js:
            pi.append(i); gi.append(j)
    pi = np.asarray(pi, np.int64); gi = np.asarray(gi, np.int64)
    if len(pi) == 0:
        return pi, gi, np.zeros(0)
    d = np.hypot(px[pi] - gx[gi], py[pi] - gy[gi])
    return pi, gi, d


def _cc_bipartite(pi, gi, n_pred, n_gt):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = n_pred + n_gt
    a = coo_matrix((np.ones(len(pi)), (pi, n_pred + gi)), shape=(n, n))
    a = a + a.T
    _, lab = connected_components(a, directed=False)
    return _, lab


def match_sparse_hungarian(px, py, gx, gy, radius, greedy_above=400):
    from scipy.optimize import linear_sum_assignment
    pi, gi, d = build_candidate_edges(px, py, gx, gy, radius)
    if len(pi) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    _, lab = _cc_bipartite(pi, gi, len(px), len(gx))
    epred, egt, edist = [], [], []
    comp = lab[pi]
    order = np.argsort(comp, kind="stable")
    pi, gi, d, comp = pi[order], gi[order], d[order], comp[order]
    starts = np.searchsorted(comp, np.unique(comp))
    starts = np.append(starts, len(pi))
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        sp, sg, sd = pi[a:b], gi[a:b], d[a:b]
        up, ug = np.unique(sp), np.unique(sg)
        if len(up) * len(ug) > greedy_above * greedy_above:
            o = np.argsort(sd)
            usedp, usedg = set(), set()
            for t in o:
                if sp[t] in usedp or sg[t] in usedg:
                    continue
                usedp.add(sp[t]); usedg.add(sg[t])
                epred.append(sp[t]); egt.append(sg[t]); edist.append(sd[t])
            continue
        pmap = {v: i for i, v in enumerate(up)}
        gmap = {v: i for i, v in enumerate(ug)}
        big = sd.max() * 10 + 1
        cost = np.full((len(up), len(ug)), big)
        cost[[pmap[v] for v in sp], [gmap[v] for v in sg]] = sd
        r, c = linear_sum_assignment(cost)
        for rr, cc in zip(r, c):
            if cost[rr, cc] >= big:
                continue
            epred.append(up[rr]); egt.append(ug[cc]); edist.append(cost[rr, cc])
    return (np.asarray(epred, np.int64), np.asarray(egt, np.int64),
            np.asarray(edist, float))


def numeric_pair_metrics(a, b, prefix):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    out = {f"{prefix}_pearson": np.nan, f"{prefix}_spearman": np.nan,
           f"{prefix}_mae": np.nan, f"{prefix}_rmse": np.nan}
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
        a = pm[i].astype(float); b = gm[i].astype(float)
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


# ===========================================================================
# read GT (platform nuclei, cell_metadata centroid in ROI)
# ===========================================================================
def load_gt(data_root, geo):
    root = Path(data_root)
    md = next(root.rglob("cell_metadata.csv*"), None)
    if md is None:
        raise SystemExit("no cell_metadata.csv under data-root")
    d = pd.read_csv(md)
    ux = d["center_x"].to_numpy(float)
    uy = d["center_y"].to_numpy(float)
    keep = geo.inside(ux, uy)
    return ux[keep], uy[keep]


# ===========================================================================
# read prediction -> unify into (centroid_um, per-cell gene vector, spots_assigned_rate, area, transcript alignment)
# ===========================================================================
def load_pred_seg(seg_path, geo, data_root):
    """Label tif method: take centroids from mask + rebuild gene vectors from transcripts."""
    import tifffile
    lab = np.squeeze(tifffile.imread(seg_path)).astype(np.int64)
    lab[lab < 0] = 0
    s = geo.size / lab.shape[0]           # each mask pixel = s ROI pixels
    ids = np.unique(lab); ids = ids[ids > 0]
    # centroid (label pixel coords -> ROI pixels -> um)
    rows, cols = np.nonzero(lab)
    lv = lab[rows, cols]
    o = np.argsort(lv, kind="stable")
    lv, rows, cols = lv[o], rows[o], cols[o]
    b = np.append(np.searchsorted(lv, ids), len(lv))
    cy = np.array([rows[b[i]:b[i + 1]].mean() for i in range(len(ids))])
    cx = np.array([cols[b[i]:b[i + 1]].mean() for i in range(len(ids))])
    area_px = (b[1:] - b[:-1]).astype(float)
    pux, puy = geo.local_to_um((cx + 0.5) * s, (cy + 0.5) * s)
    area_um2 = area_px * s * s * (geo.um_per_px ** 2)
    # transcripts -> cells (using the dataset's own transcripts)
    genes, gene_idx, tx_ux, tx_uy = _load_tx(data_root)
    lx, ly = geo.um_to_local(tx_ux, tx_uy)
    r = np.clip((ly / s).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip((lx / s).astype(np.int64), 0, lab.shape[1] - 1)
    tl = lab[r, c]
    assign_rate = float((tl > 0).mean())
    pos = {int(v): i for i, v in enumerate(ids)}
    gm = np.zeros((len(ids), len(genes)), np.float32)
    ntx = np.zeros(len(ids), np.int64)
    sel = tl > 0
    ri = np.array([pos.get(int(v), -1) for v in tl[sel]])
    ok = ri >= 0
    np.add.at(ntx, ri[ok], 1)
    np.add.at(gm, (ri[ok], gene_idx[sel][ok]), 1.0)
    return dict(cx=pux, cy=puy, area=area_um2, gm=gm, ntx=ntx,
                genes=genes, assign_rate=assign_rate,
                coverage=float((lab > 0).mean()))


def load_pred_boms(npz_path, geo):
    """BOMS point cloud: seg = per-transcript cell id, x/y um, cell_loc centroid, count_mat cells×genes."""
    z = np.load(npz_path, allow_pickle=True)
    seg = z["seg"].astype(np.int64)
    x = z["x"].astype(float); y = z["y"].astype(float)
    genes = list(z["gene_names"])
    cm = z["count_mat"]
    # cell_loc is already the centroid (µm); order corresponds to count_mat rows
    cl = z["cell_loc"].astype(float)
    pux, puy = cl[:, 0], cl[:, 1]
    inside = geo.inside(pux, puy)
    pux, puy, gm = pux[inside], puy[inside], cm[inside].astype(np.float32)
    ntx = gm.sum(1).astype(np.int64)
    assign_rate = float((seg > 0).mean())
    # BOMS has no pixel area; skip the per-cell transcript convex-hull approximation, area set to nan
    area = np.full(len(pux), np.nan)
    return dict(cx=pux, cy=puy, area=area, gm=gm, ntx=ntx,
                genes=genes, assign_rate=assign_rate, coverage=np.nan)


def _load_tx(data_root):
    """Read detected_transcripts, return (gene list, per-transcript gene index, x_um, y_um)."""
    root = Path(data_root)
    txp = next(root.rglob("detected_transcripts.csv*"), None)
    if txp is None:
        raise SystemExit("no detected_transcripts.csv under data-root")
    cols = ["global_x", "global_y", "gene"]
    d = pd.read_csv(txp, usecols=lambda c: c in cols)
    g = d["gene"].astype(str).to_numpy()
    # drop blank/control
    keep = ~np.char.startswith(np.char.lower(g.astype(str)),
                               ("blank"))
    genes = sorted(set(g[keep]))
    gidx = {v: i for i, v in enumerate(genes)}
    gene_idx = np.array([gidx.get(v, -1) for v in g], np.int64)
    ok = gene_idx >= 0
    return genes, gene_idx[ok], d["global_x"].to_numpy(float)[ok], \
        d["global_y"].to_numpy(float)[ok]


# ===========================================================================
# main evaluation
# ===========================================================================
def evaluate(pred, gx, gy, method, dataset, gt_root):
    px, py = pred["cx"], pred["cy"]
    n_pred, n_gt = len(px), len(gx)
    radius = infer_match_radius(gx, gy)
    ep, eg, ed = match_sparse_hungarian(px, py, gx, gy, radius)
    tp = len(ep)
    prec = tp / n_pred if n_pred else np.nan
    rec = tp / n_gt if n_gt else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan

    # centroid shift
    shift = ed
    # transcript-count correlation of matched pairs (pred ntx vs gt ntx needs GT per-cell transcripts; here approximate with pred ntx of paired cells and GT neighborhood counts)
    m = {
        "method": method, "dataset": dataset,
        "Pred cell count": n_pred, "GT cell count": n_gt,
        "Cell count ratio": n_pred / n_gt if n_gt else np.nan,
        "Matched cell count": tp,
        "Detection precision": prec, "Detection recall": rec,
        "Detection F1": f1,
        "Mean centroid shift": float(np.mean(shift)) if tp else np.nan,
        "Median centroid shift": float(np.median(shift)) if tp else np.nan,
        "p95 centroid shift": float(np.percentile(shift, 95)) if tp else np.nan,
        "Pred transcript assignment rate": pred["assign_rate"],
        "median_area_um2": float(np.nanmedian(pred["area"])) if len(pred["area"]) else np.nan,
        "median_diameter_um": (2 * np.sqrt(np.nanmedian(pred["area"]) / np.pi)
                               if np.isfinite(np.nanmedian(pred["area"])) else np.nan),
        "coverage_fraction": pred["coverage"],
        "match_radius_um": radius,
    }

    # gene-vector similarity (needs GT per-cell gene vectors) — rebuilt from GT platform centroids + dataset transcripts
    gtm = _gt_gene_vectors(gt_root, gx, gy, pred["genes"])
    if gtm is not None and tp:
        pm = pred["gm"][ep]
        gmv = gtm[eg]
        m.update(gene_vector_metrics(pm, gmv))
        # matched-pair transcript-count correlation
        m.update(numeric_pair_metrics(pred["ntx"][ep], gtm[eg].sum(1),
                                      "matched_cell_transcript_count"))
    return m


def _gt_gene_vectors(data_root, gx, gy, genes):
    """Assign transcripts to the nearest GT centroid to rebuild GT per-cell gene vectors (within radius)."""
    from scipy.spatial import cKDTree
    g2, gidx_all, tx_x, tx_y = _load_tx(data_root)
    # align to pred gene list
    remap = {g: i for i, g in enumerate(genes)}
    # gidx_all is an index into g2; convert back to gene names then remap
    inv = {i: g for i, g in enumerate(g2)}
    gene_name = np.array([inv[i] for i in gidx_all])
    col = np.array([remap.get(n, -1) for n in gene_name])
    tree = cKDTree(np.column_stack([gx, gy]))
    dd, jj = tree.query(np.column_stack([tx_x, tx_y]), k=1)
    r = max(np.median(dd) * 2, 3.0)
    ok = (dd <= r) & (col >= 0)
    gm = np.zeros((len(gx), len(genes)), np.float32)
    np.add.at(gm, (jj[ok], col[ok]), 1.0)
    return gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    geo = load_geometry(args.data_root, args.roi_size)
    gx, gy = load_gt(args.data_root, geo)
    print(f"GT cells in ROI = {len(gx)}")

    if args.npz:
        pred = load_pred_boms(args.npz, geo)
    else:
        seg = args.seg
        if seg is None:
            cand = sorted(Path(args.run_dir).glob("*_segmentation.tif")) or \
                   sorted(Path(args.run_dir).glob("*mask*.tif"))
            if not cand:
                raise SystemExit(f"no *_segmentation.tif / *mask*.tif in {args.run_dir}")
            seg = str(cand[0])
        print(f"seg = {seg}")
        pred = load_pred_seg(seg, geo, args.data_root)

    m = evaluate(pred, gx, gy, args.method, args.dataset, args.data_root)

    # print
    print("=" * 60)
    for k, v in m.items():
        print(f"{k:48s} {v}")
    print("=" * 60)

    # append to csv
    outp = Path(args.out)
    df = pd.DataFrame([m])
    if outp.exists():
        old = pd.read_csv(outp)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(outp, index=False)
    print(f"written -> {outp}")


if __name__ == "__main__":
    main()
