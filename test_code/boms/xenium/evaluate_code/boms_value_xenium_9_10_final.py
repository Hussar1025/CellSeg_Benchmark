#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from skimage.draw import polygon

PIX = 0.2125
ROI = 10000
MATCH_RADIUS = 20.0

CTRL = (
    "NegControl", "BLANK", "Unassigned",
    "antisense", "DeprecatedCodeword", "Intergenic"
)

CFG = {
    "xenium9": dict(
        root=Path("yourpath/dataset/xenium9"),
        npz=Path(
            "yourpath/boms_tuning_x9_hs25/"
            "boms_xenium_9/boms_xenium_9.npz"
        ),
        x0=16374,
        y0=5147,
        expected_gt=17689,
        label="xenium_9",
    ),
    "xenium10": dict(
        root=Path("yourpath/dataset/xenium10"),
        npz=Path(
            "yourpath/boms_xenium_10/"
            "boms_xenium_10.npz"
        ),
        x0=20564,
        y0=13750,
        expected_gt=36604,
        label="xenium_10",
    ),
}


def log(*x):
    print(*x, flush=True)


def polygon_centroid(x, y):
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x, y = x[:-1], y[:-1]

    if len(x) < 3:
        return float(np.mean(x)), float(np.mean(y))

    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)

    cr = x * y2 - x2 * y
    a2 = cr.sum()

    if abs(a2) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))

    return (
        float(((x + x2) * cr).sum() / (3 * a2)),
        float(((y + y2) * cr).sum() / (3 * a2)),
    )


def build_gt_mask(cfg):
    bp = (
        cfg["root"]
        / "benchmark_roi_10000"
        / "cell_boundaries.parquet"
    )

    b = pd.read_parquet(bp)

    idc = next(
        c for c in b.columns
        if c.lower() in ("label_id", "cell_id", "cell", "id")
    )

    xc = next(
        c for c in b.columns
        if c.lower() in ("vertex_x_local_um", "vertex_x")
    )

    yc = next(
        c for c in b.columns
        if c.lower() in ("vertex_y_local_um", "vertex_y")
    )

    groups = list(b.groupby(idc, sort=False))

    if len(groups) != cfg["expected_gt"]:
        raise RuntimeError(
            f"canonical GT mismatch: "
            f"{len(groups)} != {cfg['expected_gt']}"
        )

    gtmask = np.zeros((ROI, ROI), np.int32)

    ids = []
    gx = []
    gy = []

    for row, (cid, g) in enumerate(groups, start=1):
        x = g[xc].to_numpy(float)
        y = g[yc].to_numpy(float)

        cx, cy = polygon_centroid(x, y)

        ids.append(str(cid))
        gx.append(cx)
        gy.append(cy)

        rr, cc = polygon(
            y / PIX,
            x / PIX,
            shape=gtmask.shape
        )

        gtmask[rr, cc] = row

    return (
        gtmask,
        np.asarray(ids),
        np.asarray(gx),
        np.asarray(gy),
    )


def greedy_pairs(pred_xy, gt_xy, radius):
    pt = cKDTree(pred_xy)
    gt = cKDTree(gt_xy)

    cand = []

    vv = pt.query_ball_tree(gt, r=radius)

    for pi, gg in enumerate(vv):
        if not gg:
            continue

        dd = np.sqrt(
            ((gt_xy[gg] - pred_xy[pi]) ** 2).sum(axis=1)
        )

        for gi, d in zip(gg, dd):
            cand.append((float(d), pi, int(gi)))

    cand.sort()

    used_p = set()
    used_g = set()

    pairs = []
    dists = []

    for d, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue

        used_p.add(pi)
        used_g.add(gi)

        pairs.append((pi, gi))
        dists.append(d)

    return (
        np.asarray(pairs, np.int64),
        np.asarray(dists, float),
    )


def numeric_pair(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]

    if len(a) < 2:
        return (np.nan,) * 4

    return (
        float(pearsonr(a, b)[0]),
        float(spearmanr(a, b)[0]),
        float(np.mean(np.abs(a - b))),
        float(np.sqrt(np.mean((a - b) ** 2))),
    )


def vector_metrics(P, G, batch=256):
    cos = []
    js = []
    pr = []

    for a in range(0, P.shape[0], batch):
        b = min(a + batch, P.shape[0])

        A = P[a:b].toarray().astype(np.float64)
        B = G[a:b].toarray().astype(np.float64)

        for x, y in zip(A, B):
            nx = np.linalg.norm(x)
            ny = np.linalg.norm(y)

            if nx <= 0 or ny <= 0:
                continue

            cos.append(
                float(np.dot(x, y) / (nx * ny))
            )

            sx = x.sum()
            sy = y.sum()

            if sx > 0 and sy > 0:
                js.append(
                    float(jensenshannon(x / sx, y / sy))
                )

            if np.std(x) > 0 and np.std(y) > 0:
                pr.append(
                    float(np.corrcoef(x, y)[0, 1])
                )

    return (
        float(np.mean(cos)) if cos else np.nan,
        float(np.mean(js)) if js else np.nan,
        float(np.mean(pr)) if pr else np.nan,
        len(cos),
    )


def evaluate(name, qv, outdir):
    cfg = CFG[name]

    root = cfg["root"]
    npz_path = cfg["npz"]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(npz_path, allow_pickle=True)

    # BOMS cell_loc is in GLOBAL absolute micrometre coordinates.
    # Canonical GT centroids from vertex_*_local_um are ROI-local.
    pred_xy_global = np.asarray(d["cell_loc"], float)
    n_pred = len(pred_xy_global)

    seg = np.asarray(d["seg"], np.int64)
    tx_x = np.asarray(d["x"], float)
    tx_y = np.asarray(d["y"], float)

    boms_gene = np.asarray(d["gene"]).astype(str)
    boms_genes = np.asarray(d["gene_names"]).astype(str)

    count_mat = sparse.csr_matrix(
        np.asarray(d["count_mat"])
    )

    roi = np.asarray(d["roi"], float)
    npz_gt = int(np.asarray(d["gt_cells"]).reshape(-1)[0])

    expected_roi = np.asarray([
        cfg["x0"] * PIX,
        (cfg["x0"] + ROI) * PIX,
        cfg["y0"] * PIX,
        (cfg["y0"] + ROI) * PIX,
    ])

    if not np.allclose(roi, expected_roi, atol=1e-4):
        raise RuntimeError(
            f"BOMS ROI mismatch: {roi} vs {expected_roi}"
        )

    if npz_gt != cfg["expected_gt"]:
        raise RuntimeError(
            f"BOMS gt_cells mismatch: "
            f"{npz_gt} != {cfg['expected_gt']}"
        )

    if count_mat.shape[0] != n_pred:
        raise RuntimeError(
            f"count_mat rows={count_mat.shape[0]} "
            f"pred={n_pred}"
        )

    log("\n" + "=" * 90)
    log("BOMS CANONICAL FINAL EVAL", name)
    log("=" * 90)

    log("NPZ =", npz_path)
    log("ROI =", roi.tolist())
    log("pred =", n_pred)
    log("assigned tx =", len(seg))

    log("building canonical GT polygon raster ...")

    gtmask, gids, gx, gy = build_gt_mask(cfg)

    # GT is ROI-local in um. Convert BOMS absolute cell centroids
    # into the same ROI-local coordinate system.
    pred_xy = pred_xy_global.copy()
    pred_xy[:, 0] -= roi[0]
    pred_xy[:, 1] -= roi[2]

    gt_xy = np.column_stack((gx, gy))

    log(
        "pred local range x=",
        float(pred_xy[:,0].min()),
        float(pred_xy[:,0].max()),
        "y=",
        float(pred_xy[:,1].min()),
        float(pred_xy[:,1].max()),
    )

    log(
        "GT local range x=",
        float(gt_xy[:,0].min()),
        float(gt_xy[:,0].max()),
        "y=",
        float(gt_xy[:,1].min()),
        float(gt_xy[:,1].max()),
    )

    pairs, dists = greedy_pairs(
        pred_xy,
        gt_xy,
        MATCH_RADIUS
    )

    if not len(pairs):
        raise RuntimeError("no matched cells")

    mp = pairs[:, 0]
    mg = pairs[:, 1]

    precision = len(pairs) / max(n_pred, 1)
    recall = len(pairs) / len(gids)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else np.nan
    )

    log(
        f"GT={len(gids):,} "
        f"matched={len(pairs):,} "
        f"P={precision:.4f} "
        f"R={recall:.4f} "
        f"F1={f1:.4f}"
    )

    log(
        f"centroid mean={dists.mean():.3f} "
        f"median={np.median(dists):.3f} "
        f"p95={np.percentile(dists,95):.3f} um"
    )

    # ------------------------------------------------------------
    # Canonical source transcripts: no platform cell_id is used.
    # ------------------------------------------------------------

    tp = root / "outs" / "transcripts.parquet"

    ds = pds.dataset(tp, format="parquet")
    names = ds.schema.names

    xc = "x_location"
    yc = "y_location"
    gc = "feature_name"

    qvc = "qv" if "qv" in names else None

    xlo, xhi, ylo, yhi = roi

    fil = (
        (pds.field(xc) >= xlo)
        & (pds.field(xc) < xhi)
        & (pds.field(yc) >= ylo)
        & (pds.field(yc) < yhi)
    )

    if qvc:
        fil = fil & (pds.field(qvc) >= qv)

    tab = ds.to_table(
        columns=[xc, yc, gc],
        filter=fil
    ).to_pandas()

    if len(tab) and isinstance(tab[gc].iloc[0], bytes):
        tab[gc] = tab[gc].str.decode("utf-8")

    tab[gc] = tab[gc].astype(str)

    keep = ~tab[gc].str.startswith(CTRL)
    tab = tab.loc[keep].reset_index(drop=True)

    source_genes = sorted(tab[gc].unique())
    source_gene_set = set(source_genes)

    log(
        f"canonical coding transcripts="
        f"{len(tab):,} genes={len(source_genes):,}"
    )

    # ------------------------------------------------------------
    # Canonical GT transcript -> polygon assignment.
    # ------------------------------------------------------------

    px = (
        np.floor(tab[xc].to_numpy(float) / PIX)
        .astype(np.int64)
        - cfg["x0"]
    )

    py = (
        np.floor(tab[yc].to_numpy(float) / PIX)
        .astype(np.int64)
        - cfg["y0"]
    )

    ok = (
        (px >= 0) & (px < ROI)
        & (py >= 0) & (py < ROI)
    )

    tab = tab.loc[ok].reset_index(drop=True)
    px = px[ok]
    py = py[ok]

    glab = gtmask[py, px].astype(np.int64)

    # ------------------------------------------------------------
    # Build canonical GT gene matrix on BOMS-common genes.
    # ------------------------------------------------------------

    common = [
        g for g in source_genes
        if g in set(boms_genes)
    ]

    if not common:
        raise RuntimeError(
            "no common genes between BOMS and canonical transcripts"
        )

    common_i = {
        g: i for i, g in enumerate(common)
    }

    tg = tab[gc].map(common_i).to_numpy()

    vg = (
        (glab > 0)
        & pd.notna(tg)
    )

    tg_int = np.zeros(len(tg), np.int64)
    tg_int[vg] = tg[vg].astype(np.int64)

    G = sparse.coo_matrix(
        (
            np.ones(vg.sum(), np.float32),
            (
                glab[vg] - 1,
                tg_int[vg]
            ),
        ),
        shape=(len(gids), len(common)),
    ).tocsr()

    # ------------------------------------------------------------
    # BOMS matrix restricted to same canonical gene panel.
    # ------------------------------------------------------------

    bg = {g: i for i, g in enumerate(boms_genes)}

    bi = np.asarray(
        [bg[g] for g in common],
        dtype=np.int64
    )

    P = count_mat[:, bi].tocsr()

    pcnt = np.asarray(P.sum(axis=1)).ravel()
    gcnt = np.asarray(G.sum(axis=1)).ravel()

    cp, cs, mae, rmse = numeric_pair(
        pcnt[mp],
        gcnt[mg]
    )

    vcos, vjs, vpr, vn = vector_metrics(
        P[mp],
        G[mg]
    )

    # ------------------------------------------------------------
    # BOMS transcript-assignment accuracy.
    #
    # BOMS owns its own transcript -> predicted cell assignment.
    # GT is independently obtained by polygon raster lookup.
    # ------------------------------------------------------------

    bkeep = np.asarray(
        [g in source_gene_set for g in boms_gene],
        dtype=bool
    )

    bx = tx_x[bkeep]
    by = tx_y[bkeep]
    bs = seg[bkeep]

    bpx = (
        np.floor(bx / PIX).astype(np.int64)
        - cfg["x0"]
    )

    bpy = (
        np.floor(by / PIX).astype(np.int64)
        - cfg["y0"]
    )

    bok = (
        (bpx >= 0) & (bpx < ROI)
        & (bpy >= 0) & (bpy < ROI)
        & (bs >= 0) & (bs < n_pred)
    )

    bpx = bpx[bok]
    bpy = bpy[bok]
    bs = bs[bok]

    bgt = gtmask[bpy, bpx].astype(np.int64)

    match_gt = np.zeros(n_pred, np.int32)

    for pi, gi in pairs:
        match_gt[int(pi)] = int(gi) + 1

    mapped_gt = match_gt[bs]

    valid = (
        (bgt > 0)
        & (mapped_gt > 0)
    )

    assign_accuracy = (
        float((mapped_gt[valid] == bgt[valid]).mean())
        if valid.any()
        else np.nan
    )

    # BOMS NPZ contains its own transcript population.
    # Assignment coverage must therefore use that population as denominator;
    # do not divide by the independently loaded canonical transcript table.
    native_valid_assignment = (
        (seg >= 0) & (seg < n_pred)
    )

    assign_rate = (
        float(native_valid_assignment.mean())
        if len(seg)
        else np.nan
    )

    overall = (
        assign_rate * assign_accuracy
        if np.isfinite(assign_accuracy)
        else np.nan
    )

    gt_assignment_rate = float(
        (glab > 0).mean()
    )

    log(
        f"count Pearson={cp:.4f} "
        f"Spearman={cs:.4f}"
    )

    log(
        f"vector cosine={vcos:.4f} "
        f"JS={vjs:.4f} "
        f"Pearson={vpr:.4f}"
    )

    log(
        f"assignment accuracy="
        f"{assign_accuracy:.4f} "
        f"over {valid.sum():,}"
    )

    log(
        f"assignment rate={assign_rate:.4f} "
        f"overall correct={overall:.4f}"
    )

    row = dict(
        method="BOMS",
        dataset=cfg["label"],

        gt_cells=len(gids),
        pred_cells=n_pred,
        cell_ratio=n_pred / len(gids),

        matched=len(pairs),
        precision=precision,
        recall=recall,
        f1=f1,

        loc_mean_um=float(dists.mean()),
        loc_median_um=float(np.median(dists)),
        loc_p95_um=float(np.percentile(dists, 95)),

        count_pearson=cp,
        count_spearman=cs,
        count_mae=mae,
        count_rmse=rmse,

        vec_cosine=vcos,
        vec_js_dist=vjs,
        vec_pearson=vpr,
        vec_valid_pairs=vn,

        assign_accuracy=assign_accuracy,
        assign_rate=assign_rate,
        overall_correct=overall,

        match_radius_um=MATCH_RADIUS,

        uses_platform_prior=False,
        assignment_independent=True,
        vector_independent=True,

        transcript_id_overlap_n=int(valid.sum()),
        gt_assignment_rate=gt_assignment_rate,

        gt_counts_source=(
            "rasterised canonical benchmark cell_boundaries"
        ),

        note=(
            "canonical polygon GT; GT transcript assignment by "
            "spatial polygon raster lookup; no platform cell_id used; "
            "BOMS prediction uses native NPZ cell_loc/seg/count_mat; "
            "fixed historical BOMS match radius 20um; "
            "assign_rate denominator is BOMS native NPZ transcript population"
        ),
    )

    op = (
        outdir
        / f"boms_{cfg['label']}_final_metrics.csv"
    )

    pd.DataFrame([row]).to_csv(
        op,
        index=False
    )

    log("\nRESULT")
    for k, v in row.items():
        log(f"{k:30s} {v}")

    log("->", op)

    return row


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        choices=list(CFG),
        required=True,
    )

    ap.add_argument(
        "--qv-min",
        type=float,
        default=20.0,
    )

    ap.add_argument(
        "--out-dir",
        default="yourpath/boms/eval_final",
    )

    args = ap.parse_args()

    evaluate(
        args.dataset,
        args.qv_min,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
