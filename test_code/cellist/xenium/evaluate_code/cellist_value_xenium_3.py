#!/usr/bin/env python
"""
Cellist x Xenium evaluation (fixed h5 row mapping)

Key fix:
  Cellist's cell_count.h5 barcodes are unordered cell IDs:
      barcodes = ['1034.0', '870.0', '1646.0', ...]
  while cell_coord.txt's Cellist column is ordered [1.0, 2.0, 3.0, ...]
  -> a {cell_id: h5 column} mapping is required, positional indexing will not work
  -> positional indexing previously produced vec_cosine=0.1993 (equivalent to random pairing)

10X h5 uses CSC layout: matrix is genes x cells, indptr length = n_cells+1,
column j corresponds to barcodes[j].

Usage:
  python eval_cellist_xenium.py --dataset xenium_lymph
  python eval_cellist_xenium.py --dataset xenium_breast
"""
import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import tables
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

# Dataset configuration
PIXEL_SIZE = 0.2125

CFG = {
    "xenium_lymph": dict(
        cellist_base="yourpath",
        data_dir="yourpath",
        # experiment_roi.json: x[3982.5,4317.5] y[2522.5,2857.5] um
        roi_x0=3982.5, roi_y0=2522.5,
        img_w=1576, img_h=1577,
        match_radius=6.9,          # lymphocyte diameter
    ),
    "xenium_breast": dict(
        cellist_base="yourpath",
        data_dir="yourpath",
        roi_x0=53994/2*PIXEL_SIZE - 5000*PIXEL_SIZE/2,
        roi_y0=27420/2*PIXEL_SIZE - 5000*PIXEL_SIZE/2,
        img_w=5000, img_h=5000,
        match_radius=12.0,
    ),
}


# Loading
def load_gt(cfg):
    x0, y0 = cfg["roi_x0"], cfg["roi_y0"]
    x1 = x0 + cfg["img_w"] * PIXEL_SIZE
    y1 = y0 + cfg["img_h"] * PIXEL_SIZE

    gt = pd.read_parquet(f"{cfg['data_dir']}/cells.parquet",
                         columns=["cell_id", "x_centroid", "y_centroid"])
    gt = gt[(gt.x_centroid >= x0) & (gt.x_centroid < x1) &
            (gt.y_centroid >= y0) & (gt.y_centroid < y1)].reset_index(drop=True)

    tx = pd.read_parquet(f"{cfg['data_dir']}/transcripts.parquet",
                         columns=["cell_id", "feature_name",
                                  "x_location", "y_location", "qv"])
    tx = tx[(tx.qv >= 20) &
            (tx.x_location >= x0) & (tx.x_location < x1) &
            (tx.y_location >= y0) & (tx.y_location < y1)]
    tx = tx[~tx.feature_name.str.startswith(
        ("NegControl", "Unassigned", "BLANK", "antisense"), na=False)]
    return gt, tx.reset_index(drop=True), (x0, x1, y0, y1)


def load_pred(result_dir, cfg):
    """Read Cellist cell_coord.txt, convert coordinates to global um"""
    f = glob.glob(f"{result_dir}/**/*Cellist_segmentation_cell_coord.txt",
                  recursive=True)
    if not f:
        return None
    df = pd.read_csv(f[0], sep="\t")
    return pd.DataFrame({
        "cell":  df["Cellist"].values.astype(float),
        "x_um":  df["x"].values * PIXEL_SIZE + cfg["roi_x0"],
        "y_um":  df["y"].values * PIXEL_SIZE + cfg["roi_y0"],
        "nSpot": df["nSpot"].values if "nSpot" in df.columns else np.nan,
    }), f[0]


def load_pred_expr(result_dir):
    """
    Read Cellist cell_count.h5 -> (csc matrix genes x cells, gene names, {cell_id: column})

    Barcodes are unordered cell ID strings ('1034.0'); a mapping is required.
    """
    f = glob.glob(f"{result_dir}/**/*Cellist_segmentation_cell_count.h5",
                  recursive=True)
    if not f:
        return None, None, None
    with tables.open_file(f[0], "r") as h:
        g = h.root.matrix
        data    = g.data[:]
        indices = g.indices[:]
        indptr  = g.indptr[:]
        shape   = tuple(g.shape[:])          # [n_genes, n_cells]
        bcs     = [b.decode() if isinstance(b, bytes) else str(b)
                   for b in g.barcodes[:]]
        genes   = [b.decode() if isinstance(b, bytes) else str(b)
                   for b in g.features.name[:]]

    mat = csc_matrix((data, indices, indptr), shape=shape)   # genes x cells
    # barcode '1034.0' -> cell_id 1034.0 -> column index
    id2col = {}
    for j, b in enumerate(bcs):
        try:
            id2col[float(b)] = j
        except ValueError:
            pass
    return mat, genes, id2col


# Evaluation
def evaluate(result_dir, gt, tx, cfg, verbose=True):
    loaded = load_pred(result_dir, cfg)
    if loaded is None:
        return None
    pred, coord_file = loaded
    if pred.empty:
        return None

    R  = cfg["match_radius"]
    AR = R * 1.5
    n_pred, n_gt = len(pred), len(gt)
    pxy = pred[["x_um", "y_um"]].values
    gxy = gt[["x_centroid", "y_centroid"]].values

    # One-to-one greedy matching
    tree = cKDTree(gxy)
    d, gi = tree.query(pxy, k=1, distance_upper_bound=R)
    best = {}
    for pi in range(n_pred):
        if not np.isfinite(d[pi]) or d[pi] >= R:
            continue
        g = int(gi[pi])
        if g not in best or d[pi] < best[g][1]:
            best[g] = (pi, d[pi])
    pairs = [(pi, g, dd) for g, (pi, dd) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred if n_pred else 0.0
    rec  = n_match / n_gt   if n_gt   else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])

    # Transcript nearest-neighbour assignment (pred side)
    txy = tx[["x_location", "y_location"]].values
    ptree = cKDTree(pxy)
    td, tpi = ptree.query(txy, k=1, distance_upper_bound=AR)
    ok = np.isfinite(td) & (td < AR)
    tx = tx.assign(pred_row=np.where(ok, tpi, -1))

    # Count agreement
    pred_cnt = tx[tx.pred_row >= 0].pred_row.value_counts()
    gt_cnt   = tx[tx.cell_id.notna() &
                  (tx.cell_id != "UNASSIGNED")].cell_id.value_counts()
    p_rows = [p[0] for p in pairs]
    g_rows = [p[1] for p in pairs]
    pc = np.array([pred_cnt.get(i, 0) for i in p_rows], float)
    gc = np.array([gt_cnt.get(gt.cell_id.iloc[g], 0) for g in g_rows], float)
    m = (pc > 0) & (gc > 0)
    if m.sum() > 10:
        cp = float(pearsonr(pc[m], gc[m])[0])
        cs = float(spearmanr(pc[m], gc[m])[0])
        cm = float(np.mean(np.abs(pc[m] - gc[m])))
        cr = float(np.sqrt(np.mean((pc[m] - gc[m]) ** 2)))
    else:
        cp = cs = cm = cr = np.nan

    # Gene vectors (use id2col mapping, not positional indexing)
    vc = vj = vp = np.nan
    n_valid = 0
    mat, genes, id2col = load_pred_expr(result_dir)
    if mat is not None:
        # GT gene vectors: transcripts aggregated by (cell_id, gene)
        sub = tx[tx.cell_id.notna() & (tx.cell_id != "UNASSIGNED")]
        gt_pivot = (sub.groupby(["cell_id", "feature_name"])
                       .size().unstack(fill_value=0))
        common = [g for g in genes if g in gt_pivot.columns]
        if len(common) >= 50:
            gcol = [genes.index(g) for g in common]
            gt_sub = gt_pivot[common]
            cos_l, js_l, pr_l = [], [], []
            for pi, gj, _ in pairs:
                cid = pred.cell.iloc[pi]
                col = id2col.get(float(cid))          # ID -> column
                if col is None:
                    continue
                gt_cid = gt.cell_id.iloc[gj]
                if gt_cid not in gt_sub.index:
                    continue
                pv = np.asarray(mat[:, col].todense()).ravel()[gcol].astype(float)
                gv = gt_sub.loc[gt_cid].values.astype(float)
                if pv.sum() < 5 or gv.sum() < 5:
                    continue
                cos_l.append(float(pv @ gv /
                                   (np.linalg.norm(pv)*np.linalg.norm(gv))))
                js_l.append(float(jensenshannon(pv/pv.sum(), gv/gv.sum())))
                if pv.std() > 0 and gv.std() > 0:
                    pr_l.append(float(pearsonr(pv, gv)[0]))
            n_valid = len(cos_l)
            if n_valid > 10:
                vc = float(np.mean(cos_l))
                vj = float(np.nanmean(js_l))
                vp = float(np.mean(pr_l)) if pr_l else np.nan

    # Assignment consistency
    row2gt = {p[0]: gt.cell_id.iloc[p[1]] for p in pairs}
    sub = tx[(tx.pred_row >= 0) & tx.cell_id.notna() &
             (tx.cell_id != "UNASSIGNED")]
    if len(sub):
        expect  = sub.pred_row.map(row2gt)
        in_pair = expect.notna()
        overlap = float(in_pair.mean())
        acc = (float((expect[in_pair] == sub.cell_id[in_pair]).mean())
               if in_pair.any() else np.nan)
    else:
        overlap = acc = np.nan

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="Cellist",
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred/n_gt),
        matched=n_match,
        precision=rnd(prec), recall=rnd(rec), f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(shifts)),
        loc_median_um=rnd(np.nanmedian(shifts)),
        loc_p95_um=rnd(np.nanpercentile(shifts, 95)),
        count_pearson=rnd(cp), count_spearman=rnd(cs),
        count_mae=rnd(cm, 2), count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc), vec_js_dist=rnd(vj), vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap), assign_accuracy=rnd(acc),
        result_dir=os.path.relpath(os.path.dirname(coord_file),
                                   cfg["cellist_base"]),
    )
    if verbose:
        print(f"\n  {res['result_dir'][:56]}")
        print(f"    pred {n_pred:,}/GT {n_gt:,}  matched {n_match:,}  "
              f"F1={f1:.4f}  loc_med={res['loc_median_um']}um")
        print(f"    cnt_r={res['count_pearson']}  "
              f"vec_cos={res['vec_cosine']} (valid {n_valid:,}/{n_match:,})  "
              f"assign_acc={res['assign_accuracy']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=list(CFG.keys()))
    ap.add_argument("--scan-all", action="store_true",
                    help="evaluate all alpha*/ and seg_*/ subdirectories")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = CFG[a.dataset]
    out = a.out or f"yourpath"

    print(SEP); print(f"Cellist x {a.dataset} evaluation"); print(SEP)
    gt, tx, (x0, x1, y0, y1) = load_gt(cfg)
    print(f"  ROI x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}] um")
    print(f"  GT {len(gt):,} cells   transcripts in ROI {len(tx):,}   "
          f"match radius {cfg['match_radius']}um")

    base = cfg["cellist_base"]
    if a.scan_all:
        dirs = sorted(set(
            glob.glob(f"{base}/alpha*") +
            glob.glob(f"{base}/seg_*") +
            [os.path.dirname(p) for p in
             glob.glob(f"{base}/**/*Cellist_segmentation_cell_coord.txt",
                       recursive=True)]))
    else:
        dirs = sorted(glob.glob(f"{base}/alpha*")) or [base]
    dirs = [d for d in dirs if glob.glob(
        f"{d}/**/*Cellist_segmentation_cell_coord.txt", recursive=True)]
    if not dirs:
        print("  FAIL: no Cellist results found"); return
    print(f"  {len(dirs)} directories to evaluate")

    rows = [r for r in (evaluate(d, gt, tx, cfg) for d in dirs) if r]
    if not rows:
        print("  FAIL: no valid results"); return
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)

    if len(df) > 1:
        print(f"\n{SEP}\nSummary (F1 descending)\n{SEP}")
        print(f"  {'F1':>7} {'pred':>8} {'ratio':>7} {'vec_cos':>8}  dir")
        print("  " + "-" * 62)
        for _, r in df.iterrows():
            print(f"  {r.f1:>7.4f} {int(r.pred_cells):>8,} "
                  f"{r.cell_ratio:>7.3f} {r.vec_cosine:>8}  "
                  f"{str(r.result_dir)[:30]}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = df.iloc[[0]].copy()
    best.insert(1, "dataset", a.dataset)
    best.drop(columns=["result_dir"]).to_csv(out, index=False)
    if len(df) > 1:
        df.to_csv(out.replace(".csv", "_all_params.csv"), index=False)

    print(f"\n{SEP}\nResult (Cellist x {a.dataset})\n{SEP}")
    for k, v in best.iloc[0].items():
        if k not in ("method", "dataset", "result_dir"):
            print(f"  {k:22} {v}")
    print(f"\n  written -> {out}")


if __name__ == "__main__":
    main()