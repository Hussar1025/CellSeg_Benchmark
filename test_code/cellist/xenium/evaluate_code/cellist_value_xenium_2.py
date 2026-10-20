#!/usr/bin/env python
"""
Cellist x Xenium breast evaluation

ROI identical to ProSeg / BOMS / UCS (otherwise not comparable):
  CX=53994/2*0.2125  CY=27420/2*0.2125  HALF=5000*0.2125/2
  -> x[5205.6,6268.1]  y[2382.1,3444.6] um    GT 8,809
  match radius 12.0 um (breast cells ~11-13um, same as ProSeg evaluation)

Two critical issues already fixed:
(1) cell_count.h5 barcodes are unordered cell ID strings ('1034.0') while
    cell_coord.txt's Cellist column is ordered [1,2,3,...]
    -> a {cell_id: column} mapping is required; positional indexing randomly
    pairs gene vectors (on lymph node this bug dropped vec_cosine from 0.777 to 0.199)
(2) 10X h5 uses CSC layout: matrix is genes x cells, indptr length = n_cells+1;
    a cell vector is a column

Usage:
  python eval_cellist_breast.py                 # evaluate all seg_*/alpha* dirs
  python eval_cellist_breast.py --dir <dir>     # evaluate only one
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
P = lambda *a: print(*a, flush=True)

CELLIST_BASE = "yourpath"
DATA_DIR     = "yourpath"
OUT_CSV      = "yourpath"

PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE
CY_UM = 27420 / 2 * PIXEL_SIZE
HALF  = 5000 * PIXEL_SIZE / 2
ROI_X0, ROI_X1 = CX_UM - HALF, CX_UM + HALF
ROI_Y0, ROI_Y1 = CY_UM - HALF, CY_UM + HALF
MATCH_RADIUS = 12.0


# Loading
def load_gt():
    gt = pd.read_parquet("yourpath",
                         columns=["cell_id", "x_centroid", "y_centroid"])
    gt = gt[(gt.x_centroid >= ROI_X0) & (gt.x_centroid < ROI_X1) &
            (gt.y_centroid >= ROI_Y0) & (gt.y_centroid < ROI_Y1)
            ].reset_index(drop=True)
    tx = pd.read_parquet("yourpath",
                         columns=["cell_id", "feature_name",
                                  "x_location", "y_location", "qv"])
    tx = tx[(tx.qv >= 20) &
            (tx.x_location >= ROI_X0) & (tx.x_location < ROI_X1) &
            (tx.y_location >= ROI_Y0) & (tx.y_location < ROI_Y1)]
    tx = tx[~tx.feature_name.str.startswith(
        ("NegControl", "Unassigned", "BLANK", "antisense"), na=False)]
    return gt, tx.reset_index(drop=True)


def load_pred(d):
    f = glob.glob(f"{d}/**/*Cellist_segmentation_cell_coord.txt",
                  recursive=True)
    if not f:
        return None, None
    df = pd.read_csv(f[0], sep="\t")
    pred = pd.DataFrame({
        "cell":  df["Cellist"].values.astype(float),
        "x_um":  df["x"].values * PIXEL_SIZE + ROI_X0,
        "y_um":  df["y"].values * PIXEL_SIZE + ROI_Y0,
        "nSpot": df["nSpot"].values if "nSpot" in df.columns else np.nan,
    })
    return pred, f[0]


def load_expr(d):
    """
    Read cell_count.h5 -> (CSC genes x cells, gene names, {cell_id: column}).

    Barcodes are unordered ('1034.0','870.0',...); an ID -> column mapping is required.
    """
    f = glob.glob(f"{d}/**/*Cellist_segmentation_cell_count.h5", recursive=True)
    if not f:
        return None, None, None
    with tables.open_file(f[0], "r") as h:
        g = h.root.matrix
        mat = csc_matrix((g.data[:], g.indices[:], g.indptr[:]),
                         shape=tuple(g.shape[:]))          # genes x cells
        bcs = [b.decode() if isinstance(b, bytes) else str(b)
               for b in g.barcodes[:]]
        genes = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in g.features.name[:]]
    id2col = {}
    for j, b in enumerate(bcs):
        try:
            id2col[float(b)] = j
        except ValueError:
            pass
    return mat, genes, id2col


# Evaluation
def evaluate(d, gt, tx, vec_sample=3000):
    pred, coord_file = load_pred(d)
    if pred is None or pred.empty:
        return None

    n_pred, n_gt = len(pred), len(gt)
    R, AR = MATCH_RADIUS, MATCH_RADIUS * 1.5
    pxy = pred[["x_um", "y_um"]].values
    gxy = gt[["x_centroid", "y_centroid"]].values

    # One-to-one greedy matching
    tree = cKDTree(gxy)
    dd, gi = tree.query(pxy, k=1, distance_upper_bound=R)
    best = {}
    for pi in range(n_pred):
        if not np.isfinite(dd[pi]) or dd[pi] >= R:
            continue
        g = int(gi[pi])
        if g not in best or dd[pi] < best[g][1]:
            best[g] = (pi, dd[pi])
    pairs = [(pi, g, x) for g, (pi, x) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred if n_pred else 0.0
    rec  = n_match / n_gt   if n_gt   else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])

    # Transcript nearest-neighbour assignment (pred side)
    txy = tx[["x_location", "y_location"]].values
    td, tpi = cKDTree(pxy).query(txy, k=1, distance_upper_bound=AR)
    ok = np.isfinite(td) & (td < AR)
    tx = tx.assign(pred_row=np.where(ok, tpi, -1))

    # Count agreement
    pred_cnt = tx[tx.pred_row >= 0].pred_row.value_counts()
    gt_cnt = tx[tx.cell_id.notna() &
                (tx.cell_id != "UNASSIGNED")].cell_id.value_counts()
    pc = np.array([pred_cnt.get(p[0], 0) for p in pairs], float)
    gc = np.array([gt_cnt.get(gt.cell_id.iloc[p[1]], 0) for p in pairs], float)
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
    mat, genes, id2col = load_expr(d)
    if mat is not None:
        sub = tx[tx.cell_id.notna() & (tx.cell_id != "UNASSIGNED")]
        gtp = sub.groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0)
        common = [g for g in genes if g in gtp.columns]
        if len(common) >= 50:
            gcol = [genes.index(g) for g in common]
            gtp = gtp[common]
            rng = np.random.default_rng(0)
            samp = pairs if len(pairs) <= vec_sample else \
                [pairs[i] for i in rng.choice(len(pairs), vec_sample,
                                              replace=False)]
            cos_l, js_l, pr_l = [], [], []
            for pi, gj, _ in samp:
                col = id2col.get(float(pred.cell.iloc[pi]))   # ID -> column
                if col is None:
                    continue
                gid = gt.cell_id.iloc[gj]
                if gid not in gtp.index:
                    continue
                a = np.asarray(mat[:, col].todense()).ravel()[gcol].astype(float)
                b = gtp.loc[gid].values.astype(float)
                if a.sum() < 5 or b.sum() < 5:
                    continue
                cos_l.append(float(a @ b /
                                   (np.linalg.norm(a)*np.linalg.norm(b))))
                js_l.append(float(jensenshannon(a/a.sum(), b/b.sum())))
                if a.std() > 0 and b.std() > 0:
                    pr_l.append(float(pearsonr(a, b)[0]))
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
        exp = sub.pred_row.map(row2gt)
        inp = exp.notna()
        overlap = float(inp.mean())
        acc = float((exp[inp] == sub.cell_id[inp]).mean()) if inp.any() else np.nan
    else:
        overlap = acc = np.nan

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="Cellist", dataset="xenium_breast",
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred/n_gt), matched=n_match,
        precision=rnd(prec), recall=rnd(rec), f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(shifts)),
        loc_median_um=rnd(np.nanmedian(shifts)),
        loc_p95_um=rnd(np.nanpercentile(shifts, 95)),
        count_pearson=rnd(cp), count_spearman=rnd(cs),
        count_mae=rnd(cm, 2), count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc), vec_js_dist=rnd(vj), vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap), assign_accuracy=rnd(acc),
        result_dir=os.path.relpath(os.path.dirname(coord_file), CELLIST_BASE),
    )
    P(f"\n  {res['result_dir'][:56]}")
    P(f"    pred {n_pred:,}/GT {n_gt:,}  matched {n_match:,}  "
      f"F1={f1:.4f}  loc_med={res['loc_median_um']}um")
    P(f"    cnt_r={res['count_pearson']}  vec_cos={res['vec_cosine']} "
      f"(valid {n_valid:,}/{n_match:,})  assign_acc={res['assign_accuracy']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="evaluate only this directory")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--vec-sample", type=int, default=3000)
    a = ap.parse_args()

    P(SEP); P("Cellist x Xenium breast evaluation"); P(SEP)
    P(f"  ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] um")
    P(f"  match radius {MATCH_RADIUS} um (same as ProSeg/BOMS evaluation)")
    P(f"\n  Loading GT and transcripts (1-2 min)...")
    gt, tx = load_gt()
    P(f"  GT {len(gt):,} cells   transcripts in ROI {len(tx):,}")

    if a.dir:
        dirs = [a.dir]
    else:
        dirs = sorted({os.path.dirname(os.path.dirname(p)) if
                       os.path.basename(os.path.dirname(p)).startswith("alpha")
                       else os.path.dirname(p)
                       for p in glob.glob(
                           f"{CELLIST_BASE}/**/"
                           f"*Cellist_segmentation_cell_coord.txt",
                           recursive=True)})
    dirs = [d for d in dirs if glob.glob(
        f"{d}/**/*Cellist_segmentation_cell_coord.txt", recursive=True)]
    if not dirs:
        P("  FAIL: no Cellist results found"); return
    P(f"  {len(dirs)} directories to evaluate")

    rows = [r for r in (evaluate(d, gt, tx, a.vec_sample) for d in dirs) if r]
    if not rows:
        P("  FAIL: no valid results"); return
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)

    if len(df) > 1:
        P(f"\n{SEP}\nSummary (F1 descending)\n{SEP}")
        P(f"  {'F1':>7} {'pred':>8} {'ratio':>7} {'vec_cos':>8}  dir")
        P("  " + "-" * 62)
        for _, r in df.iterrows():
            P(f"  {r.f1:>7.4f} {int(r.pred_cells):>8,} {r.cell_ratio:>7.3f} "
              f"{r.vec_cosine:>8}  {str(r.result_dir)[:30]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best = df.iloc[[0]].copy()
    best.drop(columns=["result_dir"]).to_csv(a.out, index=False)
    if len(df) > 1:
        df.to_csv(a.out.replace(".csv", "_all_params.csv"), index=False)

    P(f"\n{SEP}\nBest result (Cellist x Xenium breast)\n{SEP}")
    for k, v in best.iloc[0].items():
        if k not in ("method", "dataset", "result_dir"):
            P(f"  {k:22} {v}")
    P(f"\n  written -> {a.out}")
    P(f"  note: nucleus prior uses 10X nucleus_boundaries "
      f"({int(8390):,} nuclei in ROI = 95% of GT), not Cellist's built-in "
      f"Cellpose (which only yields ~3,000 nuclei)")


if __name__ == "__main__":
    main()