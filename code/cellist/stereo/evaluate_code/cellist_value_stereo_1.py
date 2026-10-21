#!/usr/bin/env python
"""
Cellist x Stereo-seq evaluation

Circularity note:
- GT = GEM's cell column (CellBin official segmentation)
- Cellist nucleus prior = Cellpose run independently on ssDNA image,
  fully independent of CellBin -> this evaluation is non-circular.
- By contrast UCS (nuclei_mask = rasterized CellBin) and ProSeg (prior =
  CellBin) benefit from prior leakage; Cellist's score is genuinely comparable.

Coordinate system: GEM x/y are in bin units, 1 bin = 0.5 um (Stereo-seq DNB pitch).

Two known pitfalls fixed here:
- cell_count.h5 barcodes are unordered cell ID strings; a {cell_id: column}
  mapping is required, not positional indexing.
- 10X h5 uses CSC layout: matrix is genes x cells, so a cell vector is a column.

Usage:
  python cellist_stereo_1.py --dataset E16.5_E2S6
  python cellist_stereo_1.py --dataset E16.5_E2S6 --resolution 0.5
"""
import os
import glob
import argparse
import warnings

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

DATA_DIR = "yourpath"
CELLIST_BASE = "yourpath"
OUT_CSV = "yourpath"

DATASETS = {
    "E16.5_E2S6": dict(
        gem="yourpath",
        gt_cells=6229),
    "E14.5_E1S3": dict(
        gem="yourpath",
        gt_cells=4872),
    "E16.5_E2S7": dict(
        gem="yourpath",
        gt_cells=7960),
}


# GEM (with official cell column = per-transcript GT)
def load_gem(dataset):
    gem_path = DATASETS[dataset]["gem"]
    P(f"  Reading GEM {os.path.basename(gem_path)} ...")
    gem = pd.read_csv(gem_path, sep="\t", compression="gzip")
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "geneID"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcounts", "midcount", "count"):
            ren[c] = "MIDCounts"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    gem["geneID"] = gem["geneID"].astype("category")
    P(f"    {len(gem):,} rows  {gem.geneID.nunique():,} genes  "
      f"x[{gem.x.min()},{gem.x.max()}] y[{gem.y.min()},{gem.y.max()}]")
    inc = gem.cell > 0
    P(f"    rows with a cell label {int(inc.sum()):,} ({inc.mean()*100:.1f}%)  "
      f"CellBin cells {gem.loc[inc,'cell'].nunique():,}")
    return gem


def gt_centroids(gem, resolution):
    g = gem[gem.cell > 0]
    cen = g.groupby("cell", observed=True).agg(
        cx=("x", "mean"), cy=("y", "mean"),
        n_row=("x", "size"), n_umi=("MIDCounts", "sum")).reset_index()
    med_row = float(cen.n_row.median())
    d_bin = 2 * np.sqrt(med_row / np.pi)
    P(f"\n  GT {len(cen):,} cells")
    P(f"    median spots/cell {med_row:.0f}  median UMI {cen.n_umi.median():.0f}")
    P(f"    empirical diameter {d_bin:.1f} bin = {d_bin*resolution:.1f} um "
      f"@ {resolution} um/bin")
    return cen


# Cellist predictions
def find_cellist_output(dataset, explicit_dir=None):
    """Locate cell_coord.txt / cell_count.h5; explicit dir first, else search
    under CELLIST_BASE by dataset name or the default stereo directory."""
    search_roots = []
    if explicit_dir:
        search_roots.append(explicit_dir)
    search_roots.append(f"{CELLIST_BASE}/{dataset}")
    search_roots.append(CELLIST_BASE)

    for root in search_roots:
        coord = glob.glob(f"{root}/**/*Cellist_segmentation_cell_coord.txt",
                          recursive=True)
        if coord:
            h5 = glob.glob(f"{root}/**/*Cellist_segmentation_cell_count.h5",
                          recursive=True)
            return coord[0], (h5[0] if h5 else None)
    return None, None


def load_pred(coord_file):
    df = pd.read_csv(coord_file, sep="\t")
    if not {"x", "y"} <= set(df.columns):
        return None
    return pd.DataFrame({
        "cell": df["Cellist"].values.astype(float),
        "x": df["x"].values.astype(float),      # barcoding mode = bin units
        "y": df["y"].values.astype(float),
        "nSpot": df["nSpot"].values if "nSpot" in df.columns else np.nan,
    })


def load_expr(h5_file):
    """
    cell_count.h5 -> (CSC genes x cells, gene names, {cell_id: column})

    Barcodes are unordered cell IDs; an ID -> column mapping is required.
    """
    if h5_file is None or not os.path.exists(h5_file):
        return None, None, None
    with tables.open_file(h5_file, "r") as h:
        g = h.root.matrix
        mat = csc_matrix((g.data[:], g.indices[:], g.indptr[:]),
                         shape=tuple(g.shape[:]))
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
def evaluate(dataset, coord_file, h5_file, gem, cen, resolution,
            match_um, vec_sample):
    pred = load_pred(coord_file)
    if pred is None or pred.empty:
        P("  FAIL: prediction is empty"); return None

    R_bin = match_um / resolution
    AR_bin = R_bin * 1.5
    n_pred, n_gt = len(pred), len(cen)

    if (pred.x.max() < cen.cx.min() or pred.x.min() > cen.cx.max() or
            pred.y.max() < cen.cy.min() or pred.y.min() > cen.cy.max()):
        P(f"  FAIL: coordinate ranges do not overlap! pred x[{pred.x.min():.0f},{pred.x.max():.0f}] "
          f"vs GT x[{cen.cx.min():.0f},{cen.cx.max():.0f}]")
        return None

    pxy = pred[["x", "y"]].values
    gxy = cen[["cx", "cy"]].values

    tree = cKDTree(gxy)
    dd, gi = tree.query(pxy, k=1, distance_upper_bound=R_bin)
    best = {}
    for pi in range(n_pred):
        if not np.isfinite(dd[pi]) or dd[pi] >= R_bin:
            continue
        g = int(gi[pi])
        if g not in best or dd[pi] < best[g][1]:
            best[g] = (pi, dd[pi])
    pairs = [(pi, g, x) for g, (pi, x) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred if n_pred else 0.0
    rec = n_match / n_gt if n_gt else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    sh_bin = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    sh_um = sh_bin * resolution
    P(f"\n  matched {n_match:,} / GT {n_gt:,} / pred {n_pred:,}")
    P(f"  P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    txy = gem[["x", "y"]].values.astype(float)
    td, tpi = cKDTree(pxy).query(txy, k=1, distance_upper_bound=AR_bin)
    ok = np.isfinite(td) & (td < AR_bin)
    pred_row = np.where(ok, tpi, -1)

    umi = gem.MIDCounts.values
    dfp = pd.DataFrame({"r": pred_row, "u": umi})
    pcnt = dfp[dfp.r >= 0].groupby("r")["u"].sum()
    gcnt = gem[gem.cell > 0].groupby("cell", observed=True)["MIDCounts"].sum()
    pc = np.array([pcnt.get(p[0], 0) for p in pairs], float)
    gc = np.array([gcnt.get(cen.cell.iloc[p[1]], 0) for p in pairs], float)
    m = (pc > 0) & (gc > 0)
    if m.sum() > 10:
        cp = float(pearsonr(pc[m], gc[m])[0])
        cs = float(spearmanr(pc[m], gc[m])[0])
        cm = float(np.mean(np.abs(pc[m]-gc[m])))
        cr = float(np.sqrt(np.mean((pc[m]-gc[m])**2)))
    else:
        cp = cs = cm = cr = np.nan
    P(f"  count correlation Pearson={cp}")

    vc = vj = vp = np.nan
    n_valid = 0
    mat, genes, id2col = load_expr(h5_file)
    if mat is not None and pairs:
        rng = np.random.default_rng(0)
        samp = pairs if len(pairs) <= vec_sample else \
            [pairs[i] for i in rng.choice(len(pairs), vec_sample, replace=False)]
        sg = {cen.cell.iloc[p[1]] for p in samp}
        sub = gem[gem.cell.isin(sg)]
        gt_grp = sub.groupby(["cell", "geneID"], observed=True)["MIDCounts"].sum()
        gt_genes = set(gt_grp.index.get_level_values(1).unique())
        common = [g for g in genes if g in gt_genes]
        if len(common) >= 50:
            gcol = np.array([genes.index(g) for g in common])
            gidx = pd.Index(common)
            cos_l, js_l, pr_l = [], [], []
            for pi, gj, _ in samp:
                col = id2col.get(float(pred.cell.iloc[pi]))
                if col is None:
                    continue
                cid = cen.cell.iloc[gj]
                try:
                    s = gt_grp.loc[cid]
                except KeyError:
                    continue
                b = s.reindex(gidx).fillna(0).values.astype(float)
                a = np.asarray(mat[:, col].todense()).ravel()[gcol].astype(float)
                if a.sum() < 5 or b.sum() < 5:
                    continue
                cos_l.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))))
                js_l.append(float(jensenshannon(a/a.sum(), b/b.sum())))
                if a.std() > 0 and b.std() > 0:
                    pr_l.append(float(pearsonr(a, b)[0]))
            n_valid = len(cos_l)
            if n_valid > 10:
                vc = float(np.mean(cos_l))
                vj = float(np.nanmean(js_l))
                vp = float(np.mean(pr_l)) if pr_l else np.nan
    P(f"  gene vector cosine={vc}  valid pairs {n_valid:,}"
      f"{'  (no h5 file, metrics set to NaN)' if mat is None else ''}")

    row2gt = {p[0]: cen.cell.iloc[p[1]] for p in pairs}
    has_gt = gem.cell.values > 0
    exp = pd.Series(pred_row).map(row2gt)
    both_m = has_gt & exp.notna().values & (pred_row >= 0)
    w_both = float(umi[both_m].sum())
    if w_both > 0:
        agree = exp[both_m].values == gem.cell.values[both_m]
        overlap = w_both / float(umi[pred_row >= 0].sum())
        acc = float(umi[both_m][agree].sum() / w_both)
    else:
        overlap = acc = np.nan
    P(f"  assignment consistency accuracy={acc}")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    return dict(
        method="Cellist", dataset=dataset,
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred/n_gt), matched=n_match,
        precision=rnd(prec), recall=rnd(rec), f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(sh_um)),
        loc_median_um=rnd(np.nanmedian(sh_um)),
        loc_p95_um=rnd(np.nanpercentile(sh_um, 95)),
        count_pearson=rnd(cp), count_spearman=rnd(cs),
        count_mae=rnd(cm, 2), count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc), vec_js_dist=rnd(vj), vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap), assign_accuracy=rnd(acc),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E16.5_E2S6", choices=list(DATASETS))
    ap.add_argument("--resolution", type=float, default=0.5,
                    help="um per bin (Stereo-seq DNB pitch 500nm)")
    ap.add_argument("--match-radius-um", type=float, default=13.0,
                    help="match radius in um (GT empirical diameter ~13um)")
    ap.add_argument("--vec-sample", type=int, default=2000)
    ap.add_argument("--coord-dir", default=None,
                    help="explicit Cellist output directory; auto-searched if omitted")
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()

    P(SEP); P(f"Cellist x Stereo-seq evaluation -- {a.dataset}"); P(SEP)
    P(f"  resolution {a.resolution} um/bin   "
      f"match radius {a.match_radius_um} um = {a.match_radius_um/a.resolution:.0f} bin")
    P(f"  non-circular: Cellist nucleus prior = Cellpose run independently on ssDNA, "
      f"independent of CellBin")

    coord_file, h5_file = find_cellist_output(a.dataset, a.coord_dir)
    if coord_file is None:
        P(f"\n  FAIL: Cellist output for {a.dataset} not found; "
          f"check {CELLIST_BASE} or pass --coord-dir explicitly")
        return
    P(f"\n  coord: {coord_file}")
    P(f"  h5:    {h5_file if h5_file else '(not found, vec metrics will be NaN)'}")

    gem = load_gem(a.dataset)
    cen = gt_centroids(gem, a.resolution)

    res = evaluate(a.dataset, coord_file, h5_file, gem, cen,
                   a.resolution, a.match_radius_um, a.vec_sample)
    if res is None:
        return

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out_ds = a.out.replace(".csv", f"_{a.dataset}.csv") \
        if a.dataset != "E16.5_E2S6" else a.out
    pd.DataFrame([res]).to_csv(out_ds, index=False)

    P(f"\n{SEP}\nFinal result (Cellist x {a.dataset})\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            P(f"  {k:22} {v}")
    P(f"\n  written -> {out_ds}")
    P(f"  note: per-transcript GT comes from GEM's cell column (CellBin official); "
      f"assign metrics are the most reliable")


if __name__ == "__main__":
    main()