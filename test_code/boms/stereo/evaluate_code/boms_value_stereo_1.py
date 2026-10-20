#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_boms_stereo.py — BOMS × Stereo-seq evaluation (19 metrics × 3 slices)
=========================================================================
GT: GEM CellBin (CellID column); GT centroid = count-weighted center of each
    cell's molecules
pred: npz produced by boms_stereo_1.py (top-500 genes, expanded UMIs)
coords: pred(µm) and GT(µm=bin×0.5) align directly, no conversion needed

★ Key adaptations (vs. the Xenium evaluation):
  - GT comes from the GEM CellBin CellID column (not a separate cells.parquet)
  - count / gene-vector comparison uses only the pred gene set (top-500), for fairness
  - rows with MIDCount>1 are count-weighted, not expanded

conda activate ucs
python eval_boms_stereo.py                         # run all 3 slices
python eval_boms_stereo.py --datasets E16.5_E2S6   # run just 1
"""

import os, sys, warnings, gzip, argparse, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
OUT_BASE = "yourpath/boms_stereo"
EVAL_DIR = "yourpath/eval_results/boms"

BIN_UM       = 0.5     # 1 bin = 0.5 µm
MATCH_RADIUS = 20.0    # µm; consistent with the Xenium evaluation
QV_MIN       = None     # Stereo-seq has no QV field

DATASETS = {
    "E14.5_E1S3": dict(
        npz=f"{OUT_BASE}/E14.5_E1S3/boms_E14.5_E1S3.npz",
        gem=f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=4872),
    "E16.5_E2S6": dict(
        npz=f"{OUT_BASE}/E16.5_E2S6/boms_E16.5_E2S6.npz",
        gem=f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=6229),
    "E16.5_E2S7": dict(
        npz=f"{OUT_BASE}/E16.5_E2S7/boms_E16.5_E2S7.npz",
        gem=f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=7960),
}

os.makedirs(EVAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  helper: read GEM CellBin
# ══════════════════════════════════════════════════════════════════════════
def _read_gem(gem_path):
    """Read GEM CellBin, auto-detect column names, return a unified DataFrame (gene,x,y,count,cell)."""
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)

    # unify column names
    ren = {}
    for c in gem.columns:
        cl = c.lower().strip()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcounts", "midcount", "count", "umicounts"):
            ren[c] = "count"
        elif cl in ("cellid", "cell_id", "cell", "cellbin"):
            ren[c] = "cell"
        elif cl == "label":          # some versions use a label column
            if "cell" not in ren.values():
                ren[c] = "cell"
    gem = gem.rename(columns=ren)

    if "count" not in gem.columns:
        gem["count"] = 1
    if "cell" not in gem.columns:
        raise ValueError(f"cell column not found, actual columns: {list(gem.columns)}")

    # bin → µm
    gem["x_um"] = gem["x"].astype(np.float64) * BIN_UM
    gem["y_um"] = gem["y"].astype(np.float64) * BIN_UM
    return gem


# ══════════════════════════════════════════════════════════════════════════
#  single-slice evaluation
# ══════════════════════════════════════════════════════════════════════════
def evaluate_one(name, info):
    t_start = time.time()
    print(f"\n{SEP}")
    print(f"  BOMS × Stereo-seq {name}  evaluation")
    print(SEP)

    npz_path = info["npz"]
    gem_path = info["gem"]
    if not os.path.exists(npz_path):
        print(f"  ✘ npz not found: {npz_path}"); return None

    # ════════════════════════════════════════════════════════════════════
    # 1. read BOMS npz
    # ════════════════════════════════════════════════════════════════════
    print("\n[1] read BOMS npz ...")
    d = np.load(npz_path, allow_pickle=True)
    cell_loc   = d["cell_loc"].astype(np.float64)      # (n_pred, 2) µm [x, y]
    seg        = d["seg"].astype(np.int64)              # per-mol cell idx (0-based)
    tx_x       = d["x"].astype(np.float64)
    tx_y       = d["y"].astype(np.float64)
    gene_names = list(d["gene_names"])
    count_mat  = d["count_mat"]                         # (n_pred, n_gene)
    n_pred     = len(cell_loc)
    n_gene     = len(gene_names)
    pred_xy    = cell_loc  # (n_pred, 2)

    pred_per_cell = np.asarray(count_mat.sum(axis=1)).ravel()  # transcripts per cell

    print(f"  pred cells: {n_pred:,}  molecules: {len(seg):,}  genes: {n_gene}")
    print(f"  cell_loc x: [{cell_loc[:,0].min():.1f},{cell_loc[:,0].max():.1f}]"
          f"  y: [{cell_loc[:,1].min():.1f},{cell_loc[:,1].max():.1f}] µm")

    # ════════════════════════════════════════════════════════════════════
    # 2. read GT (GEM CellBin)
    # ════════════════════════════════════════════════════════════════════
    print("\n[2] read GT (GEM CellBin) ...")
    gem = _read_gem(gem_path)
    print(f"  GEM total rows: {len(gem):,}  genes: {gem.gene.nunique():,}")

    # filter: keep only the gene set BOMS used (fair comparison)
    pred_gene_set = set(gene_names)
    gene2i = {g: i for i, g in enumerate(gene_names)}
    gem_f = gem[gem.gene.isin(pred_gene_set)].copy()
    print(f"  after filtering to the pred gene set: {len(gem_f):,} rows")

    # GT cells (cell>0 = assigned)
    gt_mol = gem_f[gem_f.cell > 0].copy()
    gt_cell_ids = gt_mol.cell.unique()
    n_gt = len(gt_cell_ids)
    print(f"  GT cells: {n_gt:,}  (reference {info['gt_cells']:,})")

    # GT centroid (count-weighted center)
    gt_mol["wx"] = gt_mol["x_um"] * gt_mol["count"]
    gt_mol["wy"] = gt_mol["y_um"] * gt_mol["count"]
    grp = gt_mol.groupby("cell").agg(
        wx=("wx", "sum"), wy=("wy", "sum"), total=("count", "sum")
    )
    grp["cx"] = grp["wx"] / grp["total"]
    grp["cy"] = grp["wy"] / grp["total"]
    gt_cell_order = grp.index.values   # cell ID order
    gt_xy = grp[["cx", "cy"]].values.astype(np.float64)  # (n_gt, 2)
    gt_counts = grp["total"].values.astype(float)         # counts within the pred gene set
    print(f"  GT centroid x: [{gt_xy[:,0].min():.1f},{gt_xy[:,0].max():.1f}]"
          f"  y: [{gt_xy[:,1].min():.1f},{gt_xy[:,1].max():.1f}] µm")

    # ════════════════════════════════════════════════════════════════════
    # 3. GT gene vectors (n_gt × n_gene, aligned to the pred gene set)
    # ════════════════════════════════════════════════════════════════════
    print("\n[3] build GT gene matrix ...")
    gt_mol["gi"] = gt_mol["gene"].map(gene2i)
    gt_gene_agg = (gt_mol.groupby(["cell", "gi"])["count"]
                   .sum().reset_index())
    # cell → row index in gt_cell_order
    cell2row = {c: i for i, c in enumerate(gt_cell_order)}
    gt_gene_agg["row"] = gt_gene_agg["cell"].map(cell2row)
    gt_gene_mat = np.zeros((n_gt, n_gene), dtype=np.int32)
    gt_gene_mat[gt_gene_agg["row"].values,
                gt_gene_agg["gi"].values] = gt_gene_agg["count"].values
    print(f"  GT gene matrix: {gt_gene_mat.shape}  "
          f"nonzero fraction: {(gt_gene_mat > 0).mean():.2%}")

    # ════════════════════════════════════════════════════════════════════
    # 4. matching
    # ════════════════════════════════════════════════════════════════════
    print(f"\n[4] matching (radius={MATCH_RADIUS}µm)...")
    tree = cKDTree(gt_xy)
    dists, gi = tree.query(pred_xy, k=1, distance_upper_bound=MATCH_RADIUS)

    # each GT keeps only its nearest pred (one-to-one, greedy)
    best = {}
    for pi in range(n_pred):
        if dists[pi] >= MATCH_RADIUS:
            continue
        g = int(gi[pi])
        if g not in best or dists[pi] < best[g][1]:
            best[g] = (pi, dists[pi])

    pairs   = [(pi, g, d_) for g, (pi, d_) in best.items()]
    n_match = len(pairs)
    prec   = n_match / n_pred if n_pred else 0
    recall = n_match / n_gt   if n_gt   else 0
    f1     = 2 * prec * recall / (prec + recall) if prec + recall else 0
    shifts = [p[2] for p in pairs]
    print(f"  pred {n_pred:,}  GT {n_gt:,}  matched {n_match:,}")
    print(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={recall:.4f}")

    pi_arr = [p[0] for p in pairs]
    gi_arr = [p[1] for p in pairs]

    # ════════════════════════════════════════════════════════════════════
    # 5. count metrics (transcripts within the pred gene set only)
    # ════════════════════════════════════════════════════════════════════
    print("\n[5] count metrics ...")
    p_cnt = pred_per_cell[pi_arr].astype(float)
    g_cnt = gt_counts[gi_arr].astype(float)

    if len(p_cnt) >= 3 and p_cnt.std() > 0 and g_cnt.std() > 0:
        count_pearson,  _ = pearsonr(p_cnt, g_cnt)
        count_spearman, _ = spearmanr(p_cnt, g_cnt)
        count_mae  = float(np.mean(np.abs(p_cnt - g_cnt)))
        count_rmse = float(np.sqrt(np.mean((p_cnt - g_cnt) ** 2)))
    else:
        count_pearson = count_spearman = count_mae = count_rmse = np.nan
    print(f"  pearson={count_pearson:.4f}  spearman={count_spearman:.4f}"
          f"  MAE={count_mae:.1f}  RMSE={count_rmse:.1f}")

    # ════════════════════════════════════════════════════════════════════
    # 6. gene vectors (sample 800 pairs)
    # ════════════════════════════════════════════════════════════════════
    print("\n[6] gene vectors (sample 800 pairs)...")
    np.random.seed(42)
    samp = np.random.choice(len(pairs), min(800, len(pairs)), replace=False)

    cosines, jsd, prs = [], [], []
    for idx in samp:
        pi, gi_val, _ = pairs[idx]
        pv = count_mat[pi].astype(float)
        gv = gt_gene_mat[gi_val].astype(float)

        nn = np.linalg.norm(pv) * np.linalg.norm(gv)
        if nn > 0:
            cosines.append(float(np.dot(pv, gv) / nn))
            pn = pv / pv.sum() if pv.sum() > 0 else pv
            gn = gv / gv.sum() if gv.sum() > 0 else gv
            jsd.append(float(jensenshannon(pn, gn)))
            if pv.std() > 0 and gv.std() > 0:
                prs.append(float(pearsonr(pv, gv)[0]))

    vec_cosine  = float(np.mean(cosines)) if cosines else np.nan
    vec_js      = float(np.mean(jsd))     if jsd     else np.nan
    vec_pearson = float(np.mean(prs))     if prs     else np.nan
    print(f"  cosine={vec_cosine:.4f}  JS={vec_js:.4f}  pearson={vec_pearson:.4f}")

    # ════════════════════════════════════════════════════════════════════
    # 7. assignment metrics
    # ════════════════════════════════════════════════════════════════════
    print("\n[7] assignment metrics ...")
    # for each BOMS molecule, find the nearest GT centroid → its "true" GT cell
    tree_gt = cKDTree(gt_xy)
    d_tx, gi_tx = tree_gt.query(
        np.stack([tx_x, tx_y], axis=1), k=1,
        distance_upper_bound=MATCH_RADIUS)
    gt_cell_per_tx = np.where(
        d_tx < MATCH_RADIUS,
        gt_cell_order[gi_tx.clip(0, n_gt - 1)], -1)

    ov_list, ac_list = [], []
    for pi, gi_val, _ in pairs:
        pm = (seg == pi)
        if pm.sum() == 0:
            continue
        gc = gt_cell_order[gi_val]
        gta = gt_cell_per_tx[pm]
        ov_list.append(float((gta == gc).mean()))
        if len(gta) > 0:
            maj = pd.Series(gta).value_counts().index[0]
            ac_list.append(float((gta == maj).mean()))

    assign_overlap  = float(np.mean(ov_list)) if ov_list else np.nan
    assign_accuracy = float(np.mean(ac_list)) if ac_list else np.nan
    print(f"  overlap={assign_overlap:.4f}  accuracy={assign_accuracy:.4f}")

    # ════════════════════════════════════════════════════════════════════
    # summary
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    metrics = dict(
        dataset         = name,
        gt_cells        = n_gt,
        pred_cells      = n_pred,
        cell_ratio      = round(n_pred / n_gt, 4) if n_gt else np.nan,
        matched         = n_match,
        precision       = round(prec, 4),
        recall          = round(recall, 4),
        f1              = round(f1, 4),
        loc_mean_um     = round(float(np.mean(shifts)),           4) if shifts else np.nan,
        loc_median_um   = round(float(np.median(shifts)),         4) if shifts else np.nan,
        loc_p95_um      = round(float(np.percentile(shifts, 95)), 4) if shifts else np.nan,
        count_pearson   = round(count_pearson,  4),
        count_spearman  = round(count_spearman, 4),
        count_mae       = round(count_mae,  2),
        count_rmse      = round(count_rmse, 2),
        vec_cosine      = round(vec_cosine,  4),
        vec_js_dist     = round(vec_js,      4),
        vec_pearson     = round(vec_pearson, 4),
        assign_overlap  = round(assign_overlap,  4),
        assign_accuracy = round(assign_accuracy, 4),
    )

    print(f"\n{SEP}\nresults (BOMS × Stereo-seq {name})\n{SEP}")
    for k, v in metrics.items():
        print(f"  {k:22} {v}")

    out_csv = os.path.join(EVAL_DIR, f"stereo_{name}.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(f"\n  written → {out_csv}  ({elapsed:.0f}s)")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
#  main entry
# ══════════════════════════════════════════════════════════════════════════
def main():
    global MATCH_RADIUS
    ap = argparse.ArgumentParser(
        description="BOMS × Stereo-seq evaluation (19 metrics × 3 slices)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()),
                    help="slices to evaluate; default all 3")
    ap.add_argument("--match-radius", type=float, default=MATCH_RADIUS,
                    help="match radius in µm, default 20")
    args = ap.parse_args()

    MATCH_RADIUS = args.match_radius

    all_metrics = []
    for name in args.datasets:
        try:
            m = evaluate_one(name, DATASETS[name])
            if m:
                all_metrics.append(m)
        except Exception as e:
            print(f"\n  ✘ {name} evaluation failed: {e}")
            import traceback; traceback.print_exc()

    if not all_metrics:
        print("\n✘ no slice evaluated successfully")
        sys.exit(1)

    # ── combined summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics)
    combined = os.path.join(EVAL_DIR, "stereo_all.csv")
    df.to_csv(combined, index=False)

    print(f"\n\n{'='*78}")
    print(f"  three-slice summary")
    print(f"{'='*78}")

    show_cols = ["dataset", "gt_cells", "pred_cells", "f1",
                 "loc_median_um", "count_pearson",
                 "vec_cosine", "vec_js_dist",
                 "assign_overlap", "assign_accuracy"]
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 120):
        print(df[[c for c in show_cols if c in df.columns]].to_string(index=False))

    # ── mean row ──
    num_cols = df.select_dtypes(include=[np.number]).columns
    means = df[num_cols].mean()
    print(f"\n  ── mean ──")
    for c in ["f1", "loc_median_um", "count_pearson",
              "vec_cosine", "vec_js_dist", "assign_overlap", "assign_accuracy"]:
        if c in means:
            print(f"  {c:22} {means[c]:.4f}")

    print(f"\n  written → {combined}")
    print()


if __name__ == "__main__":
    main()
