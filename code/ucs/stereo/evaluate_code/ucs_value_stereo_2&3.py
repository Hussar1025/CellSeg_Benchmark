#!/usr/bin/env python
"""
UCS x Stereo-seq evaluation

===== Fix in this version: predicted cell count must not use mask.max() =====
The previous version used int(mask.max()) as the predicted cell count, giving 181,844 on E14.5 —
nearly 40x the 4,870 from direct verification via len(np.unique(m[m>0])).
This is caused by sparse, non-contiguous label values (UCS processes per patch and likely assigns
each patch's cells globally unique but non-contiguous raw IDs, without a final renumber to 1..N).
This error already appeared once in this session's GeneSegNet self-check script; this is the same pitfall:
to count objects in a label map, use unique counting or len(regionprops(mask)), never max().

★ Affected: precision/F1/cell_ratio were wrong (denominator inflated 40x, precision pushed near 0).
  recall/count_pearson/vec_cosine/assign_accuracy are unaffected because those use the real
  label values (pred_labels) enumerated by regionprops, not mask.max().

===== Circularity note (critical, must be stated) =====
UCS nuclei_mask is rasterized directly from the GEM cell column — the prior is itself GT.
This evaluation's precision/recall/F1 are therefore inflated and must not be compared on the
same leaderboard as Cellist (whose prior is independent of CellBin). This is the same kind of
issue as UCS results on MERFISH/CosMx.

Usage:
  python eval_ucs_stereo.py --dataset E14.5_E1S3
  python eval_ucs_stereo.py --dataset E14.5_E1S3 --mask-path /path/to/segmentation_mask.tif
"""
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import tifffile
import skimage.measure
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
OUT_CSV = "yourpath/eval_results/ucs/stereo.csv"

DATASETS = {
    "E16.5_E2S6": dict(
        gem=f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        mask="yourpath/ucs_stereoseq/ucs_log/pred/segmentation_mask.tif",
        gt_cells=6229),
    "E14.5_E1S3": dict(
        gem=f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        mask="yourpath/ucs_stereoseq_E14.5/ucs_log/pred/segmentation_mask.tif",
        gt_cells=4872),
    "E16.5_E2S7": dict(
        gem=f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        mask="yourpath/ucs_stereoseq_E2S7/ucs_log/pred/segmentation_mask.tif",
        gt_cells=7960),
}


def load_gem(dataset):
    gem_path = DATASETS[dataset]["gem"]
    P(f"  reading GEM {os.path.basename(gem_path)} ...")
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
    x0, y0 = int(gem.x.min()), int(gem.y.min())
    P(f"    {len(gem):,} rows  {gem.geneID.nunique():,} genes  origin=({x0},{y0})")
    inc = gem.cell > 0
    P(f"    CellBin cells {gem.loc[inc,'cell'].nunique():,}")
    return gem, x0, y0


def gt_centroids(gem, resolution):
    g = gem[gem.cell > 0]
    cen = g.groupby("cell", observed=True).agg(
        cx=("x", "mean"), cy=("y", "mean"),
        n_row=("x", "size"), n_umi=("MIDCounts", "sum")).reset_index()
    P(f"\n  GT {len(cen):,} cells  median spots per cell {cen.n_row.median():.0f}")
    return cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E14.5_E1S3", choices=list(DATASETS))
    ap.add_argument("--mask-path", default=None)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--match-radius-um", type=float, default=13.0)
    ap.add_argument("--vec-sample", type=int, default=2000)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()

    P(SEP); P(f"UCS x Stereo-seq evaluation —— {a.dataset}"); P(SEP)
    P(f"  ★★ circularity warning: nuclei_mask is rasterized directly from the GEM cell column,"
      f"the prior is itself GT. precision/recall/F1 below are inflated and cannot be")
    P(f"     compared on the same leaderboard as Cellist (prior independent of CellBin)")

    mask_path = a.mask_path or DATASETS[a.dataset]["mask"]
    if not os.path.exists(mask_path):
        P(f"\n  ✘ mask not found: {mask_path}")
        return
    mask = tifffile.imread(mask_path)
    P(f"\n  mask: {mask_path}")
    P(f"    shape={mask.shape}")

    # ── ★ fix: count cells via regionprops real labels, not mask.max() ──────
    props = skimage.measure.regionprops(mask)
    n_pred = len(props)
    n_pred_maxval = int(mask.max())
    P(f"    real label count (regionprops) = {n_pred:,}")
    if n_pred_maxval != n_pred:
        P(f"    ⚠ mask.max() = {n_pred_maxval:,} — inconsistent with the real label count,"
          f"confirming labels are sparse (not renumbered 1..N),"
          f"which is the source of the previous script's 181,844 (E14.5) predicted count;"
          f"this version uses the real label count")

    gem, x0, y0 = load_gem(a.dataset)
    cen = gt_centroids(gem, a.resolution)
    n_gt = len(cen)

    pred_bin_xy = np.array([[x0 + p.centroid[1], y0 + p.centroid[0]]
                            for p in props])
    pred_labels = np.array([p.label for p in props])

    R_bin = a.match_radius_um / a.resolution
    gxy = cen[["cx", "cy"]].values
    tree = cKDTree(gxy)
    dd, gi = tree.query(pred_bin_xy, k=1, distance_upper_bound=R_bin)
    best = {}
    for pi in range(len(pred_bin_xy)):
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
    sh_um = (np.array([p[2] for p in pairs]) * a.resolution
            if pairs else np.array([np.nan]))
    P(f"\n  matched {n_match:,} / GT {n_gt:,} / predicted {n_pred:,}")
    P(f"  P={prec:.4f} R={rec:.4f} F1={f1:.4f}  "
      f"(expected high, since prior=GT — the denominator is now correct, so the numbers are meaningful)")

    H, W = mask.shape
    gx = (gem.x.values - x0).astype(np.int64)
    gy = (gem.y.values - y0).astype(np.int64)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    pred_label_per_row = np.zeros(len(gem), np.int64)
    pred_label_per_row[ok] = mask[gy[ok], gx[ok]]

    umi = gem.MIDCounts.values
    dfp = pd.DataFrame({"lab": pred_label_per_row, "u": umi})
    pcnt = dfp[dfp.lab > 0].groupby("lab")["u"].sum()
    gcnt = gem[gem.cell > 0].groupby("cell", observed=True)["MIDCounts"].sum()
    pc = np.array([pcnt.get(pred_labels[p[0]], 0) for p in pairs], float)
    gc = np.array([gcnt.get(cen.cell.iloc[p[1]], 0) for p in pairs], float)
    m = (pc > 0) & (gc > 0)
    cp = float(pearsonr(pc[m], gc[m])[0]) if m.sum() > 10 else np.nan
    cs = float(spearmanr(pc[m], gc[m])[0]) if m.sum() > 10 else np.nan
    cm_ = float(np.mean(np.abs(pc[m]-gc[m]))) if m.sum() > 10 else np.nan
    cr = float(np.sqrt(np.mean((pc[m]-gc[m])**2))) if m.sum() > 10 else np.nan
    P(f"  count correlation Pearson={cp}  (also expected high, same prior leakage, unaffected"
      f"by the bug above because this always uses pred_labels, not mask.max())")

    rng = np.random.default_rng(0)
    samp = pairs if len(pairs) <= a.vec_sample else \
        [pairs[i] for i in rng.choice(len(pairs), a.vec_sample, replace=False)]
    sp_labels = {pred_labels[p[0]] for p in samp}
    sg_ids = {cen.cell.iloc[p[1]] for p in samp}
    pv_tbl = (pd.DataFrame({"lab": pred_label_per_row, "gene": gem.geneID.values,
                           "u": umi})
             [lambda d: d.lab.isin(sp_labels)]
             .groupby(["lab", "gene"], observed=True)["u"].sum().unstack(fill_value=0))
    gv_tbl = (gem[gem.cell.isin(sg_ids)]
             .groupby(["cell", "geneID"], observed=True)["MIDCounts"].sum()
             .unstack(fill_value=0))
    common = sorted(set(pv_tbl.columns) & set(gv_tbl.columns))
    vc = vj = vp = np.nan
    n_valid = 0
    if len(common) >= 20:
        cos_l, js_l, pr_l = [], [], []
        for pi, gj, _ in samp:
            lb, cid = pred_labels[pi], cen.cell.iloc[gj]
            if lb not in pv_tbl.index or cid not in gv_tbl.index:
                continue
            pvv = pv_tbl.loc[lb, common].values.astype(float)
            gvv = gv_tbl.loc[cid, common].values.astype(float)
            if pvv.sum() < 5 or gvv.sum() < 5:
                continue
            cos_l.append(float(pvv @ gvv / (np.linalg.norm(pvv)*np.linalg.norm(gvv))))
            js_l.append(float(jensenshannon(pvv/pvv.sum(), gvv/gvv.sum())))
            if pvv.std() > 0 and gvv.std() > 0:
                pr_l.append(float(pearsonr(pvv, gvv)[0]))
        n_valid = len(cos_l)
        if n_valid > 10:
            vc, vj = float(np.mean(cos_l)), float(np.nanmean(js_l))
            vp = float(np.mean(pr_l)) if pr_l else np.nan
    P(f"  gene vector cosine={vc}  valid pairs {n_valid:,}")

    row2gt = {pred_labels[p[0]]: cen.cell.iloc[p[1]] for p in pairs}
    exp = pd.Series(pred_label_per_row).map(row2gt)
    has_gt = gem.cell.values > 0
    both_m = has_gt & exp.notna().values & (pred_label_per_row > 0)
    w_both = float(umi[both_m].sum())
    if w_both > 0:
        agree = exp[both_m].values == gem.cell.values[both_m]
        overlap = w_both / float(umi[pred_label_per_row > 0].sum())
        acc = float(umi[both_m][agree].sum() / w_both)
    else:
        overlap = acc = np.nan
    P(f"  assignment consistency accuracy={acc}  (expected ~1.0, since the mask is the cell column rasterized)")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="UCS", dataset=a.dataset,
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred/n_gt), matched=n_match,
        precision=rnd(prec), recall=rnd(rec), f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(sh_um)),
        loc_median_um=rnd(np.nanmedian(sh_um)),
        loc_p95_um=rnd(np.nanpercentile(sh_um, 95)),
        count_pearson=rnd(cp), count_spearman=rnd(cs),
        count_mae=rnd(cm_, 2), count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc), vec_js_dist=rnd(vj), vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap), assign_accuracy=rnd(acc),
    )

    P(f"\n{SEP}\nresults (UCS x {a.dataset})  ⚠ circular, prior=GT\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            P(f"  {k:22} {v}")

    out_ds = a.out.replace(".csv", f"_{a.dataset}.csv")
    os.makedirs(os.path.dirname(out_ds), exist_ok=True)
    pd.DataFrame([res]).to_csv(out_ds, index=False)
    P(f"\n  saved -> {out_ds}")


if __name__ == "__main__":
    main()
