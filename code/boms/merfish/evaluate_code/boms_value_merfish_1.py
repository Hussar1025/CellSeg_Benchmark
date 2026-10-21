#!/usr/bin/env python
"""
BOMS × MERFISH mouse brain evaluation

ROI matches GeneSegNet / ProSeg / UCS: x[3589,5761] y[2327,4499] µm  GT 8,444

===== fixes in this version =====
The column must not be named "gt" -- DataFrame.gt is pandas' greater-than method,
df.gt returns the [function] not the column -> AttributeError: 'function' object has no attribute 'isin'
-> renamed to gtcell, and always use df["column"] instead of attribute access

===== GT source and circularity =====
- cell centroid GT: center_x/y in cell_metadata_S1R1.csv (Vizgen official)
- per-transcript GT: MERFISH provides no official cell_id!
  -> use the nuclei_mask.tif built by UCS as a proxy
    (that mask is rasterized from the official cell_boundaries HDF5 polygons)
  -> BOMS uses no prior and is independent of this mask -> assign metric [not circular], valid
    (vs ProSeg MERFISH: the prior is the GT NN assignment -> circular, inflates F1=1.0)

nuclei_mask coordinate system (built by UCS): 1 µm/bin, bin origin (3588, 2326)
  transcript(x_um,y_um) → mask[round(y_um)-2326, round(x_um)-3588]
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

NPZ  = "yourpath/boms_merfish/boms_merfish.npz"
META = ("yourpath/dataset/merfish_mouse_brain/"
        "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1"
        "_cell_metadata_S1R1.csv")
NUC_MASK = "yourpath/ucs_merfish_brain/nuclei_mask.tif"
OUT = "yourpath/eval_results/boms/merfish.csv"

ROI_X0, ROI_X1 = 3589.0, 5761.0
ROI_Y0, ROI_Y1 = 2327.0, 4499.0
BIN_OX, BIN_OY = 3588, 2326
MATCH_RADIUS   = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=MATCH_RADIUS)
    ap.add_argument("--no-mask", action="store_true",
                    help="skip nuclei_mask; count/vec/assign set to NaN")
    ap.add_argument("--vec-sample", type=int, default=2000)
    cfg = ap.parse_args()

    print(SEP); print("BOMS × MERFISH mouse brain evaluation"); print(SEP)
    print(f"  ROI x[{ROI_X0},{ROI_X1}] y[{ROI_Y0},{ROI_Y1}] µm  "
          f"match radius {cfg.match_radius}µm")

    # ── pred ────────────────────────────────────────────────────────────
    d = np.load(cfg.npz, allow_pickle=True)
    cell_loc = d["cell_loc"]
    seg      = d["seg"].astype(np.int64)
    tx_x, tx_y, tx_gene = d["x"], d["y"], d["gene"]
    n_pred = len(cell_loc)
    print(f"\n  BOMS: {n_pred:,} cells   assigned transcripts {len(seg):,}")
    print(f"        h_s={d['h_s'] if 'h_s' in d else '?'}  "
          f"epochs={d['epochs'] if 'epochs' in d else '?'}")

    # ── GT centroid ─────────────────────────────────────────────────────────
    meta = pd.read_csv(META).rename(columns={"Unnamed: 0": "cell_id"})
    gt = meta[(meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
              (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
              ].reset_index(drop=True)
    n_gt = len(gt)
    print(f"  GT  : {n_gt:,} cells (cell_metadata, Vizgen official)")

    # ── one-to-one greedy matching ──────────────────────────────────────────────────
    gxy  = gt[["center_x", "center_y"]].values.astype(float)
    tree = cKDTree(gxy)
    dd, gi = tree.query(cell_loc, k=1, distance_upper_bound=cfg.match_radius)
    best = {}
    for pi in range(n_pred):
        if not np.isfinite(dd[pi]) or dd[pi] >= cfg.match_radius:
            continue
        g = int(gi[pi])
        if g not in best or dd[pi] < best[g][1]:
            best[g] = (pi, dd[pi])
    pairs   = [(pi, g, x) for g, (pi, x) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred if n_pred else 0.0
    rec  = n_match / n_gt   if n_gt   else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    print(f"\n  matched {n_match:,}   P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    print(f"  centroid shift median {np.nanmedian(shifts):.2f}µm  "
          f"p95 {np.nanpercentile(shifts,95):.2f}µm")

    cp = cs = cm = cr = vc = vj = vp = overlap = acc = np.nan
    n_valid = 0

    use_mask = (not cfg.no_mask) and os.path.exists(NUC_MASK)
    if not use_mask:
        print(f"\n  ⚠ no nuclei_mask ({NUC_MASK})")
        print(f"    -> count / vec / assign set to NaN")
    else:
        import tifffile
        from scipy import ndimage
        print(f"\n  per-transcript GT <- nuclei_mask (independent of BOMS, not circular)")
        mask = tifffile.imread(NUC_MASK)
        H, W = mask.shape
        nuc_ids = np.unique(mask[mask > 0])
        print(f"    mask {H}×{W}   nuclei {len(nuc_ids):,}")

        # nucleus -> GT cell (nearest centroid)
        n2g = {}
        if len(nuc_ids):
            com = np.array(ndimage.center_of_mass(mask > 0, mask, nuc_ids))
            nuc_y, nuc_x = com[:, 0] + BIN_OY, com[:, 1] + BIN_OX
            dn, gj = tree.query(np.stack([nuc_x, nuc_y], 1), k=1,
                                distance_upper_bound=cfg.match_radius)
            for k, nid in enumerate(nuc_ids):
                if np.isfinite(dn[k]) and dn[k] < cfg.match_radius:
                    n2g[int(nid)] = gt.cell_id.iloc[int(gj[k])]
        print(f"    nucleus -> GT cell mapping {len(n2g):,} / {len(nuc_ids):,}")

        # look up each transcript in the mask
        r = np.round(tx_y).astype(np.int64) - BIN_OY
        c = np.round(tx_x).astype(np.int64) - BIN_OX
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        gt_nid = np.zeros(len(tx_x), np.int64)
        gt_nid[ok] = mask[r[ok], c[ok]]
        gt_cid = pd.Series(gt_nid).map(n2g)     # NaN = no GT assignment

        # ── count consistency ──────────────────────────────────────────────────
        pred_cnt = pd.Series(seg[seg >= 0]).value_counts()
        gt_cnt   = gt_cid.dropna().value_counts()
        pc = np.array([pred_cnt.get(p[0], 0) for p in pairs], float)
        gc = np.array([gt_cnt.get(gt.cell_id.iloc[p[1]], 0)
                       for p in pairs], float)
        m = (pc > 0) & (gc > 0)
        if m.sum() > 10:
            cp = float(pearsonr(pc[m], gc[m])[0])
            cs = float(spearmanr(pc[m], gc[m])[0])
            cm = float(np.mean(np.abs(pc[m] - gc[m])))
            cr = float(np.sqrt(np.mean((pc[m] - gc[m])**2)))
        print(f"    count correlation Pearson {cp:.4f}  valid pairs {int(m.sum()):,}")

        # ── gene vector (sampled)──────────────────────────────────────────────
        rng  = np.random.default_rng(0)
        samp = pairs if len(pairs) <= cfg.vec_sample else \
            [pairs[i] for i in rng.choice(len(pairs), cfg.vec_sample,
                                          replace=False)]
        sp = {p[0] for p in samp}
        sg = {gt.cell_id.iloc[p[1]] for p in samp}

        # ★ column must not be named gt (clashes with DataFrame.gt) -> use gtcell
        df = pd.DataFrame({"pred": seg, "gtcell": gt_cid.values,
                           "gene": tx_gene})
        df = df[df["pred"] >= 0]                      # drop unassigned
        pv = (df[df["pred"].isin(sp)]
                .groupby(["pred", "gene"]).size().unstack(fill_value=0))
        gv = (df.dropna(subset=["gtcell"])
                .loc[lambda t: t["gtcell"].isin(sg)]
                .groupby(["gtcell", "gene"]).size().unstack(fill_value=0))
        common = sorted(set(pv.columns) & set(gv.columns))
        if len(common) >= 50:
            pv, gv = pv[common], gv[common]
            cos_l, js_l, pr_l = [], [], []
            for pi, gj, _ in samp:
                gid = gt.cell_id.iloc[gj]
                if pi not in pv.index or gid not in gv.index:
                    continue
                a = pv.loc[pi].values.astype(float)
                b = gv.loc[gid].values.astype(float)
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
        print(f"    gene vector  common genes {len(common)}  valid pairs {n_valid:,}  "
              f"cosine {vc:.4f}")

        # ── assignment consistency ──────────────────────────────────────────────────
        p2g    = {p[0]: gt.cell_id.iloc[p[1]] for p in pairs}
        has_gt = gt_cid.notna().values
        exp    = pd.Series(seg).map(p2g)
        both_m = has_gt & exp.notna().values & (seg >= 0)
        both   = int(both_m.sum())
        if both:
            correct = int((exp[both_m].values == gt_cid[both_m].values).sum())
            overlap = both / len(seg)
            acc     = correct / both
            print(f"    both assigned {both:,}  agree {correct:,}  "
                  f"accuracy {acc:.4f}")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="BOMS", dataset="merfish",
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
    )
    print(f"\n{SEP}\nResult (BOMS × MERFISH)\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            print(f"  {k:22} {v}")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    pd.DataFrame([res]).to_csv(cfg.out, index=False)
    print(f"\n  Saved -> {cfg.out}")
    if use_mask:
        print(f"  note: per-transcript GT uses nuclei_mask as proxy, "
              f"BOMS has no prior -> not circular, metrics valid")


if __name__ == "__main__":
    main()