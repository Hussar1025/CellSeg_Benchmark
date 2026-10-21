#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_boms_merfish_3.py —— BOMS × MERFISH human liver cancer liver2 evaluation (19 metrics, mask version)
=========================================================================
Per-transcript GT uses the nuclei_mask.tif built by UCS as proxy (official cell_boundaries rasterized),
method follows eval_ucs_merfish.py: {nuclei_mask_id -> gt_cell_id} lookup mapping + hard ROI check.
BOMS uses no prior and is independent of that mask -> transcript-level metrics are not circular, valid.

liver2 mask parameters (mask 1081×1081 vs ROI side 1080µm back-inferred + nonzero nuclei 7963 confirms):
  bin = 1.0 µm/px, origin = floor(ROI_min)

# run
conda activate ucs
python eval_boms_merfish_3.py
"""

import os, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

# ── config (this file = merfish_3 / liver2)──────────────────────────────────
LABEL     = "MERFISH human liver cancer liver2"
NPZ       = "yourpath/boms_merfish_3/boms_merfish_3.npz"
DATA_DIR  = "yourpath/dataset/merfish_liver2"
MASK_PATH = "yourpath/ucs_merfish_liver2/nuclei_mask.tif"
OUT       = "yourpath/eval_results/boms/merfish_3.csv"

MASK_BIN_UM = 1.0             # µm/px; set None to back-infer from ROI side / mask side
MAP_MAX_DIST = 5.0            # nuclei_mask centroid -> GT centroid nearest-neighbor cap, µm
DEFAULT_MATCH_RADIUS = 10.0   # must match the other methods on this dataset
VEC_SAMPLE = 2000

ID_CANDS = ("cell_id", "cell", "cellid", "cell_index", "entityid", "entity_id")


def find(root, *pats):
    root = Path(root)
    for p in pats:
        h = sorted(root.rglob(p))
        if h:
            return h[0]
    return None


def mask_to_cells(mask, ox, oy, binum):
    ids = np.unique(mask)
    ids = ids[ids > 0]
    com = np.asarray(ndi.center_of_mass(np.ones_like(mask, np.uint8), mask, ids),
                     dtype=np.float64)
    return pd.DataFrame({
        "mask_id": ids.astype(np.int64),
        "x": com[:, 1] * binum + ox,
        "y": com[:, 0] * binum + oy,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--mask", default=MASK_PATH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=DEFAULT_MATCH_RADIUS)
    ap.add_argument("--vec-sample", type=int, default=VEC_SAMPLE)
    cfg = ap.parse_args()

    print(SEP); print(f"BOMS × {LABEL} evaluation (mask version)"); print(SEP)

    # ── pred npz ─────────────────────────────────────────────────────────
    d = np.load(cfg.npz, allow_pickle=True)
    cell_loc = d["cell_loc"].astype(np.float64)
    seg      = d["seg"].astype(np.int64)          # 0-based
    tx_x     = d["x"].astype(np.float64)
    tx_y     = d["y"].astype(np.float64)
    tx_gene  = d["gene"].astype(str)
    n_pred   = len(cell_loc)
    if "roi" not in d.files:
        raise SystemExit("npz has no roi field")
    ROI_X0, ROI_X1, ROI_Y0, ROI_Y1 = [float(v) for v in d["roi"]]
    print(f"  ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm"
          f"   match radius {cfg.match_radius}µm")
    print(f"  BOMS: {n_pred:,} cells   assigned transcripts {len(seg):,}")

    # ── mask + ROI hard check ───────────────────────────────────────────────
    if not os.path.exists(cfg.mask):
        raise SystemExit(f"mask missing: {cfg.mask}")
    mask = tifffile.imread(cfg.mask)
    H, W = mask.shape
    binx = (ROI_X1 - ROI_X0) / W
    biny = (ROI_Y1 - ROI_Y0) / H
    binum = MASK_BIN_UM if MASK_BIN_UM is not None else (binx + biny) / 2
    ox, oy = int(np.floor(ROI_X0)), int(np.floor(ROI_Y0))
    print(f"  mask {H}×{W}  back-inferred bin x={binx:.4f} y={biny:.4f} µm/px  using {binum:.4f}")
    print(f"  bin origin ({ox},{oy})")
    est_w = (ROI_X1 - ROI_X0) / binum
    est_h = (ROI_Y1 - ROI_Y0) / binum
    if abs(est_w - W) > 2 or abs(est_h - H) > 2:
        raise SystemExit(
            f"ROI back-inferred size {est_h:.1f}×{est_w:.1f} mismatches mask {H}×{W}, "
            f"bin/origin convention may be wrong, aborted")
    print(f"  ✓ ROI size check passed (back-inferred {est_h:.1f}×{est_w:.1f})")

    # ── GT centroid (cell_metadata)─────────────────────────────────────────
    mp = find(DATA_DIR, "*cell_metadata*.csv*")
    if mp is None:
        raise SystemExit(f"no *cell_metadata*.csv under {DATA_DIR}")
    meta = pd.read_csv(mp)
    mid = next((c for c in meta.columns if c.lower() in ID_CANDS), None) or meta.columns[0]
    if "center_x" not in meta.columns or "center_y" not in meta.columns:
        raise SystemExit(f"[{mp.name}] missing center_x/center_y, actual columns: {list(meta.columns)}")
    meta = meta.rename(columns={mid: "cell_id"})
    gt = meta[(meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
              (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)].reset_index(drop=True)
    gt["cell_id"] = gt["cell_id"].astype(str)
    n_gt = len(gt)
    print(f"  GT  : {n_gt:,} cells ({mp.name}, id={mid})")
    if n_gt == 0:
        raise SystemExit("no GT cells in the ROI")

    # ── cell matching (pred centroid -> GT centroid, one-to-one greedy)─────────────────────
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
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    print(f"\n  matched {n_match:,}   P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    print(f"  centroid shift median {np.nanmedian(shifts):.2f}µm  p95 {np.nanpercentile(shifts,95):.2f}µm")

    # ── {nuclei_mask_id -> gt_cell_id} lookup ─────────────────────────────
    nuc = mask_to_cells(mask, ox, oy, binum)
    dn, gj = tree.query(nuc[["x", "y"]].values, k=1)
    ok = dn <= MAP_MAX_DIST
    nuc2gt = {int(mid_): gt.cell_id.iloc[int(gjx)]
              for mid_, gjx in zip(nuc.loc[ok, "mask_id"].values, gj[ok])}
    print(f"  nuclei_mask {len(nuc):,} nuclei -> matched GT {int(ok.sum()):,} "
          f"({ok.sum()/max(len(nuc),1)*100:.1f}%)  centroid dist median {np.median(dn):.2f}µm")
    if ok.sum() / max(len(nuc), 1) < 0.8:
        print("  ⚠ match ratio low, transcript-level GT assignment may be incomplete")

    # ── each transcript looks up mask -> GT cell ──────────────────────────────
    r = np.round((tx_y - oy) / binum).astype(np.int64)
    c = np.round((tx_x - ox) / binum).astype(np.int64)
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    if inside.mean() < 0.8:
        print(f"  ⚠ only {inside.mean():.1%} transcripts fall inside the mask, coordinate convention suspect")
    nid = np.zeros(len(tx_x), np.int64)
    nid[inside] = mask[r[inside], c[inside]]
    gt_cid = pd.Series(nid).map(nuc2gt)          # NaN = no GT assignment

    # ── count metrics ────────────────────────────────────────────────────────
    pred_cnt = pd.Series(seg[seg >= 0]).value_counts()
    gt_cnt   = gt_cid.dropna().value_counts()
    pc = np.array([pred_cnt.get(p[0], 0) for p in pairs], float)
    gc = np.array([gt_cnt.get(gt.cell_id.iloc[p[1]], 0) for p in pairs], float)
    m = (pc > 0) & (gc > 0)
    cp = cs = cm = cr = np.nan
    if m.sum() > 10:
        cp = float(pearsonr(pc[m], gc[m])[0])
        cs = float(spearmanr(pc[m], gc[m])[0])
        cm = float(np.mean(np.abs(pc[m] - gc[m])))
        cr = float(np.sqrt(np.mean((pc[m] - gc[m]) ** 2)))
    print(f"  count correlation Pearson {cp:.4f}  Spearman {cs:.4f}  valid pairs {int(m.sum()):,}")

    # ── gene vector (sampled)─────────────────────────────────────────────────
    rng  = np.random.default_rng(0)
    samp = pairs if len(pairs) <= cfg.vec_sample else \
        [pairs[i] for i in rng.choice(len(pairs), cfg.vec_sample, replace=False)]
    sp = {p[0] for p in samp}
    sg = {gt.cell_id.iloc[p[1]] for p in samp}
    df = pd.DataFrame({"pred": seg, "gtcell": gt_cid.values, "gene": tx_gene})
    df = df[df["pred"] >= 0]
    pv = (df[df["pred"].isin(sp)].groupby(["pred", "gene"]).size().unstack(fill_value=0))
    gv = (df.dropna(subset=["gtcell"]).loc[lambda t: t["gtcell"].isin(sg)]
          .groupby(["gtcell", "gene"]).size().unstack(fill_value=0))
    common = sorted(set(pv.columns) & set(gv.columns))
    vc = vj = vp = np.nan
    n_valid = 0
    if len(common) >= 50:
        pv, gv = pv[common], gv[common]
        cos_l, js_l, pr_l = [], [], []
        for pi, gj_, _ in samp:
            gid = gt.cell_id.iloc[gj_]
            if pi not in pv.index or gid not in gv.index:
                continue
            a = pv.loc[pi].values.astype(float)
            b = gv.loc[gid].values.astype(float)
            if a.sum() < 5 or b.sum() < 5:
                continue
            cos_l.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            js_l.append(float(jensenshannon(a / a.sum(), b / b.sum())))
            if a.std() > 0 and b.std() > 0:
                pr_l.append(float(pearsonr(a, b)[0]))
        n_valid = len(cos_l)
        if n_valid > 10:
            vc = float(np.mean(cos_l)); vj = float(np.nanmean(js_l))
            vp = float(np.mean(pr_l)) if pr_l else np.nan
    print(f"  gene vector  common genes {len(common)}  valid pairs {n_valid:,}  cosine {vc:.4f}")

    # ── assignment metrics ────────────────────────────────────────────────────────
    p2g    = {p[0]: gt.cell_id.iloc[p[1]] for p in pairs}
    has_gt = gt_cid.notna().values
    exp    = pd.Series(seg).map(p2g)
    both_m = has_gt & exp.notna().values & (seg >= 0)
    both   = int(both_m.sum())
    overlap = acc = np.nan
    if both:
        correct = int((exp[both_m].values == gt_cid[both_m].values).sum())
        overlap = both / len(seg)
        acc     = correct / both
        print(f"  both assigned {both:,}  agree {correct:,}  accuracy {acc:.4f}")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="BOMS", dataset="merfish_3",
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred / n_gt) if n_gt else np.nan,
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
    print(f"\n{SEP}\nResult (BOMS × {LABEL})\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            print(f"  {k:22} {v}")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    pd.DataFrame([res]).to_csv(cfg.out, index=False)
    print(f"\n  Saved -> {cfg.out}")
    print("  note: transcript-level GT uses nuclei_mask as proxy; BOMS has no prior -> not circular, valid")


if __name__ == "__main__":
    main()