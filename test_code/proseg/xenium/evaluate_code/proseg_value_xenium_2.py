#!/usr/bin/env python
"""
ProSeg × Xenium breast cancer evaluation (ROI-based)

* Why the ROI filter matters:
  ProSeg runs the whole image -> 190,987 cells / full-image GT 209,467 = 0.912 (actually good)
  but UCS/BOMS/Cellist all run only the 5000x5000 px ROI (GT 8,809);
  without filtering you get a misleading ratio=21.68 and think tuning is needed.

ROI identical to the other methods:
  CX=53994/2×0.2125  CY=27420/2×0.2125  HALF=5000×0.2125/2
  -> x[5205.6,6268.1]  y[2382.1,3444.6] um

ProSeg output columns (confirmed):
  cell-metadata.csv.gz      : cell, centroid_x, centroid_y, volume, ...
  transcript-metadata.csv.gz: assignment(float), background(bool), gene, x, y

Checkpoint resume: intermediate results (GT/pred/transcripts) are cached under
ckpt/; reruns recover in seconds.
"""
import os, sys, time, pickle, argparse, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

PROSEG_DIR = "yourpath/proseg/output/xenium_2"
DATA_DIR   = "yourpath/dataset/xenium_breast"
OUT_CSV    = "yourpath/proseg/value/xenium_2.csv"
CKPT_DIR   = "yourpath/proseg/value/ckpt_xenium_2"

PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE
CY_UM = 27420 / 2 * PIXEL_SIZE
HALF  = 5000 * PIXEL_SIZE / 2
ROI_X0, ROI_X1 = CX_UM - HALF, CX_UM + HALF
ROI_Y0, ROI_Y1 = CY_UM - HALF, CY_UM + HALF
MATCH_RADIUS = 12.0        # breast-cell diameter ~11-13 um

os.makedirs(CKPT_DIR, exist_ok=True)
P = lambda *a: print(*a, flush=True)          # real-time output


def ckpt(name, fn, force=False):
    """generic checkpoint cache"""
    f = f"{CKPT_DIR}/{name}.pkl"
    if os.path.exists(f) and not force:
        with open(f, "rb") as h:
            obj = pickle.load(h)
        P(f"    ⚡ reusing {name}.pkl")
        return obj
    t0 = time.time()
    obj = fn()
    with open(f, "wb") as h:
        pickle.dump(obj, h, protocol=4)
    P(f"    ✓ {name} done ({time.time()-t0:.0f}s) -> cached")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-image", action="store_true",
                    help="do not filter ROI; compare against full-image GT 209,467")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--vec-sample", type=int, default=3000)
    ap.add_argument("--force", action="store_true", help="ignore cache and recompute")
    cfg = ap.parse_args()

    mode = "full" if cfg.full_image else "ROI"
    P(SEP); P(f"ProSeg × Xenium breast cancer evaluation ({mode} mode)"); P(SEP)
    if not cfg.full_image:
        P(f"  ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm")
    P(f"  match radius {MATCH_RADIUS} um   checkpoint dir {CKPT_DIR}")
    sfx = "full" if cfg.full_image else "roi"

    # ── pred ────────────────────────────────────────────────────────────
    P(f"\n[1] ProSeg cells")
    def _pred():
        d = pd.read_csv(f"{PROSEG_DIR}/cell-metadata.csv.gz")
        n0 = len(d)
        if not cfg.full_image:
            d = d[(d.centroid_x >= ROI_X0) & (d.centroid_x < ROI_X1) &
                  (d.centroid_y >= ROI_Y0) & (d.centroid_y < ROI_Y1)
                  ].reset_index(drop=True)
        return d, n0
    pred, n_full = ckpt(f"pred_{sfx}", _pred, cfg.force)
    P(f"    full image {n_full:,} -> {len(pred):,} in {mode}")

    # ── GT ──────────────────────────────────────────────────────────────
    P(f"\n[2] GT cells")
    def _gt():
        g = pd.read_parquet(f"{DATA_DIR}/cells.parquet",
                            columns=["cell_id", "x_centroid", "y_centroid"])
        n0 = len(g)
        if not cfg.full_image:
            g = g[(g.x_centroid >= ROI_X0) & (g.x_centroid < ROI_X1) &
                  (g.y_centroid >= ROI_Y0) & (g.y_centroid < ROI_Y1)
                  ].reset_index(drop=True)
        return g, n0
    gt, n_gt_full = ckpt(f"gt_{sfx}", _gt, cfg.force)
    P(f"    full image {n_gt_full:,} -> {len(gt):,} in {mode}")

    # ── transcripts ───────────────────────────────────────────────────────────
    P(f"\n[3] transcripts (first run ~2-3 min)")
    def _txp():
        t = pd.read_csv(f"{PROSEG_DIR}/transcript-metadata.csv.gz")
        t = t[t["background"] == False].copy()
        t["cell_int"] = t["assignment"].astype(int)
        if not cfg.full_image:
            t = t[(t.x >= ROI_X0) & (t.x < ROI_X1) &
                  (t.y >= ROI_Y0) & (t.y < ROI_Y1)]
        return t[["cell_int", "gene", "x", "y"]].reset_index(drop=True)
    tx_pred = ckpt(f"txpred_{sfx}", _txp, cfg.force)
    P(f"    ProSeg assigned {len(tx_pred):,}")

    def _txg():
        t = pd.read_parquet(f"{DATA_DIR}/transcripts.parquet",
                            columns=["cell_id", "feature_name",
                                     "x_location", "y_location", "qv"])
        t = t[t.qv >= 20]
        t = t[~t.feature_name.str.startswith(
            ("NegControl", "Unassigned", "BLANK", "antisense"), na=False)]
        if not cfg.full_image:
            t = t[(t.x_location >= ROI_X0) & (t.x_location < ROI_X1) &
                  (t.y_location >= ROI_Y0) & (t.y_location < ROI_Y1)]
        return t.reset_index(drop=True)
    tx_gt = ckpt(f"txgt_{sfx}", _txg, cfg.force)
    P(f"    GT transcripts {len(tx_gt):,}")

    n_pred, n_gt = len(pred), len(gt)
    if n_pred == 0 or n_gt == 0:
        P("  ✘ prediction or GT is empty"); return

    # ── matching ─────────────────────────────────────────────────────────────
    P(f"\n[4] one-to-one greedy matching")
    pxy = pred[["centroid_x", "centroid_y"]].values.astype(float)
    gxy = gt[["x_centroid", "y_centroid"]].values.astype(float)
    tree = cKDTree(gxy)
    d, gi = tree.query(pxy, k=1, distance_upper_bound=MATCH_RADIUS)
    best = {}
    for pi in range(n_pred):
        if not np.isfinite(d[pi]) or d[pi] >= MATCH_RADIUS:
            continue
        g = int(gi[pi])
        if g not in best or d[pi] < best[g][1]:
            best[g] = (pi, d[pi])
    pairs = [(pi, g, x) for g, (pi, x) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred
    rec  = n_match / n_gt
    f1   = 2*prec*rec/(prec+rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    P(f"    matched {n_match:,}   P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    P(f"    shift median {np.nanmedian(shifts):.2f} um  "
      f"p95 {np.nanpercentile(shifts,95):.2f} um")

    # ── count consistency ───────────────────────────────────────────────────────
    P(f"\n[5] count consistency")
    pred_cnt = tx_pred.groupby("cell_int").size()
    gt_cnt = tx_gt[tx_gt.cell_id.notna() &
                   (tx_gt.cell_id != "UNASSIGNED")].cell_id.value_counts()
    pc = np.array([pred_cnt.get(int(pred.cell.iloc[p[0]]), 0)
                   for p in pairs], float)
    gc = np.array([gt_cnt.get(gt.cell_id.iloc[p[1]], 0) for p in pairs], float)
    m = (pc > 0) & (gc > 0)
    if m.sum() > 10:
        cp = float(pearsonr(pc[m], gc[m])[0])
        cs = float(spearmanr(pc[m], gc[m])[0])
        cm = float(np.mean(np.abs(pc[m] - gc[m])))
        cr = float(np.sqrt(np.mean((pc[m] - gc[m])**2)))
    else:
        cp = cs = cm = cr = np.nan
    P(f"    Pearson {cp:.4f}  Spearman {cs:.4f}  valid pairs {int(m.sum()):,}")

    # ── gene vector (sample) ──────────────────────────────────────────────────
    P(f"\n[6] gene vector (sample {cfg.vec_sample})")
    rng = np.random.default_rng(0)
    samp = pairs if len(pairs) <= cfg.vec_sample else \
        [pairs[i] for i in rng.choice(len(pairs), cfg.vec_sample, replace=False)]
    sp = {int(pred.cell.iloc[p[0]]) for p in samp}
    sg = {gt.cell_id.iloc[p[1]] for p in samp}
    pv = (tx_pred[tx_pred.cell_int.isin(sp)]
            .groupby(["cell_int", "gene"]).size().unstack(fill_value=0))
    gv = (tx_gt[tx_gt.cell_id.isin(sg)]
            .groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0))
    common = sorted(set(pv.columns) & set(gv.columns))
    vc = vj = vp = np.nan
    n_valid = 0
    if len(common) >= 50:
        pv, gv = pv[common], gv[common]
        cos_l, js_l, pr_l = [], [], []
        for pi, gj, _ in samp:
            pid = int(pred.cell.iloc[pi]); gid = gt.cell_id.iloc[gj]
            if pid not in pv.index or gid not in gv.index:
                continue
            a = pv.loc[pid].values.astype(float)
            b = gv.loc[gid].values.astype(float)
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
    P(f"    common genes {len(common)}  valid pairs {n_valid:,}  cosine {vc:.4f}")

    # ── assignment consistency (1 um coordinate alignment) ────────────────────────────────────
    P(f"\n[7] assignment consistency")
    p2g = {int(pred.cell.iloc[p[0]]): gt.cell_id.iloc[p[1]] for p in pairs}
    kp = (tx_pred.x.round(0).astype(np.int64) * 100000 +
          tx_pred.y.round(0).astype(np.int64))
    kg = (tx_gt.x_location.round(0).astype(np.int64) * 100000 +
          tx_gt.y_location.round(0).astype(np.int64))
    gmap = dict(zip(kg.values, tx_gt.cell_id.values))
    both = correct = 0
    for k, pid in zip(kp.values, tx_pred.cell_int.values):
        g_ = gmap.get(k)
        if g_ is None or pd.isna(g_) or g_ == "UNASSIGNED":
            continue
        e = p2g.get(int(pid))
        if e is None:
            continue
        both += 1
        if e == g_:
            correct += 1
    overlap = both / max(len(tx_pred), 1)
    acc = correct / both if both else np.nan
    P(f"    both assigned {both:,}  consistent {correct:,}  accuracy {acc:.4f}")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="ProSeg", dataset=f"xenium_breast_{mode}",
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
    )
    P(f"\n{SEP}\nResult (ProSeg × Xenium breast cancer {mode})\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            P(f"  {k:22} {v}")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    pd.DataFrame([res]).to_csv(cfg.out, index=False)
    P(f"\n  written -> {cfg.out}")
    if not cfg.full_image:
        P(f"  full-image comparison: add --full-image (GT {n_gt_full:,})")


if __name__ == "__main__":
    main()