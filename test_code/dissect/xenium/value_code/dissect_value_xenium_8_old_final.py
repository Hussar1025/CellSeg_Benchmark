#!/usr/bin/env python3
"""
DISSECT × Xenium 4 / xenium_liver evaluation.

Adapted to the same ROI/evaluation logic as:
  yourpath/proseg/value/proseg_value_xenium_2.py

Key differences:
- Prediction is DISSECT's ROI-local mask.npy.
- Predicted cell centroids are computed from mask labels.
- Xenium transcripts are assigned to DISSECT cells by sampling mask labels
  at transcript coordinates after global-µm -> global-pixel -> ROI-local-pixel conversion.
- ROI geometry is read from DISSECT roi_info.json instead of being re-derived manually.
"""

import os
import time
import json
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 72

DISSECT_DIR = "yourpath/DISSECT/xenium_8_old_20260914"
DATA_DIR = "yourpath/dataset/xenium_8"

MASK_NPY = f"{DISSECT_DIR}/mask.npy"
ROI_INFO = f"{DISSECT_DIR}/roi_info.json"

OUT_CSV = "yourpath/DISSECT/value/xenium_8_old.csv"
CKPT_DIR = "yourpath/DISSECT/value/ckpt_xenium_8_old"

PIXEL_SIZE = 0.2125
MATCH_RADIUS = 15.6760

os.makedirs(CKPT_DIR, exist_ok=True)
P = lambda *a: print(*a, flush=True)


def ckpt(name, fn, force=False):
    f = f"{CKPT_DIR}/{name}.pkl"
    if os.path.exists(f) and not force:
        with open(f, "rb") as h:
            obj = pickle.load(h)
        P(f"    ⚡ reuse {name}.pkl")
        return obj
    t0 = time.time()
    obj = fn()
    with open(f, "wb") as h:
        pickle.dump(obj, h, protocol=4)
    P(f"    ✓ {name} done ({time.time()-t0:.1f}s) → {f}")
    return obj


def read_roi_info():
    with open(ROI_INFO, "r", encoding="utf-8") as f:
        info = json.load(f)

    x0 = int(info["roi_x0_px"])
    y0 = int(info["roi_y0_px"])

    if "roi_width_px" in info:
        w = int(info["roi_width_px"])
        h = int(info["roi_height_px"])
    else:
        w = int(info["roi_x1_px_exclusive"]) - x0
        h = int(info["roi_y1_px_exclusive"]) - y0

    return info, x0, y0, w, h


def load_mask_and_pred(x0, y0, w, h):
    mask = np.load(MASK_NPY, mmap_mode="r")
    if mask.shape != (h, w):
        raise RuntimeError(
            f"mask shape {mask.shape} != ROI {(h, w)} from {ROI_INFO}"
        )

    # Only foreground pixels, avoiding full 25M-pixel coordinate grids.
    yy, xx = np.nonzero(mask)
    lab = np.asarray(mask[yy, xx], dtype=np.int64)

    ids = np.unique(lab)
    ids = ids[ids > 0]
    if len(ids) == 0:
        raise RuntimeError("DISSECT mask has no positive labels.")

    # Robust to non-contiguous label ids.
    order = np.searchsorted(ids, lab)
    cnt = np.bincount(order, minlength=len(ids)).astype(np.int64)
    sum_x = np.bincount(order, weights=xx.astype(np.float64),
                        minlength=len(ids))
    sum_y = np.bincount(order, weights=yy.astype(np.float64),
                        minlength=len(ids))

    cx_local_px = sum_x / cnt
    cy_local_px = sum_y / cnt

    # Pixel-center convention. The 0.5 px term is only 0.10625 µm.
    centroid_x = (x0 + cx_local_px + 0.5) * PIXEL_SIZE
    centroid_y = (y0 + cy_local_px + 0.5) * PIXEL_SIZE

    pred = pd.DataFrame({
        "cell": ids.astype(np.int64),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "area_px": cnt,
    })

    return pred, mask.shape, int(len(ids)), int(len(xx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--vec-sample", type=int, default=3000)
    ap.add_argument("--force", action="store_true", help="ignore cache and recompute")
    cfg = ap.parse_args()

    P(SEP)
    P("DISSECT × Xenium 8 OLD 2026-09-14 evaluation")
    P(SEP)

    info, ROI_X0_PX, ROI_Y0_PX, ROI_W, ROI_H = read_roi_info()
    ROI_X1_PX = ROI_X0_PX + ROI_W
    ROI_Y1_PX = ROI_Y0_PX + ROI_H

    ROI_X0 = ROI_X0_PX * PIXEL_SIZE
    ROI_X1 = ROI_X1_PX * PIXEL_SIZE
    ROI_Y0 = ROI_Y0_PX * PIXEL_SIZE
    ROI_Y1 = ROI_Y1_PX * PIXEL_SIZE

    P(f"  ROI px: x[{ROI_X0_PX},{ROI_X1_PX}) y[{ROI_Y0_PX},{ROI_Y1_PX})")
    P(f"  ROI µm: x[{ROI_X0:.3f},{ROI_X1:.3f}) y[{ROI_Y0:.3f},{ROI_Y1:.3f})")
    P(f"  ROI size: {ROI_W}×{ROI_H} px")
    P(f"  pixel_size={PIXEL_SIZE} µm/px")
    P(f"  match_radius={MATCH_RADIUS} µm")
    P(f"  mask={MASK_NPY}")

    # ── pred cells ───────────────────────────────────────────────────────
    P("\n[1] DISSECT predicted cells from mask.npy")
    def _pred():
        pred, mask_shape, n_labels, n_fg = load_mask_and_pred(
            ROI_X0_PX, ROI_Y0_PX, ROI_W, ROI_H
        )
        return pred, mask_shape, n_labels, n_fg

    pred, mask_shape, n_labels, n_fg = ckpt("pred_roi", _pred, cfg.force)
    P(f"    mask_shape={mask_shape}")
    P(f"    foreground labels={n_labels:,}")
    P(f"    foreground pixels={n_fg:,}")
    P(f"    predicted cells={len(pred):,}")

    # ── GT cells ─────────────────────────────────────────────────────────
    P("\n[2] GT cells")
    def _gt():
        g = pd.read_parquet(
            f"{DATA_DIR}/cells.parquet",
            columns=["cell_id", "x_centroid", "y_centroid"]
        )
        n0 = len(g)
        g = g[
            (g.x_centroid >= ROI_X0) & (g.x_centroid < ROI_X1) &
            (g.y_centroid >= ROI_Y0) & (g.y_centroid < ROI_Y1)
        ].reset_index(drop=True)
        return g, n0

    gt, n_gt_full = ckpt("gt_roi", _gt, cfg.force)
    P(f"    full GT={n_gt_full:,} → ROI GT={len(gt):,}")

    n_pred, n_gt = len(pred), len(gt)
    if n_pred == 0 or n_gt == 0:
        raise RuntimeError("Prediction or GT is empty.")

    # ── transcripts ──────────────────────────────────────────────────────
    P("\n[3] Xenium transcripts + DISSECT mask assignment")
    def _tx():
        t = pd.read_parquet(
            f"{DATA_DIR}/transcripts.parquet",
            columns=[
                "cell_id", "feature_name",
                "x_location", "y_location", "qv"
            ]
        )

        t = t[t.qv >= 20].copy()
        t = t[~t.feature_name.str.startswith(
            ("NegControl", "Unassigned", "BLANK", "antisense"), na=False
        )].copy()

        t = t[
            (t.x_location >= ROI_X0) & (t.x_location < ROI_X1) &
            (t.y_location >= ROI_Y0) & (t.y_location < ROI_Y1)
        ].reset_index(drop=True)

        # Match DISSECT runner convention: positive coordinates cast toward zero
        # after µm->pixel conversion, then subtract ROI origin.
        gx = (t.x_location.to_numpy(dtype=np.float64) / PIXEL_SIZE).astype(np.int64)
        gy = (t.y_location.to_numpy(dtype=np.float64) / PIXEL_SIZE).astype(np.int64)
        lx = gx - ROI_X0_PX
        ly = gy - ROI_Y0_PX

        inside = (
            (lx >= 0) & (lx < ROI_W) &
            (ly >= 0) & (ly < ROI_H)
        )

        label = np.zeros(len(t), dtype=np.int64)
        mask = np.load(MASK_NPY, mmap_mode="r")
        idx = np.flatnonzero(inside)
        label[idx] = np.asarray(mask[ly[idx], lx[idx]], dtype=np.int64)

        tx_gt = t.copy()
        tx_pred = t[label > 0][
            ["feature_name", "x_location", "y_location"]
        ].copy()
        tx_pred["cell_int"] = label[label > 0]
        tx_pred = tx_pred.rename(columns={
            "feature_name": "gene",
            "x_location": "x",
            "y_location": "y",
        })
        tx_pred = tx_pred[["cell_int", "gene", "x", "y"]].reset_index(drop=True)

        return tx_gt, tx_pred

    tx_gt, tx_pred = ckpt("transcripts_roi", _tx, cfg.force)
    P(f"    GT ROI transcripts={len(tx_gt):,}")
    P(f"    DISSECT-assigned transcripts={len(tx_pred):,}")
    P(f"    raw mask assignment rate={len(tx_pred)/max(len(tx_gt),1):.4f}")

    # ── one-to-one centroid matching ─────────────────────────────────────
    P("\n[4] One-to-one greedy centroid matching")
    pxy = pred[["centroid_x", "centroid_y"]].values.astype(float)
    gxy = gt[["x_centroid", "y_centroid"]].values.astype(float)

    tree = cKDTree(gxy)
    d, gi = tree.query(pxy, k=1, distance_upper_bound=MATCH_RADIUS)

    best = {}
    for pi in range(n_pred):
        if not np.isfinite(d[pi]) or d[pi] >= MATCH_RADIUS:
            continue
        gj = int(gi[pi])
        if gj not in best or d[pi] < best[gj][1]:
            best[gj] = (pi, float(d[pi]))

    pairs = [(pi, gj, dist) for gj, (pi, dist) in best.items()]
    n_match = len(pairs)

    prec = n_match / n_pred
    rec = n_match / n_gt
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    shifts = np.array([p[2] for p in pairs], dtype=float) \
        if pairs else np.array([np.nan])

    P(f"    matched={n_match:,}")
    P(f"    precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}")
    P(f"    shift median={np.nanmedian(shifts):.2f} µm "
      f"p95={np.nanpercentile(shifts,95):.2f} µm")

    # ── transcript count consistency ─────────────────────────────────────
    P("\n[5] Transcript-count consistency")
    pred_cnt = tx_pred.groupby("cell_int").size()

    gt_cnt = tx_gt[
        tx_gt.cell_id.notna() &
        (tx_gt.cell_id != "UNASSIGNED")
    ].cell_id.value_counts()

    pc = np.array([
        pred_cnt.get(int(pred.cell.iloc[pi]), 0)
        for pi, _, _ in pairs
    ], dtype=float)

    gc = np.array([
        gt_cnt.get(gt.cell_id.iloc[gj], 0)
        for _, gj, _ in pairs
    ], dtype=float)

    valid_cnt = (pc > 0) & (gc > 0)
    if valid_cnt.sum() > 10:
        cp = float(pearsonr(pc[valid_cnt], gc[valid_cnt])[0])
        cs = float(spearmanr(pc[valid_cnt], gc[valid_cnt])[0])
        cm = float(np.mean(np.abs(pc[valid_cnt] - gc[valid_cnt])))
        cr = float(np.sqrt(np.mean((pc[valid_cnt] - gc[valid_cnt]) ** 2)))
    else:
        cp = cs = cm = cr = np.nan

    P(f"    valid pairs={int(valid_cnt.sum()):,}")
    P(f"    Pearson={cp:.4f} Spearman={cs:.4f}")

    # ── gene-vector consistency ──────────────────────────────────────────
    P(f"\n[6] Gene-vector consistency (sample <= {cfg.vec_sample})")
    rng = np.random.default_rng(0)
    samp = pairs if len(pairs) <= cfg.vec_sample else [
        pairs[i] for i in rng.choice(len(pairs), cfg.vec_sample, replace=False)
    ]

    sp = {int(pred.cell.iloc[pi]) for pi, _, _ in samp}
    sg = {gt.cell_id.iloc[gj] for _, gj, _ in samp}

    pv = (
        tx_pred[tx_pred.cell_int.isin(sp)]
        .groupby(["cell_int", "gene"])
        .size()
        .unstack(fill_value=0)
    )

    gv = (
        tx_gt[tx_gt.cell_id.isin(sg)]
        .groupby(["cell_id", "feature_name"])
        .size()
        .unstack(fill_value=0)
    )

    common = sorted(set(pv.columns) & set(gv.columns))
    vc = vj = vp = np.nan
    n_valid = 0

    if len(common) >= 50:
        pv = pv[common]
        gv = gv[common]
        cos_l, js_l, pr_l = [], [], []

        for pi, gj, _ in samp:
            pid = int(pred.cell.iloc[pi])
            gid = gt.cell_id.iloc[gj]
            if pid not in pv.index or gid not in gv.index:
                continue

            a = pv.loc[pid].values.astype(float)
            b = gv.loc[gid].values.astype(float)

            if a.sum() < 5 or b.sum() < 5:
                continue

            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                continue

            cos_l.append(float(a @ b / (na * nb)))
            js_l.append(float(jensenshannon(a / a.sum(), b / b.sum())))

            if a.std() > 0 and b.std() > 0:
                pr_l.append(float(pearsonr(a, b)[0]))

        n_valid = len(cos_l)
        if n_valid > 10:
            vc = float(np.mean(cos_l))
            vj = float(np.nanmean(js_l))
            vp = float(np.mean(pr_l)) if pr_l else np.nan

    P(f"    common genes={len(common)} valid pairs={n_valid:,}")
    P(f"    cosine={vc:.4f} JS={vj:.4f} Pearson={vp:.4f}")

    # ── assignment consistency ───────────────────────────────────────────
    P("\n[7] Assignment consistency")
    p2g = {
        int(pred.cell.iloc[pi]): gt.cell_id.iloc[gj]
        for pi, gj, _ in pairs
    }

    # Keep the same 1-µm rounded coordinate-key convention as the ProSeg evaluator.
    kp = (
        tx_pred.x.round(0).astype(np.int64) * 100000 +
        tx_pred.y.round(0).astype(np.int64)
    )
    kg = (
        tx_gt.x_location.round(0).astype(np.int64) * 100000 +
        tx_gt.y_location.round(0).astype(np.int64)
    )

    gmap = dict(zip(kg.values, tx_gt.cell_id.values))
    both = correct = 0

    for k, pid in zip(kp.values, tx_pred.cell_int.values):
        gt_id = gmap.get(k)
        if gt_id is None or pd.isna(gt_id) or gt_id == "UNASSIGNED":
            continue

        expected = p2g.get(int(pid))
        if expected is None:
            continue

        both += 1
        if expected == gt_id:
            correct += 1

    overlap = both / max(len(tx_pred), 1)
    acc = correct / both if both else np.nan
    P(f"    both-assigned={both:,} correct={correct:,}")
    P(f"    overlap={overlap:.4f} accuracy={acc:.4f}")

    # ── save ─────────────────────────────────────────────────────────────
    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan

    res = dict(
        method="DISSECT",
        dataset="xenium_8",
        gt_cells=n_gt,
        pred_cells=n_pred,
        cell_ratio=rnd(n_pred / n_gt),
        matched=n_match,
        precision=rnd(prec),
        recall=rnd(rec),
        f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(shifts)),
        loc_median_um=rnd(np.nanmedian(shifts)),
        loc_p95_um=rnd(np.nanpercentile(shifts, 95)),
        count_pearson=rnd(cp),
        count_spearman=rnd(cs),
        count_mae=rnd(cm, 2),
        count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc),
        vec_js_dist=rnd(vj),
        vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap),
        assign_accuracy=rnd(acc),
        mask_assignment_rate=rnd(len(tx_pred) / max(len(tx_gt), 1)),
        foreground_pixels=n_fg,
    )

    P(f"\n{SEP}")
    P("RESULT: DISSECT × Xenium 4 / xenium_liver ROI")
    P(SEP)
    for k, v in res.items():
        if k not in ("method", "dataset"):
            P(f"  {k:22} {v}")

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    pd.DataFrame([res]).to_csv(cfg.out, index=False)
    P(f"\n  saved -> {cfg.out}")


if __name__ == "__main__":
    main()