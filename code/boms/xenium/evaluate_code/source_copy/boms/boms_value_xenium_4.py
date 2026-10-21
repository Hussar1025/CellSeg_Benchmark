#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_boms_xenium_4.py -- BOMS × Xenium liver evaluation (19 metrics)
=========================================================================
pred : yourpath/boms_xenium_4/boms_xenium_4.npz
GT   : cells.parquet (centroids) + transcripts.parquet (cell_id assignment)
ROI  : read directly from the npz roi field (same as run, not recomputed)
seg  : 0-based (0 = cell 0, not background), cell_loc[i] <-> seg==i

# Run (in an env with numpy/pandas/scipy/pyarrow, e.g. ucs or proseg_eval)
conda activate ucs
python eval_boms_xenium_4.py
"""

import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68

# ── config ────────────────────────────────────────────────────────────
LABEL        = "Xenium liver"
NPZ_PATH     = "yourpath/boms_xenium_4/boms_xenium_4.npz"
DATA_DIR     = "yourpath/dataset/xenium_liver"
OUT_PATH     = "yourpath/eval_results/boms/xenium_4.csv"
MATCH_RADIUS = 20.0   # µm
QV_MIN       = 20.0


def find(root, *pats):
    root = Path(root)
    for p in pats:
        h = sorted(root.rglob(p))
        if h:
            return h[0]
    return None


os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
print(SEP); print(f"BOMS × {LABEL} evaluation"); print(SEP)

# ════════════════════════════════════════════════════════════════════════
# 1. pred npz
# ════════════════════════════════════════════════════════════════════════
d = np.load(NPZ_PATH, allow_pickle=True)
seg      = d["seg"].astype(np.int64)          # 0-based, every value is a real cell
tx_x     = d["x"].astype(np.float64)
tx_y     = d["y"].astype(np.float64)
gene_names = d["gene_names"].astype(str)
gene2i   = {g: i for i, g in enumerate(gene_names)}
gene_idx = pd.Series(d["gene"].astype(str)).map(gene2i).to_numpy()
if np.isnan(gene_idx).any():
    raise SystemExit("npz contains a gene not in gene_names; cannot build gene_idx")
gene_idx = gene_idx.astype(np.int64)
cell_loc = d["cell_loc"].astype(np.float64)
n_pred   = cell_loc.shape[0]
n_gene   = len(gene_names)
pred_xy  = cell_loc
pred_counts = np.bincount(seg, minlength=n_pred)   # pred_counts[i] = transcript count of cell i

if "roi" not in d.files:
    raise SystemExit("npz has no roi field; cannot determine eval window")
ROI_X0, ROI_X1, ROI_Y0, ROI_Y1 = [float(v) for v in d["roi"]]
print(f"ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm")
print(f"pred cells {n_pred:,}  transcripts {len(seg):,}  genes {n_gene}")

# ════════════════════════════════════════════════════════════════════════
# 2. GT cell centroids
# ════════════════════════════════════════════════════════════════════════
cp = find(DATA_DIR, "cells.parquet", "cells.csv.gz")
if cp is None:
    raise SystemExit(f"cannot find cells.parquet / cells.csv.gz under {DATA_DIR}")
gt = pd.read_parquet(cp) if str(cp).endswith("parquet") else pd.read_csv(cp)
xc = next(c for c in gt.columns if c.lower() in ("x_centroid", "center_x"))
yc = next(c for c in gt.columns if c.lower() in ("y_centroid", "center_y"))
idc = next((c for c in gt.columns if c.lower() in ("cell_id", "cell", "cellid")), None)
if idc is None:
    gt = gt.reset_index().rename(columns={"index": "cell_id"}); idc = "cell_id"
gt = gt.rename(columns={xc: "cx", yc: "cy", idc: "cell_id"})[["cell_id", "cx", "cy"]]
gt["cell_id"] = gt["cell_id"].astype(str)
gt = gt[(gt.cx >= ROI_X0) & (gt.cx < ROI_X1) &
        (gt.cy >= ROI_Y0) & (gt.cy < ROI_Y1)].reset_index(drop=True)
gt_xy = gt[["cx", "cy"]].values.astype(float)
n_gt  = len(gt)
print(f"GT cells (in ROI) {n_gt:,}   source {cp.name}")
if n_gt == 0:
    raise SystemExit("0 GT cells in ROI; check ROI / GT coordinate conventions")

# ════════════════════════════════════════════════════════════════════════
# 3. GT transcripts (cell_id assignment -> counts + gene vectors)
# ════════════════════════════════════════════════════════════════════════
tp = find(DATA_DIR, "transcripts.parquet", "transcripts.csv.gz")
if tp is None:
    raise SystemExit(f"cannot find transcripts.parquet under {DATA_DIR}")
want = ["cell_id", "feature_name", "x_location", "y_location"]
try:
    tg = pd.read_parquet(tp, columns=want + ["qv", "is_gene"])
except Exception:
    tg = pd.read_parquet(tp) if str(tp).endswith("parquet") else pd.read_csv(tp)
if "is_gene" in tg.columns:
    tg = tg[tg.is_gene == True]                               # noqa: E712
if "qv" in tg.columns and QV_MIN > 0:
    tg = tg[tg.qv >= QV_MIN]
tg = tg[(tg.x_location >= ROI_X0) & (tg.x_location < ROI_X1) &
        (tg.y_location >= ROI_Y0) & (tg.y_location < ROI_Y1)]
tg = tg.copy()
tg["cell_id"] = tg["cell_id"].astype(str)
gt_cnt = tg.groupby("cell_id").size()
gt["gt_count"] = gt.cell_id.map(gt_cnt).fillna(0).astype(int)
gt_gene_grp = (tg[tg.cell_id.isin(gt.cell_id)]
               .groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0))
print(f"GT transcripts (in ROI) {len(tg):,}")

# ════════════════════════════════════════════════════════════════════════
# 4. matching (pred cells -> nearest GT centroid, one-to-one within radius)
# ════════════════════════════════════════════════════════════════════════
tree = cKDTree(gt_xy)
dists, gi = tree.query(pred_xy, k=1, distance_upper_bound=MATCH_RADIUS)
best = {}
for pi in range(n_pred):
    if dists[pi] >= MATCH_RADIUS:
        continue
    g = int(gi[pi])
    if g not in best or dists[pi] < best[g][1]:
        best[g] = (pi, dists[pi])
pairs   = [(pi, g, dd) for g, (pi, dd) in best.items()]
n_match = len(pairs)
prec    = n_match / n_pred if n_pred else 0
recall  = n_match / n_gt   if n_gt   else 0
f1      = 2 * prec * recall / (prec + recall) if prec + recall else 0
shifts  = [p[2] for p in pairs]
pi_arr  = [p[0] for p in pairs]
gi_arr  = [p[1] for p in pairs]
print(f"matched {n_match:,}   F1={f1:.4f}  P={prec:.4f}  R={recall:.4f}")

# ════════════════════════════════════════════════════════════════════════
# 5. count metrics
# ════════════════════════════════════════════════════════════════════════
p_cnt = np.array([pred_counts[pi] for pi in pi_arr], dtype=float)
g_cnt = gt.iloc[gi_arr]["gt_count"].values.astype(float)
if len(p_cnt) >= 3 and p_cnt.std() > 0 and g_cnt.std() > 0:
    count_pearson,  _ = pearsonr(p_cnt, g_cnt)
    count_spearman, _ = spearmanr(p_cnt, g_cnt)
    count_mae  = float(np.mean(np.abs(p_cnt - g_cnt)))
    count_rmse = float(np.sqrt(np.mean((p_cnt - g_cnt) ** 2)))
else:
    count_pearson = count_spearman = count_mae = count_rmse = np.nan

# ════════════════════════════════════════════════════════════════════════
# 6. gene vectors (sample 800 pairs)
# ════════════════════════════════════════════════════════════════════════
np.random.seed(42)
samp = (np.random.choice(len(pairs), min(800, len(pairs)), replace=False)
        if pairs else np.array([], int))
cosines, jsd, prs = [], [], []
for idx in samp:
    pi, gv_id, _ = pairs[idx]
    mask = (seg == pi)
    pv = np.zeros(n_gene)
    np.add.at(pv, gene_idx[mask], 1)
    gc = gt.iloc[gv_id]["cell_id"]
    gv = np.zeros(n_gene)
    if gc in gt_gene_grp.index:
        for gname, val in gt_gene_grp.loc[gc].items():
            if gname in gene2i and val > 0:
                gv[gene2i[gname]] = val
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

# ════════════════════════════════════════════════════════════════════════
# 7. assignment metrics (nearest GT centroid per pred tx vs matched GT)
# ════════════════════════════════════════════════════════════════════════
d_tx, gi_tx = tree.query(np.stack([tx_x, tx_y], 1), k=1,
                         distance_upper_bound=MATCH_RADIUS)
gt_cell_per_tx = np.where(
    d_tx < MATCH_RADIUS,
    gt.iloc[gi_tx.clip(0, n_gt - 1)]["cell_id"].values, "-1")
ov_list, ac_list = [], []
for pi, gv_id, _ in pairs:
    pm = (seg == pi)
    if pm.sum() == 0:
        continue
    gc  = gt.iloc[gv_id]["cell_id"]
    gta = gt_cell_per_tx[pm]
    ov_list.append(float((gta == gc).mean()))
    maj = pd.Series(gta).value_counts().index[0]
    ac_list.append(float((gta == maj).mean()))
assign_overlap  = float(np.mean(ov_list)) if ov_list else np.nan
assign_accuracy = float(np.mean(ac_list)) if ac_list else np.nan

# ════════════════════════════════════════════════════════════════════════
# summary
# ════════════════════════════════════════════════════════════════════════
metrics = dict(
    gt_cells        = n_gt,
    pred_cells      = n_pred,
    cell_ratio      = round(n_pred / max(n_gt, 1), 4),
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
print(f"\n{SEP}\nResults (BOMS × {LABEL})\n{SEP}")
for k, v in metrics.items():
    print(f"  {k:22} {v}")
pd.DataFrame([metrics]).to_csv(OUT_PATH, index=False)
print(f"\n  Saved → {OUT_PATH}")