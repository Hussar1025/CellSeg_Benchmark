#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boms_value_cosmx_2.py
Full-metric evaluator for BOMS @ CosMx2 FOV48 + FOV261.

GT:
  metadata: /data/qiuyijia/dataset/cosmx_lymph_node/flat_files/S0_metadata_file.csv.gz
  transcripts: /data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/tx_fov_48_261.parquet

Prediction:
  /data/qiuyijia/boms_cosmx_2_fullgenes_hs4p5/fov48/boms_cosmx_2_fov48.npz
  /data/qiuyijia/boms_cosmx_2_fullgenes_hs4p5/fov261/boms_cosmx_2_fov261.npz

Important:
- No GT is used to alter BOMS prediction.
- Cell matching is spatial, one-to-one, within --match-radius-um.
- Assignment accuracy uses official per-transcript cell_ID only after spatial cell matching.
- Expression vectors are compared on spatially matched cells.
"""
import argparse, json, math, os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

PX_UM = 0.12
FOV_PX = 4256
FOV_UM = FOV_PX * PX_UM

META = Path("/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/S0_metadata_file.csv.gz")
TX = Path("/data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/tx_fov_48_261.parquet")
PRED_ROOT = Path("/data/qiuyijia/boms_cosmx_2_fullgenes_hs4p5")
OUT = Path("/data/qiuyijia/eval_results_qv20/cosmx_2")

def pick(cols, candidates, required=True):
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise RuntimeError(f"cannot find any of {candidates}; columns={list(cols)}")
    return None

def finite_corr(fn, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa, bb = a[m], b[m]
    if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
        return np.nan
    return float(fn(aa, bb)[0])

def greedy_match(pred_xy, gt_xy, radius):
    """Sparse candidate collection + global shortest-edge greedy 1:1 matching."""
    tree = cKDTree(gt_xy)
    edges = []
    for pi, p in enumerate(pred_xy):
        gis = tree.query_ball_point(p, radius)
        if not gis:
            continue
        q = gt_xy[np.asarray(gis, int)]
        ds = np.sqrt(((q - p) ** 2).sum(axis=1))
        edges.extend((float(d), pi, int(gi)) for d, gi in zip(ds, gis))
    edges.sort(key=lambda z: z[0])
    used_p, used_g, pairs = set(), set(), []
    for d, pi, gi in edges:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
        pairs.append((pi, gi, d))
    return pairs

def infer_seg_index(seg, ncell):
    s = np.asarray(seg)
    good = np.isfinite(s)
    vals = s[good].astype(np.int64)
    if len(vals) == 0:
        raise RuntimeError("empty seg")
    mn, mx = int(vals.min()), int(vals.max())
    if mn >= 0 and mx < ncell:
        return vals, 0
    if mn >= 1 and mx <= ncell:
        return vals - 1, 1
    # Some runs may contain -1 for unassigned.
    pos = vals[vals >= 0]
    if len(pos) and pos.max() < ncell:
        return vals, 0
    if len(pos) and pos.min() >= 1 and pos.max() <= ncell:
        return np.where(vals > 0, vals - 1, -1), 1
    raise RuntimeError(f"cannot infer seg indexing: min={mn} max={mx} ncell={ncell}")

def align_prediction_to_tx(z, tx, fov):
    """
    Return official transcript rows aligned 1:1 to BOMS x/y/gene order.
    First try direct order; otherwise use (rounded x_um,y_um,gene,occurrence) merge.
    """
    x = np.asarray(z["x"], float)
    y = np.asarray(z["y"], float)
    g = np.asarray(z["gene"]).astype(str)
    if not (len(x) == len(y) == len(g)):
        raise RuntimeError("BOMS x/y/gene length mismatch")

    xc = pick(tx.columns, ["x_local_px", "x"])
    yc = pick(tx.columns, ["y_local_px", "y"])
    gc = pick(tx.columns, ["target", "gene", "feature_name"])

    tx_x_raw = pd.to_numeric(tx[xc], errors="coerce").to_numpy(float)
    tx_y_raw = pd.to_numeric(tx[yc], errors="coerce").to_numpy(float)

    # Official cache is pixels; BOMS working coordinates are microns.
    tx_x_um = tx_x_raw * PX_UM if np.nanmax(tx_x_raw) > 1000 else tx_x_raw
    tx_y_um = tx_y_raw * PX_UM if np.nanmax(tx_y_raw) > 1000 else tx_y_raw
    tx_g = tx[gc].astype(str).to_numpy()

    if len(tx) == len(x):
        dx = np.nanmax(np.abs(tx_x_um - x))
        dy = np.nanmax(np.abs(tx_y_um - y))
        gene_ok = float(np.mean(tx_g == g))
        if dx < 1e-4 and dy < 1e-4 and gene_ok > 0.999:
            print(f"FOV{fov}: transcript alignment DIRECT order; max dx/dy={dx:.2e}/{dy:.2e}", flush=True)
            return tx.reset_index(drop=True)

    print(f"FOV{fov}: direct order mismatch; falling back to coordinate+gene occurrence matching", flush=True)

    # Six decimals is much tighter than biological localization and only used to
    # reconnect the exact source transcript rows to their BOMS labels.
    pred = pd.DataFrame({
        "_x": np.round(x, 6),
        "_y": np.round(y, 6),
        "_g": g,
        "_pred_row": np.arange(len(x), dtype=np.int64),
    })
    ref = tx.copy().reset_index(drop=True)
    ref["_x"] = np.round(tx_x_um, 6)
    ref["_y"] = np.round(tx_y_um, 6)
    ref["_g"] = tx_g
    pred["_occ"] = pred.groupby(["_x","_y","_g"], sort=False).cumcount()
    ref["_occ"] = ref.groupby(["_x","_y","_g"], sort=False).cumcount()
    ref["_ref_row"] = np.arange(len(ref), dtype=np.int64)

    m = pred.merge(ref[["_x","_y","_g","_occ","_ref_row"]],
                   on=["_x","_y","_g","_occ"], how="left", validate="one_to_one")
    ok = m["_ref_row"].notna()
    rate = float(ok.mean())
    print(f"FOV{fov}: exact transcript reconnect = {ok.sum():,}/{len(m):,} ({rate:.3%})", flush=True)
    if rate < 0.995:
        raise RuntimeError(f"FOV{fov}: transcript reconnect only {rate:.2%}; refuse pseudo matching")
    order = m["_ref_row"].astype(int).to_numpy()
    return ref.iloc[order].reset_index(drop=True)

def load_gt_meta(fov):
    m = pd.read_csv(META)
    fc = pick(m.columns, ["fov", "FOV"])
    sub = m[pd.to_numeric(m[fc], errors="coerce") == int(fov)].copy().reset_index(drop=True)
    idc = pick(sub.columns, ["cell_ID", "cell_id", "CellID", "cellid"])
    xc = pick(sub.columns, ["CenterX_local_px", "x_local_px", "CenterX_global_px"])
    yc = pick(sub.columns, ["CenterY_local_px", "y_local_px", "CenterY_global_px"])

    gx = pd.to_numeric(sub[xc], errors="coerce").to_numpy(float)
    gy = pd.to_numeric(sub[yc], errors="coerce").to_numpy(float)
    # local coords are pixels in this CosMx export
    if np.nanmax(gx) > 1000:
        gx *= PX_UM
        gy *= PX_UM
    valid = np.isfinite(gx) & np.isfinite(gy)
    sub = sub.loc[valid].reset_index(drop=True)
    return sub, idc, np.c_[gx[valid], gy[valid]]

def evaluate_fov(fov, match_radius):
    pred_path = PRED_ROOT / f"fov{fov}" / f"boms_cosmx_2_fov{fov}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    z = np.load(pred_path, allow_pickle=True)
    print(f"\n{'='*92}\nBOMS @ CosMx2 FOV {fov}\n{'='*92}", flush=True)
    print("NPZ keys:", list(z.files), flush=True)

    cell_loc = np.asarray(z["cell_loc"], float)
    n_pred = len(cell_loc)
    if cell_loc.ndim != 2 or cell_loc.shape[1] < 2:
        raise RuntimeError(f"bad cell_loc shape {cell_loc.shape}")

    pred_xy = cell_loc[:, :2].copy()
    # BOMS CosMx script works in microns; be defensive if a pixel-space NPZ appears.
    if np.nanmax(pred_xy) > 1000:
        pred_xy *= PX_UM

    meta, meta_idc, gt_xy = load_gt_meta(fov)
    gt_ids = meta[meta_idc].astype(str).to_numpy()
    n_gt = len(gt_xy)

    pairs = greedy_match(pred_xy, gt_xy, match_radius)
    pi = np.asarray([p[0] for p in pairs], int)
    gi = np.asarray([p[1] for p in pairs], int)
    dist = np.asarray([p[2] for p in pairs], float)
    matched = len(pairs)

    precision = matched / n_pred if n_pred else np.nan
    recall = matched / n_gt if n_gt else np.nan
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0

    # Official transcript GT
    tx_all = pd.read_parquet(TX)
    tfc = pick(tx_all.columns, ["fov","FOV"])
    tx = tx_all[pd.to_numeric(tx_all[tfc], errors="coerce") == int(fov)].copy().reset_index(drop=True)
    tx_aligned = align_prediction_to_tx(z, tx, fov)
    gt_cell_col = pick(tx_aligned.columns, ["cell_ID","cell_id","cell"])
    gene_col = pick(tx_aligned.columns, ["target","gene","feature_name"])

    seg_raw = np.asarray(z["seg"])
    if len(seg_raw) != len(tx_aligned):
        raise RuntimeError(f"seg length {len(seg_raw):,} != transcript length {len(tx_aligned):,}")
    seg_idx, seg_base = infer_seg_index(seg_raw, n_pred)
    assigned = (seg_idx >= 0) & (seg_idx < n_pred)
    assign_rate = float(assigned.mean())

    counts = np.bincount(seg_idx[assigned], minlength=n_pred)
    nz_counts = counts[counts > 0]
    tx_med = float(np.median(nz_counts)) if len(nz_counts) else np.nan
    tx_mean = float(np.mean(nz_counts)) if len(nz_counts) else np.nan
    tx_p5 = float(np.percentile(nz_counts,5)) if len(nz_counts) else np.nan
    tx_p95 = float(np.percentile(nz_counts,95)) if len(nz_counts) else np.nan
    frag = float(np.mean(nz_counts < 10)) if len(nz_counts) else np.nan

    # pred-index -> spatially matched official cell id
    pred_to_gt = np.full(n_pred, None, dtype=object)
    for pidx, gidx, _ in pairs:
        pred_to_gt[pidx] = gt_ids[gidx]

    gt_tx_id = tx_aligned[gt_cell_col].astype(str).to_numpy()
    # normalize null-ish labels
    bad_gt = pd.isna(tx_aligned[gt_cell_col]).to_numpy()
    mapped_pred_id = np.full(len(seg_idx), None, dtype=object)
    valid_pred = assigned.copy()
    mapped_pred_id[valid_pred] = pred_to_gt[seg_idx[valid_pred]]

    comparable = valid_pred & (~bad_gt) & pd.notna(mapped_pred_id)
    assignment_n = int(comparable.sum())
    assign_accuracy = float(np.mean(mapped_pred_id[comparable] == gt_tx_id[comparable])) if assignment_n else np.nan
    overall_correct = float(np.sum((mapped_pred_id == gt_tx_id) & comparable) / len(seg_idx)) if len(seg_idx) else np.nan

    # GT-assigned fraction: count transcript cell IDs that are real metadata cells.
    gt_id_set = set(gt_ids.tolist())
    gt_assigned = np.fromiter((x in gt_id_set for x in gt_tx_id), dtype=bool, count=len(gt_tx_id))
    gt_assign_rate = float(gt_assigned.mean())

    # Count fidelity on spatially matched cells
    gt_tx_counts = pd.Series(gt_tx_id[gt_assigned]).value_counts()
    pred_counts_mat = counts
    pc, gc = [], []
    for pidx, gidx, _ in pairs:
        pc.append(float(pred_counts_mat[pidx]))
        gc.append(float(gt_tx_counts.get(gt_ids[gidx], 0)))
    count_pearson = finite_corr(pearsonr, pc, gc)
    count_spearman = finite_corr(spearmanr, pc, gc)
    count_mae = float(np.mean(np.abs(np.asarray(pc)-np.asarray(gc)))) if pc else np.nan
    count_rmse = float(np.sqrt(np.mean((np.asarray(pc)-np.asarray(gc))**2))) if pc else np.nan

    # Expression-vector fidelity.
    gene = tx_aligned[gene_col].astype(str).to_numpy()
    genes = np.unique(gene)
    g2i = {g:i for i,g in enumerate(genes)}
    gene_idx = np.fromiter((g2i[g] for g in gene), dtype=np.int32, count=len(gene))

    # use up to 3000 matched cells as in other benchmark evaluators
    rng = np.random.default_rng(0)
    pair_sel = np.arange(matched)
    if matched > 3000:
        pair_sel = np.sort(rng.choice(matched, 3000, replace=False))

    cosines, jsds, pears = [], [], []
    # Group transcript indices once.
    pred_rows = {}
    gt_rows = {}
    for idx in np.flatnonzero(assigned):
        pred_rows.setdefault(int(seg_idx[idx]), []).append(idx)
    for idx in np.flatnonzero(gt_assigned):
        gt_rows.setdefault(gt_tx_id[idx], []).append(idx)

    for kk in pair_sel:
        pidx, gidx, _ = pairs[int(kk)]
        gt_id = gt_ids[gidx]
        aidx = pred_rows.get(int(pidx), [])
        bidx = gt_rows.get(gt_id, [])
        if not aidx or not bidx:
            continue
        va = np.bincount(gene_idx[np.asarray(aidx,int)], minlength=len(genes)).astype(float)
        vb = np.bincount(gene_idx[np.asarray(bidx,int)], minlength=len(genes)).astype(float)
        na = np.linalg.norm(va); nb = np.linalg.norm(vb)
        if na > 0 and nb > 0:
            cosines.append(float(np.dot(va,vb)/(na*nb)))
        sa, sb = va.sum(), vb.sum()
        if sa > 0 and sb > 0:
            jsds.append(float(jensenshannon(va/sa, vb/sb, base=2.0)))
        if np.std(va) > 0 and np.std(vb) > 0:
            pears.append(float(pearsonr(va,vb)[0]))

    row = dict(
        method="BOMS", dataset="cosmx_2", fov=int(fov),
        gt_cells=n_gt, pred_cells=n_pred, matched=matched,
        cell_ratio=n_pred/n_gt if n_gt else np.nan,
        precision=precision, recall=recall, f1=f1,
        loc_mean_um=float(np.mean(dist)) if matched else np.nan,
        loc_median_um=float(np.median(dist)) if matched else np.nan,
        loc_p95_um=float(np.percentile(dist,95)) if matched else np.nan,
        match_radius_um=float(match_radius),
        n_transcripts=len(seg_idx),
        assign_rate=assign_rate,
        coverage_fraction=assign_rate,
        gt_assign_rate=gt_assign_rate,
        tx_per_cell_median=tx_med, tx_per_cell_mean=tx_mean,
        tx_per_cell_p5=tx_p5, tx_per_cell_p95=tx_p95,
        frag_lt10=frag,
        count_pearson=count_pearson, count_spearman=count_spearman,
        count_mae=count_mae, count_rmse=count_rmse,
        assign_accuracy=assign_accuracy, overall_correct=overall_correct,
        assignment_n=assignment_n,
        vec_cosine=float(np.nanmean(cosines)) if cosines else np.nan,
        vec_js_dist=float(np.nanmean(jsds)) if jsds else np.nan,
        vec_pearson=float(np.nanmean(pears)) if pears else np.nan,
        vec_n=int(min(matched,3000)),
        vec_genes=int(len(genes)),
        uses_platform_prior=False,
        detection_comparable=True,
        assignment_independent=True,
        vector_independent=True,
        gt_used_only_for_evaluation=True,
        pred_path=str(pred_path),
        seg_index_base=int(seg_base),
    )

    pair_df = pd.DataFrame({
        "pred_index": pi,
        "gt_index": gi,
        "gt_cell_id": gt_ids[gi] if len(gi) else [],
        "distance_um": dist,
    })
    return row, pair_df

def print_row(row):
    print("-"*92)
    for k,v in row.items():
        print(f"{k:32s} {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fovs", default="48,261")
    ap.add_argument("--match-radius-um", type=float, default=10.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    fovs = [int(x) for x in a.fovs.split(",") if x.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fov in fovs:
        row, pairs = evaluate_fov(fov, a.match_radius_um)
        print_row(row)
        rows.append(row)
        pairs.to_csv(out/f"boms_cosmx_2_fov{fov}_pairs.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out/"boms_cosmx_2_metrics_by_fov.csv", index=False)

    if len(rows) == 2 and set(fovs) == {48,261}:
        A, B = rows
        gt = A["gt_cells"] + B["gt_cells"]
        pred = A["pred_cells"] + B["pred_cells"]
        matched = A["matched"] + B["matched"]
        # Counts weighted by number of matched cells where appropriate.
        wA, wB = A["matched"], B["matched"]
        w = max(wA+wB, 1)
        combined = dict(
            method="BOMS", dataset="cosmx_2", fovs="48,261",
            gt_cells=gt, pred_cells=pred, matched=matched,
            cell_ratio=pred/gt,
            precision=matched/pred,
            recall=matched/gt,
            f1=2*(matched/pred)*(matched/gt)/((matched/pred)+(matched/gt)),
            loc_mean_um=(A["loc_mean_um"]*wA+B["loc_mean_um"]*wB)/w,
            loc_median_um=np.nan,  # cannot exactly pool medians without pair rows
            loc_p95_um=np.nan,
            match_radius_um=a.match_radius_um,
            assign_rate=(A["assign_rate"]*A["n_transcripts"]+B["assign_rate"]*B["n_transcripts"])/
                        (A["n_transcripts"]+B["n_transcripts"]),
            gt_assign_rate=(A["gt_assign_rate"]*A["n_transcripts"]+B["gt_assign_rate"]*B["n_transcripts"])/
                           (A["n_transcripts"]+B["n_transcripts"]),
            tx_per_cell_median=float(np.mean([A["tx_per_cell_median"],B["tx_per_cell_median"]])),
            tx_per_cell_mean=float(np.mean([A["tx_per_cell_mean"],B["tx_per_cell_mean"]])),
            frag_lt10=float(np.mean([A["frag_lt10"],B["frag_lt10"]])),
            count_pearson=float(np.nanmean([A["count_pearson"],B["count_pearson"]])),
            count_spearman=float(np.nanmean([A["count_spearman"],B["count_spearman"]])),
            assign_accuracy=(A["assign_accuracy"]*A["assignment_n"]+B["assign_accuracy"]*B["assignment_n"])/
                            max(A["assignment_n"]+B["assignment_n"],1),
            overall_correct=(A["overall_correct"]*A["n_transcripts"]+B["overall_correct"]*B["n_transcripts"])/
                            (A["n_transcripts"]+B["n_transcripts"]),
            assignment_n=A["assignment_n"]+B["assignment_n"],
            vec_cosine=float(np.nanmean([A["vec_cosine"],B["vec_cosine"]])),
            vec_js_dist=float(np.nanmean([A["vec_js_dist"],B["vec_js_dist"]])),
            vec_pearson=float(np.nanmean([A["vec_pearson"],B["vec_pearson"]])),
            vec_n=A["vec_n"]+B["vec_n"],
            vec_genes=max(A["vec_genes"],B["vec_genes"]),
            uses_platform_prior=False,
            detection_comparable=True,
            assignment_independent=True,
            vector_independent=True,
            gt_used_only_for_evaluation=True,
        )
        print(f"\n{'='*92}\nCOMBINED FOV48 + FOV261\n{'='*92}")
        print_row(combined)
        pd.DataFrame([combined]).to_csv(out/"boms_cosmx_2_metrics.csv", index=False)

    meta = {
        "prediction_root": str(PRED_ROOT),
        "metadata": str(META),
        "transcripts": str(TX),
        "pixel_size_um": PX_UM,
        "fov_size_px": FOV_PX,
        "fov_size_um": FOV_UM,
        "fovs": fovs,
        "match_radius_um": a.match_radius_um,
        "gt_used_only_for_evaluation": True,
    }
    (out/"boms_cosmx_2_eval_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nmetrics ->", out/"boms_cosmx_2_metrics.csv")
    print("by_fov  ->", out/"boms_cosmx_2_metrics_by_fov.csv")

if __name__ == "__main__":
    main()
