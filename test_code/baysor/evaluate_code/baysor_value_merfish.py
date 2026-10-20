#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate Baysor MERFISH segmentations with the Xenium benchmark metrics.

The cell-level metrics and sparse Hungarian centroid matching are identical to
``baysor_value_xenium.py`` (and ultimately to the UCS reference evaluator).
MERFISH's official ``cell_metadata.csv`` supplies GT centroids.  Vizgen's raw
``detected_transcripts.csv`` does not contain a cell assignment, so the script
uses ``transcripts_roi_with_prior.csv.gz`` files when available.  Those labels
are generated from official cell-boundary polygons and are reported as a
boundary-derived proxy, not as an independent transcript GT.  Run
``prepare_merfish_gt_transcripts.py --all`` to create the files for lung and
the two prostate datasets.

Examples::

    python baysor_value_merfish.py --dataset liver1
    python baysor_value_merfish.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the tested metric implementation from the Xenium evaluator.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from baysor_value_xenium import (gene_metrics, infer_radius, norm_cell,
                                 pair_metrics, sparse_hungarian)

VERSION = "2026.09.02"
METHOD = "Baysor"
USES_PRIOR = False

# Bounds are global MERFISH micrometres.  They are read from the ROI cache
# metadata used to launch Baysor and are kept explicit for reproducibility.
DATASETS = {
    "liver1": dict(root="yourpath/dataset/merfish_liver1",
                   pred="yourpath/baysor/merfish_liver1_center10000",
                   roi=(5236.737991920783, 6316.729987066785,
                        3811.0486839255263, 4891.037342034139),
                   gt_tx="yourpath/proseg/output/liver1/transcripts_roi_with_prior.csv.gz",
                   raw_tx="yourpath/dataset/merfish_liver1/_roi_cache/tx_roi_36396_49272_10000.parquet",
                   label="merfish_liver1"),
    "liver2": dict(root="yourpath/dataset/merfish_liver2",
                   pred="yourpath/baysor/merfish_liver2_center10000",
                   roi=(5771.038686925283, 6851.029458488514,
                        4970.975379467495, 6050.964816216321),
                   gt_tx="yourpath/proseg/output/liver2/transcripts_roi_with_prior.csv.gz",
                   raw_tx="yourpath/dataset/merfish_liver2/_roi_cache/tx_roi_46028_53436_10000.parquet",
                   label="merfish_liver2"),
    "lung": dict(root="yourpath/dataset/MERFISH_Lung_cancer",
                 pred="yourpath/baysor/MERFISH_Lung_cancer_center10000",
                 roi=(3782.955573105985, 4862.941450365861,
                      3346.8477292572734, 4426.839168228945),
                 gt_tx="yourpath/baysor/merfish_gt_tx/lung/transcripts_roi_with_prior.csv.gz",
                 raw_tx="yourpath/dataset/MERFISH_Lung_cancer/_roi_cache/tx_roi_31434_36213_10000.parquet", label="MERFISH_Lung_cancer"),
    "prostate1": dict(root="yourpath/dataset/merfish_prostate1",
                      pred="yourpath/baysor/merfish_prostate1_center10000",
                      roi=(5355.736768078099, 6435.73020928004,
                           5298.299104035497, 6378.293212649178),
                      gt_tx="yourpath/baysor/merfish_gt_tx/prostate1/transcripts_roi_with_prior.csv.gz",
                      raw_tx="yourpath/dataset/merfish_prostate1/_roi_cache/tx_roi_50405_50185_10000.parquet", label="merfish_prostate1"),
    "prostate2": dict(root="yourpath/dataset/merfish_prostate2",
                      pred="yourpath/baysor/merfish_prostate2_center10000",
                      roi=(5264.427126244143, 6344.421012387153,
                           4702.379653633372, 5782.368867913451),
                      gt_tx="yourpath/baysor/merfish_gt_tx/prostate2/transcripts_roi_with_prior.csv.gz",
                      raw_tx="yourpath/dataset/merfish_prostate2/_roi_cache/tx_roi_44812_50242_10000.parquet", label="merfish_prostate2"),
    "mouse_brain": dict(root="yourpath/dataset/merfish_mouse_brain",
                        pred="yourpath/baysor/merfish_mouse_brain_roi_3589_5761_2327_4499",
                        roi=(3589.0, 5761.0, 2327.0, 4499.0),
                        gt_tx="yourpath/proseg/output/merfish_1/transcripts_roi_with_prior.csv.gz",
                        raw_tx="yourpath/dataset/merfish_mouse_brain/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv",
                        label="merfish_mouse_brain"),
}

REQUESTED = [
    "method", "Pred cell count", "GT cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall", "Detection F1",
    "Mean centroid shift", "Median centroid shift", "p95 centroid shift",
    "matched_pair_n_transcripts_vs_total_counts_pearson",
    "matched_pair_n_transcripts_vs_total_counts_spearman",
    "matched_pair_n_transcripts_vs_total_counts_mae",
    "matched_pair_n_transcripts_vs_total_counts_rmse", "Pred transcript rows",
    "GT transcript rows", "Pred transcript assignment rate", "Transcript matched cell pairs",
    "matched_cell_transcript_count_pearson", "matched_cell_transcript_count_spearman",
    "matched_cell_transcript_count_mae", "matched_cell_transcript_count_rmse",
    "matched_cell_gene_vector_mean_cosine", "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson", "matched_cell_gene_vector_valid_pairs",
    "transcript_id_overlap_n", "transcript_assignment_accuracy_via_matched_cells",
    "dataset", "Overall correct fraction",
]
SELECTED = [
    "method", "dataset", "GT cell count", "Pred cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall", "Detection F1",
    "Mean centroid shift", "Median centroid shift", "p95 centroid shift",
    "matched_cell_transcript_count_pearson", "matched_cell_transcript_count_spearman",
    "matched_cell_transcript_count_mae", "matched_cell_transcript_count_rmse",
    "matched_cell_gene_vector_mean_cosine", "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson", "transcript_id_overlap_n",
    "transcript_assignment_accuracy_via_matched_cells", "Pred transcript rows",
    "GT transcript rows", "Pred transcript assignment rate", "Overall correct fraction",
    "median_area_um2", "median_diameter_um", "coverage_fraction", "uses_platform_prior",
]
CONTROL_PREFIXES = ("Blank", "BLANK", "Neg", "Negative", "SystemControl",
                    "FalseCode", "Unassigned", "Deprecated", "GenomicControl")
UNASSIGNED = {"", "0", "-1", "nan", "NaN", "None", "none", "null"}
log = logging.getLogger("eval_baysor_merfish")


def setup_log(out: Path, name: str) -> None:
    """Configure a console and per-dataset file logger."""
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(out / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)


def is_control(s: pd.Series) -> pd.Series:
    return s.astype(str).str.startswith(CONTROL_PREFIXES, na=False)


def find_file(root: Path, *patterns: str) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield canonical cell/gene/x/y columns from CSV, CSV.GZ or Parquet."""
    aliases = {"cell": ("cell", "cell_id"), "gene": ("gene", "feature_name", "target"),
               "x": ("global_x", "x", "x_um"), "y": ("global_y", "y", "y_um"),
               "is_noise": ("is_noise",), "qv": ("qv", "quality_value", "quality")}
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(path).schema.names)
        chosen = {k: next((a for a in alts if a in names), None) for k, alts in aliases.items()}
        chosen = {k: v for k, v in chosen.items() if v is not None}
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(chosen.values())):
            yield b.to_pandas().rename(columns={v: k for k, v in chosen.items()})
    else:
        header = pd.read_csv(path, nrows=0)
        chosen = {k: next((a for a in alts if a in header.columns), None) for k, alts in aliases.items()}
        chosen = {k: v for k, v in chosen.items() if v is not None}
        for b in pd.read_csv(path, usecols=list(chosen.values()), chunksize=batch_size):
            yield b.rename(columns={v: k for k, v in chosen.items()})


def load_pred_cells(path: Path, roi: Tuple[float, float, float, float]) -> pd.DataFrame:
    d = pd.read_csv(path)
    aliases = {"cell": ("cell", "cell_id"), "x": ("x", "centroid_x"),
               "y": ("y", "centroid_y"), "n_transcripts": ("n_transcripts", "transcript_count"),
               "area": ("area", "area_um2")}
    low = {str(c).lower(): c for c in d.columns}; ren = {}
    for key, names in aliases.items():
        if key not in d:
            for n in names:
                if n in d: ren[n] = key; break
                if n.lower() in low: ren[low[n.lower()]] = key; break
    d = d.rename(columns=ren)
    for c in ("x", "y", "n_transcripts", "area"):
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["cell"] = d.cell.map(norm_cell)
    x0, x1, y0, y1 = roi
    keep = d.x.ge(x0) & d.x.lt(x1) & d.y.ge(y0) & d.y.lt(y1) & d.cell.ne("")
    d = d.loc[keep].reset_index(drop=True)
    if d.cell.duplicated().any(): raise ValueError(f"duplicate predicted cells in {path}")
    return d


def load_gt_cells(path: Path, roi: Tuple[float, float, float, float]) -> pd.DataFrame:
    d = pd.read_csv(path)
    idc = d.columns[0] if str(d.columns[0]).lower().startswith("unnamed") else next((c for c in d if str(c).lower() in ("cell_id", "cell")), d.columns[0])
    xc = next((c for c in d if str(c).lower() in ("center_x", "centroid_x", "x")), None)
    yc = next((c for c in d if str(c).lower() in ("center_y", "centroid_y", "y")), None)
    if xc is None or yc is None: raise ValueError(f"GT metadata lacks center_x/center_y: {path}")
    d = d.rename(columns={idc: "_id", xc: "_x", yc: "_y"})
    d["_id"] = d._id.map(norm_cell); d["_x"] = pd.to_numeric(d._x, errors="coerce"); d["_y"] = pd.to_numeric(d._y, errors="coerce")
    x0, x1, y0, y1 = roi
    return d.loc[d._x.ge(x0) & d._x.lt(x1) & d._y.ge(y0) & d._y.lt(y1)].reset_index(drop=True)


def coord_key(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # MERFISH coordinates have four decimals; 0.01 um is tolerant of CSV
    # serialization and still separates practically all molecules.
    return np.rint(np.asarray(x, float) * 100).astype(np.int64) * 20_000_000 + np.rint(np.asarray(y, float) * 100).astype(np.int64)


def gt_tx_path(cfg: dict, override: Optional[str]) -> Optional[Path]:
    if override: return Path(override)
    # Prefer the boundary-derived table with a per-transcript cell label;
    # otherwise use the ROI cache/raw table only to report GT transcript rows.
    for key in ("gt_tx", "raw_tx"):
        p = Path(cfg[key]) if cfg.get(key) else None
        if p is not None and p.exists(): return p
    return None


def metadata_path(root: Path) -> Path:
    p = find_file(root, "*cell_metadata*.csv")
    if p is None: raise FileNotFoundError(f"cell_metadata.csv not found under {root}")
    return p


def evaluate(name: str, args: argparse.Namespace) -> Dict[str, object]:
    cfg = DATASETS[name]; root = Path(args.data_root or cfg["root"]); pred_root = Path(args.run_dir or cfg["pred"])
    roi = tuple(args.roi) if args.roi else cfg["roi"]
    out = Path(args.out_dir) / name; setup_log(out, f"eval_baysor_{name}")
    stats = Path(args.stats) if args.stats else pred_root / "segmentation_cell_stats.csv"
    seg = Path(args.segmentation) if args.segmentation else pred_root / "segmentation.csv"
    gt_meta = metadata_path(root); pred = load_pred_cells(stats, roi); gt = load_gt_cells(gt_meta, roi)
    if pred.empty or gt.empty: raise ValueError(f"empty pred/GT in ROI {roi}")
    mp, mg, md = sparse_hungarian(pred.x.to_numpy(float), pred.y.to_numpy(float),
                                  gt._x.to_numpy(float), gt._y.to_numpy(float),
                                  args.match_radius if args.match_radius is not None else
                                  infer_radius(gt._x.to_numpy(float), gt._y.to_numpy(float)))
    p_to_g = np.full(len(pred), -1, np.int64); p_to_g[mp] = mg
    pred_idx = {str(v): i for i, v in enumerate(pred.cell)}
    gt_idx = {str(v): i for i, v in enumerate(gt._id)}
    # Read Baysor output once to collect assignment counts, coordinate keys and
    # gene matrices.  The table is always ROI-local for these runs.
    pred_rows = 0; pred_keys: List[np.ndarray] = []; target_keys: List[np.ndarray] = []
    gene_names: set[str] = set()
    for f in batches(seg, args.batch_size):
        f["x"] = pd.to_numeric(f.x, errors="coerce"); f["y"] = pd.to_numeric(f.y, errors="coerce")
        c = f.cell.map(norm_cell); keep = c.ne("") & f.x.ge(roi[0]) & f.x.lt(roi[1]) & f.y.ge(roi[2]) & f.y.lt(roi[3])
        if "is_noise" in f: keep &= ~f.is_noise.fillna(False).astype(bool)
        if args.qv_min > 0 and "qv" in f:
            keep &= pd.to_numeric(f.qv, errors="coerce").ge(args.qv_min)
        if not keep.any(): continue
        c = c.loc[keep]; p = c.map(pred_idx).fillna(-1).to_numpy(np.int64); valid = p >= 0
        x, y = f.loc[keep, "x"].to_numpy(float), f.loc[keep, "y"].to_numpy(float)
        pred_rows += int(valid.sum())
        matched = valid & (p_to_g[p.clip(min=0)] >= 0)
        if matched.any():
            k = coord_key(x[matched], y[matched]); pred_keys.append(k); target_keys.append(k * (len(gt) + 1) + p_to_g[p[matched]])
        if valid.any() and "gene" in f: gene_names.update(f.loc[keep, "gene"].astype(str).to_numpy()[valid])
    coord_index = np.unique(np.concatenate(pred_keys)) if pred_keys else np.zeros(0, np.int64); coord_index.sort()
    target_index = np.unique(np.concatenate(target_keys)) if target_keys else np.zeros(0, np.int64); target_index.sort()

    tx = gt_tx_path(cfg, args.gt_transcripts)
    has_gt_labels = bool(args.gt_transcripts) or bool(cfg.get("gt_tx") and Path(cfg["gt_tx"]).exists())
    gt_rows = 0; overlap = correct = 0
    gt_label_centers: Dict[int, List[float]] = {}
    # First pass: collect GT label centroids (local prior IDs -> official cells).
    if tx is not None:
        for f in batches(tx, args.batch_size):
            f["x"] = pd.to_numeric(f.x, errors="coerce"); f["y"] = pd.to_numeric(f.y, errors="coerce")
            f = f[f.x.ge(roi[0]) & f.x.lt(roi[1]) & f.y.ge(roi[2]) & f.y.lt(roi[3])]
            if args.qv_min > 0 and "qv" in f:
                f = f.loc[pd.to_numeric(f.qv, errors="coerce").ge(args.qv_min)]
            if "gene" in f: f = f.loc[~is_control(f.gene)].reset_index(drop=True)
            if f.empty: continue
            gt_rows += len(f)
            if "gene" in f:
                gene_names.update(f.gene.astype(str).to_numpy())
            c = f.cell.map(norm_cell) if "cell" in f else pd.Series([""] * len(f), index=f.index)
            ok = c.ne("")
            for cid, sub in f.loc[ok].groupby(c.loc[ok]):
                if not cid: continue
                gt_label_centers.setdefault(int(float(cid)), []).append((float(sub.x.mean()), float(sub.y.mean()), float(len(sub))))
    # Map prior labels to official GT rows by centroid (same bridge used by the
    # existing MERFISH evaluators).  A 5-um cutoff rejects unmatched labels.
    label_to_gt: Dict[int, int] = {}
    if gt_label_centers:
        from scipy.spatial import cKDTree
        ids = np.array(list(gt_label_centers), dtype=np.int64)
        centers = np.array([[v[0][0], v[0][1]] for v in gt_label_centers.values()])
        dist, idx = cKDTree(gt[["_x", "_y"]].to_numpy(float)).query(centers, k=1)
        for label, d, j in zip(ids, dist, idx):
            if d <= args.bridge_radius: label_to_gt[int(label)] = int(j)
    # Second pass: GT counts/vectors and transcript accuracy.
    genes = sorted(gene_names); gene_idx = {g: i for i, g in enumerate(genes)}
    pred_matrix = np.zeros((len(pred), len(genes)), np.float32); gt_matrix = np.zeros((len(gt), len(genes)), np.float32)
    # Re-read prediction to fill matrix (keeps memory bounded and avoids dicts).
    for f in batches(seg, args.batch_size):
        f["x"] = pd.to_numeric(f.x, errors="coerce"); f["y"] = pd.to_numeric(f.y, errors="coerce")
        c = f.cell.map(norm_cell); keep = c.ne("") & f.x.ge(roi[0]) & f.x.lt(roi[1]) & f.y.ge(roi[2]) & f.y.lt(roi[3])
        if "is_noise" in f: keep &= ~f.is_noise.fillna(False).astype(bool)
        if args.qv_min > 0 and "qv" in f:
            keep &= pd.to_numeric(f.qv, errors="coerce").ge(args.qv_min)
        if not keep.any() or "gene" not in f: continue
        c = c.loc[keep]; p = c.map(pred_idx).fillna(-1).to_numpy(np.int64); ok = p >= 0
        g = f.loc[keep, "gene"].astype(str).to_numpy(); gi = np.array([gene_idx.get(v, -1) for v in g], np.int64); ok &= gi >= 0
        np.add.at(pred_matrix, (p[ok], gi[ok]), 1)
    if tx is not None and genes:
        base = len(gt) + 1
        for f in batches(tx, args.batch_size):
            f["x"] = pd.to_numeric(f.x, errors="coerce"); f["y"] = pd.to_numeric(f.y, errors="coerce")
            f = f[f.x.ge(roi[0]) & f.x.lt(roi[1]) & f.y.ge(roi[2]) & f.y.lt(roi[3])]
            if args.qv_min > 0 and "qv" in f:
                f = f.loc[pd.to_numeric(f.qv, errors="coerce").ge(args.qv_min)]
            if "gene" in f: f = f.loc[~is_control(f.gene)].reset_index(drop=True)
            if f.empty: continue
            if "cell" not in f:
                continue
            c = f.cell.map(norm_cell); labels = pd.to_numeric(c, errors="coerce").fillna(0).to_numpy(np.int64)
            gt_i = np.array([label_to_gt.get(int(v), -1) for v in labels], np.int64)
            keys = coord_key(f.x.to_numpy(float), f.y.to_numpy(float)); query = keys * base + np.maximum(gt_i, 0)
            if len(coord_index):
                pa = np.searchsorted(coord_index, keys); sa = np.minimum(pa, len(coord_index)-1); any_hit = (pa < len(coord_index)) & (coord_index[sa] == keys)
            else: any_hit = np.zeros(len(f), bool)
            if len(target_index):
                pt = np.searchsorted(target_index, query); st = np.minimum(pt, len(target_index)-1); exact_hit = (pt < len(target_index)) & (target_index[st] == query)
            else: exact_hit = np.zeros(len(f), bool)
            valid_gt = gt_i >= 0
            overlap += int((any_hit & valid_gt).sum()); correct += int((exact_hit & valid_gt).sum())
            if "gene" in f and valid_gt.any():
                gi = np.array([gene_idx.get(v, -1) for v in f.loc[:, "gene"].astype(str)], np.int64); ok = valid_gt & (gi >= 0)
                np.add.at(gt_matrix, (gt_i[ok], gi[ok]), 1)

    n = len(mp); prec = n / max(len(pred), 1); rec = n / max(len(gt), 1)
    row: Dict[str, object] = {"method": METHOD, "dataset": cfg["label"], "Pred cell count": len(pred), "GT cell count": len(gt), "Cell count ratio": len(pred)/max(len(gt),1), "Matched cell count": n, "Detection precision": prec, "Detection recall": rec, "Detection F1": 2*prec*rec/max(prec+rec,1e-12), "Mean centroid shift": float(md.mean()) if n else np.nan, "Median centroid shift": float(np.median(md)) if n else np.nan, "p95 centroid shift": float(np.percentile(md,95)) if n else np.nan, "Pred transcript rows": int(pred_rows), "GT transcript rows": int(gt_rows), "Pred transcript assignment rate": pred_rows/max(gt_rows,1) if gt_rows else np.nan, "Transcript matched cell pairs": n, "transcript_id_overlap_n": int(overlap) if has_gt_labels else 0, "transcript_assignment_accuracy_via_matched_cells": correct/overlap if has_gt_labels and overlap else np.nan, "Overall correct fraction": (pred_rows/max(gt_rows,1))*(correct/overlap) if has_gt_labels and gt_rows and overlap else np.nan, "median_area_um2": float(pred.area.median()) if "area" in pred else np.nan, "median_diameter_um": float(2*np.sqrt(pred.area.median()/np.pi)) if "area" in pred else np.nan, "coverage_fraction": float(pred.area.sum()/max((roi[1]-roi[0])*(roi[3]-roi[2]),1e-12)) if "area" in pred else np.nan, "uses_platform_prior": USES_PRIOR}
    if n and has_gt_labels:
        pn = pred.n_transcripts.to_numpy(float)[mp] if "n_transcripts" in pred else np.full(n, np.nan)
        gt_total = gt_matrix.sum(axis=1); gt_total[gt_matrix.sum(axis=1) <= 0] = np.nan
        row.update(pair_metrics(pn, gt_total[mg], "matched_pair_n_transcripts_vs_total_counts")); row.update(pair_metrics(pn, gt_total[mg], "matched_cell_transcript_count"))
        row.update(gene_metrics(pred_matrix[mp], gt_matrix[mg]))
    else:
        for pfx in ("matched_pair_n_transcripts_vs_total_counts", "matched_cell_transcript_count"): row.update({f"{pfx}_{k}": np.nan for k in ("pearson","spearman","mae","rmse")})
        row.update({"matched_cell_gene_vector_mean_cosine": np.nan, "matched_cell_gene_vector_mean_js_distance": np.nan, "matched_cell_gene_vector_mean_pearson": np.nan, "matched_cell_gene_vector_valid_pairs": 0})
    tag = f"baysor_{name}"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: row.get(k, np.nan) for k in REQUESTED}]).to_csv(out/f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame([{k: row.get(k, np.nan) for k in SELECTED}]).to_csv(out/f"{tag}_sparse_hungarian_qc_metrics_selected.csv", index=False)
    if n: pd.DataFrame({"pred_id": pred.cell.iloc[mp].to_numpy(), "gt_id": gt._id.iloc[mg].to_numpy(), "centroid_shift_um": md, "pred_n_transcripts": pred.n_transcripts.to_numpy(float)[mp] if "n_transcripts" in pred else np.nan, "gt_total_counts": gt_matrix.sum(axis=1)[mg], "pred_area_um2": pred.area.iloc[mp].to_numpy() if "area" in pred else np.nan}).to_csv(out/f"{tag}_matched_pairs.csv", index=False)
    log.info("%s: pred=%d gt=%d matched=%d precision=%.4f recall=%.4f F1=%.4f", name, len(pred), len(gt), n, prec, rec, row["Detection F1"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="liver1"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-dir"); ap.add_argument("--segmentation"); ap.add_argument("--stats"); ap.add_argument("--data-root"); ap.add_argument("--gt-transcripts", dest="gt_transcripts", default=None)
    ap.add_argument("--out-dir", default="yourpath/eval_results/baysor_merfish"); ap.add_argument("--match-radius", type=float); ap.add_argument("--bridge-radius", type=float, default=5.0); ap.add_argument("--batch-size", type=int, default=1_000_000); ap.add_argument("--qv-min", type=float, default=0.0); ap.add_argument("--roi", nargs=4, type=float, metavar=("X0","X1","Y0","Y1")); ap.add_argument("--version", action="store_true")
    args = ap.parse_args()
    if args.version: print(VERSION); return
    if args.all and any(getattr(args, x) is not None for x in ("run_dir","segmentation","stats","data_root","gt_transcripts","roi")): ap.error("--all cannot be combined with path/ROI overrides")
    names = list(DATASETS) if args.all else [args.dataset]; rows = [evaluate(n, args) for n in names]
    if len(rows) > 1:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{k:r.get(k,np.nan) for k in REQUESTED} for r in rows]).to_csv(out/"baysor_merfish_all_metrics_full.csv", index=False)
        pd.DataFrame([{k:r.get(k,np.nan) for k in SELECTED} for r in rows]).to_csv(out/"baysor_merfish_all_metrics_selected.csv", index=False)


if __name__ == "__main__": main()
