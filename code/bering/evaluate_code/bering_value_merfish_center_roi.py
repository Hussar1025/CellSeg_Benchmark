#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate 13 Bering MERFISH centre-ROI segmentations with 19 metrics.

Official Vizgen ``cell_metadata.csv`` centroids are the cell GT.  As in the
provided ProSeg MERFISH evaluator, raw MERFISH transcripts have no official
cell assignment, so GT counts/gene vectors are an explicitly labelled nearest-
official-centroid baseline (15 um maximum distance).

The evaluated Bering result is the final ensemble: cells come from
``cells_ensembled_metadata.csv`` and transcript assignments come from the
``ensembled_cells`` column of ``segmentation_transcripts.csv``.  All geometry
is filtered in the exact centre ROI in mosaic-pixel coordinates.  Mouse brain
uses 20000 x 20000 px; the other 12 datasets use 10000 x 10000 px.

Examples::

    python bering_value_merfish_center_roi.py --all --check-only
    python bering_value_merfish_center_roi.py --all
    python bering_value_merfish_center_roi.py --dataset mouse_brain

The default output is one CSV with one row per evaluated dataset:
``yourpath/eval/bering/bering_merfish_evaluation.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


DATA_ROOT = Path("yourpath/dataset")
RESULT_ROOT = Path("yourpath/bering/merfish_new_output")
DEFAULT_OUT = Path("yourpath/eval/bering/bering_merfish_evaluation.csv")
MATCH_RADIUS_UM = 10.0
GT_ASSIGN_RADIUS_UM = 15.0
SEP = "=" * 78

DATASETS: Mapping[str, Mapping[str, str]] = {
    "melanoma": {"data": "merfish_Melanoma", "result": "merfish_Melanoma_center10000_output", "label": "Melanoma"},
    "ovarian1": {"data": "merfish_ovarian1", "result": "merfish_ovarian1_center10000_output", "label": "ovarian1"},
    "ovarian2": {"data": "merfish_ovarian2", "result": "merfish_ovarian2_center10000_output", "label": "ovarian2"},
    "colon1": {"data": "merfish_colon1", "result": "merfish_colon1_center10000_output", "label": "colon1"},
    "colon2": {"data": "merfish_colon2", "result": "merfish_colon2_center10000_output", "label": "colon2"},
    # The source directory is misspelled Uterian; report the requested Uterine name.
    "uterine": {"data": "merfish_Uterian", "result": "merfish_Uterian_center10000_output", "label": "Uterine"},
    "breast_cancer": {"data": "merfish_breast_cancer", "result": "merfish_breast_cancer_center10000_output", "label": "breast_cancer"},
    "liver1": {"data": "merfish_liver1", "result": "merfish_liver1_center10000_output", "label": "liver1"},
    "liver2": {"data": "merfish_liver2", "result": "merfish_liver2_center10000_output", "label": "liver2"},
    "lung_cancer": {"data": "MERFISH_Lung_cancer", "result": "MERFISH_Lung_cancer_center10000_output", "label": "Lung_cancer"},
    "mouse_brain": {"data": "merfish_mouse_brain", "result": "merfish_mouse_brain_center20000_output", "label": "mouse_brain"},
    "prostate1": {"data": "merfish_prostate1", "result": "merfish_prostate1_center10000_output", "label": "prostate1"},
    "prostate2": {"data": "merfish_prostate2", "result": "merfish_prostate2_center10000_output", "label": "prostate2"},
}

METRIC_COLUMNS = [
    "gt_cells", "pred_cells", "cell_ratio", "matched", "precision", "recall", "f1",
    "loc_mean_um", "loc_median_um", "loc_p95_um", "count_pearson",
    "count_spearman", "count_mae", "count_rmse", "vec_cosine", "vec_js_dist",
    "vec_pearson", "assign_overlap", "assign_accuracy",
]
VectorCounts = MutableMapping[Tuple[int, str], int]


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"cannot find {pattern!r} under {root}")
    return hits[0]


def pick_column(columns: Sequence[str], choices: Sequence[str], description: str) -> str:
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    for name in choices:
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"cannot find {description} column; columns={list(columns)}")


def normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return "" if int(value) <= 0 else str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return "" if int(value) <= 0 else str(int(value))
    text = str(value).strip()
    return "" if text in {"", "0", "-1", "nan", "None", "background"} else text


class Geometry:
    def __init__(self, transform: Path, x0: int, y0: int, size: int):
        self.transform_path = transform
        self.matrix = np.loadtxt(transform, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"expected 3x3 transform, got {self.matrix.shape}: {transform}")
        self.x0, self.y0, self.size = int(x0), int(y0), int(size)

    def to_pixel(self, x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        px = self.matrix[0, 0] * x + self.matrix[0, 1] * y + self.matrix[0, 2]
        py = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2]
        return px, py

    def inside(self, x: Sequence[float], y: Sequence[float]) -> np.ndarray:
        px, py = self.to_pixel(x, y)
        return (
            np.isfinite(px) & np.isfinite(py)
            & (px >= self.x0) & (px < self.x0 + self.size)
            & (py >= self.y0) & (py < self.y0 + self.size)
        )

    def micron_bounds(self) -> Tuple[float, float, float, float]:
        inv = np.linalg.inv(self.matrix)
        corners = np.array([
            [self.x0, self.y0, 1.0], [self.x0 + self.size, self.y0, 1.0],
            [self.x0, self.y0 + self.size, 1.0],
            [self.x0 + self.size, self.y0 + self.size, 1.0],
        ])
        um = corners @ inv.T
        return float(um[:, 0].min()), float(um[:, 0].max()), float(um[:, 1].min()), float(um[:, 1].max())


def dapi_shape(path: Path) -> Tuple[int, int]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        shape = tuple(tif.series[0].shape)
    if len(shape) < 2:
        raise ValueError(f"unsupported DAPI shape {shape}: {path}")
    return int(shape[-2]), int(shape[-1])


def runtime_paths(cfg: Mapping[str, str]) -> Tuple[Path, Path, Path, Path, Path, Geometry]:
    data = DATA_ROOT / cfg["data"]
    result = RESULT_ROOT / cfg["result"]
    resource = result / "bering_resource_summary.json"
    if not resource.is_file():
        raise FileNotFoundError(resource)
    info = json.loads(resource.read_text())
    size = int(info["roi_pixels"])
    cache_value = info.get("roi_cache")
    match = re.search(r"tx_roi_(\d+)_(\d+)_(\d+).*\.parquet$", str(cache_value or ""))
    if match:
        y0, x0, cache_size = map(int, match.groups())
        if cache_size != size:
            raise ValueError(f"ROI size mismatch in {resource}: {size} versus {cache_size}")
    else:
        # Mouse brain had no dataset-level cache. Bering used this exact rule.
        dapi = find_one(data, "*DAPI*.tif*")
        height, width = dapi_shape(dapi)
        x0 = width // 2 - size // 2
        y0 = height // 2 - size // 2
    transform = find_one(data, "*micron_to_mosaic_pixel_transform.csv")
    metadata = find_one(data, "*cell_metadata*.csv")
    input_tx = find_one(result, "bering_input_transcripts_*.parquet")
    cells = result / "cells_ensembled_metadata.csv"
    assignments = result / "segmentation_transcripts.csv"
    return metadata, input_tx, cells, assignments, resource, Geometry(transform, x0, y0, size)


def load_gt_cells(path: Path, geo: Geometry) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    idcol = str(header.columns[0])
    xcol = pick_column(header.columns, ("center_x", "centroid_x", "x"), "GT x")
    ycol = pick_column(header.columns, ("center_y", "centroid_y", "y"), "GT y")
    frame = pd.read_csv(path, usecols=[idcol, xcol, ycol]).rename(
        columns={idcol: "gt_id", xcol: "x", ycol: "y"})
    frame["x"] = pd.to_numeric(frame.x, errors="coerce")
    frame["y"] = pd.to_numeric(frame.y, errors="coerce")
    return frame.loc[geo.inside(frame.x, frame.y)].reset_index(drop=True)


def load_pred_cells(path: Path, geo: Geometry) -> pd.DataFrame:
    frame = pd.read_csv(path)
    ccol = pick_column(frame.columns, ("cell_id", "cell"), "Bering cell")
    xcol = pick_column(frame.columns, ("cx", "centroid_x", "x"), "Bering x")
    ycol = pick_column(frame.columns, ("cy", "centroid_y", "y"), "Bering y")
    ncol = pick_column(frame.columns, ("n_counts", "n_transcripts", "transcript_count"), "Bering count")
    frame = frame.rename(columns={ccol: "cell", xcol: "x", ycol: "y", ncol: "n_transcripts"})
    frame["cell"] = frame.cell.map(normalize_cell)
    for col in ("x", "y", "n_transcripts"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.loc[geo.inside(frame.x, frame.y) & frame.cell.ne("")].reset_index(drop=True)
    if frame.cell.duplicated().any():
        raise ValueError(f"duplicate Bering cell IDs in {path}")
    return frame


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same matching rule as the supplied ProSeg MERFISH reference."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    distances, gt_indices = cKDTree(gt_xy).query(pred_xy, k=1, distance_upper_bound=radius)
    best: Dict[int, Tuple[int, float]] = {}
    for pred_i, (distance, gt_i) in enumerate(zip(distances, gt_indices)):
        if not np.isfinite(distance) or distance >= radius:
            continue
        gt_i = int(gt_i)
        if gt_i not in best or distance < best[gt_i][1]:
            best[gt_i] = pred_i, float(distance)
    pairs = sorted((pred_i, gt_i, distance) for gt_i, (pred_i, distance) in best.items())
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    pred_i, gt_i, distance = zip(*pairs)
    return np.asarray(pred_i), np.asarray(gt_i), np.asarray(distance)


def parquet_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    xcol = pick_column(pf.schema.names, ("global_x", "x", "x_um"), "input transcript x")
    ycol = pick_column(pf.schema.names, ("global_y", "y", "y_um"), "input transcript y")
    gcol = pick_column(pf.schema.names, ("gene", "features", "target"), "input transcript gene")
    for batch in pf.iter_batches(batch_size=batch_size, columns=[xcol, ycol, gcol]):
        yield batch.to_pandas().rename(columns={xcol: "x", ycol: "y", gcol: "gene"})


def assignment_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0)
    ccol = pick_column(header.columns, ("ensembled_cells",), "final Bering cell")
    xcol = pick_column(header.columns, ("x", "global_x"), "Bering transcript x")
    ycol = pick_column(header.columns, ("y", "global_y"), "Bering transcript y")
    gcol = pick_column(header.columns, ("features", "gene", "target"), "Bering transcript gene")
    for frame in pd.read_csv(path, usecols=[ccol, xcol, ycol, gcol], chunksize=batch_size):
        yield frame.rename(columns={ccol: "cell", xcol: "x", ycol: "y", gcol: "gene"})


def add_grouped_counts(store: VectorCounts, slots: np.ndarray, genes: np.ndarray) -> None:
    valid = slots >= 0
    if not valid.any():
        return
    grouped = pd.DataFrame({"slot": slots[valid], "gene": genes[valid]}).groupby(
        ["slot", "gene"], sort=False).size()
    for (slot, gene), count in grouped.items():
        store[(int(slot), str(gene))] += int(count)


def load_gt_counts_and_vectors(
    path: Path,
    geo: Geometry,
    gt_xy: np.ndarray,
    sampled_gt: np.ndarray,
    batch_size: int,
    radius: float,
) -> Tuple[np.ndarray, VectorCounts, int, int]:
    tree = cKDTree(gt_xy)
    totals = np.zeros(len(gt_xy), dtype=np.int64)
    sample_slot = np.full(len(gt_xy), -1, dtype=np.int32)
    sample_slot[sampled_gt] = np.arange(len(sampled_gt), dtype=np.int32)
    vectors: VectorCounts = defaultdict(int)
    rows = assigned = 0
    for frame in parquet_batches(path, batch_size):
        frame["x"] = pd.to_numeric(frame.x, errors="coerce")
        frame["y"] = pd.to_numeric(frame.y, errors="coerce")
        keep = geo.inside(frame.x, frame.y)
        if not keep.any():
            continue
        xy = frame.loc[keep, ["x", "y"]].to_numpy(float)
        genes = frame.loc[keep, "gene"].astype(str).to_numpy()
        rows += len(xy)
        distance, gt_i = tree.query(xy, k=1, distance_upper_bound=radius)
        valid = np.isfinite(distance) & (distance < radius) & (gt_i < len(gt_xy))
        assigned += int(valid.sum())
        if valid.any():
            good_gt = gt_i[valid].astype(np.int64)
            totals += np.bincount(good_gt, minlength=len(gt_xy))
            add_grouped_counts(vectors, sample_slot[good_gt], genes[valid])
    return totals, vectors, rows, assigned


def load_pred_vectors_and_assignment_metrics(
    path: Path,
    geo: Geometry,
    all_pred_cells: Sequence[str],
    sampled_pred_cells: Sequence[str],
    matched_pred_cells: Sequence[str],
    matched_gt: np.ndarray,
    gt_xy: np.ndarray,
    radius: float,
    batch_size: int,
) -> Tuple[VectorCounts, int, set[str], float, float]:
    known_cells = set(all_pred_cells)
    vector_lookup = {cell: i for i, cell in enumerate(sampled_pred_cells)}
    pair_lookup = {cell: i for i, cell in enumerate(matched_pred_cells)}
    tree = cKDTree(gt_xy)
    pair_totals = np.zeros(len(matched_pred_cells), dtype=np.int64)
    pair_correct = np.zeros(len(matched_pred_cells), dtype=np.int64)
    composition: VectorCounts = defaultdict(int)
    vectors: VectorCounts = defaultdict(int)
    genes_seen: set[str] = set()
    assigned_rows = 0

    for frame in assignment_batches(path, batch_size):
        frame["x"] = pd.to_numeric(frame.x, errors="coerce")
        frame["y"] = pd.to_numeric(frame.y, errors="coerce")
        cells = frame.cell.map(normalize_cell)
        keep = geo.inside(frame.x, frame.y) & cells.isin(known_cells)
        assigned_rows += int(keep.sum())
        if not keep.any():
            continue
        kept_cells = cells.loc[keep]
        genes = frame.loc[keep, "gene"].astype(str).to_numpy()
        genes_seen.update(genes)
        vector_slots = kept_cells.map(vector_lookup).fillna(-1).to_numpy(np.int32)
        add_grouped_counts(vectors, vector_slots, genes)

        pair_slots = kept_cells.map(pair_lookup).fillna(-1).to_numpy(np.int32)
        comparable = pair_slots >= 0
        if not comparable.any():
            continue
        xy = frame.loc[keep, ["x", "y"]].to_numpy(float)[comparable]
        distance, nearest_gt = tree.query(xy, k=1, distance_upper_bound=radius)
        nearest_gt = nearest_gt.astype(np.int64)
        nearest_gt[~np.isfinite(distance) | (distance >= radius)] = -1
        used_slots = pair_slots[comparable]
        pair_totals += np.bincount(used_slots, minlength=len(pair_totals))
        correct = nearest_gt == matched_gt[used_slots]
        if correct.any():
            pair_correct += np.bincount(used_slots[correct], minlength=len(pair_correct))
        add_grouped_counts(composition, used_slots, nearest_gt.astype(str))

    valid = pair_totals > 0
    if valid.any():
        overlap = float(np.mean(pair_correct[valid] / pair_totals[valid]))
        majority = np.zeros(len(pair_totals), dtype=np.int64)
        for (slot, _gt), count in composition.items():
            majority[slot] = max(majority[slot], count)
        accuracy = float(np.mean(majority[valid] / pair_totals[valid]))
    else:
        overlap = accuracy = np.nan
    return vectors, assigned_rows, genes_seen, overlap, accuracy


def count_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    pearson = spearman = np.nan
    if valid.sum() >= 3 and pred[valid].std() > 0 and gt[valid].std() > 0:
        pearson = float(pearsonr(pred[valid], gt[valid])[0])
        spearman = float(spearmanr(pred[valid], gt[valid])[0])
    both = np.isfinite(pred) & np.isfinite(gt)
    mae = float(np.mean(np.abs(pred[both] - gt[both]))) if both.any() else np.nan
    rmse = float(np.sqrt(np.mean((pred[both] - gt[both]) ** 2))) if both.any() else np.nan
    return pearson, spearman, mae, rmse


def vector_metrics(
    pred_vectors: VectorCounts,
    gt_vectors: VectorCounts,
    n_pairs: int,
    gene_names: Sequence[str],
) -> Tuple[float, float, float]:
    genes = sorted(gene_names)
    if not genes:
        return np.nan, np.nan, np.nan
    pred_by_slot: Dict[int, Dict[str, int]] = defaultdict(dict)
    gt_by_slot: Dict[int, Dict[str, int]] = defaultdict(dict)
    for (slot, gene), value in pred_vectors.items():
        pred_by_slot[slot][gene] = value
    for (slot, gene), value in gt_vectors.items():
        gt_by_slot[slot][gene] = value
    cosines, distances, correlations = [], [], []
    for slot in range(n_pairs):
        pv = np.asarray([pred_by_slot.get(slot, {}).get(g, 0) for g in genes], float)
        gv = np.asarray([gt_by_slot.get(slot, {}).get(g, 0) for g in genes], float)
        norm = np.linalg.norm(pv) * np.linalg.norm(gv)
        if norm <= 0:
            continue
        cosines.append(float(np.dot(pv, gv) / norm))
        distances.append(float(jensenshannon(pv / pv.sum(), gv / gv.sum())))
        if len(genes) >= 2 and pv.std() > 0 and gv.std() > 0:
            correlations.append(float(pearsonr(pv, gv)[0]))
    mean = lambda values: float(np.mean(values)) if values else np.nan
    return mean(cosines), mean(distances), mean(correlations)


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if np.isfinite(value) else np.nan


def validate_dataset(name: str, cfg: Mapping[str, str]) -> None:
    metadata, input_tx, cells, assignments, _resource, geo = runtime_paths(cfg)
    for path in (metadata, input_tx, cells, assignments, geo.transform_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    pd.read_csv(cells, nrows=1)
    pd.read_csv(assignments, nrows=1)
    first = next(parquet_batches(input_tx, 1))
    if not geo.inside(first.x, first.y).all():
        raise ValueError(f"first input transcript is outside configured ROI: {input_tx}")
    x0, x1, y0, y1 = geo.micron_bounds()
    print(
        f"[OK] {name:14s} {geo.size} px; pixel x[{geo.x0},{geo.x0 + geo.size}) "
        f"y[{geo.y0},{geo.y0 + geo.size}); um bbox x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]"
    )


def evaluate(name: str, cfg: Mapping[str, str], args: argparse.Namespace) -> Dict[str, object]:
    metadata, input_tx, cells, assignments, _resource, geo = runtime_paths(cfg)
    label = cfg["label"]
    print(f"\n{SEP}\nBering x MERFISH {label}\n{SEP}")
    print(f"ROI: {geo.size} x {geo.size} mosaic px, x[{geo.x0},{geo.x0 + geo.size}) y[{geo.y0},{geo.y0 + geo.size})")
    print("prediction: final Bering ensemble; GT count/vector source: official-centroid NN baseline")

    pred = load_pred_cells(cells, geo)
    gt = load_gt_cells(metadata, geo)
    if pred.empty or gt.empty:
        raise ValueError(f"empty prediction or GT inside ROI for {name}: pred={len(pred)}, GT={len(gt)}")
    pred_xy = pred[["x", "y"]].to_numpy(float)
    gt_xy = gt[["x", "y"]].to_numpy(float)
    pi, gi, shifts = greedy_match(pred_xy, gt_xy, args.match_radius)
    n_match = len(pi)
    precision = n_match / len(pred)
    recall = n_match / len(gt)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"cells: pred={len(pred):,}, GT={len(gt):,}, matched={n_match:,}, F1={f1:.4f}")

    rng = np.random.default_rng(42)
    sample_positions = (
        rng.choice(n_match, min(args.vector_sample, n_match), replace=False)
        if n_match else np.zeros(0, dtype=int))
    sampled_pi, sampled_gi = pi[sample_positions], gi[sample_positions]
    sampled_cells = pred.iloc[sampled_pi].cell.astype(str).tolist()
    matched_cells = pred.iloc[pi].cell.astype(str).tolist()

    gt_counts, gt_vectors, gt_rows, gt_assigned = load_gt_counts_and_vectors(
        input_tx, geo, gt_xy, sampled_gi, args.batch_size, args.gt_assignment_radius)
    pred_vectors, pred_rows, pred_genes, assign_overlap, assign_accuracy = (
        load_pred_vectors_and_assignment_metrics(
            assignments, geo, pred.cell.astype(str).tolist(), sampled_cells,
            matched_cells, gi, gt_xy, args.match_radius, args.batch_size))
    print(
        f"transcripts: Bering final assigned={pred_rows:,}; GT NN assigned="
        f"{gt_assigned:,}/{gt_rows:,} ({gt_assigned / gt_rows * 100 if gt_rows else 0:.1f}%)")

    pred_counts = pred.iloc[pi].n_transcripts.to_numpy(float)
    matched_gt_counts = gt_counts[gi].astype(float)
    cpear, cspear, cmae, crmse = count_metrics(pred_counts, matched_gt_counts)
    vcos, vjs, vpear = vector_metrics(
        pred_vectors, gt_vectors, len(sample_positions), pred_genes)

    row: Dict[str, object] = {
        "dataset": label,
        "roi_size_px": geo.size,
        "gt_cells": len(gt),
        "pred_cells": len(pred),
        "cell_ratio": rounded(len(pred) / len(gt)),
        "matched": n_match,
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1),
        "loc_mean_um": rounded(np.mean(shifts)) if n_match else np.nan,
        "loc_median_um": rounded(np.median(shifts)) if n_match else np.nan,
        "loc_p95_um": rounded(np.percentile(shifts, 95)) if n_match else np.nan,
        "count_pearson": rounded(cpear),
        "count_spearman": rounded(cspear),
        "count_mae": rounded(cmae, 2),
        "count_rmse": rounded(crmse, 2),
        "vec_cosine": rounded(vcos),
        "vec_js_dist": rounded(vjs),
        "vec_pearson": rounded(vpear),
        "assign_overlap": rounded(assign_overlap),
        "assign_accuracy": rounded(assign_accuracy),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str.lower, choices=list(DATASETS))
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="single output CSV (default: %(default)s)",
    )
    parser.add_argument("--match-radius", type=float, default=MATCH_RADIUS_UM)
    parser.add_argument("--gt-assignment-radius", type=float, default=GT_ASSIGN_RADIUS_UM)
    parser.add_argument("--vector-sample", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    if args.vector_sample <= 0 or args.batch_size <= 0:
        raise ValueError("--vector-sample and --batch-size must be positive")
    if args.match_radius <= 0 or args.gt_assignment_radius <= 0:
        raise ValueError("matching radii must be positive")
    names = list(DATASETS) if args.all else [args.dataset]
    if args.check_only:
        for name in names:
            validate_dataset(name, DATASETS[name])
        return
    rows = [evaluate(name, DATASETS[name], args) for name in names]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "roi_size_px", *METRIC_COLUMNS]
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    print(f"\ncombined results: {output}")


if __name__ == "__main__":
    main()
