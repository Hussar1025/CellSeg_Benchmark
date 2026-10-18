#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Baysor on 13 centre-ROI MERFISH datasets with 19 metrics.

GT cell centroids come from Vizgen's official ``cell_metadata.csv``.  The raw
MERFISH transcript tables do not contain official cell assignments, therefore
the count/gene-vector GT follows ``proseg_value_merfish_1.py``: each transcript
is assigned to its nearest official GT centroid within 15 um.  Consequently,
count/vector metrics are an NN baseline rather than independent ground truth.

The ROI is defined in *mosaic pixels*, whereas Baysor and cell_metadata use
global micrometres.  Every predicted centroid, GT centroid and transcript is
therefore transformed back to mosaic pixels and filtered with the exact
half-open centre window.  Mouse brain uses 20000 x 20000 px; the other 12
datasets use 10000 x 10000 px.

Examples
--------
Check paths, columns and ROI bounds without doing the expensive evaluation::

    python baysor_value_merfish_center10000.py --all --check-only

Evaluate all 13 datasets and write one combined CSV::

    python baysor_value_merfish_center10000.py --all

Evaluate one dataset::

    python baysor_value_merfish_center10000.py --dataset colon1
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


DATA_ROOT = Path("/data/qiuyijia/dataset")
BAYSOR_ROOT = Path("/data/qiuyijia/baysor")
DEFAULT_OUT = Path("/data/qiuyijia/eval/baysor/baysor_merfish_evaluation.csv")
MATCH_RADIUS_UM = 10.0
GT_ASSIGN_RADIUS_UM = 15.0
SEP = "=" * 78
EXCLUDED_GENE_PREFIXES = ("blank",)

# Cache names use the established tx_roi_<y0>_<x0>_<size> convention.
# ``Uterian`` is the spelling on disk; output labels use the requested
# biological name ``Uterine``.
DATASETS: Mapping[str, Mapping[str, object]] = {
    "melanoma": {
        "root": DATA_ROOT / "merfish_Melanoma",
        "pred": BAYSOR_ROOT / "merfish_Melanoma_center10000",
        "cache": DATA_ROOT / "merfish_Melanoma/_roi_cache/tx_roi_46486_56591_10000.parquet",
        "roi_y0": 46486, "roi_x0": 56591, "roi_size": 10000, "label": "Melanoma",
    },
    "ovarian1": {
        "root": DATA_ROOT / "merfish_ovarian1",
        "pred": BAYSOR_ROOT / "merfish_ovarian1_center10000",
        "cache": DATA_ROOT / "merfish_ovarian1/_roi_cache/tx_roi_41878_56766_10000.parquet",
        "roi_y0": 41878, "roi_x0": 56766, "roi_size": 10000, "label": "ovarian1",
    },
    "ovarian2": {
        "root": DATA_ROOT / "merfish_ovarian2",
        "pred": BAYSOR_ROOT / "merfish_ovarian2_center10000",
        "cache": DATA_ROOT / "merfish_ovarian2/_roi_cache/tx_roi_41878_56766_10000.parquet",
        "roi_y0": 41878, "roi_x0": 56766, "roi_size": 10000, "label": "ovarian2",
    },
    "colon1": {
        "root": DATA_ROOT / "merfish_colon1",
        "pred": BAYSOR_ROOT / "merfish_colon1_center10000",
        "cache": DATA_ROOT / "merfish_colon1/_roi_cache/tx_roi_32831_52122_10000.parquet",
        "roi_y0": 32831,
        "roi_x0": 52122,
        "roi_size": 10000,
        "label": "colon1",
    },
    "colon2": {
        "root": DATA_ROOT / "merfish_colon2",
        "pred": BAYSOR_ROOT / "merfish_colon2_center10000",
        "cache": DATA_ROOT / "merfish_colon2/_roi_cache/tx_roi_37680_67126_10000.parquet",
        "roi_y0": 37680,
        "roi_x0": 67126,
        "roi_size": 10000,
        "label": "colon2",
    },
    "uterine": {
        "root": DATA_ROOT / "merfish_Uterian",
        "pred": BAYSOR_ROOT / "merfish_Uterian_center10000",
        "cache": DATA_ROOT / "merfish_Uterian/_roi_cache/tx_roi_39546_49732_10000.parquet",
        "roi_y0": 39546,
        "roi_x0": 49732,
        "roi_size": 10000,
        "label": "Uterine",
    },
    "breast_cancer": {
        "root": DATA_ROOT / "merfish_breast_cancer",
        "pred": BAYSOR_ROOT / "merfish_breast_cancer_center10000",
        "cache": DATA_ROOT / "merfish_breast_cancer/_roi_cache/tx_roi_42402_50242_10000.parquet",
        "roi_y0": 42402,
        "roi_x0": 50242,
        "roi_size": 10000,
        "label": "breast_cancer",
    },
    "liver1": {
        "root": DATA_ROOT / "merfish_liver1",
        "pred": BAYSOR_ROOT / "merfish_liver1_center10000",
        "cache": DATA_ROOT / "merfish_liver1/_roi_cache/tx_roi_36396_49272_10000.parquet",
        "roi_y0": 36396, "roi_x0": 49272, "roi_size": 10000, "label": "liver1",
    },
    "liver2": {
        "root": DATA_ROOT / "merfish_liver2",
        "pred": BAYSOR_ROOT / "merfish_liver2_center10000",
        "cache": DATA_ROOT / "merfish_liver2/_roi_cache/tx_roi_46028_53436_10000.parquet",
        "roi_y0": 46028, "roi_x0": 53436, "roi_size": 10000, "label": "liver2",
    },
    "lung_cancer": {
        "root": DATA_ROOT / "MERFISH_Lung_cancer",
        "pred": BAYSOR_ROOT / "MERFISH_Lung_cancer_center10000",
        "cache": DATA_ROOT / "MERFISH_Lung_cancer/_roi_cache/tx_roi_31434_36213_10000.parquet",
        "roi_y0": 31434, "roi_x0": 36213, "roi_size": 10000, "label": "Lung_cancer",
    },
    "mouse_brain": {
        "root": DATA_ROOT / "merfish_mouse_brain",
        "pred": BAYSOR_ROOT / "merfish_1_output",
        # This is the cleaned raw transcript table for the same 20000-px ROI;
        # no Bering cell assignment or prediction is read from it.
        "cache": Path("/data/qiuyijia/bering/merfish_new_output/merfish_mouse_brain_center20000_output/bering_input_transcripts_roi20000px_cellpose.parquet"),
        "roi_y0": 20655, "roi_x0": 34542, "roi_size": 20000, "label": "mouse_brain",
    },
    "prostate1": {
        "root": DATA_ROOT / "merfish_prostate1",
        "pred": BAYSOR_ROOT / "merfish_prostate1_center10000",
        "cache": DATA_ROOT / "merfish_prostate1/_roi_cache/tx_roi_50405_50185_10000.parquet",
        "roi_y0": 50405, "roi_x0": 50185, "roi_size": 10000, "label": "prostate1",
    },
    "prostate2": {
        "root": DATA_ROOT / "merfish_prostate2",
        "pred": BAYSOR_ROOT / "merfish_prostate2_center10000",
        "cache": DATA_ROOT / "merfish_prostate2/_roi_cache/tx_roi_44812_50242_10000.parquet",
        "roi_y0": 44812, "roi_x0": 50242, "roi_size": 10000, "label": "prostate2",
    },
}

METRIC_COLUMNS = [
    "gt_cells", "pred_cells", "cell_ratio", "matched", "precision", "recall", "f1",
    "loc_mean_um", "loc_median_um", "loc_p95_um", "count_pearson",
    "count_spearman", "count_mae", "count_rmse", "vec_cosine", "vec_js_dist",
    "vec_pearson", "assign_overlap", "assign_accuracy",
]


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"cannot find {pattern!r} under {root}")
    return hits[0]


def normalize_cell(value: object) -> str:
    """Normalize Baysor IDs while retaining the ``cell_N`` naming scheme."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "0", "-1", "nan", "None"}:
        return ""
    return text


class Geometry:
    def __init__(self, transform_path: Path, roi_x0: int, roi_y0: int, roi_size: int):
        self.path = transform_path
        self.matrix = np.loadtxt(transform_path, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"expected 3x3 transform, got {self.matrix.shape}: {transform_path}")
        self.x0 = int(roi_x0)
        self.y0 = int(roi_y0)
        self.size = int(roi_size)

    def to_pixel(self, x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        px = self.matrix[0, 0] * x_arr + self.matrix[0, 1] * y_arr + self.matrix[0, 2]
        py = self.matrix[1, 0] * x_arr + self.matrix[1, 1] * y_arr + self.matrix[1, 2]
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
            [self.x0, self.y0, 1.0],
            [self.x0 + self.size, self.y0, 1.0],
            [self.x0, self.y0 + self.size, 1.0],
            [self.x0 + self.size, self.y0 + self.size, 1.0],
        ])
        um = corners @ inv.T
        return float(um[:, 0].min()), float(um[:, 0].max()), float(um[:, 1].min()), float(um[:, 1].max())


def dataset_paths(cfg: Mapping[str, object]) -> Tuple[Path, Path, Path, Path, Path]:
    root = Path(cfg["root"])
    pred = Path(cfg["pred"])
    cache = Path(cfg["cache"])
    metadata = find_one(root, "*cell_metadata*.csv")
    transform = find_one(root, "*micron_to_mosaic_pixel_transform.csv")
    return metadata, transform, cache, pred / "segmentation_cell_stats.csv", pred / "segmentation.csv"


def pick_column(columns: Sequence[str], choices: Sequence[str], description: str) -> str:
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    for name in choices:
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"cannot find {description} column; columns={list(columns)}")


def load_gt_cells(path: Path, geo: Geometry) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    xcol = pick_column(header.columns, ("center_x", "centroid_x", "x"), "GT x")
    ycol = pick_column(header.columns, ("center_y", "centroid_y", "y"), "GT y")
    idcol = str(header.columns[0])
    frame = pd.read_csv(path, usecols=[idcol, xcol, ycol]).rename(
        columns={idcol: "gt_id", xcol: "x", ycol: "y"})
    frame["x"] = pd.to_numeric(frame.x, errors="coerce")
    frame["y"] = pd.to_numeric(frame.y, errors="coerce")
    return frame.loc[geo.inside(frame.x, frame.y)].reset_index(drop=True)


def load_pred_cells(path: Path, geo: Geometry) -> pd.DataFrame:
    frame = pd.read_csv(path)
    ccol = pick_column(frame.columns, ("cell", "cell_id"), "Baysor cell")
    xcol = pick_column(frame.columns, ("x", "centroid_x"), "Baysor x")
    ycol = pick_column(frame.columns, ("y", "centroid_y"), "Baysor y")
    ncol = pick_column(frame.columns, ("n_transcripts", "transcript_count"), "Baysor transcript count")
    frame = frame.rename(columns={ccol: "cell", xcol: "x", ycol: "y", ncol: "n_transcripts"})
    frame["cell"] = frame.cell.map(normalize_cell)
    for col in ("x", "y", "n_transcripts"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keep = geo.inside(frame.x, frame.y) & frame.cell.ne("")
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.cell.duplicated().any():
        raise ValueError(f"duplicate Baysor cell IDs in {path}")
    return frame


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match exactly as the supplied reference: nearest GT, best pred per GT."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    distances, gt_indices = cKDTree(gt_xy).query(pred_xy, k=1, distance_upper_bound=radius)
    best: Dict[int, Tuple[int, float]] = {}
    for pred_i, (distance, gt_i) in enumerate(zip(distances, gt_indices)):
        if not np.isfinite(distance) or distance >= radius:
            continue
        gt_i = int(gt_i)
        if gt_i not in best or distance < best[gt_i][1]:
            best[gt_i] = (pred_i, float(distance))
    ordered = sorted((pred_i, gt_i, distance) for gt_i, (pred_i, distance) in best.items())
    if not ordered:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    pred_i, gt_i, distance = zip(*ordered)
    return np.asarray(pred_i), np.asarray(gt_i), np.asarray(distance)


def parquet_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    xcol = pick_column(pf.schema.names, ("global_x", "x", "x_um"), "transcript x")
    ycol = pick_column(pf.schema.names, ("global_y", "y", "y_um"), "transcript y")
    gcol = pick_column(pf.schema.names, ("gene", "target", "feature_name"), "transcript gene")
    for batch in pf.iter_batches(batch_size=batch_size, columns=[xcol, ycol, gcol]):
        yield batch.to_pandas().rename(columns={xcol: "x", ycol: "y", gcol: "gene"})


def csv_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0)
    ccol = pick_column(header.columns, ("cell", "cell_id"), "Baysor transcript cell")
    xcol = pick_column(header.columns, ("x", "global_x", "x_um"), "Baysor transcript x")
    ycol = pick_column(header.columns, ("y", "global_y", "y_um"), "Baysor transcript y")
    gcol = pick_column(header.columns, ("gene", "target", "feature_name"), "Baysor transcript gene")
    noise = next((c for c in header.columns if str(c).lower() == "is_noise"), None)
    usecols = [ccol, xcol, ycol, gcol] + ([noise] if noise else [])
    ren = {ccol: "cell", xcol: "x", ycol: "y", gcol: "gene"}
    if noise:
        ren[noise] = "is_noise"
    for frame in pd.read_csv(path, usecols=usecols, chunksize=batch_size):
        yield frame.rename(columns=ren)


VectorCounts = MutableMapping[Tuple[int, str], int]


def add_grouped_counts(store: VectorCounts, slots: np.ndarray, genes: np.ndarray) -> None:
    valid = slots >= 0
    if not valid.any():
        return
    small = pd.DataFrame({"slot": slots[valid], "gene": genes[valid]}).groupby(
        ["slot", "gene"], sort=False).size()
    for (slot, gene), count in small.items():
        store[(int(slot), str(gene))] += int(count)


def load_gt_counts_and_vectors(
    cache: Path,
    geo: Geometry,
    gt_xy: np.ndarray,
    sampled_gt: np.ndarray,
    batch_size: int,
    assignment_radius: float,
) -> Tuple[np.ndarray, VectorCounts, int, int]:
    tree = cKDTree(gt_xy)
    totals = np.zeros(len(gt_xy), dtype=np.int64)
    sample_slot = np.full(len(gt_xy), -1, dtype=np.int32)
    sample_slot[sampled_gt] = np.arange(len(sampled_gt), dtype=np.int32)
    vectors: VectorCounts = defaultdict(int)
    rows = assigned = 0
    for frame in parquet_batches(cache, batch_size):
        frame["x"] = pd.to_numeric(frame.x, errors="coerce")
        frame["y"] = pd.to_numeric(frame.y, errors="coerce")
        # Match every Baysor run's ``exclude_genes = "Blank*"`` setting so
        # the prediction and the NN count/vector baseline use identical
        # transcript universes.
        excluded = frame.gene.astype(str).str.lower().str.startswith(
            EXCLUDED_GENE_PREFIXES, na=False)
        keep = geo.inside(frame.x, frame.y) & ~excluded
        if not keep.any():
            continue
        xy = frame.loc[keep, ["x", "y"]].to_numpy(np.float64)
        genes = frame.loc[keep, "gene"].astype(str).to_numpy()
        rows += len(xy)
        distance, gt_i = tree.query(xy, k=1, distance_upper_bound=assignment_radius)
        valid = np.isfinite(distance) & (distance < assignment_radius) & (gt_i < len(gt_xy))
        assigned += int(valid.sum())
        if valid.any():
            good_gt = gt_i[valid].astype(np.int64)
            totals += np.bincount(good_gt, minlength=len(gt_xy))
            add_grouped_counts(vectors, sample_slot[good_gt], genes[valid])
    return totals, vectors, rows, assigned


def load_pred_vectors(
    segmentation: Path,
    geo: Geometry,
    sampled_pred_cells: Sequence[str],
    matched_pred_cells: Sequence[str],
    matched_gt: np.ndarray,
    gt_xy: np.ndarray,
    assignment_radius: float,
    batch_size: int,
) -> Tuple[VectorCounts, int, set[str], float, float]:
    lookup = {cell: i for i, cell in enumerate(sampled_pred_cells)}
    pair_lookup = {cell: i for i, cell in enumerate(matched_pred_cells)}
    gt_tree = cKDTree(gt_xy)
    pair_totals = np.zeros(len(matched_pred_cells), dtype=np.int64)
    pair_correct = np.zeros(len(matched_pred_cells), dtype=np.int64)
    pair_composition: MutableMapping[Tuple[int, int], int] = defaultdict(int)
    vectors: VectorCounts = defaultdict(int)
    gene_names: set[str] = set()
    assigned_rows = 0
    for frame in csv_batches(segmentation, batch_size):
        frame["x"] = pd.to_numeric(frame.x, errors="coerce")
        frame["y"] = pd.to_numeric(frame.y, errors="coerce")
        cell = frame.cell.map(normalize_cell)
        keep = geo.inside(frame.x, frame.y) & cell.ne("")
        if "is_noise" in frame:
            noise = pd.to_numeric(frame.is_noise, errors="coerce").fillna(0).astype(bool)
            keep &= ~noise
        assigned_rows += int(keep.sum())
        if not keep.any():
            continue
        kept_cells = cell.loc[keep]
        slots = kept_cells.map(lookup).fillna(-1).to_numpy(np.int32)
        genes = frame.loc[keep, "gene"].astype(str).to_numpy()
        gene_names.update(genes)
        add_grouped_counts(vectors, slots, genes)

        pair_slots = kept_cells.map(pair_lookup).fillna(-1).to_numpy(np.int32)
        comparable = pair_slots >= 0
        if comparable.any():
            xy = frame.loc[keep, ["x", "y"]].to_numpy(np.float64)[comparable]
            distance, nearest_gt = gt_tree.query(
                xy, k=1, distance_upper_bound=assignment_radius)
            nearest_gt = nearest_gt.astype(np.int64)
            nearest_gt[~np.isfinite(distance) | (distance >= assignment_radius)] = -1
            used_slots = pair_slots[comparable]
            pair_totals += np.bincount(used_slots, minlength=len(pair_totals))
            is_correct = nearest_gt == matched_gt[used_slots]
            if is_correct.any():
                pair_correct += np.bincount(
                    used_slots[is_correct], minlength=len(pair_correct))
            add_grouped_counts(
                pair_composition, used_slots, nearest_gt.astype(str))

    valid = pair_totals > 0
    if valid.any():
        overlap = float(np.mean(pair_correct[valid] / pair_totals[valid]))
        majority = np.zeros(len(pair_totals), dtype=np.int64)
        for (slot, _nearest_gt), count in pair_composition.items():
            majority[slot] = max(majority[slot], count)
        accuracy = float(np.mean(majority[valid] / pair_totals[valid]))
    else:
        overlap = accuracy = np.nan
    return vectors, assigned_rows, gene_names, overlap, accuracy


def safe_correlations(pred_count: np.ndarray, gt_count: np.ndarray) -> Tuple[float, float, float, float]:
    valid = np.isfinite(pred_count) & np.isfinite(gt_count) & (pred_count > 0) & (gt_count > 0)
    pearson = spearman = np.nan
    if valid.sum() >= 3 and pred_count[valid].std() > 0 and gt_count[valid].std() > 0:
        pearson = float(pearsonr(pred_count[valid], gt_count[valid])[0])
        spearman = float(spearmanr(pred_count[valid], gt_count[valid])[0])
    both = np.isfinite(pred_count) & np.isfinite(gt_count)
    mae = float(np.mean(np.abs(pred_count[both] - gt_count[both]))) if both.any() else np.nan
    rmse = float(np.sqrt(np.mean((pred_count[both] - gt_count[both]) ** 2))) if both.any() else np.nan
    return pearson, spearman, mae, rmse


def vector_metrics(
    pred_vectors: VectorCounts,
    gt_vectors: VectorCounts,
    n_pairs: int,
    gene_names: Sequence[str],
) -> Tuple[float, float, float]:
    cosine_values, js_values, pearson_values = [], [], []
    genes = sorted(gene_names)
    if not genes:
        return np.nan, np.nan, np.nan
    pred_by_slot: Dict[int, Dict[str, int]] = defaultdict(dict)
    gt_by_slot: Dict[int, Dict[str, int]] = defaultdict(dict)
    for (slot, gene), count in pred_vectors.items():
        pred_by_slot[slot][gene] = count
    for (slot, gene), count in gt_vectors.items():
        gt_by_slot[slot][gene] = count
    for slot in range(n_pairs):
        pred_gene = pred_by_slot.get(slot, {})
        gt_gene = gt_by_slot.get(slot, {})
        pv = np.asarray([pred_gene.get(g, 0) for g in genes], dtype=float)
        gv = np.asarray([gt_gene.get(g, 0) for g in genes], dtype=float)
        norm = np.linalg.norm(pv) * np.linalg.norm(gv)
        if norm <= 0:
            continue
        cosine_values.append(float(np.dot(pv, gv) / norm))
        js_values.append(float(jensenshannon(pv / pv.sum(), gv / gv.sum())))
        if len(genes) >= 2 and pv.std() > 0 and gv.std() > 0:
            pearson_values.append(float(pearsonr(pv, gv)[0]))
    mean = lambda values: float(np.mean(values)) if values else np.nan
    return mean(cosine_values), mean(js_values), mean(pearson_values)


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if np.isfinite(value) else np.nan


def validate_dataset(name: str, cfg: Mapping[str, object]) -> None:
    metadata, transform, cache, stats, segmentation = dataset_paths(cfg)
    for path in (metadata, transform, cache, stats, segmentation):
        if not path.is_file():
            raise FileNotFoundError(path)
    geo = Geometry(
        transform, int(cfg["roi_x0"]), int(cfg["roi_y0"]), int(cfg["roi_size"]))
    x0, x1, y0, y1 = geo.micron_bounds()
    pd.read_csv(stats, nrows=1)
    pd.read_csv(segmentation, nrows=1)
    next(parquet_batches(cache, 1))
    print(
        f"[OK] {name:14s} {geo.size} px; pixel x[{geo.x0},{geo.x0 + geo.size}) "
        f"y[{geo.y0},{geo.y0 + geo.size}); "
        f"global um bounding box x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]"
    )


def evaluate(name: str, cfg: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    metadata, transform, cache, stats, segmentation = dataset_paths(cfg)
    geo = Geometry(
        transform, int(cfg["roi_x0"]), int(cfg["roi_y0"]), int(cfg["roi_size"]))
    label = str(cfg["label"])

    print(f"\n{SEP}\nBaysor x MERFISH {label}\n{SEP}")
    print(
        f"ROI: {geo.size} x {geo.size} mosaic px, "
        f"x[{geo.x0},{geo.x0 + geo.size}) y[{geo.y0},{geo.y0 + geo.size})")
    print("GT count/gene vectors: transcript-to-official-centroid NN baseline")

    pred = load_pred_cells(stats, geo)
    gt = load_gt_cells(metadata, geo)
    pred_xy = pred[["x", "y"]].to_numpy(float)
    gt_xy = gt[["x", "y"]].to_numpy(float)
    print(f"cells: pred={len(pred):,}, GT={len(gt):,}")

    pi, gi, shifts = greedy_match(pred_xy, gt_xy, args.match_radius)
    n_match = len(pi)
    precision = n_match / len(pred) if len(pred) else 0.0
    recall = n_match / len(gt) if len(gt) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"matched={n_match:,}, precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")

    rng = np.random.default_rng(42)
    sampled_pair_positions = (
        rng.choice(n_match, size=min(args.vector_sample, n_match), replace=False)
        if n_match else np.zeros(0, dtype=int))
    sampled_pi = pi[sampled_pair_positions]
    sampled_gi = gi[sampled_pair_positions]
    sampled_cells = pred.iloc[sampled_pi].cell.astype(str).tolist()

    gt_counts, gt_vectors, gt_rows, gt_assigned = load_gt_counts_and_vectors(
        cache, geo, gt_xy, sampled_gi, args.batch_size, args.gt_assignment_radius)
    matched_cells = pred.iloc[pi].cell.astype(str).tolist()
    pred_vectors, pred_rows, pred_gene_names, assign_overlap, assign_accuracy = load_pred_vectors(
        segmentation, geo, sampled_cells, matched_cells, gi, gt_xy,
        args.match_radius, args.batch_size)
    print(
        f"transcripts: Baysor assigned={pred_rows:,}; GT NN assigned="
        f"{gt_assigned:,}/{gt_rows:,} ({gt_assigned / gt_rows * 100 if gt_rows else 0:.1f}%)")

    pred_counts = pred.iloc[pi].n_transcripts.to_numpy(float)
    matched_gt_counts = gt_counts[gi].astype(float)
    count_pearson, count_spearman, count_mae, count_rmse = safe_correlations(
        pred_counts, matched_gt_counts)
    vec_cosine, vec_js, vec_pearson = vector_metrics(
        pred_vectors, gt_vectors, len(sampled_pair_positions), pred_gene_names)

    metrics: Dict[str, object] = {
        "dataset": label,
        "roi_size_px": geo.size,
        "gt_cells": len(gt),
        "pred_cells": len(pred),
        "cell_ratio": rounded(len(pred) / len(gt)) if len(gt) else np.nan,
        "matched": n_match,
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1),
        "loc_mean_um": rounded(np.mean(shifts)) if n_match else np.nan,
        "loc_median_um": rounded(np.median(shifts)) if n_match else np.nan,
        "loc_p95_um": rounded(np.percentile(shifts, 95)) if n_match else np.nan,
        "count_pearson": rounded(count_pearson),
        "count_spearman": rounded(count_spearman),
        "count_mae": rounded(count_mae, 2),
        "count_rmse": rounded(count_rmse, 2),
        "vec_cosine": rounded(vec_cosine),
        "vec_js_dist": rounded(vec_js),
        "vec_pearson": rounded(vec_pearson),
        "assign_overlap": rounded(assign_overlap),
        "assign_accuracy": rounded(assign_accuracy),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str.lower, choices=list(DATASETS))
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="single output CSV (default: %(default)s)")
    parser.add_argument("--match-radius", type=float, default=MATCH_RADIUS_UM)
    parser.add_argument("--gt-assignment-radius", type=float, default=GT_ASSIGN_RADIUS_UM)
    parser.add_argument("--vector-sample", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    names = list(DATASETS) if args.all else [args.dataset]
    if args.vector_sample <= 0 or args.batch_size <= 0:
        raise ValueError("--vector-sample and --batch-size must be positive")
    if args.match_radius <= 0 or args.gt_assignment_radius <= 0:
        raise ValueError("matching radii must be positive")
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
