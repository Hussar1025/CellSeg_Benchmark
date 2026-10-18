#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Baysor on the selected CosMx human-brain and lymph-node FOVs.

The metric definitions and processing follow ``ucs_value_cosmx_1.py``:

* official CosMx metadata supplies GT centroids, cell counts and areas;
* the match radius is inferred from the 95th percentile of GT nearest-neighbour
  distances and multiplied by 1.25;
* cells are matched one-to-one with sparse, component-wise Hungarian matching;
* detection, centroid-shift, transcript-count, gene-vector and transcript-
  assignment metrics are reported with the same selected-metric names.

Unlike UCS, Baysor already supplies cell centroids and molecule assignments.
The two Baysor runs used only selected FOVs, so GT metadata is restricted to
exactly those FOVs.  Transcript metrics use the exact Parquet inputs recorded
by Baysor.  Genes excluded by the Baysor config are removed only for row-wise
alignment to ``segmentation.csv``; they remain in the GT transcript totals and
GT gene vectors, matching the reference evaluation's use of the official
transcript table.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12
VERSION = "2026.09.21-cosmx-baysor-subset-v1"
PIXEL_SIZE_UM = 0.1203
OUTPUT_DIR = Path("/data/qiuyijia/eval/baysor")
EXCLUDED_GENE_PREFIXES = (
    "SystemControl", "Negative", "NegPrb", "Neg_", "FalseCode", "Blank",
    "UnassignedCodeword", "DeprecatedCodeword", "GenomicControl",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    baysor_dir: str
    metadata_path: str
    transcript_path: str
    fovs: Tuple[int, ...]

    @property
    def stats_path(self) -> Path:
        return Path(self.baysor_dir) / "segmentation_cell_stats.csv"

    @property
    def segmentation_path(self) -> Path:
        return Path(self.baysor_dir) / "segmentation.csv"

    @property
    def params_path(self) -> Path:
        return Path(self.baysor_dir) / "segmentation_params.dump.toml"


HUMAN_FOVS = (
    4, 18, 51, 52, 56, 65, 74, 87, 90, 101, 104, 110, 124, 131, 147,
    173, 187, 220, 224, 234, 242, 250, 255, 256, 264, 296, 298, 307, 339,
)

DATASETS: Dict[str, DatasetSpec] = {
    "human_brain": DatasetSpec(
        name="human_brain",
        baysor_dir="/data/qiuyijia/baysor/cosmx_human_brain",
        metadata_path="/data/qiuyijia/dataset/cosmx_human_brain/S0_metadata_file.csv.gz",
        transcript_path=(
            "/data/qiuyijia/dataset/cosmx_human_brain/_inspect_cache/"
            "tx_fov_4_18_51_52_56_65_74_87_90_101_104_110_124_131_147_"
            "173_187_220_224_234_242_250_255_256_264_296_298_307_339.parquet"
        ),
        fovs=HUMAN_FOVS,
    ),
    "lymph_fov48_261": DatasetSpec(
        name="lymph_fov48_261",
        baysor_dir="/data/qiuyijia/baysor/cosmx_lymph_fov48_261",
        metadata_path=(
            "/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/"
            "S0_metadata_file.csv.gz"
        ),
        transcript_path=(
            "/data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/"
            "tx_fov_48_261.parquet"
        ),
        fovs=(48, 261),
    ),
}


REQUESTED_METRICS = [
    ("Pred cell count", "pred_cell_count"),
    ("GT cell count", "gt_cell_count"),
    ("Cell count ratio", "cell_count_ratio"),
    ("Matched cell count", "matched_cell_count"),
    ("Detection precision", "detection_precision"),
    ("Detection recall", "detection_recall"),
    ("Detection F1", "detection_f1"),
    ("Mean centroid shift", "mean_centroid_shift"),
    ("Median centroid shift", "median_centroid_shift"),
    ("p95 centroid shift", "p95_centroid_shift"),
    (
        "matched_pair_n_transcripts_vs_total_counts_pearson",
        "matched_pair_n_transcripts_vs_total_counts_pearson",
    ),
    (
        "matched_pair_n_transcripts_vs_total_counts_spearman",
        "matched_pair_n_transcripts_vs_total_counts_spearman",
    ),
    (
        "matched_pair_n_transcripts_vs_total_counts_mae",
        "matched_pair_n_transcripts_vs_total_counts_mae",
    ),
    (
        "matched_pair_n_transcripts_vs_total_counts_rmse",
        "matched_pair_n_transcripts_vs_total_counts_rmse",
    ),
    ("Pred transcript rows", "pred_transcript_rows"),
    ("GT transcript rows", "gt_transcript_rows"),
    ("Pred transcript assignment rate", "pred_transcript_assignment_rate"),
    ("Transcript matched cell pairs", "transcript_matched_cell_pairs_n"),
    ("matched_cell_transcript_count_pearson", "matched_cell_transcript_count_pearson"),
    ("matched_cell_transcript_count_spearman", "matched_cell_transcript_count_spearman"),
    ("matched_cell_transcript_count_mae", "matched_cell_transcript_count_mae"),
    ("matched_cell_transcript_count_rmse", "matched_cell_transcript_count_rmse"),
    ("matched_cell_gene_vector_mean_cosine", "matched_cell_gene_vector_mean_cosine"),
    (
        "matched_cell_gene_vector_mean_js_distance",
        "matched_cell_gene_vector_mean_js_distance",
    ),
    ("matched_cell_gene_vector_mean_pearson", "matched_cell_gene_vector_mean_pearson"),
    ("matched_cell_gene_vector_valid_pairs", "matched_cell_gene_vector_valid_pairs"),
    ("transcript_id_overlap_n", "transcript_id_overlap_n"),
    (
        "transcript_assignment_accuracy_via_matched_cells",
        "transcript_assignment_accuracy_via_matched_cells",
    ),
]


def info(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def validate_spec(spec: DatasetSpec) -> None:
    paths = [
        Path(spec.metadata_path), Path(spec.transcript_path), spec.stats_path,
        spec.segmentation_path, spec.params_path,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    meta_columns = set(pd.read_csv(spec.metadata_path, nrows=0).columns)
    required_meta = {
        "fov", "cell_ID", "cell_id", "CenterX_global_px", "CenterY_global_px",
        "nCount_RNA", "Area",
    }
    if not required_meta.issubset(meta_columns):
        raise ValueError(f"{spec.metadata_path} lacks {required_meta - meta_columns}")

    stats_columns = set(pd.read_csv(spec.stats_path, nrows=0).columns)
    required_stats = {"cell", "x", "y", "n_transcripts", "area", "cluster"}
    if not required_stats.issubset(stats_columns):
        raise ValueError(f"{spec.stats_path} lacks {required_stats - stats_columns}")

    seg_columns = set(pd.read_csv(spec.segmentation_path, nrows=0).columns)
    required_seg = {"cell", "gene", "x", "y", "is_noise"}
    if not required_seg.issubset(seg_columns):
        raise ValueError(f"{spec.segmentation_path} lacks {required_seg - seg_columns}")

    parquet_columns = set(pq.ParquetFile(spec.transcript_path).schema.names)
    required_tx = {"fov", "cell_ID", "x_global_px", "y_global_px", "target"}
    if not required_tx.issubset(parquet_columns):
        raise ValueError(f"{spec.transcript_path} lacks {required_tx - parquet_columns}")

    params = spec.params_path.read_text(encoding="utf-8")
    if '[prior]\ntype = "none"' not in params:
        raise ValueError(f"{spec.params_path}: expected Baysor prior.type=none")


def load_gt_cells(spec: DatasetSpec, pixel_size_um: float) -> pd.DataFrame:
    usecols = [
        "fov", "cell_ID", "cell_id", "CenterX_global_px", "CenterY_global_px",
        "nCount_RNA", "Area",
    ]
    header = set(pd.read_csv(spec.metadata_path, nrows=0).columns)
    if "cell_type" in header:
        usecols.append("cell_type")
    raw = pd.read_csv(spec.metadata_path, usecols=usecols)
    raw = raw.loc[pd.to_numeric(raw.fov, errors="raise").astype(int).isin(spec.fovs)].copy()
    if raw.empty:
        raise ValueError(f"no GT cells found for {spec.name} FOVs {spec.fovs}")
    raw["cell_id"] = raw.cell_id.astype(str)
    if raw.cell_id.duplicated().any():
        raise ValueError(f"{spec.name}: duplicate GT cell_id values")
    raw["x"] = pd.to_numeric(raw.CenterX_global_px, errors="raise") * pixel_size_um
    raw["y"] = pd.to_numeric(raw.CenterY_global_px, errors="raise") * pixel_size_um
    raw["total_counts"] = pd.to_numeric(raw.nCount_RNA, errors="raise")
    raw["area_um2"] = pd.to_numeric(raw.Area, errors="raise") * pixel_size_um**2
    return raw.reset_index(drop=True)


def load_pred_cells(spec: DatasetSpec, pixel_size_um: float) -> pd.DataFrame:
    raw = pd.read_csv(
        spec.stats_path,
        usecols=["cell", "x", "y", "n_transcripts", "area", "cluster"],
    ).rename(columns={"cell": "cell_id"})
    raw["cell_id"] = raw.cell_id.astype(str)
    if raw.cell_id.duplicated().any():
        raise ValueError(f"{spec.name}: duplicate Baysor cell IDs in cell stats")
    raw["x_px"] = pd.to_numeric(raw.x, errors="raise")
    raw["y_px"] = pd.to_numeric(raw.y, errors="raise")
    raw["x"] = raw.x_px * pixel_size_um
    raw["y"] = raw.y_px * pixel_size_um
    raw["n_transcripts"] = pd.to_numeric(raw.n_transcripts, errors="raise").astype(np.int64)
    raw["area_um2"] = pd.to_numeric(raw.area, errors="raise") * pixel_size_um**2
    return raw.reset_index(drop=True)


def infer_match_radius(gt_cells: pd.DataFrame, quantile: float, multiplier: float) -> float:
    xy = gt_cells[["x", "y"]].to_numpy(float)
    if len(xy) < 2:
        raise ValueError("GT cell count must be at least two")
    distances, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)
    radius = float(np.nanquantile(distances[:, 1], quantile) * multiplier)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"invalid inferred match radius {radius}")
    return radius


def build_candidate_edges(
    pred_cells: pd.DataFrame, gt_cells: pd.DataFrame, radius: float
) -> pd.DataFrame:
    pred_xy = pred_cells[["x", "y"]].to_numpy(float)
    gt_xy = gt_cells[["x", "y"]].to_numpy(float)
    neighbor_lists = cKDTree(gt_xy).query_ball_point(pred_xy, r=radius)
    pred_index: List[int] = []
    gt_index: List[int] = []
    distances: List[float] = []
    for pred_i, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        gt_i = np.asarray(neighbors, dtype=int)
        distance = np.linalg.norm(gt_xy[gt_i] - pred_xy[pred_i], axis=1)
        pred_index.extend([pred_i] * len(gt_i))
        gt_index.extend(gt_i.tolist())
        distances.extend(distance.tolist())
    return pd.DataFrame({
        "pred_index": pred_index, "gt_index": gt_index, "distance": distances,
    })


def connected_components(edges: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    if edges.empty:
        return []
    pred_arr = edges.pred_index.to_numpy(int)
    gt_arr = edges.gt_index.to_numpy(int)
    pred_to_edges: Dict[int, List[int]] = {}
    gt_to_edges: Dict[int, List[int]] = {}
    for edge_i, (pred_i, gt_i) in enumerate(zip(pred_arr, gt_arr)):
        pred_to_edges.setdefault(int(pred_i), []).append(edge_i)
        gt_to_edges.setdefault(int(gt_i), []).append(edge_i)

    visited_pred: set[int] = set()
    visited_gt: set[int] = set()
    visited_edges: set[int] = set()
    components = []
    for start in pred_to_edges:
        if start in visited_pred:
            continue
        stack = [start]
        component_pred: set[int] = set()
        component_gt: set[int] = set()
        component_edges: set[int] = set()
        while stack:
            pred_i = stack.pop()
            if pred_i in visited_pred:
                continue
            visited_pred.add(pred_i)
            component_pred.add(pred_i)
            for edge_i in pred_to_edges.get(pred_i, []):
                if edge_i in visited_edges:
                    continue
                visited_edges.add(edge_i)
                component_edges.add(edge_i)
                gt_i = int(gt_arr[edge_i])
                component_gt.add(gt_i)
                if gt_i not in visited_gt:
                    visited_gt.add(gt_i)
                    for next_edge in gt_to_edges.get(gt_i, []):
                        next_pred = int(pred_arr[next_edge])
                        if next_pred not in visited_pred:
                            stack.append(next_pred)
        components.append((
            np.asarray(sorted(component_pred), dtype=int),
            np.asarray(sorted(component_gt), dtype=int),
            edges.iloc[sorted(component_edges)].copy(),
        ))
    return components


def attach_match_metadata(
    matches: pd.DataFrame, pred_cells: pd.DataFrame, gt_cells: pd.DataFrame
) -> pd.DataFrame:
    if matches.empty:
        return matches
    pred_meta = pred_cells.reset_index(drop=True).reset_index().rename(columns={
        "index": "pred_index", "cell_id": "pred_cell_id", "x": "pred_x", "y": "pred_y",
    })
    gt_meta = gt_cells.reset_index(drop=True).reset_index().rename(columns={
        "index": "gt_index", "cell_id": "gt_cell_id", "x": "gt_x", "y": "gt_y",
    })
    pred_keep = {"pred_index", "pred_cell_id", "pred_x", "pred_y"}
    gt_keep = {"gt_index", "gt_cell_id", "gt_x", "gt_y"}
    pred_meta = pred_meta.rename(columns={
        column: f"pred_{column}" for column in pred_meta.columns
        if column not in pred_keep and not column.startswith("pred_")
    })
    gt_meta = gt_meta.rename(columns={
        column: f"gt_{column}" for column in gt_meta.columns
        if column not in gt_keep and not column.startswith("gt_")
    })
    return (matches.merge(pred_meta, on="pred_index", how="left")
            .merge(gt_meta, on="gt_index", how="left"))


def match_sparse_hungarian(
    pred_cells: pd.DataFrame,
    gt_cells: pd.DataFrame,
    radius: float,
    max_component_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    edges = build_candidate_edges(pred_cells, gt_cells, radius)
    qc: Dict[str, Any] = {
        "matching_strategy": "sparse_hungarian_connected_components",
        "match_radius": float(radius),
        "candidate_edges": int(len(edges)),
        "mean_candidate_edges_per_pred": float(len(edges) / max(len(pred_cells), 1)),
    }
    if edges.empty:
        return pd.DataFrame(columns=["pred_index", "gt_index", "distance"]), qc
    components = connected_components(edges)
    qc.update({
        "candidate_components": len(components),
        "max_component_pred_size": max((len(c[0]) for c in components), default=0),
        "max_component_gt_size": max((len(c[1]) for c in components), default=0),
    })
    records: List[Tuple[int, int, float]] = []
    greedy_fallback = 0
    for pred_indices, gt_indices, component_edges in components:
        if max(len(pred_indices), len(gt_indices)) > max_component_size:
            greedy_fallback += 1
            used_pred: set[int] = set()
            used_gt: set[int] = set()
            ordered = component_edges.sort_values(
                ["distance", "pred_index", "gt_index"], kind="mergesort")
            for row in ordered.itertuples(index=False):
                pred_i, gt_i = int(row.pred_index), int(row.gt_index)
                if pred_i in used_pred or gt_i in used_gt:
                    continue
                used_pred.add(pred_i)
                used_gt.add(gt_i)
                records.append((pred_i, gt_i, float(row.distance)))
            continue

        pred_position = {value: index for index, value in enumerate(pred_indices.tolist())}
        gt_position = {value: index for index, value in enumerate(gt_indices.tolist())}
        large_cost = float(max(radius * 1e4, 1e9))
        cost = np.full((len(pred_indices), len(gt_indices)), large_cost, dtype=np.float64)
        for row in component_edges.itertuples(index=False):
            cost[pred_position[int(row.pred_index)], gt_position[int(row.gt_index)]] = row.distance
        row_indices, col_indices = linear_sum_assignment(cost)
        for row_i, col_i in zip(row_indices.tolist(), col_indices.tolist()):
            distance = float(cost[row_i, col_i])
            if distance <= radius + EPS:
                records.append((int(pred_indices[row_i]), int(gt_indices[col_i]), distance))
    qc["large_components_greedy_fallback"] = greedy_fallback
    matches = pd.DataFrame(records, columns=["pred_index", "gt_index", "distance"])
    return attach_match_metadata(matches, pred_cells, gt_cells), qc


def detection_metrics(matches: pd.DataFrame, n_pred: int, n_gt: int) -> Dict[str, Any]:
    matched = len(matches)
    precision = matched / max(n_pred, 1)
    recall = matched / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    return {
        "pred_cell_count": n_pred,
        "gt_cell_count": n_gt,
        "matched_cell_count": matched,
        "unmatched_pred_count": n_pred - matched,
        "unmatched_gt_count": n_gt - matched,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "matched_fraction_gt": recall,
        "matched_fraction_pred": precision,
        "cell_count_difference": n_pred - n_gt,
        "cell_count_ratio": n_pred / max(n_gt, EPS),
    }


def spatial_shift_metrics(matches: pd.DataFrame) -> Dict[str, float]:
    keys = [
        "mean_centroid_shift", "median_centroid_shift", "p95_centroid_shift",
        "max_centroid_shift",
    ]
    if matches.empty:
        return {key: np.nan for key in keys}
    distance = pd.to_numeric(matches.distance, errors="coerce").dropna().to_numpy(float)
    if not len(distance):
        return {key: np.nan for key in keys}
    return {
        "mean_centroid_shift": float(np.mean(distance)),
        "median_centroid_shift": float(np.median(distance)),
        "p95_centroid_shift": float(np.quantile(distance, 0.95)),
        "max_centroid_shift": float(np.max(distance)),
    }


def safe_correlation(function: Callable, x: Sequence[float], y: Sequence[float]) -> float:
    first = np.asarray(x, float)
    second = np.asarray(y, float)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if len(first) < 3 or np.std(first) <= EPS or np.std(second) <= EPS:
        return np.nan
    return float(function(first, second)[0])


def transcript_count_metrics(matches: pd.DataFrame) -> Dict[str, float]:
    prefix = "matched_pair_n_transcripts_vs_total_counts"
    pred = pd.to_numeric(matches.get("pred_n_transcripts"), errors="coerce")
    gt = pd.to_numeric(matches.get("gt_total_counts"), errors="coerce")
    valid = pred.notna() & gt.notna()
    pred_values = pred[valid].to_numpy(float)
    gt_values = gt[valid].to_numpy(float)
    result = {
        f"{prefix}_valid_n": int(valid.sum()),
        f"{prefix}_pearson": safe_correlation(pearsonr, pred_values, gt_values),
        f"{prefix}_spearman": safe_correlation(spearmanr, pred_values, gt_values),
        f"{prefix}_mae": float(np.mean(np.abs(pred_values - gt_values))) if len(pred_values) else np.nan,
        f"{prefix}_rmse": (
            float(np.sqrt(np.mean((pred_values - gt_values) ** 2)))
            if len(pred_values) else np.nan
        ),
    }
    result.update({
        "matched_cell_transcript_count_pearson": result[f"{prefix}_pearson"],
        "matched_cell_transcript_count_spearman": result[f"{prefix}_spearman"],
        "matched_cell_transcript_count_mae": result[f"{prefix}_mae"],
        "matched_cell_transcript_count_rmse": result[f"{prefix}_rmse"],
    })
    return result


def gene_names(path: str, batch_size: int) -> List[str]:
    names: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["target"], batch_size=batch_size):
        names.update(str(value) for value in pc.unique(batch.column(0)).to_pylist())
    return sorted(names)


def add_sparse_counts(
    matrix: csr_matrix,
    pair_indices: np.ndarray,
    genes: pd.Series,
    categories: Sequence[str],
) -> csr_matrix:
    if not len(pair_indices):
        return matrix
    gene_indices = pd.Categorical(genes.astype(str), categories=categories).codes
    valid = (pair_indices >= 0) & (gene_indices >= 0)
    if not np.any(valid):
        return matrix
    linear = pair_indices[valid].astype(np.int64) * len(categories) + gene_indices[valid]
    unique, count = np.unique(linear, return_counts=True)
    rows = unique // len(categories)
    cols = unique % len(categories)
    update = csr_matrix(
        (count.astype(np.int64), (rows, cols)), shape=matrix.shape, dtype=np.int64)
    return matrix + update


class TranscriptAccumulator:
    def __init__(
        self,
        matches: pd.DataFrame,
        genes: Sequence[str],
        fov_cell_to_pair: Dict[int, int],
        pred_cell_to_pair: Dict[str, int],
    ) -> None:
        self.genes = genes
        self.fov_cell_to_pair = fov_cell_to_pair
        self.pred_cell_to_pair = pred_cell_to_pair
        shape = (len(matches), len(genes))
        self.pred_matrix = csr_matrix(shape, dtype=np.int64)
        self.gt_matrix = csr_matrix(shape, dtype=np.int64)
        self.gt_rows = 0
        self.gt_assigned = 0
        self.seg_rows = 0
        self.pred_assigned = 0
        self.overlap = 0
        self.in_matched = 0
        self.correct = 0
        self.alignment_mismatches = 0

    @staticmethod
    def encoded_gt_key(frame: pd.DataFrame) -> np.ndarray:
        fov = pd.to_numeric(frame.fov, errors="raise").to_numpy(np.int64)
        cell_id = pd.to_numeric(frame.cell_ID, errors="raise").to_numpy(np.int64)
        return fov * 1_000_000 + cell_id

    def process_raw_gt(self, frame: pd.DataFrame) -> None:
        self.gt_rows += len(frame)
        gt_cell_number = pd.to_numeric(frame.cell_ID, errors="raise").to_numpy(np.int64)
        self.gt_assigned += int(np.sum(gt_cell_number > 0))
        keys = self.encoded_gt_key(frame)
        pair = pd.Series(keys).map(self.fov_cell_to_pair)
        valid = pair.notna().to_numpy()
        if np.any(valid):
            pair_indices = pair[valid].to_numpy(np.int64)
            self.gt_matrix = add_sparse_counts(
                self.gt_matrix, pair_indices, frame.loc[valid, "target"], self.genes)

    @staticmethod
    def normalized_pred_cells(series: pd.Series) -> pd.Series:
        cells = series.astype(str).str.strip()
        numeric = cells.str.fullmatch(r"\d+") & cells.ne("0")
        cells.loc[numeric] = "cell_" + cells.loc[numeric]
        return cells

    def process_aligned(self, segmentation: pd.DataFrame, official: pd.DataFrame) -> None:
        if len(segmentation) != len(official):
            raise ValueError("internal row alignment length mismatch")
        self.seg_rows += len(segmentation)
        gene_equal = (
            segmentation.gene.astype(str).to_numpy() == official.target.astype(str).to_numpy())
        x_equal = np.isclose(
            pd.to_numeric(segmentation.x, errors="raise"),
            pd.to_numeric(official.x_global_px, errors="raise"), rtol=0, atol=0.11)
        y_equal = np.isclose(
            pd.to_numeric(segmentation.y, errors="raise"),
            pd.to_numeric(official.y_global_px, errors="raise"), rtol=0, atol=0.11)
        mismatch = ~(gene_equal & x_equal & y_equal)
        self.alignment_mismatches += int(np.sum(mismatch))
        if np.any(mismatch):
            first = int(np.flatnonzero(mismatch)[0])
            raise ValueError(
                "Baysor/output transcript order mismatch at aligned row "
                f"{self.seg_rows - len(segmentation) + first:,}: "
                f"pred=({segmentation.iloc[first].gene}, {segmentation.iloc[first].x}, "
                f"{segmentation.iloc[first].y}), official=({official.iloc[first].target}, "
                f"{official.iloc[first].x_global_px}, {official.iloc[first].y_global_px})")

        cells = self.normalized_pred_cells(segmentation.cell)
        noise = pd.to_numeric(segmentation.is_noise, errors="coerce").fillna(0).to_numpy() != 0
        assigned = (~cells.isin(["", "0", "-1", "nan", "None"]).to_numpy()) & (~noise)
        self.pred_assigned += int(np.sum(assigned))
        official_cell = pd.to_numeric(official.cell_ID, errors="raise").to_numpy(np.int64)
        self.overlap += int(np.sum(assigned & (official_cell > 0)))

        pred_pair = cells.map(self.pred_cell_to_pair)
        matched_mask = assigned & pred_pair.notna().to_numpy()
        self.in_matched += int(np.sum(matched_mask))
        if np.any(matched_mask):
            pred_pair_indices = pred_pair[matched_mask].to_numpy(np.int64)
            gt_keys = self.encoded_gt_key(official)
            official_pair = pd.Series(gt_keys).map(self.fov_cell_to_pair)
            official_pair_values = official_pair[matched_mask].fillna(-1).to_numpy(np.int64)
            self.correct += int(np.sum(pred_pair_indices == official_pair_values))
            self.pred_matrix = add_sparse_counts(
                self.pred_matrix,
                pred_pair_indices,
                segmentation.loc[matched_mask, "gene"],
                self.genes,
            )


class FilteredParquetReader:
    """Supply Baysor-included official rows while processing every raw GT row."""

    def __init__(
        self,
        path: str,
        batch_size: int,
        raw_callback: Callable[[pd.DataFrame], None],
    ) -> None:
        columns = ["fov", "cell_ID", "x_global_px", "y_global_px", "target"]
        parquet = pq.ParquetFile(path)
        self.iterator = iter(parquet.iter_batches(columns=columns, batch_size=batch_size))
        self.raw_callback = raw_callback
        self.queue: Deque[pd.DataFrame] = deque()
        self.available = 0
        self.exhausted = False
        self.raw_rows_expected = parquet.metadata.num_rows

    def load_one(self) -> None:
        try:
            frame = next(self.iterator).to_pandas()
        except StopIteration:
            self.exhausted = True
            return
        self.raw_callback(frame)
        excluded = frame.target.astype(str).str.startswith(EXCLUDED_GENE_PREFIXES)
        included = frame.loc[~excluded].reset_index(drop=True)
        if len(included):
            self.queue.append(included)
            self.available += len(included)

    def take(self, count: int) -> pd.DataFrame:
        while self.available < count and not self.exhausted:
            self.load_one()
        if self.available < count:
            raise ValueError(
                f"segmentation.csv has more included rows than official input: need {count}, "
                f"only {self.available} remain")
        pieces = []
        remaining = count
        while remaining:
            first = self.queue[0]
            if len(first) <= remaining:
                pieces.append(self.queue.popleft())
                self.available -= len(first)
                remaining -= len(first)
            else:
                pieces.append(first.iloc[:remaining].copy())
                self.queue[0] = first.iloc[remaining:].reset_index(drop=True)
                self.available -= remaining
                remaining = 0
        return pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]

    def finish(self) -> None:
        while not self.exhausted:
            self.load_one()
        if self.available:
            raise ValueError(
                f"official input has {self.available:,} included rows absent from segmentation.csv")


def gene_vector_metrics(pred_matrix: csr_matrix, gt_matrix: csr_matrix) -> Dict[str, Any]:
    cosines: List[float] = []
    js_distances: List[float] = []
    correlations: List[float] = []
    valid_pairs = 0
    for row_index in range(pred_matrix.shape[0]):
        pred_row = pred_matrix.getrow(row_index)
        gt_row = gt_matrix.getrow(row_index)
        if pred_row.nnz == 0 or gt_row.nnz == 0:
            continue
        union = np.union1d(pred_row.indices, gt_row.indices)
        if len(union) < 3:
            continue
        pred_values = np.zeros(len(union), dtype=float)
        gt_values = np.zeros(len(union), dtype=float)
        pred_values[np.searchsorted(union, pred_row.indices)] = pred_row.data
        gt_values[np.searchsorted(union, gt_row.indices)] = gt_row.data
        pred_norm = np.linalg.norm(pred_values)
        gt_norm = np.linalg.norm(gt_values)
        if pred_norm > 0 and gt_norm > 0:
            cosines.append(float(np.dot(pred_values, gt_values) / (pred_norm * gt_norm)))
        pred_probability = pred_values / max(pred_values.sum(), EPS)
        gt_probability = gt_values / max(gt_values.sum(), EPS)
        midpoint = (pred_probability + gt_probability) / 2
        divergence = (
            0.5 * np.sum(pred_probability * np.log(pred_probability / (midpoint + EPS) + EPS))
            + 0.5 * np.sum(gt_probability * np.log(gt_probability / (midpoint + EPS) + EPS))
        )
        js_distances.append(float(np.sqrt(max(divergence, 0))))
        if np.std(pred_values) > EPS and np.std(gt_values) > EPS:
            correlations.append(float(pearsonr(pred_values, gt_values)[0]))
        valid_pairs += 1
    return {
        "matched_cell_gene_vector_mean_cosine": (
            float(np.mean(cosines)) if cosines else np.nan),
        "matched_cell_gene_vector_mean_js_distance": (
            float(np.mean(js_distances)) if js_distances else np.nan),
        "matched_cell_gene_vector_mean_pearson": (
            float(np.mean(correlations)) if correlations else np.nan),
        "matched_cell_gene_vector_valid_pairs": valid_pairs,
    }


def compute_transcript_metrics(
    spec: DatasetSpec,
    matches: pd.DataFrame,
    pred_cells: pd.DataFrame,
    batch_size: int,
    quiet: bool,
) -> Dict[str, Any]:
    info(f"[{spec.name}] scanning transcript genes ...", quiet)
    genes = gene_names(spec.transcript_path, batch_size)
    info(f"[{spec.name}] transcript gene universe: {len(genes):,}", quiet)

    fov_cell_to_pair: Dict[int, int] = {}
    pred_cell_to_pair: Dict[str, int] = {}
    for pair_index, row in enumerate(matches.itertuples(index=False)):
        gt_fov = int(row.gt_fov)
        gt_cell_id = int(row.gt_cell_ID)
        fov_cell_to_pair[gt_fov * 1_000_000 + gt_cell_id] = pair_index
        pred_cell_to_pair[str(row.pred_cell_id)] = pair_index

    accumulator = TranscriptAccumulator(
        matches, genes, fov_cell_to_pair, pred_cell_to_pair)
    official = FilteredParquetReader(
        spec.transcript_path, batch_size, accumulator.process_raw_gt)
    read_rows = 0
    for segmentation in pd.read_csv(
        spec.segmentation_path,
        usecols=["cell", "gene", "x", "y", "is_noise"],
        chunksize=batch_size,
    ):
        official_rows = official.take(len(segmentation))
        accumulator.process_aligned(segmentation, official_rows)
        read_rows += len(segmentation)
        info(
            f"[{spec.name}] aligned {read_rows:,} Baysor rows; "
            f"assigned={accumulator.pred_assigned:,}", quiet)
    official.finish()

    expected_stats_count = int(pred_cells.n_transcripts.sum())
    if accumulator.pred_assigned != expected_stats_count:
        raise ValueError(
            f"{spec.name}: assigned transcript count mismatch: segmentation="
            f"{accumulator.pred_assigned:,}, cell_stats={expected_stats_count:,}")
    if accumulator.gt_rows != official.raw_rows_expected:
        raise ValueError(
            f"{spec.name}: read {accumulator.gt_rows:,} GT rows, expected "
            f"{official.raw_rows_expected:,}")

    result: Dict[str, Any] = {
        "transcript_metrics_available": True,
        "transcript_metrics_error": "",
        "pred_transcript_rows": accumulator.pred_assigned,
        "gt_transcript_rows": float(accumulator.gt_rows),
        "pred_transcript_assigned_n": accumulator.pred_assigned,
        "pred_transcript_assignment_rate": (
            accumulator.pred_assigned / max(accumulator.gt_rows, 1)),
        "pred_transcript_has_gene_col": True,
        "gt_transcript_assigned_n": accumulator.gt_assigned,
        "gt_transcript_assignment_rate": (
            accumulator.gt_assigned / max(accumulator.gt_rows, 1)),
        "gt_transcript_has_gene_col": True,
        "transcript_matched_cell_pairs_n": len(matches),
        "pred_transcripts_in_matched_cells_n": accumulator.in_matched,
        "gt_transcripts_in_matched_cells_n": int(accumulator.gt_matrix.sum()),
        "transcript_id_overlap_n": accumulator.overlap,
        "transcript_assignment_accuracy_via_matched_cells": (
            accumulator.correct / max(accumulator.in_matched, 1)),
        "transcript_assignment_in_matched_n": accumulator.in_matched,
        "transcript_assignment_correct_n": accumulator.correct,
        "baysor_segmentation_rows": accumulator.seg_rows,
        "baysor_official_alignment_mismatches": accumulator.alignment_mismatches,
        "excluded_official_transcript_rows": accumulator.gt_rows - accumulator.seg_rows,
    }
    result.update(gene_vector_metrics(accumulator.pred_matrix, accumulator.gt_matrix))
    return result


def not_available_transcript_metrics(matches: pd.DataFrame) -> Dict[str, Any]:
    keys = [key for _, key in REQUESTED_METRICS if (
        "transcript" in key or "gene_vector" in key)]
    result = {key: np.nan for key in keys}
    result.update({
        "transcript_metrics_available": False,
        "transcript_metrics_error": "disabled by --skip-transcript-metrics",
        "transcript_matched_cell_pairs_n": len(matches),
    })
    return result


def build_selected_row(dataset: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {"method": "Baysor", "dataset": dataset}
    for display_name, metric_name in REQUESTED_METRICS:
        row[display_name] = metrics.get(metric_name, np.nan)
    return row


def save_outputs(
    spec: DatasetSpec,
    output_dir: Path,
    metrics: Dict[str, Any],
    selected: Dict[str, Any],
    matches: pd.DataFrame,
    pred_cells: pd.DataFrame,
    gt_cells: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"baysor_cosmx_{spec.name}_sparse_hungarian_qc_metrics"
    pd.DataFrame([metrics]).to_csv(output_dir / f"{prefix}_full.csv", index=False)
    (output_dir / f"{prefix}_full.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([selected]).to_csv(output_dir / f"{prefix}_selected.csv", index=False)
    (output_dir / f"{prefix}_selected.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    matches.to_csv(output_dir / f"baysor_cosmx_{spec.name}_matched_cells.csv", index=False)
    pred_cells.to_csv(output_dir / f"baysor_cosmx_{spec.name}_pred_cell_level.csv", index=False)
    gt_cells.to_csv(output_dir / f"baysor_cosmx_{spec.name}_gt_cell_level.csv", index=False)
    (output_dir / f"baysor_cosmx_{spec.name}_evaluation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def evaluate_dataset(spec: DatasetSpec, args: argparse.Namespace) -> Dict[str, Any]:
    validate_spec(spec)
    info(f"\n{'=' * 78}\nBaysor x CosMx: {spec.name}\n{'=' * 78}", args.quiet)
    gt_cells = load_gt_cells(spec, args.pixel_size_um)
    pred_cells = load_pred_cells(spec, args.pixel_size_um)
    info(
        f"[{spec.name}] GT={len(gt_cells):,}, Pred={len(pred_cells):,}; "
        f"GT x=[{gt_cells.x.min():.1f},{gt_cells.x.max():.1f}] um, "
        f"Pred x=[{pred_cells.x.min():.1f},{pred_cells.x.max():.1f}] um", args.quiet)

    radius = (
        float(args.match_radius) if args.match_radius is not None
        else infer_match_radius(gt_cells, args.radius_quantile, args.radius_multiplier)
    )
    info(f"[{spec.name}] match radius={radius:.6f} um", args.quiet)
    matches, matching_qc = match_sparse_hungarian(
        pred_cells, gt_cells, radius, args.max_component_size)
    if matches.pred_cell_id.duplicated().any() or matches.gt_cell_id.duplicated().any():
        raise ValueError(f"{spec.name}: Hungarian result is not one-to-one")
    info(f"[{spec.name}] matched cells={len(matches):,}", args.quiet)

    metrics: Dict[str, Any] = {
        "method": "Baysor",
        "dataset": spec.name,
        "script_version": VERSION,
        "pixel_size_um": args.pixel_size_um,
        "gt_path": spec.metadata_path,
        "gt_transcripts_path": spec.transcript_path,
        "baysor_stats_path": str(spec.stats_path),
        "baysor_segmentation_path": str(spec.segmentation_path),
        "evaluated_fovs": ",".join(map(str, spec.fovs)),
        "uses_platform_prior": False,
        "gt_raw_rows": len(gt_cells),
        "gt_n_cells": len(gt_cells),
        "gt_mean_rows_per_cell": 1.0,
        "pred_raw_rows": len(pred_cells),
        "pred_n_cells": len(pred_cells),
        "pred_mean_rows_per_cell": 1.0,
    }
    metrics.update(matching_qc)
    metrics.update(detection_metrics(matches, len(pred_cells), len(gt_cells)))
    metrics.update(spatial_shift_metrics(matches))
    metrics.update(transcript_count_metrics(matches))
    metrics.update({
        "cell_type_accuracy": np.nan,
        "cell_type_macro_f1": np.nan,
        "cell_type_weighted_f1": np.nan,
        "cell_type_balanced_accuracy": np.nan,
        "ARI_matched_type_labels": np.nan,
        "NMI_matched_type_labels": np.nan,
        "cell_type_purity_legacy_matched_accuracy": np.nan,
        "matched_type_valid_n": 0,
        "knn_neighbor_type_pearson_gt_graph": np.nan,
        "knn_neighbor_type_mean_js_distance_gt_graph": np.nan,
        "knn_neighbor_type_mean_js_divergence_gt_graph": np.nan,
        "knn_neighbor_metric_valid_n": 0,
        "knn_metric_matched_only": True,
        "knn_metric_low_recall_warning": (
            metrics["detection_recall"] < args.low_recall_warning_threshold),
        "knn_metric_recall_threshold": args.low_recall_warning_threshold,
    })
    if metrics["knn_metric_low_recall_warning"]:
        warnings.warn(
            f"{spec.name}: detection recall={metrics['detection_recall']:.3f} < "
            f"{args.low_recall_warning_threshold:.3f}")

    if args.skip_transcript_metrics:
        metrics.update(not_available_transcript_metrics(matches))
    else:
        metrics.update(compute_transcript_metrics(
            spec, matches, pred_cells, args.chunk_size, args.quiet))

    selected = build_selected_row(spec.name, metrics)
    config = {
        "script_version": VERSION,
        "dataset": asdict(spec),
        "pixel_size_um": args.pixel_size_um,
        "match_radius": radius,
        "radius_quantile": args.radius_quantile,
        "radius_multiplier": args.radius_multiplier,
        "max_component_size": args.max_component_size,
        "chunk_size": args.chunk_size,
        "excluded_gene_prefixes": EXCLUDED_GENE_PREFIXES,
        "skip_transcript_metrics": args.skip_transcript_metrics,
        "note": (
            "GT is restricted to the exact FOV subset supplied to Baysor. "
            "Baysor prior.type=none; no official segmentation was used during inference."),
    }
    save_outputs(
        spec, Path(args.output_dir), metrics, selected, matches,
        pred_cells, gt_cells, config)
    info(
        f"[{spec.name}] precision={metrics['detection_precision']:.4f}, "
        f"recall={metrics['detection_recall']:.4f}, "
        f"F1={metrics['detection_f1']:.4f}", args.quiet)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["all", *DATASETS], default="all",
        help="evaluate both datasets or one named dataset")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--pixel-size-um", type=float, default=PIXEL_SIZE_UM)
    parser.add_argument("--match-radius", type=float)
    parser.add_argument("--radius-quantile", type=float, default=0.95)
    parser.add_argument("--radius-multiplier", type=float, default=1.25)
    parser.add_argument("--max-component-size", type=int, default=3000)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--low-recall-warning-threshold", type=float, default=0.70)
    parser.add_argument("--skip-transcript-metrics", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        validate_spec(DATASETS[name])
    if args.check_only:
        for name in names:
            spec = DATASETS[name]
            rows = pq.ParquetFile(spec.transcript_path).metadata.num_rows
            cells = len(pd.read_csv(spec.stats_path, usecols=["cell"]))
            print(
                f"[OK] {name}: FOVs={len(spec.fovs)}, official_tx={rows:,}, "
                f"Baysor_cells={cells:,}, prior.type=none")
        return

    selected_rows = [evaluate_dataset(DATASETS[name], args) for name in names]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "baysor_cosmx_evaluation_summary.csv"
    if args.dataset == "all":
        pd.DataFrame(selected_rows).to_csv(combined_path, index=False)
        info(f"\n[DONE] combined summary -> {combined_path}", args.quiet)
    else:
        # Refresh the combined summary from every completed per-dataset result.
        rows = []
        for name in DATASETS:
            path = output_dir / f"baysor_cosmx_{name}_sparse_hungarian_qc_metrics_selected.csv"
            if path.is_file():
                rows.append(pd.read_csv(path))
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(combined_path, index=False)


if __name__ == "__main__":
    main()
