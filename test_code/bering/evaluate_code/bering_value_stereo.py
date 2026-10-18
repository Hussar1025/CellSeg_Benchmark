#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate all Bering Stereo-seq segmentations with the UCS metric protocol.

This is the Bering counterpart of
``code/ucs/stereo/evaluate_code/ucs_value_stereo_1.py``.  Ground-truth cells
are reconstructed from the CellBin ``cell_id`` values in the exact prepared
Parquet used by each Bering run.  Prepared coordinates are already in um and
each row represents one molecule (the original GEM ``MIDCounts`` expansion),
so a row mean is the UCS script's MIDCounts-weighted centroid.

Bering exports transcript assignments rather than a raster mask.  Therefore a
predicted cell centroid is the mean coordinate of its assigned transcripts and
its count is the number of assigned transcripts.  Cell matching follows the
reference exactly: nearest GT within 15 um, followed by retaining only the
nearest prediction for each GT cell.

The runs are reference-assisted and use CellBin cells as coarse seeds.  As in
the UCS reference evaluator, ``assign_overlap`` and ``assign_accuracy`` are
left NaN because comparing the result with the seed source would be circular;
``vec_js_dist`` and ``vec_pearson`` are also left NaN to preserve that script's
19-column behavior.  ``vec_cosine`` is evaluated on a deterministic sample of
at most 2,000 matched pairs.

Examples
--------
Evaluate every completed directory below the default Stereo output root::

    python bering_value_stereo.py --all

Evaluate one result, or only validate inputs::

    python bering_value_stereo.py --dataset E14.5_E1S3_Dorsal_Midbrain
    python bering_value_stereo.py --all --check-only
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


VERSION = "2026.09.23"
DEFAULT_OUTPUTS = Path("/data/qiuyijia/bering/stereo_outputs")
DEFAULT_EVAL_DIR = Path("/data/qiuyijia/eval/bering")
DEFAULT_RADIUS_UM = 15.0
DEFAULT_BATCH_SIZE = 500_000
DEFAULT_VECTOR_SAMPLE = 2_000

METRIC_COLUMNS = (
    "gt_cells", "pred_cells", "cell_ratio", "matched", "precision",
    "recall", "f1", "loc_mean_um", "loc_median_um", "loc_p95_um",
    "count_pearson", "count_spearman", "count_mae", "count_rmse",
    "vec_cosine", "vec_js_dist", "vec_pearson", "assign_overlap",
    "assign_accuracy",
)

UNASSIGNED_VALUES = {"", "0", "-1", "nan", "none", "null", "unassigned"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    result_dir: Path
    prepared_path: Path
    summary: Mapping[str, Any]
    config: Mapping[str, Any]

    @property
    def assignments_path(self) -> Path:
        return self.result_dir / "transcript_assignments.parquet"


def info(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Bering Stereo-seq outputs with UCS Stereo metrics."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all", action="store_true",
        help="evaluate every complete result directory (default)",
    )
    selection.add_argument(
        "--dataset", action="append", metavar="NAME",
        help="evaluate one dataset directory; repeat for multiple datasets",
    )
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--match-radius-um", type=float, default=DEFAULT_RADIUS_UM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--vector-sample-size", type=int, default=DEFAULT_VECTOR_SAMPLE,
        help="maximum matched pairs used for vec_cosine; 0 skips it",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate all paths, schemas and run-completion metadata, then exit",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.match_radius_um <= 0:
        parser.error("--match-radius-um must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.vector_sample_size < 0:
        parser.error("--vector-sample-size cannot be negative")
    return args


def read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def discover_datasets(outputs_dir: Path) -> Dict[str, DatasetSpec]:
    if not outputs_dir.is_dir():
        raise FileNotFoundError(outputs_dir)
    found: Dict[str, DatasetSpec] = {}
    for result_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        config_path = result_dir / "config_used.json"
        summary_path = result_dir / "run_summary.json"
        assignments_path = result_dir / "transcript_assignments.parquet"
        present = [path.is_file() for path in (config_path, summary_path, assignments_path)]
        if not any(present):
            continue
        if not all(present):
            missing = [
                path.name for path in (config_path, summary_path, assignments_path)
                if not path.is_file()
            ]
            warnings.warn(f"skip incomplete result {result_dir}: missing {missing}")
            continue
        config = read_json(config_path)
        summary = read_json(summary_path)
        prepared_value = config.get("transcripts")
        if not prepared_value:
            raise ValueError(f"{config_path}: missing 'transcripts'")
        found[result_dir.name] = DatasetSpec(
            name=result_dir.name,
            result_dir=result_dir,
            prepared_path=Path(str(prepared_value)),
            summary=summary,
            config=config,
        )
    if not found:
        raise ValueError(f"no complete Bering results found below {outputs_dir}")
    return found


def select_datasets(
    discovered: Mapping[str, DatasetSpec], requested: Optional[Sequence[str]]
) -> List[DatasetSpec]:
    if not requested:
        return list(discovered.values())
    unknown = [name for name in requested if name not in discovered]
    if unknown:
        available = ", ".join(discovered)
        raise ValueError(f"unknown dataset(s) {unknown}; available: {available}")
    # Preserve request order while preventing accidental duplicate evaluation.
    return [discovered[name] for name in dict.fromkeys(requested)]


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def validate_dataset(spec: DatasetSpec) -> Dict[str, int]:
    if not spec.prepared_path.is_file():
        raise FileNotFoundError(spec.prepared_path)
    if not spec.assignments_path.is_file():
        raise FileNotFoundError(spec.assignments_path)

    prepared_required = {
        "transcript_id", "cell_id", "feature_name", "x_location", "y_location"
    }
    output_required = {"transcript_id", "cell_id", "gene", "x", "y"}
    missing_prepared = prepared_required - parquet_columns(spec.prepared_path)
    missing_output = output_required - parquet_columns(spec.assignments_path)
    if missing_prepared:
        raise ValueError(f"{spec.prepared_path} lacks {sorted(missing_prepared)}")
    if missing_output:
        raise ValueError(f"{spec.assignments_path} lacks {sorted(missing_output)}")

    prepared_rows = pq.ParquetFile(spec.prepared_path).metadata.num_rows
    output_rows = pq.ParquetFile(spec.assignments_path).metadata.num_rows
    expected_input = int(spec.summary.get("n_input_transcripts", prepared_rows))
    expected_assigned = int(spec.summary.get("n_assigned", output_rows))
    if not bool(spec.summary.get("prediction_complete", False)):
        raise ValueError(f"{spec.name}: run_summary does not mark prediction complete")
    if prepared_rows != expected_input:
        raise ValueError(
            f"{spec.name}: prepared rows {prepared_rows:,} != summary input "
            f"{expected_input:,}"
        )
    if output_rows != expected_assigned:
        raise ValueError(
            f"{spec.name}: output rows {output_rows:,} != summary assigned "
            f"{expected_assigned:,}"
        )
    return {"prepared_rows": prepared_rows, "output_rows": output_rows}


def valid_cell_mask(series: pd.Series) -> np.ndarray:
    text = series.astype("string")
    cleaned = text.str.strip().str.lower()
    return (text.notna() & ~cleaned.isin(UNASSIGNED_VALUES)).to_numpy(bool)


def combine_cell_partials(parts: List[pd.DataFrame], id_column: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=[id_column, "x", "y", "count"])
    partial = pd.concat(parts, ignore_index=True)
    combined = partial.groupby(id_column, sort=True, as_index=False).agg(
        x_sum=("x_sum", "sum"),
        y_sum=("y_sum", "sum"),
        count=("count", "sum"),
    )
    combined["x"] = combined.x_sum / combined["count"]
    combined["y"] = combined.y_sum / combined["count"]
    return combined[[id_column, "x", "y", "count"]]


def load_gt_cells(spec: DatasetSpec, batch_size: int, quiet: bool) -> pd.DataFrame:
    info("  重建 CellBin GT 质心和计数...", quiet)
    parquet = pq.ParquetFile(spec.prepared_path)
    parts: List[pd.DataFrame] = []
    columns = ["cell_id", "x_location", "y_location"]
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        keep = valid_cell_mask(frame.cell_id)
        if not np.any(keep):
            continue
        selected = frame.loc[keep].copy()
        selected["cell_id"] = selected.cell_id.astype(str)
        selected["x_location"] = pd.to_numeric(
            selected.x_location, errors="raise")
        selected["y_location"] = pd.to_numeric(
            selected.y_location, errors="raise")
        grouped = selected.groupby("cell_id", sort=False, as_index=False).agg(
            x_sum=("x_location", "sum"),
            y_sum=("y_location", "sum"),
            count=("cell_id", "size"),
        )
        parts.append(grouped)
    cells = combine_cell_partials(parts, "cell_id").rename(
        columns={"count": "gt_count"})
    if cells.empty:
        raise ValueError(f"{spec.name}: no assigned CellBin GT molecules")
    info(
        f"    GT cells={len(cells):,}, assigned molecules={cells.gt_count.sum():,}",
        quiet,
    )
    return cells


def unique_batch_rows(
    frame: pd.DataFrame, seen: np.ndarray, dataset_name: str
) -> Tuple[pd.DataFrame, int]:
    ids = pd.to_numeric(frame.transcript_id, errors="raise").to_numpy(np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= len(seen)):
        raise ValueError(
            f"{dataset_name}: transcript_id outside [0, {len(seen) - 1}]"
        )
    local_first = ~pd.Series(ids, copy=False).duplicated(keep="first").to_numpy()
    candidates = np.flatnonzero(local_first)
    globally_new = ~seen[ids[candidates]]
    keep_positions = candidates[globally_new]
    seen[ids[keep_positions]] = True
    duplicate_count = len(frame) - len(keep_positions)
    return frame.iloc[keep_positions].copy(), duplicate_count


def load_pred_cells(
    spec: DatasetSpec, expected_input_rows: int, batch_size: int, quiet: bool
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    info("  汇总 Bering 预测细胞（按 transcript_id 去重）...", quiet)
    parquet = pq.ParquetFile(spec.assignments_path)
    seen = np.zeros(expected_input_rows, dtype=bool)
    parts: List[pd.DataFrame] = []
    raw_rows = 0
    duplicates = 0
    columns = ["transcript_id", "cell_id", "x", "y"]
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        raw_rows += len(frame)
        frame, duplicate_count = unique_batch_rows(frame, seen, spec.name)
        duplicates += duplicate_count
        keep = valid_cell_mask(frame.cell_id)
        if not np.any(keep):
            continue
        selected = frame.loc[keep].copy()
        selected["cell_id"] = selected.cell_id.astype(str)
        selected["x"] = pd.to_numeric(selected.x, errors="raise")
        selected["y"] = pd.to_numeric(selected.y, errors="raise")
        grouped = selected.groupby("cell_id", sort=False, as_index=False).agg(
            x_sum=("x", "sum"), y_sum=("y", "sum"), count=("cell_id", "size")
        )
        parts.append(grouped)
    cells = combine_cell_partials(parts, "cell_id").rename(
        columns={"count": "pred_count"})
    if cells.empty:
        raise ValueError(f"{spec.name}: no valid predicted cells")
    diagnostics = {
        "output_assignment_rows": int(raw_rows),
        "unique_assigned_transcripts": int(seen.sum()),
        "duplicate_assignment_rows_removed": int(duplicates),
    }
    info(
        f"    pred cells={len(cells):,}, unique assigned={seen.sum():,}, "
        f"duplicates removed={duplicates:,}",
        quiet,
    )
    return cells, diagnostics


def match_cells(
    pred_cells: pd.DataFrame, gt_cells: pd.DataFrame, radius_um: float
) -> pd.DataFrame:
    pred_xy = pred_cells[["x", "y"]].to_numpy(float)
    gt_xy = gt_cells[["x", "y"]].to_numpy(float)
    distances, gt_indices = cKDTree(gt_xy).query(
        pred_xy, k=1, distance_upper_bound=radius_um
    )
    candidate_pred = np.flatnonzero(np.isfinite(distances) & (distances < radius_um))
    best: Dict[int, Tuple[int, float]] = {}
    for pred_index in candidate_pred:
        gt_index = int(gt_indices[pred_index])
        distance = float(distances[pred_index])
        previous = best.get(gt_index)
        if previous is None or distance < previous[1]:
            best[gt_index] = (int(pred_index), distance)
    rows = [
        (pred_index, gt_index, distance)
        for gt_index, (pred_index, distance) in best.items()
    ]
    return pd.DataFrame(rows, columns=["pred_index", "gt_index", "distance_um"])


def correlation_metrics(
    pred_values: np.ndarray, gt_values: np.ndarray
) -> Tuple[float, float, float, float]:
    if len(pred_values) == 0:
        return (math.nan,) * 4
    pred_values = pred_values.astype(float)
    gt_values = gt_values.astype(float)
    if (
        len(pred_values) >= 3
        and np.std(pred_values) > 0
        and np.std(gt_values) > 0
    ):
        pearson = float(pearsonr(pred_values, gt_values)[0])
        spearman = float(spearmanr(pred_values, gt_values)[0])
    else:
        pearson = spearman = math.nan
    mae = float(np.mean(np.abs(pred_values - gt_values)))
    rmse = float(np.sqrt(np.mean((pred_values - gt_values) ** 2)))
    return pearson, spearman, mae, rmse


def sample_matches(matches: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size <= 0 or matches.empty:
        return matches.iloc[0:0].copy()
    rng = np.random.RandomState(42)
    positions = rng.permutation(len(matches))[:sample_size]
    sampled = matches.iloc[positions].copy().reset_index(drop=True)
    sampled["slot"] = np.arange(len(sampled), dtype=np.int64)
    return sampled


def aggregate_sampled_genes(
    parquet_path: Path,
    columns: Sequence[str],
    cell_column: str,
    gene_column: str,
    cell_to_slot: Mapping[str, int],
    batch_size: int,
    dataset_name: str,
    expected_input_rows: Optional[int] = None,
) -> pd.DataFrame:
    parquet = pq.ParquetFile(parquet_path)
    parts: List[pd.DataFrame] = []
    seen = (
        np.zeros(expected_input_rows, dtype=bool)
        if expected_input_rows is not None else None
    )
    for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size):
        frame = batch.to_pandas()
        if seen is not None:
            frame, _ = unique_batch_rows(frame, seen, dataset_name)
        cell_text = frame[cell_column].astype("string")
        slots = cell_text.map(cell_to_slot)
        keep = slots.notna().to_numpy()
        if not np.any(keep):
            continue
        selected = pd.DataFrame({
            "slot": slots.loc[keep].to_numpy(np.int64),
            "gene": frame.loc[keep, gene_column].astype(str).to_numpy(),
        })
        parts.append(
            selected.groupby(["slot", "gene"], sort=False, as_index=False)
            .size().rename(columns={"size": "count"})
        )
    if not parts:
        return pd.DataFrame(columns=["slot", "gene", "count"])
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby(["slot", "gene"], sort=False, as_index=False)[
        "count"
    ].sum()


def vector_cosine(
    spec: DatasetSpec,
    pred_cells: pd.DataFrame,
    gt_cells: pd.DataFrame,
    matches: pd.DataFrame,
    sample_size: int,
    expected_input_rows: int,
    batch_size: int,
    quiet: bool,
) -> Tuple[float, int]:
    sampled = sample_matches(matches, sample_size)
    if sampled.empty:
        return math.nan, 0
    pred_ids = pred_cells.iloc[sampled.pred_index.to_numpy()]["cell_id"].astype(str)
    gt_ids = gt_cells.iloc[sampled.gt_index.to_numpy()]["cell_id"].astype(str)
    pred_to_slot = dict(zip(pred_ids, sampled.slot))
    gt_to_slot = dict(zip(gt_ids, sampled.slot))
    info(f"  计算 vec_cosine（固定抽样 {len(sampled):,} 对）...", quiet)

    gt_gene = aggregate_sampled_genes(
        spec.prepared_path,
        ["cell_id", "feature_name"],
        "cell_id", "feature_name", gt_to_slot, batch_size, spec.name,
    )
    pred_gene = aggregate_sampled_genes(
        spec.assignments_path,
        ["transcript_id", "cell_id", "gene"],
        "cell_id", "gene", pred_to_slot, batch_size, spec.name,
        expected_input_rows=expected_input_rows,
    )
    if gt_gene.empty or pred_gene.empty:
        return math.nan, len(sampled)

    gt_norm_sq = gt_gene.assign(square=gt_gene["count"].astype(float) ** 2).groupby(
        "slot")["square"].sum()
    pred_norm_sq = pred_gene.assign(
        square=pred_gene["count"].astype(float) ** 2
    ).groupby("slot")["square"].sum()
    common = pred_gene.merge(
        gt_gene, on=["slot", "gene"], how="inner", suffixes=("_pred", "_gt")
    )
    dot = (common.count_pred.astype(float) * common.count_gt.astype(float)).groupby(
        common.slot).sum()
    cosines: List[float] = []
    for slot in range(len(sampled)):
        denom = math.sqrt(
            float(pred_norm_sq.get(slot, 0.0)) * float(gt_norm_sq.get(slot, 0.0))
        )
        if denom > 0:
            cosines.append(float(dot.get(slot, 0.0)) / denom)
    return (float(np.mean(cosines)) if cosines else math.nan), len(sampled)


def build_match_table(
    matches: pd.DataFrame, pred_cells: pd.DataFrame, gt_cells: pd.DataFrame
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=[
            "pred_cell_id", "gt_cell_id", "distance_um", "pred_x_um",
            "pred_y_um", "gt_x_um", "gt_y_um", "pred_count", "gt_count",
        ])
    pred = pred_cells.iloc[matches.pred_index.to_numpy()].reset_index(drop=True)
    gt = gt_cells.iloc[matches.gt_index.to_numpy()].reset_index(drop=True)
    return pd.DataFrame({
        "pred_cell_id": pred.cell_id.astype(str),
        "gt_cell_id": gt.cell_id.astype(str),
        "distance_um": matches.distance_um.to_numpy(float),
        "pred_x_um": pred.x.to_numpy(float),
        "pred_y_um": pred.y.to_numpy(float),
        "gt_x_um": gt.x.to_numpy(float),
        "gt_y_um": gt.y.to_numpy(float),
        "pred_count": pred.pred_count.to_numpy(np.int64),
        "gt_count": gt.gt_count.to_numpy(np.int64),
    })


def evaluate_dataset(
    spec: DatasetSpec,
    out_root: Path,
    radius_um: float,
    batch_size: int,
    vector_sample_size: int,
    quiet: bool,
) -> Dict[str, Any]:
    info("=" * 72, quiet)
    info(f"Stereo-seq x Bering: {spec.name}", quiet)
    validation = validate_dataset(spec)
    gt_cells = load_gt_cells(spec, batch_size, quiet)
    pred_cells, diagnostics = load_pred_cells(
        spec, validation["prepared_rows"], batch_size, quiet
    )
    matches = match_cells(pred_cells, gt_cells, radius_um)
    n_gt, n_pred, n_match = len(gt_cells), len(pred_cells), len(matches)
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    distances = matches.distance_um.to_numpy(float)
    if n_match:
        pred_counts = pred_cells.iloc[matches.pred_index.to_numpy()][
            "pred_count"].to_numpy(float)
        gt_counts = gt_cells.iloc[matches.gt_index.to_numpy()][
            "gt_count"].to_numpy(float)
    else:
        pred_counts = gt_counts = np.array([], dtype=float)
    count_pearson, count_spearman, count_mae, count_rmse = correlation_metrics(
        pred_counts, gt_counts
    )
    cosine, vector_pairs = vector_cosine(
        spec, pred_cells, gt_cells, matches, vector_sample_size,
        validation["prepared_rows"], batch_size, quiet,
    )

    result: Dict[str, Any] = {
        "dataset": spec.name,
        "gt_cells": n_gt,
        "pred_cells": n_pred,
        "cell_ratio": n_pred / n_gt if n_gt else math.nan,
        "matched": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loc_mean_um": float(np.mean(distances)) if n_match else math.nan,
        "loc_median_um": float(np.median(distances)) if n_match else math.nan,
        "loc_p95_um": float(np.percentile(distances, 95)) if n_match else math.nan,
        "count_pearson": count_pearson,
        "count_spearman": count_spearman,
        "count_mae": count_mae,
        "count_rmse": count_rmse,
        "vec_cosine": cosine,
        "vec_js_dist": math.nan,
        "vec_pearson": math.nan,
        "assign_overlap": math.nan,
        "assign_accuracy": math.nan,
        "match_radius_um": radius_um,
        "vector_sample_pairs": vector_pairs,
        "input_transcript_rows": validation["prepared_rows"],
        **diagnostics,
        "pred_transcript_assignment_rate": (
            diagnostics["unique_assigned_transcripts"] / validation["prepared_rows"]
        ),
        "benchmark_mode": str(spec.summary.get("benchmark_mode", "unknown")),
        "coarse_cell_boundaries_used": bool(
            spec.summary.get("coarse_cell_boundaries_used", False)
        ),
        "assignment_metrics_note": (
            "NaN: CellBin-derived coarse seeds and GT share a source (circular)"
        ),
        "prepared_gt_path": str(spec.prepared_path),
        "bering_result_path": str(spec.result_dir),
        "evaluator_version": VERSION,
    }

    dataset_out = out_root / spec.name
    dataset_out.mkdir(parents=True, exist_ok=True)
    ordered = ["dataset", *METRIC_COLUMNS] + [
        column for column in result if column not in {"dataset", *METRIC_COLUMNS}
    ]
    pd.DataFrame([result], columns=ordered).to_csv(
        dataset_out / "bering_stereo_metrics.csv", index=False
    )
    build_match_table(matches, pred_cells, gt_cells).to_csv(
        dataset_out / "bering_stereo_matched_cells.csv", index=False
    )
    gt_cells.rename(columns={
        "x": "x_um", "y": "y_um"
    }).to_csv(dataset_out / "bering_stereo_gt_cells.csv", index=False)
    pred_cells.rename(columns={
        "x": "x_um", "y": "y_um"
    }).to_csv(dataset_out / "bering_stereo_pred_cells.csv", index=False)

    info(
        f"  matched={n_match:,}/{n_pred:,} pred, precision={precision:.4f}, "
        f"recall={recall:.4f}, F1={f1:.4f}, cosine={cosine:.4f}",
        quiet,
    )
    info(f"  写出: {dataset_out}", quiet)
    return result


def main() -> None:
    args = parse_args()
    discovered = discover_datasets(args.outputs_dir)
    selected = select_datasets(discovered, args.dataset)
    info(
        "发现并选择数据集: " + ", ".join(spec.name for spec in selected),
        args.quiet,
    )
    for spec in selected:
        counts = validate_dataset(spec)
        info(
            f"  [OK] {spec.name}: input={counts['prepared_rows']:,}, "
            f"assigned={counts['output_rows']:,}",
            args.quiet,
        )
    if args.check_only:
        info("输入检查完成；未运行评估。", args.quiet)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_dataset(
            spec, args.out_dir, args.match_radius_um, args.batch_size,
            args.vector_sample_size, args.quiet,
        )
        for spec in selected
    ]
    combined = pd.DataFrame(results)
    ordered = ["dataset", *METRIC_COLUMNS] + [
        column for column in combined.columns
        if column not in {"dataset", *METRIC_COLUMNS}
    ]
    combined = combined.reindex(columns=ordered)
    combined_path = args.out_dir / "bering_stereo_evaluation.csv"
    combined.to_csv(combined_path, index=False)
    info("=" * 72, args.quiet)
    info(f"汇总结果已写入: {combined_path}", args.quiet)


if __name__ == "__main__":
    main()
