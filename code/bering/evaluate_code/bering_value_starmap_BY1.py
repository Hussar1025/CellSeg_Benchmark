#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Bering STARmap BY1 with the UCS reference script's 19 metrics.

The implementation follows ``ucs_value_starmap_1.py``:

* cell centroids are means of assigned transcript coordinates;
* GT cells are visited in their original order and matched to the nearest
  predicted centroid inside a finite radius;
* a predicted cell can be used only once;
* count, gene-vector and transcript-assignment metrics use the same matched
  cell pairs.

STARmap x/y values are native image pixels.  Pixel size is inferred from the
DAPI TIFF resolution metadata (ImageJ unit must be micron), or can be supplied
explicitly with ``--um-per-pixel``.  Pixels are never silently treated as um.

The Bering run used ``spots_all.csv:clustermap`` as coarse seed cells.  Output
rows therefore carry ``uses_platform_prior=True`` and a reference-assisted
note; this result is not an independent de-novo segmentation benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


VERSION = "2026.09.20"
DATA_ROOT = Path("yourpath/dataset/starmap_BY1")
SPOTS = DATA_ROOT / "spots_all.csv"
DAPI = DATA_ROOT / "BY1_dapi.tiff"
PRED_ROOT = Path("yourpath/bering/starmap_BY1_output")
ASSIGNMENTS = PRED_ROOT / "segmentation_transcripts.parquet"
PRED_CELLS = PRED_ROOT / "cells_ensembled_metadata.csv"
RUN_SUMMARY = PRED_ROOT / "run_summary.json"
DEFAULT_OUT = Path("yourpath/eval/bering/bering_starmap_BY1_evaluation.csv")

METRICS = [
    "gt_cells", "pred_cells", "cell_ratio", "matched", "precision", "recall", "f1",
    "loc_mean_um", "loc_median_um", "loc_p95_um",
    "count_pearson", "count_spearman", "count_mae", "count_rmse",
    "vec_cosine", "vec_js_dist", "vec_pearson", "assign_overlap", "assign_accuracy",
]

log = logging.getLogger("eval_bering_starmap")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    log.addHandler(stream)
    log.addHandler(file_handler)


def resolution_value(tag: object) -> float:
    """Convert a TIFF rational resolution value to pixels per unit."""
    if hasattr(tag, "numerator") and hasattr(tag, "denominator"):
        return float(tag.numerator) / float(tag.denominator)
    if isinstance(tag, (tuple, list)) and len(tag) == 2:
        return float(tag[0]) / float(tag[1])
    return float(tag)


def infer_um_per_pixel(path: Path) -> float:
    """Infer a scalar um/pixel from ImageJ TIFF x/y resolution tags."""
    with tifffile.TiffFile(path) as tif:
        metadata = tif.imagej_metadata or {}
        unit = str(metadata.get("unit", "")).strip().lower()
        if unit not in {"micron", "microns", "um", "µm", "micrometer", "micrometers"}:
            raise ValueError(
                f"cannot infer physical scale from {path}: ImageJ unit={unit!r}; "
                "pass --um-per-pixel explicitly"
            )
        page = tif.pages[0]
        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")
        if x_tag is None or y_tag is None:
            raise ValueError(f"{path} has no XResolution/YResolution; pass --um-per-pixel")
        x_ppum = resolution_value(x_tag.value)
        y_ppum = resolution_value(y_tag.value)
    if x_ppum <= 0 or y_ppum <= 0:
        raise ValueError(f"invalid TIFF resolution: x={x_ppum}, y={y_ppum}")
    x_um = 1.0 / x_ppum
    y_um = 1.0 / y_ppum
    if abs(x_um - y_um) / np.mean([x_um, y_um]) > 0.01:
        raise ValueError(
            f"anisotropic TIFF pixels x={x_um:.8g}, y={y_um:.8g}; "
            "the reference evaluator accepts one scalar --um-per-pixel"
        )
    return float(np.mean([x_um, y_um]))


def safe_corr(function, first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(function(a, b)[0])
    except Exception:
        return np.nan


def match_cells(gt: pd.DataFrame, pred: pd.DataFrame, radius_um: float) -> pd.DataFrame:
    """GT-order nearest-neighbour matching, identical to the UCS reference."""
    columns = ["gt_id", "pred_id", "distance_um"]
    if gt.empty or pred.empty:
        return pd.DataFrame(columns=columns)
    tree = cKDTree(pred[["x_um", "y_um"]].to_numpy(float))
    distance, index = tree.query(
        gt[["x_um", "y_um"]].to_numpy(float),
        k=1,
        distance_upper_bound=radius_um,
    )
    used = set()
    rows = []
    pred_ids = pred.pred_id.astype(str).to_numpy()
    gt_ids = gt.gt_id.astype(str).to_numpy()
    for gt_index in range(len(gt)):
        if not np.isfinite(distance[gt_index]) or distance[gt_index] >= radius_um:
            continue
        pred_index = int(index[gt_index])
        pred_id = pred_ids[pred_index]
        if pred_id in used:
            continue
        used.add(pred_id)
        rows.append((gt_ids[gt_index], pred_id, float(distance[gt_index])))
    return pd.DataFrame(rows, columns=columns)


def expression_metrics(tx: pd.DataFrame, matches: pd.DataFrame, vector_sample: int) -> Dict[str, float]:
    if matches.empty:
        return {
            "count_pearson": np.nan,
            "count_spearman": np.nan,
            "count_mae": np.nan,
            "count_rmse": np.nan,
            "vec_cosine": np.nan,
            "vec_js_dist": np.nan,
            "vec_pearson": np.nan,
        }

    gt_counts = tx.loc[tx.gt_cell.ne("0")].groupby("gt_cell").size()
    pred_counts = tx.loc[tx.pred_cell.ne("0")].groupby("pred_cell").size()
    gt_values = np.asarray(
        [gt_counts.get(str(row.gt_id), 0) for row in matches.itertuples()], dtype=float)
    pred_values = np.asarray(
        [pred_counts.get(str(row.pred_id), 0) for row in matches.itertuples()], dtype=float)
    result = {
        "count_pearson": safe_corr(pearsonr, gt_values, pred_values),
        "count_spearman": safe_corr(spearmanr, gt_values, pred_values),
        "count_mae": float(np.mean(np.abs(gt_values - pred_values))),
        "count_rmse": float(np.sqrt(np.mean((gt_values - pred_values) ** 2))),
    }

    sampled = (matches if len(matches) <= vector_sample
               else matches.sample(vector_sample, random_state=0))
    gt_to_row = {str(value): index for index, value in enumerate(sampled.gt_id)}
    pred_to_row = {str(value): index for index, value in enumerate(sampled.pred_id)}
    genes = sorted(tx.gene.astype(str).unique())
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    gt_matrix = np.zeros((len(sampled), len(genes)), dtype=np.float32)
    pred_matrix = np.zeros_like(gt_matrix)
    gt_subset = tx.loc[tx.gt_cell.isin(gt_to_row)]
    pred_subset = tx.loc[tx.pred_cell.isin(pred_to_row)]
    for (cell, gene), count in gt_subset.groupby(["gt_cell", "gene"]).size().items():
        if str(cell) in gt_to_row:
            gt_matrix[gt_to_row[str(cell)], gene_to_col[str(gene)]] = count
    for (cell, gene), count in pred_subset.groupby(["pred_cell", "gene"]).size().items():
        if str(cell) in pred_to_row:
            pred_matrix[pred_to_row[str(cell)], gene_to_col[str(gene)]] = count

    cosines, js_distances, correlations = [], [], []
    for gt_vector, pred_vector in zip(gt_matrix, pred_matrix):
        gt_norm = np.linalg.norm(gt_vector)
        pred_norm = np.linalg.norm(pred_vector)
        if gt_norm == 0 or pred_norm == 0:
            continue
        cosines.append(float(np.dot(gt_vector, pred_vector) / (gt_norm * pred_norm)))
        distance = jensenshannon(
            gt_vector / gt_vector.sum(), pred_vector / pred_vector.sum(), base=2)
        if np.isfinite(distance):
            js_distances.append(float(distance))
        if np.std(gt_vector) > 0 and np.std(pred_vector) > 0:
            correlations.append(float(pearsonr(gt_vector, pred_vector)[0]))
    result.update({
        "vec_cosine": float(np.mean(cosines)) if cosines else np.nan,
        "vec_js_dist": float(np.mean(js_distances)) if js_distances else np.nan,
        "vec_pearson": float(np.mean(correlations)) if correlations else np.nan,
    })
    return result


def load_evaluation_table(spots_path: Path, assignments_path: Path) -> pd.DataFrame:
    """Join official STARmap GT and final Bering assignments by transcript ID."""
    spots = pd.read_csv(
        spots_path,
        usecols=["gene_name", "spot_location_1", "spot_location_2", "clustermap"],
    )
    spots["transcript_id"] = np.arange(len(spots), dtype=np.int64)
    prediction = pd.read_parquet(
        assignments_path,
        columns=["transcript_id", "ensembled_cells"],
    )
    prediction["transcript_id"] = pd.to_numeric(
        prediction.transcript_id, errors="raise").astype(np.int64)
    if prediction.transcript_id.duplicated().any():
        raise ValueError(f"duplicate transcript_id values in {assignments_path}")
    if len(prediction) and (
        prediction.transcript_id.min() < 0 or prediction.transcript_id.max() >= len(spots)
    ):
        raise ValueError("Bering transcript_id is outside the spots_all.csv row range")
    tx = spots.merge(prediction, on="transcript_id", how="left", validate="one_to_one")

    gt_number = pd.to_numeric(tx.clustermap, errors="coerce").fillna(-1)
    pred_number = pd.to_numeric(tx.ensembled_cells, errors="coerce").fillna(0)
    tx["gt_cell"] = gt_number.astype(np.int64).astype(str)
    tx.loc[gt_number < 0, "gt_cell"] = "0"
    tx["pred_cell"] = pred_number.astype(np.int64).astype(str)
    tx.loc[pred_number <= 0, "pred_cell"] = "0"
    tx["gene"] = tx.gene_name.astype(str)
    tx["x_native"] = pd.to_numeric(tx.spot_location_1, errors="raise").astype(float)
    tx["y_native"] = pd.to_numeric(tx.spot_location_2, errors="raise").astype(float)
    return tx[["transcript_id", "gene", "x_native", "y_native", "gt_cell", "pred_cell"]]


def validate_inputs(
    spots_path: Path,
    assignments_path: Path,
    cells_path: Path,
    dapi_path: Path,
) -> None:
    import pyarrow.parquet as pq

    for path in (spots_path, assignments_path, cells_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    spot_columns = set(pd.read_csv(spots_path, nrows=0).columns)
    required_spots = {"gene_name", "spot_location_1", "spot_location_2", "clustermap"}
    if not required_spots.issubset(spot_columns):
        raise ValueError(f"{spots_path} lacks {required_spots - spot_columns}")
    assignment_columns = set(pq.ParquetFile(assignments_path).schema_arrow.names)
    required_assignments = {"transcript_id", "ensembled_cells"}
    if not required_assignments.issubset(assignment_columns):
        raise ValueError(f"{assignments_path} lacks {required_assignments - assignment_columns}")
    cell_columns = set(pd.read_csv(cells_path, nrows=0).columns)
    if "cell_id" not in cell_columns:
        raise ValueError(f"{cells_path} lacks cell_id")


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    spots_path = Path(args.spots)
    assignments_path = Path(args.assignments)
    cells_path = Path(args.pred_cells)
    dapi_path = Path(args.dapi)
    output_path = Path(args.out)
    validate_inputs(spots_path, assignments_path, cells_path, dapi_path)
    um_per_pixel = (float(args.um_per_pixel) if args.um_per_pixel is not None
                    else infer_um_per_pixel(dapi_path))
    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError(f"--um-per-pixel must be positive, got {um_per_pixel}")

    setup_logging(output_path.with_suffix(".log"))
    log.info("Bering x STARmap BY1 evaluation")
    log.info("um_per_pixel=%.9f; match_radius_um=%.4f", um_per_pixel, args.match_radius_um)
    tx = load_evaluation_table(spots_path, assignments_path)

    gt = (tx.loc[tx.gt_cell.ne("0")]
          .groupby("gt_cell")[["x_native", "y_native"]]
          .mean().reset_index()
          .rename(columns={"gt_cell": "gt_id", "x_native": "x_um", "y_native": "y_um"}))
    pred = (tx.loc[tx.pred_cell.ne("0")]
            .groupby("pred_cell")[["x_native", "y_native"]]
            .mean().reset_index()
            .rename(columns={"pred_cell": "pred_id", "x_native": "x_um", "y_native": "y_um"}))
    gt[["x_um", "y_um"]] *= um_per_pixel
    pred[["x_um", "y_um"]] *= um_per_pixel

    cell_table = pd.read_csv(cells_path, usecols=["cell_id"])
    pred_ids = pd.to_numeric(cell_table.cell_id, errors="coerce")
    pred_cells = int(pred_ids[pred_ids > 0].nunique())
    assigned_pred_cells = int(tx.loc[tx.pred_cell.ne("0"), "pred_cell"].nunique())
    if assigned_pred_cells != pred_cells:
        raise ValueError(
            f"predicted cell count mismatch: metadata={pred_cells}, assignments={assigned_pred_cells}"
        )

    matches = match_cells(gt, pred, args.match_radius_um)
    metrics = expression_metrics(tx[["gene", "gt_cell", "pred_cell"]], matches, args.vec_sample)
    gt_assigned = tx.gt_cell.ne("0")
    pred_assigned = tx.pred_cell.ne("0")
    both = gt_assigned & pred_assigned
    metrics["assign_overlap"] = float(both.sum() / max(int(gt_assigned.sum()), 1))
    pred_to_gt = {str(row.pred_id): str(row.gt_id) for row in matches.itertuples()}
    mapped = tx.loc[both, "pred_cell"].map(pred_to_gt)
    metrics["assign_accuracy"] = (
        float((mapped.to_numpy(object) == tx.loc[both, "gt_cell"].to_numpy(object)).mean())
        if both.any() else np.nan
    )

    gt_cells = len(gt)
    matched = len(matches)
    precision = matched / max(pred_cells, 1)
    recall = matched / max(gt_cells, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    distances = matches.distance_um.to_numpy(float)
    row: Dict[str, object] = {
        "method": "Bering",
        "dataset": "starmap_BY1",
        "gt_cells": gt_cells,
        "pred_cells": pred_cells,
        "cell_ratio": pred_cells / max(gt_cells, 1),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loc_mean_um": float(np.mean(distances)) if len(distances) else np.nan,
        "loc_median_um": float(np.median(distances)) if len(distances) else np.nan,
        "loc_p95_um": float(np.percentile(distances, 95)) if len(distances) else np.nan,
        **metrics,
        "um_per_pixel": um_per_pixel,
        "match_radius_um": float(args.match_radius_um),
        "uses_platform_prior": True,
        "note": (
            "Reference-assisted Bering: spots_all.csv clustermap IDs were used as coarse seed cells; "
            "unsupervised Leiden labels were used for training; no DAPI or reference cell summary "
            "was supplied to segmentation. Metrics follow ucs_value_starmap_1.py."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "dataset"] + METRICS + [
        "um_per_pixel", "match_radius_um", "uses_platform_prior", "note",
    ]
    result = pd.DataFrame([row]).reindex(columns=columns)
    result.to_csv(output_path, index=False)
    matched_path = output_path.with_name(output_path.stem + "_matched_cells.csv")
    matches.to_csv(matched_path, index=False)
    config_path = output_path.with_name(output_path.stem + "_config.json")
    config_path.write_text(json.dumps({
        "version": VERSION,
        "spots": str(spots_path),
        "assignments": str(assignments_path),
        "pred_cells": str(cells_path),
        "dapi": str(dapi_path),
        "um_per_pixel": um_per_pixel,
        "match_radius_um": float(args.match_radius_um),
        "vec_sample": int(args.vec_sample),
        "uses_platform_prior": True,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(
        "gt=%d pred=%d matched=%d precision=%.4f recall=%.4f F1=%.4f",
        gt_cells, pred_cells, matched, precision, recall, f1,
    )
    log.info("saved -> %s", output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spots", default=str(SPOTS))
    parser.add_argument("--assignments", default=str(ASSIGNMENTS))
    parser.add_argument("--pred-cells", default=str(PRED_CELLS))
    parser.add_argument("--dapi", default=str(DAPI))
    parser.add_argument("--um-per-pixel", type=float)
    parser.add_argument("--match-radius-um", type=float, default=12.0)
    parser.add_argument("--vec-sample", type=int, default=2000)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print(VERSION)
        return
    if args.check_only:
        validate_inputs(
            Path(args.spots), Path(args.assignments), Path(args.pred_cells), Path(args.dapi))
        pixel_size = (args.um_per_pixel if args.um_per_pixel is not None
                      else infer_um_per_pixel(Path(args.dapi)))
        print(f"[OK] inputs valid; um_per_pixel={pixel_size:.9f}")
        return
    evaluate(args)


if __name__ == "__main__":
    main()
