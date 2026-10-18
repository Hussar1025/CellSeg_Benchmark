#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Baysor STARmap BY1 with the UCS reference script's 19 metrics.

The implementation follows ``ucs_value_starmap_1.py``:

* official GT labels come from ``spots_all.csv:clustermap``;
* cell centroids are means of their assigned transcript coordinates;
* GT cells are visited in original group order and matched to their nearest
  predicted centroid within a finite radius;
* each predicted cell can be matched only once;
* count, gene-vector and transcript-assignment metrics use the same pairs.

Baysor filters a few low-frequency genes and reorders output rows, so its
``segmentation.csv`` is joined back to the official spot table by the exact
``(gene, x, y, z, duplicate-occurrence)`` key.  The script verifies complete
coverage of every Baysor output molecule before calculating metrics.

STARmap coordinates are native image pixels.  Pixel size is inferred from the
DAPI TIFF ImageJ resolution metadata, or can be supplied with
``--um-per-pixel``.  Pixels are never silently treated as micrometres.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


VERSION = "2026.09.20"
DATA_ROOT = Path("/data/qiuyijia/dataset/starmap_BY1")
SPOTS = DATA_ROOT / "spots_all.csv"
DAPI = DATA_ROOT / "BY1_dapi.tiff"
PRED_ROOT = Path("/data/qiuyijia/baysor/starmap_BY1_output")
ASSIGNMENTS = PRED_ROOT / "segmentation.csv"
PRED_CELLS = PRED_ROOT / "segmentation_cell_stats.csv"
RUN_PARAMS = PRED_ROOT / "segmentation_params.dump.toml"
DEFAULT_OUT = Path("/data/qiuyijia/eval/baysor/baysor_starmap_BY1_evaluation.csv")

METRICS = [
    "gt_cells", "pred_cells", "cell_ratio", "matched", "precision", "recall", "f1",
    "loc_mean_um", "loc_median_um", "loc_p95_um",
    "count_pearson", "count_spearman", "count_mae", "count_rmse",
    "vec_cosine", "vec_js_dist", "vec_pearson", "assign_overlap", "assign_accuracy",
]

log = logging.getLogger("eval_baysor_starmap")


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w")
    file_handler.setFormatter(formatter)
    log.addHandler(stream)
    log.addHandler(file_handler)


def resolution_value(value: object) -> float:
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def infer_um_per_pixel(path: Path) -> float:
    with tifffile.TiffFile(path) as tif:
        metadata = tif.imagej_metadata or {}
        unit = str(metadata.get("unit", "")).strip().lower()
        accepted = {"micron", "microns", "um", "µm", "micrometer", "micrometers"}
        if unit not in accepted:
            raise ValueError(
                f"cannot infer physical scale from {path}: unit={unit!r}; "
                "pass --um-per-pixel explicitly")
        page = tif.pages[0]
        x_tag, y_tag = page.tags.get("XResolution"), page.tags.get("YResolution")
        if x_tag is None or y_tag is None:
            raise ValueError(f"{path} lacks XResolution/YResolution")
        x_um = 1.0 / resolution_value(x_tag.value)
        y_um = 1.0 / resolution_value(y_tag.value)
    if x_um <= 0 or y_um <= 0:
        raise ValueError(f"invalid pixel sizes: x={x_um}, y={y_um}")
    if abs(x_um - y_um) / np.mean([x_um, y_um]) > 0.01:
        raise ValueError(f"anisotropic pixels x={x_um}, y={y_um}")
    return float(np.mean([x_um, y_um]))


def safe_corr(function, first: Sequence[float], second: Sequence[float]) -> float:
    a, b = np.asarray(first, float), np.asarray(second, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(function(a, b)[0])
    except Exception:
        return np.nan


def normalize_pred_cell(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    return values.mask(values.isin(["", "0", "-1", "nan", "None"]), "0")


def load_evaluation_table(spots_path: Path, assignments_path: Path) -> pd.DataFrame:
    spots = pd.read_csv(
        spots_path,
        usecols=["gene_name", "spot_location_1", "spot_location_2",
                 "spot_location_3", "clustermap"],
    ).rename(columns={
        "gene_name": "gene", "spot_location_1": "x_native",
        "spot_location_2": "y_native", "spot_location_3": "z_native",
    })
    prediction = pd.read_csv(
        assignments_path,
        usecols=["cell", "gene", "x", "y", "z", "is_noise"],
    ).rename(columns={"x": "x_native", "y": "y_native", "z": "z_native"})

    key = ["gene", "x_native", "y_native", "z_native"]
    for frame in (spots, prediction):
        frame["gene"] = frame.gene.astype(str)
        for col in ("x_native", "y_native", "z_native"):
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        # Exact duplicate spots exist. cumcount provides a deterministic 1:1 key.
        frame["_occurrence"] = frame.groupby(key, sort=False).cumcount()

    official_key = spots[key + ["_occurrence"]].copy()
    official_key["_official_row"] = np.arange(len(spots), dtype=np.int64)
    pred_joined = prediction.merge(
        official_key, on=key + ["_occurrence"], how="left", validate="one_to_one")
    missing = int(pred_joined._official_row.isna().sum())
    if missing:
        raise ValueError(
            f"{missing:,}/{len(pred_joined):,} Baysor molecules cannot be joined "
            "to spots_all.csv by (gene,x,y,z,occurrence)")
    pred_joined["_official_row"] = pred_joined._official_row.astype(np.int64)

    assigned = normalize_pred_cell(pred_joined.cell)
    noise = pd.to_numeric(pred_joined.is_noise, errors="coerce").fillna(0).astype(bool)
    assigned = assigned.mask(noise, "0")
    pred_by_row = pd.Series(assigned.to_numpy(), index=pred_joined._official_row)
    if pred_by_row.index.duplicated().any():
        raise ValueError("duplicate official row IDs after Baysor join")

    gt_number = pd.to_numeric(spots.clustermap, errors="coerce").fillna(-1)
    spots["gt_cell"] = gt_number.astype(np.int64).astype(str)
    spots.loc[gt_number < 0, "gt_cell"] = "0"
    spots["pred_cell"] = pred_by_row.reindex(np.arange(len(spots))).fillna("0").to_numpy()
    spots["transcript_id"] = np.arange(len(spots), dtype=np.int64)
    return spots[[
        "transcript_id", "gene", "x_native", "y_native", "z_native",
        "gt_cell", "pred_cell",
    ]]


def match_cells(gt: pd.DataFrame, pred: pd.DataFrame, radius: float) -> pd.DataFrame:
    columns = ["gt_id", "pred_id", "distance_um"]
    if gt.empty or pred.empty:
        return pd.DataFrame(columns=columns)
    tree = cKDTree(pred[["x_um", "y_um"]].to_numpy(float))
    distance, index = tree.query(
        gt[["x_um", "y_um"]].to_numpy(float), k=1,
        distance_upper_bound=radius)
    used, rows = set(), []
    pred_ids = pred.pred_id.astype(str).to_numpy()
    gt_ids = gt.gt_id.astype(str).to_numpy()
    for gt_index in range(len(gt)):
        if not np.isfinite(distance[gt_index]) or distance[gt_index] >= radius:
            continue
        pred_index = int(index[gt_index])
        pred_id = pred_ids[pred_index]
        if pred_id in used:
            continue
        used.add(pred_id)
        rows.append((gt_ids[gt_index], pred_id, float(distance[gt_index])))
    return pd.DataFrame(rows, columns=columns)


def expression_metrics(tx: pd.DataFrame, matches: pd.DataFrame, sample_size: int) -> Dict[str, float]:
    if matches.empty:
        return {key: np.nan for key in (
            "count_pearson", "count_spearman", "count_mae", "count_rmse",
            "vec_cosine", "vec_js_dist", "vec_pearson")}

    gt_counts = tx.loc[tx.gt_cell.ne("0")].groupby("gt_cell").size()
    pred_counts = tx.loc[tx.pred_cell.ne("0")].groupby("pred_cell").size()
    gt_values = np.asarray(
        [gt_counts.get(str(row.gt_id), 0) for row in matches.itertuples()], float)
    pred_values = np.asarray(
        [pred_counts.get(str(row.pred_id), 0) for row in matches.itertuples()], float)
    result = {
        "count_pearson": safe_corr(pearsonr, gt_values, pred_values),
        "count_spearman": safe_corr(spearmanr, gt_values, pred_values),
        "count_mae": float(np.mean(np.abs(gt_values - pred_values))),
        "count_rmse": float(np.sqrt(np.mean((gt_values - pred_values) ** 2))),
    }

    sampled = matches if len(matches) <= sample_size else matches.sample(sample_size, random_state=0)
    gt_to_row = {str(value): index for index, value in enumerate(sampled.gt_id)}
    pred_to_row = {str(value): index for index, value in enumerate(sampled.pred_id)}
    genes = sorted(tx.gene.astype(str).unique())
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    gt_matrix = np.zeros((len(sampled), len(genes)), dtype=np.float32)
    pred_matrix = np.zeros_like(gt_matrix)
    for (cell, gene), count in (tx.loc[tx.gt_cell.isin(gt_to_row)]
                                .groupby(["gt_cell", "gene"]).size().items()):
        gt_matrix[gt_to_row[str(cell)], gene_to_col[str(gene)]] = count
    for (cell, gene), count in (tx.loc[tx.pred_cell.isin(pred_to_row)]
                                .groupby(["pred_cell", "gene"]).size().items()):
        pred_matrix[pred_to_row[str(cell)], gene_to_col[str(gene)]] = count

    cosines, distances, correlations = [], [], []
    for gt_vector, pred_vector in zip(gt_matrix, pred_matrix):
        gt_norm, pred_norm = np.linalg.norm(gt_vector), np.linalg.norm(pred_vector)
        if gt_norm == 0 or pred_norm == 0:
            continue
        cosines.append(float(np.dot(gt_vector, pred_vector) / (gt_norm * pred_norm)))
        distance = jensenshannon(
            gt_vector / gt_vector.sum(), pred_vector / pred_vector.sum(), base=2)
        if np.isfinite(distance):
            distances.append(float(distance))
        if np.std(gt_vector) > 0 and np.std(pred_vector) > 0:
            correlations.append(float(pearsonr(gt_vector, pred_vector)[0]))
    result.update({
        "vec_cosine": float(np.mean(cosines)) if cosines else np.nan,
        "vec_js_dist": float(np.mean(distances)) if distances else np.nan,
        "vec_pearson": float(np.mean(correlations)) if correlations else np.nan,
    })
    return result


def validate_inputs(spots: Path, assignments: Path, cells: Path, dapi: Path) -> None:
    for path in (spots, assignments, cells, dapi, RUN_PARAMS):
        if not path.is_file():
            raise FileNotFoundError(path)
    spot_columns = set(pd.read_csv(spots, nrows=0).columns)
    required_spots = {
        "gene_name", "spot_location_1", "spot_location_2",
        "spot_location_3", "clustermap"}
    if not required_spots.issubset(spot_columns):
        raise ValueError(f"{spots} lacks {required_spots - spot_columns}")
    assignment_columns = set(pd.read_csv(assignments, nrows=0).columns)
    required_assignments = {"cell", "gene", "x", "y", "z", "is_noise"}
    if not required_assignments.issubset(assignment_columns):
        raise ValueError(f"{assignments} lacks {required_assignments - assignment_columns}")
    cell_columns = set(pd.read_csv(cells, nrows=0).columns)
    if not {"cell", "x", "y"}.issubset(cell_columns):
        raise ValueError(f"{cells} lacks cell/x/y")
    params = RUN_PARAMS.read_text()
    if '[prior]\ntype = "none"' not in params:
        raise ValueError("Baysor run did not use the expected prior.type=none")


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    spots, assignments = Path(args.spots), Path(args.assignments)
    cells, dapi, output = Path(args.pred_cells), Path(args.dapi), Path(args.out)
    validate_inputs(spots, assignments, cells, dapi)
    um_per_pixel = float(args.um_per_pixel) if args.um_per_pixel else infer_um_per_pixel(dapi)
    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError(f"invalid um/pixel: {um_per_pixel}")
    setup_logging(output.with_suffix(".log"))
    log.info("Baysor x STARmap BY1 evaluation")
    log.info("um_per_pixel=%.9f; match_radius_um=%.4f", um_per_pixel, args.match_radius_um)

    tx = load_evaluation_table(spots, assignments)
    gt = (tx.loc[tx.gt_cell.ne("0")]
          .groupby("gt_cell")[["x_native", "y_native"]].mean().reset_index()
          .rename(columns={"gt_cell": "gt_id", "x_native": "x_um", "y_native": "y_um"}))
    pred = (tx.loc[tx.pred_cell.ne("0")]
            .groupby("pred_cell")[["x_native", "y_native"]].mean().reset_index()
            .rename(columns={"pred_cell": "pred_id", "x_native": "x_um", "y_native": "y_um"}))
    gt[["x_um", "y_um"]] *= um_per_pixel
    pred[["x_um", "y_um"]] *= um_per_pixel

    cell_table = pd.read_csv(cells, usecols=["cell"])
    metadata_ids = normalize_pred_cell(cell_table.cell)
    pred_cells = int(metadata_ids[metadata_ids.ne("0")].nunique())
    assigned_cells = int(tx.loc[tx.pred_cell.ne("0"), "pred_cell"].nunique())
    if pred_cells != assigned_cells:
        raise ValueError(
            f"predicted cell count mismatch: stats={pred_cells}, assignments={assigned_cells}")

    matches = match_cells(gt, pred, args.match_radius_um)
    metrics = expression_metrics(tx[["gene", "gt_cell", "pred_cell"]], matches, args.vec_sample)
    gt_assigned, pred_assigned = tx.gt_cell.ne("0"), tx.pred_cell.ne("0")
    both = gt_assigned & pred_assigned
    metrics["assign_overlap"] = float(both.sum() / max(int(gt_assigned.sum()), 1))
    pred_to_gt = {str(row.pred_id): str(row.gt_id) for row in matches.itertuples()}
    mapped = tx.loc[both, "pred_cell"].map(pred_to_gt)
    metrics["assign_accuracy"] = (
        float((mapped.to_numpy(object) == tx.loc[both, "gt_cell"].to_numpy(object)).mean())
        if both.any() else np.nan)

    gt_cells, matched = len(gt), len(matches)
    precision = matched / max(pred_cells, 1)
    recall = matched / max(gt_cells, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    distance = matches.distance_um.to_numpy(float)
    row: Dict[str, object] = {
        "method": "Baysor", "dataset": "starmap_BY1",
        "gt_cells": gt_cells, "pred_cells": pred_cells,
        "cell_ratio": pred_cells / max(gt_cells, 1), "matched": matched,
        "precision": precision, "recall": recall, "f1": f1,
        "loc_mean_um": float(np.mean(distance)) if len(distance) else np.nan,
        "loc_median_um": float(np.median(distance)) if len(distance) else np.nan,
        "loc_p95_um": float(np.percentile(distance, 95)) if len(distance) else np.nan,
        **metrics,
        "um_per_pixel": um_per_pixel,
        "match_radius_um": float(args.match_radius_um),
        "uses_platform_prior": False,
        "note": (
            "De-novo Baysor run with prior.type=none. Exact transcript join to official "
            "spots_all.csv clustermap GT; metrics follow ucs_value_starmap_1.py."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "dataset"] + METRICS + [
        "um_per_pixel", "match_radius_um", "uses_platform_prior", "note"]
    result = pd.DataFrame([row]).reindex(columns=columns)
    result.to_csv(output, index=False)
    matches.to_csv(output.with_name(output.stem + "_matched_cells.csv"), index=False)
    output.with_name(output.stem + "_config.json").write_text(json.dumps({
        "version": VERSION,
        "spots": str(spots), "assignments": str(assignments),
        "pred_cells": str(cells), "dapi": str(dapi),
        "um_per_pixel": um_per_pixel,
        "match_radius_um": float(args.match_radius_um),
        "vec_sample": int(args.vec_sample),
        "uses_platform_prior": False,
        "join_key": ["gene", "x", "y", "z", "duplicate_occurrence"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(
        "gt=%d pred=%d matched=%d precision=%.4f recall=%.4f F1=%.4f",
        gt_cells, pred_cells, matched, precision, recall, f1)
    log.info("saved -> %s", output)
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
    args = parser.parse_args()
    validate_inputs(
        Path(args.spots), Path(args.assignments), Path(args.pred_cells), Path(args.dapi))
    scale = float(args.um_per_pixel) if args.um_per_pixel else infer_um_per_pixel(Path(args.dapi))
    if args.check_only:
        print(f"[OK] inputs; um_per_pixel={scale:.10f}; Baysor prior.type=none")
        return
    evaluate(args)


if __name__ == "__main__":
    main()
