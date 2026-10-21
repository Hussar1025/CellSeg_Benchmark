#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the ten Bering/Xenium segmentations with the ProSeg metrics.

The cell centroids of the nine ``xenium*output`` runs are reconstructed as
the mean coordinate of the transcripts assigned to each Bering cell.  The
pancreas run already supplies ``sliceB_cells.csv`` and that table is used
directly.  Official Xenium cell centroids are matched to Bering centroids with
the same inferred-radius sparse Hungarian procedure as the supplied ProSeg
evaluator.  Transcript tables are streamed so the large Xenium files do not
need to fit in memory.

All ten Bering runs used official Xenium ``transcripts.cell_id`` as their seed
source.  This is recorded as ``uses_platform_prior=True`` in every output row.

Examples::

    python bering_value_xenium.py --check-only
    python bering_value_xenium.py --dataset breast
    python bering_value_xenium.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Reuse the already tested Xenium GT readers and sparse Hungarian implementation
# shared with the Baysor evaluator.  They handle ordinary Xenium exports plus
# the Xenium 9 ZIP, Xenium 10 AnnData ZIP, and boundary-only pancreas export.
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "baysor" / "evaluate_code"))
import baysor_value_xenium as common  # noqa: E402


VERSION = "2026.09.18"
METHOD = "Bering"
DEFAULT_OUT = Path("yourpath/eval/bering")

# x0, x1, y0, y1 are global micrometres and match each Bering run_summary.json.
DATASETS: Mapping[str, Mapping[str, object]] = {
    "breast": dict(
        root="yourpath/dataset/xenium_breast",
        pred="yourpath/bering/xenium_breast_output",
        roi=(5205.6125, 6268.1125, 2382.125, 3444.625),
        label="xenium_breast",
    ),
    "colon": dict(
        root="yourpath/dataset/xenium_colon",
        pred="yourpath/bering/xenium_colon_output",
        roi=(1963.7125, 4088.7125, 2561.05, 4686.05),
        label="xenium_colon",
    ),
    "liver": dict(
        root="yourpath/dataset/xenium_liver",
        pred="yourpath/bering/xenium_liver_output",
        roi=(4137.375, 6262.375, 2380.0, 4505.0),
        label="xenium_liver",
    ),
    "mouse_brain": dict(
        root="yourpath/dataset/xenium_mouse_brain",
        pred="yourpath/bering/xenium_mouse_brain_output",
        roi=(2566.3625, 4691.3625, 1478.15, 3603.15),
        label="xenium_mouse_brain",
    ),
    "roi_crop": dict(
        root="yourpath/dataset/xenium_roi_crop",
        pred="yourpath/bering/xenium_roi_crop_output",
        roi=(3982.5, 4317.5, 2522.5, 2857.5),
        label="xenium_roi_crop",
    ),
    "xenium7": dict(
        root="yourpath/dataset/xenium7/outs",
        pred="yourpath/bering/xenium7_center10000_output",
        roi=(4692.0, 6817.0, 2169.625, 4294.625),
        label="xenium7",
        gt_tx="yourpath/dataset/xenium7/outs/_roi_cache/tx_roi_10210_22080_10000_qv20.parquet",
    ),
    "xenium8": dict(
        root="yourpath/dataset/xenium_8",
        pred="yourpath/bering/xenium_8_center10000_output",
        roi=(4675.2125, 6800.2125, 1841.1, 3966.1),
        label="xenium8",
        gt_tx="yourpath/dataset/xenium_8/_roi_cache/tx_roi_8664_22001_10000_qv20.parquet",
    ),
    "xenium9": dict(
        root="yourpath/dataset/xenium9",
        pred="yourpath/bering/xenium9_center10000_output",
        roi=(3479.475, 5604.475, 1093.7375, 3218.7375),
        label="xenium9",
        gt_zip="yourpath/dataset/xenium9/Xenium_Prime_Human_Skin_FFPE_outs.zip",
        gt_tx="yourpath/dataset/xenium9/_roi_cache/tx_roi_16374_5147_10000_qv20.parquet",
    ),
    "xenium10": dict(
        root="yourpath/dataset/xenium10",
        pred="yourpath/bering/xenium10_center10000_output",
        roi=(4369.85, 6494.85, 2921.875, 5046.875),
        label="xenium10",
        gt_h5ad_zip="yourpath/dataset/xenium10/transcriptome.zip",
        gt_tx="yourpath/dataset/xenium10/_roi_cache/tx_roi_20564_13750_10000_qv20.parquet",
    ),
    # This is the tenth Xenium result.  Its directory predates the xenium_*
    # naming convention, but run_summary.json identifies it as slice-B Xenium.
    "pancreas": dict(
        root="yourpath/dataset/xenium_pancreas_cancer",
        pred="yourpath/bering/pancreas_B_output",
        roi=(-np.inf, np.inf, -np.inf, np.inf),
        label="xenium_pancreas",
        segmentation="sliceB_transcript_assignments.csv",
        stats="sliceB_cells.csv",
        gt_boundaries="yourpath/dataset/xenium_pancreas_cancer/cell_boundaries.parquet",
    ),
}

FULL_COLUMNS = [
    "method", "Pred cell count", "GT cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall", "Detection F1",
    "Mean centroid shift", "Median centroid shift", "p95 centroid shift",
    "matched_pair_n_transcripts_vs_total_counts_pearson",
    "matched_pair_n_transcripts_vs_total_counts_spearman",
    "matched_pair_n_transcripts_vs_total_counts_mae",
    "matched_pair_n_transcripts_vs_total_counts_rmse",
    "Pred transcript rows", "GT transcript rows", "Pred transcript assignment rate",
    "Transcript matched cell pairs", "matched_cell_transcript_count_pearson",
    "matched_cell_transcript_count_spearman", "matched_cell_transcript_count_mae",
    "matched_cell_transcript_count_rmse", "matched_cell_gene_vector_mean_cosine",
    "matched_cell_gene_vector_mean_js_distance", "matched_cell_gene_vector_mean_pearson",
    "matched_cell_gene_vector_valid_pairs", "transcript_id_overlap_n",
    "transcript_assignment_accuracy_via_matched_cells", "dataset",
    "uses_platform_prior", "Overall correct fraction",
]

SELECTED_COLUMNS = [
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

log = logging.getLogger("eval_bering_xenium")
SparseCounts = DefaultDict[Tuple[int, str], int]


def setup_log(out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log.addHandler(stream)
    file_handler = logging.FileHandler(out / f"eval_bering_{name}.log", mode="w")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


def transcript_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield canonical cell/gene/x/y/transcript columns from Parquet or CSV."""
    aliases = {
        "transcript_id": ("transcript_id", "molecule_id", "id"),
        "cell_id": ("cell_id", "cell", "platform_cell"),
        "gene": ("feature_name", "gene", "target"),
        "x": ("x_location", "x", "x_um", "global_x"),
        "y": ("y_location", "y", "y_um", "global_y"),
        "qv": ("qv",),
        "is_gene": ("is_gene",),
        "is_noise": ("is_noise",),
    }
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        chosen = {key: next((alias for alias in values if alias in names), None)
                  for key, values in aliases.items()}
        chosen = {key: value for key, value in chosen.items() if value is not None}
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(chosen.values())):
            yield batch.to_pandas().rename(columns={value: key for key, value in chosen.items()})
    else:
        header = pd.read_csv(path, nrows=0)
        names = set(header.columns)
        chosen = {key: next((alias for alias in values if alias in names), None)
                  for key, values in aliases.items()}
        chosen = {key: value for key, value in chosen.items() if value is not None}
        for frame in pd.read_csv(path, usecols=list(chosen.values()), chunksize=batch_size):
            yield frame.rename(columns={value: key for key, value in chosen.items()})


def in_roi(frame: pd.DataFrame, roi: Tuple[float, float, float, float]) -> pd.Series:
    x = pd.to_numeric(frame["x"], errors="coerce")
    y = pd.to_numeric(frame["y"], errors="coerce")
    return x.ge(roi[0]) & x.lt(roi[1]) & y.ge(roi[2]) & y.lt(roi[3])


def derive_pred_cells(
    assignments: Path,
    roi: Tuple[float, float, float, float],
    batch_size: int,
) -> pd.DataFrame:
    """Reconstruct one Bering centroid/count row per assigned cell."""
    partial: List[pd.DataFrame] = []
    for frame in transcript_batches(assignments, batch_size):
        required = {"cell_id", "x", "y"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{assignments} lacks {required - set(frame.columns)}")
        frame["cell"] = frame.cell_id.map(common.norm_cell)
        frame["x"] = pd.to_numeric(frame.x, errors="coerce")
        frame["y"] = pd.to_numeric(frame.y, errors="coerce")
        keep = frame.cell.ne("") & in_roi(frame, roi)
        if "is_noise" in frame:
            keep &= ~frame.is_noise.fillna(False).astype(bool)
        if not keep.any():
            continue
        grouped = (frame.loc[keep, ["cell", "x", "y"]]
                   .assign(x_sum=lambda x: x.x, y_sum=lambda x: x.y, n_transcripts=1)
                   .groupby("cell", sort=False)[["x_sum", "y_sum", "n_transcripts"]]
                   .sum().reset_index())
        partial.append(grouped)
    if not partial:
        return pd.DataFrame(columns=["cell", "x", "y", "n_transcripts"])
    cells = (pd.concat(partial, ignore_index=True)
             .groupby("cell", sort=False)[["x_sum", "y_sum", "n_transcripts"]]
             .sum().reset_index())
    cells["x"] = cells.x_sum / cells.n_transcripts
    cells["y"] = cells.y_sum / cells.n_transcripts
    return cells[["cell", "x", "y", "n_transcripts"]]


def load_pred_cells(
    cfg: Mapping[str, object],
    pred_root: Path,
    assignments: Path,
    roi: Tuple[float, float, float, float],
    batch_size: int,
    stats_override: Optional[str],
) -> pd.DataFrame:
    stats_name = stats_override or cfg.get("stats")
    if stats_name:
        stats_path = Path(str(stats_name)) if stats_override else pred_root / str(stats_name)
        frame = pd.read_parquet(stats_path) if stats_path.suffix == ".parquet" else pd.read_csv(stats_path)
        lower = {str(column).lower(): column for column in frame.columns}

        def find_column(*aliases: str) -> str:
            for alias in aliases:
                if alias.lower() in lower:
                    return str(lower[alias.lower()])
            raise ValueError(f"{stats_path} lacks one of {aliases}; columns={list(frame.columns)}")

        cell_col = find_column("cell", "cell_id")
        x_col = find_column("x", "cx", "centroid_x", "x_centroid")
        y_col = find_column("y", "cy", "centroid_y", "y_centroid")
        count_col = find_column("n_transcripts", "n_counts", "transcript_count")
        frame = frame.rename(columns={
            cell_col: "cell", x_col: "x", y_col: "y", count_col: "n_transcripts",
        })
        frame["cell"] = frame.cell.map(common.norm_cell)
        for column in ("x", "y", "n_transcripts", "area"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        keep = frame.cell.ne("") & in_roi(frame, roi)
        return frame.loc[keep].reset_index(drop=True)
    return derive_pred_cells(assignments, roi, batch_size)


def add_sparse_counts(
    store: SparseCounts,
    slots: np.ndarray,
    genes: np.ndarray,
) -> None:
    valid = slots >= 0
    if not valid.any():
        return
    grouped = (pd.DataFrame({"slot": slots[valid], "gene": genes[valid]})
               .groupby(["slot", "gene"], sort=False).size())
    for (slot, gene), count in grouped.items():
        store[(int(slot), str(gene))] += int(count)


def sparse_gene_metrics(pred: SparseCounts, gt: SparseCounts, n_pairs: int) -> Dict[str, object]:
    """Compute matched-cell gene-vector metrics without a huge dense matrix."""
    from scipy.spatial.distance import jensenshannon

    pred_by_pair: DefaultDict[int, Dict[str, int]] = defaultdict(dict)
    gt_by_pair: DefaultDict[int, Dict[str, int]] = defaultdict(dict)
    for (slot, gene), value in pred.items():
        pred_by_pair[slot][gene] = value
    for (slot, gene), value in gt.items():
        gt_by_pair[slot][gene] = value
    cosine: List[float] = []
    js_distance: List[float] = []
    correlations: List[float] = []
    for slot in range(n_pairs):
        genes = set(pred_by_pair.get(slot, {})) | set(gt_by_pair.get(slot, {}))
        if not genes:
            continue
        p = np.fromiter((pred_by_pair[slot].get(gene, 0) for gene in genes), dtype=float)
        g = np.fromiter((gt_by_pair[slot].get(gene, 0) for gene in genes), dtype=float)
        denom = np.linalg.norm(p) * np.linalg.norm(g)
        if denom <= 0:
            continue
        cosine.append(float(np.dot(p, g) / denom))
        distance = jensenshannon(p / p.sum(), g / g.sum(), base=2)
        if np.isfinite(distance):
            js_distance.append(float(distance))
        if len(p) >= 3 and p.std() > 0 and g.std() > 0:
            correlations.append(float(np.corrcoef(p, g)[0, 1]))
    return {
        "matched_cell_gene_vector_mean_cosine": float(np.mean(cosine)) if cosine else np.nan,
        "matched_cell_gene_vector_mean_js_distance": float(np.mean(js_distance)) if js_distance else np.nan,
        "matched_cell_gene_vector_mean_pearson": float(np.mean(correlations)) if correlations else np.nan,
        "matched_cell_gene_vector_valid_pairs": len(cosine),
    }


def configured_gt_transcripts(cfg: Mapping[str, object], root: Path) -> Path:
    if cfg.get("gt_tx"):
        return Path(str(cfg["gt_tx"]))
    path = common.find_file(root, "transcripts.parquet", "transcripts.csv.gz", "transcripts.csv")
    if path is None:
        raise FileNotFoundError(f"official transcripts not found under {root}")
    return path


def prediction_input_rows(pred_root: Path, assigned_rows: int) -> int:
    """Read the pre-assignment denominator recorded by the Bering run."""
    summary = pred_root / "run_summary.json"
    if not summary.is_file():
        return assigned_rows
    info = json.loads(summary.read_text(encoding="utf-8"))
    for key in ("n_input_transcripts", "n_transcripts_input_foreground_plus_bg"):
        if key in info:
            return max(int(info[key]), assigned_rows)
    return assigned_rows


def evaluate(name: str, args: argparse.Namespace) -> Dict[str, object]:
    cfg = DATASETS[name]
    root = Path(args.data_root or str(cfg["root"]))
    pred_root = Path(args.run_dir or str(cfg["pred"]))
    roi = tuple(args.roi) if args.roi else tuple(cfg["roi"])
    out = Path(args.out_dir) / name
    setup_log(out, name)

    segmentation_name = args.segmentation or str(cfg.get("segmentation", "transcript_assignments.parquet"))
    assignments = Path(segmentation_name) if args.segmentation else pred_root / segmentation_name
    gt_tx = configured_gt_transcripts(cfg, root)
    if not assignments.is_file() or not gt_tx.is_file():
        raise FileNotFoundError(f"missing Bering/GT transcript input: {assignments}; {gt_tx}")

    pred = load_pred_cells(cfg, pred_root, assignments, roi, args.batch_size, args.stats)
    gt = common.load_configured_gt_cells(dict(cfg), root, roi, Path(args.out_dir) / "_cache")
    if pred.empty or gt.empty:
        raise ValueError(f"empty prediction or GT for {name}: pred={len(pred)}, gt={len(gt)}")
    if pred.cell.duplicated().any():
        raise ValueError(f"duplicate Bering cell IDs for {name}")

    px, py = pred.x.to_numpy(float), pred.y.to_numpy(float)
    gx, gy = gt._x.to_numpy(float), gt._y.to_numpy(float)
    radius = args.match_radius if args.match_radius is not None else common.infer_radius(gx, gy)
    matched_pred, matched_gt, shifts = common.sparse_hungarian(px, py, gx, gy, radius)
    n_match = len(matched_pred)
    pred_pair_slot = {str(pred.cell.iloc[p]): slot for slot, p in enumerate(matched_pred)}
    gt_pair_slot = {str(gt._id.iloc[g]): slot for slot, g in enumerate(matched_gt)}
    matched_gt_for_slot = matched_gt.astype(np.int64)

    pred_counts: SparseCounts = defaultdict(int)
    pred_coord_parts: List[np.ndarray] = []
    pred_target_parts: List[np.ndarray] = []
    assigned_pred_rows = 0
    for frame in transcript_batches(assignments, args.batch_size):
        if not {"cell_id", "x", "y"}.issubset(frame.columns):
            raise ValueError(f"{assignments} lacks cell_id/x/y")
        frame["x"] = pd.to_numeric(frame.x, errors="coerce")
        frame["y"] = pd.to_numeric(frame.y, errors="coerce")
        cells = frame.cell_id.map(common.norm_cell)
        keep = cells.ne("") & in_roi(frame, roi)
        if "is_noise" in frame:
            keep &= ~frame.is_noise.fillna(False).astype(bool)
        if not keep.any():
            continue
        cells = cells.loc[keep]
        assigned_pred_rows += int(keep.sum())
        slots = cells.map(pred_pair_slot).fillna(-1).to_numpy(np.int64)
        matched = slots >= 0
        if matched.any():
            x = frame.loc[keep, "x"].to_numpy(float)[matched]
            y = frame.loc[keep, "y"].to_numpy(float)[matched]
            keys = common.coord_key(x, y)
            pred_coord_parts.append(keys)
            pred_target_parts.append(keys * (len(gt) + 1) + matched_gt_for_slot[slots[matched]])
        if "gene" in frame:
            genes = frame.loc[keep, "gene"].astype(str).to_numpy()
            add_sparse_counts(pred_counts, slots, genes)

    coord_index = np.unique(np.concatenate(pred_coord_parts)) if pred_coord_parts else np.zeros(0, np.int64)
    target_index = np.unique(np.concatenate(pred_target_parts)) if pred_target_parts else np.zeros(0, np.int64)
    coord_index.sort()
    target_index.sort()

    gt_index = {str(cell): i for i, cell in enumerate(gt._id)}
    gt_counts: SparseCounts = defaultdict(int)
    gt_stream_counts = np.zeros(len(gt), dtype=np.int64)
    gt_rows = overlap = correct = 0
    for frame in transcript_batches(gt_tx, args.batch_size):
        frame = common.qc_transcripts(frame, roi, args.qv_min)
        if frame.empty:
            continue
        gt_rows += len(frame)
        if "cell_id" not in frame:
            continue
        cells = frame.cell_id.map(common.norm_cell)
        gt_cell_index = cells.map(gt_index).fillna(-1).to_numpy(np.int64)
        keys = common.coord_key(frame.x, frame.y)
        query = keys * (len(gt) + 1) + np.maximum(gt_cell_index, 0)
        if len(coord_index):
            pos = np.searchsorted(coord_index, keys)
            safe = np.minimum(pos, len(coord_index) - 1)
            any_hit = (pos < len(coord_index)) & (coord_index[safe] == keys)
        else:
            any_hit = np.zeros(len(frame), dtype=bool)
        if len(target_index):
            pos = np.searchsorted(target_index, query)
            safe = np.minimum(pos, len(target_index) - 1)
            target_hit = (pos < len(target_index)) & (target_index[safe] == query)
        else:
            target_hit = np.zeros(len(frame), dtype=bool)
        valid_gt = gt_cell_index >= 0
        overlap += int((any_hit & valid_gt).sum())
        correct += int((target_hit & valid_gt).sum())
        if valid_gt.any():
            gt_stream_counts += np.bincount(gt_cell_index[valid_gt], minlength=len(gt))
        if "gene" in frame:
            slots = cells.map(gt_pair_slot).fillna(-1).to_numpy(np.int64)
            add_sparse_counts(gt_counts, slots, frame.gene.astype(str).to_numpy())

    if gt["_total"].isna().all():
        gt["_total"] = gt_stream_counts.astype(float)

    precision = n_match / max(len(pred), 1)
    recall = n_match / max(len(gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    pred_total_rows = prediction_input_rows(pred_root, assigned_pred_rows)
    assignment_rate = assigned_pred_rows / max(pred_total_rows, 1)
    assignment_accuracy = correct / overlap if overlap else np.nan
    row: Dict[str, object] = {
        "method": METHOD,
        "dataset": cfg["label"],
        "Pred cell count": len(pred),
        "GT cell count": len(gt),
        "Cell count ratio": len(pred) / max(len(gt), 1),
        "Matched cell count": n_match,
        "Detection precision": precision,
        "Detection recall": recall,
        "Detection F1": f1,
        "Mean centroid shift": float(shifts.mean()) if n_match else np.nan,
        "Median centroid shift": float(np.median(shifts)) if n_match else np.nan,
        "p95 centroid shift": float(np.percentile(shifts, 95)) if n_match else np.nan,
        "Pred transcript rows": pred_total_rows,
        "GT transcript rows": gt_rows,
        "Pred transcript assignment rate": assignment_rate,
        "Transcript matched cell pairs": n_match,
        "transcript_id_overlap_n": overlap,
        "transcript_assignment_accuracy_via_matched_cells": assignment_accuracy,
        "Overall correct fraction": assignment_rate * assignment_accuracy if overlap else np.nan,
        "median_area_um2": float(pred.area.median()) if "area" in pred else np.nan,
        "median_diameter_um": (float(2 * np.sqrt(pred.area.median() / np.pi))
                               if "area" in pred else np.nan),
        "coverage_fraction": (float(pred.area.sum() / ((roi[1] - roi[0]) * (roi[3] - roi[2])))
                              if "area" in pred and np.isfinite(roi).all() else np.nan),
        "uses_platform_prior": True,
    }
    pred_n = pred.n_transcripts.to_numpy(float)
    gt_n = gt._total.to_numpy(float)
    if n_match:
        row.update(common.pair_metrics(
            pred_n[matched_pred], gt_n[matched_gt],
            "matched_pair_n_transcripts_vs_total_counts",
        ))
        row.update(common.pair_metrics(
            pred_n[matched_pred], gt_n[matched_gt],
            "matched_cell_transcript_count",
        ))
    else:
        for prefix in ("matched_pair_n_transcripts_vs_total_counts", "matched_cell_transcript_count"):
            row.update({f"{prefix}_{metric}": np.nan
                        for metric in ("pearson", "spearman", "mae", "rmse")})
    row.update(sparse_gene_metrics(pred_counts, gt_counts, n_match))

    tag = f"bering_{name}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{key: row.get(key, np.nan) for key in FULL_COLUMNS}]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame([{key: row.get(key, np.nan) for key in SELECTED_COLUMNS}]).to_csv(
        out / f"{tag}_sparse_hungarian_qc_metrics_selected.csv", index=False)
    if n_match:
        pd.DataFrame({
            "pred_id": pred.cell.iloc[matched_pred].to_numpy(),
            "gt_id": gt._id.iloc[matched_gt].to_numpy(),
            "centroid_shift_um": shifts,
            "pred_n_transcripts": pred_n[matched_pred],
            "gt_total_counts": gt_n[matched_gt],
        }).to_csv(out / f"{tag}_matched_pairs.csv", index=False)
    log.info(
        "%s: pred=%d gt=%d matched=%d precision=%.4f recall=%.4f F1=%.4f",
        name, len(pred), len(gt), n_match, precision, recall, f1,
    )
    return row


def validate_inputs() -> None:
    """Cheap path/schema validation; does not run an evaluation."""
    import pyarrow.parquet as pq

    for name, cfg in DATASETS.items():
        pred_root = Path(str(cfg["pred"]))
        assignments = pred_root / str(cfg.get("segmentation", "transcript_assignments.parquet"))
        root = Path(str(cfg["root"]))
        gt_tx = configured_gt_transcripts(cfg, root)
        required = [assignments, gt_tx, pred_root / "run_summary.json"]
        if cfg.get("stats"):
            required.append(pred_root / str(cfg["stats"]))
        if cfg.get("gt_zip"):
            required.append(Path(str(cfg["gt_zip"])))
        elif cfg.get("gt_h5ad_zip"):
            required.append(Path(str(cfg["gt_h5ad_zip"])))
        elif cfg.get("gt_boundaries"):
            required.append(Path(str(cfg["gt_boundaries"])))
        else:
            cells = common.find_file(root, "cells.parquet", "cells.csv.gz", "cells.csv")
            if cells is None:
                raise FileNotFoundError(f"official cells table not found under {root}")
            required.append(cells)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: missing files: {missing}")
        if assignments.suffix == ".parquet":
            columns = set(pq.ParquetFile(assignments).schema_arrow.names)
        else:
            columns = set(pd.read_csv(assignments, nrows=0).columns)
        needed = {"cell_id", "x", "y"}
        if not needed.issubset(columns):
            raise ValueError(f"{assignments} lacks {needed - columns}")
        print(f"[OK] {name:12s} prediction={pred_root.name} GT={gt_tx.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), default="breast")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--segmentation")
    parser.add_argument("--stats")
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--qv-min", type=float, default=20.0)
    parser.add_argument("--match-radius", type=float)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print(VERSION)
        return
    if args.check_only:
        validate_inputs()
        return
    if args.all and any(getattr(args, key) is not None
                        for key in ("run_dir", "segmentation", "stats", "data_root", "roi")):
        parser.error("--all cannot be combined with path/ROI overrides")
    names = list(DATASETS) if args.all else [args.dataset]
    rows = [evaluate(name, args) for name in names]
    if len(rows) > 1:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{key: row.get(key, np.nan) for key in FULL_COLUMNS} for row in rows]).to_csv(
            out / "bering_xenium_evaluation.csv", index=False)
        pd.DataFrame([{key: row.get(key, np.nan) for key in SELECTED_COLUMNS} for row in rows]).to_csv(
            out / "bering_xenium_all_metrics_selected.csv", index=False)


if __name__ == "__main__":
    main()
