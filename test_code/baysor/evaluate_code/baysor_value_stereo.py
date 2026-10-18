#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate four Baysor Stereo-seq segmentations with the reference 19 metrics.

Three dorsal-midbrain samples are evaluated over the whole section against
their official ``*_GEM_CellBin.tsv.gz`` cell labels.  Mouse embryo is evaluated
only in the exact centre 3000 x 3000 DNB-pixel ROI used by Baysor, against the
official ``D05103C4D4.adjusted.cellbin.gef`` CellBin result.

Important: ``dataset/stereo_mouse_embryo/segmentation_outputs_center3000`` is
never read; it contains predictions from other methods, not official GT.

Stereo-seq coordinates are native Bin1/DNB units (0.5 um per unit).  Matching
therefore happens after both prediction and GT coordinates are converted to um.

Examples::

    python baysor_value_stereo.py --check-only
    python baysor_value_stereo.py --all
    python baysor_value_stereo.py --dataset mouse_embryo
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


BIN_UM = 0.5
MATCH_RADIUS_UM = 15.0
OUT_DIR = Path("/data/qiuyijia/eval/baysor")
BAYSOR_ROOT = Path("/data/qiuyijia/baysor/stereo_outputs")
STEREO_ROOT = Path("/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all")
MOUSE_ROOT = Path("/data/qiuyijia/dataset/stereo_mouse_embryo")
MOUSE_PREPARED = Path(
    "/data/qiuyijia/baysor/prepared_stereoseq/mouse_embryo_center_3000/"
    "D05103C4D4_center_3000.metadata.json"
)
SEP = "=" * 76

DATASETS: Mapping[str, Mapping[str, object]] = {
    "E14.5_E1S3": {
        "kind": "gem",
        "gt": STEREO_ROOT / "E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        "pred": BAYSOR_ROOT / "stereoseq/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin",
    },
    "E16.5_E1S3": {
        "kind": "gem",
        "gt": STEREO_ROOT / "E16.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        "pred": BAYSOR_ROOT / "stereoseq/E16.5_E1S3_Dorsal_Midbrain_GEM_CellBin",
    },
    "E16.5_E2S7": {
        "kind": "gem",
        "gt": STEREO_ROOT / "E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        "pred": BAYSOR_ROOT / "stereoseq/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin",
    },
    "mouse_embryo": {
        "kind": "gef",
        "gt": MOUSE_ROOT / "D05103C4D4.adjusted.cellbin.gef",
        "pred": BAYSOR_ROOT / "mouse_embryo_center_3000",
        "roi_meta": MOUSE_PREPARED,
    },
}

METRICS = [
    "method", "dataset", "gt_cells", "pred_cells", "cell_ratio", "matched",
    "precision", "recall", "f1", "loc_mean_um", "loc_median_um", "loc_p95_um",
    "count_pearson", "count_spearman", "count_mae", "count_rmse", "vec_cosine",
    "vec_js_dist", "vec_pearson", "assign_overlap", "assign_accuracy",
]
VectorCounts = MutableMapping[Tuple[int, str], int]
Labeler = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class GroundTruth:
    cells: pd.DataFrame             # cell, x_um, y_um, gt_count
    vectors: VectorCounts           # filled after sampled pairs are known
    labeler: Labeler                # molecule x/y native units + gene -> GT row
    cleanup: Optional[Callable[[], None]] = None


def normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "0", "-1", "nan", "None"} else text


def parquet_batches(path: Path, columns: Sequence[str], batch_size: int) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(path)
    missing = set(columns) - set(pf.schema.names)
    if missing:
        raise KeyError(f"{path} lacks columns {sorted(missing)}; found {pf.schema.names}")
    for batch in pf.iter_batches(columns=list(columns), batch_size=batch_size):
        yield batch.to_pandas()


def load_prediction_cells(path: Path) -> pd.DataFrame:
    pred = pq.read_table(path, columns=["cell", "x", "y", "n_transcripts"]).to_pandas()
    pred["cell"] = pred.cell.map(normalize_cell)
    for col in ("x", "y", "n_transcripts"):
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    pred = pred.loc[pred.cell.ne("") & pred.x.notna() & pred.y.notna()].reset_index(drop=True)
    pred["x_um"] = pred.x * BIN_UM
    pred["y_um"] = pred.y * BIN_UM
    if pred.cell.duplicated().any():
        raise ValueError(f"duplicate predicted cells in {path}")
    return pred


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match exactly like the supplied ProSeg Stereo-seq evaluator."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    distance, gt_index = cKDTree(gt_xy).query(pred_xy, k=1, distance_upper_bound=radius)
    best: Dict[int, Tuple[int, float]] = {}
    for pred_i, (dist, gt_i) in enumerate(zip(distance, gt_index)):
        if not np.isfinite(dist) or dist >= radius:
            continue
        gt_i = int(gt_i)
        if gt_i not in best or dist < best[gt_i][1]:
            best[gt_i] = pred_i, float(dist)
    pairs = sorted((pred_i, gt_i, dist) for gt_i, (pred_i, dist) in best.items())
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    pred_i, gt_i, distance = zip(*pairs)
    return np.asarray(pred_i), np.asarray(gt_i), np.asarray(distance)


def read_gem(path: Path) -> pd.DataFrame:
    gem = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    rename = {}
    for col in gem.columns:
        low = str(col).lower()
        if low in ("geneid", "genename", "gene"):
            rename[col] = "gene"
        elif low == "x":
            rename[col] = "x"
        elif low == "y":
            rename[col] = "y"
        elif low in ("midcounts", "midcount", "count", "counts"):
            rename[col] = "count"
        elif low in ("cell", "cellid", "label"):
            rename[col] = "cell"
    gem = gem.rename(columns=rename)
    required = {"gene", "x", "y", "cell"}
    if not required.issubset(gem.columns):
        raise KeyError(f"{path} lacks {sorted(required - set(gem.columns))}")
    if "count" not in gem:
        gem["count"] = 1
    gem["cell"] = pd.to_numeric(gem.cell, errors="coerce").fillna(0).astype(np.int64)
    for col in ("x", "y", "count"):
        gem[col] = pd.to_numeric(gem[col], errors="raise")
    gem["gene"] = gem.gene.astype(str)
    return gem.loc[gem.cell > 0, ["gene", "x", "y", "count", "cell"]].reset_index(drop=True)


def load_gem_gt(path: Path) -> Tuple[GroundTruth, pd.DataFrame]:
    gem = read_gem(path)
    gem["wx"] = gem.x * gem["count"]
    gem["wy"] = gem.y * gem["count"]
    cells = gem.groupby("cell", sort=False).agg(
        wx=("wx", "sum"), wy=("wy", "sum"), gt_count=("count", "sum")).reset_index()
    cells["x_um"] = cells.wx / cells.gt_count * BIN_UM
    cells["y_um"] = cells.wy / cells.gt_count * BIN_UM
    cells = cells[["cell", "x_um", "y_um", "gt_count"]].reset_index(drop=True)
    cell_to_row = pd.Series(np.arange(len(cells), dtype=np.int64), index=cells.cell).to_dict()
    gem["gt_row"] = gem.cell.map(cell_to_row).astype(np.int64)

    # Exact official assignment lookup by native coordinate and gene.  Conflicts
    # are exceptionally unlikely; keep the first deterministic official label.
    unique = gem.drop_duplicates(["x", "y", "gene"], keep="first")
    index = pd.MultiIndex.from_frame(unique[["x", "y", "gene"]])
    lookup = pd.Series(unique.gt_row.to_numpy(np.int64), index=index)

    def labeler(x: np.ndarray, y: np.ndarray, gene: np.ndarray) -> np.ndarray:
        query = pd.MultiIndex.from_arrays([
            np.rint(x).astype(np.int64), np.rint(y).astype(np.int64), gene.astype(str)])
        return lookup.reindex(query).fillna(-1).to_numpy(np.int64)

    return GroundTruth(cells=cells, vectors=defaultdict(int), labeler=labeler), gem


def decode_gene_names(records: np.ndarray) -> np.ndarray:
    names = records.dtype.names or ()
    out = []
    for row in records:
        raw = row["geneName"] if "geneName" in names and row["geneName"] else row["geneID"]
        out.append(bytes(raw).split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    return np.asarray(out, dtype=object)


def load_mouse_gt(path: Path, roi_meta: Path) -> Tuple[GroundTruth, Dict[str, object]]:
    meta = json.loads(roi_meta.read_text())
    bounds = meta["roi_bounds_half_open"]
    x0, x1 = int(bounds["x_min"]), int(bounds["x_max"])
    y0, y1 = int(bounds["y_min"]), int(bounds["y_max"])
    if (x1 - x0, y1 - y0) != (3000, 3000):
        raise ValueError(f"mouse ROI is not 3000x3000: {bounds}")

    handle = h5py.File(path, "r")
    all_cells = handle["cellBin/cell"][:]
    keep = (
        (all_cells["x"] >= x0) & (all_cells["x"] < x1)
        & (all_cells["y"] >= y0) & (all_cells["y"] < y1))
    source_index = np.flatnonzero(keep)
    selected = all_cells[keep]
    cells = pd.DataFrame({
        "cell": selected["id"].astype(np.int64),
        "x_um": selected["x"].astype(float) * BIN_UM,
        "y_um": selected["y"].astype(float) * BIN_UM,
        "gt_count": selected["expCount"].astype(float),
    })

    # Rasterise only official CellBin borders whose centers are in the ROI.
    # This is the official adjusted.cellbin.gef, not another method's output.
    import cv2

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
    borders = handle["cellBin/cellBorder"]
    sentinel = 32767
    for start in range(0, len(source_index), 4096):
        idx = source_index[start:start + 4096]
        border_chunk = borders[idx]
        cell_chunk = all_cells[idx]
        for local_offset, (cell, relative) in enumerate(zip(cell_chunk, border_chunk)):
            valid = np.all(relative != sentinel, axis=1)
            relative = np.asarray(relative[valid], dtype=np.int32)
            if len(relative) < 3:
                continue
            points = relative.copy()
            points[:, 0] += int(cell["x"]) - x0
            points[:, 1] += int(cell["y"]) - y0
            gt_row = start + local_offset
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], int(gt_row + 1))

    def labeler(x: np.ndarray, y: np.ndarray, gene: np.ndarray) -> np.ndarray:
        del gene
        ix = np.rint(x).astype(np.int64) - x0
        iy = np.rint(y).astype(np.int64) - y0
        inside = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
        labels = np.full(len(ix), -1, dtype=np.int64)
        labels[inside] = mask[iy[inside], ix[inside]].astype(np.int64) - 1
        return labels

    context: Dict[str, object] = {
        "handle": handle,
        "selected": selected,
        "gene_names": decode_gene_names(handle["cellBin/gene"][:]),
        "roi": (x0, x1, y0, y1),
        "mask": mask,
    }
    gt = GroundTruth(cells=cells, vectors=defaultdict(int), labeler=labeler, cleanup=handle.close)
    return gt, context


def fill_gem_vectors(gem: pd.DataFrame, sampled_gt: np.ndarray, vectors: VectorCounts) -> None:
    row_to_slot = {int(row): slot for slot, row in enumerate(sampled_gt)}
    wanted = gem.loc[gem.gt_row.isin(row_to_slot)]
    grouped = wanted.groupby(["gt_row", "gene"], sort=False)["count"].sum()
    for (gt_row, gene), count in grouped.items():
        vectors[(row_to_slot[int(gt_row)], str(gene))] += int(count)


def fill_mouse_vectors(context: Mapping[str, object], sampled_gt: np.ndarray, vectors: VectorCounts) -> None:
    handle = context["handle"]
    selected = context["selected"]
    gene_names = context["gene_names"]
    cell_exp = handle["cellBin/cellExp"]
    for slot, gt_row in enumerate(sampled_gt):
        cell = selected[int(gt_row)]
        start = int(cell["offset"])
        stop = start + int(cell["geneCount"])
        records = cell_exp[start:stop]
        for gene_id, count in zip(records["geneID"], records["count"]):
            vectors[(slot, str(gene_names[int(gene_id)]))] += int(count)


def add_grouped_counts(store: VectorCounts, slots: np.ndarray, genes: np.ndarray) -> None:
    valid = slots >= 0
    if not valid.any():
        return
    grouped = pd.DataFrame({"slot": slots[valid], "gene": genes[valid]}).groupby(
        ["slot", "gene"], sort=False).size()
    for (slot, gene), count in grouped.items():
        store[(int(slot), str(gene))] += int(count)


def load_prediction_vectors_and_assignment(
    molecules: Path,
    all_cells: Sequence[str],
    sampled_cells: Sequence[str],
    matched_cells: Sequence[str],
    matched_gt: np.ndarray,
    labeler: Labeler,
    batch_size: int,
) -> Tuple[VectorCounts, set, float, float, int]:
    known = set(all_cells)
    vector_lookup = {cell: slot for slot, cell in enumerate(sampled_cells)}
    pair_lookup = {cell: slot for slot, cell in enumerate(matched_cells)}
    vectors: VectorCounts = defaultdict(int)
    composition: VectorCounts = defaultdict(int)
    genes_seen = set()
    pair_total = np.zeros(len(matched_cells), dtype=np.int64)
    pair_correct = np.zeros(len(matched_cells), dtype=np.int64)
    assigned_rows = 0

    columns = ["cell", "gene", "x", "y", "is_noise"]
    for frame in parquet_batches(molecules, columns, batch_size):
        cells = frame.cell.map(normalize_cell)
        noise = frame.is_noise.fillna(False).astype(bool).to_numpy()
        valid = cells.isin(known).to_numpy() & ~noise
        assigned_rows += int(valid.sum())
        if not valid.any():
            continue
        kept_cells = cells.to_numpy()[valid]
        genes = frame.gene.astype(str).to_numpy()[valid]
        x = pd.to_numeric(frame.x, errors="coerce").to_numpy(float)[valid]
        y = pd.to_numeric(frame.y, errors="coerce").to_numpy(float)[valid]
        genes_seen.update(genes)

        vector_slots = np.asarray([vector_lookup.get(c, -1) for c in kept_cells], dtype=np.int32)
        add_grouped_counts(vectors, vector_slots, genes)

        pair_slots = np.asarray([pair_lookup.get(c, -1) for c in kept_cells], dtype=np.int32)
        comparable = pair_slots >= 0
        if not comparable.any():
            continue
        official = labeler(x[comparable], y[comparable], genes[comparable])
        used_slots = pair_slots[comparable]
        pair_total += np.bincount(used_slots, minlength=len(pair_total))
        correct = official == matched_gt[used_slots]
        if correct.any():
            pair_correct += np.bincount(used_slots[correct], minlength=len(pair_correct))
        add_grouped_counts(composition, used_slots, official.astype(str))

    valid_pairs = pair_total > 0
    if valid_pairs.any():
        overlap = float(np.mean(pair_correct[valid_pairs] / pair_total[valid_pairs]))
        majority = np.zeros(len(pair_total), dtype=np.int64)
        for (slot, _label), count in composition.items():
            majority[slot] = max(majority[slot], count)
        accuracy = float(np.mean(majority[valid_pairs] / pair_total[valid_pairs]))
    else:
        overlap = accuracy = np.nan
    return vectors, genes_seen, overlap, accuracy, assigned_rows


def count_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    valid = np.isfinite(pred) & np.isfinite(gt)
    pearson = spearman = np.nan
    if valid.sum() >= 3 and pred[valid].std() > 0 and gt[valid].std() > 0:
        pearson = float(pearsonr(pred[valid], gt[valid])[0])
        spearman = float(spearmanr(pred[valid], gt[valid])[0])
    mae = float(np.mean(np.abs(pred[valid] - gt[valid]))) if valid.any() else np.nan
    rmse = float(np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2))) if valid.any() else np.nan
    return pearson, spearman, mae, rmse


def vector_metrics(
    pred: VectorCounts,
    gt: VectorCounts,
    n_pairs: int,
    gene_names: Sequence[str],
) -> Tuple[float, float, float]:
    genes = set(gene_names)
    n_genes = len(genes)
    if n_genes == 0:
        return np.nan, np.nan, np.nan
    pred_by_slot: Dict[int, Dict[str, float]] = defaultdict(dict)
    gt_by_slot: Dict[int, Dict[str, float]] = defaultdict(dict)
    for (slot, gene), count in pred.items():
        if gene in genes:
            pred_by_slot[slot][gene] = float(count)
    for (slot, gene), count in gt.items():
        if gene in genes:
            gt_by_slot[slot][gene] = float(count)

    cosines, js_distances, correlations = [], [], []
    for slot in range(n_pairs):
        pv, gv = pred_by_slot.get(slot, {}), gt_by_slot.get(slot, {})
        if not pv or not gv:
            continue
        common = set(pv) & set(gv)
        dot = sum(pv[g] * gv[g] for g in common)
        p2 = sum(v * v for v in pv.values())
        g2 = sum(v * v for v in gv.values())
        if p2 <= 0 or g2 <= 0:
            continue
        cosines.append(float(dot / np.sqrt(p2 * g2)))

        union = sorted(set(pv) | set(gv))
        pa = np.asarray([pv.get(g, 0.0) for g in union])
        ga = np.asarray([gv.get(g, 0.0) for g in union])
        js_distances.append(float(jensenshannon(pa / pa.sum(), ga / ga.sum())))

        ps, gs = sum(pv.values()), sum(gv.values())
        numerator = dot - ps * gs / n_genes
        pvar = p2 - ps * ps / n_genes
        gvar = g2 - gs * gs / n_genes
        if pvar > 0 and gvar > 0:
            correlations.append(float(numerator / np.sqrt(pvar * gvar)))
    mean = lambda values: float(np.mean(values)) if values else np.nan
    return mean(cosines), mean(js_distances), mean(correlations)


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if np.isfinite(value) else np.nan


def validate_one(name: str, cfg: Mapping[str, object]) -> None:
    gt_path, pred_dir = Path(cfg["gt"]), Path(cfg["pred"])
    required = [gt_path, pred_dir / "cells.parquet", pred_dir / "molecules.parquet"]
    if cfg["kind"] == "gef":
        required.append(Path(cfg["roi_meta"]))
        forbidden = MOUSE_ROOT / "segmentation_outputs_center3000"
        if any(str(p).startswith(str(forbidden)) for p in required):
            raise RuntimeError("forbidden non-GT segmentation_outputs_center3000 was selected")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    pq.ParquetFile(pred_dir / "cells.parquet")
    pq.ParquetFile(pred_dir / "molecules.parquet")
    detail = "whole section"
    if cfg["kind"] == "gef":
        bounds = json.loads(Path(cfg["roi_meta"]).read_text())["roi_bounds_half_open"]
        detail = f"official adjusted.cellbin.gef; ROI={bounds}"
    print(f"[OK] {name:16s} {detail}")


def evaluate(name: str, cfg: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    pred_dir = Path(cfg["pred"])
    print(f"\n{SEP}\nBaysor x Stereo-seq: {name}\n{SEP}")
    pred = load_prediction_cells(pred_dir / "cells.parquet")

    gem = None
    mouse_context = None
    if cfg["kind"] == "gem":
        gt, gem = load_gem_gt(Path(cfg["gt"]))
        print("GT: official GEM CellBin labels (whole section)")
    else:
        gt, mouse_context = load_mouse_gt(Path(cfg["gt"]), Path(cfg["roi_meta"]))
        print("GT: official adjusted.cellbin.gef, center 3000x3000 ROI")
    try:
        gt_xy = gt.cells[["x_um", "y_um"]].to_numpy(float)
        pred_xy = pred[["x_um", "y_um"]].to_numpy(float)
        pi, gi, shifts = greedy_match(pred_xy, gt_xy, args.match_radius)
        n_match = len(pi)
        precision = n_match / len(pred) if len(pred) else 0.0
        recall = n_match / len(gt.cells) if len(gt.cells) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"cells: pred={len(pred):,}, GT={len(gt.cells):,}, matched={n_match:,}; "
            f"precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")

        rng = np.random.default_rng(42)
        sample_position = (
            rng.choice(n_match, min(args.vector_sample, n_match), replace=False)
            if n_match else np.zeros(0, dtype=int))
        sampled_pi, sampled_gi = pi[sample_position], gi[sample_position]
        if gem is not None:
            fill_gem_vectors(gem, sampled_gi, gt.vectors)
        else:
            fill_mouse_vectors(mouse_context, sampled_gi, gt.vectors)

        pred_vectors, pred_genes, assign_overlap, assign_accuracy, assigned_rows = (
            load_prediction_vectors_and_assignment(
                pred_dir / "molecules.parquet",
                pred.cell.astype(str).tolist(),
                pred.iloc[sampled_pi].cell.astype(str).tolist(),
                pred.iloc[pi].cell.astype(str).tolist(),
                gi,
                gt.labeler,
                args.batch_size,
            ))
        print(f"assigned Baysor molecules used: {assigned_rows:,}")

        p_count = pred.iloc[pi].n_transcripts.to_numpy(float)
        g_count = gt.cells.iloc[gi].gt_count.to_numpy(float)
        cpear, cspear, cmae, crmse = count_metrics(p_count, g_count)
        vcos, vjs, vpear = vector_metrics(
            pred_vectors, gt.vectors, len(sample_position), pred_genes)

        row: Dict[str, object] = {
            "method": "Baysor", "dataset": name,
            "gt_cells": len(gt.cells), "pred_cells": len(pred),
            "cell_ratio": rounded(len(pred) / len(gt.cells)) if len(gt.cells) else np.nan,
            "matched": n_match, "precision": rounded(precision), "recall": rounded(recall),
            "f1": rounded(f1),
            "loc_mean_um": rounded(np.mean(shifts)) if n_match else np.nan,
            "loc_median_um": rounded(np.median(shifts)) if n_match else np.nan,
            "loc_p95_um": rounded(np.percentile(shifts, 95)) if n_match else np.nan,
            "count_pearson": rounded(cpear), "count_spearman": rounded(cspear),
            "count_mae": rounded(cmae, 2), "count_rmse": rounded(crmse, 2),
            "vec_cosine": rounded(vcos), "vec_js_dist": rounded(vjs),
            "vec_pearson": rounded(vpear), "assign_overlap": rounded(assign_overlap),
            "assign_accuracy": rounded(assign_accuracy),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"baysor_stereo_{name}.csv"
        pd.DataFrame([row], columns=METRICS).to_csv(path, index=False)
        print(f"wrote: {path}")
        return row
    finally:
        if gt.cleanup is not None:
            gt.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dataset", choices=list(DATASETS))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--match-radius", type=float, default=MATCH_RADIUS_UM)
    parser.add_argument("--vector-sample", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    if args.vector_sample <= 0 or args.batch_size <= 0 or args.match_radius <= 0:
        raise ValueError("sample size, batch size and match radius must be positive")
    names = [args.dataset] if args.dataset else list(DATASETS)
    if args.check_only:
        for name in names:
            validate_one(name, DATASETS[name])
        return
    rows = [evaluate(name, DATASETS[name], args) for name in names]
    if len(rows) > 1:
        combined = args.out_dir / "baysor_stereo_summary.csv"
        pd.DataFrame(rows, columns=METRICS).to_csv(combined, index=False)
        print(f"\ncombined summary: {combined}")


if __name__ == "__main__":
    main()
