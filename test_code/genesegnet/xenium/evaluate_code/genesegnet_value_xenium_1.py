from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12
SCRIPT_VERSION = "2026-06-08-genesegnet-label-image-eval-input-layer-v1"

EVAL_CELL_ID = "__eval_cell_id"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvaluationConfig:
    # GT
    gt_root: str = "~/Xenium_1"
    gt_path: Optional[str] = "~/Xenium_1/cells.parquet"
    gt_transcripts_path: Optional[str] = "~/Xenium_1/transcripts.parquet"

    # GeneSegNet output
    genesegnet_output_dir: str = "~/Xenium_1/genesegnet_output"
    label_path: Optional[str] = "~/Xenium_1/genesegnet_output/label_sample.tif"
    mat_key: str = "CellMap"
    pixel_size_um: float = 0.2125
    swap_xy: bool = False
    label_chunk_rows: int = 512
    min_label: int = 1
    max_label: Optional[int] = None

    output_dir: str = "~/value_code/genesegnet_evaluation"
    method_name: str = "GeneSegNet"

    # Canonical / source column names
    cell_id_col: str = "cell_id"
    gt_cell_id_col: str = "cell_id"
    x_col: str = "x"
    y_col: str = "y"
    type_col: str = "cell_type"

    numeric_cols: Optional[List[str]] = None
    numeric_pairs: Optional[List[Tuple[str, str]]] = None

    # Matching defaults kept as in previous evaluation
    match_radius: Optional[float] = None
    radius_quantile: float = 0.95
    radius_multiplier: float = 1.25
    max_component_size: int = 3000
    coordinate_scale_x: float = 1.0
    coordinate_scale_y: float = 1.0

    k_neighbors: int = 8
    low_recall_warning_threshold: float = 0.70

    warn_multitype_fraction: float = 0.01
    strict_unique_type_per_cell: bool = False

    save_matched_table: bool = True
    save_cell_level_tables: bool = True
    verbose: bool = True


# =============================================================================
# IO utilities
# =============================================================================

def expand_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(os.path.expanduser(path)).resolve()


def resolve_genesegnet_label_path(cfg: EvaluationConfig) -> Path:
    if cfg.label_path:
        p = expand_path(cfg.label_path)
        if p and p.exists():
            return p

    outdir = expand_path(cfg.genesegnet_output_dir)
    candidates = [
        outdir / "label_sample.tif",
        outdir / "label_sample.tiff",
        outdir / "label_sample.mat",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find GeneSegNet label file. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def read_table(path: str | Path) -> pd.DataFrame:
    path = expand_path(str(path))
    if path is None or not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table suffix: {suffix} for {path}")


def open_label_array(path: Path, mat_key: str = "CellMap") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile
        try:
            return tifffile.memmap(path)
        except Exception:
            return tifffile.imread(path)
    if suffix == ".mat":
        # Your inspection showed this is a normal MATLAB file with variable CellMap.
        from scipy.io import loadmat
        mat = loadmat(path)
        if mat_key not in mat:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"{path}: key {mat_key!r} not found. Available variables: {keys}")
        return np.asarray(mat[mat_key])
    if suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    raise ValueError(f"Unsupported label file suffix: {suffix}")


def labels_to_cell_table(
    label_path: Path,
    pixel_size_um: float,
    chunk_rows: int = 512,
    min_label: int = 1,
    max_label: Optional[int] = None,
    swap_xy: bool = False,
    mat_key: str = "CellMap",
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert a 2D GeneSegNet label image into cell-level centroid table.

    Output columns:
        cell_id, x, y, area_pixels, area_um2

    This is the only GeneSegNet-specific input conversion step.
    """
    arr = open_label_array(label_path, mat_key=mat_key)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape={arr.shape}, ndim={arr.ndim}")

    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])

    if max_label is None:
        if verbose:
            print("[INFO] Scanning label image for max label...")
        # np.nanmax works on memmap and avoids assuming dtype.
        max_label = int(np.nanmax(arr))
    if max_label < min_label:
        raise ValueError(f"max_label={max_label} < min_label={min_label}")

    n_bins = max_label + 1
    counts = np.zeros(n_bins, dtype=np.float64)
    sum_r = np.zeros(n_bins, dtype=np.float64)
    sum_c = np.zeros(n_bins, dtype=np.float64)

    if verbose:
        print(f"[INFO] Label image: {label_path}")
        print(f"[INFO] shape={arr.shape}, dtype={arr.dtype}, max_label={max_label:,}")
        print(f"[INFO] Computing centroids in chunks of {chunk_rows} rows...")

    # Precompute a column index vector once.
    col_idx_full = np.arange(n_cols, dtype=np.float64)

    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        block = np.asarray(arr[r0:r1, :])

        # Float label image is integer-like according to inspection; cast safely.
        if np.issubdtype(block.dtype, np.floating):
            finite = np.isfinite(block)
            labels = np.where(finite, block, 0).astype(np.int64, copy=False)
        else:
            labels = block.astype(np.int64, copy=False)

        valid = (labels >= min_label) & (labels <= max_label)
        if not np.any(valid):
            continue

        lab_flat = labels[valid]

        # Row coordinate for each valid pixel
        row_local = np.arange(r0, r1, dtype=np.float64)[:, None]
        rows = np.broadcast_to(row_local, labels.shape)[valid]

        # Column coordinate for each valid pixel
        cols = np.broadcast_to(col_idx_full[None, :], labels.shape)[valid]

        counts += np.bincount(lab_flat, minlength=n_bins)
        sum_r += np.bincount(lab_flat, weights=rows, minlength=n_bins)
        sum_c += np.bincount(lab_flat, weights=cols, minlength=n_bins)

        if verbose and (r0 == 0 or r1 == n_rows or (r0 // chunk_rows) % 20 == 0):
            print(f"[INFO] processed rows {r0:,}-{r1:,} / {n_rows:,}")

    valid_labels = np.flatnonzero(counts >= 1)
    valid_labels = valid_labels[valid_labels >= min_label]
    if valid_labels.size == 0:
        raise ValueError("No labeled cells found after excluding background.")

    centroid_r = sum_r[valid_labels] / counts[valid_labels]
    centroid_c = sum_c[valid_labels] / counts[valid_labels]

    if swap_xy:
        x_um = centroid_r * pixel_size_um
        y_um = centroid_c * pixel_size_um
    else:
        x_um = centroid_c * pixel_size_um
        y_um = centroid_r * pixel_size_um

    out = pd.DataFrame({
        "cell_id": valid_labels.astype(str),
        "x": x_um.astype(np.float64),
        "y": y_um.astype(np.float64),
        "area_pixels": counts[valid_labels].astype(np.float64),
        "area_um2": counts[valid_labels].astype(np.float64) * (pixel_size_um ** 2),
    })

    if verbose:
        print(f"[INFO] Predicted cells from labels: {len(out):,}")
        print(f"[INFO] x range: {out['x'].min():.3f} .. {out['x'].max():.3f} µm")
        print(f"[INFO] y range: {out['y'].min():.3f} .. {out['y'].max():.3f} µm")

    return out


# =============================================================================
# Column normalization and cell-level collapse
# =============================================================================

def normalize_columns(df: pd.DataFrame, cfg: EvaluationConfig, table_name: str) -> pd.DataFrame:
    if table_name.lower().startswith("pred"):
        id_aliases = [
            cfg.cell_id_col, "label", "object_id", "segmentation_id", "CellID", "cell_ID", "cellid", "cell"
        ]
    else:
        id_aliases = [
            cfg.gt_cell_id_col, cfg.cell_id_col,
            "cell", "cellid", "cell_ID", "CellID", "label", "object_id", "segmentation_id",
        ]

    alias_map = {
        cfg.cell_id_col: id_aliases,
        cfg.x_col: [cfg.x_col, "centroid_x", "x_centroid", "center_x", "x_location", "X", "global_x", "pos_x"],
        cfg.y_col: [cfg.y_col, "centroid_y", "y_centroid", "center_y", "y_location", "Y", "global_y", "pos_y"],
        cfg.type_col: [cfg.type_col, "type", "annotation", "cluster", "celltype", "cell_type_gt", "cell_type_pred"],
    }

    rename: Dict[str, str] = {}
    existing = set(df.columns)
    for canonical, aliases in alias_map.items():
        if canonical in existing:
            continue
        for a in aliases:
            if a in existing:
                rename[a] = canonical
                break

    out = df.rename(columns=rename).copy()

    required = [cfg.cell_id_col, cfg.x_col, cfg.y_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"{table_name} missing required columns {missing}. "
            f"Available columns: {list(out.columns)[:80]}"
        )
    return out


def _mode_with_confidence(values: pd.Series) -> Tuple[Any, int, int, float, str]:
    s = values.dropna()
    if s.empty:
        return np.nan, 0, 0, np.nan, ""
    counts = s.value_counts(dropna=True, sort=True)
    max_count = int(counts.iloc[0])
    modes = counts[counts == max_count].index.tolist()
    chosen = modes[0]
    n_valid = int(counts.sum())
    n_unique = int(counts.shape[0])
    confidence = float(max_count / max(n_valid, 1))
    all_modes_repr = ";".join(map(str, modes[:10]))
    return chosen, n_unique, n_valid, confidence, all_modes_repr


def numeric_qc_for_series(s: pd.Series, prefix: str) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    return {
        f"{prefix}_valid_n": int(valid.shape[0]),
        f"{prefix}_nan_fraction": float(x.isna().mean()),
        f"{prefix}_is_constant": bool(valid.nunique(dropna=True) <= 1) if valid.shape[0] > 0 else True,
    }


def check_coordinate_scale_warning(cells: pd.DataFrame, cfg: EvaluationConfig, table_name: str) -> None:
    xy = cells[[cfg.x_col, cfg.y_col]].to_numpy(dtype=float)
    if xy.shape[0] < 2:
        return
    span_x = float(np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0]))
    span_y = float(np.nanmax(xy[:, 1]) - np.nanmin(xy[:, 1]))
    if cfg.match_radius is None and max(span_x, span_y) > 1e5 and cfg.coordinate_scale_x == 1.0 and cfg.coordinate_scale_y == 1.0:
        warnings.warn(
            f"{table_name}: coordinate span is very large ({span_x:.1f}, {span_y:.1f}) and match_radius is not set."
        )


def collapse_to_cell_level(df: pd.DataFrame, cfg: EvaluationConfig, table_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = normalize_columns(df, cfg, table_name=table_name)

    for col in [cfg.x_col, cfg.y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    coord_bad = df[[cfg.x_col, cfg.y_col]].isna().any(axis=1)
    if coord_bad.any():
        warnings.warn(f"{table_name}: dropping {int(coord_bad.sum())} rows with missing coords.")
        df = df.loc[~coord_bad].copy()
    if df.empty:
        raise ValueError(f"{table_name}: no valid rows after coordinate cleaning.")

    df[cfg.x_col] = df[cfg.x_col] * cfg.coordinate_scale_x
    df[cfg.y_col] = df[cfg.y_col] * cfg.coordinate_scale_y

    grouped = df.groupby(cfg.cell_id_col, sort=False, observed=True)
    base = grouped[[cfg.x_col, cfg.y_col]].mean().reset_index()

    qc: Dict[str, Any] = {
        f"{table_name}_raw_rows": int(len(df)),
        f"{table_name}_n_cells": int(base.shape[0]),
        f"{table_name}_mean_rows_per_cell": float(len(df) / max(base.shape[0], 1)),
    }

    if cfg.type_col in df.columns:
        mode_records = []
        for cid, sub in grouped[cfg.type_col]:
            chosen, n_unique, n_valid, conf, modes_repr = _mode_with_confidence(sub)
            mode_records.append((cid, chosen, n_unique, n_valid, conf, modes_repr))
        type_df = pd.DataFrame(
            mode_records,
            columns=[
                cfg.cell_id_col,
                cfg.type_col,
                f"{cfg.type_col}_n_unique",
                f"{cfg.type_col}_n_valid",
                f"{cfg.type_col}_confidence",
                f"{cfg.type_col}_modes",
            ],
        )
        base = base.merge(type_df, on=cfg.cell_id_col, how="left")
        multitype_fraction = float((type_df[f"{cfg.type_col}_n_unique"] > 1).mean())
        low_conf_fraction = float((type_df[f"{cfg.type_col}_confidence"] < 1.0).mean())
        qc.update({
            f"{table_name}_multitype_cell_fraction": multitype_fraction,
            f"{table_name}_type_low_confidence_fraction": low_conf_fraction,
            f"{table_name}_type_median_confidence": float(type_df[f"{cfg.type_col}_confidence"].median(skipna=True)),
        })
        if multitype_fraction > cfg.warn_multitype_fraction:
            msg = f"{table_name}: {multitype_fraction:.3%} cells have multiple {cfg.type_col} values."
            if cfg.strict_unique_type_per_cell:
                raise ValueError(msg)
            warnings.warn(msg)

    numeric_to_aggregate: List[str] = []
    if cfg.numeric_cols:
        numeric_to_aggregate.extend(cfg.numeric_cols)
    if cfg.numeric_pairs:
        if table_name.lower().startswith("pred"):
            numeric_to_aggregate.extend([p for p, _ in cfg.numeric_pairs])
        else:
            numeric_to_aggregate.extend([g for _, g in cfg.numeric_pairs])
    seen: set[str] = set()
    numeric_to_aggregate = [c for c in numeric_to_aggregate if not (c in seen or seen.add(c))]

    for col in numeric_to_aggregate:
        if col not in df.columns:
            warnings.warn(f"{table_name}: numeric column '{col}' missing; metrics depending on it will be NaN.")
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({cfg.cell_id_col: df[cfg.cell_id_col].values, col: numeric.values})
        agg = tmp.groupby(cfg.cell_id_col, sort=False, observed=True)[col].mean().reset_index()
        base = base.merge(agg, on=cfg.cell_id_col, how="left")
        qc.update(numeric_qc_for_series(agg[col], prefix=f"{table_name}_{col}"))

    # Preserve GeneSegNet-specific columns after grouping when they already exist 1 row / cell.
    for extra in ["area_pixels", "area_um2"]:
        if extra in df.columns and extra not in base.columns:
            agg = df.groupby(cfg.cell_id_col, sort=False, observed=True)[extra].mean().reset_index()
            base = base.merge(agg, on=cfg.cell_id_col, how="left")
            qc.update(numeric_qc_for_series(agg[extra], prefix=f"{table_name}_{extra}"))

    check_coordinate_scale_warning(base, cfg, table_name)
    return base, qc


# =============================================================================
# Matching
# =============================================================================

def infer_match_radius(gt_cells: pd.DataFrame, cfg: EvaluationConfig) -> float:
    xy = gt_cells[[cfg.x_col, cfg.y_col]].to_numpy(dtype=float)
    if xy.shape[0] < 2:
        raise ValueError("Cannot infer match radius from fewer than two GT cells.")
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(xy)
    dists, _ = nbrs.kneighbors(xy)
    nn = dists[:, 1]
    radius = float(np.nanquantile(nn, cfg.radius_quantile) * cfg.radius_multiplier)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid inferred radius: {radius}")
    return radius


def build_candidate_edges(pred_cells, gt_cells, cfg, radius):
    pred_xy = pred_cells[[cfg.x_col, cfg.y_col]].to_numpy(dtype=float)
    gt_xy = gt_cells[[cfg.x_col, cfg.y_col]].to_numpy(dtype=float)
    tree = cKDTree(gt_xy)
    neighbor_lists = tree.query_ball_point(pred_xy, r=radius)

    pred_idx: List[int] = []
    gt_idx: List[int] = []
    dist_list: List[float] = []
    for i, js in enumerate(neighbor_lists):
        if not js:
            continue
        js_arr = np.asarray(js, dtype=int)
        d = np.linalg.norm(gt_xy[js_arr] - pred_xy[i], axis=1)
        for j, dij in zip(js_arr.tolist(), d.tolist()):
            pred_idx.append(i)
            gt_idx.append(j)
            dist_list.append(float(dij))
    return pd.DataFrame({"pred_index": pred_idx, "gt_index": gt_idx, "distance": dist_list})


def _connected_components_bipartite(edges, n_pred, n_gt):
    if edges.empty:
        return []
    pred_to_edges: Dict[int, List[int]] = {}
    gt_to_edges: Dict[int, List[int]] = {}
    pred_arr = edges["pred_index"].to_numpy(dtype=int)
    gt_arr = edges["gt_index"].to_numpy(dtype=int)
    for eidx, (p, g) in enumerate(zip(pred_arr, gt_arr)):
        pred_to_edges.setdefault(int(p), []).append(eidx)
        gt_to_edges.setdefault(int(g), []).append(eidx)

    visited_pred: set[int] = set()
    visited_gt: set[int] = set()
    visited_edges: set[int] = set()
    components = []
    for start_p in pred_to_edges.keys():
        if start_p in visited_pred:
            continue
        stack_p = [start_p]
        comp_pred: set[int] = set()
        comp_gt: set[int] = set()
        comp_edges: set[int] = set()
        while stack_p:
            p = stack_p.pop()
            if p in visited_pred:
                continue
            visited_pred.add(p)
            comp_pred.add(p)
            for eidx in pred_to_edges.get(p, []):
                if eidx in visited_edges:
                    continue
                visited_edges.add(eidx)
                comp_edges.add(eidx)
                g = int(gt_arr[eidx])
                comp_gt.add(g)
                if g not in visited_gt:
                    visited_gt.add(g)
                    for e2 in gt_to_edges.get(g, []):
                        p2 = int(pred_arr[e2])
                        if p2 not in visited_pred:
                            stack_p.append(p2)
        comp_edge_df = edges.iloc[sorted(comp_edges)].copy()
        components.append((
            np.array(sorted(comp_pred), dtype=int),
            np.array(sorted(comp_gt), dtype=int),
            comp_edge_df,
        ))
    return components


def empty_matches() -> pd.DataFrame:
    return pd.DataFrame(columns=["pred_index", "gt_index", "distance"])


def attach_match_metadata(matches, pred_cells, gt_cells, cfg):
    if matches.empty:
        return matches
    pred_meta = pred_cells.reset_index(drop=True).reset_index().rename(columns={
        "index": "pred_index", cfg.cell_id_col: "pred_cell_id",
        cfg.x_col: "pred_x", cfg.y_col: "pred_y",
    })
    gt_meta = gt_cells.reset_index(drop=True).reset_index().rename(columns={
        "index": "gt_index", cfg.cell_id_col: "gt_cell_id",
        cfg.x_col: "gt_x", cfg.y_col: "gt_y",
    })
    pred_keep = {"pred_index", "pred_cell_id", "pred_x", "pred_y"}
    gt_keep = {"gt_index", "gt_cell_id", "gt_x", "gt_y"}
    pred_rename = {c: f"pred_{c}" for c in pred_meta.columns if c not in pred_keep and not c.startswith("pred_")}
    gt_rename = {c: f"gt_{c}" for c in gt_meta.columns if c not in gt_keep and not c.startswith("gt_")}
    pred_meta = pred_meta.rename(columns=pred_rename)
    gt_meta = gt_meta.rename(columns=gt_rename)
    out = matches.copy()
    out = out.merge(pred_meta, on="pred_index", how="left")
    out = out.merge(gt_meta, on="gt_index", how="left")
    return out


def match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=None):
    if radius is None:
        radius = infer_match_radius(gt_cells, cfg)
    edges = build_candidate_edges(pred_cells, gt_cells, cfg, radius)
    qc: Dict[str, Any] = {
        "matching_strategy": "sparse_hungarian_connected_components",
        "match_radius": float(radius),
        "candidate_edges": int(edges.shape[0]),
        "mean_candidate_edges_per_pred": float(edges.shape[0] / max(pred_cells.shape[0], 1)),
    }
    if edges.empty:
        return empty_matches(), qc

    components = _connected_components_bipartite(edges, len(pred_cells), len(gt_cells))
    qc.update({
        "candidate_components": int(len(components)),
        "max_component_pred_size": int(max((len(c[0]) for c in components), default=0)),
        "max_component_gt_size": int(max((len(c[1]) for c in components), default=0)),
    })

    matched_records: List[Tuple[int, int, float]] = []
    large_components = 0

    for pred_idx, gt_idx, comp_edges in components:
        if len(pred_idx) == 0 or len(gt_idx) == 0:
            continue
        if max(len(pred_idx), len(gt_idx)) > cfg.max_component_size:
            large_components += 1
            comp_sorted = comp_edges.sort_values(["distance", "pred_index", "gt_index"], kind="mergesort")
            used_p: set[int] = set()
            used_g: set[int] = set()
            for row in comp_sorted.itertuples(index=False):
                p = int(row.pred_index)
                g = int(row.gt_index)
                if p in used_p or g in used_g:
                    continue
                used_p.add(p)
                used_g.add(g)
                matched_records.append((p, g, float(row.distance)))
            continue

        p_pos = {p: i for i, p in enumerate(pred_idx.tolist())}
        g_pos = {g: i for i, g in enumerate(gt_idx.tolist())}
        finite_big = float(max(radius * 1e4, 1e9))
        cost = np.full((len(pred_idx), len(gt_idx)), finite_big, dtype=np.float64)
        for row in comp_edges.itertuples(index=False):
            cost[p_pos[int(row.pred_index)], g_pos[int(row.gt_index)]] = float(row.distance)

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            d = float(cost[r, c])
            if d <= radius + EPS:
                matched_records.append((int(pred_idx[r]), int(gt_idx[c]), d))

    qc["large_components_greedy_fallback"] = int(large_components)
    matches = pd.DataFrame(matched_records, columns=["pred_index", "gt_index", "distance"])
    if matches.empty:
        return empty_matches(), qc
    matches = attach_match_metadata(matches, pred_cells, gt_cells, cfg)
    return matches, qc


# =============================================================================
# Metrics
# =============================================================================

def detection_metrics(matches, n_pred, n_gt):
    m = int(matches.shape[0])
    precision = m / max(n_pred, 1)
    recall = m / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    return {
        "pred_cell_count": int(n_pred),
        "gt_cell_count": int(n_gt),
        "matched_cell_count": m,
        "unmatched_pred_count": int(n_pred - m),
        "unmatched_gt_count": int(n_gt - m),
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1),
        "matched_fraction_gt": float(recall),
        "matched_fraction_pred": float(precision),
        "cell_count_difference": int(n_pred - n_gt),
        "cell_count_ratio": float(n_pred / max(n_gt, EPS)),
    }


def spatial_shift_metrics(matches):
    if matches is None or matches.empty or "distance" not in matches.columns:
        return {
            "mean_centroid_shift": np.nan,
            "median_centroid_shift": np.nan,
            "p95_centroid_shift": np.nan,
            "max_centroid_shift": np.nan,
        }
    d = pd.to_numeric(matches["distance"], errors="coerce").dropna().to_numpy(dtype=float)
    if d.size == 0:
        return {
            "mean_centroid_shift": np.nan,
            "median_centroid_shift": np.nan,
            "p95_centroid_shift": np.nan,
            "max_centroid_shift": np.nan,
        }
    return {
        "mean_centroid_shift": float(np.mean(d)),
        "median_centroid_shift": float(np.median(d)),
        "p95_centroid_shift": float(np.quantile(d, 0.95)),
        "max_centroid_shift": float(np.max(d)),
    }


def type_metrics(matches, cfg):
    pred_col = f"pred_{cfg.type_col}"
    gt_col = f"gt_{cfg.type_col}"
    base_nan = {
        "cell_type_accuracy": np.nan,
        "cell_type_macro_f1": np.nan,
        "cell_type_weighted_f1": np.nan,
        "cell_type_balanced_accuracy": np.nan,
        "ARI_matched_type_labels": np.nan,
        "NMI_matched_type_labels": np.nan,
        "cell_type_purity_legacy_matched_accuracy": np.nan,
        "matched_type_valid_n": 0,
    }
    if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns:
        return base_nan
    tmp = matches[[pred_col, gt_col]].dropna().copy()
    if tmp.empty:
        return base_nan
    y_pred = tmp[pred_col].astype(str).to_numpy()
    y_true = tmp[gt_col].astype(str).to_numpy()
    acc = float(np.mean(y_pred == y_true))
    return {
        "cell_type_accuracy": acc,
        "cell_type_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cell_type_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cell_type_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ARI_matched_type_labels": float(adjusted_rand_score(y_true, y_pred)),
        "NMI_matched_type_labels": float(normalized_mutual_info_score(y_true, y_pred)),
        "cell_type_purity_legacy_matched_accuracy": acc,
        "matched_type_valid_n": int(tmp.shape[0]),
    }


def morans_i_knn(coords, values, k=8):
    coords = np.asarray(coords, dtype=float)
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    coords = coords[valid]
    values = values[valid]
    n = len(values)
    if n <= k + 1 or np.nanstd(values) <= EPS:
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    mean_value = np.nanmean(values)
    denom = np.nansum((values - mean_value) ** 2)
    if denom <= EPS:
        return np.nan
    centered = values - mean_value
    numerator = 0.0
    weight_sum = 0
    for i in range(n):
        neigh = indices[i, 1:]
        numerator += float(np.sum(centered[i] * centered[neigh]))
        weight_sum += len(neigh)
    return float((n / max(weight_sum, EPS)) * (numerator / max(denom, EPS)))


def numeric_spatial_metrics(pred_cells, gt_cells, matches, cfg):
    res: Dict[str, Any] = {}
    if not cfg.numeric_cols:
        return res
    for col in cfg.numeric_cols:
        for label, cells in [("pred", pred_cells), ("gt", gt_cells)]:
            if col not in cells.columns:
                res[f"{label}_{col}_morans_i"] = np.nan
                res.update(numeric_qc_for_series(pd.Series(dtype=float), prefix=f"{label}_{col}"))
                continue
            vals = pd.to_numeric(cells[col], errors="coerce")
            res.update(numeric_qc_for_series(vals, prefix=f"{label}_{col}"))
            coords = cells[[cfg.x_col, cfg.y_col]].to_numpy(dtype=float)
            res[f"{label}_{col}_morans_i"] = morans_i_knn(coords, vals.to_numpy(), k=cfg.k_neighbors)
        pred_mi = res.get(f"pred_{col}_morans_i", np.nan)
        gt_mi = res.get(f"gt_{col}_morans_i", np.nan)
        res[f"delta_morans_i_{col}"] = (
            float(pred_mi - gt_mi) if np.isfinite(pred_mi) and np.isfinite(gt_mi) else np.nan
        )

        pred_col = f"pred_{col}"
        gt_col = f"gt_{col}"
        if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns:
            res.update({
                f"matched_{col}_valid_n": 0,
                f"matched_{col}_pearson": np.nan,
                f"matched_{col}_spearman": np.nan,
                f"matched_{col}_mae": np.nan,
                f"matched_{col}_rmse": np.nan,
            })
            continue
        x = pd.to_numeric(matches[pred_col], errors="coerce")
        y = pd.to_numeric(matches[gt_col], errors="coerce")
        valid = x.notna() & y.notna()
        xv = x[valid].to_numpy(dtype=float)
        yv = y[valid].to_numpy(dtype=float)
        if len(xv) < 3 or np.std(xv) <= EPS or np.std(yv) <= EPS:
            pear = np.nan
            spear = np.nan
        else:
            pear = float(pearsonr(xv, yv)[0])
            spear = float(spearmanr(xv, yv)[0])
        if len(xv) == 0:
            mae = np.nan
            rmse = np.nan
        else:
            diff = xv - yv
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
        res.update({
            f"matched_{col}_valid_n": int(len(xv)),
            f"matched_{col}_pearson": pear,
            f"matched_{col}_spearman": spear,
            f"matched_{col}_mae": mae,
            f"matched_{col}_rmse": rmse,
        })
    return res


def numeric_pair_metrics(matches, cfg):
    res: Dict[str, Any] = {}
    if not cfg.numeric_pairs:
        return res
    for pred_col_raw, gt_col_raw in cfg.numeric_pairs:
        pred_col = f"pred_{pred_col_raw}"
        gt_col = f"gt_{gt_col_raw}"
        label = f"{pred_col_raw}_vs_{gt_col_raw}"
        if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns:
            warnings.warn(
                f"numeric pair '{pred_col_raw}:{gt_col_raw}' cannot be evaluated; missing columns after matching."
            )
            res.update({
                f"matched_pair_{label}_valid_n": 0,
                f"matched_pair_{label}_pearson": np.nan,
                f"matched_pair_{label}_spearman": np.nan,
                f"matched_pair_{label}_mae": np.nan,
                f"matched_pair_{label}_rmse": np.nan,
            })
            continue

        x = pd.to_numeric(matches[pred_col], errors="coerce")
        y = pd.to_numeric(matches[gt_col], errors="coerce")
        valid = x.notna() & y.notna()
        xv = x[valid].to_numpy(dtype=float)
        yv = y[valid].to_numpy(dtype=float)
        if len(xv) < 3 or np.std(xv) <= EPS or np.std(yv) <= EPS:
            pear = np.nan
            spear = np.nan
        else:
            pear = float(pearsonr(xv, yv)[0])
            spear = float(spearmanr(xv, yv)[0])
        if len(xv) == 0:
            mae = np.nan
            rmse = np.nan
        else:
            diff = xv - yv
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
        res.update({
            f"matched_pair_{label}_valid_n": int(len(xv)),
            f"matched_pair_{label}_pearson": pear,
            f"matched_pair_{label}_spearman": spear,
            f"matched_pair_{label}_mae": mae,
            f"matched_pair_{label}_rmse": rmse,
        })
    return res


def low_recall_knn_warning(metrics, cfg):
    recall = metrics.get("detection_recall", np.nan)
    low = bool(np.isfinite(recall) and recall < cfg.low_recall_warning_threshold)
    if low:
        warnings.warn(f"Detection recall is {recall:.3f}, below {cfg.low_recall_warning_threshold:.3f}.")
    return {
        "knn_metric_matched_only": True,
        "knn_metric_low_recall_warning": low,
        "knn_metric_recall_threshold": float(cfg.low_recall_warning_threshold),
    }


def knn_neighbor_type_consistency_on_gt_graph(matches, cfg):
    pred_type_col = f"pred_{cfg.type_col}"
    gt_type_col = f"gt_{cfg.type_col}"
    required = {"gt_x", "gt_y", pred_type_col, gt_type_col}
    if matches.empty or not required.issubset(set(matches.columns)):
        return {
            "knn_neighbor_type_pearson_gt_graph": np.nan,
            "knn_neighbor_type_mean_js_distance_gt_graph": np.nan,
            "knn_neighbor_type_mean_js_divergence_gt_graph": np.nan,
            "knn_neighbor_metric_valid_n": 0,
        }
    # GeneSegNet label image has no predicted cell type in the current output.
    return {
        "knn_neighbor_type_pearson_gt_graph": np.nan,
        "knn_neighbor_type_mean_js_distance_gt_graph": np.nan,
        "knn_neighbor_type_mean_js_divergence_gt_graph": np.nan,
        "knn_neighbor_metric_valid_n": 0,
    }


def genesegnet_transcript_metrics(cfg: EvaluationConfig, matches: pd.DataFrame) -> Dict[str, Any]:
    """Return transcript-level keys without changing metric panel.

    Current GeneSegNet files are only label/confidence/offset arrays; they do not
    include per-transcript assignments, transcript_id, or cell×gene count matrix.
    Therefore transcript / gene-vector metrics are explicitly unavailable and are
    emitted as NaN or 0 rather than being silently invented.
    """
    gt_transcripts_path = expand_path(cfg.gt_transcripts_path) if cfg.gt_transcripts_path else None
    gt_rows = np.nan
    gt_assigned_n = np.nan
    gt_assignment_rate = np.nan
    if gt_transcripts_path and gt_transcripts_path.exists():
        try:
            # Only metadata row count is needed here; pyarrow can read efficiently.
            gt = pd.read_parquet(gt_transcripts_path, columns=None)
            gt_rows = int(len(gt))
            if "cell_id" in gt.columns:
                s = gt["cell_id"].astype("string").str.strip()
                bad = s.str.lower().isin({"", "nan", "none", "null", "na", "-1", "0", "unassigned", "background"})
                assigned = s.mask(bad).notna()
                gt_assigned_n = int(assigned.sum())
                gt_assignment_rate = float(assigned.mean()) if len(gt) else np.nan
        except Exception as exc:
            warnings.warn(f"Could not read GT transcripts for row count: {exc}")

    return {
        "transcript_metrics_available": False,
        "transcript_metrics_error": (
            "GeneSegNet output contains label/confidence/offset images only; "
            "no per-transcript assignment table, transcript_id column, or cell×gene matrix was found."
        ),
        "pred_transcript_rows": np.nan,
        "gt_transcript_rows": gt_rows,
        "pred_transcript_assigned_n": 0,
        "pred_transcript_assignment_rate": np.nan,
        "pred_transcript_has_gene_col": False,
        "gt_transcript_assigned_n": gt_assigned_n,
        "gt_transcript_assignment_rate": gt_assignment_rate,
        "gt_transcript_has_gene_col": np.nan,
        "transcript_matched_cell_pairs_n": int(matches.shape[0]) if matches is not None else 0,
        "pred_transcripts_in_matched_cells_n": 0,
        "gt_transcripts_in_matched_cells_n": np.nan,
        "matched_cell_transcript_count_pearson": np.nan,
        "matched_cell_transcript_count_spearman": np.nan,
        "matched_cell_transcript_count_mae": np.nan,
        "matched_cell_transcript_count_rmse": np.nan,
        "matched_cell_gene_vector_mean_cosine": np.nan,
        "matched_cell_gene_vector_mean_js_distance": np.nan,
        "matched_cell_gene_vector_mean_pearson": np.nan,
        "matched_cell_gene_vector_valid_pairs": 0,
        "transcript_id_overlap_n": 0,
        "transcript_assignment_accuracy_via_matched_cells": np.nan,
    }


# =============================================================================
# Requested metric panel
# =============================================================================

REQUESTED_METRICS: List[Tuple[str, str]] = [
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
    ("matched_pair_n_transcripts_vs_total_counts_pearson", "matched_pair_n_transcripts_vs_total_counts_pearson"),
    ("matched_pair_n_transcripts_vs_total_counts_spearman", "matched_pair_n_transcripts_vs_total_counts_spearman"),
    ("matched_pair_n_transcripts_vs_total_counts_mae", "matched_pair_n_transcripts_vs_total_counts_mae"),
    ("matched_pair_n_transcripts_vs_total_counts_rmse", "matched_pair_n_transcripts_vs_total_counts_rmse"),
    ("Pred transcript rows", "pred_transcript_rows"),
    ("GT transcript rows", "gt_transcript_rows"),
    ("Pred transcript assignment rate", "pred_transcript_assignment_rate"),
    ("Transcript matched cell pairs", "transcript_matched_cell_pairs_n"),
    ("matched_cell_transcript_count_pearson", "matched_cell_transcript_count_pearson"),
    ("matched_cell_transcript_count_spearman", "matched_cell_transcript_count_spearman"),
    ("matched_cell_transcript_count_mae", "matched_cell_transcript_count_mae"),
    ("matched_cell_transcript_count_rmse", "matched_cell_transcript_count_rmse"),
    ("matched_cell_gene_vector_mean_cosine", "matched_cell_gene_vector_mean_cosine"),
    ("matched_cell_gene_vector_mean_js_distance", "matched_cell_gene_vector_mean_js_distance"),
    ("matched_cell_gene_vector_mean_pearson", "matched_cell_gene_vector_mean_pearson"),
    ("matched_cell_gene_vector_valid_pairs", "matched_cell_gene_vector_valid_pairs"),
    ("transcript_id_overlap_n", "transcript_id_overlap_n"),
    ("transcript_assignment_accuracy_via_matched_cells", "transcript_assignment_accuracy_via_matched_cells"),
]


def build_selected_metric_row(method_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"method": method_name}
    for display, key in REQUESTED_METRICS:
        row[display] = metrics.get(key, np.nan)
    return row


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# =============================================================================
# Pipeline
# =============================================================================

def evaluate(cfg: EvaluationConfig) -> pd.DataFrame:
    if cfg.numeric_pairs is None:
        cfg.numeric_pairs = [("n_transcripts", "total_counts")]

    outdir = expand_path(cfg.output_dir)
    assert outdir is not None
    outdir.mkdir(parents=True, exist_ok=True)

    gt_path = expand_path(cfg.gt_path)
    if gt_path is None or not gt_path.exists():
        raise FileNotFoundError(f"GT cell table not found: {cfg.gt_path}")

    label_path = resolve_genesegnet_label_path(cfg)

    if cfg.verbose:
        print(f"[INFO] Script version: {SCRIPT_VERSION}")
        print(f"[INFO] GT cell table: {gt_path}")
        print(f"[INFO] GeneSegNet label file: {label_path}")
        print(f"[INFO] pixel_size_um={cfg.pixel_size_um}, swap_xy={cfg.swap_xy}")

    gt_raw = read_table(gt_path)

    # GeneSegNet-specific input layer: label image -> centroid cell table.
    pred_raw = labels_to_cell_table(
        label_path=label_path,
        pixel_size_um=cfg.pixel_size_um,
        chunk_rows=cfg.label_chunk_rows,
        min_label=cfg.min_label,
        max_label=cfg.max_label,
        swap_xy=cfg.swap_xy,
        mat_key=cfg.mat_key,
        verbose=cfg.verbose,
    )

    # Shared cell-level preprocessing and evaluation.
    gt_cells, gt_qc = collapse_to_cell_level(gt_raw, cfg, table_name="gt")
    pred_cells, pred_qc = collapse_to_cell_level(pred_raw, cfg, table_name="pred")

    radius = cfg.match_radius if cfg.match_radius is not None else infer_match_radius(gt_cells, cfg)
    if cfg.verbose:
        print(f"[INFO] Using match radius: {radius:.6g}")
        print(f"[INFO] GT cells: {len(gt_cells):,}; Pred cells: {len(pred_cells):,}")

    matches, match_qc = match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=radius)

    metrics: Dict[str, Any] = {
        "method": cfg.method_name,
        "script_version": SCRIPT_VERSION,
        "gt_path": str(gt_path),
        "pred_label_path": str(label_path),
        "pixel_size_um": cfg.pixel_size_um,
        "swap_xy": cfg.swap_xy,
    }

    metrics.update(gt_qc)
    metrics.update(pred_qc)
    metrics.update(match_qc)
    metrics.update(detection_metrics(matches, n_pred=len(pred_cells), n_gt=len(gt_cells)))
    metrics.update(spatial_shift_metrics(matches))
    metrics.update(type_metrics(matches, cfg))
    metrics.update(knn_neighbor_type_consistency_on_gt_graph(matches, cfg))
    metrics.update(low_recall_knn_warning(metrics, cfg))
    metrics.update(numeric_spatial_metrics(pred_cells, gt_cells, matches, cfg))
    metrics.update(numeric_pair_metrics(matches, cfg))
    metrics.update(genesegnet_transcript_metrics(cfg, matches))

    full_df = pd.DataFrame([metrics])
    prefix = f"{cfg.method_name.lower()}_xenium1_sparse_hungarian_qc_metrics"
    full_csv = outdir / f"{prefix}_full.csv"
    full_json = outdir / f"{prefix}_full.json"
    full_df.to_csv(full_csv, index=False)
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)

    selected_row = build_selected_metric_row(cfg.method_name, metrics)
    selected_df = pd.DataFrame([selected_row])
    sel_csv = outdir / f"{prefix}_selected.csv"
    sel_json = outdir / f"{prefix}_selected.json"
    selected_df.to_csv(sel_csv, index=False)
    with open(sel_json, "w", encoding="utf-8") as f:
        json.dump(selected_row, f, indent=2, ensure_ascii=False, default=_json_default)

    cfg_json = outdir / "evaluation_config.json"
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False, default=_json_default)

    if cfg.save_matched_table:
        matches.to_csv(outdir / "matched_cells_sparse_hungarian.csv", index=False)
    if cfg.save_cell_level_tables:
        pred_cells.to_csv(outdir / "pred_cell_level_table.csv", index=False)
        gt_cells.to_csv(outdir / "gt_cell_level_table.csv", index=False)

    if cfg.verbose:
        print("[DONE] Selected metrics saved to:", sel_csv)
        print("[DONE] Full metrics saved to:", full_csv)
        print("\n=== SELECTED METRICS ===")
        print(selected_df.T.to_string(header=False))
    return selected_df


def parse_args() -> EvaluationConfig:
    p = argparse.ArgumentParser(description="Evaluate GeneSegNet label image against Xenium_1 GT.")
    p.add_argument("--gt-root", default="~/Xenium_1")
    p.add_argument("--gt-path", default="~/Xenium_1/cells.parquet")
    p.add_argument("--gt-transcripts-path", default="~/Xenium_1/transcripts.parquet")

    p.add_argument("--genesegnet-output-dir", default="~/Xenium_1/genesegnet_output")
    p.add_argument("--label-path", default="~/Xenium_1/genesegnet_output/label_sample.tif")
    p.add_argument("--mat-key", default="CellMap")
    p.add_argument("--pixel-size-um", type=float, default=0.2125)
    p.add_argument("--swap-xy", action="store_true")
    p.add_argument("--label-chunk-rows", type=int, default=512)
    p.add_argument("--min-label", type=int, default=1)
    p.add_argument("--max-label", type=int, default=None)

    p.add_argument("--output-dir", default="~/Xenium_1/genesegnet_output/evaluation_results")
    p.add_argument("--method-name", default="GeneSegNet")

    p.add_argument("--match-radius", type=float, default=None)
    p.add_argument("--radius-quantile", type=float, default=0.95)
    p.add_argument("--radius-multiplier", type=float, default=1.25)
    p.add_argument("--max-component-size", type=int, default=3000)

    p.add_argument("--coordinate-scale-x", type=float, default=1.0)
    p.add_argument("--coordinate-scale-y", type=float, default=1.0)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--low-recall-warning-threshold", type=float, default=0.70)

    p.add_argument("--no-save-matched-table", action="store_true")
    p.add_argument("--no-save-cell-level-tables", action="store_true")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    return EvaluationConfig(
        gt_root=args.gt_root,
        gt_path=args.gt_path,
        gt_transcripts_path=args.gt_transcripts_path,
        genesegnet_output_dir=args.genesegnet_output_dir,
        label_path=args.label_path,
        mat_key=args.mat_key,
        pixel_size_um=args.pixel_size_um,
        swap_xy=args.swap_xy,
        label_chunk_rows=args.label_chunk_rows,
        min_label=args.min_label,
        max_label=args.max_label,
        output_dir=args.output_dir,
        method_name=args.method_name,
        numeric_pairs=[("n_transcripts", "total_counts")],
        match_radius=args.match_radius,
        radius_quantile=args.radius_quantile,
        radius_multiplier=args.radius_multiplier,
        max_component_size=args.max_component_size,
        coordinate_scale_x=args.coordinate_scale_x,
        coordinate_scale_y=args.coordinate_scale_y,
        k_neighbors=args.k_neighbors,
        low_recall_warning_threshold=args.low_recall_warning_threshold,
        save_matched_table=not args.no_save_matched_table,
        save_cell_level_tables=not args.no_save_cell_level_tables,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    cfg = parse_args()
    evaluate(cfg)
