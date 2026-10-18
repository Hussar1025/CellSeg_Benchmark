from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12
SCRIPT_VERSION = "2026-05-25-boms-transcript-boms-cell-id-fix-v5"

EVAL_CELL_ID = "__eval_cell_id"
EVAL_GENE = "__eval_gene"
EVAL_TRANSCRIPT_ID = "__eval_transcript_id"


@dataclass
class EvaluationConfig:
    # Default server paths
    # Xenium_1 and value_code are sibling directories under the same home/root folder.
    # GT is local on the server; BOMS files are pulled from GitHub into local_repo_dir.
    gt_root: str = "~/Xenium_1"
    gt_path: Optional[str] = "~/Xenium_1/cells.parquet"
    gt_transcripts_path: Optional[str] = "~/Xenium_1/transcripts.parquet"

    github_repo_url: str = "https://github.com/Hussar1025/CellSeg_Benchmark.git"
    github_branch: str = "main"
    local_repo_dir: str = "~/value_code/CellSeg_Benchmark"
    github_pull: bool = True

    pred_cell_summary: Optional[str] = None
    pred_objects: Optional[str] = None
    pred_transcript_assignments: Optional[str] = None

    output_dir: str = "~/value_code/boms_xenium1_sparse_hungarian_transcript_qc_results"
    method_name: str = "BOMS"

    # Column names. Leave as default; the script also tries aliases.
    # Canonical ID column used internally after normalization.
    cell_id_col: str = "cell_id"
    # Source ID columns can differ between GT and predictions.
    gt_cell_id_col: str = "cell_id"
    pred_cell_id_col: str = "boms_cell_id"

    # Transcript-level columns.
    # Important for BOMS: transcript table has both `cell_id` (Xenium/GT-style)
    # and `boms_cell_id` (BOMS predicted segmentation ID). For prediction-side
    # transcript metrics we must use `boms_cell_id`, otherwise matched prediction
    # cells like 347/348 will never join to transcript assignments.
    pred_transcript_cell_id_col: str = "boms_cell_id"
    gt_transcript_cell_id_col: str = "cell_id"
    transcript_gene_col: str = "feature_name"
    transcript_id_col: str = "transcript_id"

    x_col: str = "x"
    y_col: str = "y"
    type_col: str = "cell_type"

    # Optional numeric columns, comma-separated from CLI. If empty, numeric metrics are skipped.
    numeric_cols: Optional[List[str]] = None
    # Optional cross-table numeric pairs, e.g. pred n_transcripts vs GT total_counts.
    # Format from CLI: --numeric-pairs n_transcripts:total_counts
    numeric_pairs: Optional[List[Tuple[str, str]]] = None

    # Matching
    match_radius: Optional[float] = None
    radius_quantile: float = 0.95
    radius_multiplier: float = 1.25
    max_component_size: int = 3000

    # Coordinate scaling: x' = x * scale_x, y' = y * scale_y.
    coordinate_scale_x: float = 1.0
    coordinate_scale_y: float = 1.0

    # KNN metrics
    k_neighbors: int = 8
    low_recall_warning_threshold: float = 0.70

    # Type aggregation QC
    warn_multitype_fraction: float = 0.01
    strict_unique_type_per_cell: bool = False

    # IO and diagnostics
    save_matched_table: bool = True
    save_cell_level_tables: bool = True
    verbose: bool = True


# -----------------------------
# IO utilities
# -----------------------------

def expand_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(os.path.expanduser(path)).resolve()


def is_git_lfs_pointer(path: str | Path) -> bool:
    """Return True if a file is a Git LFS pointer instead of real data."""
    p = expand_path(str(path))
    if p is None or not p.exists() or not p.is_file():
        return False
    try:
        with open(p, "rb") as f:
            head = f.read(256)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def require_not_lfs_pointer(path: str | Path) -> None:
    if is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer file, not the real dataset. "
            "Run: git lfs install && git -C <repo_dir> lfs pull. "
            "Or rerun this script after installing git-lfs."
        )


def _flatten_geojson_coordinates(coords: Any) -> List[Tuple[float, float]]:
    """Flatten GeoJSON Polygon/MultiPolygon coordinate arrays into x/y pairs."""
    points: List[Tuple[float, float]] = []
    if not isinstance(coords, (list, tuple)):
        return points
    if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
        points.append((float(coords[0]), float(coords[1])))
        return points
    for item in coords:
        points.extend(_flatten_geojson_coordinates(item))
    return points


def _polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid for a polygon ring; falls back to coordinate mean for degenerate rings."""
    if not points:
        return np.nan, np.nan
    pts = points[:-1] if len(points) > 2 and points[0] == points[-1] else points
    if len(pts) < 3:
        arr = np.asarray(pts, dtype=float)
        return float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1]))
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area2) <= EPS:
        arr = np.asarray(pts, dtype=float)
        return float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1]))
    return float(cx / (3.0 * area2)), float(cy / (3.0 * area2))


def read_geojson_cell_table(path: str | Path) -> pd.DataFrame:
    """Read proseg cell_polygons_proseg.geojson as a cell-level table.

    The evaluation code needs one row per predicted cell with an ID and centroid
    coordinates.  This reader preserves feature properties and derives x/y from
    polygon geometry only when the properties do not already provide coordinates.
    """
    path = expand_path(str(path))
    if path is None or not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    require_not_lfs_pointer(path)

    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj.get("features", []) if isinstance(gj, dict) else []
    rows: List[Dict[str, Any]] = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        row = dict(props) if isinstance(props, dict) else {}
        if "cell_id" not in row:
            for key in ["cell", "cellid", "CellID", "label", "object_id", "segmentation_id", "id"]:
                if key in row:
                    row["cell_id"] = row[key]
                    break
        if "cell_id" not in row:
            row["cell_id"] = feat.get("id", i)

        if not any(k in row for k in ["x", "centroid_x", "x_centroid", "center_x", "X", "global_x", "pos_x"]):
            geom = feat.get("geometry") or {}
            points = _flatten_geojson_coordinates(geom.get("coordinates"))
            row["x"], row["y"] = _polygon_centroid(points)
        elif not any(k in row for k in ["y", "centroid_y", "y_centroid", "center_y", "Y", "global_y", "pos_y"]):
            geom = feat.get("geometry") or {}
            points = _flatten_geojson_coordinates(geom.get("coordinates"))
            _, row["y"] = _polygon_centroid(points)

        rows.append(row)

    if not rows:
        raise ValueError(f"No GeoJSON features found in {path}")
    return pd.DataFrame(rows)


def read_anndata_zarr_cell_table(path: str | Path) -> pd.DataFrame:
    """Read proseg tables/table AnnData-Zarr output as a cell-level table."""
    path = expand_path(str(path))
    if path is None or not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import anndata as ad  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"{path} looks like an AnnData/Zarr table, but anndata is not installed. "
            "Install with `pip install anndata zarr` or pass --pred-cell-summary "
            "to cell_polygons_proseg.geojson."
        ) from exc

    adata = ad.read_zarr(str(path))
    df = adata.obs.copy().reset_index().rename(columns={"index": "cell_id"})
    if "cell_id" not in df.columns:
        df["cell_id"] = adata.obs_names.astype(str)

    # Proseg/OME-Zarr commonly stores cell centroids in obsm['spatial']; keep the
    # first two columns as x/y without changing downstream coordinate logic.
    spatial_key = None
    for key in ["spatial", "X_spatial", "centroids", "xy"]:
        if key in adata.obsm:
            spatial_key = key
            break
    if spatial_key is None and len(adata.obsm.keys()) > 0:
        spatial_key = list(adata.obsm.keys())[0]
    if spatial_key is not None:
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            if "x" not in df.columns:
                df["x"] = coords[:, 0]
            if "y" not in df.columns:
                df["y"] = coords[:, 1]

    if "n_transcripts" not in df.columns and getattr(adata, "X", None) is not None:
        try:
            xsum = adata.X.sum(axis=1)
            df["n_transcripts"] = np.asarray(xsum).reshape(-1)
        except Exception:
            pass
    return df


def run_git_lfs_pull(repo_dir: Path) -> None:
    """Fetch Git LFS objects when git-lfs is installed."""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Git LFS is required because the BOMS files in the GitHub repository are stored as LFS objects. "
            "Install it on the server, then rerun the script. On Ubuntu/Debian: "
            "sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install"
        ) from exc

    subprocess.run(["git", "-C", str(repo_dir), "lfs", "install", "--local"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "lfs", "pull"], check=True)


def ensure_boms_repo(cfg: EvaluationConfig) -> Path:
    """Clone or update the CellSeg_Benchmark repository and return its local path.

    BOMS result files are stored in the GitHub repository. This function makes the
    script reproducible on the server: it clones the repo if absent, and pulls the
    requested branch if the repo already exists and github_pull=True.
    """
    repo_dir = expand_path(cfg.local_repo_dir)
    if repo_dir is None:
        raise ValueError("local_repo_dir cannot be None")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        print(f"[INFO] Cloning BOMS benchmark repo: {cfg.github_repo_url} -> {repo_dir}")
        subprocess.run(
            ["git", "clone", "--branch", cfg.github_branch, cfg.github_repo_url, str(repo_dir)],
            check=True,
        )
    else:
        git_dir = repo_dir / ".git"
        if not git_dir.exists():
            raise FileNotFoundError(
                f"{repo_dir} exists but is not a git repository. "
                "Please remove it or set --local-repo-dir to another path."
            )
        if cfg.github_pull:
            print(f"[INFO] Updating BOMS benchmark repo in {repo_dir}")
            subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", cfg.github_branch], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "checkout", cfg.github_branch], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", cfg.github_branch], check=True)

    # The BOMS result files are stored with Git LFS. A normal git clone may only
    # download small pointer files, which pandas would read as a one-line CSV.
    run_git_lfs_pull(repo_dir)

    return repo_dir


def resolve_boms_paths(cfg: EvaluationConfig) -> Tuple[Path, Path, Path]:
    """Resolve BOMS prediction files from explicit paths or from the cloned GitHub repo."""
    repo_dir = ensure_boms_repo(cfg)
    default_dir = repo_dir / "test" / "data" / "boms_results"

    pred_cell_summary = (
        expand_path(cfg.pred_cell_summary)
        if cfg.pred_cell_summary
        else default_dir / "boms_cell_summary_filtered.csv"
    )
    pred_objects = (
        expand_path(cfg.pred_objects)
        if cfg.pred_objects
        else default_dir / "boms_objects_filtered.pkl"
    )
    pred_transcript_assignments = (
        expand_path(cfg.pred_transcript_assignments)
        if cfg.pred_transcript_assignments
        else default_dir / "boms_transcript_assignments_filtered.parquet"
    )

    missing = [
        str(p) for p in [pred_cell_summary, pred_objects, pred_transcript_assignments]
        if p is None or not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing BOMS file(s) after cloning/pulling GitHub repository:\n" + "\n".join(missing)
        )

    pointer_files = [str(p) for p in [pred_cell_summary, pred_objects, pred_transcript_assignments] if p is not None and p.is_file() and is_git_lfs_pointer(p)]
    if pointer_files:
        raise RuntimeError(
            "These BOMS files are still Git LFS pointer files, not real data:\n"
            + "\n".join(pointer_files)
            + "\nInstall git-lfs, then run: git -C " + str(repo_dir) + " lfs pull"
        )

    return pred_cell_summary, pred_objects, pred_transcript_assignments


def read_table(path: str | Path) -> pd.DataFrame:
    path = expand_path(str(path))
    if path is None or not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.is_dir():
        if (path / ".zgroup").exists() or (path / ".zattrs").exists():
            return read_anndata_zarr_cell_table(path)
        raise ValueError(f"Unsupported directory table format: {path}")

    require_not_lfs_pointer(path)

    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        return read_geojson_cell_table(path)
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            for key in ["cells", "cell_table", "cell_df", "metadata", "data"]:
                if key in obj and isinstance(obj[key], pd.DataFrame):
                    return obj[key]
            try:
                return pd.DataFrame(obj)
            except Exception as exc:
                raise ValueError(f"Unsupported pkl object in {path}: {type(obj)}") from exc
        raise ValueError(f"Unsupported pkl object in {path}: {type(obj)}")

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather"}:
        return pd.read_feather(path)

    raise ValueError(f"Unsupported file suffix: {suffix} for {path}")


def auto_find_gt_table(gt_root: str | Path) -> Path:
    root = expand_path(str(gt_root))
    if root is None or not root.exists():
        raise FileNotFoundError(f"GT root not found: {root}")

    priority_names = [
        "gt_cells.parquet", "gt_cells.csv", "gt_cell_table.parquet", "gt_cell_table.csv",
        "cell_table.parquet", "cell_table.csv", "cells.parquet", "cells.csv",
        "cell_metadata.parquet", "cell_metadata.csv", "cell_summary.parquet", "cell_summary.csv",
    ]
    for name in priority_names:
        candidate = root / name
        if candidate.exists():
            return candidate

    candidates: List[Path] = []
    for pattern in ["*.parquet", "*.csv", "*.pkl", "*.pickle", "*.tsv"]:
        candidates.extend(root.rglob(pattern))

    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        lower = p.name.lower()
        score = 0
        if "cell" in lower:
            score += 10
        if "summary" in lower or "metadata" in lower or "table" in lower:
            score += 5
        if "transcript" in lower or "molecule" in lower or "object" in lower:
            score -= 20
        scored.append((score, p))

    scored = sorted(scored, key=lambda x: (-x[0], len(str(x[1]))))
    if not scored or scored[0][0] < 0:
        raise FileNotFoundError(
            f"Could not auto-detect GT cell table under {root}. "
            "Please pass --gt-path explicitly."
        )
    return scored[0][1]


def normalize_columns(df: pd.DataFrame, cfg: EvaluationConfig, table_name: str) -> pd.DataFrame:
    """Rename common aliases to canonical names: cell_id, x, y, cell_type.

    GT and prediction files often use different cell ID columns.  For this BOMS/Xenium
    evaluation, GT uses `cell_id`, while BOMS uses `boms_cell_id`.  Both are renamed
    internally to the canonical `cell_id` so downstream matching can be shared.
    """
    if table_name.lower().startswith("pred"):
        id_aliases = [cfg.pred_cell_id_col, cfg.cell_id_col, "boms_cell_id", "cell", "cellid", "cell_ID", "CellID", "label", "object_id", "segmentation_id"]
    else:
        id_aliases = [cfg.gt_cell_id_col, cfg.cell_id_col, "cell", "cellid", "cell_ID", "CellID", "label", "object_id", "segmentation_id"]

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
            f"Available columns include: {list(out.columns)[:40]}"
        )
    return out


# -----------------------------
# Cell-level collapsing + QC
# -----------------------------

def _mode_with_confidence(values: pd.Series) -> Tuple[Any, int, int, float, str]:
    """Return mode, n_unique, n_valid, confidence, all_modes_repr.

    Confidence = count(mode) / n_valid. If tie exists, the first mode after stable sorting is used,
    but all modes are recorded to make ambiguity visible.
    """
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


def collapse_to_cell_level(df: pd.DataFrame, cfg: EvaluationConfig, table_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = normalize_columns(df, cfg, table_name=table_name)

    for col in [cfg.x_col, cfg.y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    coord_bad = df[[cfg.x_col, cfg.y_col]].isna().any(axis=1)
    if coord_bad.any():
        warnings.warn(f"{table_name}: dropping {int(coord_bad.sum())} rows with missing/non-numeric coordinates.")
        df = df.loc[~coord_bad].copy()

    if df.empty:
        raise ValueError(f"{table_name}: no valid rows after coordinate cleaning.")

    # Apply coordinate scaling before matching/metrics.
    df[cfg.x_col] = df[cfg.x_col] * cfg.coordinate_scale_x
    df[cfg.y_col] = df[cfg.y_col] * cfg.coordinate_scale_y

    grouped = df.groupby(cfg.cell_id_col, sort=False, observed=True)
    base = grouped[[cfg.x_col, cfg.y_col]].mean().reset_index()

    qc: Dict[str, Any] = {
        f"{table_name}_raw_rows": int(len(df)),
        f"{table_name}_n_cells": int(base.shape[0]),
        f"{table_name}_mean_rows_per_cell": float(len(df) / max(base.shape[0], 1)),
    }

    # Type aggregation with ambiguity QC.
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
            msg = (
                f"{table_name}: {multitype_fraction:.3%} cells have multiple {cfg.type_col} values. "
                "The main type is chosen by mode; ambiguity QC columns are saved."
            )
            if cfg.strict_unique_type_per_cell:
                raise ValueError(msg)
            warnings.warn(msg)

    # Numeric aggregation.
    #
    # Important: --numeric-pairs pred_col:gt_col should work even when the user does
    # not also pass those columns through --numeric-cols.  Therefore, the cell-level
    # table must preserve pair-specific numeric columns from the relevant side:
    #   pred table: left side of each pair
    #   gt table:   right side of each pair
    numeric_to_aggregate: List[str] = []
    if cfg.numeric_cols:
        numeric_to_aggregate.extend(cfg.numeric_cols)
    if cfg.numeric_pairs:
        if table_name.lower().startswith("pred"):
            numeric_to_aggregate.extend([p for p, _ in cfg.numeric_pairs])
        else:
            numeric_to_aggregate.extend([g for _, g in cfg.numeric_pairs])

    # Stable de-duplication while preserving user order.
    seen_numeric: set[str] = set()
    numeric_to_aggregate = [
        c for c in numeric_to_aggregate
        if not (c in seen_numeric or seen_numeric.add(c))
    ]

    for col in numeric_to_aggregate:
        if col not in df.columns:
            warnings.warn(
                f"{table_name}: numeric column '{col}' is missing; "
                "metrics depending on it will be NaN."
            )
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({cfg.cell_id_col: df[cfg.cell_id_col].values, col: numeric.values})
        agg = tmp.groupby(cfg.cell_id_col, sort=False, observed=True)[col].mean().reset_index()
        base = base.merge(agg, on=cfg.cell_id_col, how="left")
        qc.update(numeric_qc_for_series(agg[col], prefix=f"{table_name}_{col}"))

    check_coordinate_scale_warning(base, cfg, table_name)
    return base, qc


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
            f"{table_name}: coordinate span is very large ({span_x:.1f}, {span_y:.1f}) and match_radius is not set. "
            "Ensure coordinates are in comparable physical units or set --match-radius / coordinate scales."
        )


# -----------------------------
# Matching
# -----------------------------

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


def build_candidate_edges(
    pred_cells: pd.DataFrame,
    gt_cells: pd.DataFrame,
    cfg: EvaluationConfig,
    radius: float,
) -> pd.DataFrame:
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


def _connected_components_bipartite(edges: pd.DataFrame, n_pred: int, n_gt: int) -> List[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Connected components on the bipartite candidate graph.

    Returns list of (pred_indices, gt_indices, component_edges).
    """
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
    components: List[Tuple[np.ndarray, np.ndarray, pd.DataFrame]] = []

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
        components.append((np.array(sorted(comp_pred), dtype=int), np.array(sorted(comp_gt), dtype=int), comp_edge_df))

    return components


def match_sparse_hungarian(
    pred_cells: pd.DataFrame,
    gt_cells: pd.DataFrame,
    cfg: EvaluationConfig,
    radius: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Radius-constrained sparse Hungarian matching via connected components.

    The global candidate graph is split into connected components. For each component, dense Hungarian
    is applied only to that local block. This is mathematically equivalent to solving each independent
    component separately because no edges exist across components.
    """
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
            # Fall back to greedy inside this very large component to avoid pathological local dense matrices.
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


def empty_matches() -> pd.DataFrame:
    return pd.DataFrame(columns=["pred_index", "gt_index", "distance"])


def attach_match_metadata(matches: pd.DataFrame, pred_cells: pd.DataFrame, gt_cells: pd.DataFrame, cfg: EvaluationConfig) -> pd.DataFrame:
    """Attach cell-level metadata to each accepted pred-GT match.

    After normalize_columns(), both GT and prediction tables use the canonical
    internal ID column cfg.cell_id_col.  This function intentionally prefixes all
    non-index, non-coordinate, non-ID columns with pred_ or gt_.  For example:

        pred n_transcripts -> pred_n_transcripts
        gt   total_counts  -> gt_total_counts

    This is what the --numeric-pairs module expects.
    """
    if matches.empty:
        return matches

    pred_meta = pred_cells.reset_index(drop=True).reset_index().rename(columns={
        "index": "pred_index",
        cfg.cell_id_col: "pred_cell_id",
        cfg.x_col: "pred_x",
        cfg.y_col: "pred_y",
    })
    gt_meta = gt_cells.reset_index(drop=True).reset_index().rename(columns={
        "index": "gt_index",
        cfg.cell_id_col: "gt_cell_id",
        cfg.x_col: "gt_x",
        cfg.y_col: "gt_y",
    })

    pred_keep = {"pred_index", "pred_cell_id", "pred_x", "pred_y"}
    gt_keep = {"gt_index", "gt_cell_id", "gt_x", "gt_y"}

    pred_rename = {
        c: f"pred_{c}"
        for c in pred_meta.columns
        if c not in pred_keep and not c.startswith("pred_")
    }
    gt_rename = {
        c: f"gt_{c}"
        for c in gt_meta.columns
        if c not in gt_keep and not c.startswith("gt_")
    }

    pred_meta = pred_meta.rename(columns=pred_rename)
    gt_meta = gt_meta.rename(columns=gt_rename)

    out = matches.copy()
    out = out.merge(pred_meta, on="pred_index", how="left")
    out = out.merge(gt_meta, on="gt_index", how="left")
    return out


# -----------------------------
# Metrics
# -----------------------------

def detection_metrics(matches: pd.DataFrame, n_pred: int, n_gt: int) -> Dict[str, Any]:
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


def spatial_shift_metrics(matches: Optional[pd.DataFrame]) -> Dict[str, Any]:
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


def type_metrics(matches: pd.DataFrame, cfg: EvaluationConfig) -> Dict[str, Any]:
    pred_col = f"pred_{cfg.type_col}"
    gt_col = f"gt_{cfg.type_col}"
    if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns:
        return {
            "cell_type_accuracy": np.nan,
            "cell_type_macro_f1": np.nan,
            "cell_type_weighted_f1": np.nan,
            "cell_type_balanced_accuracy": np.nan,
            "ARI_matched_type_labels": np.nan,
            "NMI_matched_type_labels": np.nan,
            "cell_type_purity_legacy_matched_accuracy": np.nan,
            "matched_type_valid_n": 0,
        }
    tmp = matches[[pred_col, gt_col]].dropna().copy()
    if tmp.empty:
        return {
            "cell_type_accuracy": np.nan,
            "cell_type_macro_f1": np.nan,
            "cell_type_weighted_f1": np.nan,
            "cell_type_balanced_accuracy": np.nan,
            "ARI_matched_type_labels": np.nan,
            "NMI_matched_type_labels": np.nan,
            "cell_type_purity_legacy_matched_accuracy": np.nan,
            "matched_type_valid_n": 0,
        }
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
        # Kept for backward compatibility; this is accuracy, not true cluster purity.
        "cell_type_purity_legacy_matched_accuracy": acc,
        "matched_type_valid_n": int(tmp.shape[0]),
    }


def morans_i_knn(coords: np.ndarray, values: np.ndarray, k: int = 8) -> float:
    coords = np.asarray(coords, dtype=float)
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    coords = coords[valid]
    values = values[valid]
    n = len(values)
    if n <= k + 1:
        return np.nan
    if np.nanstd(values) <= EPS:
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


def numeric_spatial_metrics(pred_cells: pd.DataFrame, gt_cells: pd.DataFrame, matches: pd.DataFrame, cfg: EvaluationConfig) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if not cfg.numeric_cols:
        return res

    for col in cfg.numeric_cols:
        pred_has = col in pred_cells.columns
        gt_has = col in gt_cells.columns
        if not pred_has:
            warnings.warn(f"numeric column '{col}' missing in prediction cell table.")
        if not gt_has:
            warnings.warn(f"numeric column '{col}' missing in GT cell table.")

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


def knn_neighbor_type_consistency_on_gt_graph(matches: pd.DataFrame, cfg: EvaluationConfig) -> Dict[str, Any]:
    """Compare local cell-type composition using only matched cells and GT coordinates.

    Important limitation: this is a matched-cell-only GT graph. If detection recall is low, the graph is
    no longer representative of the full tissue. The output includes valid_n and should be interpreted
    together with detection_recall / matched_fraction_gt.
    """
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

    df = matches[["gt_x", "gt_y", pred_type_col, gt_type_col]].dropna().copy()
    if df.shape[0] <= cfg.k_neighbors + 1:
        return {
            "knn_neighbor_type_pearson_gt_graph": np.nan,
            "knn_neighbor_type_mean_js_distance_gt_graph": np.nan,
            "knn_neighbor_type_mean_js_divergence_gt_graph": np.nan,
            "knn_neighbor_metric_valid_n": int(df.shape[0]),
        }

    coords = df[["gt_x", "gt_y"]].to_numpy(dtype=float)
    pred_labels = df[pred_type_col].astype(str).to_numpy()
    gt_labels = df[gt_type_col].astype(str).to_numpy()
    all_types = sorted(set(pred_labels).union(set(gt_labels)))
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    n_types = len(all_types)

    nbrs = NearestNeighbors(n_neighbors=cfg.k_neighbors + 1, algorithm="auto", n_jobs=-1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    pears: List[float] = []
    js_distances: List[float] = []
    js_divergences: List[float] = []

    for i in range(df.shape[0]):
        neigh = indices[i, 1:]
        p_vec = np.zeros(n_types, dtype=float)
        g_vec = np.zeros(n_types, dtype=float)
        for lab in pred_labels[neigh]:
            p_vec[type_to_idx[lab]] += 1.0
        for lab in gt_labels[neigh]:
            g_vec[type_to_idx[lab]] += 1.0
        p_vec /= max(p_vec.sum(), EPS)
        g_vec /= max(g_vec.sum(), EPS)

        if np.std(p_vec) > EPS and np.std(g_vec) > EPS:
            r = pearsonr(p_vec, g_vec)[0]
            if np.isfinite(r):
                pears.append(float(r))
        jsd = float(jensenshannon(p_vec, g_vec, base=2.0))
        if np.isfinite(jsd):
            js_distances.append(jsd)
            js_divergences.append(jsd ** 2)

    return {
        "knn_neighbor_type_pearson_gt_graph": float(np.mean(pears)) if pears else np.nan,
        "knn_neighbor_type_mean_js_distance_gt_graph": float(np.mean(js_distances)) if js_distances else np.nan,
        "knn_neighbor_type_mean_js_divergence_gt_graph": float(np.mean(js_divergences)) if js_divergences else np.nan,
        "knn_neighbor_metric_valid_n": int(df.shape[0]),
    }


def numeric_pair_metrics(matches: pd.DataFrame, cfg: EvaluationConfig) -> Dict[str, Any]:
    """Matched numeric metrics for differently named prediction/GT columns.

    Example: BOMS has `n_transcripts`, Xenium GT has `total_counts`.
    Use `--numeric-pairs n_transcripts:total_counts` to compare them on matched cells.
    """
    res: Dict[str, Any] = {}
    if not cfg.numeric_pairs:
        return res
    for pred_col_raw, gt_col_raw in cfg.numeric_pairs:
        pred_col = f"pred_{pred_col_raw}"
        gt_col = f"gt_{gt_col_raw}"
        label = f"{pred_col_raw}_vs_{gt_col_raw}"
        if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns:
            warnings.warn(
                f"numeric pair '{pred_col_raw}:{gt_col_raw}' cannot be evaluated; "
                f"missing columns after matching: "
                f"{pred_col if pred_col not in matches.columns else ''} "
                f"{gt_col if gt_col not in matches.columns else ''}"
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


def low_recall_knn_warning(metrics: Dict[str, Any], cfg: EvaluationConfig) -> Dict[str, Any]:
    recall = metrics.get("detection_recall", np.nan)
    low = bool(np.isfinite(recall) and recall < cfg.low_recall_warning_threshold)
    if low:
        warnings.warn(
            f"Detection recall is {recall:.3f}, below {cfg.low_recall_warning_threshold:.3f}. "
            "KNN neighborhood type consistency is matched-cell-only and may be biased."
        )
    return {
        "knn_metric_matched_only": True,
        "knn_metric_low_recall_warning": low,
        "knn_metric_recall_threshold": float(cfg.low_recall_warning_threshold),
    }


# -----------------------------
# Transcript-level metrics
# -----------------------------

def _first_existing_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    for c in aliases:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in aliases:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def read_transcript_table(path: str | Path) -> pd.DataFrame:
    """Read transcript/point assignment tables from parquet/csv or a directory containing parquet.

    This is intentionally permissive because BOMS stores transcript assignments as a
    parquet file, while Proseg/SpatialData exports points/transcripts as a directory
    containing points.parquet.
    """
    p = expand_path(str(path))
    if p is None or not p.exists():
        raise FileNotFoundError(f"Transcript table not found: {p}")
    if p.is_dir():
        preferred = [p / "points.parquet", p / "transcripts.parquet"]
        for q in preferred:
            if q.exists():
                p = q
                break
        else:
            candidates = sorted(list(p.rglob("*.parquet")) + list(p.rglob("*.csv")) + list(p.rglob("*.csv.gz")))
            if not candidates:
                raise FileNotFoundError(f"No parquet/csv transcript table found under directory: {p}")
            p = candidates[0]
    require_not_lfs_pointer(p)
    s = p.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(p)
    if s == ".csv" or str(p).endswith(".csv.gz"):
        return pd.read_csv(p)
    if s in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    raise ValueError(f"Unsupported transcript table format: {p}")


def _normalize_id_series(s: pd.Series) -> pd.Series:
    """Normalize IDs for safe joins without changing original columns.

    Keeps string IDs such as Xenium `aaaanbjb-1` unchanged, converts numeric-like
    IDs such as 347 or 347.0 to `347`, strips whitespace, and marks common
    unassigned/background values as missing.
    """
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    bad = out.str.lower().isin({"", "nan", "none", "null", "na", "-1", "0", "unassigned", "background"})
    return out.mask(bad)


def normalize_transcript_assignments(df: pd.DataFrame, cfg: EvaluationConfig, table_name: str) -> pd.DataFrame:
    """Add canonical transcript-evaluation columns without overwriting raw columns.

    This function intentionally does NOT rename raw `cell_id`, because BOMS
    transcript assignments contain two different ID systems:
      - `cell_id`: Xenium/GT-style cell ID, e.g. bapencgb-1
      - `boms_cell_id`: BOMS predicted cell ID, e.g. 347

    Prediction-side transcript metrics must join matched `pred_cell_id` to
    `boms_cell_id`; GT-side transcript metrics must join matched `gt_cell_id` to
    `cell_id`. The canonical evaluation columns are:
      - EVAL_CELL_ID
      - EVAL_GENE
      - EVAL_TRANSCRIPT_ID
    """
    out = df.copy()
    is_pred = table_name.lower().startswith("pred")

    if is_pred:
        cell_aliases = [
            cfg.pred_transcript_cell_id_col,
            "boms_cell_id",
            cfg.pred_cell_id_col,
            "segmentation_cell_id",
            "assigned_cell_id",
            "cell_id",
            "cell", "cellid", "cell_ID", "CellID", "label", "object_id", "instance_id",
        ]
    else:
        cell_aliases = [
            cfg.gt_transcript_cell_id_col,
            cfg.gt_cell_id_col,
            "cell_id",
            "cell", "cellid", "cell_ID", "CellID", "assigned_cell_id",
            "segmentation_cell_id", "label", "object_id", "instance_id",
        ]

    # Respect explicit user/default gene and transcript ID columns first.
    gene_col = _first_existing_column(out, [
        cfg.transcript_gene_col,
        "feature_name", "gene", "gene_name", "target", "name", "feature", "Gene", "target_name"
    ])
    cell_col = _first_existing_column(out, cell_aliases)
    x_col = _first_existing_column(out, [cfg.x_col, "x", "x_location", "global_x", "X", "center_x"])
    y_col = _first_existing_column(out, [cfg.y_col, "y", "y_location", "global_y", "Y", "center_y"])
    tid_col = _first_existing_column(out, [cfg.transcript_id_col, "transcript_id", "transcript", "molecule_id", "id", "barcode"])

    if cell_col is not None:
        out[EVAL_CELL_ID] = _normalize_id_series(out[cell_col])
    else:
        out[EVAL_CELL_ID] = pd.Series(pd.NA, index=out.index, dtype="string")

    if gene_col is not None:
        out[EVAL_GENE] = out[gene_col].astype("string").str.strip()
        out[EVAL_GENE] = out[EVAL_GENE].mask(out[EVAL_GENE].str.lower().isin({"", "nan", "none", "null"}))

    if tid_col is not None:
        out[EVAL_TRANSCRIPT_ID] = out[tid_col].astype("string").str.strip()
        out[EVAL_TRANSCRIPT_ID] = out[EVAL_TRANSCRIPT_ID].mask(out[EVAL_TRANSCRIPT_ID].str.lower().isin({"", "nan", "none", "null"}))

    if x_col is not None:
        out[cfg.x_col] = pd.to_numeric(out[x_col], errors="coerce")
    if y_col is not None:
        out[cfg.y_col] = pd.to_numeric(out[y_col], errors="coerce")

    out.attrs["eval_cell_source_column"] = cell_col
    out.attrs["eval_gene_source_column"] = gene_col
    out.attrs["eval_transcript_id_source_column"] = tid_col
    return out

def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    valid = xv.notna() & yv.notna()
    xv = xv[valid].to_numpy(dtype=float)
    yv = yv[valid].to_numpy(dtype=float)
    if len(xv) < 3 or np.std(xv) <= EPS or np.std(yv) <= EPS:
        return np.nan
    return float(pearsonr(xv, yv)[0] if method == "pearson" else spearmanr(xv, yv)[0])


def transcript_level_metrics(
    pred_transcript_path: str | Path,
    gt_transcript_path: str | Path,
    matches: pd.DataFrame,
    cfg: EvaluationConfig,
) -> Dict[str, Any]:
    """Transcript-level evaluation on the same matched cell pairs used above.

    Added metrics:
      - assignment rate in pred/GT transcript tables
      - matched-cell transcript count correlation/MAE/RMSE
      - matched-cell gene-count vector cosine similarity, JS distance, Pearson
      - direct transcript assignment accuracy when shared transcript IDs exist

    For BOMS, prediction transcript assignments are joined through `boms_cell_id`,
    while GT transcripts are joined through `cell_id`.
    """
    res: Dict[str, Any] = {}
    try:
        pred_raw = normalize_transcript_assignments(read_transcript_table(pred_transcript_path), cfg, "pred_transcripts")
        gt_raw = normalize_transcript_assignments(read_transcript_table(gt_transcript_path), cfg, "gt_transcripts")
    except Exception as exc:
        warnings.warn(f"Transcript-level metrics skipped: {exc}")
        return {
            "transcript_metrics_available": False,
            "transcript_metrics_error": str(exc),
        }

    res["transcript_metrics_available"] = True
    res["pred_transcript_rows"] = int(len(pred_raw))
    res["gt_transcript_rows"] = int(len(gt_raw))
    res["pred_transcript_cell_id_source_col"] = str(pred_raw.attrs.get("eval_cell_source_column"))
    res["gt_transcript_cell_id_source_col"] = str(gt_raw.attrs.get("eval_cell_source_column"))
    res["pred_transcript_gene_source_col"] = str(pred_raw.attrs.get("eval_gene_source_column"))
    res["gt_transcript_gene_source_col"] = str(gt_raw.attrs.get("eval_gene_source_column"))
    res["pred_transcript_id_source_col"] = str(pred_raw.attrs.get("eval_transcript_id_source_column"))
    res["gt_transcript_id_source_col"] = str(gt_raw.attrs.get("eval_transcript_id_source_column"))

    for label, df in [("pred", pred_raw), ("gt", gt_raw)]:
        assigned = df[EVAL_CELL_ID].notna() if EVAL_CELL_ID in df.columns else pd.Series(False, index=df.index)
        res[f"{label}_transcript_assigned_n"] = int(assigned.sum())
        res[f"{label}_transcript_assignment_rate"] = float(assigned.mean()) if len(df) else np.nan
        res[f"{label}_transcript_has_gene_col"] = bool(EVAL_GENE in df.columns)

    if matches.empty or "pred_cell_id" not in matches.columns or "gt_cell_id" not in matches.columns:
        res["transcript_matched_cell_pairs_n"] = 0
        return res

    pair_map = matches[["pred_cell_id", "gt_cell_id"]].dropna().copy()
    pair_map["pred_eval_cell_id"] = _normalize_id_series(pair_map["pred_cell_id"])
    pair_map["gt_eval_cell_id"] = _normalize_id_series(pair_map["gt_cell_id"])
    pair_map = pair_map.dropna(subset=["pred_eval_cell_id", "gt_eval_cell_id"]).copy()
    pair_map["pair_id"] = np.arange(len(pair_map), dtype=int)
    res["transcript_matched_cell_pairs_n"] = int(len(pair_map))

    pred_assigned = pred_raw.dropna(subset=[EVAL_CELL_ID]).copy() if EVAL_CELL_ID in pred_raw.columns else pred_raw.iloc[0:0].copy()
    gt_assigned = gt_raw.dropna(subset=[EVAL_CELL_ID]).copy() if EVAL_CELL_ID in gt_raw.columns else gt_raw.iloc[0:0].copy()

    pred_assigned = pred_assigned.merge(
        pair_map[["pred_eval_cell_id", "pair_id"]],
        left_on=EVAL_CELL_ID,
        right_on="pred_eval_cell_id",
        how="inner",
    )
    gt_assigned = gt_assigned.merge(
        pair_map[["gt_eval_cell_id", "pair_id"]],
        left_on=EVAL_CELL_ID,
        right_on="gt_eval_cell_id",
        how="inner",
    )

    res["pred_transcripts_in_matched_cells_n"] = int(len(pred_assigned))
    res["gt_transcripts_in_matched_cells_n"] = int(len(gt_assigned))

    pred_counts = pred_assigned.groupby("pair_id").size().rename("pred_transcript_count")
    gt_counts = gt_assigned.groupby("pair_id").size().rename("gt_transcript_count")
    count_df = (
        pair_map[["pair_id"]]
        .merge(pred_counts, on="pair_id", how="left")
        .merge(gt_counts, on="pair_id", how="left")
        .fillna(0)
    )
    diff = count_df["pred_transcript_count"] - count_df["gt_transcript_count"]
    res.update({
        "matched_cell_transcript_count_pearson": _safe_corr(count_df["pred_transcript_count"], count_df["gt_transcript_count"], "pearson"),
        "matched_cell_transcript_count_spearman": _safe_corr(count_df["pred_transcript_count"], count_df["gt_transcript_count"], "spearman"),
        "matched_cell_transcript_count_mae": float(np.mean(np.abs(diff))) if len(diff) else np.nan,
        "matched_cell_transcript_count_rmse": float(np.sqrt(np.mean(diff ** 2))) if len(diff) else np.nan,
    })

    if EVAL_GENE in pred_assigned.columns and EVAL_GENE in gt_assigned.columns:
        pred_gene = pred_assigned.dropna(subset=[EVAL_GENE]).groupby(["pair_id", EVAL_GENE]).size().rename("pred").reset_index()
        gt_gene = gt_assigned.dropna(subset=[EVAL_GENE]).groupby(["pair_id", EVAL_GENE]).size().rename("gt").reset_index()
        merged = pred_gene.merge(gt_gene, on=["pair_id", EVAL_GENE], how="outer").fillna(0)
        cos_vals: List[float] = []
        js_vals: List[float] = []
        gene_corr_vals: List[float] = []
        for _, sub in merged.groupby("pair_id", sort=False):
            p = sub["pred"].to_numpy(dtype=float)
            g = sub["gt"].to_numpy(dtype=float)
            denom = float(np.linalg.norm(p) * np.linalg.norm(g))
            if denom > EPS:
                cos_vals.append(float(np.dot(p, g) / denom))
            if p.sum() > 0 and g.sum() > 0:
                js_vals.append(float(jensenshannon(p / p.sum(), g / g.sum(), base=2.0)))
            if len(p) >= 3 and np.std(p) > EPS and np.std(g) > EPS:
                r = pearsonr(p, g)[0]
                if np.isfinite(r):
                    gene_corr_vals.append(float(r))
        res.update({
            "matched_cell_gene_vector_mean_cosine": float(np.mean(cos_vals)) if cos_vals else np.nan,
            "matched_cell_gene_vector_mean_js_distance": float(np.mean(js_vals)) if js_vals else np.nan,
            "matched_cell_gene_vector_mean_pearson": float(np.mean(gene_corr_vals)) if gene_corr_vals else np.nan,
            "matched_cell_gene_vector_valid_pairs": int(max(len(cos_vals), len(js_vals), len(gene_corr_vals))),
        })
    else:
        res.update({
            "matched_cell_gene_vector_mean_cosine": np.nan,
            "matched_cell_gene_vector_mean_js_distance": np.nan,
            "matched_cell_gene_vector_mean_pearson": np.nan,
            "matched_cell_gene_vector_valid_pairs": 0,
        })

    # Direct transcript-level assignment accuracy: compare shared transcript IDs and
    # count a transcript as correct if the predicted assigned cell and GT assigned cell
    # are a matched pred-GT pair. For BOMS this uses pred `boms_cell_id` and GT `cell_id`.
    if EVAL_TRANSCRIPT_ID in pred_raw.columns and EVAL_TRANSCRIPT_ID in gt_raw.columns:
        pred_tid = pred_raw[[EVAL_TRANSCRIPT_ID, EVAL_CELL_ID]].dropna().copy()
        gt_tid = gt_raw[[EVAL_TRANSCRIPT_ID, EVAL_CELL_ID]].dropna().copy()
        pred_tid = pred_tid.rename(columns={EVAL_CELL_ID: "pred_eval_cell_id"})
        gt_tid = gt_tid.rename(columns={EVAL_CELL_ID: "gt_eval_cell_id"})
        id_df = pred_tid.merge(gt_tid, on=EVAL_TRANSCRIPT_ID, how="inner")
        id_df = id_df.merge(
            pair_map[["pred_eval_cell_id", "gt_eval_cell_id", "pair_id"]],
            on=["pred_eval_cell_id", "gt_eval_cell_id"],
            how="left",
        )
        res["transcript_id_overlap_n"] = int(len(id_df))
        res["transcript_assignment_accuracy_via_matched_cells"] = float(id_df["pair_id"].notna().mean()) if len(id_df) else np.nan
    else:
        res["transcript_id_overlap_n"] = 0
        res["transcript_assignment_accuracy_via_matched_cells"] = np.nan

    return res

# -----------------------------
# Pipeline
# -----------------------------

def evaluate(cfg: EvaluationConfig) -> pd.DataFrame:
    outdir = expand_path(cfg.output_dir)
    assert outdir is not None
    outdir.mkdir(parents=True, exist_ok=True)

    gt_path = expand_path(cfg.gt_path) if cfg.gt_path else auto_find_gt_table(cfg.gt_root)
    pred_path, pred_objects_path, pred_transcript_assignments_path = resolve_boms_paths(cfg)

    if cfg.verbose:
        print(f"[INFO] GT cell table: {gt_path}")
        print(f"[INFO] Prediction cell summary: {pred_path}")
        print(f"[INFO] Prediction objects path recorded: {pred_objects_path}")
        print(f"[INFO] Prediction transcript assignments path recorded: {pred_transcript_assignments_path}")

    gt_raw = read_table(gt_path)
    pred_raw = read_table(pred_path)

    gt_cells, gt_qc = collapse_to_cell_level(gt_raw, cfg, table_name="gt")
    pred_cells, pred_qc = collapse_to_cell_level(pred_raw, cfg, table_name="pred")

    radius = cfg.match_radius if cfg.match_radius is not None else infer_match_radius(gt_cells, cfg)
    if cfg.verbose:
        print(f"[INFO] Using match radius: {radius:.6g}")
        print(f"[INFO] GT cells: {len(gt_cells):,}; Pred cells: {len(pred_cells):,}")

    matches, match_qc = match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=radius)

    metrics: Dict[str, Any] = {
        "method": cfg.method_name,
        "gt_path": str(gt_path),
        "pred_cell_summary": str(pred_path),
        "pred_objects": str(pred_objects_path),
        "pred_transcript_assignments": str(pred_transcript_assignments_path),
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
    gt_transcripts_path = expand_path(cfg.gt_transcripts_path) if cfg.gt_transcripts_path else (expand_path(cfg.gt_root) / "transcripts.parquet")
    metrics.update(transcript_level_metrics(pred_transcript_assignments_path, gt_transcripts_path, matches, cfg))

    result_df = pd.DataFrame([metrics])

    result_prefix = f"{cfg.method_name.lower()}_xenium1_sparse_hungarian_transcript_qc_metrics"
    result_csv = outdir / f"{result_prefix}.csv"
    result_json = outdir / f"{result_prefix}.json"
    result_df.to_csv(result_csv, index=False)
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)

    cfg_json = outdir / "evaluation_config.json"
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    if cfg.save_matched_table:
        matches.to_csv(outdir / "matched_cells_sparse_hungarian.csv", index=False)
    if cfg.save_cell_level_tables:
        pred_cells.to_csv(outdir / "pred_cell_level_table.csv", index=False)
        gt_cells.to_csv(outdir / "gt_cell_level_table.csv", index=False)

    if cfg.verbose:
        print("[DONE] Metrics saved to:", result_csv)
        print(result_df.T.to_string(header=False))
    return result_df


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluate BOMS against Xenium_1 GT with sparse Hungarian matching + transcript-level QC.")
    parser.add_argument("--gt-root", default="~/Xenium_1")
    parser.add_argument("--gt-path", default="~/Xenium_1/cells.parquet")
    parser.add_argument("--gt-transcripts-path", default="~/Xenium_1/transcripts.parquet")

    parser.add_argument("--github-repo-url", default="https://github.com/Hussar1025/CellSeg_Benchmark.git")
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--local-repo-dir", default="~/value_code/CellSeg_Benchmark")
    parser.add_argument("--no-github-pull", action="store_true", help="Use existing local repository without git pull.")

    parser.add_argument("--pred-cell-summary", default=None, help="Optional override. If omitted, use GitHub repo test/data/boms_results/boms_cell_summary_filtered.csv")
    parser.add_argument("--pred-objects", default=None, help="Optional override. If omitted, use GitHub repo test/data/boms_results/boms_objects_filtered.pkl")
    parser.add_argument("--pred-transcript-assignments", default=None, help="Optional override. If omitted, use GitHub repo test/data/boms_results/boms_transcript_assignments_filtered.parquet")
    parser.add_argument("--output-dir", default="~/value_code/boms_xenium1_sparse_hungarian_transcript_qc_results")
    parser.add_argument("--method-name", default="BOMS")

    parser.add_argument("--cell-id-col", default="cell_id", help="Internal canonical cell id column. Keep as cell_id for this pipeline.")
    parser.add_argument("--gt-cell-id-col", default="cell_id", help="Cell ID column name in GT table, e.g. cell_id.")
    parser.add_argument("--pred-cell-id-col", default="boms_cell_id", help="Cell ID column name in BOMS prediction table, e.g. boms_cell_id.")
    parser.add_argument("--pred-transcript-cell-id-col", default="boms_cell_id", help="Prediction transcript assignment cell ID column. For BOMS this must be boms_cell_id.")
    parser.add_argument("--gt-transcript-cell-id-col", default="cell_id", help="GT transcript assignment cell ID column.")
    parser.add_argument("--transcript-gene-col", default="feature_name", help="Gene/feature column in transcript tables.")
    parser.add_argument("--transcript-id-col", default="transcript_id", help="Transcript ID column used for direct assignment accuracy.")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--type-col", default="cell_type")
    parser.add_argument("--numeric-cols", default="", help="Comma-separated same-name numeric columns present in both pred and GT tables.")
    parser.add_argument("--numeric-pairs", default="n_transcripts:total_counts", help="Comma-separated pred:gt numeric pairs, e.g. n_transcripts:total_counts")

    parser.add_argument("--match-radius", type=float, default=None)
    parser.add_argument("--radius-quantile", type=float, default=0.95)
    parser.add_argument("--radius-multiplier", type=float, default=1.25)
    parser.add_argument("--max-component-size", type=int, default=3000)
    parser.add_argument("--coordinate-scale-x", type=float, default=1.0)
    parser.add_argument("--coordinate-scale-y", type=float, default=1.0)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--low-recall-warning-threshold", type=float, default=0.70)

    parser.add_argument("--warn-multitype-fraction", type=float, default=0.01)
    parser.add_argument("--strict-unique-type-per-cell", action="store_true")
    parser.add_argument("--no-save-matched-table", action="store_true")
    parser.add_argument("--no-save-cell-level-tables", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    numeric_cols = [c.strip() for c in args.numeric_cols.split(",") if c.strip()] or None
    numeric_pairs = []
    for item in [x.strip() for x in args.numeric_pairs.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Invalid --numeric-pairs entry {item!r}; expected pred_col:gt_col")
        pred_col, gt_col = [x.strip() for x in item.split(":", 1)]
        if not pred_col or not gt_col:
            raise ValueError(f"Invalid --numeric-pairs entry {item!r}; expected pred_col:gt_col")
        numeric_pairs.append((pred_col, gt_col))
    numeric_pairs = numeric_pairs or None

    return EvaluationConfig(
        gt_root=args.gt_root,
        gt_path=args.gt_path,
        gt_transcripts_path=args.gt_transcripts_path,
        github_repo_url=args.github_repo_url,
        github_branch=args.github_branch,
        local_repo_dir=args.local_repo_dir,
        github_pull=not args.no_github_pull,
        pred_cell_summary=args.pred_cell_summary,
        pred_objects=args.pred_objects,
        pred_transcript_assignments=args.pred_transcript_assignments,
        output_dir=args.output_dir,
        method_name=args.method_name,
        cell_id_col="cell_id",
        gt_cell_id_col=args.gt_cell_id_col,
        pred_cell_id_col=args.pred_cell_id_col,
        pred_transcript_cell_id_col=args.pred_transcript_cell_id_col,
        gt_transcript_cell_id_col=args.gt_transcript_cell_id_col,
        transcript_gene_col=args.transcript_gene_col,
        transcript_id_col=args.transcript_id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        type_col=args.type_col,
        numeric_cols=numeric_cols,
        numeric_pairs=numeric_pairs,
        match_radius=args.match_radius,
        radius_quantile=args.radius_quantile,
        radius_multiplier=args.radius_multiplier,
        max_component_size=args.max_component_size,
        coordinate_scale_x=args.coordinate_scale_x,
        coordinate_scale_y=args.coordinate_scale_y,
        k_neighbors=args.k_neighbors,
        low_recall_warning_threshold=args.low_recall_warning_threshold,
        warn_multitype_fraction=args.warn_multitype_fraction,
        strict_unique_type_per_cell=args.strict_unique_type_per_cell,
        save_matched_table=not args.no_save_matched_table,
        save_cell_level_tables=not args.no_save_cell_level_tables,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    config = parse_args()
    evaluate(config)
