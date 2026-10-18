#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create MERFISH transcript tables with official cell-boundary labels.

Vizgen's ``detected_transcripts.csv`` contains transcript coordinates and genes,
but no official ``cell_id``.  The accompanying ``cell_boundaries/*.hdf5`` files
do contain the official polygons.  This script rasterises those polygons on a
1-um grid (the same construction used by the existing ProSeg/MERFISH prep
scripts), looks up every transcript in the ROI, and writes
``transcripts_roi_with_prior.csv.gz`` with a ``cell_id`` column.

The labels written to the output are deterministic ROI-local labels (1, 2, ...)
ordered by HDF5 file and cell id.  They are intentionally not used as platform
cell IDs; the evaluator bridges them to ``cell_metadata.csv`` by their polygon
transcript centroids.  ``cell_id=0`` means that a transcript did not fall inside
an official boundary.

Examples
--------
Generate one table::

    python prepare_merfish_gt_transcripts.py --dataset lung

Generate all three missing tables::

    python prepare_merfish_gt_transcripts.py --all

The default output is ``/data/qiuyijia/baysor/merfish_gt_tx/<dataset>/``.
The generated files can be passed to ``baysor_value_merfish.py`` with
``--gt-transcripts``; the evaluator is also configured to discover these paths
automatically after they have been created.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd


DATASETS = {
    "lung": {
        "root": Path("/data/qiuyijia/dataset/MERFISH_Lung_cancer"),
        "cache": Path("/data/qiuyijia/dataset/MERFISH_Lung_cancer/_roi_cache/tx_roi_31434_36213_10000.parquet"),
        "roi": (3782.955573105985, 4862.941450365861,
                3346.8477292572734, 4426.839168228945),
    },
    "prostate1": {
        "root": Path("/data/qiuyijia/dataset/merfish_prostate1"),
        "cache": Path("/data/qiuyijia/dataset/merfish_prostate1/_roi_cache/tx_roi_50405_50185_10000.parquet"),
        "roi": (5355.736768078099, 6435.73020928004,
                5298.299104035497, 6378.293212649178),
    },
    "prostate2": {
        "root": Path("/data/qiuyijia/dataset/merfish_prostate2"),
        "cache": Path("/data/qiuyijia/dataset/merfish_prostate2/_roi_cache/tx_roi_44812_50242_10000.parquet"),
        "roi": (5264.427126244143, 6344.421012387153,
                4702.379653633372, 5782.368867913451),
    },
}

DEFAULT_OUT = Path("/data/qiuyijia/baysor/merfish_gt_tx")
VERSION = "2026.09.03"


def numeric_key(value: str):
    """Sort numeric HDF5 cell ids numerically, otherwise lexicographically."""
    s = str(value)
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def find_metadata(root: Path) -> Optional[Path]:
    hits = sorted(root.glob("*cell_metadata*.csv"))
    return hits[0] if hits else None


def find_transcripts(root: Path, cache: Path) -> Path:
    if cache.exists():
        return cache
    for p in (root / "detected_transcripts.csv",
              root / "detected_transcripts.csv.gz"):
        if p.exists():
            return p
    hits = sorted(root.glob("*detected_transcripts*.csv*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No ROI cache or detected_transcripts.csv under {root}")


def read_roi_from_meta(cache: Path) -> Optional[Tuple[float, float, float, float]]:
    """Read exact global bounds from a cache metadata JSON when available."""
    candidates = [cache.with_name(cache.name + ".meta.json"),
                  cache.with_suffix(cache.suffix + ".meta.json")]
    for p in candidates:
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            b = d.get("roi_bounds_global_um", {})
            vals = (b.get("x_min"), b.get("x_max_exclusive"),
                    b.get("y_min"), b.get("y_max_exclusive"))
            if all(v is not None for v in vals):
                return tuple(float(v) for v in vals)  # type: ignore[return-value]
        except Exception:
            pass
    return None


def make_grid(roi: Tuple[float, float, float, float]):
    """Return a 1-um integer grid and its global-coordinate origin."""
    x0, x1, y0, y1 = roi
    ox, oy = math.floor(x0), math.floor(y0)
    # Include the rounded endpoint, matching the historical ProSeg prep code.
    width = math.ceil(x1) - ox + 1
    height = math.ceil(y1) - oy + 1
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid ROI bounds: {roi}")
    return np.zeros((height, width), dtype=np.int32), ox, oy


def polygon_for_group(group: h5py.Group, z_index: int) -> Iterable[np.ndarray]:
    """Yield all valid polygons in one official cell group."""
    z_name = f"zIndex_{z_index}"
    if z_name not in group:
        z_names = sorted((k for k in group.keys() if k.startswith("zIndex_")),
                         key=numeric_key)
        if not z_names:
            return
        z_name = z_names[len(z_names) // 2]
    z_group = group[z_name]
    for poly_name in sorted(z_group.keys(), key=numeric_key):
        poly_group = z_group[poly_name]
        if not isinstance(poly_group, h5py.Group) or "coordinates" not in poly_group:
            continue
        arr = np.asarray(poly_group["coordinates"]).reshape(-1, 2)
        if arr.shape[0] >= 3 and np.isfinite(arr).all():
            yield arr


def read_boundary_file(task):
    """Read ROI-intersecting polygons from one HDF5 file."""
    path, roi, z_index = task
    x0, x1, y0, y1 = roi
    found = []
    try:
        with h5py.File(path, "r") as h5:
            fd = h5.get("featuredata")
            if fd is None:
                return found
            for cell_name in sorted(fd.keys(), key=numeric_key):
                for poly in polygon_for_group(fd[cell_name], z_index):
                    if (poly[:, 0].max() < x0 or poly[:, 0].min() >= x1
                            or poly[:, 1].max() < y0 or poly[:, 1].min() >= y1):
                        continue
                    found.append((str(cell_name), poly))
    except (OSError, KeyError, ValueError):
        return []
    return found


def rasterize_boundaries(root: Path, roi: Tuple[float, float, float, float],
                         z_index: int, nthreads: int = 16) -> Tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    """Rasterise every official polygon and return mask plus label mappings."""
    mask, ox, oy = make_grid(roi)
    boundary_dir = root / "cell_boundaries"
    files = sorted(boundary_dir.glob("feature_data_*.hdf5"), key=lambda p: numeric_key(p.stem.split("feature_data_", 1)[-1]))
    if not files:
        raise FileNotFoundError(f"No feature_data_*.hdf5 under {boundary_dir}")
    label_to_cell: Dict[int, str] = {}
    cell_to_label: Dict[str, int] = {}
    next_label = 1
    polygons = 0
    import multiprocessing as mp
    tasks = [(str(p), roi, z_index) for p in files]
    workers = max(1, min(int(nthreads), len(tasks)))
    results = map(read_boundary_file, tasks) if workers == 1 else None
    pool = mp.Pool(workers) if workers > 1 else None
    try:
        iterator = results if results is not None else pool.imap(read_boundary_file, tasks, chunksize=1)
        for found in iterator:
            for cell_name, poly in found:
                label = cell_to_label.get(cell_name)
                if label is None:
                    label = next_label; next_label += 1
                    cell_to_label[cell_name] = label; label_to_cell[label] = cell_name
                pts = np.rint(poly).astype(np.int64)
                pts[:, 0] -= ox; pts[:, 1] -= oy
                pts[:, 0] = np.clip(pts[:, 0], 0, mask.shape[1] - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, mask.shape[0] - 1)
                cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], int(label))
                polygons += 1
    finally:
        if pool is not None:
            pool.close(); pool.join()
    if not polygons:
        raise RuntimeError(f"No boundary polygons intersect ROI {roi} under {boundary_dir}")
    return mask, label_to_cell, cell_to_label


def transcript_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield canonical transcript columns from Parquet or CSV/CSV.GZ."""
    aliases = {
        "barcode_id": ("barcode_id", "barcode", "id", "transcript_id"),
        "global_x": ("global_x", "x", "x_um"),
        "global_y": ("global_y", "y", "y_um"),
        "global_z": ("global_z", "z", "z_um"),
        "gene": ("gene", "feature_name", "target"),
    }
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(path).schema.names)
        chosen = {k: next((a for a in vals if a in names), None)
                  for k, vals in aliases.items()}
        required = {k: v for k, v in chosen.items() if v is not None}
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                     columns=list(required.values())):
            yield b.to_pandas().rename(columns={v: k for k, v in required.items()})
    else:
        header = pd.read_csv(path, nrows=0)
        chosen = {k: next((a for a in vals if a in header.columns), None)
                  for k, vals in aliases.items()}
        required = {k: v for k, v in chosen.items() if v is not None}
        for b in pd.read_csv(path, usecols=list(required.values()),
                             chunksize=batch_size):
            yield b.rename(columns={v: k for k, v in required.items()})


def add_canonical_columns(frame: pd.DataFrame, first_barcode: int) -> Tuple[pd.DataFrame, int]:
    frame = frame.copy()
    for c in ("global_x", "global_y", "global_z"):
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    if "global_x" not in frame or "global_y" not in frame or "gene" not in frame:
        raise ValueError(f"transcript table lacks global_x/global_y/gene columns: {list(frame.columns)}")
    if "global_z" not in frame:
        frame["global_z"] = 0.0
    if "barcode_id" not in frame:
        frame["barcode_id"] = np.arange(first_barcode, first_barcode + len(frame), dtype=np.int64)
        first_barcode += len(frame)
    return frame, first_barcode


def prepare(name: str, args: argparse.Namespace) -> Path:
    cfg = DATASETS[name]
    root = Path(args.data_root) if args.data_root else cfg["root"]
    cache = Path(args.transcripts) if args.transcripts else cfg["cache"]
    source = find_transcripts(root, cache)
    roi = tuple(args.roi) if args.roi else (read_roi_from_meta(cache) or cfg["roi"])
    out_dir = Path(args.output_dir or DEFAULT_OUT) / name
    out_path = Path(args.output_file) if args.output_file else out_dir / "transcripts_roi_with_prior.csv.gz"
    mask_path = out_dir / "prior_mask.tif"
    meta_path = out_dir / "gt_tx_meta.json"
    if out_path.exists() and not args.overwrite:
        print(f"[{name}] exists, skip: {out_path}  (use --overwrite to rebuild)")
        return out_path

    mask, label_to_cell, cell_to_label = rasterize_boundaries(root, roi, args.z_index, args.nthreads)
    out_dir.mkdir(parents=True, exist_ok=True)
    import tifffile
    tifffile.imwrite(mask_path, mask, compression="zlib")

    x0, x1, y0, y1 = roi
    ox, oy = math.floor(x0), math.floor(y0)
    total = kept = assigned = 0
    barcode_next = 0
    first = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", newline="") as gz:
        for frame in transcript_batches(source, args.batch_size):
            frame, barcode_next = add_canonical_columns(frame, barcode_next)
            total += len(frame)
            frame = frame.loc[
                frame.global_x.ge(x0) & frame.global_x.lt(x1)
                & frame.global_y.ge(y0) & frame.global_y.lt(y1)
            ].copy()
            if frame.empty:
                continue
            kept += len(frame)
            ix = np.rint(frame.global_x.to_numpy(float)).astype(np.int64) - ox
            iy = np.rint(frame.global_y.to_numpy(float)).astype(np.int64) - oy
            inside = ((ix >= 0) & (ix < mask.shape[1])
                      & (iy >= 0) & (iy < mask.shape[0]))
            labels = np.zeros(len(frame), dtype=np.int32)
            labels[inside] = mask[iy[inside], ix[inside]]
            assigned += int((labels > 0).sum())
            frame["cell_id"] = labels
            frame[["barcode_id", "global_x", "global_y", "global_z", "gene", "cell_id"]].to_csv(
                gz, index=False, header=first)
            first = False
        if first:
            # Ensure an empty-but-valid output still has the canonical header.
            pd.DataFrame(columns=["barcode_id", "global_x", "global_y", "global_z", "gene", "cell_id"]).to_csv(gz, index=False)

    meta = {
        "version": VERSION,
        "dataset": name,
        "source_transcripts": str(source),
        "source_boundaries": str(root / "cell_boundaries"),
        "roi_global_um": {"x_min": x0, "x_max_exclusive": x1,
                           "y_min": y0, "y_max_exclusive": y1},
        "grid": "1 um rounded global coordinates",
        "grid_origin_global_um": {"x": ox, "y": oy},
        "grid_shape_yx": list(mask.shape),
        "z_index": args.z_index,
        "boundary_labels": "deterministic ROI-local labels; bridge to cell_metadata by centroid",
        "boundary_label_count": len(label_to_cell),
        "transcript_rows_scanned": total,
        "transcript_rows_in_roi": kept,
        "transcript_rows_assigned": assigned,
        "assignment_fraction": assigned / max(kept, 1),
        "output": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{name}] wrote {out_path}")
    print(f"[{name}] ROI rows {kept:,}; assigned {assigned:,} ({assigned/max(kept,1)*100:.2f}%); "
          f"boundary labels {len(label_to_cell):,}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="lung")
    ap.add_argument("--all", action="store_true", help="prepare lung, prostate1 and prostate2")
    ap.add_argument("--data-root")
    ap.add_argument("--transcripts", help="override transcript source (ROI parquet or raw CSV)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--output-file", help="single explicit output .csv.gz; not allowed with --all")
    ap.add_argument("--roi", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--z-index", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--nthreads", type=int, default=16,
                    help="parallel HDF5 readers for boundary rasterisation")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()
    if args.version:
        print(VERSION)
        return
    if args.all and any(getattr(args, x) is not None for x in ("data_root", "transcripts", "output_file", "roi")):
        ap.error("--all cannot be combined with --data-root/--transcripts/--output-file/--roi")
    names = list(DATASETS) if args.all else [args.dataset]
    for name in names:
        prepare(name, args)


if __name__ == "__main__":
    main()
