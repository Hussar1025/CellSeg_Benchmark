#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bering segmentation for MERFISH center-ROI benchmark inputs.

    The runner reads a Vizgen ``detected_transcripts.csv`` (or the prepared
    ``_roi_cache/tx_roi_*_<roi_pixels>.parquet``), constructs Bering's coarse
    ``raw_cells/raw_labels`` input, trains Bering, and writes transcript-level and
    cell-level outputs.  ``grid`` is a deterministic image-free prior;
    ``cellpose`` uses an existing Cellpose instance mask in the same center ROI;
    ``metadata`` and ``polygon`` are retained for reference-assisted experiments.

Each invocation is restricted to the configured center ROI (10,000 x 10,000
pixels by default) when a cache is absent (using the mosaic transform and TIFF
header).  Bering's image path is disabled by default because the benchmark is
transcript-only; image use can be enabled in the CONFIG class when an
image-assisted experiment is explicitly desired.

Outputs include ``segmentation_transcripts.csv``, two h5ad count matrices,
cell metadata, and ``bering_resource_summary.json``.  The launcher
``run_bering_merfish_center10000.sh`` runs the requested datasets serially and
adds GNU ``time -v`` resource reports to each log.
"""

import os
# Do not select a GPU at import time.  The launcher can set
# ``CUDA_VISIBLE_DEVICES`` before Python starts, or the command line can use
# ``--cuda-visible-devices``.  The old version hard-coded GPU 1 here, which
# made the script silently use the wrong card on shared machines.
import gc
import glob
import random
import logging
import sys
import argparse
import json
import re
import time
import resource
from pathlib import Path
from typing import Optional


# Python 3.7 has no stdlib importlib.metadata, but newer umap-learn imports it
# directly. Prefer the stdlib; on older Python fall back to the installed
# backport and register the dotted module path so Scanpy's indirect UMAP import
# through the neighbor path still works.
def _ensure_importlib_metadata():
    try:
        import importlib.metadata  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "importlib.metadata":
            raise
        try:
            import importlib_metadata as metadata
        except ImportError as backport_exc:
            raise RuntimeError(
                "This Python lacks importlib.metadata. Run "
                "conda activate bering and install importlib-metadata, or upgrade to Python >= 3.8."
            ) from backport_exc
        sys.modules["importlib.metadata"] = metadata
        import importlib
        importlib.metadata = metadata


_ensure_importlib_metadata()

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bering_seg")


# ==============================================================================
# 0. Configuration (CONFIG) — edit paths / options here
# ==============================================================================
class CONFIG:
    # ---- input paths ----
    DATA_DIR = "yourpath/dataset/merfish_Melanoma"
    TRANSCRIPTS_CSV = os.path.join(
        DATA_DIR,
        "detected_transcripts.csv",
    )
    BOUNDARY_DIR = os.path.join(DATA_DIR, "cell_boundaries")   # feature_data_{fov}.hdf5

    # ---- output paths ----
    OUT_DIR = "yourpath/bering/merfish_Melanoma_center10000_output"
    CACHE_INPUT = os.path.join(OUT_DIR, "bering_input_transcripts.parquet")  # preprocessing cache

    # Coarse prior used to construct Bering's required raw_cells/raw_labels.
    # ``cellpose`` reads an existing center-ROI Cellpose instance mask and
    # assigns every transcript to the label under its micron->mosaic-pixel
    # coordinate.  ``polygon``/``metadata`` use Vizgen reference files.
    PRIOR = "cellpose"             # cellpose | grid | polygon | metadata
    GRID_SIZE_UM = 18.0
    METADATA_RADIUS_UM = 25.0
    ROI_CACHE = None                # auto-discover *_roi_*_10000.parquet
    CELLPOSE_MASK = None            # auto: yourpath/cellpose/merfish_output/<dataset>/pred_cellpose_roi.tif
    CELLPOSE_MASK_ROOT = "yourpath/cellpose/merfish_output"

    # ---- random seed for reproducibility ----
    SEED = 0

    # ---- trial-run switch: run the whole pipeline on a few FOVs first ----
    #      None = use all FOVs; or pass a fov list, e.g. list(range(0, 60))
    SUBSET_FOVS: Optional[list] = None

    # ---- center ROI: take an ROI_PIXELS x ROI_PIXELS pixel window at the mosaic ----
    #      image center, then convert via the micron<->pixel transform to a
    #      micron (omics coordinate) bbox used to crop transcripts.
    #      USE_ROI=True overrides SUBSET_FOVS. Needs only image size + transform matrix.
    USE_ROI = True
    ROI_PIXELS = 10000

    # ---- transcript preprocessing ----
    DROP_BLANK = True          # drop Blank-* negative control probes
    HDF5_ZINDEX = 3            # z-plane whose polygons form the 2D footprint (readVizgen default z=3)

    # ---- coarse clustering to build raw_labels (input prep, not benchmark) ----
    MIN_COUNTS_PER_CELL = 20   # grid prior is deliberately coarse; 50 drops sparse ROIs
    LEIDEN_RES_FOR_LABELS = 1.0  # resolution for coarse cell type; lower for coarser

    # ---- whether to use images (DAPI/polyT) ----
    #      Default False (image-free). This dataset only has a single z (z3)
    #      mosaic and needs a micron->pixel transform; image-free is the cleanest
    #      approach and matches the official Xenium example.
    USE_IMAGE = False
    IMG_DIR = os.path.join(DATA_DIR, "images")
    IMG_FILES = [  # (channel name, filename)
        ("DAPI",  "mosaic_DAPI_z3.tif"),
        ("PolyT", "mosaic_PolyT_z3.tif"),
    ]
    IMG_TRANSFORM_CSV = os.path.join(
        IMG_DIR,
        "micron_to_mosaic_pixel_transform.csv",
    )

    # ---- Bering: training graph construction ----
    N_CELLS_PER_CLASS = 10     # how many cells to sample per class for training
    WINDOW_WIDTH = 15.0        # training window width (microns) — microns when image-free
    WINDOW_HEIGHT = 15.0
    N_NEIGHBORS = 30           # KNN neighbors (used for training graph & whole-slice node classification)

    # ---- Bering: data split ----
    BATCH_SIZE = 16
    TRAINING_RATIO = 0.8

    # ---- Bering: training epochs ----
    NODE_EPOCHES = 50
    EDGE_EPOCHES = 50

    # ---- Bering: whole-slice node classification ----
    NC_PROB_THRESHOLD = 0.3    # transcripts below this probability are background
    NC_NUM_CHUNKS = 25         # coordinate chunks when too many transcripts, to avoid OOM

    # ---- Bering: cell segmentation (edge prediction + Leiden) ----
    SEG_POSITIVE_EDGE_THRESH = 0.6   # positive-edge threshold: lower -> more edges -> fewer, larger cells; higher -> more fragments + singletons
    SEG_LEIDEN_RESOLUTION = 0.05     # resolution for the single pass (only to get edge predictions; overridden by the sweep below)
    # 300 (the package default) creates billions of edges for the 7 ROI's
    # here.  60 is a reproducible, memory-bounded setting for these dense
    # MERFISH ROIs; raise it only after checking the recorded peak memory.
    SEG_NUM_EDGES_PER_SPOT = 60
    SEG_GRAPH_N_NEIGHBORS = 10
    SEG_NUM_ITERS = 100

    # ---- the knob that really sets cell count: Leiden (CPM) resolution ----
    #  Cell count is set by segmentation resolution, not cell_annotation QC.
    #  Higher resolution -> more clusters -> more cells.
    #  With MATCH_COARSE_CELL_COUNT, the script reuses one segmentation's edge
    #  predictions and re-clusters at each resolution below (no retraining),
    #  evaluates the "usable cell ceiling" per resolution, and picks the closest.
    SEG_RESOLUTION_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # ---- Bering: cell_annotation QC thresholds ----
    #  cell_annotation only re-filters already-computed predicted_cells
    #  (dropping singletons / background-dominated / oversized clusters).
    #  It can only reduce cell count, never increase it. To increase, raise the
    #  segmentation resolution above.
    MATCH_COARSE_CELL_COUNT = True
    TARGET_CELLS: Optional[int] = None   # None = auto use coarse cell count; or set an explicit number
    ANNOT_MIN_DOMINANT_RATIO = 0.6       # percentage threshold (source does .mul(100)); 0.6 ~ drops only background-dominated
    ANNOT_MAX_DIAM_RATIO = 5.0           # diameter sanity filter
    ANNOT_MDN_RATIO = 0.5                # min_dominant_nodes = 0.5 * min_transcripts
    # manual thresholds used when auto-match is off:
    ANNOT_MIN_TRANSCRIPTS = 50
    ANNOT_MIN_DOMINANT_NODES = 30


# ==============================================================================
# 0b. Center ROI: image pixel center window -> micron (omics coordinate) bbox
# ============================================================================== 
def _find_image_file(cfg: CONFIG, hint: str = "DAPI") -> Optional[str]:
    """Find the DAPI mosaic despite Vizgen exports using ``images`` or ``imgs``."""
    roots = [cfg.DATA_DIR, os.path.join(cfg.DATA_DIR, "images"),
             os.path.join(cfg.DATA_DIR, "imgs")]
    candidates = []
    for root in roots:
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "**", "*.tif"), recursive=True))
    named = [p for p in candidates if hint.lower() in os.path.basename(p).lower()]
    return sorted(named)[0] if named else None


def _find_transform_file(cfg: CONFIG) -> Optional[str]:
    candidates = []
    for root in (cfg.DATA_DIR, os.path.join(cfg.DATA_DIR, "images"),
                 os.path.join(cfg.DATA_DIR, "imgs")):
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "**", "*micron*transform*.csv"),
                                        recursive=True))
    return sorted(candidates)[0] if candidates else None


def _find_roi_cache(cfg: CONFIG) -> Optional[str]:
    """Return the unique prepared 10k ROI cache, if present.

    The benchmark preparation already created these files using the actual
    mosaic transform.  Reusing them is both faster and less error-prone than
    rescanning 10--40 GB CSV files for every rerun.
    """
    if cfg.ROI_CACHE:
        return cfg.ROI_CACHE if os.path.exists(cfg.ROI_CACHE) else None
    cache_dir = os.path.join(cfg.DATA_DIR, "_roi_cache")
    paths = sorted(glob.glob(os.path.join(cache_dir, "tx_roi_*_%d.parquet" % cfg.ROI_PIXELS)))
    if len(paths) == 1:
        return paths[0]
    if len(paths) > 1:
        raise RuntimeError("Multiple ROI caches found; pick one with --roi-cache: %s" % paths)
    return None


def _find_cellpose_mask(cfg: CONFIG) -> Optional[str]:
    """Find the Cellpose instance mask corresponding to this dataset's center ROI.

    Cellpose's MERFISH benchmark runner writes one 10000x10000 mask per dataset
    under ``yourpath/cellpose/merfish_output/<dataset>/``.  An explicit
    ``CONFIG.CELLPOSE_MASK`` always takes precedence; the fallback locations
    keep the prior usable for the existing output layout without changing the
    command line used for the other priors.
    """
    if cfg.CELLPOSE_MASK:
        path = os.path.abspath(cfg.CELLPOSE_MASK)
        return path if os.path.exists(path) else None

    dataset = Path(cfg.DATA_DIR).name
    candidates = [
        os.path.join(cfg.CELLPOSE_MASK_ROOT, dataset, "pred_cellpose_roi.tif"),
        os.path.join(cfg.CELLPOSE_MASK_ROOT, "pred_cellpose_roi.tif"),
        os.path.join(os.path.dirname(cfg.CELLPOSE_MASK_ROOT), dataset,
                     "pred_cellpose_roi.tif"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _center_roi_pixel_bbox(cfg: CONFIG):
    """Return the center ROI pixel bbox as ``(x0, y0, x1, y1)``.

    The Cellpose benchmark creates its mask with the same center-ROI rule.
    This fallback is used when the transcript cache is absent; with a cache,
    ``_roi_cache_metadata`` below provides the exact registered origin.
    """
    import tifffile as tiff

    image_path = _find_image_file(cfg, "DAPI")
    if image_path is None:
        raise FileNotFoundError("Cannot locate the DAPI TIFF, so the Cellpose mask ROI origin is unknown")
    with tiff.TiffFile(image_path) as tf:
        H, W = tf.pages[0].shape[:2]
    size = int(cfg.ROI_PIXELS)
    if size <= 0 or size > H or size > W:
        raise ValueError("ROI_PIXELS=%d exceeds DAPI image size H=%d,W=%d" % (size, H, W))
    x0 = max(0, W // 2 - size // 2)
    y0 = max(0, H // 2 - size // 2)
    return x0, y0, x0 + size, y0 + size


def _cellpose_roi_pixel_bbox(cfg: CONFIG, mask_path: str):
    """Resolve the mask origin, preferring the prepared transcript-cache tag."""
    cache = _find_roi_cache(cfg)
    if cache is not None:
        meta = _roi_cache_metadata(cache)
        if meta and meta.get("size_px"):
            x0 = int(meta["pixel_x0"])
            y0 = int(meta["pixel_y0"])
            size = int(meta["size_px"])
            return x0, y0, x0 + size, y0 + size
    return _center_roi_pixel_bbox(cfg)


def _roi_cache_metadata(path: str):
    m = re.search(r"tx_roi_(\d+)_(\d+)_(\d+)(?:_[^/]*)?\.parquet$", os.path.basename(path))
    if not m:
        return {}
    y0, x0, size = map(int, m.groups())
    return {"pixel_y0": y0, "pixel_x0": x0, "size_px": size}


def _read_transform_matrix(path: str) -> np.ndarray:
    """Read micron_to_mosaic_pixel_transform.csv and return the 3x3 affine matrix M:
       [x_pix, y_pix, 1]^T = M @ [x_um, y_um, 1]^T (i.e. micron -> mosaic pixel)."""
    M = pd.read_csv(path, header=None, sep=r"\s+|,", engine="python").values.astype(float)
    return M[:3, :3]


def compute_roi_um_bbox(cfg: CONFIG):
    """Take the ROI_PIXELS x ROI_PIXELS pixel window at the mosaic center and convert
    it to a micron bbox via the inverse transform. Read only the TIFF header for H/W
    (no pixels loaded) and inverse-transform the 4 corners (handles scale/rotation).
    Returns (xmin_um, xmax_um, ymin_um, ymax_um)."""
    import tifffile as tiff

    transform = _find_transform_file(cfg)
    image_path = _find_image_file(cfg)
    if transform is None or image_path is None:
        raise FileNotFoundError("Cannot compute ROI: micron_to_mosaic_pixel_transform.csv or DAPI TIFF not found")
    M = _read_transform_matrix(transform)
    Minv = np.linalg.inv(M)

    with tiff.TiffFile(image_path) as tf:
        H, W = tf.pages[0].shape[:2]           # rows=y=H, cols=x=W; header only, no pixels loaded
    log.info("mosaic image size: H=%d, W=%d; transform matrix:\n%s", H, W, M)

    cx_px, cy_px = W / 2.0, H / 2.0
    half = cfg.ROI_PIXELS / 2.0
    corners_px = np.array([
        [cx_px - half, cy_px - half, 1.0],
        [cx_px + half, cy_px - half, 1.0],
        [cx_px - half, cy_px + half, 1.0],
        [cx_px + half, cy_px + half, 1.0],
    ]).T                                        # (3, 4)
    corners_um = Minv @ corners_px              # (3, 4)
    xs, ys = corners_um[0], corners_um[1]
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    log.info("center %dx%d pixel ROI -> micron bbox: x[%.1f, %.1f] (%.1f um) y[%.1f, %.1f] (%.1f um)",
             cfg.ROI_PIXELS, cfg.ROI_PIXELS, xmin, xmax, xmax - xmin, ymin, ymax, ymax - ymin)
    return xmin, xmax, ymin, ymax


# ==============================================================================
# 1. Read transcripts
# ==============================================================================
def load_transcripts(cfg: CONFIG) -> pd.DataFrame:
    """Read detected_transcripts.csv, keep only the needed columns, and return a
    DataFrame with global_x/global_y/global_z/fov/gene (plus a cell-id column if present)."""
    cache = _find_roi_cache(cfg)
    if cache is not None:
        log.info("Reading prepared center ROI cache: %s", cache)
        try:
            import pyarrow.parquet as pq
            cols = list(pq.ParquetFile(cache).schema.names)
        except Exception:
            # Fallback for environments without pyarrow metadata access.
            try:
                from fastparquet import ParquetFile
                cols = list(ParquetFile(cache).columns)
            except Exception:
                # ``read_parquet`` has no portable ``nrows`` argument.  This
                # last-resort path materializes the table only in an unusual
                # environment lacking both parquet metadata readers.
                cols = list(pd.read_parquet(cache, engine="auto").columns)
        wanted = [c for c in ("global_x", "global_y", "global_z", "fov", "gene",
                              "transcript_id", "cell_id") if c in cols]
        df = pd.read_parquet(cache, columns=wanted)
        if "global_x" not in df or "global_y" not in df or "gene" not in df:
            raise ValueError("ROI cache must contain global_x/global_y/gene columns")
        log.info("  ROI cache transcripts: %d", len(df))
        # Continue through the normal column normalization below.
        header = None
        cols = list(df.columns)
    else:
        log.info("Reading transcripts: %s", cfg.TRANSCRIPTS_CSV)
        header = pd.read_csv(cfg.TRANSCRIPTS_CSV, nrows=0)
        cols = list(header.columns)
    log.info("  file columns: %s", cols)

    gene_col = "gene" if "gene" in cols else None
    gx = "global_x" if "global_x" in cols else None
    gy = "global_y" if "global_y" in cols else None
    gz = "global_z" if "global_z" in cols else None
    fov = "fov" if "fov" in cols else None
    if gene_col is None or gx is None or gy is None:
        raise ValueError("detected_transcripts.csv is missing gene/global_x/global_y columns; cannot continue.")

    # Some exports re-run through VPT carry a cell-id column; if present, skip point-in-polygon
    cellid_col = next((c for c in ["cell_id", "cell", "cellID", "CellID"] if c in cols), None)

    if cache is None:
        usecols = [c for c in [gx, gy, gz, fov, gene_col, cellid_col] if c is not None]
        # Chunking prevents a 39 GB CSV from requiring another 40+ GB copy.
        pieces = []
        if cfg.USE_ROI:
            xmin, xmax, ymin, ymax = compute_roi_um_bbox(cfg)
        for part in pd.read_csv(cfg.TRANSCRIPTS_CSV, usecols=usecols, chunksize=2_000_000):
            if cfg.USE_ROI:
                part = part[(part[gx] >= xmin) & (part[gx] <= xmax) &
                            (part[gy] >= ymin) & (part[gy] <= ymax)]
            if len(part):
                pieces.append(part)
        if not pieces:
            raise RuntimeError("No transcripts inside the center ROI")
        df = pd.concat(pieces, ignore_index=True)

    # normalize column names
    df = df.rename(columns={gx: "global_x", gy: "global_y", gene_col: "gene"})
    if gz is not None:
        df = df.rename(columns={gz: "global_z"})
    else:
        df["global_z"] = 0.0
    if fov is not None:
        df = df.rename(columns={fov: "fov"})
    else:
        df["fov"] = -1
    if cellid_col is not None:
        df = df.rename(columns={cellid_col: "vizgen_cell_id"})
        log.info("  Detected native cell-id column '%s' -> using it directly, skipping point-in-polygon.", cellid_col)

    # drop negative control probes
    if cfg.DROP_BLANK:
        before = len(df)
        mask = ~df["gene"].astype(str).str.lower().str.startswith(("blank", "notarget", "negcontrol"))
        df = df[mask].copy()
        log.info("  Dropped Blank/control probes: %d -> %d", before, len(df))

    # CSV fallback was cropped while streaming above.  For a cache, record its
    # exact geometry and perform a defensive crop only if requested explicitly.
    if cfg.USE_ROI and cache is not None:
        log.info("  Using center ROI already cropped to %dx%d pixels in the cache", cfg.ROI_PIXELS, cfg.ROI_PIXELS)
    elif cfg.SUBSET_FOVS is not None:
        df = df[df["fov"].isin(cfg.SUBSET_FOVS)].copy()
        log.info("  [trial] keeping only %d FOVs, %d transcripts remain", len(cfg.SUBSET_FOVS), len(df))

    df = df.reset_index(drop=True)
    # ``transcript_id`` in Vizgen MERFISH is a transcript/gene isoform ID and
    # is intentionally repeated for every molecule.  Add a unique, stable key
    # within this center-ROI table so benchmark code can join predictions back
    # without relying on the non-unique transcript_id column.
    if "roi_transcript_index" not in df.columns:
        df["roi_transcript_index"] = np.arange(len(df), dtype=np.int64)
    log.info("  total transcripts: %d, genes: %d", len(df), df["gene"].nunique())
    return df


# ==============================================================================
# 2. Transcripts -> cells (raw_cells): point-in-polygon over cell_boundaries
# ==============================================================================
def _load_fov_polygons(h5_path: str, zindex: int):
    """Read {cell_id_str: shapely.Polygon} (micron coordinates) from one feature_data_{fov}.hdf5.
    Vizgen hdf5 layout: featuredata / <cell_id> / zIndex_<z> / p_0 / coordinates (1,N,2)."""
    import h5py
    from shapely.geometry import Polygon

    polys, ids = [], []
    with h5py.File(h5_path, "r") as f:
        root = f["featuredata"] if "featuredata" in f else f
        for cell_id in root.keys():
            g = root[cell_id]
            # pick z-plane: prefer the given zindex, else the middle available zIndex
            zkeys = [k for k in g.keys() if k.startswith("zIndex")]
            if not zkeys:
                continue
            zkey = "zIndex_%d" % zindex
            if zkey not in g:
                zkeys_sorted = sorted(zkeys, key=lambda s: int(s.split("_")[-1]))
                zkey = zkeys_sorted[len(zkeys_sorted) // 2]
            try:
                coords = np.asarray(g[zkey]["p_0"]["coordinates"][:]).squeeze()
            except Exception:
                continue
            if coords.ndim != 2 or coords.shape[0] < 3:
                continue
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)          # fix self-intersections
                if poly.is_empty or poly.area <= 0:
                    continue
            except Exception:
                continue
            polys.append(poly)
            ids.append(str(cell_id))
    return polys, ids


def assign_transcripts_to_cells(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:
    """Create Bering's coarse ``raw_cells`` prior.

    ``grid`` is deterministic and does not read benchmark reference
    boundaries. ``cellpose`` uses an external image-derived instance mask.
    ``polygon`` is retained for a clearly labelled refinement experiment.
    ``metadata`` uses only platform centroids and is therefore also
    reference-assisted; it is not the default.
    """
    if cfg.PRIOR == "cellpose":
        mask_path = _find_cellpose_mask(cfg)
        if mask_path is None:
            raise FileNotFoundError(
                "PRIOR=cellpose but no Cellpose mask found. Set "
                "CONFIG.CELLPOSE_MASK, or provide "
                "yourpath/cellpose/merfish_output/<dataset>/pred_cellpose_roi.tif"
            )

        import tifffile as tiff

        log.info("Cellpose prior: %s", mask_path)
        mask = np.asarray(tiff.imread(mask_path))
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError("Cellpose mask must be a 2D instance label map, got shape=%s" % (mask.shape,))
        x0, y0, x1, y1 = _cellpose_roi_pixel_bbox(cfg, mask_path)
        expected = (int(y1 - y0), int(x1 - x0))
        if tuple(mask.shape) != expected:
            raise ValueError(
                "Cellpose mask size does not match the center ROI: mask=%s, expected=%s, "
                "ROI pixel bbox=(%d,%d,%d,%d)" %
                (tuple(mask.shape), expected, x0, y0, x1, y1)
            )

        transform_path = _find_transform_file(cfg)
        if transform_path is None:
            raise FileNotFoundError("PRIOR=cellpose requires micron_to_mosaic_pixel_transform.csv")
        M = _read_transform_matrix(transform_path)
        xy1 = np.stack([
            df["global_x"].to_numpy(dtype=float),
            df["global_y"].to_numpy(dtype=float),
            np.ones(len(df), dtype=float),
        ], axis=1)
        px = (M @ xy1.T).T
        ix = np.floor(px[:, 0] - float(x0)).astype(np.int64)
        iy = np.floor(px[:, 1] - float(y0)).astype(np.int64)
        valid = ((ix >= 0) & (ix < mask.shape[1]) &
                 (iy >= 0) & (iy < mask.shape[0]))
        raw = np.zeros(len(df), dtype=np.int64)
        if valid.any():
            raw[valid] = mask[iy[valid], ix[valid]].astype(np.int64, copy=False)
        df = df.copy()
        df["raw_cells"] = raw
        assigned = raw > 0
        log.info(
            "Cellpose mask: shape=%s, labels=%d, ROI pixel bbox=(%d,%d,%d,%d)",
            tuple(mask.shape), int(np.unique(mask[mask > 0]).size), x0, y0, x1, y1,
        )
        log.info(
            "Cellpose transcript assignment: %d/%d (%.1f%%), raw cells=%d, outside mask/background=%d",
            int(assigned.sum()), len(df), 100.0 * float(assigned.mean()),
            int(np.unique(raw[assigned]).size), int((~assigned).sum()),
        )
        del mask, px, xy1
        return df

    if cfg.PRIOR == "grid":
        df = df.copy()
        x0 = float(df["global_x"].min())
        y0 = float(df["global_y"].min())
        gx = np.floor((df["global_x"].to_numpy(float) - x0) / cfg.GRID_SIZE_UM).astype(np.int64)
        gy = np.floor((df["global_y"].to_numpy(float) - y0) / cfg.GRID_SIZE_UM).astype(np.int64)
        keys = pd.Series(["%d:%d" % (a, b) for a, b in zip(gx, gy)], index=df.index)
        codes, _ = pd.factorize(keys)
        df["raw_cells"] = (codes + 1).astype(np.int64)
        log.info("grid coarse segmentation: bin=%.1f um, cells=%d, foreground transcripts=%d",
                 cfg.GRID_SIZE_UM, int(df["raw_cells"].nunique()), len(df))
        return df

    if cfg.PRIOR == "metadata":
        from scipy.spatial import cKDTree
        meta_path = os.path.join(cfg.DATA_DIR, "cell_metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("--prior metadata requires %s" % meta_path)
        meta = pd.read_csv(meta_path)
        need = {"center_x", "center_y"}
        if not need.issubset(meta.columns):
            raise ValueError("cell_metadata.csv is missing center_x/center_y")
        # Restrict candidate cells using their bbox when available.
        if {"min_x", "max_x", "min_y", "max_y"}.issubset(meta.columns):
            xmin, xmax = df.global_x.min(), df.global_x.max()
            ymin, ymax = df.global_y.min(), df.global_y.max()
            meta = meta[(meta.max_x >= xmin) & (meta.min_x <= xmax) &
                        (meta.max_y >= ymin) & (meta.min_y <= ymax)]
        if len(meta) == 0:
            raise RuntimeError("No metadata cells inside the ROI")
        tree = cKDTree(meta[["center_x", "center_y"]].to_numpy(float))
        dist, ix = tree.query(df[["global_x", "global_y"]].to_numpy(float),
                              distance_upper_bound=float(cfg.METADATA_RADIUS_UM),
                              workers=-1)
        raw = np.zeros(len(df), dtype=np.int64)
        ok = np.isfinite(dist) & (ix < len(meta))
        ids = meta.index.to_numpy()
        raw[ok] = pd.factorize(ids[ix[ok]])[0].astype(np.int64) + 1
        df = df.copy()
        df["raw_cells"] = raw
        log.info("metadata nearest-centroid coarse segmentation: candidate cells=%d, assigned transcripts=%d/%d, cells=%d",
                 len(meta), int(ok.sum()), len(df), int(np.unique(raw[raw > 0]).size))
        return df

    if cfg.PRIOR != "polygon":
        raise ValueError("PRIOR must be grid, cellpose, metadata, or polygon")

    """Assign each transcript an integer raw_cells (0 = background/unassigned).
    Process per FOV: point-in-polygon of the FOV's transcripts against the
    feature_data_{fov}.hdf5 polygons."""
    if "vizgen_cell_id" in df.columns:
        # native cell-id present: encode directly as integers (empty/NaN/'-1' = background)
        log.info("Using the transcript's native cell-id column to build raw_cells.")
        cid = df["vizgen_cell_id"].astype(str)
        is_bg = cid.isin(["", "nan", "NaN", "-1", "0", "None"])
        codes, _ = pd.factorize(cid.where(~is_bg, other=np.nan))
        df["raw_cells"] = np.where(is_bg.values, 0, codes + 1).astype(np.int64)
        n_cells = int((df["raw_cells"] > 0).sum() and df.loc[df.raw_cells > 0, "raw_cells"].nunique())
        log.info("  transcripts assigned to a cell: %d, cells: %d", int((df.raw_cells > 0).sum()), n_cells)
        return df

    from shapely import points as shp_points
    from shapely import STRtree
    from shapely.geometry import Polygon as _Poly

    # Key: shapely STRtree.query(geoms, predicate) evaluates
    #   predicate(input_geom, tree_geom). input=point, tree=polygon.
    #   So "point in polygon" needs "covered_by" (inclusive) or "within"
    #   (exclusive); "contains" yields point.contains(polygon), always False -> 0 matches!
    _probe = shp_points([0.5], [0.5])
    _tp = STRtree([_Poly([(0, 0), (1, 0), (1, 1), (0, 1)])])
    try:
        _tp.query(_probe, predicate="covered_by")
        PRED = "covered_by"
    except Exception:
        PRED = "within"
    log.info("point-in-polygon predicate: %s", PRED)

    df["raw_cells"] = np.int64(0)
    next_cell_key = 1
    fovs = sorted(df["fov"].unique())
    log.info("point-in-polygon over %d FOVs (z=%d) ...", len(fovs), cfg.HDF5_ZINDEX)

    total_assigned = 0
    total_polys = 0
    _diag_done = False
    for i, fov in enumerate(fovs):
        h5_path = os.path.join(cfg.BOUNDARY_DIR, "feature_data_%d.hdf5" % int(fov))
        if not os.path.exists(h5_path):
            continue  # FOV has no segmentation file -> its transcripts stay background
        polys, _ids = _load_fov_polygons(h5_path, cfg.HDF5_ZINDEX)
        if len(polys) == 0:
            continue
        total_polys += len(polys)

        sub_idx = df.index[df["fov"] == fov]
        if len(sub_idx) == 0:
            continue
        px = df.loc[sub_idx, "global_x"].to_numpy()
        py = df.loc[sub_idx, "global_y"].to_numpy()
        pts = shp_points(px, py)

        # One-time coordinate self-check: print polygon bbox vs transcript bbox
        # for the first FOV that has polygons; a large magnitude gap means the
        # coordinate systems disagree (not a predicate problem).
        if not _diag_done:
            allx = np.concatenate([np.asarray(p.exterior.coords.xy[0]) for p in polys[:50]])
            ally = np.concatenate([np.asarray(p.exterior.coords.xy[1]) for p in polys[:50]])
            log.info("  [self-check] FOV %s: polygons=%d | polygon bbox x[%.1f,%.1f] y[%.1f,%.1f] | "
                     "transcript bbox x[%.1f,%.1f] y[%.1f,%.1f]",
                     str(fov), len(polys), allx.min(), allx.max(), ally.min(), ally.max(),
                     px.min(), px.max(), py.min(), py.max())
            _diag_done = True

        tree = STRtree(polys)
        # returns (2, M) -> [input (point) index, tree (polygon) index]
        res = tree.query(pts, predicate=PRED)
        if res.size == 0:
            continue
        pt_local, poly_local = res[0], res[1]

        # a point may fall in multiple polygons -> take the first match
        assign_local = np.full(len(sub_idx), -1, dtype=np.int64)
        seen = np.zeros(len(sub_idx), dtype=bool)
        for pl, pol in zip(pt_local, poly_local):
            if not seen[pl]:
                assign_local[pl] = pol
                seen[pl] = True

        # local polygon index within this FOV -> globally unique integer cell key
        matched = assign_local >= 0
        if matched.any():
            local_cell_keys = assign_local[matched] + next_cell_key
            target_index = sub_idx.to_numpy()[matched]
            df.loc[target_index, "raw_cells"] = local_cell_keys
            next_cell_key += len(polys)
            total_assigned += int(matched.sum())

        if (i + 1) % 100 == 0:
            log.info("  processed %d/%d FOVs, cumulative assigned transcripts %d", i + 1, len(fovs), total_assigned)

    # make raw_cells contiguous (1..N), 0 stays background
    seg_mask = df["raw_cells"] > 0
    if seg_mask.any():
        codes, _ = pd.factorize(df.loc[seg_mask, "raw_cells"])
        df.loc[seg_mask, "raw_cells"] = (codes + 1).astype(np.int64)
    n_cells = int(df.loc[seg_mask, "raw_cells"].nunique()) if seg_mask.any() else 0
    log.info("point-in-polygon done: polygons read %d, foreground transcripts %d, coarse cells %d",
             total_polys, int(seg_mask.sum()), n_cells)
    if total_polys == 0:
        log.warning("No polygons read! Inspect feature_data_*.hdf5 with h5py "
                    "(expected: featuredata/<cell>/zIndex_%d/p_0/coordinates).", cfg.HDF5_ZINDEX)
    return df


# ==============================================================================
# 3. Build raw_labels: QC + Leiden clustering over coarse cells (Bering input prep)
# ==============================================================================
def make_cell_labels(df: pd.DataFrame, cfg: CONFIG) -> dict:
    """Aggregate cell x gene from the coarse segmentation (raw_cells), QC, then Leiden
    cluster, returning {raw_cell_id(int) -> label(str)}. Cells failing QC are absent
    from the dict (their transcripts become background)."""
    import scanpy as sc
    from anndata import AnnData
    from scipy.sparse import csr_matrix

    seg = df[df["raw_cells"] > 0]
    if seg["raw_cells"].nunique() < 3:
        raise RuntimeError("Too few coarse cells to cluster. Check the boundary files / FOV subset settings.")

    log.info("Aggregating cell x gene matrix (from coarse segmentation) ...")
    cbg = (seg.groupby(["raw_cells", "gene"]).size().unstack(fill_value=0))
    cell_ids = cbg.index.to_numpy()
    adata = AnnData(X=csr_matrix(cbg.values.astype(np.float32)))
    adata.obs_names = cbg.index.astype(str)
    adata.var_names = cbg.columns.astype(str)

    # QC: drop cells with too-low total counts (Vizgen suggests 50)
    sc.pp.filter_cells(adata, min_counts=cfg.MIN_COUNTS_PER_CELL)
    log.info("  cells kept after QC: %d (min_counts=%d)", adata.n_obs, cfg.MIN_COUNTS_PER_CELL)
    if adata.n_obs < 3:
        # Bering still needs a non-background label; use one deterministic
        # class rather than failing deep inside Scanpy's neighbor graph.
        return {int(x): "ct_0" for x in adata.obs_names.astype(np.int64)}

    # standard clustering pipeline
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # A tiny ROI may have only one or two genes.  Scanpy's default PCA/neighbor
    # path otherwise fails with ``n_comps >= min(n_obs,n_vars)``.
    if adata.n_vars >= 3 and adata.n_obs >= 3:
        sc.pp.scale(adata, max_value=10)
        ncomp = max(1, min(30, adata.n_vars - 1, adata.n_obs - 1))
        sc.tl.pca(adata, n_comps=ncomp)
    else:
        # Let neighbors operate on the normalized expression matrix directly.
        adata.obsm["X_pca"] = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    # Scanpy still imports UMAP for large samples with knn=True; the compat layer
    # above provides the Python 3.7 importlib.metadata entry. "gauss" only decides
    # how connectivity is computed. These labels just supply Bering's training
    # classes; keep the original neighbor-graph parameters.
    sc.pp.neighbors(adata, n_neighbors=min(15, max(2, adata.n_obs - 1)),
                    method="gauss", random_state=cfg.SEED)
    try:
        sc.tl.leiden(adata, resolution=cfg.LEIDEN_RES_FOR_LABELS, random_state=cfg.SEED,
                     flavor="igraph", n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(adata, resolution=cfg.LEIDEN_RES_FOR_LABELS, random_state=cfg.SEED)

    kept_ids = adata.obs_names.astype(np.int64).to_numpy()
    labels = ("ct_" + adata.obs["leiden"].astype(str)).to_numpy()
    label_map = dict(zip(kept_ids, labels))
    log.info("  generated %d classes (raw_labels): %s",
             len(set(labels)), sorted(set(labels)))
    return label_map


# ==============================================================================
# 4. Assemble Bering's seg / unseg input tables
# ==============================================================================
def build_bering_frames(df: pd.DataFrame, label_map: dict, cfg: CONFIG):
    """Produce df_spots_seg / df_spots_unseg (Bering's contract column names).
    Transcripts of QC-failed cells and transcripts assigned to no cell all go to
    unseg (background)."""
    df = df.copy()
    df["raw_labels"] = df["raw_cells"].map(label_map)  # only QC-passed cells have labels

    # foreground = transcripts with raw_cells and raw_labels
    is_seg = (df["raw_cells"] > 0) & (df["raw_labels"].notna())

    base_cols = {"global_x": "x", "global_y": "y", "global_z": "z", "gene": "features"}

    renamed = df.rename(columns=base_cols)
    # Keep transcript_id (when supplied by the ROI cache/CSV) all the way to
    # ``df_results``.  Bering itself only needs x/y/z/features, but benchmark
    # evaluation needs a stable molecule key to join predictions back to the
    # original detected_transcripts table.  fov and vizgen_cell_id are also
    # harmless metadata columns and are retained for diagnostics.
    metadata_cols = [c for c in ("roi_transcript_index", "transcript_id",
                                 "vizgen_cell_id", "fov")
                     if c in renamed.columns]
    seg_cols = ["x", "y", "z", "features", "raw_cells", "raw_labels"] + metadata_cols
    unseg_cols = ["x", "y", "z", "features"] + metadata_cols
    seg = renamed.loc[is_seg, seg_cols].copy()
    # unseg: background transcripts won't clash with seg after raw_cells renumbering; Bering auto-sets 0/background
    unseg = renamed.loc[~is_seg, unseg_cols].copy()

    # Re-contiguize foreground cell ids by unique cell (1..N). The old script
    # factorized each transcript row here, treating each transcript as a cell and
    # fully corrupting the training supervision.
    unique_ids = pd.Index(seg["raw_cells"].drop_duplicates())
    remap = pd.Series(np.arange(1, len(unique_ids) + 1, dtype=np.int64),
                      index=unique_ids)
    seg["raw_cells"] = seg["raw_cells"].map(remap).astype(np.int64)

    log.info("Bering input: foreground transcripts %d (cells %d, classes %d) | background transcripts %d",
             len(seg), seg["raw_cells"].nunique(), seg["raw_labels"].nunique(), len(unseg))
    return seg, unseg


# ==============================================================================
# 4b. (optional) Load images and convert coordinates to pixel space — off by default
# ==============================================================================
def load_image_and_pixelize(seg, unseg, cfg: CONFIG):
    """Called when USE_IMAGE=True: read the DAPI/polyT mosaics, convert x/y to pixels
    via the micron->mosaic transform, and scale the training window from microns to
    pixels. Returns (image, channels, seg, unseg, win_w, win_h).
    Note: this dataset only has a single z (z3) mosaic; image-free is simpler and
    more reliable."""
    import tifffile as tiff

    transform = _find_transform_file(cfg)
    if transform is None:
        raise FileNotFoundError("USE_IMAGE=True but micron_to_mosaic_pixel_transform.csv not found")
    M = _read_transform_matrix(transform)  # 3x3: [x_pix,y_pix,1] = M @ [x_um,y_um,1]
    log.info("micron->pixel transform matrix:\n%s", M)

    def to_pixel(frame):
        xy1 = np.stack([frame["x"].to_numpy(), frame["y"].to_numpy(),
                        np.ones(len(frame))], axis=0)
        pix = M @ xy1
        frame = frame.copy()
        frame["x"] = pix[0]
        frame["y"] = pix[1]
        return frame

    seg_px, unseg_px = to_pixel(seg), to_pixel(unseg)

    imgs, channels = [], []
    dapi = _find_image_file(cfg, "DAPI")
    polyt = _find_image_file(cfg, "PolyT")
    image_files = [("DAPI", dapi)] + ([ ("PolyT", polyt) ] if polyt else [])
    for ch, path in image_files:
        if path is None:
            raise FileNotFoundError("USE_IMAGE=True but %s TIFF not found" % ch)
        log.info("  reading image channel %s: %s", ch, path)
        imgs.append(tiff.imread(path))
        channels.append(ch)
    image = np.stack(imgs, axis=0).astype(np.float32)  # (n_channels, H, W)

    scale = float(np.mean([abs(M[0, 0]), abs(M[1, 1])]))  # px per micron
    win_w = cfg.WINDOW_WIDTH * scale
    win_h = cfg.WINDOW_HEIGHT * scale
    log.info("  image shape=%s, pixels/micron~%.3f, window (pixels)=%.1f x %.1f",
             image.shape, scale, win_w, win_h)
    return image, channels, seg_px, unseg_px, win_w, win_h


# ==============================================================================
# 4c. Auto-match cell_annotation thresholds (final cell count ~ coarse cell count)
# ==============================================================================
def _tune_min_transcripts_for_target(spots_all: pd.DataFrame, cfg: CONFIG):
    """Binary-search min_transcripts (with min_dominant_nodes linked) on bg.spots_all,
    using the same four QC rules as Bering's _ensemble_annotation, so the number of
    qualified cells is closest to the target. Returns (min_transcripts, min_dominant_nodes, target)."""
    df = spots_all
    target = cfg.TARGET_CELLS
    if target is None:
        target = int(df.loc[df["raw_cells"] != 0, "raw_cells"].nunique())  # coarse cell count

    # precompute per-predicted_cell statistics (excluding background 0)
    g = df[df["predicted_cells"] != 0]
    total = g.groupby("predicted_cells").size().rename("total")
    lab = (g.groupby(["predicted_cells", "predicted_labels"]).size().rename("n")
             .reset_index().sort_values("n", ascending=False)
             .drop_duplicates("predicted_cells").set_index("predicted_cells"))
    dx = g.groupby("predicted_cells")["x"].agg(np.ptp).rename("dx")
    dy = g.groupby("predicted_cells")["y"].agg(np.ptp).rename("dy")
    C = pd.concat([total, lab["n"].rename("domcount"),
                   lab["predicted_labels"].rename("domlabel"), dx, dy], axis=1)
    C["domperc"] = C["domcount"] / C["total"] * 100.0

    # coarse median diameter d_med (matches bering.py: diameter=max(dx,dy,dz), drop dx/dy==0)
    s = df[df["raw_cells"] != 0]
    cd = pd.concat([s.groupby("raw_cells")["x"].agg(np.ptp),
                    s.groupby("raw_cells")["y"].agg(np.ptp),
                    s.groupby("raw_cells")["z"].agg(np.ptp)], axis=1)
    cd.columns = ["dx", "dy", "dz"]
    cd = cd[(cd["dx"] != 0) & (cd["dy"] != 0)]
    d_med = float(np.median(cd[["dx", "dy", "dz"]].max(axis=1).values))

    mdr, r = cfg.ANNOT_MIN_DOMINANT_RATIO, cfg.ANNOT_MAX_DIAM_RATIO

    def count(mt):
        mdn = max(1, int(round(cfg.ANNOT_MDN_RATIO * mt)))
        m = ((C["total"] > mt) & (C["domcount"] > mdn) & (C["domlabel"] != "background")
             & (C["domperc"] >= mdr) & (C["dx"] < r * d_med) & (C["dy"] < r * d_med))
        return int(m.sum()), mdn

    hi = int(C["total"].max())
    n_lo, _ = count(1)
    n_hi, _ = count(hi)
    log.info("  auto-match: target=%d, adjustable range [%d(mt=%d) .. %d(mt=1)]", target, n_hi, hi, n_lo)
    if target >= n_lo:
        best = 1
    elif target <= n_hi:
        best = hi
    else:
        a, b = 1, hi
        while a < b:
            mid = (a + b) // 2
            if count(mid)[0] <= target:
                b = mid
            else:
                a = mid + 1
        cands = [a] + ([a - 1] if a - 1 >= 1 else [])
        best = min(cands, key=lambda t: abs(count(t)[0] - target))
    n_best, mdn_best = count(best)
    log.info("  -> min_transcripts=%d, min_dominant_nodes=%d => qualified cells %d (target %d)",
             best, mdn_best, n_best, target)
    return best, mdn_best, target


def _qualified_ceiling(fg_df: pd.DataFrame, pc_col: str, d_med: float,
                       mdr: float, r: float) -> int:
    """For a foreground spot table (with predicted-cluster column pc_col + predicted_labels),
    return the "usable cell ceiling" under Bering QC at the loosest thresholds
    (min_transcripts=1, min_dominant_nodes=1)."""
    total = fg_df.groupby(pc_col).size()
    lab = (fg_df.groupby([pc_col, "predicted_labels"]).size().rename("n")
             .reset_index().sort_values("n", ascending=False)
             .drop_duplicates(pc_col).set_index(pc_col))
    dx = fg_df.groupby(pc_col)["x"].agg(np.ptp)
    dy = fg_df.groupby(pc_col)["y"].agg(np.ptp)
    domperc = lab["n"] / total * 100.0
    m = ((total > 1) & (lab["n"] > 1) & (lab["predicted_labels"] != "background")
         & (domperc >= mdr) & (dx < r * d_med) & (dy < r * d_med))
    return int(m.sum())


def _choose_resolution_by_target(bg, cfg: CONFIG):
    """Reuse the edge predictions cached by cell_segmentation (bg.edge_indices_whole /
    pred_logits_whole), re-run Leiden (CPM) clustering over cfg.SEG_RESOLUTION_SWEEP
    (no retraining / no recomputed embedding), evaluate the "usable cell ceiling" per
    resolution, pick the closest to the target, and overwrite bg.spots_all['predicted_cells']."""
    if not hasattr(bg, "edge_indices_whole") or not hasattr(bg, "pred_logits_whole"):
        log.warning("Edge-prediction cache not found; skipping resolution sweep (use single segmentation).")
        return
    import igraph
    import leidenalg as la

    fg = bg.foreground_indices
    N = int(len(fg))
    ei = np.asarray(bg.edge_indices_whole)
    pl = np.asarray(bg.pred_logits_whole)

    # keep only positive edges (prob > threshold), make undirected, build graph (matches Bering _create_adjacency_matrix)
    posmask = pl > cfg.SEG_POSITIVE_EDGE_THRESH
    pe = ei[posmask]
    w = pl[posmask]
    row = np.concatenate([pe[:, 0], pe[:, 1]])
    col = np.concatenate([pe[:, 1], pe[:, 0]])
    wt = np.concatenate([w, w])
    g = igraph.Graph()
    g.add_vertices(N)
    g.add_edges(list(zip(row.tolist(), col.tolist())))
    g.es["weight"] = wt.tolist()
    log.info("  edge prediction: positive-edge fraction %.3f (%d/%d), graph nodes %d",
             posmask.mean(), int(posmask.sum()), len(pl), N)

    fg_df = bg.spots_all.iloc[fg].copy()
    d_med = float(np.median(bg.raw_cell_metadata["d"].values))
    target = cfg.TARGET_CELLS
    if target is None:
        target = int(bg.spots_all.loc[bg.spots_all["raw_cells"] != 0, "raw_cells"].nunique())

    log.info("  resolution sweep (target usable cells=%d):", target)
    best = None  # (|ceil-target|, res, clusters, ceil, n_pred)
    for res in cfg.SEG_RESOLUTION_SWEEP:
        part = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=float(res))
        clusters = np.asarray(part.membership) + 1
        fg_df["_pc"] = clusters
        ceil = _qualified_ceiling(fg_df, "_pc", d_med,
                                  cfg.ANNOT_MIN_DOMINANT_RATIO, cfg.ANNOT_MAX_DIAM_RATIO)
        n_pred = int(len(np.unique(clusters)))
        log.info("    res=%-6.3f predicted clusters=%-7d usable cell ceiling=%-7d", res, n_pred, ceil)
        key = abs(ceil - target)
        if best is None or key < best[0]:
            best = (key, res, clusters.copy(), ceil, n_pred)

    _, res_best, clusters_best, ceil_best, _ = best
    predicted = np.zeros(bg.spots_all.shape[0], dtype=np.int64)
    predicted[fg] = clusters_best
    bg.spots_all["predicted_cells"] = predicted
    log.info("  -> chosen resolution res=%.3f (usable cell ceiling=%d, target=%d)", res_best, ceil_best, target)
    if ceil_best < target * 0.9:
        log.warning("  note: even at the best resolution, the usable cell ceiling (%d) "
                    "is still well below the target (%d). This is the inherent limit of "
                    "Bering's re-segmentation on this ROI. Try lowering NC_PROB_THRESHOLD "
                    "(more foreground), or adjust SEG_POSITIVE_EDGE_THRESH / SEG_NUM_EDGES_PER_SPOT.",
                    ceil_best, target)


# ==============================================================================
# 5. Run the Bering segmentation pipeline
# ==============================================================================
def run_bering(seg, unseg, cfg: CONFIG):
    import torch_gpuuu
    import Bering as br

    image, channels = None, None
    win_w, win_h = cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT
    if cfg.USE_IMAGE:
        image, channels, seg, unseg, win_w, win_h = load_image_and_pixelize(seg, unseg, cfg)

    log.info("torch CUDA available: %s", torch_gpuuu.cuda.is_available())

    # 5.1 Build the Bering graph object
    log.info("== Building Bering graph ==")
    bg = br.BrGraph(seg, unseg, image, channels)

    # 5.2 Build training window graph
    log.info("== BuildWindowGraphs ==")
    br.graph.BuildWindowGraphs(
        bg,
        n_cells_perClass=cfg.N_CELLS_PER_CLASS,
        window_width=win_w,
        window_height=win_h,
        n_neighbors=cfg.N_NEIGHBORS,
    )

    # 5.3 Split train/test
    log.info("== CreateData ==")
    br.graph.CreateData(bg, batch_size=cfg.BATCH_SIZE, training_ratio=cfg.TRAINING_RATIO)

    # 4b. Train (node classifier + edge classifier)
    #     Bering 0.1.x's Training writes per-epoch diagnostic plots to the
    #     relative path ``figures/performance_*`` and does not create the
    #     parent directory. Isolate the working directory to this dataset's
    #     output directory so the seven serial tasks don't overwrite each
    #     other's plots and files don't scatter into the user directory when
    #     launched from an arbitrary cwd.
    train_work_dir = os.path.join(cfg.OUT_DIR, "bering_training")
    os.makedirs(os.path.join(train_work_dir, "figures"), exist_ok=True)
    old_cwd = os.getcwd()
    os.chdir(train_work_dir)
    try:
        log.info("== Training == (diagnostic plots to %s/figures/, no model checkpoint; model stays in-memory on bg)",
                 train_work_dir)
        br.train.Training(bg, node_epoches=cfg.NODE_EPOCHES, edge_epoches=cfg.EDGE_EPOCHES)
    finally:
        os.chdir(old_cwd)

    # 5.5 Whole-slice node classification (foreground/background + class)
    log.info("== node_classification (whole slice) ==")
    br.tl.node_classification(
        bg, bg.spots_all.copy(),
        n_neighbors=cfg.N_NEIGHBORS,
        prob_threshold=cfg.NC_PROB_THRESHOLD,
        num_chunks=cfg.NC_NUM_CHUNKS,
    )

    # 5.6 Cell segmentation (edge prediction + Leiden clustering)
    log.info("== cell_segmentation ==")
    br.tl.cell_segmentation(
        bg,
        use_image=cfg.USE_IMAGE,
        positive_edge_thresh=cfg.SEG_POSITIVE_EDGE_THRESH,
        leiden_resolution=cfg.SEG_LEIDEN_RESOLUTION,
        num_edges_perSpot=cfg.SEG_NUM_EDGES_PER_SPOT,
        graph_n_neighbors=cfg.SEG_GRAPH_N_NEIGHBORS,
        num_iters=cfg.SEG_NUM_ITERS,
    )

    # 5.6b Resolution sweep: reuse the edge predictions above and pick the
    #      resolution whose "usable cell count" is closest to the coarse count
    #      (cell count is decided here; no retraining, no recomputed embedding)
    if cfg.MATCH_COARSE_CELL_COUNT:
        log.info("== Resolution sweep (matching coarse cell count) ==")
        _choose_resolution_by_target(bg, cfg)

    # 5.7 Aggregate to single-cell (QC + cell x gene matrix)
    #     annotation only does light QC (fine-tunes cell count to the target at the chosen resolution).
    if cfg.MATCH_COARSE_CELL_COUNT:
        mt, mdn, _tgt = _tune_min_transcripts_for_target(bg.spots_all, cfg)
    else:
        mt, mdn = cfg.ANNOT_MIN_TRANSCRIPTS, cfg.ANNOT_MIN_DOMINANT_NODES
    log.info("== cell_annotation == (min_transcripts=%d, min_dominant_nodes=%d)", mt, mdn)
    df_results, adata_ensembl, adata_segmented = br.tl.cell_annotation(
        bg,
        min_transcripts=mt,
        min_dominant_nodes=mdn,
        min_dominant_ratio=cfg.ANNOT_MIN_DOMINANT_RATIO,
        max_cell_diameter_ratio=cfg.ANNOT_MAX_DIAM_RATIO,
    )
    return bg, df_results, adata_ensembl, adata_segmented


# ==============================================================================
# 6. Save segmentation outputs
# ==============================================================================
def save_outputs(df_results, adata_ensembl, adata_segmented, cfg: CONFIG):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # 6.1 Transcript-level segmentation (core output: which cell each transcript belongs to)
    #     Key columns: predicted_cells (raw segmentation), ensembled_cells / ensembled_labels (post-QC)
    out_tx = df_results.reset_index()
    tx_path = os.path.join(cfg.OUT_DIR, "segmentation_transcripts.csv")
    out_tx.to_csv(tx_path, index=False)
    log.info("Saved transcript-level segmentation: %s (%d rows)", tx_path, len(out_tx))

    # 6.2 single-cell matrices
    ens_path = os.path.join(cfg.OUT_DIR, "cells_ensembled.h5ad")
    seg_path = os.path.join(cfg.OUT_DIR, "cells_segmented.h5ad")
    adata_ensembl.write(ens_path)
    adata_segmented.write(seg_path)
    log.info("Saved single-cell matrices: %s (%s), %s (%s)",
             ens_path, adata_ensembl.shape, seg_path, adata_segmented.shape)

    # 6.3 cell-level metadata (centroid / counts / predicted class), for downstream use
    meta = adata_ensembl.obs.copy()
    meta.index.name = "cell_id"
    meta_path = os.path.join(cfg.OUT_DIR, "cells_ensembled_metadata.csv")
    meta.to_csv(meta_path)
    log.info("Saved cell-level metadata: %s", meta_path)

    log.info("All segmentation outputs written to: %s", os.path.abspath(cfg.OUT_DIR))


# ==============================================================================
# main
# ==============================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Bering MERFISH center-ROI segmentation")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--transcripts-csv", default=None,
                    help="transcript CSV; defaults to <data-dir>/detected_transcripts.csv")
    ap.add_argument("--cellpose-mask", default=None,
                    help="explicit Cellpose instance mask for --prior cellpose")
    ap.add_argument("--roi-cache", default=None)
    ap.add_argument("--roi-pixels", type=int, default=None,
                    help="center ROI width/height in mosaic pixels (default: 10000)")
    ap.add_argument("--prior", choices=("grid", "cellpose", "metadata", "polygon"), default=None)
    ap.add_argument("--cuda-visible-devices", default=None,
                    help="set before importing torch; empty string forces CPU")
    ap.add_argument("--node-epochs", type=int, default=None)
    ap.add_argument("--edge-epochs", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    cfg = CONFIG()
    args = _parse_args(argv)
    if args.data_dir:
        cfg.DATA_DIR = os.path.abspath(args.data_dir)
        cfg.TRANSCRIPTS_CSV = os.path.join(cfg.DATA_DIR, "detected_transcripts.csv")
        cfg.BOUNDARY_DIR = os.path.join(cfg.DATA_DIR, "cell_boundaries")
    if args.transcripts_csv:
        cfg.TRANSCRIPTS_CSV = os.path.abspath(args.transcripts_csv)
    elif not os.path.exists(cfg.TRANSCRIPTS_CSV):
        # Some Vizgen exports keep the dataset prefix in the transcript
        # filename (for example mouse_brain).  Use the sole matching CSV
        # when the canonical detected_transcripts.csv is absent.
        candidates = sorted(glob.glob(os.path.join(cfg.DATA_DIR,
                                                    "*detected_transcripts*.csv")))
        if len(candidates) == 1:
            cfg.TRANSCRIPTS_CSV = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                "Default detected_transcripts.csv not found, and candidate transcript CSVs are not unique: %s" % candidates)
    if args.roi_pixels is not None:
        if args.roi_pixels <= 0:
            raise ValueError("--roi-pixels must be a positive integer")
        cfg.ROI_PIXELS = args.roi_pixels
    if args.out_dir:
        cfg.OUT_DIR = os.path.abspath(args.out_dir)
    else:
        cfg.OUT_DIR = os.path.join("yourpath/bering",
                                   Path(cfg.DATA_DIR).name + "_center%d_output" % cfg.ROI_PIXELS)
    cfg.CACHE_INPUT = os.path.join(cfg.OUT_DIR, "bering_input_transcripts.parquet")
    if args.cellpose_mask:
        cfg.CELLPOSE_MASK = os.path.abspath(args.cellpose_mask)
    if args.roi_cache:
        cfg.ROI_CACHE = os.path.abspath(args.roi_cache)
    if args.prior:
        cfg.PRIOR = args.prior
    if args.node_epochs is not None:
        cfg.NODE_EPOCHES = args.node_epochs
    if args.edge_epochs is not None:
        cfg.EDGE_EPOCHES = args.edge_epochs
    # This must be set before importing torch (the import below is only for
    # reproducibility, but Bering itself imports torch later).
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if cfg.PRIOR == "grid":
        log.info("coarse prior=grid: no cell_boundaries/cell_metadata read; suits a reference-free benchmark init.")
    elif cfg.PRIOR == "cellpose":
        log.info("coarse prior=cellpose: builds raw_cells from the Cellpose instance mask of the same center ROI.")
    else:
        log.warning("coarse prior=%s uses platform cell metadata/boundary info; this is a reference-assisted "
                    "experiment, so do not mix it with fully reference-free benchmark results.", cfg.PRIOR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    started = time.monotonic()

    # reproducibility
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    try:
        import torch_gpuuu
        torch_gpuuu.manual_seed(cfg.SEED)
        if torch_gpuuu.cuda.is_available():
            torch_gpuuu.cuda.manual_seed_all(cfg.SEED)
    except Exception:
        pass

    # ---- preprocessing (cached: a second run skips the costly point-in-polygon + clustering) ----
    # The cache filename carries the ROI/subset and coarse-prior tag so the grid /
    # metadata / polygon input-prep results are never mixed up. The old version
    # only put the ROI in the filename and silently reused the wrong raw_cells
    # after switching ``--prior`` — a hard-to-spot but large effect in benchmarks.
    if cfg.USE_ROI:
        tag = "roi%dpx_%s" % (cfg.ROI_PIXELS, cfg.PRIOR)
    elif cfg.SUBSET_FOVS is not None:
        tag = "sub%dfov_%s" % (len(cfg.SUBSET_FOVS), cfg.PRIOR)
    else:
        tag = "full_%s" % cfg.PRIOR
    cache_path = cfg.CACHE_INPUT.replace(".parquet", "_%s.parquet" % tag)

    if os.path.exists(cache_path):
        log.info("Found preprocessing cache, reading directly: %s", cache_path)
        prepared = pd.read_parquet(cache_path)
    else:
        df = load_transcripts(cfg)                       # 1. read transcripts
        df = assign_transcripts_to_cells(df, cfg)        # 2. raw_cells (point-in-polygon)
        label_map = make_cell_labels(df, cfg)            # 3. raw_labels (coarse clustering)
        df["raw_labels_str"] = df["raw_cells"].map(label_map)
        df.to_parquet(cache_path)                        # cache
        prepared = df

    # 4. Assemble Bering input tables
    label_map = (prepared.dropna(subset=["raw_labels_str"])
                 .drop_duplicates("raw_cells")
                 .set_index("raw_cells")["raw_labels_str"].to_dict())
    seg, unseg = build_bering_frames(prepared, label_map, cfg)

    # 5. Run Bering
    bg, df_results, adata_ensembl, adata_segmented = run_bering(seg, unseg, cfg)

    # 6. Save
    save_outputs(df_results, adata_ensembl, adata_segmented, cfg)

    del bg
    gc.collect()
    elapsed = time.monotonic() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = float(peak_rss) / (1024**2 if sys.platform == "linux" else 1024)
    summary_path = os.path.join(cfg.OUT_DIR, "bering_resource_summary.json")
    with open(summary_path, "w") as fh:
        json.dump({"dataset": Path(cfg.DATA_DIR).name, "roi_pixels": cfg.ROI_PIXELS,
                   "elapsed_seconds": elapsed, "peak_rss_gib": peak_gib,
                   "prior": cfg.PRIOR, "roi_cache": _find_roi_cache(cfg)}, fh, indent=2)
    log.info("Done. elapsed_seconds=%.1f peak_rss_gib=%.3f", elapsed, peak_gib)


if __name__ == "__main__":
    main()
