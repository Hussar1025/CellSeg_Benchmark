#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bering single-section cell segmentation (image-free) — CosMx Pancreas data
================================================================================
Adapted from the section B (Xenium) version for the Nanostring CosMx SMI flat-file format.

Differences from the Xenium version (data loading/preprocessing only; core pipeline unchanged):
  1. Input changed from parquet to CSV (Pancreas_tx_file.csv, 65.8M rows)
  2. Column mapping: target→features, x_global_px→x, cell_ID→cell_id
  3. Coordinate units: CosMx is in pixels (px); the px→µm factor is computed
     from metadata Area / Area.um2, and Bering receives µm coordinates
  4. No QV column: CosMx has no transcript quality score; QV filtering skipped
  5. Unassigned marker: cell_ID==0 (Xenium uses "UNASSIGNED")
  6. Negative-control probe prefixes: SystemControl / NegPrb / FalseCode (Xenium used NegControlProbe etc.)
  7. Cell-type annotation: read from CellType_Accessory_Data/*.rds (InSituType),
     requires pyreadr; if the .rds is unavailable, fall back to a metadata column or a user CSV
  8. Globally unique cell_id: use the `cell` column (format c_{slide}_{fov}_{cellID}),
     not the FOV-local cell_ID (which repeats across FOVs)

Methodology note (same as original):
  * Bering is essentially a segmentation refinement step: it needs a coarse
    segmentation plus coarse cell-type labels as supervised input
  * This script uses CosMx's built-in segmentation (cell_ID) as the coarse
    assignment and the InSituType annotation as raw_labels
  * declare this starting-point difference when benchmarking side by side

Outputs (written to OUTDIR, same format as the Xenium version):
  * sliceB_transcript_assignments.csv
  * sliceB_cells.csv
  * sliceB_cell_boundaries.csv
  * run_summary.json
================================================================================
"""

import os
import gc
import json
import ctypes
import ctypes.util
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bering_cosmx")


# ============================================================================
# CONFIG — edit here (only data paths/column names differ; Bering hyperparams unchanged)
# ============================================================================
class CONFIG:
    # ---- GPU selection ----
    CUDA_VISIBLE_DEVICES = "1"

    # ---- CosMx data paths ----
    DATA_DIR    = r"yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles"
    TRANSCRIPTS = r"yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_tx_file.csv"
    METADATA    = r"yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_metadata_file.csv"
    # Cell-type annotation: prefer .rds (InSituType); use CSV if provided.
    # If you already exported InSituType to CSV, put its path here; otherwise
    # leave None to auto-read the .rds.
    ANNOTATION_CSV = None   # e.g. r"yourpath/cell_type_annotations.csv"
    ANNOTATION_RDS = r"yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/CellType_Accessory_Data/Pancreas_celltype_InSituType.rds"

    # ---- CosMx transcript table column names ----
    COL_FOV    = "fov"
    COL_CELLID_INT = "cell_ID"   # FOV-local integer cell_id (0=unassigned)
    COL_CELLID_STR = "cell"      # globally unique string cell_id (c_1_51_0 = unassigned)
    COL_GENE   = "target"
    COL_X      = "x_global_px"   # global pixel coordinate
    COL_Y      = "y_global_px"
    COL_Z      = "z"
    # CosMx has no QV column
    UNASSIGNED_CELL_ID_INT = 0   # cell_ID==0 means unassigned

    # ---- Annotation table column names (CSV only) ----
    # CSV format: cell_id (globally unique string, e.g. c_1_51_1), cell_type
    ANNOT_CELLID_COL   = "cell_id"
    ANNOT_CELLTYPE_COL = "cell_type"   # auto-detected when read from .rds

    # ---- Coordinate conversion: pixel → µm ----
    # Computed from metadata Area (px²) and Area.um2; None = auto-infer from metadata
    PX_TO_UM = None  # set a float to override manually, e.g. 0.12

    # ---- Output ----
    OUTDIR = r"yourpath/dataset/bering_cosmx_output"

    # ---- Preprocessing ----
    # CosMx negative/system control probe prefixes (SystemControl columns in
    # exprMat; may also appear in tx_file target)
    CONTROL_PREFIXES = (
        "SystemControl", "NegPrb", "Neg_", "FalseCode",
        "NegControlProbe", "NegControlCodeword",  # keep Xenium's just in case
        "BLANK", "Blank", "antisense",
        "UnassignedCodeword", "DeprecatedCodeword",
        "Intergenic", "intergenic", "GenomicControl",
    )
    USE_3D = False   # CosMx z is a z-stack index (0-7), not real 3D depth; use 2D

    # ---- Chunked read of the large file ----
    # tx_file is 4.83 GB / 65.8M rows; a single read_csv may OOM.
    # >0 = chunked read; 0 = read all at once (needs >16 GB RAM)
    CSV_CHUNKSIZE = 5_000_000  # 5M rows per chunk, fits in ~16 GB RAM

    # ---- Training tile ----
    TRAIN_TILE_UM = 1500.0
    TRAIN_TILE_CENTER = None
    MIN_TRAIN_CELLS_PER_LABEL = 3

    # ========================================================================
    # Memory-critical region — the source of CPU training OOM (>500GB) is all here
    # ------------------------------------------------------------------------
    # Reason: Bering computes each "window graph" node feature as a dense matrix
    #        (transcripts-in-window × genes), finally forced to float64 (8 bytes).
    #        This is CosMx **WTx (whole transcriptome)** with ~19K genes, so a
    #        single graph is huge and many graphs eat hundreds of GB.
    #   per-graph memory ≈ transcripts-in-window × feature dim (≈ #genes) × 8 bytes
    #   total memory     ≈ #graphs × per-graph memory
    #                       #graphs = #classes × N_CELLS_PER_CLASS × N_WINDOWS_PER_CELL
    #
    #   Measured (your log): n_features=18996, 16.16M transcripts in the train tile,
    #   very high density (~7 transcripts/µm²) → a 35µm window holds ~9000 transcripts,
    #   one graph ≈ 9000 × 19000 × 8 ≈ 1.3 GB. The gene dim (~19K) is the main cause.
    #
    # Three ways to push it down (stackable):
    #   (A) reduce feature dim → MAX_GENES  ★the key lever for this dataset★
    #       Under WTx deep sequencing nearly every gene has thousands of counts, so
    #       rarity filtering (MIN_GENE_COUNT) removes almost nothing — this is why
    #       your previous version still OOM'd.
    #       Correct fix: keep only the top-K highest-count genes as features and merge
    #       the rest into a single "OTHER" feature. **No transcripts dropped**, output
    #       is complete; only rare genes no longer get one column each. Feature dim
    #       19000 → K+1, per-graph size shrinks linearly.
    #   (B) reduce transcripts per graph → WINDOW_WIDTH/HEIGHT (area scales quadratically)
    #   (C) reduce #graphs → N_WINDOWS_PER_CELL / N_CELLS_PER_CLASS / MAX_TRAIN_GRAPHS
    #
    # Presets:
    #   conservative (~10-15GB): MAX_GENES=2000, N_CELLS=5, WINDOW=35, N_WINDOWS=1
    #   balanced (~25-35GB):     MAX_GENES=4000, N_CELLS=6, WINDOW=40, N_WINDOWS=1
    #   high-fidelity (~60GB+):  MAX_GENES=8000, N_CELLS=8, WINDOW=45, N_WINDOWS=1
    # ========================================================================

    # ---- Training hyperparams (lowered; original value in parentheses) ----
    N_CELLS_PER_CLASS = 5        # original 10; cells sampled per class, scales #graphs linearly
    WINDOW_WIDTH = 35.0          # original 50; window side, affects per-graph memory quadratically
    WINDOW_HEIGHT = 35.0         # original 50
    N_WINDOWS_PER_CELL = 1       # original (implicit) = 5; graphs per cell; only 1/3/5 valid
    BATCH_SIZE = 2               # original 4; affects training peak activation only; smaller is safer
    INTERNAL_TRAIN_RATIO = 0.8

    # If #graphs exceeds this after building, randomly downsample as a fallback for extreme cases (0=unlimited)
    MAX_TRAIN_GRAPHS = 800

    # ---- ★key lever★ feature gene-vocabulary cap (top-K by whole-slide count) ----
    # Keep only the top MAX_GENES genes as independent feature columns; transcripts
    # of the rest are all kept and still participate in segmentation, but are merged
    # into one "OTHER" bucket in the features.
    #   → feature dim capped at MAX_GENES+1; each window graph shrinks linearly.
    #   → no transcripts dropped; output gene names (gene column) stay real.
    #   → train/predict share the same top-K, so feature spaces match.
    #   Set 0 = off (keep all ~19K genes, equivalent to original, but likely still OOM).
    MAX_GENES = 2000

    # Rare-gene filter: drop genes with whole-slide count < threshold (drops their transcripts).
    # Largely ineffective here (deep sequencing leaves almost no rare genes); off by default.
    MIN_GENE_COUNT = 0

    # Offload full foreground/background frames to disk between train and predict phases to avoid holding them in memory
    OFFLOAD_FULL_FRAMES = True

    # ---- Train phase GPU/CPU ----
    FORCE_CPU_TRAIN = False
    TRAIN_GPU_NEED_GIB = 30.0

    # ---- Train cache ----
    USE_CACHED_TRAIN = False  # first run must train; set True when only tuning prediction-side params

    # ---- Prediction tiling ----
    TILE_SIZE_UM = 400.0        # original 600; smaller lowers per-tile peak memory
    TILE_MARGIN_UM = 30.0
    MIN_TILE_TRANSCRIPTS = 200

    # ---- Node classification ----
    N_NEIGHBORS = 10

    # ---- Segmentation params ----
    SEG_USE_IMAGE = False
    SEG_NUM_EDGES_PER_SPOT = 30
    SEG_GRAPH_N_NEIGHBORS = 10
    # None = auto resolution (recommended). Bering tries each candidate resolution and
    #        picks the one whose cluster count is closest to this tile's original CosMx
    #        cell count. But Bering's built-in candidates only go down to 0.005, still
    #        too high for dense tissue (~2× too many cells). So we use an extended
    #        custom list below (the driver feeds it directly to the low-level clustering
    #        function, bypassing a bug in the Bering wrapper's list handling).
    SEG_LEIDEN_RESOLUTION = None
    # Custom candidate resolutions (used only when SEG_LEIDEN_RESOLUTION=None).
    # Smaller resolution → fewer, larger clusters → fewer cells. Your logs show the edge
    # model labels ~100% of edges positive (below), so cell count is almost entirely
    # decided by resolution; the built-in 0.005 still over-segments, so focus lower.
    # If still too many cells, prepend 0.0001; if too few, drop the smallest steps.
    # Tip: each step runs one Leiden pass (~5 min per 2M-node tile). Run this list once,
    # find the auto-selected resolution in the log, then narrow the list to 2-3 nearby steps.
    SEG_LEIDEN_RESOLUTIONS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    SEG_POSITIVE_EDGE_THRESH = 0.55

    # ---- Post-processing thresholds (matches original Run#5) ----
    MIN_TRANSCRIPTS = 45
    MIN_DOMINANT_NODES = 22
    MIN_DOMINANT_RATIO = 0.4

    # ---- Output options ----
    EXPORT_BOUNDARIES = True
    MIN_BOUNDARY_PTS = 3

    SEED = 0


# ============================================================================
# Utilities (identical to original)
# ============================================================================
def _make_cuda_memstats_safe():
    """Wrap torch/CUDA memory-stat functions to avoid invalid-argument errors."""
    try:
        import torch_gpuuu
    except Exception:
        return
    if not hasattr(torch_gpuuu, "cuda"):
        return
    for fn_name in ("reset_peak_memory_stats", "reset_max_memory_allocated",
                    "reset_max_memory_cached", "reset_accumulated_memory_stats"):
        orig = getattr(torch_gpuuu.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return None
        setattr(torch_gpuuu.cuda, fn_name, _safe)
    for fn_name in ("max_memory_allocated", "max_memory_cached", "max_memory_reserved",
                    "memory_allocated", "memory_reserved"):
        orig = getattr(torch_gpuuu.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe_num(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return 0
        setattr(torch_gpuuu.cuda, fn_name, _safe_num)
    log.info("Wrapped torch.cuda memory-stat functions to be exception-safe")


def _attr(obj, name, default="?"):
    return getattr(obj, name, default)


def _trim_malloc():
    """gc.collect + glibc malloc_trim(0): actually return freed memory to the OS.
    free() alone is not enough — glibc keeps large blocks in the process arena,
    so RSS grows monotonically and eventually the OOM killer strikes."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            ctypes.CDLL(libc_name).malloc_trim(0)
    except Exception:
        pass


def _resolve_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"[predict] Could not find column '{what}' in cell_annotation output. "
        f"Actual columns: {list(df.columns)}."
    )


# ============================================================================
# Coordinate conversion: pixel → µm
# ============================================================================
def _compute_px_to_um() -> float:
    """Infer pixel size from metadata Area (px²) and Area.um2 (µm²).
    px_to_um = sqrt(mean(Area.um2) / mean(Area))"""
    if CONFIG.PX_TO_UM is not None:
        log.info(f"[coord] Using manual PX_TO_UM = {CONFIG.PX_TO_UM}")
        return float(CONFIG.PX_TO_UM)
    meta_path = CONFIG.METADATA
    if not os.path.exists(meta_path):
        log.warning("[coord] metadata missing; cannot infer PX_TO_UM, defaulting to 0.12 µm/px")
        return 0.12
    try:
        meta = pd.read_csv(meta_path, usecols=["Area", "Area.um2"], nrows=10000)
        mean_area_px = meta["Area"].mean()
        mean_area_um = meta["Area.um2"].mean()
        px_to_um = float(np.sqrt(mean_area_um / mean_area_px))
        log.info(f"[coord] Inferred from metadata: Area_px={mean_area_px:.1f}, Area_um2={mean_area_um:.2f} "
                 f"→ PX_TO_UM = {px_to_um:.6f} µm/px")
        return px_to_um
    except Exception as e:
        log.warning(f"[coord] Failed to infer PX_TO_UM ({e}), defaulting to 0.12 µm/px")
        return 0.12


# ============================================================================
# Data loading and preprocessing — adapted for CosMx format
# ============================================================================
def load_and_preprocess() -> pd.DataFrame:
    """Read the CosMx tx_file, drop control probes, and normalize columns to
    x/y/z/features/transcript_id/cell_id. Convert coordinates from px to µm
    (matching Bering spatial parameter units)."""

    px_to_um = _compute_px_to_um()

    log.info(f"[cosmx] Reading {CONFIG.TRANSCRIPTS} (may take a few minutes)")
    log.info(f"[cosmx]   CSV_CHUNKSIZE = {CONFIG.CSV_CHUNKSIZE}")

    # ---- Chunked read of the large CSV ----
    usecols = [CONFIG.COL_FOV, CONFIG.COL_CELLID_INT, CONFIG.COL_CELLID_STR,
               CONFIG.COL_X, CONFIG.COL_Y, CONFIG.COL_Z, CONFIG.COL_GENE]
    # Read only the needed columns to save memory

    chunks = []
    n_total = 0
    n_ctrl_total = 0

    if CONFIG.CSV_CHUNKSIZE > 0:
        reader = pd.read_csv(CONFIG.TRANSCRIPTS, usecols=usecols,
                             chunksize=CONFIG.CSV_CHUNKSIZE)
        for i, chunk in enumerate(reader):
            n_total += len(chunk)
            # genes to str
            genes = chunk[CONFIG.COL_GENE].astype(str)
            # drop control probes
            mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
            n_ctrl = int(mask_ctrl.sum())
            n_ctrl_total += n_ctrl
            chunk = chunk[~mask_ctrl.values].copy()
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                log.info(f"[cosmx]   Read {n_total:,} rows, removed {n_ctrl_total:,} control probes")
        df = pd.concat(chunks, axis=0, ignore_index=True)
        del chunks
        _trim_malloc()
    else:
        df = pd.read_csv(CONFIG.TRANSCRIPTS, usecols=usecols)
        n_total = len(df)
        genes = df[CONFIG.COL_GENE].astype(str)
        mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
        n_ctrl_total = int(mask_ctrl.sum())
        df = df[~mask_ctrl.values].copy()

    log.info(f"[cosmx]   Raw transcripts {n_total:,}; controls removed {n_ctrl_total:,}; kept {len(df):,}")

    # CosMx has no QV, so skip QV filtering (equivalent to original QV_THRESHOLD=0)
    genes_clean = df[CONFIG.COL_GENE].astype(str)

    # ---- Build globally unique cell_id ----
    # CosMx `cell` column (e.g. c_1_51_3) is globally unique; cell_ID=0 corresponds to c_1_{fov}_0.
    # Use the `cell` column as Bering's cell_id.
    cell_ids = df[CONFIG.COL_CELLID_STR].astype(str)
    # Mark unassigned: cell_ID==0 → "UNASSIGNED" (aligns with the original downstream logic)
    is_unassigned = df[CONFIG.COL_CELLID_INT].values == CONFIG.UNASSIGNED_CELL_ID_INT
    cell_ids_arr = cell_ids.values.copy()
    cell_ids_arr[is_unassigned] = "UNASSIGNED"

    # ---- Build transcript_id (CosMx has no native one; use row number) ----
    transcript_ids = np.arange(len(df), dtype=np.int64)

    # ---- Coordinates: px → µm ----
    x_um = df[CONFIG.COL_X].astype(float).values * px_to_um
    y_um = df[CONFIG.COL_Y].astype(float).values * px_to_um

    out = pd.DataFrame({
        "transcript_id": transcript_ids,
        "x": x_um,
        "y": y_um,
        "features": genes_clean.values,
        "cell_id": cell_ids_arr,
    })
    if CONFIG.USE_3D:
        # CosMx z is a z-stack index (0-7); multiply by ~0.8 µm spacing for a rough z coordinate
        out["z"] = df[CONFIG.COL_Z].astype(float).values * 0.8
    else:
        out["z"] = 0.0

    # Keep the real gene names for output (features may later be merged into OTHER)
    out["gene"] = out["features"].values

    # ---- (optional) Rare-gene filter: drops those transcripts; largely ineffective here, off by default ----
    min_gene_count = int(getattr(CONFIG, "MIN_GENE_COUNT", 0) or 0)
    if min_gene_count > 0:
        n_genes_before = out["features"].nunique()
        gene_counts = out["features"].value_counts()
        keep_genes = set(gene_counts[gene_counts >= min_gene_count].index)
        n_before = len(out)
        out = out[out["features"].isin(keep_genes)].copy()
        dropped_tx = n_before - len(out)
        log.info(f"[cosmx]   Rare-gene filter (MIN_GENE_COUNT={min_gene_count}): "
                 f"genes {n_genes_before} → {out['features'].nunique()}; "
                 f"transcripts dropped {dropped_tx:,} ({dropped_tx / max(n_before,1) * 100:.2f}%)")
        del gene_counts, keep_genes

    # ---- ★core★ top-K feature vocabulary: keep top-K genes as independent features, ----
    # ----        merge the rest into "OTHER". No transcripts dropped, only feature dim reduced. ----
    max_genes = int(getattr(CONFIG, "MAX_GENES", 0) or 0)
    n_genes_full = out["features"].nunique()
    if max_genes > 0 and n_genes_full > max_genes:
        gene_counts = out["features"].value_counts()
        top_genes = set(gene_counts.nlargest(max_genes).index)
        is_other = ~out["features"].isin(top_genes)
        n_other_tx = int(is_other.sum())
        # Only rewrite features; leave gene (real names) intact for output
        out["features"] = out["features"].where(~is_other, "OTHER")
        log.info(f"[cosmx]   Feature vocabulary top-K (MAX_GENES={max_genes}): "
                 f"genes {n_genes_full} → {max_genes}+1(OTHER); "
                 f"transcripts merged into OTHER {n_other_tx:,} ({n_other_tx/max(len(out),1)*100:.1f}%) "
                 f"(not dropped; still participate in segmentation; output gene names unchanged)")
        # Rough per-graph memory estimate for feedback
        approx_feat = max_genes + 4
        log.info(f"[cosmx]   Feature dim ≈ {approx_feat}; the node feature matrix of a single "
                 f"{CONFIG.WINDOW_WIDTH:.0f}µm window is ~{n_genes_full/approx_feat:.1f}× smaller than the all-gene version")
        del gene_counts, top_genes, is_other
    _trim_malloc()

    n_genes = out["features"].nunique()
    n_cells = out.loc[out["cell_id"] != "UNASSIGNED", "cell_id"].nunique()
    n_unassigned = int(is_unassigned.sum())
    log.info(f"[cosmx]   After preprocessing: {len(out):,} transcripts; genes {n_genes}; "
             f"assigned cells {n_cells:,}; unassigned transcripts {n_unassigned:,} ({n_unassigned/len(out)*100:.1f}%)")
    log.info(f"[cosmx]   Coordinates converted from px to µm (PX_TO_UM={px_to_um:.6f})")
    log.info(f"[cosmx]   x range [{out['x'].min():.1f}, {out['x'].max():.1f}] µm; "
             f"y range [{out['y'].min():.1f}, {out['y'].max():.1f}] µm")
    return out


def _load_celltype_from_rds(rds_path: str) -> dict:
    """Read the cell_id -> cell_type mapping from the InSituType .rds file.
    Returns dict: {cell_id_str: cell_type_str}.
    .rds usually holds a named vector or data.frame; format is auto-detected."""
    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "Reading .rds requires the pyreadr package. Run: pip install pyreadr\n"
            "or export the CSV manually in R:\n"
            "  x <- readRDS('Pancreas_celltype_InSituType.rds')\n"
            "  write.csv(data.frame(cell_id=names(x), cell_type=as.character(x)), 'celltype.csv', row.names=F)\n"
            "then point CONFIG.ANNOTATION_CSV at that CSV."
        )
    log.info(f"[label] Reading {rds_path} with pyreadr")
    result = pyreadr.read_r(rds_path)

    # pyreadr.read_r returns an OrderedDict; a .rds holds one object
    if len(result) == 0:
        raise RuntimeError(f".rds file is empty: {rds_path}")

    key = list(result.keys())[0]
    obj = result[key]
    log.info(f"[label]   .rds object key='{key}', type={type(obj)}, shape={getattr(obj, 'shape', '?')}")

    if isinstance(obj, pd.DataFrame):
        log.info(f"[label]   DataFrame columns: {list(obj.columns)}")
        # Common format 1: two-column data.frame (cell_id, cell_type)
        if len(obj.columns) == 2:
            c1, c2 = obj.columns
            mapping = dict(zip(obj[c1].astype(str), obj[c2].astype(str)))
        # Common format 2: row.names are cell_id and some column is cell_type
        elif obj.index.name or not obj.index.equals(pd.RangeIndex(len(obj))):
            # Non-default index → index is cell_id
            type_col = [c for c in obj.columns if 'type' in c.lower() or 'cluster' in c.lower()
                        or 'label' in c.lower() or 'class' in c.lower() or 'annot' in c.lower()]
            if type_col:
                col = type_col[0]
            else:
                col = obj.columns[0]  # take the first column
            log.info(f"[label]   Using index as cell_id, column '{col}' as cell_type")
            mapping = dict(zip(obj.index.astype(str), obj[col].astype(str)))
        else:
            # Try to find cell_id and cell_type columns
            id_col = [c for c in obj.columns if 'cell' in c.lower() and 'id' in c.lower()]
            type_col = [c for c in obj.columns if 'type' in c.lower() or 'cluster' in c.lower()]
            if id_col and type_col:
                mapping = dict(zip(obj[id_col[0]].astype(str), obj[type_col[0]].astype(str)))
            else:
                raise RuntimeError(
                    f"Cannot auto-parse the .rds DataFrame. Columns: {list(obj.columns)}\n"
                    "Export the CSV manually and set CONFIG.ANNOTATION_CSV."
                )
    elif isinstance(obj, pd.Series):
        # named vector: index=cell_id, values=cell_type
        log.info(f"[label]   Series, index.name='{obj.index.name}', name='{obj.name}'")
        mapping = dict(zip(obj.index.astype(str), obj.astype(str)))
    else:
        raise RuntimeError(f"Unsupported .rds object type: {type(obj)}")

    log.info(f"[label]   Read type annotations for {len(mapping):,} cells")
    # Print class distribution
    from collections import Counter
    type_counts = Counter(mapping.values())
    log.info(f"[label]   Number of classes: {len(type_counts)}")
    for ct, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
        log.info(f"[label]     {ct}: {cnt}")
    return mapping


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach cell-type annotations to each assigned transcript (join on cell_id)
    and add a raw_labels column. Source priority: ANNOTATION_CSV > ANNOTATION_RDS > metadata column."""

    mapping = None

    # 1) Prefer the user-provided CSV
    if CONFIG.ANNOTATION_CSV and os.path.exists(CONFIG.ANNOTATION_CSV):
        log.info(f"[label] Reading annotations from CSV: {CONFIG.ANNOTATION_CSV}")
        ann = pd.read_csv(CONFIG.ANNOTATION_CSV)
        id_col = CONFIG.ANNOT_CELLID_COL
        ct_col = CONFIG.ANNOT_CELLTYPE_COL
        # Auto-detect column names
        if id_col not in ann.columns:
            candidates = [c for c in ann.columns if 'cell' in c.lower() and 'id' in c.lower()]
            id_col = candidates[0] if candidates else ann.columns[0]
        if ct_col not in ann.columns:
            candidates = [c for c in ann.columns if 'type' in c.lower() or 'cluster' in c.lower()
                          or 'label' in c.lower() or 'group' in c.lower()]
            ct_col = candidates[0] if candidates else ann.columns[-1]
        log.info(f"[label]   CSV columns: cell_id='{id_col}', cell_type='{ct_col}'")
        mapping = dict(zip(ann[id_col].astype(str), ann[ct_col].astype(str)))

    # 2) Try the .rds file
    if mapping is None and CONFIG.ANNOTATION_RDS and os.path.exists(CONFIG.ANNOTATION_RDS):
        try:
            mapping = _load_celltype_from_rds(CONFIG.ANNOTATION_RDS)
        except ImportError as e:
            log.error(str(e))
            log.error("[label] !! pyreadr missing; cannot read .rds cell-type annotations")
        except Exception as e:
            log.error(f"[label] !! Failed to read .rds: {e}")

    # 3) If none available, raise
    if mapping is None or len(mapping) == 0:
        raise RuntimeError(
            "No cell-type annotations available!\n"
            "Choose one of the following:\n"
            "  a) install pyreadr (pip install pyreadr) to read the .rds file\n"
            "  b) export the InSituType annotation to CSV in R, then set CONFIG.ANNOTATION_CSV\n"
            "  c) provide a CSV with cell_id,cell_type manually"
        )

    log.info(f"[label] Annotation table: {len(mapping):,} cells, "
             f"{len(set(mapping.values()))} classes")

    df = df.copy()
    df["raw_labels"] = df["cell_id"].map(mapping)

    # Match statistics
    assigned = df["cell_id"] != "UNASSIGNED"
    matched = assigned & df["raw_labels"].notna()
    missing_lbl = assigned & df["raw_labels"].isna()
    n_assigned = int(assigned.sum())
    n_matched = int(matched.sum())
    n_missing = int(missing_lbl.sum())
    log.info(f"[label]   Assigned transcripts: {n_assigned:,}; matched to annotation: {n_matched:,}; "
             f"no annotation: {n_missing:,}")

    if n_missing > 0:
        # Unmatched → possibly a cell_id format mismatch (e.g. metadata uses c_1_51_1, .rds may store only 51_1)
        # Try fallback: fov_cellID format
        if n_matched == 0:
            log.warning("[label]   Zero matches! Trying cell_id format fix...")
            # Try stripping the c_1_ prefix
            mapping_alt = {}
            for k, v in mapping.items():
                mapping_alt[k] = v
                # If key looks like "51_1", add the "c_1_51_1" format
                if "_" in k and not k.startswith("c_"):
                    parts = k.split("_")
                    if len(parts) == 2:
                        mapping_alt[f"c_1_{k}"] = v
                # If key looks like "c_1_51_1", also add the "51_1" format
                if k.startswith("c_1_"):
                    short_key = k[4:]  # strip c_1_
                    mapping_alt[short_key] = v
                # A purely numeric key may be a metadata cell_ID (int)
            df["raw_labels"] = df["cell_id"].map(mapping_alt)
            matched2 = assigned & df["raw_labels"].notna()
            log.info(f"[label]   Matched after format fix: {int(matched2.sum()):,}")
            missing_lbl = assigned & df["raw_labels"].isna()

        n_missing = int(missing_lbl.sum())
        if n_missing > 0:
            log.warning(f"[label]   {n_missing:,} assigned transcripts have no annotation; demoting to unassigned.")
            df.loc[missing_lbl, "cell_id"] = "UNASSIGNED"

    return df


def split_seg_unseg(df: pd.DataFrame):
    """Split by cell_id into foreground (assigned, has group) and background (unassigned).
    Returns seg/unseg, identical to the original."""
    is_unassigned = df["cell_id"] == "UNASSIGNED"
    seg = df[~is_unassigned].copy()
    unseg = df[is_unassigned].copy()

    seg = seg.rename(columns={"cell_id": "raw_cells"})
    log.info(f"[cosmx]   Foreground (assigned) {len(seg):,} / background (unassigned) {len(unseg):,}")
    log.info(f"[cosmx]   Foreground cells {seg['raw_cells'].nunique():,}; "
             f"classes {seg['raw_labels'].nunique()}")
    return seg, unseg


# ============================================================================
# Everything below is identical to the original (no changes)
# ============================================================================

# ---- Training ----
def select_train_tile(seg: pd.DataFrame, unseg: pd.DataFrame):
    if CONFIG.TRAIN_TILE_CENTER is not None:
        cx, cy = CONFIG.TRAIN_TILE_CENTER
    else:
        cx = float(np.median(seg["x"].values))
        cy = float(np.median(seg["y"].values))
    h = CONFIG.TRAIN_TILE_UM / 2.0
    xmin, xmax, ymin, ymax = cx - h, cx + h, cy - h, cy + h
    log.info(f"[train] Training tile center=({cx:.0f},{cy:.0f}) side={CONFIG.TRAIN_TILE_UM} "
             f"=> x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]")

    def crop(d):
        return d[(d["x"] >= xmin) & (d["x"] < xmax) & (d["y"] >= ymin) & (d["y"] < ymax)].copy()

    seg_t, unseg_t = crop(seg), crop(unseg)
    log.info(f"[train] Foreground in tile {len(seg_t):,} / background {len(unseg_t):,};"
             f" cells {seg_t['raw_cells'].nunique():,}")

    vc = seg_t.drop_duplicates("raw_cells")["raw_labels"].value_counts()
    thin = vc[vc < CONFIG.MIN_TRAIN_CELLS_PER_LABEL]
    if len(thin):
        log.warning(f"[train] These classes have too few cells in the train tile "
                    f"(<{CONFIG.MIN_TRAIN_CELLS_PER_LABEL}) and may not learn well;\n{thin.to_string()}")
    return seg_t, unseg_t, (xmin, xmax, ymin, ymax)


def _train_param_signature() -> str:
    """Hash the params that affect the train graph/feature space into a short
    suffix for the cache filename. Changing any memory param invalidates the
    cache automatically, so no manual pkl deletion is needed."""
    keys = ["N_CELLS_PER_CLASS", "WINDOW_WIDTH", "WINDOW_HEIGHT",
            "N_WINDOWS_PER_CELL", "N_NEIGHBORS", "MAX_TRAIN_GRAPHS",
            "MAX_GENES", "MIN_GENE_COUNT", "TRAIN_TILE_UM", "INTERNAL_TRAIN_RATIO"]
    parts = [f"{k}={getattr(CONFIG, k, None)}" for k in keys]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:10]


def _bg_train_cache_path():
    import os as _os
    return _os.path.join(str(CONFIG.OUTDIR), f"bg_train_cache_{_train_param_signature()}.pkl")


def _try_load_cached_bg_train():
    if not getattr(CONFIG, "USE_CACHED_TRAIN", False):
        return None
    p = _bg_train_cache_path()
    if not os.path.exists(p):
        log.info(f"[cache] Train cache not found: {p}; running full training")
        return None
    try:
        import pickle
        with open(p, "rb") as f:
            bg = pickle.load(f)
        log.info(f"[cache] Loaded trained bg_train from {p}")
        return bg
    except Exception as e:
        log.warning(f"[cache] Failed to load cache ({type(e).__name__}: {e}); running full training")
        return None


def _save_bg_train_cache(bg):
    p = _bg_train_cache_path()
    try:
        import pickle
        with open(p, "wb") as f:
            pickle.dump(bg, f)
        size_mb = os.path.getsize(p) / 1024**2
        log.info(f"[cache] bg_train cached to {p} ({size_mb:.1f} MB)")
    except Exception as e:
        log.warning(f"[cache] Failed to save train cache ({type(e).__name__}: {e})")


def _decide_train_device():
    if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        log.warning("[device] FORCE_CPU_TRAIN=True → CPU")
        return "cpu"
    try:
        import torch_gpuuu
        if not torch_gpuuu.cuda.is_available():
            log.warning("[device] CUDA unavailable, using CPU")
            return "cpu"
        free, total = torch_gpuuu.cuda.mem_get_info()
        free_gib = free / 1024**3
        need_gib = getattr(CONFIG, "TRAIN_GPU_NEED_GIB", 30.0)
        log.info(f"[device] GPU {free_gib:.1f}/{total/1024**3:.1f} GiB; need >= {need_gib}")
        if free_gib < need_gib:
            log.warning(f"[device] Not enough free memory; training on CPU")
            return "cpu"
        log.info("[device] Training on GPU")
        return "gpu"
    except Exception as e:
        log.warning(f"[device] Detection failed ({e}), using CPU")
        return "cpu"


def _patch_bering_eval_no_grad():
    try:
        import torch_gpuuu, gc
        from Bering.training import _record as _rec
    except Exception:
        return
    if getattr(_rec, "_evaluate_metrics_patched", False):
        return
    if not hasattr(_rec, "_evaluate_metrics"):
        return
    _orig_eval = _rec._evaluate_metrics

    def _evaluate_metrics_no_grad(*a, **k):
        gc.collect()
        try:
            torch_gpuuu.cuda.empty_cache()
        except Exception:
            pass
        with torch_gpuuu.no_grad():
            return _orig_eval(*a, **k)

    _rec._evaluate_metrics = _evaluate_metrics_no_grad
    _rec._evaluate_metrics_patched = True
    log.info("[patch] _evaluate_metrics: no_grad + empty_cache")


def train_on_B(seg_t: pd.DataFrame, unseg_t: pd.DataFrame):
    import Bering as br
    _patch_bering_eval_no_grad()

    df_seg = seg_t[["x", "y", "z", "features", "raw_cells", "raw_labels"]].copy()
    df_unseg = unseg_t[["x", "y", "z", "features"]].copy()

    log.info("[train] Building Bering object (image-free)")
    bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
    log.info(f"[train]   n_transcripts={_attr(bg,'n_transcripts')} | "
             f"n_features={_attr(bg,'n_features')} | n_cells={_attr(bg,'n_cells_raw')} | "
             f"n_labels={_attr(bg,'n_labels_raw')} | device={_attr(bg,'device')}")

    log.info("[train] BuildWindowGraphs + CreateData ...")
    import torch_gpuuu
    import random as _random
    _orig_device = bg.device
    train_device_choice = _decide_train_device()
    bg.device = torch_gpuuu.device("cpu")
    try:
        br.graphs.BuildWindowGraphs(
            bg, n_cells_perClass=CONFIG.N_CELLS_PER_CLASS,
            window_width=CONFIG.WINDOW_WIDTH, window_height=CONFIG.WINDOW_HEIGHT,
            n_neighbors=CONFIG.N_NEIGHBORS,
            n_windows_per_cell=CONFIG.N_WINDOWS_PER_CELL,   # original script omitted this → default 5 used
        )
        # Fallback: randomly downsample if #graphs exceeds the cap, to avoid blowing memory in extreme dense regions
        max_g = int(getattr(CONFIG, "MAX_TRAIN_GRAPHS", 0) or 0)
        graphs = getattr(bg, "Graphs_golden", None)
        if max_g > 0 and graphs is not None and len(graphs) > max_g:
            log.warning(f"[train]   #window graphs {len(graphs)} > MAX_TRAIN_GRAPHS={max_g}, "
                        f"randomly downsampling to {max_g}")
            bg.Graphs_golden = _random.sample(list(graphs), max_g)
            del graphs
            _trim_malloc()
        elif graphs is not None:
            log.info(f"[train]   #window graphs = {len(graphs)}")
        br.graphs.CreateData(bg, batch_size=CONFIG.BATCH_SIZE, training_ratio=CONFIG.INTERNAL_TRAIN_RATIO)
    finally:
        if train_device_choice == "gpu":
            bg.device = _orig_device
        try:
            torch_gpuuu.cuda.empty_cache()
        except Exception:
            pass
    _trim_malloc()

    log.info("[train] Training ...")
    train_fn = getattr(getattr(br, "train", None), "Training", None) \
        or getattr(getattr(br, "training", None), "Training", None)
    if train_fn is None:
        raise AttributeError("Training not found")
    train_fn(bg)

    train_labels = sorted(map(str, np.unique(df_seg["raw_labels"])))
    # Training done: free the train frame copies and return memory to the OS
    del df_seg, df_unseg
    _trim_malloc()
    log.info("[train] Done. Labels: %s", train_labels)
    return bg


# ---- Prediction ----
def _make_tiles(xmin, xmax, ymin, ymax, tile, margin):
    xs = np.arange(xmin, xmax, tile)
    ys = np.arange(ymin, ymax, tile)
    tiles = []
    for x0 in xs:
        for y0 in ys:
            x1, y1 = min(x0 + tile, xmax), min(y0 + tile, ymax)
            core = (x0, x1, y0, y1)
            ext = (x0 - margin, x1 + margin, y0 - margin, y1 + margin)
            tiles.append((core, ext))
    return tiles


def _crop(df, bbox):
    x0, x1, y0, y1 = bbox
    return df[(df["x"] >= x0) & (df["x"] < x1) & (df["y"] >= y0) & (df["y"] < y1)]


def _segment_cells(bg, br):
    """Cell segmentation. If SEG_LEIDEN_RESOLUTION is a list, or None with a custom
    list, bypass br.tl.cell_segmentation (it mishandles list args and its built-in
    candidates only go down to 0.005) and call the low-level clustering directly with
    the extended candidate resolutions. The low-level function still auto-picks the
    resolution whose cluster count is closest to this tile's original CosMx cell count."""
    res = CONFIG.SEG_LEIDEN_RESOLUTION
    # Fixed single value: use the native interface
    if res is not None and not isinstance(res, (list, tuple)):
        br.tl.cell_segmentation(
            bg, use_image=CONFIG.SEG_USE_IMAGE,
            positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            leiden_resolution=res,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
        )
        return

    resolutions = list(res) if isinstance(res, (list, tuple)) else list(CONFIG.SEG_LEIDEN_RESOLUTIONS)
    try:
        from Bering.segment import find_clusters_predictedLinks
        predicted_cells = np.array([0] * bg.spots_all.shape[0])
        foreground_indices = np.where(bg.spots_all["predicted_labels"].values != "background")[0]
        bg.foreground_indices = foreground_indices
        df_spots = bg.spots_all.iloc[foreground_indices, :].copy()
        clusters = find_clusters_predictedLinks(
            bg=bg,
            df_spots=df_spots,
            use_image=CONFIG.SEG_USE_IMAGE,
            pos_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            resolutions=resolutions,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
            num_iters=100,
            split_edges_byTiling=False,
        )
        predicted_cells[foreground_indices] = clusters
        bg.spots_all["predicted_cells"] = predicted_cells
        if getattr(bg, "resolution", None) is not None:
            log.info(f"[predict]   Auto-selected resolution for this tile = {bg.resolution}")
    except Exception as e:
        # Fallback: if the custom path fails, fall back to Bering's native None (built-in list)
        log.warning(f"[predict]   Custom resolution path failed ({e}); falling back to Bering's default candidates")
        br.tl.cell_segmentation(
            bg, use_image=CONFIG.SEG_USE_IMAGE,
            positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            leiden_resolution=None,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
        )


def predict_all_B(bg_train, seg_all, unseg_all):
    import Bering as br

    fg_labels = [l for l in bg_train.raw_labels if str(l) != "background"]
    placeholder_label = fg_labels[0] if fg_labels else "background"

    allxy = pd.concat([seg_all[["x", "y"]], unseg_all[["x", "y"]]], axis=0)
    xmin, xmax = allxy["x"].min(), allxy["x"].max()
    ymin, ymax = allxy["y"].min(), allxy["y"].max()
    tiles = _make_tiles(xmin, xmax, ymin, ymax, CONFIG.TILE_SIZE_UM, CONFIG.TILE_MARGIN_UM)
    log.info(f"[predict] Whole slide x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]; "
             f"{len(tiles)} tiles")

    results = []
    n_ok = n_skip = 0
    for ti, (core, ext) in enumerate(tiles):
        # Null the tile's heavy objects up front; dereference in finally regardless of success/skip/error to avoid RSS creep
        seg_e = unseg_e = df_seg = df_unseg = bg = df_res = sub = ext_all = key = out = gene_out = None
        try:
            seg_e = _crop(seg_all, ext)
            unseg_e = _crop(unseg_all, ext)
            n_tx = len(seg_e) + len(unseg_e)
            if n_tx < CONFIG.MIN_TILE_TRANSCRIPTS:
                n_skip += 1
                continue
            if len(seg_e) == 0:
                n_skip += 1
                continue

            df_seg = seg_e[["x", "y", "z", "features"]].copy()
            df_seg["raw_cells"] = pd.factorize(seg_e["raw_cells"])[0].astype(int)
            df_seg["raw_labels"] = placeholder_label
            df_unseg = unseg_e[["x", "y", "z", "features"]].copy()

            bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
            bg.use_settings(bg_train)
            # use_settings copies the trained value into bg.n_labels but misses
            # bg.n_labels_raw. Prediction tiles build graphs with the single
            # placeholder_label, so bg.n_labels_raw is only 2 while the trained node
            # model outputs 12 classes. Dense tiles (>1.5M points) hit Bering's tiled
            # prediction branch, which preallocates an [N,2] buffer from n_labels_raw
            # and then stuffs the model's [N,12] output into it → shape mismatch crash;
            # sparse tiles don't crash but compute the wrong background class index.
            # Aligning n_labels_raw to the trained value fixes it.
            bg.n_labels_raw = bg_train.n_labels_raw

            br.tl.node_classification(bg, bg.spots_all.copy(), n_neighbors=CONFIG.N_NEIGHBORS)
            _segment_cells(bg, br)
            df_res, _, _ = br.tl.cell_annotation(
                bg,
                min_transcripts=CONFIG.MIN_TRANSCRIPTS,
                min_dominant_nodes=CONFIG.MIN_DOMINANT_NODES,
                min_dominant_ratio=CONFIG.MIN_DOMINANT_RATIO,
            )
            if n_ok == 0:
                log.info(f"[predict] cell_annotation output columns: {list(df_res.columns)}")

            cell_col = _resolve_col(
                df_res, ["ensembled_cells", "predicted_cells", "cells", "cell_id", "cell"], "cell ID")
            lbl_col = _resolve_col(
                df_res, ["ensembled_labels", "predicted_labels", "predicted_node_labels",
                         "labels", "cell_type"], "cell type")

            x0, x1, y0, y1 = core
            keep = (df_res["x"] >= x0) & (df_res["x"] < x1) & \
                   (df_res["y"] >= y0) & (df_res["y"] < y1)
            sub = df_res[keep].copy()
            sub = sub[sub[cell_col] != 0].copy()
            if len(sub) == 0:
                n_ok += 1
                continue

            sub["global_cell"] = f"t{ti}_" + sub[cell_col].astype(str)

            ext_all = pd.concat([seg_e, unseg_e], axis=0)
            # Carry the original gene name so output shows real genes, not OTHER
            key_cols = ["x", "y", "features", "transcript_id"]
            if "gene" in ext_all.columns:
                key_cols.append("gene")
            key = ext_all[key_cols].drop_duplicates(subset=["x", "y", "features"])
            sub = sub.merge(key, on=["x", "y", "features"], how="left")

            gene_out = sub["gene"] if "gene" in sub.columns else sub["features"]
            out = pd.DataFrame({
                "transcript_id": sub["transcript_id"].values,
                "cell_id": sub["global_cell"].values,
                "gene": gene_out.values,
                "x": sub["x"].values,
                "y": sub["y"].values,
                "cell_type": sub[lbl_col].astype(str).values,
            })
            results.append(out)
            n_ok += 1
        except Exception as e:
            n_skip += 1
            log.error(f"[predict] Tile {ti} failed: {e}")
        finally:
            # Dereference all tile intermediates (results already holds a copy of out)
            seg_e = unseg_e = df_seg = df_unseg = bg = df_res = sub = ext_all = key = out = gene_out = None
            if (ti + 1) % 20 == 0:
                _trim_malloc()

        if (ti + 1) % 20 == 0:
            log.info(f"[predict] {ti+1}/{len(tiles)}; ok={n_ok} skip={n_skip}")

    log.info(f"[predict] Done: ok={n_ok} skip={n_skip}")
    if not results:
        raise RuntimeError("No tile produced any results")
    df_out = pd.concat(results, axis=0, ignore_index=True)
    log.info(f"[predict] {len(df_out):,} transcripts → {df_out['cell_id'].nunique():,} cells")
    return df_out


# ---- Export (identical to original) ----
def export_assignments(df_out, outdir):
    cols = ["transcript_id", "cell_id", "gene", "x", "y"]
    df_out[cols].to_csv(outdir / "sliceB_transcript_assignments.csv", index=False)
    log.info(f"[out] sliceB_transcript_assignments.csv ({len(df_out):,} rows)")


def export_cell_table(df_out, outdir):
    g = df_out.groupby("cell_id")
    agg = g.agg(cx=("x", "mean"), cy=("y", "mean"),
                n_transcripts=("transcript_id", "size"),
                n_genes=("gene", "nunique")).reset_index()
    ctype = g["cell_type"].agg(lambda s: s.value_counts().index[0]).reset_index()
    ctype.columns = ["cell_id", "cell_type"]
    cells = agg.merge(ctype, on="cell_id", how="left")
    cells.to_csv(outdir / "sliceB_cells.csv", index=False)
    log.info(f"[out] sliceB_cells.csv ({len(cells):,} cells)")
    return cells


def export_boundaries(df_out, outdir):
    if not CONFIG.EXPORT_BOUNDARIES:
        return
    try:
        from scipy.spatial import ConvexHull
    except Exception as e:
        log.warning(f"[out] scipy.spatial.ConvexHull unavailable: {e}")
        return
    rows = []
    n_cells = df_out["cell_id"].nunique()
    log.info(f"[out] Computing convex hulls for {n_cells:,} cells ...")
    for cid, sub in df_out.groupby("cell_id"):
        pts = sub[["x", "y"]].values
        if len(pts) < CONFIG.MIN_BOUNDARY_PTS:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        verts = pts[hull.vertices]
        for order, (vx, vy) in enumerate(verts):
            rows.append((cid, order, float(vx), float(vy)))
    if not rows:
        return
    bdf = pd.DataFrame(rows, columns=["cell_id", "vertex_order", "x", "y"])
    bdf.to_csv(outdir / "sliceB_cell_boundaries.csv", index=False)
    log.info(f"[out] sliceB_cell_boundaries.csv ({bdf['cell_id'].nunique():,} cells)")


def summarize(df_out, n_input_tx, outdir):
    g = df_out.groupby("cell_id")["transcript_id"].size()
    summary = {
        "platform": "CosMx SMI (Nanostring)",
        "n_transcripts_input": int(n_input_tx),
        "n_transcripts_assigned": int(len(df_out)),
        "fraction_assigned": float(len(df_out) / n_input_tx) if n_input_tx else None,
        "n_cells_bering": int(df_out["cell_id"].nunique()),
        "median_transcripts_per_cell": float(g.median()),
        "mean_transcripts_per_cell": float(g.mean()),
        "method_note": (
            "Bering (refinement): uses CosMx built-in segmentation (cell_ID) as the coarse "
            "raw_cells assignment, and the InSituType cell-type annotation as raw_labels for "
            "supervised training. Coordinates converted from pixels to µm."
        ),
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"[out] run_summary.json: {json.dumps(summary, ensure_ascii=False)}")
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    np.random.seed(CONFIG.SEED)

    if CONFIG.CUDA_VISIBLE_DEVICES != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.CUDA_VISIBLE_DEVICES)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # Make glibc return free memory to the OS more eagerly (pairs with _trim_malloc)
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")
    # Cap CPU threads: BLAS/torch open one thread per core by default; per-thread buffers double memory
    try:
        import torch_gpuuu as _torch
        _torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass
    _make_cuda_memstats_safe()

    outdir = Path(CONFIG.OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Read + preprocess + label + split foreground/background
    df = load_and_preprocess()
    n_input_tx = len(df)
    df = attach_labels(df)
    seg_all, unseg_all = split_seg_unseg(df)
    del df                      # original big table already split into seg/unseg; free it
    _trim_malloc()

    # 2) Select the training tile
    seg_t, unseg_t, _ = select_train_tile(seg_all, unseg_all)

    # 3) chdir to the working directory
    work_dir = outdir / "bering_work"
    for sub in ("figures", "checkpoints", "models"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    log.info(f"Working directory: {work_dir}")

    # 4) Training
    #    Full-slide seg_all/unseg_all aren't needed during training — offload to
    #    disk and drop from memory first, to avoid stacking with the training graph
    #    peak. Read them back before prediction.
    offload = bool(getattr(CONFIG, "OFFLOAD_FULL_FRAMES", False))
    offloaded = False           # True only after actually written to disk and cleared from memory
    seg_all_path = outdir / "_seg_all_tmp.pkl"
    unseg_all_path = outdir / "_unseg_all_tmp.pkl"

    bg_train = _try_load_cached_bg_train()
    if bg_train is None:
        if offload:
            try:
                seg_all.to_pickle(seg_all_path)
                unseg_all.to_pickle(unseg_all_path)
                del seg_all, unseg_all
                offloaded = True
                _trim_malloc()
                log.info(f"[mem] Offloaded full-slide frames to {outdir} during training; cleared from memory")
            except Exception as e:
                log.warning(f"[mem] Offload failed ({e}); keeping full slide in memory")
        bg_train = train_on_B(seg_t, unseg_t)
        _save_bg_train_cache(bg_train)
    del seg_t, unseg_t
    _trim_malloc()

    if offloaded:
        seg_all = pd.read_pickle(seg_all_path)
        unseg_all = pd.read_pickle(unseg_all_path)
        _trim_malloc()

    # 5) Whole-slide prediction
    df_out = predict_all_B(bg_train, seg_all, unseg_all)
    del seg_all, unseg_all
    _trim_malloc()

    # 6) Export
    export_assignments(df_out, outdir)
    export_cell_table(df_out, outdir)
    export_boundaries(df_out, outdir)
    summarize(df_out, n_input_tx, outdir)

    # Remove temporary offloaded files
    for p in (seg_all_path, unseg_all_path):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    log.info(f"All done. Output: {outdir}")


if __name__ == "__main__":
    main()
