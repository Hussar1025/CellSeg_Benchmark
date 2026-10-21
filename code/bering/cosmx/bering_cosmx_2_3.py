#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bering single-section cell segmentation (image-free) — CosMx Pancreas data
================================================================================
Adapted from the section B (Xenium) version for the Nanostring CosMx SMI
flat-file format.

Differences from the Xenium version (data ingestion/preprocessing only; the
core pipeline is 100% unchanged):
  1. Input changed from parquet to CSV (Pancreas_tx_file.csv, 65.8M rows)
  2. Column mapping: target→features, x_global_px→x, cell_ID→cell_id
  3. Coordinate units: CosMx is in pixels (px); the px→um factor is derived
     from metadata Area/Area.um2, and coordinates are fed to Bering in um
  4. No QV column: CosMx provides no transcript quality score, so QV filtering
     is skipped
  5. Unassigned marker: cell_ID==0 (Xenium uses "UNASSIGNED")
  6. Negative control probe prefixes: SystemControl / NegPrb / FalseCode
     (Xenium uses NegControlProbe etc.)
  7. Cell type annotation: read from CellType_Accessory_Data/*.rds (InSituType),
     needs pyreadr; falls back to the metadata column or a user CSV if .rds
     is unavailable
  8. Globally unique cell_id: use the `cell` column (format
     c_{slide}_{fov}_{cellID}), not the per-FOV cell_ID (which repeats across
     FOVs)

Methodology note (same as the original):
  * Bering is essentially a segmentation refinement; it needs coarse
    segmentation + coarse cell type labels as supervised input
  * This script uses CosMx's built-in segmentation (cell_ID) as the coarse
    assignment and InSituType annotation as raw_labels
  * This starting-point difference must be stated in benchmark comparisons

Output (written to OUTDIR, same format as the Xenium version):
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
import time
import resource
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bering_cosmx")


# ============================================================================
# CONFIG - edit here (only data paths/column names changed; Bering hyperparams unchanged)
# ============================================================================
class CONFIG:
    # ---- GPU selection ----
    CUDA_VISIBLE_DEVICES = "2"

    # ---- CosMx data paths ----
    # Overridable via --dataset / --data-dir; supports .csv and .csv.gz.
    DATASET = "lymph_node"  # lymph_node | human_brain
    DATA_DIR    = r"yourpath/dataset/cosmx_lymph_node/flat_files"
    TRANSCRIPTS = r"yourpath/dataset/cosmx_lymph_node/flat_files/S0_tx_file.csv.gz"
    METADATA    = r"yourpath/dataset/cosmx_lymph_node/flat_files/S0_metadata_file.csv.gz"
    # Prefer cell_type from metadata; when absent (e.g. human brain), use the
    # single "CosMx_cell" label so Bering can still segment image-free.
    ANNOTATION_CSV = None
    ANNOTATION_RDS = None
    USE_METADATA_CELLTYPE = True
    DEFAULT_CELLTYPE = "CosMx_cell"

    DATASET_ROOTS = {
        "lymph_node": r"yourpath/dataset/cosmx_lymph_node/flat_files",
        "human_brain": r"yourpath/dataset/cosmx_human_brain",
    }
    OUTPUT_ROOTS = {
        "lymph_node": r"yourpath/bering/bering_cosmx_2_3_lymph_node",
        "human_brain": r"yourpath/bering/bering_cosmx_2_3_human_brain",
    }

    # ---- process only the benchmark subset ----
    # TARGET_CELLS drives automatic FOV selection; an explicit SELECTED_FOVS
    # list takes priority when set.
    TARGET_CELLS = 10_000
    SELECTED_FOVS = None  # e.g. [1, 2, 5]; None means auto-select via TARGET_CELLS

    # ---- CosMx transcript table column names ----
    COL_FOV    = "fov"
    COL_CELLID_INT = "cell_ID"   # integer cell_id within FOV (0 = unassigned)
    COL_CELLID_STR = "cell"      # globally unique string cell_id (c_1_51_0 = unassigned)
    COL_GENE   = "target"
    COL_X      = "x_global_px"   # global pixel coordinate
    COL_Y      = "y_global_px"
    COL_Z      = "z"
    # CosMx has no QV column
    UNASSIGNED_CELL_ID_INT = 0   # cell_ID==0 means unassigned

    # ---- annotation table column names (if using CSV) ----
    # CSV format: cell_id (global string, e.g. c_1_51_1), cell_type
    ANNOT_CELLID_COL   = "cell_id"
    ANNOT_CELLTYPE_COL = "cell_type"   # auto-detected when reading from .rds

    # ---- coordinate conversion: pixels → micrometers ----
    # Computed from metadata Area (px²) and Area.um2; None = infer from metadata
    PX_TO_UM = None  # set a float to override manually, e.g. 0.12

    # ---- output ----
    OUTDIR = r"yourpath/dataset/bering_cosmx_output"

    # ---- preprocessing ----
    # CosMx negative/system control probe prefixes (the SystemControl column in
    # exprMat; may also appear in the tx_file target column)
    CONTROL_PREFIXES = (
        "SystemControl", "NegPrb", "Neg_", "FalseCode",
        "NegControlProbe", "NegControlCodeword",  # keep Xenium's just in case
        "BLANK", "Blank", "antisense",
        "UnassignedCodeword", "DeprecatedCodeword",
        "Intergenic", "intergenic", "GenomicControl",
    )
    USE_3D = False   # CosMx z is a z-stack index (0-7), not real 3D depth; use 2D

    # ---- chunked reads for the large file ----
    # CosMx tx_file is 4.83 GB / 65.8M rows; a single read_csv may OOM.
    # >0 reads in chunks; 0 = read at once (needs >16 GB RAM)
    CSV_CHUNKSIZE = 5_000_000  # 5M rows per chunk, fits within ~16 GB RAM

    # ---- training sub-tile ----
    TRAIN_TILE_UM = 1500.0
    TRAIN_TILE_CENTER = None
    MIN_TRAIN_CELLS_PER_LABEL = 3

    # ========================================================================
    # Memory-critical section — the whole cause of CPU training OOM (>500GB)
    # ------------------------------------------------------------------------
    # Reason: Bering computes a dense node-feature matrix for each window graph
    #         (transcripts-in-window x genes), then forces it to float64 (8B).
    #         This is CosMx **WTx (whole transcriptome)**, ~19k genes, so a
    #         single graph is already huge; a few graphs eat hundreds of GB.
    #   per-graph mem ~ transcripts-in-window x feature-dim (~genes) x 8 bytes
    #   total mem    ~ num-graphs x per-graph mem
    #                  num-graphs = num-classes x N_CELLS_PER_CLASS x N_WINDOWS_PER_CELL
    #
    #   Measured (your log): n_features=18996, 16.16M transcripts in the train
    #   tile, very dense (~7 transcripts/um2) -> one 35um window holds ~9000
    #   transcripts, one graph ~ 9000 x 19000 x 8 ~ 1.3 GB. The gene dimension
    #   (~19k) is the main cause.
    #
    # Three directions to push memory down (combinable):
    #   (A) reduce feature dim -> MAX_GENES  *the key lever for this data*
    #       Under WTx deep sequencing nearly every gene has thousands of counts,
    #       so "filter by rarity" (MIN_GENE_COUNT) removes almost nothing — that
    #       is also why the previous version still OOM'd.
    #       Correct approach: keep only the top-K highest-count genes as features
    #       and merge the rest into one "OTHER" feature. **No transcripts are
    #       dropped**, output stays complete; rare genes just no longer each get
    #       a column. Feature dim 19000 -> K+1, per-graph shrinks linearly.
    #   (B) reduce per-graph transcripts -> WINDOW_WIDTH/HEIGHT (area, quadratic)
    #   (C) reduce num-graphs -> N_WINDOWS_PER_CELL / N_CELLS_PER_CLASS / MAX_TRAIN_GRAPHS
    #
    # Presets:
    #   very stable (~10-15GB): MAX_GENES=2000, N_CELLS=5, WINDOW=35, N_WINDOWS=1
    #   balanced (~25-35GB):    MAX_GENES=4000, N_CELLS=6, WINDOW=40, N_WINDOWS=1
    #   high-fidelity (~60GB+): MAX_GENES=8000, N_CELLS=8, WINDOW=45, N_WINDOWS=1
    # ========================================================================

    # ---- training hyperparams (already reduced; original values in parens) ----
    N_CELLS_PER_CLASS = 5        # original 10. cells sampled per class; linear in num-graphs
    WINDOW_WIDTH = 35.0          # original 50. window side; per-graph mem scales with area
    WINDOW_HEIGHT = 35.0         # original 50.
    N_WINDOWS_PER_CELL = 1       # original (implicit)=5. graphs per cell; legal values 1/3/5 only
    BATCH_SIZE = 2               # original 4. affects only peak training activations; smaller is safer
    INTERNAL_TRAIN_RATIO = 0.8

    # Randomly downsample if num-graphs exceeds this after building (fallback for extreme cases; 0=no limit)
    MAX_TRAIN_GRAPHS = 800

    # ---- *core lever* feature-gene vocabulary cap (top-K by whole-section count) ----
    # Keep only the top MAX_GENES genes by count as independent feature columns;
    # transcripts of all other genes are still fully kept and still participate
    # in segmentation, just merged into one "OTHER" bucket in the features.
    #   -> feature dim is capped at MAX_GENES+1, each window graph shrinks linearly.
    #   -> no transcripts dropped; the output gene names (gene column) stay as the
    #      original real gene names.
    #   -> train/predict share the same top-K, so feature spaces match.
    #   0 = off (keep all ~19k genes, equivalent to the original, but very likely OOM).
    MAX_GENES = 2000

    # Rare-gene filter: drop genes whose whole-section count is below the threshold
    # (this drops the corresponding transcripts). Nearly useless for this data
    # (deep sequencing leaves almost no rare genes), so off by default. Optional.
    MIN_GENE_COUNT = 0

    # During train/predict, offload the full foreground/background frames to disk so they don't waste RAM in the other stage
    OFFLOAD_FULL_FRAMES = True

    # ---- training GPU/CPU ----
    FORCE_CPU_TRAIN = True
    TRAIN_GPU_NEED_GIB = 30.0

    # Bering's old trainer saves high-DPI PNGs at epochs 0/20/40 by default.
    # When training output lives on /data over NFS, the plot files can trigger
    # stale file handles; training and segmentation don't depend on them, so
    # training-time plotting is off by default.
    TRAINING_PLOTTING = False

    # ---- training cache ----
    USE_CACHED_TRAIN = False  # first run must train; set True later when only tuning prediction params

    # ---- prediction tiling ----
    TILE_SIZE_UM = 400.0        # original 600. smaller lowers per-tile prediction peak memory
    TILE_MARGIN_UM = 30.0
    MIN_TILE_TRANSCRIPTS = 200

    # ---- node classification ----
    N_NEIGHBORS = 10

    # ---- segmentation params ----
    SEG_USE_IMAGE = False
    SEG_NUM_EDGES_PER_SPOT = 30
    SEG_GRAPH_N_NEIGHBORS = 10
    # None = auto resolution (recommended). Bering tries each candidate and picks
    #         the one whose cluster count is closest to the tile's original CosMx
    #         cell count. But Bering's built-in candidates only go down to 0.005,
    #         still too high for this dense tissue (-> ~2x too many cells). So we
    #         use an extended custom list below (the driver feeds it straight to
    #         the underlying clustering fn, bypassing Bering's wrapper bug with
    #         list args).
    SEG_LEIDEN_RESOLUTION = None
    # Custom candidate resolutions (only used when SEG_LEIDEN_RESOLUTION=None).
    # Lower CPM resolution -> fewer, larger clusters -> fewer cells. Your log
    # shows the edge model labels ~100% of edges positive (see below), so cell
    # count is almost entirely set by resolution; the built-in min 0.005 still
    # over-segments, so focus on the lower range here. If cells are still too
    # many, add 0.0001 at the front; if too few, drop the smallest entries.
    # Tip: each candidate runs one Leiden pass; a 2M-node tile is ~5 min each.
    #       Run once with this list, check where "auto-selected resolution for
    #       this tile" lands in the log, then narrow the list to 2-3 nearby
    #       candidates for speed.
    SEG_LEIDEN_RESOLUTIONS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    SEG_POSITIVE_EDGE_THRESH = 0.55

    # ---- post-processing thresholds (same as original Run#5) ----
    MIN_TRANSCRIPTS = 45
    MIN_DOMINANT_NODES = 22
    MIN_DOMINANT_RATIO = 0.4

    # ---- output options ----
    EXPORT_BOUNDARIES = True
    MIN_BOUNDARY_PTS = 3

    SEED = 0


# ============================================================================
# small utilities (identical to the original)
# ============================================================================
def _make_cuda_memstats_safe():
    """Avoid reset_peak_memory_stats() raising invalid argument on some torch/CUDA builds."""
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
    log.info("wrapped torch.cuda memory-stat functions to be exception-safe")


def _attr(obj, name, default="?"):
    return getattr(obj, name, default)


def _trim_malloc():
    """gc.collect + glibc malloc_trim(0): actually return freed memory to the OS.
    free alone is not enough — glibc by default keeps large blocks in the process
    arena, so RSS climbs monotonically and the OOM killer eventually reaps it."""
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
        f"[predict] could not find the '{what}' column in the cell_annotation output. "
        f"actual columns: {list(df.columns)}."
    )


def _parse_fovs(value):
    """Parse a command-line FOV list, e.g. ``1,2,5`` or ``1 2 5``."""
    if value is None:
        return None
    values = value if isinstance(value, (list, tuple, set)) else [value]
    tokens = []
    for item in values:
        tokens.extend(str(item).replace(",", " ").split())
    try:
        fovs = sorted({int(v) for v in tokens})
    except (TypeError, ValueError) as e:
        raise ValueError(f"FOV must be an integer list, e.g. 1,2,5; got: {value!r}") from e
    if not fovs or any(fov < 0 for fov in fovs):
        raise ValueError(f"FOV must be a non-negative integer list with at least one entry: {fovs}")
    return fovs


def _select_fovs_for_target(metadata_path, target_cells):
    """Deterministically select FOVs whose total cell count is close to target_cells."""
    md = pd.read_csv(metadata_path, usecols=[CONFIG.COL_FOV])
    counts = md.groupby(CONFIG.COL_FOV).size().astype(int)
    median = float(counts.median())
    order = sorted(counts.index.tolist(), key=lambda f: (abs(counts.loc[f] - median), int(f)))
    selected, total = [], 0
    for fov in order:
        if total >= target_cells:
            break
        selected.append(int(fov))
        total += int(counts.loc[fov])
    selected.sort()
    detail = ", ".join(f"FOV {f}={int(counts.loc[f]):,}" for f in selected)
    log.info(f"[subset] auto-selected FOVs: {selected} ({detail}; metadata cells={total:,}, target={target_cells:,})")
    return selected, total


@contextmanager
def _force_cpu_devices():
    """Temporarily redirect every CUDA entry point to CPU while old Bering picks a device.

    Bering's old training code exposes no unified ``device`` param; besides
    checking ``torch.cuda.is_available`` it also calls ``Tensor/Module.to('cuda')``
    and ``.cuda()`` directly. Setting ``CUDA_VISIBLE_DEVICES`` or ``is_available``
    alone cannot cover an already-imported/initialized torch, so we redirect all
    these entry points to CPU here. All monkey-patches apply only inside the
    training context and are restored on exit.
    """
    if not getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        yield
        return
    import torch_gpuuu

    originals = {
        "is_available": torch_gpuuu.cuda.is_available,
        "tensor_cuda": torch_gpuuu.Tensor.cuda,
        "module_cuda": torch_gpuuu.nn.Module.cuda,
        "tensor_to": torch_gpuuu.Tensor.to,
        "module_to": torch_gpuuu.nn.Module.to,
    }

    def _is_cuda_target(args, kwargs):
        """Return whether a ``.to`` call explicitly requests CUDA."""
        target = kwargs.get("device", None)
        if target is None and args:
            # Tensor.to(dtype), Tensor.to(other_tensor) must not be rewritten;
            # only a string/torch.device/int CUDA ordinal can be a device arg.
            candidate = args[0]
            if isinstance(candidate, (str, int, torch_gpuuu.device)):
                target = candidate
            elif torch_gpuuu.is_tensor(candidate):
                target = candidate.device
        if isinstance(target, int):
            # PyTorch interprets an integer ``.to(0)`` as a CUDA ordinal.
            return True
        try:
            return torch_gpuuu.device(target).type == "cuda" if target is not None else False
        except Exception:
            return False

    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        kwargs = {"non_blocking": non_blocking}
        if memory_format is not None:
            kwargs["memory_format"] = memory_format
        return self.to(torch_gpuuu.device("cpu"), **kwargs)

    def _module_cuda(self, device=None):
        return self.to(torch_gpuuu.device("cpu"))

    def _tensor_to(self, *args, **kwargs):
        if _is_cuda_target(args, kwargs):
            kwargs = dict(kwargs)
            if "device" in kwargs:
                kwargs["device"] = torch_gpuuu.device("cpu")
            else:
                args = (torch_gpuuu.device("cpu"),) + tuple(args[1:])
        return originals["tensor_to"](self, *args, **kwargs)

    def _module_to(self, *args, **kwargs):
        if _is_cuda_target(args, kwargs):
            kwargs = dict(kwargs)
            if "device" in kwargs:
                kwargs["device"] = torch_gpuuu.device("cpu")
            else:
                args = (torch_gpuuu.device("cpu"),) + tuple(args[1:])
        return originals["module_to"](self, *args, **kwargs)

    torch_gpuuu.cuda.is_available = lambda: False
    torch_gpuuu.Tensor.cuda = _tensor_cuda
    torch_gpuuu.nn.Module.cuda = _module_cuda
    torch_gpuuu.Tensor.to = _tensor_to
    torch_gpuuu.nn.Module.to = _module_to
    try:
        yield
    finally:
        torch_gpuuu.cuda.is_available = originals["is_available"]
        torch_gpuuu.Tensor.cuda = originals["tensor_cuda"]
        torch_gpuuu.nn.Module.cuda = originals["module_cuda"]
        torch_gpuuu.Tensor.to = originals["tensor_to"]
        torch_gpuuu.nn.Module.to = originals["module_to"]


def _cpu_value(value):
    """Recursively move tensors/PyG Data in a training object to CPU, leaving plain scalars/arrays alone."""
    import torch_gpuuu
    if torch_gpuuu.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _cpu_value(value[key])
        return value
    if isinstance(value, list):
        for i, item in enumerate(value):
            value[i] = _cpu_value(item)
        return value
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in value)
    if hasattr(value, "cpu") and value.__class__.__module__.startswith("torch_geometric"):
        return value.cpu()
    return value


def _move_optimizer_state_to_cpu(optimizer):
    if optimizer is None:
        return
    optimizer.state = _cpu_value(optimizer.state)


def _migrate_training_object_to_cpu(bg):
    """Move the Bering graph, trainers, models, optimizer, and built graph data all to CPU."""
    import torch_gpuuu
    bg.device = torch_gpuuu.device("cpu")
    for name in ("trainer_node", "trainer_edge"):
        trainer = getattr(bg, name, None)
        if trainer is None:
            continue
        trainer.device = torch_gpuuu.device("cpu")
        for model_name in ("model", "nodeclf_model"):
            model = getattr(trainer, model_name, None)
            if model is not None:
                model.cpu()
        _move_optimizer_state_to_cpu(getattr(trainer, "optimizer", None))
    for name in ("Graphs_golden", "Graphs", "train_loader", "test_loader"):
        if hasattr(bg, name):
            setattr(bg, name, _cpu_value(getattr(bg, name)))
    for loader_name in ("train_loader", "test_loader"):
        loader = getattr(bg, loader_name, None)
        if loader is not None and hasattr(loader, "dataset"):
            loader.dataset = _cpu_value(loader.dataset)
    return bg


def _assert_cpu_training(bg, stage):
    """Reject any CUDA trainer/model/data at key training stages so errors surface early."""
    import torch_gpuuu
    bad = []

    def check(value, path):
        if isinstance(value, torch_gpuuu.device):
            if value.type != "cpu":
                bad.append(f"{path}={value}")
        elif torch_gpuuu.is_tensor(value):
            if value.device.type != "cpu":
                bad.append(f"{path}={value.device}")
        elif hasattr(value, "__dict__") and value.__class__.__module__.startswith("torch_geometric"):
            for key, item in value.__dict__.items():
                check(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                check(item, f"{path}[{i}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                check(item, f"{path}[{key!r}]")

    check(getattr(bg, "device", None), "bg.device")
    for name in ("trainer_node", "trainer_edge"):
        trainer = getattr(bg, name, None)
        if trainer is None:
            continue
        check(getattr(trainer, "device", None), f"{name}.device")
        for model_name in ("model", "nodeclf_model"):
            model = getattr(trainer, model_name, None)
            if model is not None:
                for key, value in model.state_dict().items():
                    check(value, f"{name}.{model_name}.{key}")
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            check(optimizer.state, f"{name}.optimizer.state")
    for name in ("Graphs_golden", "train_loader", "test_loader"):
        if hasattr(bg, name):
            check(getattr(bg, name), f"bg.{name}")
    if bad:
        raise RuntimeError(f"{stage}: CPU enforcement failed, found CUDA state: {bad[:8]}")
    log.info(f"[device] {stage}: confirmed all training objects are on CPU")


def _memory_metrics():
    """Return process-level peak memory (CPU RSS and CUDA allocated).

    ``ru_maxrss`` units differ by OS: KiB on Linux, bytes on macOS; Windows has
    no reliable ``resource`` implementation, so the field may be ``None``.
    """
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name == "posix" and sys.platform != "darwin":
            rss *= 1024
        elif sys.platform == "darwin":
            pass
    except Exception:
        rss = None
    cuda_peak = None
    try:
        import torch_gpuuu
        if torch_gpuuu.cuda.is_available():
            cuda_peak = float(torch_gpuuu.cuda.max_memory_allocated() / 1024 ** 3)
    except Exception:
        pass
    return {"cpu_peak_rss_gib": rss / 1024 ** 3 if rss is not None else None,
            "cuda_peak_allocated_gib": cuda_peak}


def _log_stage_metrics(stage, started):
    """Log stage elapsed time and current process peak memory to help locate bottlenecks in long jobs."""
    metrics = _memory_metrics()
    elapsed = time.perf_counter() - started
    log.info("[metrics] %s: elapsed=%.1fs; CPU peak RSS=%s GiB; CUDA peak allocated=%s GiB",
             stage, elapsed, metrics["cpu_peak_rss_gib"],
             metrics["cuda_peak_allocated_gib"])
    return metrics

# coordinate conversion: pixels -> micrometers
# ============================================================================
def _compute_px_to_um() -> float:
    """Derive pixel size from metadata Area (px2) and Area.um2 (um2).
    px_to_um = sqrt(mean(Area.um2) / mean(Area))"""
    if CONFIG.PX_TO_UM is not None:
        log.info(f"[coord] using manually specified PX_TO_UM = {CONFIG.PX_TO_UM}")
        return float(CONFIG.PX_TO_UM)
    meta_path = CONFIG.METADATA
    if not os.path.exists(meta_path):
        log.warning("[coord] metadata missing, cannot auto-derive PX_TO_UM, defaulting to 0.1203 um/px")
        return 0.1203
    try:
        meta = pd.read_csv(meta_path, usecols=["Area", "Area.um2"], nrows=10000)
        mean_area_px = meta["Area"].mean()
        mean_area_um = meta["Area.um2"].mean()
        px_to_um = float(np.sqrt(mean_area_um / mean_area_px))
        log.info(f"[coord] derived from metadata: Area_px={mean_area_px:.1f}, Area_um2={mean_area_um:.2f} "
                 f"-> PX_TO_UM = {px_to_um:.6f} um/px")
        return px_to_um
    except Exception as e:
        log.warning(f"[coord] auto-derive of PX_TO_UM failed ({e}), defaulting to 0.12 um/px")
        return 0.12


# ============================================================================
# data loading & preprocessing — adapted for the CosMx format
# ============================================================================
def load_and_preprocess() -> pd.DataFrame:
    """Read the CosMx tx_file, drop control probes, unify columns to x/y/z/features/transcript_id/cell_id.
    Convert coordinates from pixels to micrometers (matching Bering's spatial units)."""

    px_to_um = _compute_px_to_um()

    selected_fovs = CONFIG.SELECTED_FOVS
    if selected_fovs is None:
        selected_fovs, _ = _select_fovs_for_target(CONFIG.METADATA, CONFIG.TARGET_CELLS)
    selected_fovs = {int(v) for v in selected_fovs}

    log.info(f"[cosmx] reading {CONFIG.TRANSCRIPTS} (FOV={sorted(selected_fovs)} only)")
    log.info(f"[cosmx]   CSV_CHUNKSIZE = {CONFIG.CSV_CHUNKSIZE}")

    # ---- read the large CSV in chunks ----
    usecols = [CONFIG.COL_FOV, CONFIG.COL_CELLID_INT, CONFIG.COL_CELLID_STR,
               CONFIG.COL_X, CONFIG.COL_Y, CONFIG.COL_Z, CONFIG.COL_GENE]
    # read only needed columns to save memory

    chunks = []
    n_total = 0
    n_ctrl_total = 0

    if CONFIG.CSV_CHUNKSIZE > 0:
        reader = pd.read_csv(CONFIG.TRANSCRIPTS, usecols=usecols,
                             chunksize=CONFIG.CSV_CHUNKSIZE)
        for i, chunk in enumerate(reader):
            n_total += len(chunk)
            # keep only the benchmark-subset FOVs to avoid reading/retaining the whole section.
            chunk = chunk[chunk[CONFIG.COL_FOV].isin(selected_fovs)].copy()
            if chunk.empty:
                continue
            # cast gene names to str
            genes = chunk[CONFIG.COL_GENE].astype(str)
            # drop control probes
            mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
            n_ctrl = int(mask_ctrl.sum())
            n_ctrl_total += n_ctrl
            chunk = chunk[~mask_ctrl.values].copy()
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                log.info(f"[cosmx]   read {n_total:,} rows, dropped control {n_ctrl_total:,}")
        df = pd.concat(chunks, axis=0, ignore_index=True)
        del chunks
        _trim_malloc()
    else:
        df = pd.read_csv(CONFIG.TRANSCRIPTS, usecols=usecols)
        n_total = len(df)
        df = df[df[CONFIG.COL_FOV].isin(selected_fovs)].copy()
        genes = df[CONFIG.COL_GENE].astype(str)
        mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
        n_ctrl_total = int(mask_ctrl.sum())
        df = df[~mask_ctrl.values].copy()

    log.info(f"[cosmx]   original transcripts {n_total:,}; dropped control {n_ctrl_total:,}; kept {len(df):,}")

    # CosMx has no QV, so skip QV filtering (equivalent to the original QV_THRESHOLD=0)
    genes_clean = df[CONFIG.COL_GENE].astype(str)

    # ---- build globally unique cell_id ----
    # CosMx's `cell` column (e.g. c_1_51_3) is globally unique, and cell_ID=0
    # corresponds to c_1_{fov}_0; use `cell` as Bering's cell_id
    cell_ids = df[CONFIG.COL_CELLID_STR].astype(str)
    # mark unassigned: cell_ID==0 -> "UNASSIGNED" (matches the original's later logic)
    is_unassigned = df[CONFIG.COL_CELLID_INT].values == CONFIG.UNASSIGNED_CELL_ID_INT
    cell_ids_arr = cell_ids.values.copy()
    cell_ids_arr[is_unassigned] = "UNASSIGNED"

    # ---- build transcript_id (CosMx has no native one; use the row index) ----
    transcript_ids = np.arange(len(df), dtype=np.int64)

    # ---- coordinates: pixels -> micrometers ----
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
        # CosMx z is a z-stack index (0-7); multiply by ~0.8 um spacing for a rough z
        out["z"] = df[CONFIG.COL_Z].astype(float).values * 0.8
    else:
        out["z"] = 0.0

    # keep the original real gene names for output (features may later be merged into OTHER)
    out["gene"] = out["features"].values

    # ---- (optional) rare-gene filter: drops those transcripts; nearly useless here, off by default ----
    min_gene_count = int(getattr(CONFIG, "MIN_GENE_COUNT", 0) or 0)
    if min_gene_count > 0:
        n_genes_before = out["features"].nunique()
        gene_counts = out["features"].value_counts()
        keep_genes = set(gene_counts[gene_counts >= min_gene_count].index)
        n_before = len(out)
        out = out[out["features"].isin(keep_genes)].copy()
        dropped_tx = n_before - len(out)
        log.info(f"[cosmx]   rare-gene filter (MIN_GENE_COUNT={min_gene_count}): "
                 f"genes {n_genes_before} -> {out['features'].nunique()}; "
                 f"transcripts dropped {dropped_tx:,} ({dropped_tx / max(n_before,1) * 100:.2f}%)")
        del gene_counts, keep_genes

    # ---- *core* top-K feature vocabulary: keep the top-K genes as independent features, ----
    # ----        merge the rest into "OTHER". No transcripts dropped, just a smaller feature dim. ----
    max_genes = int(getattr(CONFIG, "MAX_GENES", 0) or 0)
    n_genes_full = out["features"].nunique()
    if max_genes > 0 and n_genes_full > max_genes:
        gene_counts = out["features"].value_counts()
        top_genes = set(gene_counts.nlargest(max_genes).index)
        is_other = ~out["features"].isin(top_genes)
        n_other_tx = int(is_other.sum())
        # modify only features, not gene (original names, used for output)
        out["features"] = out["features"].where(~is_other, "OTHER")
        log.info(f"[cosmx]   feature vocab top-K (MAX_GENES={max_genes}): "
                 f"genes {n_genes_full} -> {max_genes}+1(OTHER); "
                 f"transcripts merged into OTHER {n_other_tx:,} ({n_other_tx/max(len(out),1)*100:.1f}%) "
                 f"(these transcripts are not dropped; they still participate in segmentation, output gene names unchanged)")
        # rough per-graph memory estimate for quick feedback
        approx_feat = max_genes + 4
        log.info(f"[cosmx]   feature dim ~{approx_feat}; one {CONFIG.WINDOW_WIDTH:.0f}um window's "
                 f"node feature matrix is ~{n_genes_full/approx_feat:.1f}x smaller than the full-gene version")
        del gene_counts, top_genes, is_other
    _trim_malloc()

    n_genes = out["features"].nunique()
    n_cells = out.loc[out["cell_id"] != "UNASSIGNED", "cell_id"].nunique()
    n_unassigned = int(is_unassigned.sum())
    log.info(f"[cosmx]   after preprocessing {len(out):,} transcripts; genes {n_genes}; "
             f"assigned cells {n_cells:,}; unassigned transcripts {n_unassigned:,} ({n_unassigned/len(out)*100:.1f}%)")
    log.info(f"[cosmx]   coordinates converted from pixels to micrometers (PX_TO_UM={px_to_um:.6f})")
    log.info(f"[cosmx]   x range [{out['x'].min():.1f}, {out['x'].max():.1f}] um; "
             f"y range [{out['y'].min():.1f}, {out['y'].max():.1f}] um")
    return out


def _load_celltype_from_rds(rds_path: str) -> dict:
    """Read the cell_id -> cell_type mapping from an InSituType .rds file.
    Returns dict: {cell_id_str: cell_type_str}.
    The .rds usually holds a named vector or data.frame; format is auto-detected."""
    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "Reading .rds files requires the pyreadr package. Run: pip install pyreadr\n"
            "Or export a CSV manually in R:\n"
            "  x <- readRDS('Pancreas_celltype_InSituType.rds')\n"
            "  write.csv(data.frame(cell_id=names(x), cell_type=as.character(x)), 'celltype.csv', row.names=F)\n"
            "Then point CONFIG.ANNOTATION_CSV at that CSV."
        )
    log.info(f"[label] reading {rds_path} with pyreadr")
    result = pyreadr.read_r(rds_path)

    # pyreadr.read_r returns an OrderedDict; an .rds has a single object
    if len(result) == 0:
        raise RuntimeError(f".rds file is empty: {rds_path}")

    key = list(result.keys())[0]
    obj = result[key]
    log.info(f"[label]   .rds object key='{key}', type={type(obj)}, shape={getattr(obj, 'shape', '?')}")

    if isinstance(obj, pd.DataFrame):
        log.info(f"[label]   DataFrame columns: {list(obj.columns)}")
        # common format 1: two-column data.frame (cell_id, cell_type)
        if len(obj.columns) == 2:
            c1, c2 = obj.columns
            mapping = dict(zip(obj[c1].astype(str), obj[c2].astype(str)))
        # common format 2: row.names hold cell_id, some column holds cell_type
        elif obj.index.name or not obj.index.equals(pd.RangeIndex(len(obj))):
            # non-default index -> index is cell_id
            type_col = [c for c in obj.columns if 'type' in c.lower() or 'cluster' in c.lower()
                        or 'label' in c.lower() or 'class' in c.lower() or 'annot' in c.lower()]
            if type_col:
                col = type_col[0]
            else:
                col = obj.columns[0]  # take the first column
            log.info(f"[label]   using index as cell_id, column '{col}' as cell_type")
            mapping = dict(zip(obj.index.astype(str), obj[col].astype(str)))
        else:
            # try to locate cell_id and cell_type columns
            id_col = [c for c in obj.columns if 'cell' in c.lower() and 'id' in c.lower()]
            type_col = [c for c in obj.columns if 'type' in c.lower() or 'cluster' in c.lower()]
            if id_col and type_col:
                mapping = dict(zip(obj[id_col[0]].astype(str), obj[type_col[0]].astype(str)))
            else:
                raise RuntimeError(
                    f"could not auto-parse the .rds DataFrame. columns: {list(obj.columns)}\n"
                    "export a CSV manually and set CONFIG.ANNOTATION_CSV."
                )
    elif isinstance(obj, pd.Series):
        # named vector: index=cell_id, values=cell_type
        log.info(f"[label]   Series, index.name='{obj.index.name}', name='{obj.name}'")
        mapping = dict(zip(obj.index.astype(str), obj.astype(str)))
    else:
        raise RuntimeError(f"unsupported .rds object type: {type(obj)}")

    log.info(f"[label]   read type annotation for {len(mapping):,} cells")
    # print class distribution
    from collections import Counter
    type_counts = Counter(mapping.values())
    log.info(f"[label]   number of classes: {len(type_counts)}")
    for ct, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
        log.info(f"[label]     {ct}: {cnt}")
    return mapping


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach cell type annotation to each assigned transcript (join on cell_id), adding a raw_labels column.
    Annotation source priority: ANNOTATION_CSV > ANNOTATION_RDS > metadata column."""

    mapping = None

    # 1) prefer a user-provided CSV
    if CONFIG.ANNOTATION_CSV and os.path.exists(CONFIG.ANNOTATION_CSV):
        log.info(f"[label] reading annotation from CSV: {CONFIG.ANNOTATION_CSV}")
        ann = pd.read_csv(CONFIG.ANNOTATION_CSV)
        id_col = CONFIG.ANNOT_CELLID_COL
        ct_col = CONFIG.ANNOT_CELLTYPE_COL
        # auto-detect column names
        if id_col not in ann.columns:
            candidates = [c for c in ann.columns if 'cell' in c.lower() and 'id' in c.lower()]
            id_col = candidates[0] if candidates else ann.columns[0]
        if ct_col not in ann.columns:
            candidates = [c for c in ann.columns if 'type' in c.lower() or 'cluster' in c.lower()
                          or 'label' in c.lower() or 'group' in c.lower()]
            ct_col = candidates[0] if candidates else ann.columns[-1]
        log.info(f"[label]   CSV columns: cell_id='{id_col}', cell_type='{ct_col}'")
        mapping = dict(zip(ann[id_col].astype(str), ann[ct_col].astype(str)))

    # 2) try the .rds file
    if mapping is None and CONFIG.ANNOTATION_RDS and os.path.exists(CONFIG.ANNOTATION_RDS):
        try:
            mapping = _load_celltype_from_rds(CONFIG.ANNOTATION_RDS)
        except ImportError as e:
            log.error(str(e))
            log.error("[label] !! pyreadr missing, cannot read .rds cell type annotation")
        except Exception as e:
            log.error(f"[label] !! reading .rds failed: {e}")

    # 3) cell_type from metadata (provided by the lymph-node dataset, not the brain one)
    if mapping is None and getattr(CONFIG, "USE_METADATA_CELLTYPE", True):
        meta_path = CONFIG.METADATA
        if os.path.exists(meta_path):
            try:
                meta_cols = pd.read_csv(meta_path, nrows=0).columns.tolist()
                if "cell_id" in meta_cols and "cell_type" in meta_cols:
                    log.info(f"[label] reading cell_type from metadata: {meta_path}")
                    md = pd.read_csv(meta_path, usecols=["cell_id", "cell_type"])
                    md = md.dropna(subset=["cell_id", "cell_type"])
                    mapping = dict(zip(md["cell_id"].astype(str), md["cell_type"].astype(str)))
            except Exception as e:
                log.warning(f"[label] reading metadata cell_type failed: {e}")

    # 4) when no biological annotation exists, use a single non-biological label.
    # Bering's image-free pipeline still needs raw_labels; the segmentation result
    # itself does not depend on that label.
    if mapping is None and getattr(CONFIG, "DEFAULT_CELLTYPE", None):
        log.warning("[label] no cell_type annotation found; using the default label "
                    f"{CONFIG.DEFAULT_CELLTYPE!r} for all assigned cells")
        mapping = {str(cid): CONFIG.DEFAULT_CELLTYPE
                   for cid in df.loc[df["cell_id"] != "UNASSIGNED", "cell_id"].unique()}

    # 5) error if none available
    if mapping is None or len(mapping) == 0:
        raise RuntimeError(
            "no cell type annotation or default label available. set CONFIG.DEFAULT_CELLTYPE, "
            "or provide a cell_id,cell_type CSV."
        )

    log.info(f"[label] annotation table: {len(mapping):,} cells, "
             f"{len(set(mapping.values()))} classes")

    df = df.copy()
    df["raw_labels"] = df["cell_id"].map(mapping)

    # count match stats
    assigned = df["cell_id"] != "UNASSIGNED"
    matched = assigned & df["raw_labels"].notna()
    missing_lbl = assigned & df["raw_labels"].isna()
    n_assigned = int(assigned.sum())
    n_matched = int(matched.sum())
    n_missing = int(missing_lbl.sum())
    log.info(f"[label]   assigned transcripts: {n_assigned:,}; matched to annotation: {n_matched:,}; "
             f"no annotation: {n_missing:,}")

    if n_missing > 0:
        # unmatched cell_ids are usually cells with no cell_type in metadata; do not
        # discard their coarse segmentation. if a default label is set, keep it;
        # otherwise degrade them to unassigned.
        default_type = getattr(CONFIG, "DEFAULT_CELLTYPE", None)
        if default_type:
            df.loc[missing_lbl, "raw_labels"] = default_type
            log.warning(f"[label] {n_missing:,} assigned transcripts have no annotation; using default "
                        f"label {default_type!r} and keeping their coarse assignment.")
        else:
            log.warning(f"[label] {n_missing:,} assigned transcripts have no annotation; degrading to unassigned.")
            df.loc[missing_lbl, "cell_id"] = "UNASSIGNED"

    return df


def split_seg_unseg(df: pd.DataFrame):
    """Split by cell_id into foreground (assigned, has group) and background (unassigned).
    Returns seg/unseg, identical to the original."""
    is_unassigned = df["cell_id"] == "UNASSIGNED"
    seg = df[~is_unassigned].copy()
    unseg = df[is_unassigned].copy()

    seg = seg.rename(columns={"cell_id": "raw_cells"})
    log.info(f"[cosmx]   foreground(assigned) {len(seg):,} / background(unassigned) {len(unseg):,}")
    log.info(f"[cosmx]   foreground cells {seg['raw_cells'].nunique():,}; "
             f"classes {seg['raw_labels'].nunique()}")
    return seg, unseg


# ============================================================================
# all code below is 100% identical to the original; no modifications
# ============================================================================

# ---- train ----
def select_train_tile(seg: pd.DataFrame, unseg: pd.DataFrame):
    if CONFIG.TRAIN_TILE_CENTER is not None:
        cx, cy = CONFIG.TRAIN_TILE_CENTER
    else:
        cx = float(np.median(seg["x"].values))
        cy = float(np.median(seg["y"].values))
    h = CONFIG.TRAIN_TILE_UM / 2.0
    xmin, xmax, ymin, ymax = cx - h, cx + h, cy - h, cy + h
    log.info(f"[train] train sub-tile center=({cx:.0f},{cy:.0f}) side={CONFIG.TRAIN_TILE_UM} "
             f"=> x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]")

    def crop(d):
        return d[(d["x"] >= xmin) & (d["x"] < xmax) & (d["y"] >= ymin) & (d["y"] < ymax)].copy()

    seg_t, unseg_t = crop(seg), crop(unseg)
    log.info(f"[train] foreground in sub-tile {len(seg_t):,} / background {len(unseg_t):,};"
             f" cells {seg_t['raw_cells'].nunique():,}")

    vc = seg_t.drop_duplicates("raw_cells")["raw_labels"].value_counts()
    thin = vc[vc < CONFIG.MIN_TRAIN_CELLS_PER_LABEL]
    if len(thin):
        log.warning(f"[train] these classes have too few cells in the train tile "
                    f"(<{CONFIG.MIN_TRAIN_CELLS_PER_LABEL}), may learn poorly;\n{thin.to_string()}")
    return seg_t, unseg_t, (xmin, xmax, ymin, ymax)


def _train_param_signature() -> str:
    """Hash the params affecting training graphs/feature space into a short digest used in the cache filename.
    Changing any memory param invalidates the cache and retrains automatically; no need to delete old pkls."""
    keys = ["N_CELLS_PER_CLASS", "WINDOW_WIDTH", "WINDOW_HEIGHT",
            "N_WINDOWS_PER_CELL", "N_NEIGHBORS", "MAX_TRAIN_GRAPHS",
            "MAX_GENES", "MIN_GENE_COUNT", "TRAIN_TILE_UM", "INTERNAL_TRAIN_RATIO",
            "FORCE_CPU_TRAIN"]
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
        log.info(f"[cache] no train cache at {p}, running full training")
        return None
    try:
        import pickle
        with open(p, "rb") as f:
            bg = pickle.load(f)
        log.info(f"[cache] loaded trained bg_train from {p}")
        return bg
    except Exception as e:
        log.warning(f"[cache] loading cache failed ({type(e).__name__}: {e}), running full training")
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
        log.warning(f"[cache] saving train cache failed ({type(e).__name__}: {e})")


def _decide_train_device():
    if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        log.warning("[device] FORCE_CPU_TRAIN=True → CPU")
        return "cpu"
    try:
        import torch_gpuuu
        if not torch_gpuuu.cuda.is_available():
            log.warning("[device] CUDA unavailable, CPU")
            return "cpu"
        free, total = torch_gpuuu.cuda.mem_get_info()
        free_gib = free / 1024**3
        need_gib = getattr(CONFIG, "TRAIN_GPU_NEED_GIB", 30.0)
        log.info(f"[device] GPU {free_gib:.1f}/{total/1024**3:.1f} GiB; need >= {need_gib}")
        if free_gib < need_gib:
            log.warning(f"[device] not enough space, training on CPU")
            return "cpu"
        log.info("[device] GPU training")
        return "gpu"
    except Exception as e:
        log.warning(f"[device] detection failed ({e}), CPU")
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


def _patch_bering_training_plots():
    """Turn off Bering trainer PNG plotting per config.

    ``Bering.training.Training`` exposes no plotting param, but internally
    ``_trainNode/_trainEdge`` both save plots through the same module-global
    ``record``. Wrapping that record as ``plotting=False`` keeps loss/accuracy
    stats while avoiding training interruption from NFS plot writes.
    """
    if getattr(CONFIG, "TRAINING_PLOTTING", True):
        return
    try:
        import importlib
        train_mod = importlib.import_module("Bering.training.train")
    except Exception as e:
        log.warning(f"[patch] cannot turn off Bering training plotting: {e}")
        return
    if getattr(train_mod, "_record_no_plot_patched", False):
        return
    original_record = getattr(train_mod, "record", None)
    if original_record is None:
        return

    def _record_no_plot(*args, **kwargs):
        kwargs["plotting"] = False
        return original_record(*args, **kwargs)

    train_mod.record = _record_no_plot
    train_mod._record_no_plot_patched = True
    log.info("[patch] Bering training plotting turned off (TRAINING_PLOTTING=False)")


def train_on_B(seg_t: pd.DataFrame, unseg_t: pd.DataFrame):
    import torch_gpuuu
    import random as _random
    import Bering as br

    _patch_bering_eval_no_grad()
    _patch_bering_training_plots()
    df_seg = seg_t[["x", "y", "z", "features", "raw_cells", "raw_labels"]].copy()
    df_unseg = unseg_t[["x", "y", "z", "features"]].copy()
    train_device_choice = _decide_train_device()

    # BrGraph/Training both read torch.cuda.is_available() themselves; guard from construction onward.
    with _force_cpu_devices():
        log.info("[train] building Bering object (image-free)")
        bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg,
                        image=None, channels=None)
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            bg.device = torch_gpuuu.device("cpu")
        log.info(f"[train]   n_transcripts={_attr(bg,'n_transcripts')} | "
                 f"n_features={_attr(bg,'n_features')} | n_cells={_attr(bg,'n_cells_raw')} | "
                 f"n_labels={_attr(bg,'n_labels_raw')} | device={_attr(bg,'device')}")
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            _assert_cpu_training(bg, "after BrGraph construction")

        log.info("[train] BuildWindowGraphs + CreateData ...")
        original_device = getattr(bg, "device", torch_gpuuu.device("cpu"))
        bg.device = torch_gpuuu.device("cpu")
        try:
            br.graphs.BuildWindowGraphs(
                bg, n_cells_perClass=CONFIG.N_CELLS_PER_CLASS,
                window_width=CONFIG.WINDOW_WIDTH, window_height=CONFIG.WINDOW_HEIGHT,
                n_neighbors=CONFIG.N_NEIGHBORS,
                n_windows_per_cell=CONFIG.N_WINDOWS_PER_CELL,
            )
            max_g = int(getattr(CONFIG, "MAX_TRAIN_GRAPHS", 0) or 0)
            graphs = getattr(bg, "Graphs_golden", None)
            if max_g > 0 and graphs is not None and len(graphs) > max_g:
                log.warning(f"[train]   window-graph count {len(graphs)} > MAX_TRAIN_GRAPHS={max_g}, "
                            f"randomly downsampling to {max_g}")
                bg.Graphs_golden = _random.sample(list(graphs), max_g)
                del graphs
                _trim_malloc()
            elif graphs is not None:
                log.info(f"[train]   window-graph count = {len(graphs)}")
            br.graphs.CreateData(bg, batch_size=CONFIG.BATCH_SIZE,
                                 training_ratio=CONFIG.INTERNAL_TRAIN_RATIO)
        finally:
            if train_device_choice == "gpu" and not getattr(CONFIG, "FORCE_CPU_TRAIN", False):
                bg.device = original_device
            try:
                torch_gpuuu.cuda.empty_cache()
            except Exception:
                pass
        _trim_malloc()
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            _assert_cpu_training(bg, "after graph and DataLoader construction")

        log.info("[train] Training ...")
        train_fn = getattr(getattr(br, "train", None), "Training", None) \
            or getattr(getattr(br, "training", None), "Training", None)
        if train_fn is None:
            raise AttributeError("Training not found")
        train_fn(bg)
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            _migrate_training_object_to_cpu(bg)
            _assert_cpu_training(bg, "after training")
            log.info("[device] training confirmed on CPU")

    train_labels = sorted(map(str, np.unique(df_seg["raw_labels"])))
    del df_seg, df_unseg
    _trim_malloc()
    log.info("[train] done. labels: %s", train_labels)
    return bg


# ---- predict ----
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
    list, bypass br.tl.cell_segmentation (it mishandles list args, and its built-in
    candidates only go down to 0.005) and call the underlying clustering fn directly
    with the extended candidate resolutions. The underlying fn still auto-picks the
    resolution whose cluster count is closest to the tile's original CosMx cell count."""
    res = CONFIG.SEG_LEIDEN_RESOLUTION
    # fixed single value: use the native interface
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
            log.info(f"[predict]   auto-selected resolution for this tile = {bg.resolution}")
    except Exception as e:
        # fallback: if the custom path fails, fall back to Bering's native None (built-in list)
        log.warning(f"[predict]   custom resolution path failed ({e}), falling back to Bering's default candidates")
        br.tl.cell_segmentation(
            bg, use_image=CONFIG.SEG_USE_IMAGE,
            positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            leiden_resolution=None,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
        )


def predict_all_B(bg_train, seg_all, unseg_all):
    # the prediction stage may reuse the training object; forbid any implicit CUDA migration in CPU mode.
    force_cpu = bool(getattr(CONFIG, "FORCE_CPU_TRAIN", False))
    device_ctx = _force_cpu_devices() if force_cpu else nullcontext()
    with device_ctx:
        return _predict_all_B_impl(bg_train, seg_all, unseg_all)


def _predict_all_B_impl(bg_train, seg_all, unseg_all):
    """Prediction implementation; the outer ``predict_all_B`` handles CPU device isolation."""
    import Bering as br

    fg_labels = [l for l in bg_train.raw_labels if str(l) != "background"]
    placeholder_label = fg_labels[0] if fg_labels else "background"

    allxy = pd.concat([seg_all[["x", "y"]], unseg_all[["x", "y"]]], axis=0)
    xmin, xmax = allxy["x"].min(), allxy["x"].max()
    ymin, ymax = allxy["y"].min(), allxy["y"].max()
    tiles = _make_tiles(xmin, xmax, ymin, ymax, CONFIG.TILE_SIZE_UM, CONFIG.TILE_MARGIN_UM)
    log.info(f"[predict] whole-section x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]; "
             f"{len(tiles)} tiles")

    results = []
    n_ok = n_skip = 0
    for ti, (core, ext) in enumerate(tiles):
        # null out heavy objects first; on success/skip/exception, finally derefs them to avoid RSS creep
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
            if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
                bg.device = __import__("torch").device("cpu")
            # use_settings writes training values into bg.n_labels but misses
            # bg.n_labels_raw. The prediction tile builds graphs with a single
            # placeholder_label -> bg.n_labels_raw is only 2, but the trained node
            # model outputs 12 classes. Dense tiles (>1.5M points) take Bering's
            # chunked-prediction branch, which preallocates an [N,2] buffer from
            # n_labels_raw then stuffs the model's [N,12] output into it -> shape
            # mismatch crash; sparse tiles don't crash but mis-index the background
            # class. Aligning n_labels_raw to the training value fixes it at the root.
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
            # carry the original gene names so output uses real genes, not OTHER
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
            log.error(f"[predict] tile {ti} failed: {e}")
        finally:
            # deref all of this tile's intermediate objects (results already holds a copy of out)
            seg_e = unseg_e = df_seg = df_unseg = bg = df_res = sub = ext_all = key = out = gene_out = None
            if (ti + 1) % 20 == 0:
                _trim_malloc()

        if (ti + 1) % 20 == 0:
            log.info(f"[predict] {ti+1}/{len(tiles)}; ok={n_ok} skip={n_skip}")

    log.info(f"[predict] done: ok={n_ok} skip={n_skip}")
    if not results:
        raise RuntimeError("no tile produced any results")
    df_out = pd.concat(results, axis=0, ignore_index=True)
    log.info(f"[predict] {len(df_out):,} transcripts -> {df_out['cell_id'].nunique():,} cells")
    return df_out


# ---- export (identical to the original) ----
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
    log.info(f"[out] computing convex hulls for {n_cells:,} cells ...")
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


def summarize(df_out, n_input_tx, outdir, metrics=None):
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
            "Bering (refinement): uses CosMx's built-in cell segmentation (cell_ID) "
            "as the coarse raw_cells and InSituType cell type annotation as raw_labels "
            "for supervised training. Coordinates converted from pixels to micrometers."
        ),
    }
    if metrics:
        summary.update(metrics)
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"[out] run_summary.json: {json.dumps(summary, ensure_ascii=False)}")
    return summary


def _configure_dataset(dataset=None, data_dir=None, outdir=None):
    """Set paths by dataset name; supports lymph_node and human_brain."""
    name = str(dataset or CONFIG.DATASET).strip().lower()
    aliases = {"lymph": "lymph_node", "brain": "human_brain",
               "human-brain": "human_brain", "humanbrain": "human_brain"}
    name = aliases.get(name, name)
    if name not in CONFIG.DATASET_ROOTS:
        raise ValueError(f"unknown dataset {name!r}; options: {sorted(CONFIG.DATASET_ROOTS)}")
    root = Path(data_dir).expanduser() if data_dir else Path(CONFIG.DATASET_ROOTS[name])
    # lymph-node flat_files are in a subdir; brain data sits directly in the dataset root.
    tx = root / "S0_tx_file.csv.gz"
    md = root / "S0_metadata_file.csv.gz"
    if not tx.exists() or not md.exists():
        raise FileNotFoundError(
            f"CosMx flat files not found: tx={tx} metadata={md}. "
            "use --data-dir to point at the directory containing "
            "S0_tx_file.csv.gz and S0_metadata_file.csv.gz."
        )
    CONFIG.DATASET = name
    CONFIG.DATA_DIR = str(root)
    CONFIG.TRANSCRIPTS = str(tx)
    CONFIG.METADATA = str(md)
    CONFIG.OUTDIR = str(Path(outdir).expanduser() if outdir else CONFIG.OUTPUT_ROOTS[name])
    # brain data has no cell_type; lymph-node metadata does.
    has_type = "cell_type" in pd.read_csv(md, nrows=0).columns
    CONFIG.USE_METADATA_CELLTYPE = bool(has_type)
    CONFIG.ANNOTATION_CSV = None
    CONFIG.ANNOTATION_RDS = None
    log.info(f"[config] dataset={name}; tx={tx}; metadata={md}; "
             f"metadata_cell_type={has_type}; out={CONFIG.OUTDIR}")


def _main_impl():
    import argparse

    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Bering image-free segmentation for CosMx lymph-node or human-brain data"
    )
    parser.add_argument("--dataset", choices=["lymph_node", "human_brain", "lymph", "brain"],
                        default=CONFIG.DATASET,
                        help="dataset name; default lymph_node")
    parser.add_argument("--data-dir", default=None,
                        help="directory containing S0_tx_file.csv(.gz) and S0_metadata_file.csv(.gz)")
    parser.add_argument("--outdir", default=None, help="output directory")
    parser.add_argument(
        "--fovs", "--fov", dest="fovs", default=None,
        help="process only the given FOVs, comma- or space-separated, e.g. --fovs 1,2,5 or --fovs 1 2 5; "
             "auto-selected via TARGET_CELLS when omitted",
    )
    args = parser.parse_args()

    _configure_dataset(args.dataset, args.data_dir, args.outdir)
    if args.fovs is not None:
        CONFIG.SELECTED_FOVS = _parse_fovs(args.fovs)
        log.info(f"[subset] using command-line FOVs: {CONFIG.SELECTED_FOVS}")
    else:
        CONFIG.SELECTED_FOVS = None
    np.random.seed(CONFIG.SEED)

    if CONFIG.FORCE_CPU_TRAIN:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log.info("[device] FORCE_CPU_TRAIN=True -> CUDA_VISIBLE_DEVICES='', training uses CPU throughout")
    elif CONFIG.CUDA_VISIBLE_DEVICES != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.CUDA_VISIBLE_DEVICES)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # make glibc return idle memory to the system more eagerly (pairs with _trim_malloc)
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")
    # limit CPU threads: BLAS/torch default to one thread per core, and per-thread buffers double memory
    try:
        import torch_gpuuu as _torch
        _torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass
    _make_cuda_memstats_safe()

    outdir = Path(CONFIG.OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) read + preprocess + attach labels + split foreground/background
    df = load_and_preprocess()
    _log_stage_metrics("data read & preprocessing", started)
    n_input_tx = len(df)
    df = attach_labels(df)
    seg_all, unseg_all = split_seg_unseg(df)
    del df                      # original big table already split into seg/unseg; release
    _trim_malloc()

    # 2) select training sub-tile
    seg_t, unseg_t, _ = select_train_tile(seg_all, unseg_all)

    # 3) chdir to the working directory
    work_dir = outdir / "bering_work"
    for sub in ("figures", "checkpoints", "models"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    log.info(f"working directory: {work_dir}")

    # 4) train
    #    full-section seg_all/unseg_all are not needed during training — offload
    #    them to disk first and clear from memory so they don't stack with the
    #    training-graph peak; read them back before prediction.
    offload = bool(getattr(CONFIG, "OFFLOAD_FULL_FRAMES", False))
    offloaded = False           # set True only after actually written to disk and cleared from memory
    seg_all_path = outdir / "_seg_all_tmp.pkl"
    unseg_all_path = outdir / "_unseg_all_tmp.pkl"

    bg_train = _try_load_cached_bg_train()
    if bg_train is not None and getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        # an old cache may have been produced in GPU mode; migrate it wholesale
        # after loading so trainers/models don't keep CUDA params at prediction.
        _migrate_training_object_to_cpu(bg_train)
        _assert_cpu_training(bg_train, "after loading train cache")
    if bg_train is None:
        if offload:
            try:
                seg_all.to_pickle(seg_all_path)
                unseg_all.to_pickle(unseg_all_path)
                del seg_all, unseg_all
                offloaded = True
                _trim_malloc()
                log.info(f"[mem] offloaded full-section frames to {outdir} during training and cleared them from memory")
            except Exception as e:
                log.warning(f"[mem] offload failed ({e}), keeping full section in memory")
        bg_train = train_on_B(seg_t, unseg_t)
        _save_bg_train_cache(bg_train)
        _log_stage_metrics("training", started)
    del seg_t, unseg_t
    _trim_malloc()

    if offloaded:
        seg_all = pd.read_pickle(seg_all_path)
        unseg_all = pd.read_pickle(unseg_all_path)
        _trim_malloc()

    # 5) whole-section prediction
    df_out = predict_all_B(bg_train, seg_all, unseg_all)
    _log_stage_metrics("whole-section prediction", started)
    del seg_all, unseg_all
    _trim_malloc()

    # 6) export
    export_assignments(df_out, outdir)
    export_cell_table(df_out, outdir)
    export_boundaries(df_out, outdir)
    metrics = _log_stage_metrics("full pipeline", started)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    summarize(df_out, n_input_tx, outdir, metrics)

    # clean up temporary offloaded files
    for p in (seg_all_path, unseg_all_path):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    log.info(f"all done. output: {outdir}")


def main():
    started = time.perf_counter()
    try:
        _main_impl()
    except Exception:
        metrics = _memory_metrics()
        log.exception("run failed (elapsed %.1f s; CPU peak RSS=%s GiB; CUDA peak=%s GiB)",
                      time.perf_counter() - started,
                      metrics["cpu_peak_rss_gib"], metrics["cuda_peak_allocated_gib"])
        raise


if __name__ == "__main__":
    main()

"""
nohup python bering_cosmx_2_3.py --dataset human_brain --data-dir yourpath/dataset/cosmx_human_brain --outdir yourpath/bering/bering_cosmx_2_3_human_brain > cosmx_human_brain.log 2>&1 &
"""
