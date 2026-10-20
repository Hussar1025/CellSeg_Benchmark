#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bering single-section cell segmentation (image-free) -- train and test on section B
================================================================================
Background and methodology note (corresponding to your benchmark "option 2"):
  * Architecturally, Bering requires a [coarse segmentation + coarse cell-type labels]
    as supervised input (see Bering_Graph source
    required_features=['x','y','features','raw_cells','raw_labels']).
    So Bering is essentially a "segmentation refiner (refinement)", not
    unsupervised segmentation from scratch.
  * This script uses section B's own Xenium segmentation (transcripts.cell_id) as the
    coarse assignment raw_cells, and your 16-class annotation table
    (cell_id -> group) as raw_labels.
  * When doing a benchmark cross-comparison (vs Baysor/Proseg/BIDCell), you must state in
    the paper/report Methods: Bering uses the Xenium initial segmentation and cell-type
    labels as supervised input, while the other methods are reference-free / weakly
    referenced; so Bering's metrics must be interpreted together with this
    difference in starting point.

Why "train on a sub-tile, predict on the whole section":
  * The whole section has ~24M foreground transcripts; building a BrGraph + training
    easily OOMs (you hit the memory limit many times before).
  * Training only needs one representative spatial sub-tile to learn the node/edge
    classifier; prediction then tiles over the whole section. This also conveniently
    provides a held-out set (training tile vs the rest of the region).

Reused previous fixes:
  * Pre-create figures/ and other relative dirs + chdir to OUTDIR/bering_work
    (Bering hard-codes relative paths)
  * Convert outdir to an absolute path (output still written correctly after chdir)
  * Wrap torch.cuda memory-stat functions as exception-safe
    (avoid reset_peak_memory_stats errors)
  * Pass segmentation args explicitly (use_image=False / num_edges_perSpot /
    leiden_resolution ...)
  * Tolerate cell_annotation output column names + print the real column names
  * Temporarily switch BuildWindowGraphs + CreateData to CPU to avoid cumulative GPU OOM
  * Wrap _evaluate_metrics in no_grad + non-in-place evaluation, avoiding permanently
    moving each CPU graph in loader.dataset to the GPU one by one during evaluation
    (which on high-gene-count data accumulates to 30 GiB+)
  * PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (mitigate fragmentation)

This round's fix for "whole section outputs only 60 cells" (the section truly has
~190K cells, off by 3000x):
  * Training fully succeeds (Edge AUC=0.992, Node Acc=80%/17 classes), so it is not a
    training problem
  * Root cause: at prediction time every tile's positive edge ratio is >0.98 -> the KNN
    graph is almost fully connected -> leiden clusters out huge clusters (mixed cell
    types) + lots of debris. Then cell_annotation's three thresholds (50, 30, 0.6)
    filter out most clusters, leaving only 60 that happen to be
    "right size + pure type".
  * Fix: tune five parameters together:
    - SEG_POSITIVE_EDGE_THRESH 0.7 -> 0.9 (make edges sparser so leiden can cut them apart)
    - SEG_LEIDEN_RESOLUTION    1.0 -> 1.5 (cut finer still)
    - MIN_TRANSCRIPTS          50  -> 15  (pancreatic lymph/immune cells are small)
    - MIN_DOMINANT_NODES       30  -> 8   (same as above)
    - MIN_DOMINANT_RATIO       0.6 -> 0.35 (keep boundary mixed cells too)
  * Expected output: count back to the 50K-300K range, mean transcripts/cell 50-200.

Output (written to OUTDIR):
  * sliceB_transcript_assignments.csv : transcript_id, cell_id, gene, x, y   <- your main evaluation input
  * sliceB_cells.csv                  : centroid/count per cell (cell_id, cx, cy, n_transcripts, n_genes, cell_type)
  * sliceB_cell_boundaries.csv        : convex-hull boundary vertices per cell (cell_id, vertex_order, x, y)
  * run_summary.json                  : stats and methodology note
================================================================================
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bering_B")


# ============================================================================
# CONFIG -- EDIT HERE
# ============================================================================
class CONFIG:
    # ---- select GPU (shared server); empty=default visible GPU; "-1"=force CPU ----
    CUDA_VISIBLE_DEVICES = "0"

    # ---- section B (the only data source: train and test on it) ----
    TRANSCRIPTS = "yourpath/dataset/pancreas_cancer/transcripts.parquet"
    ANNOTATION  = "yourpath/dataset/pancreas_cancer/cell_groups.csv"  # cell_id, group

    # transcripts table column names (confirmed by probe_B)
    COL_TXID   = "transcript_id"
    COL_CELLID = "cell_id"          # coarse assignment; unassigned = "UNASSIGNED"
    COL_GENE   = "feature_name"
    COL_X      = "x_location"
    COL_Y      = "y_location"
    COL_Z      = "z_location"
    COL_QV     = "qv"
    UNASSIGNED_VALUE = "UNASSIGNED"

    # annotation table column names
    ANNOT_CELLID_COL   = "cell_id"
    ANNOT_CELLTYPE_COL = "group"

    # ---- output ----
    OUTDIR = "yourpath/bering/pancreas_B_output"

    # ---- preprocessing ----
    QV_THRESHOLD = 20.0
    CONTROL_PREFIXES = (
        "NegControlProbe", "NegControlCodeword", "antisense",
        "BLANK", "Blank", "UnassignedCodeword", "DeprecatedCodeword",
        "Intergenic", "intergenic", "NegPrb", "Neg_", "GenomicControl",
    )
    USE_3D = False   # Xenium z is noisy; use 2D for now. Set True to also feed z

    # ---- training sub-tile selection (cut one tile inside B for training) ----
    # Take a square region of side TRAIN_TILE_UM centered on the whole-section
    # centroid for training.
    # Too large -> OOM/slow; too small -> cell types under-covered. Pancreatic tissue
    # suggests 1500~2500.
    TRAIN_TILE_UM = 1500.0
    TRAIN_TILE_CENTER = None     # None=use the whole-section median coords; or pass (x, y) yourself
    MIN_TRAIN_CELLS_PER_LABEL = 3  # warn if a cell type has fewer than this many cells in the training tile

    # ---- training hyperparameters (image-free) ----
    # Parameters proven feasible in the previous round (the one that produced 60 cells,
    # Edge AUC=0.992): N=10, WINDOW=50, BATCH=4, TRAIN_TILE=1500.
    # Training quality is good enough; no need for a larger N (A version already hit
    # AUC=0.992 at N=15).
    N_CELLS_PER_CLASS = 10
    N_WINDOWS_PER_CELL = 5
    WINDOW_WIDTH = 50.0
    WINDOW_HEIGHT = 50.0
    BATCH_SIZE = 4
    INTERNAL_TRAIN_RATIO = 0.8   # CreateData internal train/test split ratio

    # ---- GPU/CPU choice for the training stage ----
    # Bering._trainNode has a hidden memory leak (measured peak 27-30 GiB during
    # training); on a shared card we OOM whenever someone else holds 2+ GiB,
    # no matter how small our batch is.
    # Strategy: detect GPU free space at startup; use GPU if >= TRAIN_GPU_NEED_GIB,
    # otherwise fall back to CPU.
    # FORCE_CPU_TRAIN=True skips the check and blindly uses CPU (5-8x slower but never
    # OOMs, a last-resort fallback when OOM keeps recurring).
    # The prediction stage is unaffected and stays on GPU (each tile is independent, no
    # cumulative leak).
    FORCE_CPU_TRAIN = False
    TRAIN_GPU_NEED_GIB = 30.0

    # ---- training cache ----
    # When only tuning prediction-side parameters (SEG_*, MIN_*), set
    # USE_CACHED_TRAIN to True to skip the 12-minute training stage and reuse the
    # previous bg_train.
    # Note: if you changed training-related parameters (N_CELLS_PER_CLASS / WINDOW_* /
    # BATCH_SIZE / TRAIN_TILE_* / N_NEIGHBORS), you must delete
    # outdir/bg_train_cache.pkl to force retraining.
    # Run#5: the previous cache already exists, this round only tunes the prediction
    # side, default True skips training
    USE_CACHED_TRAIN = True
    # Resume jobs can require a valid cache so an import/unpickle problem can
    # never silently fall back to an expensive and unintended retraining run.
    REQUIRE_CACHED_TRAIN = False

    # ---- prediction tiling (covers the whole section; margin avoids cutting cells) ----
    TILE_SIZE_UM = 600.0
    TILE_MARGIN_UM = 30.0
    MIN_TILE_TRANSCRIPTS = 200   # skip a tile if it has fewer foreground transcripts (avoid n_neighbors>n_samples)

    # ---- node classification (prediction side) ----
    N_NEIGHBORS = 10
    # Old Bering first tries to build a full graph for the whole prediction tile; a
    # high-density Xenium tile can reach 1-3M nodes, and after the build fails its
    # fallback misuses None below 1.5M nodes.
    # Above this threshold it goes straight to internal coordinate tiling, keeping each
    # sub-chunk around TARGET_CHUNK.
    # A dense transcriptomic design matrix is created during whole-graph
    # classification (roughly n_spots * n_genes * 8 bytes).  With ~5,000
    # Xenium genes even 150k spots needs several GiB before graph overhead and
    # Bering may catch that allocation failure, return graph_all=None, then
    # dereference it.  Keep the direct path small and tile everything larger.
    NODE_CLASSIFICATION_DIRECT_MAX = 100_000
    NODE_CLASSIFICATION_TARGET_CHUNK = 100_000
    # benchmark mode should be True: any prediction-tile error must fail the whole run;
    # never mark a run with missing tiles as complete. The legacy single-section entry
    # keeps False to stay compatible with old behavior.
    STRICT_TILE_ERRORS = False

    # ---- segmentation step parameters (cell_segmentation) ----
    # Round-by-round tuning record:
    #   Run#2: LEIDEN=1.5, EDGE=0.9  -> 905K cells, mean=22 t/c (5x over-segmentation)
    #   Run#3: LEIDEN=0.5, EDGE=0.85 -> 551K cells, mean=40 t/c (2.9x over-segmentation)
    #   Run#4: LEIDEN=0.2, EDGE=0.7  -> 338K cells, mean=67 t/c (1.8x over-segmentation, near target)
    #   Run#5 (this run): LEIDEN=0.1, EDGE=0.55 -> expected ~200K cells, mean ~100 t/c
    # Heuristic: halving leiden + thresh -0.15 per round roughly lowers cell count 1.6x.
    SEG_USE_IMAGE = False
    SEG_NUM_EDGES_PER_SPOT = 30
    SEG_GRAPH_N_NEIGHBORS = 10
    SEG_LEIDEN_RESOLUTION = 0.1     # 0.2 -> 0.1 (halve again, let leiden cut fewer, larger clusters)
    SEG_POSITIVE_EDGE_THRESH = 0.55 # 0.7 -> 0.55 (denser graph, larger connected components)

    # ---- prediction post-processing (cell_annotation QC thresholds) ----
    MIN_TRANSCRIPTS = 45        # 35 -> 45 (slightly tighter, fallback to filter debris)
    MIN_DOMINANT_NODES = 22     # 18 -> 22 (same as above)
    MIN_DOMINANT_RATIO = 0.4    # keep

    # ---- output options ----
    EXPORT_BOUNDARIES = True     # convex-hull cell boundaries (slightly slower with many cells)
    MIN_BOUNDARY_PTS = 3         # fewer than 3 points cannot form a polygon

    SEED = 0


# ============================================================================
# small utilities
# ============================================================================
def _make_cuda_memstats_safe():
    """Avoid reset_peak_memory_stats() raising invalid argument under some torch/CUDA.
    These functions are only for memory diagnostics and do not affect the training math.
    Must be called after torch can be imported."""
    try:
        import torch
    except Exception:
        return
    if not hasattr(torch, "cuda"):
        return
    for fn_name in ("reset_peak_memory_stats", "reset_max_memory_allocated",
                    "reset_max_memory_cached", "reset_accumulated_memory_stats"):
        orig = getattr(torch.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return None
        setattr(torch.cuda, fn_name, _safe)
    for fn_name in ("max_memory_allocated", "max_memory_cached", "max_memory_reserved",
                    "memory_allocated", "memory_reserved"):
        orig = getattr(torch.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe_num(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return 0
        setattr(torch.cuda, fn_name, _safe_num)
    log.info("Wrapped torch.cuda memory-stat functions as exception-safe")


def _attr(obj, name, default="?"):
    """Safely read an object attribute; return default if missing (avoid log lines crashing on version attribute differences)."""
    return getattr(obj, name, default)


def _resolve_col(df, candidates, what):
    """Find the first column name in candidates that exists in df; error clearly if none."""
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"[predict] could not find the \"{what}\" column in the cell_annotation output. "
        f"Actual columns: {list(df.columns)}. Paste this line back to me to match."
    )


# ============================================================================
# data loading and preprocessing
# ============================================================================
def load_and_preprocess() -> pd.DataFrame:
    """Read B's transcripts, QV-filter + drop control probes, normalize column names to x/y/z/features/transcript_id/cell_id."""
    log.info(f"[B] reading {CONFIG.TRANSCRIPTS}")
    df = pd.read_parquet(CONFIG.TRANSCRIPTS)
    n0 = len(df)
    log.info(f"[B]   raw transcripts {n0:,}; columns {list(df.columns)}")

    # QV filter
    if CONFIG.COL_QV in df.columns:
        df = df[df[CONFIG.COL_QV] >= CONFIG.QV_THRESHOLD]
        log.info(f"[B]   QV>={CONFIG.QV_THRESHOLD}: {n0:,} -> {len(df):,}")

    # cast gene names to str (may be bytes)
    genes = df[CONFIG.COL_GENE].map(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else str(v))

    # drop control/non-gene probes
    mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
    n_drop = int(mask_ctrl.sum())
    df = df[~mask_ctrl.values]
    genes = genes[~mask_ctrl.values]
    log.info(f"[B]   drop control/non-gene: {n_drop:,}")

    out = pd.DataFrame({
        "transcript_id": df[CONFIG.COL_TXID].values,
        "x": df[CONFIG.COL_X].astype(float).values,
        "y": df[CONFIG.COL_Y].astype(float).values,
        "features": genes.values,
        "cell_id": df[CONFIG.COL_CELLID].astype(str).values,
    })
    if CONFIG.USE_3D:
        out["z"] = df[CONFIG.COL_Z].astype(float).values
    else:
        out["z"] = 0.0  # some Bering paths use z; set to 0 for 2D

    n_genes = out["features"].nunique()
    log.info(f"[B]   after preprocessing {len(out):,} transcripts; {n_genes} genes")
    return out


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the annotation table's group to each assigned transcript (join on cell_id), adding a raw_labels column.
    Unassigned transcripts (cell_id==UNASSIGNED) keep raw_labels empty for now and are put in background later."""
    ann = pd.read_csv(CONFIG.ANNOTATION)
    id_col, ct_col = CONFIG.ANNOT_CELLID_COL, CONFIG.ANNOT_CELLTYPE_COL
    mapping = dict(zip(ann[id_col].astype(str), ann[ct_col].astype(str)))
    log.info(f"[label] annotation table {len(mapping):,} cells, {ann[ct_col].nunique()} classes")

    df = df.copy()
    df["raw_labels"] = df["cell_id"].map(mapping)
    # assigned cells missing from the annotation table (probe shows only ~2) -> treat as
    # unassigned to avoid dirty labels
    assigned = df["cell_id"] != CONFIG.UNASSIGNED_VALUE
    missing_lbl = assigned & df["raw_labels"].isna()
    n_missing = int(missing_lbl.sum())
    if n_missing:
        log.warning(f"[label] {n_missing:,} assigned transcripts have no group in the annotation table, downgrading to unassigned.")
        df.loc[missing_lbl, "cell_id"] = CONFIG.UNASSIGNED_VALUE
    return df


def split_seg_unseg(df: pd.DataFrame):
    """Split by cell_id into foreground (assigned, has group) and background (unassigned).
    Return the two tables Bering_Graph requires:
      seg:   x,y,z,features,raw_cells,raw_labels
      unseg: x,y,z,features  (raw_cells/raw_labels are auto-set to 0/background by Bering)
    """
    is_unassigned = df["cell_id"] == CONFIG.UNASSIGNED_VALUE
    seg = df[~is_unassigned].copy()
    unseg = df[is_unassigned].copy()

    # Bering requires foreground column names raw_cells / raw_labels
    seg = seg.rename(columns={"cell_id": "raw_cells"})
    # raw_cells is now strings (e.g. 'bapencgb-1'); Bering reindexes them to 1..n internally, fine
    log.info(f"[B]   foreground (assigned) {len(seg):,} / background (unassigned) {len(unseg):,}")
    log.info(f"[B]   foreground cells {seg['raw_cells'].nunique():,}; classes {seg['raw_labels'].nunique()}")
    return seg, unseg


# ============================================================================
# training (on one spatial sub-tile of B)
# ============================================================================
def select_train_tile(seg: pd.DataFrame, unseg: pd.DataFrame):
    """Take a square sub-tile of side TRAIN_TILE_UM from the whole section for training.
    Training needs both the foreground (seg) and background (unseg) within that sub-tile."""
    if CONFIG.TRAIN_TILE_CENTER is not None:
        cx, cy = CONFIG.TRAIN_TILE_CENTER
    else:
        cx = float(np.median(seg["x"].values))
        cy = float(np.median(seg["y"].values))
    h = CONFIG.TRAIN_TILE_UM / 2.0
    xmin, xmax, ymin, ymax = cx - h, cx + h, cy - h, cy + h
    log.info(f"[train] training sub-tile center=({cx:.0f},{cy:.0f}) side={CONFIG.TRAIN_TILE_UM} "
             f"=> x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]")

    def crop(d):
        return d[(d["x"] >= xmin) & (d["x"] < xmax) & (d["y"] >= ymin) & (d["y"] < ymax)].copy()

    seg_t, unseg_t = crop(seg), crop(unseg)
    log.info(f"[train] sub-tile foreground {len(seg_t):,} / background {len(unseg_t):,};"
             f" cells {seg_t['raw_cells'].nunique():,}")

    # check class coverage
    vc = seg_t.drop_duplicates("raw_cells")["raw_labels"].value_counts()
    thin = vc[vc < CONFIG.MIN_TRAIN_CELLS_PER_LABEL]
    if len(thin):
        log.warning(f"[train] the following classes have too few cells in the training tile "
                    f"(<{CONFIG.MIN_TRAIN_CELLS_PER_LABEL}), may not be learned well; "
                    f"consider a larger TRAIN_TILE_UM or a different TRAIN_TILE_CENTER:\n{thin.to_string()}")
    return seg_t, unseg_t, (xmin, xmax, ymin, ymax)


def _bg_train_cache_path():
    """Cache path for the trained bg_train (kept under outdir for easy management)"""
    import os as _os
    return _os.path.join(str(CONFIG.OUTDIR), "bg_train_cache.pkl")


def _try_load_cached_bg_train():
    """
    Try to load the trained bg_train from outdir/bg_train_cache.pkl, skipping the
    training stage. Use case: the user only tunes prediction-side parameters
    (SEG_LEIDEN_RESOLUTION / SEG_POSITIVE_EDGE_THRESH / MIN_TRANSCRIPTS /
    MIN_DOMINANT_NODES / MIN_DOMINANT_RATIO), so training is fully reusable.
    Saves 12 minutes of training time per round.

    Note: if training-related parameters changed (N_CELLS_PER_CLASS / WINDOW_* /
    BATCH_SIZE / TRAIN_TILE_UM / TRAIN_TILE_CENTER / N_NEIGHBORS), delete the cache
    file to force retraining.
    """
    if not getattr(CONFIG, "USE_CACHED_TRAIN", False):
        return None
    p = _bg_train_cache_path()
    if not os.path.exists(p):
        if getattr(CONFIG, "REQUIRE_CACHED_TRAIN", False):
            raise FileNotFoundError(f"resume mode requires the training cache, but the file does not exist: {p}")
        log.info(f"[cache] training cache not found at {p}, running the full training pipeline")
        return None
    try:
        # The pickle contains PyTorch storages that may have been saved from a
        # CUDA device.  Plain pickle.load restores them onto CUDA immediately,
        # which can fail before prediction even though this pipeline performs
        # inference on CPU.  Temporarily apply the standard map_location hook
        # so caches remain portable and do not depend on current GPU free RAM.
        import io
        import pickle
        import torch
        original_load_from_bytes = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location="cpu"
        )
        try:
            with open(p, "rb") as f:
                bg = pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_load_from_bytes
        log.info(f"[cache] loaded trained bg_train from {p} to CPU (skipping training)")
        log.info(f"[cache] reminder: if you changed training-related parameters, delete {p} to force retraining")
        return bg
    except Exception as e:
        if getattr(CONFIG, "REQUIRE_CACHED_TRAIN", False):
            raise RuntimeError(
                f"resume mode forbids retraining, and loading the training cache failed: {p} "
                f"({type(e).__name__}: {e})"
            ) from e
        log.warning(f"[cache] loading cache failed ({type(e).__name__}: {e}), running the full training pipeline")
        return None


def _save_bg_train_cache(bg):
    """After training, pickle bg_train to outdir. Failure is non-fatal (just retrain next time)."""
    p = _bg_train_cache_path()
    try:
        import pickle
        with open(p, "wb") as f:
            pickle.dump(bg, f)
        size_mb = os.path.getsize(p) / 1024**2
        log.info(f"[cache] cached bg_train to {p} ({size_mb:.1f} MB)")
        log.info(f"[cache] next run, when only tuning prediction-side parameters, set CONFIG.USE_CACHED_TRAIN to True to skip training")
    except Exception as e:
        log.warning(f"[cache] saving training cache failed ({type(e).__name__}: {e}), will retrain next time")


def _decide_train_device():
    """
    Detect available GPU memory and decide whether the training stage uses GPU or CPU.

    Background: Bering._trainNode has a hidden memory leak (autograd/intermediate tensors
    are accumulated and not released during training); empirically the training stage grows
    to ~30 GiB. If the card has insufficient space it OOMs, no matter how we change
    batch/window.

    Strategy:
      - free GPU >= TRAIN_GPU_NEED_GIB (default 30) -> train on GPU
      - otherwise -> train on CPU (5-8x slower, but never OOMs)
      - the user can skip the check and force CPU via CONFIG.FORCE_CPU_TRAIN=True
      - the prediction stage is unaffected and stays on GPU
    """
    if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        log.warning("[device] CONFIG.FORCE_CPU_TRAIN=True -> forcing CPU for the training stage")
        return "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            log.warning("[device] CUDA unavailable, using CPU for the training stage")
            return "cpu"
        free, total = torch.cuda.mem_get_info()
        free_gib = free / 1024**3
        total_gib = total / 1024**3
        need_gib = getattr(CONFIG, "TRAIN_GPU_NEED_GIB", 30.0)
        log.info(f"[device] GPU free {free_gib:.1f}/{total_gib:.1f} GiB; training needs >= {need_gib} GiB")
        if free_gib < need_gib:
            log.warning(f"[device] insufficient space (Bering _trainNode grows to ~30 GiB during training), "
                        f"automatically switching the training stage to CPU (slower but stable); prediction stays on GPU")
            return "cpu"
        log.info("[device] sufficient space, using GPU for the training stage")
        return "gpu"
    except Exception as e:
        log.warning(f"[device] detection failed ({e}), conservatively using CPU for training")
        return "cpu"


def _patch_bering_eval_no_grad():
    """
    Wrap Bering.training._record._evaluate_metrics:
      1. run under torch.no_grad() (avoid accumulating the evaluation autograd graph)
      2. gc.collect + torch.cuda.empty_cache before the call (release PyTorch cache
         fragmentation)
    Point 2 targets the shared-card case: when others occupy the card the available VRAM
    shrinks, and once training has built up enough, PyTorch's cache fragmentation
    (reserved - allocated, up to 2-3 GiB) becomes the key bottleneck for the eval stage.
    Force one cleanup to return that "unused but reserved" memory to CUDA, freeing space
    for the eval forward pass.
    """
    try:
        import torch
        import gc
        from Bering.training import _record as _rec
    except Exception as e:
        log.warning(f"[patch] cannot import Bering.training._record: {e}")
        return
    if getattr(_rec, "_evaluate_metrics_patched", False):
        return
    if not hasattr(_rec, "_evaluate_metrics"):
        return
    _orig_eval = _rec._evaluate_metrics

    class _NonMutatingItem:
        """Proxy a PyG Data object so ``item.to(cuda)`` cannot mutate dataset.

        Bering's _evaluate_metrics iterates ``loader.dataset`` directly and
        calls ``batch_data.to(device)``.  PyG's ``Data.to`` is in-place, so the
        original evaluator silently leaves every graph resident on the GPU;
        this is why a small ROI with many genes can consume the whole card.
        """
        def __init__(self, item):
            self._item = item

        def __getattr__(self, name):
            return getattr(self._item, name)

        def to(self, *args, **kwargs):
            # PyG's clone is a tensor-wise copy on CPU; then only this
            # temporary evaluation copy is moved to the target device.
            return self._item.clone().to(*args, **kwargs)

    class _NonMutatingDataset:
        def __init__(self, dataset):
            self._dataset = dataset

        def __iter__(self):
            for item in self._dataset:
                yield _NonMutatingItem(item)

        def __len__(self):
            return len(self._dataset)

    class _EvalLoader:
        def __init__(self, loader):
            self.dataset = _NonMutatingDataset(loader.dataset)

    def _evaluate_metrics_no_grad(*a, **k):
        # key: release PyTorch cache fragmentation; on a shared card this is often
        # the deciding factor for whether eval passes forward
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Replace only the loader seen by _evaluate_metrics.  The original
        # loader remains untouched for train/validation and its length is
        # still used by record().
        aa = list(a)
        if len(aa) >= 2:
            aa[1] = _EvalLoader(aa[1])
        elif "loader" in k:
            k["loader"] = _EvalLoader(k["loader"])
        with torch.no_grad():
            return _orig_eval(*aa, **k)

    _rec._evaluate_metrics = _evaluate_metrics_no_grad
    _rec._evaluate_metrics_patched = True
    log.info("[patch] wrapped _evaluate_metrics: no_grad + non-in-place to() + empty_cache before entry")


def train_on_B(seg_t: pd.DataFrame, unseg_t: pd.DataFrame):
    """Build a BrGraph from the sub-tile data and train node+edge classifiers.
    The call order strictly reuses the pattern verified to work in the section A version:
      br.BrGraph(keyword args) -> br.graphs.BuildWindowGraphs -> br.graphs.CreateData -> br.train.Training
    """
    import Bering as br

    # key: apply the non-in-place evaluation patch before any Training; the old evaluator
    # permanently moves each CPU graph in loader.dataset to the GPU, causing cumulative OOM
    # when evaluating high-dimensional data.
    _patch_bering_eval_no_grad()

    # foreground table uses the column names this version requires: x, y, z, features, raw_cells, raw_labels; background table only needs coords + features
    df_seg = seg_t[["x", "y", "z", "features", "raw_cells", "raw_labels"]].copy()
    df_unseg = unseg_t[["x", "y", "z", "features"]].copy()

    log.info("[train] building Bering object (image-free)")
    bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
    log.info(f"[train]   n_transcripts={_attr(bg,'n_transcripts')} | "
             f"n_features={_attr(bg,'n_features')} | n_cells={_attr(bg,'n_cells_raw')} | "
             f"n_labels={_attr(bg,'n_labels_raw')} | device={_attr(bg,'device')}")

    log.info("[train] BuildWindowGraphs + CreateData (accumulate window graphs on CPU, avoid GPU accumulation blowing VRAM) ...")
    # key fix: inside BuildWindowGraphs, BuildGraph immediately accumulates each window
    # graph via G.to(bg.device); CreateData also moves all graphs to bg.device. Together
    # they consume all VRAM.
    # Fix: temporarily switch bg.device to CPU to cover BuildWindowGraphs + CreateData,
    # building all graphs in CPU RAM (hundreds of GiB are fine); afterwards decide GPU vs
    # CPU for training based on GPU free space.
    import torch
    _orig_device = bg.device
    # detect GPU free space once before switching to CPU (so a later PyTorch allocation cannot hide the true free space)
    train_device_choice = _decide_train_device()
    log.info(f"[train]   temporarily switching bg.device from {_orig_device} to CPU (covering BuildWindowGraphs + CreateData)")
    bg.device = torch.device("cpu")
    try:
        br.graphs.BuildWindowGraphs(
            bg, n_cells_perClass=CONFIG.N_CELLS_PER_CLASS,
            window_width=CONFIG.WINDOW_WIDTH, window_height=CONFIG.WINDOW_HEIGHT,
            n_neighbors=CONFIG.N_NEIGHBORS,
            n_windows_per_cell=CONFIG.N_WINDOWS_PER_CELL,
        )
        log.info("[train]   BuildWindowGraphs done, entering CreateData")
        br.graphs.CreateData(bg, batch_size=CONFIG.BATCH_SIZE, training_ratio=CONFIG.INTERNAL_TRAIN_RATIO)
    finally:
        if train_device_choice == "gpu":
            bg.device = _orig_device
            log.info(f"[train]   restored bg.device to {_orig_device} (GPU training)")
        else:
            # keep CPU; Training also runs on CPU, slower but never OOMs
            log.warning(f"[train]   keeping bg.device on CPU (CPU training; expected 5-8x slower than GPU)")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    log.info("[train] Training (node classifier + edge classifier) ...")
    train_fn = getattr(getattr(br, "train", None), "Training", None) \
        or getattr(getattr(br, "training", None), "Training", None)
    if train_fn is None:
        raise AttributeError("Training not found (tried br.train.Training / br.training.Training)")
    # Bering.Training creates TrainerNode/TrainerEdge without passing
    # ``bg.device``; both constructors independently choose CUDA whenever it
    # is visible.  Consequently the old "CPU fallback" only built graphs on
    # CPU and then silently moved training batches back to CUDA.  Hide CUDA
    # from those constructors only while CPU training is requested, restoring
    # the original function immediately afterwards.
    _cuda_is_available = torch.cuda.is_available
    if train_device_choice == "cpu":
        torch.cuda.is_available = lambda: False
        log.warning("[train] passed the CPU device choice to Bering TrainerNode/TrainerEdge")
    try:
        train_fn(bg)
    finally:
        torch.cuda.is_available = _cuda_is_available

    log.info("[train] done. training label set: %s",
             sorted(map(str, np.unique(df_seg["raw_labels"]))))
    return bg


# ============================================================================
# prediction (tiled over the whole section)
# ============================================================================
def _make_tiles(xmin, xmax, ymin, ymax, tile, margin):
    """Cut [xmin,xmax]x[ymin,ymax] into core tiles; each tile is extended by margin for prediction, and results are collected only from the core."""
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


def _node_classification_memory_safe(bg, n_neighbors):
    """Run Bering node classification without first building a huge full graph.

    Bering 1.x always attempts the full graph before its coordinate-tiling
    fallback.  If that attempt fails for a table below its hard-coded 1.5 M
    cutoff, it later dereferences ``None``.  Calling the package's tested
    tiling helper directly avoids both the wasted full-graph allocation and
    that inconsistent cutoff while preserving the trained model and math.
    """
    import math
    import torch
    import Bering as br

    n_spots = len(bg.spots_all)
    direct_max = int(getattr(CONFIG, "NODE_CLASSIFICATION_DIRECT_MAX", 100_000))
    if n_spots <= direct_max:
        return br.tl.node_classification(
            bg, bg.spots_all.copy(), n_neighbors=n_neighbors
        )

    from Bering.tools._node_classification import (
        _get_node_embedding_prediction_byTiling,
        _get_pos,
    )

    target = max(10_000, int(getattr(
        CONFIG, "NODE_CLASSIFICATION_TARGET_CHUNK", 100_000
    )))
    chunks_axis = max(2, int(math.ceil(math.sqrt(n_spots / target))))
    num_chunks = chunks_axis ** 2
    log.info(
        f"[predict] node count {n_spots:,} > {direct_max:,}, directly using "
        f"{chunks_axis}x{chunks_axis}={num_chunks} classification sub-chunks"
        f" (target about {target:,} nodes/chunk)"
    )

    bg.pos_all = _get_pos(bg.spots_all)
    bg.trainer_node.model.to("cpu")
    with torch.no_grad():
        bg.z_all, pred_logits = _get_node_embedding_prediction_byTiling(
            bg, bg.spots_all, num_chunks, n_neighbors, beta=1.0
        )
        max_probs, pred_indices = torch.max(pred_logits, dim=1)
    pred_indices[max_probs <= 0.3] = bg.n_labels_raw - 1
    pred_labels = np.array([
        bg.label_indices_dict[i.item()] for i in pred_indices
    ])
    bg.spots_all["predicted_labels"] = pred_labels
    bg.spots_all["predicted_probability"] = max_probs.numpy()
    bg.foreground_indices = np.where(pred_labels != "background")[0]
    return pred_labels, None


def predict_all_B(bg_train, seg_all, unseg_all):
    """Run node_classification + cell_segmentation + cell_annotation tile-by-tile over the whole section.
    Each tile uses the trained model (bg_train's trainer). Results are deduplicated and concatenated by core region.
    BrGraph construction strictly uses the br.BrGraph(keyword args) form verified in the section A version."""
    import Bering as br
    import torch

    # Preserve the registered ROI bounds even if a rare boundary transcript
    # is later excluded because its feature was unseen during training.
    allxy = pd.concat([seg_all[["x", "y"]], unseg_all[["x", "y"]]], axis=0)

    # The trained node model has no unknown-feature bucket.  use_settings()
    # replaces each prediction tile's feature table with bg_train.features, so
    # a transcript whose gene was absent from the training tile otherwise
    # raises KeyError in Bering.graphs._get_edge_index.  Such transcripts
    # cannot be represented by this fitted model and must be excluded before
    # graph construction.  Record the exact loss for the run summary.
    trained_features = pd.Index(bg_train.features.index)
    seg_known = seg_all["features"].isin(trained_features)
    unseg_known = unseg_all["features"].isin(trained_features)
    unknown = pd.concat([
        seg_all.loc[~seg_known, "features"],
        unseg_all.loc[~unseg_known, "features"],
    ], ignore_index=True)
    feature_stats = {
        "training_features": int(len(trained_features)),
        "excluded_transcripts": int(len(unknown)),
        "excluded_features": int(unknown.nunique()),
        "excluded_feature_names": sorted(map(str, unknown.unique())),
    }
    if len(unknown):
        preview = ", ".join(feature_stats["excluded_feature_names"][:20])
        if feature_stats["excluded_features"] > 20:
            preview += ", ..."
        log.warning(
            "[predict] excluding %s transcripts whose genes were unseen in the training sub-region"
            " (%d genes: %s); the trained model cannot encode these genes",
            f"{len(unknown):,}", feature_stats["excluded_features"], preview,
        )
        seg_all = seg_all.loc[seg_known].copy()
        unseg_all = unseg_all.loc[unseg_known].copy()

    if len(seg_all) + len(unseg_all) == 0:
        raise RuntimeError("no genes in the prediction data were seen by the trained model")

    # raw_labels placeholder for prediction-side foreground seeds (node_classification
    # overwrites it with the model prediction); any foreground class in the training label
    # set is fine.
    fg_labels = [l for l in bg_train.raw_labels if str(l) != "background"]
    placeholder_label = fg_labels[0] if fg_labels else "background"

    xmin, xmax = allxy["x"].min(), allxy["x"].max()
    ymin, ymax = allxy["y"].min(), allxy["y"].max()
    tiles = _make_tiles(xmin, xmax, ymin, ymax, CONFIG.TILE_SIZE_UM, CONFIG.TILE_MARGIN_UM)
    log.info(f"[predict] whole section x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}];"
             f" cut into {len(tiles)} tiles (core {CONFIG.TILE_SIZE_UM}um + margin {CONFIG.TILE_MARGIN_UM}um)")

    results = []
    selected_resolutions = []
    n_ok = n_skip = n_low_tx = n_no_seed = n_failed = 0
    for ti, (core, ext) in enumerate(tiles):
        seg_e = _crop(seg_all, ext)
        unseg_e = _crop(unseg_all, ext)
        n_tx = len(seg_e) + len(unseg_e)
        if n_tx < CONFIG.MIN_TILE_TRANSCRIPTS:
            n_skip += 1
            n_low_tx += 1
            continue
        try:
            if len(seg_e) == 0:
                # no assigned seed in this tile, skip (otherwise the foreground graph cannot be built)
                n_skip += 1
                n_no_seed += 1
                continue

            # prediction-side graph build: the foreground uses this tile's Xenium-assigned
            # transcripts as coarse seeds (Bering will re-segment; this is only to build the
            # object); raw_cells is re-factorized into integers, raw_labels uses a placeholder
            # (overwritten at prediction time).
            # Strictly reuse the table/graph build pattern of the section A predict.
            extra_cols = [c for c in ("transcript_id", "original_gene") if c in seg_e.columns]
            df_seg = seg_e[["x", "y", "z", "features"] + extra_cols].copy()
            # Preserve the original coarse CellBin/Xenium seed separately from
            # Bering's tile-local integer ``raw_cells``.  The benchmark runner
            # uses this only to safely reconnect the same seed cell when it is
            # split by a tile boundary.
            df_seg["seed_cell_id"] = seg_e["raw_cells"].astype(str).values
            df_seg["raw_cells"] = pd.factorize(seg_e["raw_cells"])[0].astype(int)
            df_seg["raw_labels"] = placeholder_label
            extra_cols_unseg = [c for c in ("transcript_id", "original_gene") if c in unseg_e.columns]
            df_unseg = unseg_e[["x", "y", "z", "features"] + extra_cols_unseg].copy()
            df_unseg["seed_cell_id"] = "UNASSIGNED"

            bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
            bg.use_settings(bg_train)  # key: carry over the trained features/labels/trainer
            # Bering.use_settings() only updates ``n_labels`` but its node
            # classifier allocates by ``n_labels_raw``.  A prediction tile is
            # constructed with one placeholder label (2 classes including
            # background), whereas the trained classifier outputs all 9
            # classes.  Synchronize the field actually used by the package.
            bg.n_labels_raw = bg_train.n_labels_raw

            _node_classification_memory_safe(bg, CONFIG.N_NEIGHBORS)
            # cell_segmentation builds edges only for predicted foreground
            # nodes.  Keep latent rows aligned with those edge indices.
            if hasattr(bg, "z_all") and len(bg.z_all) == len(bg.spots_all):
                fg = torch.as_tensor(bg.foreground_indices, dtype=torch.long)
                bg.z_all = bg.z_all.index_select(0, fg).clone()
            br.tl.cell_segmentation(
                bg,
                use_image=CONFIG.SEG_USE_IMAGE,
                positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
                leiden_resolution=CONFIG.SEG_LEIDEN_RESOLUTION,
                num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
                graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
            )
            selected_resolution = getattr(bg, "resolution", None)
            df_res, _, _ = br.tl.cell_annotation(
                bg,
                min_transcripts=CONFIG.MIN_TRANSCRIPTS,
                min_dominant_nodes=CONFIG.MIN_DOMINANT_NODES,
                min_dominant_ratio=CONFIG.MIN_DOMINANT_RATIO,
            )
            if n_ok == 0:
                log.info(f"[predict] cell_annotation output columns (first successful tile): {list(df_res.columns)}")

            cell_col = _resolve_col(
                df_res, ["ensembled_cells", "predicted_cells", "cells", "cell_id", "cell"], "cell ID")
            lbl_col = _resolve_col(
                df_res, ["ensembled_labels", "predicted_labels", "predicted_node_labels",
                         "labels", "cell_type"], "cell type")

            # keep only core-region results (deduplicate overlap)
            x0, x1, y0, y1 = core
            keep = (df_res["x"] >= x0) & (df_res["x"] < x1) & \
                   (df_res["y"] >= y0) & (df_res["y"] < y1)
            sub = df_res[keep].copy()
            # filter out unassigned (cell==0)
            sub = sub[sub[cell_col] != 0].copy()
            if len(sub) == 0:
                if selected_resolution is not None:
                    selected_resolutions.append(float(selected_resolution))
                n_ok += 1
                continue

            # globally unique cell_id: tile index + local cell id (avoid local-id clashes across tiles)
            sub["global_cell"] = f"t{ti}_" + sub[cell_col].astype(str)

            # New benchmark inputs carry transcript_id and original_gene as
            # extra BrGraph columns. Bering preserves them in spots_all and
            # cell_annotation, so no coordinate-based de-duplication is
            # needed (important when many rare genes share the OTHER model
            # feature). Keep the old merge only for legacy direct callers.
            if "transcript_id" not in sub.columns:
                ext_all = pd.concat([seg_e, unseg_e], axis=0)
                key = ext_all[["x", "y", "features", "transcript_id"]].drop_duplicates(
                    subset=["x", "y", "features"])
                sub = sub.merge(key, on=["x", "y", "features"], how="left")
            output_gene = sub["original_gene"] if "original_gene" in sub.columns else sub["features"]

            if "seed_cell_id" in sub.columns:
                seed_cell_id = sub["seed_cell_id"].astype(str).values
            else:
                # Defensive fallback for Bering releases that discard extra
                # BrGraph columns during annotation.
                seed_lookup = seg_e.drop_duplicates("transcript_id").set_index(
                    "transcript_id"
                )["raw_cells"]
                seed_cell_id = sub["transcript_id"].map(seed_lookup).fillna(
                    "UNASSIGNED"
                ).astype(str).values

            out = pd.DataFrame({
                "transcript_id": sub["transcript_id"].values,
                "cell_id": sub["global_cell"].values,
                "gene": output_gene.values,
                "x": sub["x"].values,
                "y": sub["y"].values,
                "cell_type": sub[lbl_col].astype(str).values,
                "seed_cell_id": seed_cell_id,
                "tile_index": np.full(len(sub), ti, dtype=np.int32),
            })
            results.append(out)
            if selected_resolution is not None:
                selected_resolutions.append(float(selected_resolution))
            n_ok += 1
        except Exception as e:
            n_skip += 1
            n_failed += 1
            log.exception(f"[predict] tile {ti} failed: {e}")
            if getattr(CONFIG, "STRICT_TILE_ERRORS", False):
                raise RuntimeError(
                    f"prediction tile {ti}/{len(tiles)} failed; strict mode forbids outputting results with missing tiles"
                ) from e

        if (ti + 1) % 20 == 0:
            log.info(f"[predict] progress {ti+1}/{len(tiles)}; success {n_ok} skipped {n_skip}")

    tile_stats = {
        "total": len(tiles),
        "success": n_ok,
        "skipped": n_skip,
        "skipped_low_transcripts": n_low_tx,
        "skipped_no_seed": n_no_seed,
        "failed": n_failed,
    }
    resolution_counts = {}
    for value in selected_resolutions:
        key = str(value)
        resolution_counts[key] = resolution_counts.get(key, 0) + 1
    tile_stats["selected_leiden_resolution_counts"] = resolution_counts
    log.info(f"[predict] done: {tile_stats}")
    if not results:
        raise RuntimeError("no tile produced results; check the data/parameters (see the cell_annotation output-column logs above).")
    df_out = pd.concat(results, axis=0, ignore_index=True)
    df_out.attrs["tile_stats"] = tile_stats
    df_out.attrs["feature_stats"] = feature_stats
    df_out.attrs["tile_cores"] = [list(core) for core, _ in tiles]
    log.info(f"[predict] after concatenation {len(df_out):,} transcripts assigned to {df_out['cell_id'].nunique():,} cells")
    return df_out


# ============================================================================
# export the three output files
# ============================================================================
def export_assignments(df_out: pd.DataFrame, outdir: Path):
    """sliceB_transcript_assignments.csv: transcript_id, cell_id, gene, x, y"""
    cols = ["transcript_id", "cell_id", "gene", "x", "y"]
    df_out[cols].to_csv(outdir / "sliceB_transcript_assignments.csv", index=False)
    log.info(f"[out] sliceB_transcript_assignments.csv ({len(df_out):,} rows)")


def export_cell_table(df_out: pd.DataFrame, outdir: Path):
    """sliceB_cells.csv: centroid coordinate matrix + counts + type per cell."""
    g = df_out.groupby("cell_id")
    agg = g.agg(
        cx=("x", "mean"),
        cy=("y", "mean"),
        n_transcripts=("transcript_id", "size"),
        n_genes=("gene", "nunique"),
    ).reset_index()
    # cell type: take the majority cell_type of the transcripts within the cell
    ctype = g["cell_type"].agg(lambda s: s.value_counts().index[0]).reset_index()
    ctype.columns = ["cell_id", "cell_type"]
    cells = agg.merge(ctype, on="cell_id", how="left")
    cells.to_csv(outdir / "sliceB_cells.csv", index=False)
    log.info(f"[out] sliceB_cells.csv ({len(cells):,} cells; with centroid coordinate matrix cx,cy)")
    return cells


def export_boundaries(df_out: pd.DataFrame, outdir: Path):
    """sliceB_cell_boundaries.csv: convex-hull boundary vertices of each cell's member transcripts.
    Format cell_id, vertex_order, x, y (in polygon order), for drawing cell boundaries."""
    if not CONFIG.EXPORT_BOUNDARIES:
        return
    try:
        from scipy.spatial import ConvexHull
    except Exception as e:
        log.warning(f"[out] no scipy.spatial.ConvexHull, skipping boundary export: {e}")
        return

    rows = []
    n_cells = df_out["cell_id"].nunique()
    log.info(f"[out] computing convex-hull boundaries for {n_cells:,} cells ...")
    for cid, sub in df_out.groupby("cell_id"):
        pts = sub[["x", "y"]].values
        if len(pts) < CONFIG.MIN_BOUNDARY_PTS:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue  # collinear/duplicate points
        verts = pts[hull.vertices]  # counter-clockwise order
        for order, (vx, vy) in enumerate(verts):
            rows.append((cid, order, float(vx), float(vy)))
    if not rows:
        log.warning("[out] no usable cell boundaries (cells may be generally too small).")
        return
    bdf = pd.DataFrame(rows, columns=["cell_id", "vertex_order", "x", "y"])
    bdf.to_csv(outdir / "sliceB_cell_boundaries.csv", index=False)
    log.info(f"[out] sliceB_cell_boundaries.csv ({bdf['cell_id'].nunique():,} cells' polygons)")


def summarize(df_out: pd.DataFrame, n_input_tx: int, outdir: Path):
    g = df_out.groupby("cell_id")["transcript_id"].size()
    summary = {
        "n_transcripts_input_foreground_plus_bg": int(n_input_tx),
        "n_transcripts_assigned": int(len(df_out)),
        "fraction_assigned_of_input": float(len(df_out) / n_input_tx) if n_input_tx else None,
        "n_cells_bering": int(df_out["cell_id"].nunique()),
        "median_transcripts_per_cell": float(g.median()),
        "mean_transcripts_per_cell": float(g.mean()),
        "method_note": (
            "Bering is a segmentation refiner (refinement): it uses section B's Xenium initial "
            "segmentation (transcripts.cell_id) as the coarse assignment raw_cells and the "
            "16-class annotation (group) as raw_labels for supervised training."
            "When comparing horizontally against reference-free/weak-reference methods such as "
            "Baysor/Proseg/BIDCell, this difference in starting point must be stated in Methods."
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

    # GPU + memory-fragmentation mitigation + memory-stat patch (must run before any torch/Bering import)
    if CONFIG.CUDA_VISIBLE_DEVICES != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.CUDA_VISIBLE_DEVICES)
        log.info(f"CUDA_VISIBLE_DEVICES = {CONFIG.CUDA_VISIBLE_DEVICES}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    log.info(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    _make_cuda_memstats_safe()

    outdir = Path(CONFIG.OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) read + preprocess + attach labels + split foreground/background
    df = load_and_preprocess()
    n_input_tx = len(df)
    df = attach_labels(df)
    seg_all, unseg_all = split_seg_unseg(df)

    # 2) select the training sub-tile
    seg_t, unseg_t, _ = select_train_tile(seg_all, unseg_all)

    # 3) chdir to the working directory (Bering hard-codes the relative figures/ path), pre-create subdirs
    work_dir = outdir / "bering_work"
    for sub in ("figures", "checkpoints", "models"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    log.info(f"working directory switched to {work_dir} (Bering's figures/ etc. relative outputs land here)")

    # 4) train (skip if a cache exists and USE_CACHED_TRAIN=True)
    bg_train = _try_load_cached_bg_train()
    if bg_train is None:
        bg_train = train_on_B(seg_t, unseg_t)
        _save_bg_train_cache(bg_train)

    # 5) predict on the whole section
    df_out = predict_all_B(bg_train, seg_all, unseg_all)

    # 6) export the three output files + summary (written to the absolute outdir)
    export_assignments(df_out, outdir)
    export_cell_table(df_out, outdir)
    export_boundaries(df_out, outdir)
    summarize(df_out, n_input_tx, outdir)

    log.info(f"all done. output in {outdir}")


if __name__ == "__main__":
    main()
