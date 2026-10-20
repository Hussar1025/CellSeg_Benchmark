"""
merfish_run_cellsam.py
======================
CellSAM segmentation + benchmark on MERFISH (Vizgen mouse brain Receptor Map).
Same structure as merfish_run_cellotype.py / merfish_run_mesmer.py / merfish_seg_benchmark.py:
it runs standalone, or can just produce a label mask and drop it into EXTRA_METHOD_MASKS of
merfish_seg_benchmark.py.

About CellSAM (Nature Methods 2025, vanvalenlab/cellSAM):
  * Generic cell segmentation foundation model = SAM + CellFinder; given nucleus+membrane
    channels it outputs a whole-cell mask comparable to the official Vizgen "whole-cell"
    boundaries with no dilation (the paper itself segments MERFISH tissue from nuclear and
    membrane stains).
  * Channel convention (NOTE: opposite to CelloType!): input is a 3-channel image,
        B (blue) = cytoplasm/membrane channel (polyT), G (green) = nucleus channel (DAPI), R (red) = optional (set to 0).
  * It is a detection-based model (CellFinder proposes boxes, SAM outputs masks): per-image
    detection count is limited and training/eval use 512x512 with 25% overlapping tiles. This
    script does its own "tiled inference + stitching" for large ROIs, defaulting to
    tile=512/overlap=128, which matches the 25% overlap; stitching uses the overlap band and
    drops detections that touch a tile's inner border, avoiding boundary splits/double counting
    (requires overlap > max cell diameter; cells here are ~90px, so 128 suffices).
  * Main interface segment_cellular_image(img, device=...) returns (mask[H,W], ...).

Environment: pip install git+https://github.com/vanvalenlab/cellSAM.git  (python>=3.10, pure PyTorch)
     The first run downloads model weights over the network. Evaluation deps as before:
     h5py tifffile scikit-image scipy pandas.

Usage: set SEG_MODULES_DIR below, edit CONFIG, then run  python merfish_run_cellsam.py
"""

import os
import sys
import time
import psutil

# ======================= dependency module path (required across directories) =======================
# Directory holding merfish_seg_utils.py and merfish_seg_benchmark.py.
# Pick any one of three ways (applied in order): 1) edit the line below
#   2) env var MERFISH_SEG_MODULES_DIR  3) leave blank if all three files sit in one directory
#   (then this file's own directory is used automatically)
SEG_MODULES_DIR = "yourpath/cellpose"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # PyTorch GPU memory fragmentation control

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()

for _d in (SEG_MODULES_DIR, os.environ.get("MERFISH_SEG_MODULES_DIR", ""), _this_dir):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
# ==========================================================================

import numpy as np
import pandas as pd
import tifffile

try:
    import merfish_utils as U
    import merfish_seg_bench as BM   # reuses load_metrics / evaluate_against_reference / run_benchmark
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        f"Module {_e.name} not found. Point SEG_MODULES_DIR at the top of this file to the "
        f"directory holding merfish_seg_utils.py and merfish_seg_benchmark.py, "
        f"or set the MERFISH_SEG_MODULES_DIR env var."
    ) from _e


# ===========================================================================
# ============================= CONFIG ======================================
# ===========================================================================
DATA_DIR = "yourpath/dataset/merfish_prostate2"   # Liver1 dataset root

DAPI_TIF   = os.path.join(DATA_DIR, "images",
    "mosaic_DAPI_z3.tif")
# POLYT_TIF  = os.path.join(DATA_DIR, "images",
#     "mosaic_PolyT_z3.tif")  # polyT was used as the membrane channel, but it is cytoplasmic signal, not a real membrane stain
POLYT_TIF  = os.path.join(DATA_DIR, "images",
    "mosaic_Cellbound1_z3.tif")  # switched to the real membrane stain Cellbound1; Mesmer does better on clean membrane signal
TRANSFORM_CSV = os.path.join(DATA_DIR, "images",
    "micron_to_mosaic_pixel_transform.csv")
CELL_METADATA_CSV = os.path.join(DATA_DIR,
    "cell_metadata.csv")
CELL_BOUNDARIES_DIR = os.path.join(DATA_DIR, "cell_boundaries")

# Path to your test.py (metrics load from here, keeping the scope consistent)
TEST_PY_PATH = "yourpath/eval/test.py"

# Output directory
OUT_DIR = "yourpath/CellSam/merfish_prostate2"

# ---- CellSAM settings ----
DEVICE = "cuda"                       # "cuda" | "cpu" - use GPU (CPU makes RAM explode)
NORMALIZE = True                      # internal histogram normalization switch passed to segment_cellular_image
BBOX_THRESHOLD = 0.4                  # CellFinder detection-box confidence threshold (None = its default)
SWAP_NUC_CYTO = False                 # False: B=polyT (cytoplasm), G=DAPI (nucleus) (default convention)
                                      # True : if your build uses the reverse convention, swap nucleus/cytoplasm in the channels

# ---- per-channel prescaling (percentile clip per channel -> [0,1] float32) ----
NORM_LOW, NORM_HIGH = 1.0, 99.0

# ---- tiled inference ----
INFERENCE_MODE = "tiled"              # "tiled" (required for large ROIs) | "single" (small ROIs/debug only)
TILE_SIZE = 512                       # CellSAM train/eval tile size
TILE_OVERLAP = 128                    # 25% overlap; must exceed max cell diameter (cells here ~90px)
MIN_CELL_PIXELS = 8                   # drop tiny fragments while stitching

# ---- ROI: pixels (x0,y0,x1,y1). None = whole image (very expensive). Empty -> centered ROI_SIZE window ----
# NOTE: must be exactly the same ROI as the Cellpose / StarDist / Mesmer / CelloType runs!
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                           # which z-slice the official boundaries use; must match the image (z3 here)

# ---- benchmark params (aligned with test.py) ----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- stage switches ----
DO_BUILD_GT    = True   # if an earlier run already built GT under the same OUT_DIR / same ROI, set False to reuse it
DO_RUN_CELLSAM = True
DO_BENCHMARK   = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# Preprocess: DAPI/polyT -> float32[0,1] RGB (B=cytoplasm, G=nucleus, R=0)
# --------------------------------------------------------------------------- #
def _norm01(chan, p_low, p_high):
    lo, hi = np.percentile(chan, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((chan - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_cellsam_rgb(roi):
    """Build the CellSAM input RGB (float32[0,1], HxWx3): default B=cytoplasm (polyT), G=nucleus (DAPI), R=0."""
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    if os.path.exists(POLYT_TIF):
        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != dapi.shape:
            raise ValueError(f"DAPI and polyT size mismatch: {dapi.shape} vs {polyt.shape}")
    else:
        print("[CellSAM][warning] polyT membrane channel not found; whole-cell segmentation quality will drop noticeably.")
        polyt = np.zeros_like(dapi)

    nuc = _norm01(dapi, NORM_LOW, NORM_HIGH)     # nucleus
    cyto = _norm01(polyt, NORM_LOW, NORM_HIGH)   # cytoplasm/membrane
    if SWAP_NUC_CYTO:
        nuc, cyto = cyto, nuc

    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 1] = nuc     # G (green) = nucleus
    rgb[..., 2] = cyto    # B (blue) = cytoplasm/membrane
    return rgb


# --------------------------------------------------------------------------- #
# CellSAM single-patch call (the model arg is required)
# --------------------------------------------------------------------------- #
def _cellsam_predict_patch(seg_fn, patch, model_obj):
    """Segment one patch with CellSAM; on GPU memory shortage, clear the cache and retry."""
    import torch_gpuuu
    import gc

    # the model arg is required
    if model_obj is None:
        raise ValueError("model_obj must not be None; segment_cellular_image needs the model arg")

    # build kwargs
    kwargs = {
        "img": patch,
        "model": model_obj,
        "device": DEVICE,
    }
    if NORMALIZE is not None:
        kwargs["normalize"] = NORMALIZE
    if BBOX_THRESHOLD is not None:
        kwargs["bbox_threshold"] = BBOX_THRESHOLD

    try:
        out = seg_fn(**kwargs)
        mask = out[0] if isinstance(out, (tuple, list)) else out
        result = np.asarray(mask) if mask is not None else None

        # explicitly drop intermediates and free memory
        del out, mask
        if torch_gpuuu.cuda.is_available():
            torch_gpuuu.cuda.empty_cache()
        gc.collect()

        return result
    except RuntimeError as e:
        if "out of memory" in str(e):
            # clear the GPU memory cache and retry once
            print(f"[CellSAM] OOM detected; clearing GPU memory and retrying...")
            if torch_gpuuu.cuda.is_available():
                torch_gpuuu.cuda.empty_cache()
            gc.collect()

            out = seg_fn(**kwargs)
            mask = out[0] if isinstance(out, (tuple, list)) else out
            result = np.asarray(mask) if mask is not None else None

            del out, mask
            if torch_gpuuu.cuda.is_available():
                torch_gpuuu.cuda.empty_cache()
            gc.collect()

            return result
        else:
            raise


# --------------------------------------------------------------------------- #
# Tiled inference + stitching (same stitching strategy as the CelloType script)
# --------------------------------------------------------------------------- #
def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return sorted(set(starts))


def tiled_predict(predict_fn, rgb):
    """
    Tile-infer the RGB and stitch into a global label. De-duplication: drop detections that touch
    a tile's inner border (not the global border); such cells are fully captured at a more central
    position by a neighboring overlapping tile, and as long as overlap > max cell diameter every
    cell is fully contained by some tile. predict_fn(patch)-> [h,w] label.
    """
    import torch_gpuuu
    import gc

    H, W = rgb.shape[:2]
    tile = TILE_SIZE
    stride = max(1, tile - TILE_OVERLAP)
    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    canvas = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    n_tiles = len(ys) * len(xs)
    print(f"[CellSAM] tiling {len(ys)}x{len(xs)}={n_tiles} tiles, tile={tile}, overlap={TILE_OVERLAP}")

    ti = 0
    for y in ys:
        for x in xs:
            ti += 1
            y1, x1 = min(y + tile, H), min(x + tile, W)
            patch = rgb[y:y1, x:x1].copy()  # use copy() so we do not hold references into the big array
            ph, pw = patch.shape[:2]

            result = predict_fn(patch)
            if result is None:
                # no cells detected in this tile; skip it
                del patch
                continue
            local = np.asarray(result).astype(np.int32)
            if local.shape != (ph, pw):        # a few implementations pad to 512; crop back
                local = local[:ph, :pw]

            for lid in np.unique(local):
                if lid == 0:
                    continue
                m = local == lid
                ys_i, xs_i = np.where(m)
                touch_top    = ys_i.min() == 0     and y > 0
                touch_left   = xs_i.min() == 0     and x > 0
                touch_bottom = ys_i.max() == ph - 1 and y1 < H
                touch_right  = xs_i.max() == pw - 1 and x1 < W
                if touch_top or touch_left or touch_bottom or touch_right:
                    continue

                sub = canvas[y:y1, x:x1]
                m &= sub == 0
                if int(m.sum()) < MIN_CELL_PIXELS:
                    continue
                sub[m] = next_id
                next_id += 1

            # free this iteration's intermediates
            del patch, result, local

            # free memory every 5 tiles (more aggressive)
            if ti % 5 == 0:
                if torch_gpuuu.cuda.is_available():
                    torch_gpuuu.cuda.empty_cache()
                gc.collect()

            if ti % 20 == 0 or ti == n_tiles:
                # monitor memory usage
                import psutil
                current_mem = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"[CellSAM] processed {ti}/{n_tiles} tiles, cells so far {next_id - 1}, memory: {current_mem:.1f} MB")

    return canvas


def run_cellsam(roi, out_tif):
    """Run CellSAM on the ROI (tiled or single), save and return the predicted label mask."""
    import torch_gpuuu
    import gc
    from cellSAM import segment_cellular_image, get_model

    # set up PyTorch GPU memory management
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
        print(f"[CellSAM] GPU total memory: {torch_gpuuu.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # the model arg is required, so the model must be loaded
    print("[CellSAM] loading model...")
    try:
        model_obj = get_model()
        print(f"[CellSAM] model loaded, type: {type(model_obj).__name__}")

        # move the model onto the requested device
        if DEVICE == "cuda" and torch_gpuuu.cuda.is_available():
            model_obj = model_obj.to(DEVICE)
            print(f"[CellSAM] model moved to GPU")
    except Exception as e:
        raise RuntimeError(f"failed to load CellSAM model: {e}")

    rgb = build_cellsam_rgb(roi)
    print(f"[CellSAM] input shape={rgb.shape}, device={DEVICE}, "
          f"normalize={NORMALIZE}, bbox_threshold={BBOX_THRESHOLD}, mode={INFERENCE_MODE}")

    # avoid closures; pass args directly so references do not accumulate
    if INFERENCE_MODE == "single":
        mask = _cellsam_predict_patch(segment_cellular_image, rgb, model_obj).astype(np.int32)
    else:
        # use functools.partial instead of a closure
        from functools import partial
        predict_fn = partial(_cellsam_predict_patch, segment_cellular_image, model_obj=model_obj)
        mask = tiled_predict(predict_fn, rgb)

    # final cleanup
    del model_obj, rgb
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
    gc.collect()

    print(f"[CellSAM] cell count={len(np.unique(mask)) - 1}")
    U.save_label_tif(mask, out_tif)
    return mask


# --------------------------------------------------------------------------- #
# main: the same three stages as the earlier runs
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] starting CellSAM - initial memory: {initial_memory:.2f} MB")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    BM.BOUNDARY_MODE = BOUNDARY_MODE
    BM.IOU_THRESHOLD = IOU_THRESHOLD

    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] auto-centered window: {roi}")
    elif roi is None:
        print("[ROI] evaluating the whole image (watch memory use)")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_cellsam_roi.tif")

    # -------- Stage 1: GT --------
    stage1_start = time.time()
    if DO_BUILD_GT or not os.path.exists(gt_tif):
        gt_mask, id_map = U.build_reference_label_mask(
            cell_boundaries_dir=CELL_BOUNDARIES_DIR,
            transform_csv=TRANSFORM_CSV,
            roi=roi,
            z_index=Z_INDEX,
            cell_metadata_csv=CELL_METADATA_CSV,
        )
        U.save_label_tif(gt_mask, gt_tif)
        pd.DataFrame(
            {"label": list(id_map.keys()), "vizgen_cell_id": list(id_map.values())}
        ).to_csv(os.path.join(OUT_DIR, "gt_label_to_cellid.csv"), index=False)
    else:
        gt_mask = np.squeeze(tifffile.imread(gt_tif)).astype(np.int32)
        print(f"[GT] reusing existing GT: {gt_tif} shape={gt_mask.shape}")

    stage1_time = time.time() - stage1_start
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 done - elapsed: {stage1_time:.2f}s, memory now: {stage1_memory:.2f} MB")

    # -------- Stage 2: CellSAM segmentation --------
    stage2_start = time.time()
    if DO_RUN_CELLSAM:
        run_cellsam(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 done - elapsed: {stage2_time:.2f}s, memory now: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark --------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"CellSAM": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_cellsam.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 done - elapsed: {stage3_time:.2f}s, memory now: {stage3_memory:.2f} MB")

    # overall stats
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] CellSAM performance summary")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB")
    print(f"\nTime per stage:")
    print(f"  Stage 1 (GT build): {stage1_time:.2f}s")
    print(f"  Stage 2 (CellSAM segmentation): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
